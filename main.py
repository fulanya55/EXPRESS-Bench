import json
import sys
import time
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"
)
os.environ["MAGNUM_LOG"] = "quiet"

import numpy as np
np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
from pathlib import Path
import quaternion
import magnum as mn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from habitat_sim.utils import viz_utils as vut
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    store_obs
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from gpt import (
    configure_usage,
    gpt_4o_mini,
    set_current_question,
    usage_snapshot,
    write_usage_summary,
)
from evaluation import score


def resolve_scene_files(scene_data_path, scene_id):
    """Resolve both the repository layout and a flat HM3D scene directory."""
    root = Path(scene_data_path).expanduser()
    scene_name = Path(scene_id).name
    asset_name = scene_name.split("-", 1)[1] if "-" in scene_name else scene_name
    scene_dirs = [root / Path(scene_id), root / scene_name]
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue
        glb_candidates = [
            scene_dir / f"{asset_name}.basis.glb",
            scene_dir / f"{asset_name}.glb",
        ]
        for glb_path in glb_candidates:
            if glb_path.is_file():
                navmesh_path = scene_dir / f"{asset_name}.navmesh"
                return str(glb_path), str(navmesh_path)
    searched = "\n".join(str(path) for path in scene_dirs)
    raise FileNotFoundError(
        f"Could not find HM3D scene {scene_id!r} under {root}. Searched:\n{searched}"
    )


def setup_distributed(cfg):
    """Use one process per GPU when launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.distributed.is_initialized():
            # A CPU-side process group is enough for barriers and avoids
            # competing with Habitat's CUDA context for NCCL resources.
            torch.distributed.init_process_group(backend="gloo", init_method="env://")
        cfg.vlm.device = f"cuda:{local_rank}"
    return rank, local_rank, world_size


def aggregate_usage(output_dir, world_size):
    """Merge per-rank usage summaries into one easy-to-read JSON file."""
    summaries = []
    for rank in range(world_size):
        path = Path(output_dir) / f"api_usage.rank{rank}.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                summaries.append(json.load(handle))
    merged = {
        "calls": sum(item.get("calls", 0) for item in summaries),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in summaries),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in summaries),
        "total_tokens": sum(item.get("total_tokens", 0) for item in summaries),
        "cost_known": all(item.get("cost_known", False) for item in summaries),
        "cost_usd": sum(item.get("cost_usd", 0.0) for item in summaries),
        "ranks": summaries,
    }
    if summaries:
        merged["input_usd_per_1m"] = summaries[0].get("input_usd_per_1m")
        merged["output_usd_per_1m"] = summaries[0].get("output_usd_per_1m")
    path = Path(output_dir) / "api_usage.json"
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def write_progress(output_dir, rank, completed, total, usage):
    """Publish a small atomic status file for the other torchrun ranks."""
    path = Path(output_dir) / f"progress.rank{rank}.json"
    payload = {
        "completed": completed,
        "total": total,
        "calls": usage["calls"],
        "total_tokens": usage["total_tokens"],
        "cost_known": usage["cost_known"],
        "cost_usd": usage["cost_usd"],
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def read_progress(output_dir, world_size):
    states = []
    for rank in range(world_size):
        path = Path(output_dir) / f"progress.rank{rank}.json"
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as handle:
                    states.append(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue
    return {
        "completed": sum(item.get("completed", 0) for item in states),
        "total": sum(item.get("total", 0) for item in states),
        "calls": sum(item.get("calls", 0) for item in states),
        "total_tokens": sum(item.get("total_tokens", 0) for item in states),
        "cost_known": len(states) == world_size and all(
            item.get("cost_known", False) for item in states
        ),
        "cost_usd": sum(item.get("cost_usd", 0.0) for item in states),
    }


def usage_difference(after, before):
    """Return the API usage attributable to one question."""
    return {
        "calls": after["calls"] - before["calls"],
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
        "cost_known": after["cost_known"],
        "cost_usd": after["cost_usd"] - before["cost_usd"],
        "input_usd_per_1m": after["input_usd_per_1m"],
        "output_usd_per_1m": after["output_usd_per_1m"],
    }


def main(cfg):
    rank, local_rank, world_size = setup_distributed(cfg)
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)
    scene_data_path = cfg.scene_data_path
    output_dir = cfg.output_dir
    seed = cfg.seed

    if rank == 0:
        for pattern in ("progress.rank*.json", "progress.rank*.tmp", "api_usage.rank*.json", "api_usage.rank*.jsonl"):
            for stale_path in Path(output_dir).glob(pattern):
                stale_path.unlink(missing_ok=True)
    if world_size > 1:
        torch.distributed.barrier()
    
    # Prompt
    region_prompt = "./prompt/region.txt"
    explore_prompt = "./prompt/explore.txt"
    answer_prompt = "./prompt/answer.txt"
    random_answer_prompt = "./prompt/random_answer.txt"
    score_prompt = "./prompt/evaluation.txt" 

    # Load dataset
    with open(cfg.dataset_path, 'r', encoding='utf-8') as file:
        questions_data = json.load(file)
    logging.info(f"Loaded {len(questions_data)} questions.")

    configure_usage(
        output_dir=output_dir,
        rank=rank,
        input_price_usd_per_1m=getattr(cfg, "api_input_usd_per_1m", None),
        output_price_usd_per_1m=getattr(cfg, "api_output_usd_per_1m", None),
    )

    # Load VLM. With torchrun each process gets a separate copy on one GPU.
    vlm = VLM(cfg.vlm)

    # Each rank gets a disjoint deterministic slice of the dataset.
    question_indices = list(range(rank, len(questions_data), world_size))
    results_all = []
    write_progress(output_dir, rank, 0, len(question_indices), usage_snapshot())
    progress = tqdm(
        question_indices,
        desc=f"Eval GPU {local_rank}" if world_size > 1 else "Eval",
        position=rank if world_size > 1 else 0,
        dynamic_ncols=True,
        unit="question",
    )
    for question_ind in progress:
        set_current_question(question_ind)
        question_usage_start = usage_snapshot()

        result = {"question_ind": question_ind}
        question_data = questions_data[question_ind]
        question = question_data["question"]
        answer = question_data["answer"]
        question_type = question_data["type"]
        scene, navmesh_file = resolve_scene_files(scene_data_path, question_data["scene_id"])
        if not os.path.isfile(navmesh_file):
            raise FileNotFoundError(
                f"Missing navmesh for {question_data['scene_id']}: {navmesh_file}"
            )
        init_pts = question_data["start_position"]
        init_rotation = question_data["start_rotation"]
        goal_pts = question_data["goal_position"]
        init_angle = 0

        logging.info(f"\n========\nIndex: {question_ind}\nScene: {scene}\nQuestion: {question}\nAnswer: {answer}")
        episode_data_dir = os.path.join(output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        sim_settings = {
            "scene": scene,
            "gpu_device_id": local_rank,
            "scene_dataset": os.path.join(scene_data_path, "hm3d/hm3d_annotated_basis.scene_dataset_config.json"),
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
            "color_sensor": cfg.rgb_sensor,
            "depth_sensor": cfg.depth_sensor,
            "semantic_sensor": cfg.semantic_sensor
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        rotation = np.quaternion(init_rotation[0], init_rotation[1], init_rotation[2], init_rotation[3])
        angle = init_angle

        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # Identify task-related regions
        regions_dict = {'A':"bathroom", 'B': "bedroom", 'C': "dining room", 'D':"garage", 'E': "kitchen", 'F': "laundry room", "G": "living room", "H": "office", "I": "rec room", "J": "study", "K": "hallway", "L": "entryway", "M": "laboratory", "N": "workout room", "O": "warehouse", "P": "lounge", "Q": "balcony", "R": "staircase", "S": "cloakroom", "T": "unknown"}  
        letters = list(regions_dict.keys())
        regions = list(regions_dict.values())
        ex_prompt = f"QUESTION: {question}\nREGION: "
        regs = gpt_4o_mini(region_prompt, ex_prompt)
        regs = regs.split(",")
        regs_list = []
        for r in regs:
            if (r != 'unknown') and (r in regions):
                regs_list.append(regions.index(r))
        logging.info(f"regs: {regs}")
        result["regs_list"] = regs
        result["path"] = {}
        
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            regs=regs_list,
            max_exp=3,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        pts_pixs = np.empty((0, 2))
        last_pts = pts
        path_len = 0
        find_answer = False

        for cnt_step in range(num_step):
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            pts_normal = pos_habitat_to_normal(pts)
            result["path"][step_name] = {"pts": agent_state.position, "rotation": agent_state.rotation}
            
            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]

            if cfg.save_obs:
                plt.imsave(
                    os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                )
            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )

            if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
                # TSDF fusion
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                )

                # Terminate exploration
                ex_prompt = f"QUESTION: {question}"
                img_path = f"{episode_data_dir}/{cnt_step}.png"
                stop_explore = gpt_4o_mini(explore_prompt, ex_prompt, img_path)
                if "yes" in stop_explore.lower():
                    ex_prompt = f"Q: {question}\nA: "
                    gen_answer = gpt_4o_mini(answer_prompt, ex_prompt, img_path)
                    gen_answer = gen_answer.replace("A:", "").strip()
                    logging.info(f"gen_answer: {gen_answer}")
                    find_answer = True
                    break

                # Semantic map
                prompt_points_pix = []
                prompt_points_pix, fig1 = (
                    tsdf_planner.find_prompt_points_within_view(
                        pts_normal,
                        img_width,
                        img_height,
                        cam_intr,
                        cam_pose_tsdf,
                        **cfg.visual_prompt,
                    )
                )
                fig1.tight_layout()
                fig1.savefig(
                    os.path.join(
                        episode_data_dir, "{}_prompt_points.png".format(cnt_step)
                    )
                )

                # Get VLM prediction
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                draw_letters = ["A", "B", "C", "D"]
                fnt = ImageFont.truetype(
                    "data/Open_Sans/static/OpenSans-Regular.ttf",
                    30,
                )
                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                    rgb_im_draw = rgb_im.copy()
                    draw = ImageDraw.Draw(rgb_im_draw)
                    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                        draw.ellipse(
                            (
                                point_pix[0] - cfg.visual_prompt.circle_radius,
                                point_pix[1] - cfg.visual_prompt.circle_radius,
                                point_pix[0] + cfg.visual_prompt.circle_radius,
                                point_pix[1] + cfg.visual_prompt.circle_radius,
                            ),
                            fill=(200, 200, 200, 255),
                            outline=(0, 0, 0, 255),
                            width=3,
                        )
                        draw.text(
                            tuple(point_pix.astype(int).tolist()),
                            draw_letters[prompt_point_ind],
                            font=fnt,
                            fill=(0, 0, 0, 255),
                            anchor="mm",
                            font_size=12,
                        )
                    rgb_im_draw.save(
                        os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
                    )

                    prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                    lsv = vlm.get_loss(
                        rgb_im_draw,
                        prompt_lsv,
                        draw_letters[:actual_num_prompt_points],
                    )
                    lsv *= actual_num_prompt_points / 3
                    prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                    gsv = vlm.get_loss(rgb_im, prompt_gsv, ["Yes", "No"])[0]
                    gsv = (
                        np.exp(gsv / cfg.gsv_T) / cfg.gsv_F
                    )
                    sv = lsv * gsv
                    logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

                    # Integrate semantics only if there is any prompted point
                    tsdf_planner.integrate_sem(
                        sem_pix=sv,
                        radius=1.0,
                        obs_weight=1.0,
                    )

                    # Get region prediction
                    prompt_reg = f"You need to determine what region the picture shows. All possible regions are shown in the dictionary {regions_dict}. Answer with the key corresponding to the region in the dictionary."
                    pred = vlm.get_loss(
                        rgb_im,
                        prompt_reg,
                        letters,
                    )
                    array_reg = np.array(pred)
                    region = regions[np.argmax(array_reg)]
                    logging.info(f"Predicted region: {region}, Confidence: {np.max(array_reg)}")

                    # Update TSDF with the predicted region
                    if region != "unknown":
                        prompt_pt = f"Based on the content of the picture, please judge which letter indicates the location that belongs to the {region} area. Answer with a single letter."
                        pred = vlm.get_loss(
                            rgb_im_draw,
                            prompt_pt,
                            draw_letters[:actual_num_prompt_points],
                        )
                        array_pt = np.array(pred)
                        point = draw_letters[np.argmax(array_pt)]
                        logging.info(f"Predicted point: {point}, Confidence: {np.max(array_pt)}")

                        # Update only the region with the highest probability and its coordinates
                        tsdf_planner.integrate_reg(
                            [np.argmax(array_pt)],
                            [np.argmax(array_reg)],
                            radius=1.0,
                        )
                    
                    # Draw semantic map and region map
                    fig2 = tsdf_planner.draw_map(regions_dict)
                    fig2.tight_layout()
                    fig2.savefig(
                        os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1))
                    )

            else:
                logging.info("Skipping black image!")

            # Determine whether to use GOE or FBE
            if regs_list != []:
                state, exp_list = tsdf_planner.get_exp_state()
            else:
                state = True
            found_path = True

            # GOE
            if state == False:
                agent_state = agent.get_state()
                cur_pt = agent_state.position
                pts_normal, pts_pix, fig3 = tsdf_planner.find_next_point_region(cur_pt, regions_dict)
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                pts = pathfinder.snap_point(pts)
                reg = [i["count"] for i in exp_list]
                idx = np.argmax(reg)
                direction = exp_list[idx]["dir"]
                if direction == 0:
                    pts = pathfinder.snap_point(pts)
                    path = habitat_sim.ShortestPath()
                    path.requested_start = np.array(last_pts)
                    path.requested_end = pts
                    found_path = simulator.pathfinder.find_path(path)
                    start_time = time.time()
                    while not found_path:
                        end_time = time.time()
                        if end_time - start_time > 60:
                            break
                        pts_normal, pts_pix, fig3 = tsdf_planner.find_next_point_region(cur_pt, regions_dict)
                        pts_normal = np.append(pts_normal, floor_height)
                        pts = pos_normal_to_habitat(pts_normal)
                        pts = pathfinder.snap_point(pts)
                        simulator.pathfinder.seed(seed)
                        path = habitat_sim.ShortestPath()
                        path.requested_start = np.array(last_pts)
                        path.requested_end = pts
                        found_path = simulator.pathfinder.find_path(path)
                    if found_path:
                        path_len += path.geodesic_distance
                        fig3.savefig(
                            os.path.join(episode_data_dir, "{}_region.png".format(cnt_step + 1))
                        )
                else:
                    for _ in range(3):
                        obs = simulator.step("turn_left")
                    agent_state = agent.get_state()
                    pts = agent_state.position
                    rotation = agent_state.rotation

                region = regions[exp_list[idx]["region"]]
                logging.info(f"Region: {region}")

            # FBE       
            if (state == True) or (found_path == False):
                pts_normal, angle, pts_pix, fig4 = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner,
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig4.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig4.tight_layout()
                fig4.savefig(
                    os.path.join(episode_data_dir, "{}_semantic.png".format(cnt_step + 1))
                )
                
                rotation = quat_to_coeffs(
                    quat_from_angle_axis(angle, np.array([0, 1, 0]))
                    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                ).tolist()

                pts = pathfinder.snap_point(pts)
                if np.isnan(pts).any():
                    pts = pathfinder.get_random_navigable_point_near(last_pts, 3)
                path = habitat_sim.ShortestPath()
                path.requested_start = np.array(last_pts)
                path.requested_end = pts
                found_path = simulator.pathfinder.find_path(path)
                if found_path:
                    path_len += path.geodesic_distance

            last_pts = pts

                
        if not find_answer:
            gen_answer = gpt_4o_mini(random_answer_prompt, question)
            gen_answer = gen_answer.replace("A:", "").strip()
            logging.info(f"gen_answer: {gen_answer}")

        ex_prompt = f"Question: {question}\nAnswer: {answer}\nResponse: {gen_answer}\nYour mark: "
        img_path = os.path.join(episode_data_dir, f"{cnt_step}.png")
        EAC = gpt_4o_mini(score_prompt, ex_prompt, img_path)
        question_usage = usage_difference(usage_snapshot(), question_usage_start)

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Index: {question_ind}")
        logging.info(f"Scene: {scene}")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Find_answer: {find_answer}")
        logging.info(f"Gen_answer: {gen_answer}")
        logging.info(f"EAC: {EAC}")
        logging.info(f"Path_len: {path_len}")
        logging.info(
            f"API usage: {question_usage['calls']} calls, "
            f"{question_usage['total_tokens']} tokens, "
            f"${question_usage['cost_usd']:.6f}"
            if question_usage["cost_known"]
            else f"API usage: {question_usage['calls']} calls, "
            f"{question_usage['total_tokens']} tokens, cost=?"
        )

        # Distance to target point
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(pts)
        path.requested_end = np.array(goal_pts)
        found_path = simulator.pathfinder.find_path(path)
        goal_dis = path.geodesic_distance
        
        result.update({
            "scene": scene, 
            "type": question_type, 
            "question": question, 
            "answer": answer,
            "geodesic_distance": question_data["geodesic_distance"],
            "cnt_step": cnt_step,
            "gen_answer": gen_answer,
            "EAC": EAC,
            "end_pts": pts,
            "goal_dis": goal_dis,
            "path_len": path_len,
            "api_usage": question_usage,
        })

        # Save data
        results_all.append(result)
        with open(os.path.join(episode_data_dir, f"result.pkl"), "wb") as f:
            pickle.dump(result, f)
        with open(os.path.join(episode_data_dir, "api_usage.json"), "w", encoding="utf-8") as f:
            json.dump(question_usage, f, ensure_ascii=False, indent=2)
        if (question_ind+1) % cfg.save_freq == 0:
            with open(os.path.join(output_dir, f"results.rank{rank}_{question_ind+1}.pkl"), "wb") as f:
                pickle.dump(results_all, f)
        stats = usage_snapshot()
        write_progress(output_dir, rank, len(results_all), len(question_indices), stats)
        global_stats = read_progress(output_dir, world_size)
        cost_text = f"${global_stats['cost_usd']:.4f}" if global_stats["cost_known"] else "cost=?"
        progress.set_postfix(
            done=f"{global_stats['completed']}/{len(questions_data)}",
            calls=global_stats["calls"],
            tokens=global_stats["total_tokens"],
            cost=cost_text,
        )
        set_current_question(None)

    progress.close()
    local_results_path = Path(output_dir) / f"results.rank{rank}.pkl"
    with local_results_path.open("wb") as f:
        pickle.dump(results_all, f)
    write_usage_summary(output_dir, rank=rank)

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        if world_size > 1:
            results_all = []
            for rank_index in range(world_size):
                rank_path = Path(output_dir) / f"results.rank{rank_index}.pkl"
                with rank_path.open("rb") as f:
                    results_all.extend(pickle.load(f))
            results_all.sort(key=lambda item: item["question_ind"])
        with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
            pickle.dump(results_all, f)
        C_avg, C_star_avg, E_path, d_T_avg = score(results_all)
        usage = aggregate_usage(output_dir, world_size)
        cost_text = f"${usage['cost_usd']:.6f}" if usage["cost_known"] else "unknown (set API unit prices)"
        logging.info(
            f"\nC_avg: {C_avg}\nC_star_avg: {C_star_avg}\n"
            f"E_path: {E_path}\nd_T_avg: {d_T_avg}\n"
            f"API calls: {usage['calls']}\nAPI tokens: {usage['total_tokens']}\n"
            f"API cost: {cost_text}"
        )
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="USD per 1M input tokens (overrides api_input_usd_per_1m)",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="USD per 1M output tokens (overrides api_output_usd_per_1m)",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    if args.input_price is not None:
        cfg.api_input_usd_per_1m = args.input_price
    if args.output_price is not None:
        cfg.api_output_usd_per_1m = args.output_price
    OmegaConf.resolve(cfg)

    # Set up logging
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    logging_name = f"log.rank{rank}.log" if world_size > 1 else "log.log"
    logging_path = os.path.join(cfg.output_dir, logging_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    main(cfg)

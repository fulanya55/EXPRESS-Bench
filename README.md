# [Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering](https://arxiv.org/pdf/2503.11117)      
A Large-scale Embodied Question Answering (EQA) benchmark and method

### Abstract
Embodied Question Answering (EQA) is a challenging task in embodied intelligence that requires agents to dynamically explore 3D environments, actively gather visual information, and perform multi-step reasoning to answer questions. However, current EQA approaches suffer from critical limitations in exploration efficiency, dataset design, and evaluation metrics. Moreover, existing datasets often introduce biases or prior knowledge, leading to disembodied reasoning, while frontier-based exploration strategies struggle in cluttered environments and fail to ensure fine-grained exploration of task-relevant areas. To address these challenges, we construct the **EXP**loration-awa**R**e **E**mbodied que**S**tion an**S**wering **Bench**mark (EXPRESS-Bench), the largest dataset designed specifically to evaluate both exploration and reasoning capabilities. EXPRESS-Bench consists of 777 exploration trajectories and 2,044 question-trajectory pairs. To improve exploration efficiency, we propose Fine-EQA, a hybrid exploration model that integrates frontier-based and goal-oriented navigation to guide agents toward task-relevant regions more effectively. Additionally, we introduce a novel evaluation metric, Exploration-Answer Consistency (EAC), which ensures faithful assessment by measuring the alignment between answer grounding and exploration reliability. Extensive experimental comparisons with state-of-the-art EQA models demonstrate the effectiveness of our EXPRESS-Bench in advancing embodied exploration and question reasoning.

### Installation

Set up the environment (Linux, Python 3.10; this checkout uses `uv`):
```
conda env create -f environment.yml
conda activate fine-eqa
pip install -e .
```

Install the latest version of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) on headless machines:

```
conda install habitat-sim headless -c conda-forge -c aihabitat
```

Install [Prismatic VLM](https://github.com/TRI-ML/prismatic-vlms):
```
git https://github.com/TRI-ML/prismatic-vlms.git
cd prismatic-vlms
pip install -e .
```

### EXPRESS-Bench

EXPRESS-Bench comprises 777 exploration trajectories and 2,044 question-trajectory pairs. The corresponding question-answer pairs are stored in [express-bench.json](https://github.com/kxxxxxxxxxx/EXPRESS-Bench/tree/main/data/express-bench.json), while the full set of episodes for EXPRESS-Bench can be accessed from [[Google Drive](https://drive.google.com/file/d/1_FyeWi62d7NcB2VtBQPwkHSpsiWAQaL3/view?usp=sharing)], [[Baidu](https://pan.baidu.com/s/1s_q_QedXMFQzgvY4Ty6Unw?pwd=mj3f)] and [[ModelScope](https://www.modelscope.cn/datasets/kxxxxxxx/EXPRESS-Bench)]. 

To obtain the train and val splits of the HM3D dataset, please download them [here](https://github.com/matterport/habitat-matterport-3dresearch). Note that semantic annotations are required, and access must be requested in advance.

Afterward, your [data](https://github.com/kxxxxxxxxxx/EXPRESS-Bench/tree/main/data) directory structure should be:

```
|→ data
	|→ episode
		|→ 0000-00006-HkseAnWCgqk
		|→ ...
	|→ hm3d
		|→ train
			|→ 00000-kfPV7w3FaU5
			|→ ...
		|→ val
			|→ 00800-TEEsavR23oF
			|→ ...
	|→ express-bench.json
```

### Fine-EQA

To run the Fine-EQA model, you can use the following command:

```
uv run python main.py -cf fine_eqa.yaml
```

The evaluator writes `api_usage.json` and `api_usage.rank*.jsonl` under
`output_dir`. The OpenAI-compatible endpoint returns token usage, but not the
account's USD tariff. Set `api_input_usd_per_1m` and `api_output_usd_per_1m` in
`fine_eqa.yaml` (or pass `--input-price` and `--output-price`) to show the
running dollar cost in the progress bar and the final log.

For one process per GPU, launch with `torchrun`. Questions are sharded by
index, each process loads one VLM copy on its assigned GPU, and rank 0 merges
`results.rank*.pkl` before scoring:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 main.py -cf fine_eqa.yaml \
  --input-price <USD-per-1M-input-tokens> \
  --output-price <USD-per-1M-output-tokens>
```

Each rank displays its progress bar; the `done`, `tokens`, and `cost` fields
are aggregated across ranks as status files are updated.

### Acknowledgement
This project is built upon the [explore-eqa](https://github.com/Stanford-ILIAD/explore-eqa). We sincerely thank the authors for their excellent work and open-source contribution, which served as a solid foundation for our development.

### Citation
If you use this code for your research, please cite our paper.      
```
@inproceedings{EXPRESSBench,
title={Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering},
author={Jiang, Kaixuan and Liu, Yang and Chen, Weixing and Luo, Jingzhou and Chen, Ziliang and Pan, Ling and Li, Guanbin and Lin, Liang},
year={2025}
booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)}
}
``` 
If you have any question about this code, feel free to reach (jiangkx3@mail2.sysu.edu.cn or liuy856@mail.sysu.edu.cn). 

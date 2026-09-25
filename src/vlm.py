import time
import logging
import os
from pathlib import Path
import torch
import numpy as np

class VLM:
    def __init__(self, cfg):
        start_time = time.time()
        # Keep all local model locations in the YAML config. This avoids
        # relying on shell startup files and makes migration a two-line edit.
        model_root_value = getattr(cfg, "model_root", None) or "/root/wxwu/model"
        model_root = Path(str(model_root_value)).expanduser()
        llama_root_value = getattr(cfg, "llama_path", None) or model_root / "Llama-2-7b-hf"
        llama_root = Path(str(llama_root_value)).expanduser()
        os.environ["PRISMATIC_MODEL_ROOT"] = str(model_root)
        os.environ["PRISMATIC_LLAMA2_7B_PATH"] = str(llama_root)

        # Import after setting the paths: Prismatic reads its local backbone
        # locations while constructing its model registries.
        from prismatic import load

        model_id = str(cfg.model_id)
        model_path = Path(model_id)
        if not model_path.is_absolute():
            local_candidate = model_root / model_path
            if local_candidate.is_dir():
                model_id = str(local_candidate)

        self.model = load(model_id, hf_token=cfg.hf_token)
        self.model.to(cfg.device, dtype=torch.bfloat16)
        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

    def generate(self, prompt, image, T=0.4, max_tokens=512):
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=T,
            max_new_tokens=max_tokens,
            min_length=1,
        )
        return generated_text

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        """Return normalized first-token probabilities for the requested strings."""
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        pixel_values = self.model.vision_backbone.image_transform(image)
        probabilities = self.model.generate_batch(
            pixel_values,
            [prompt_text],
            return_string_probabilities=tokens,
            max_new_tokens=1,
        )[0]
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if get_smx:
            return probabilities / np.sum(probabilities)
        return probabilities

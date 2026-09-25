import time
import logging
import os
from pathlib import Path
import torch
import numpy as np

# The released checkpoint contains the LLM weights, while the Prismatic
# loader still needs the local Llama-2 config and tokenizer during setup.
LOCAL_LLAMA_ROOT = Path("/root/wxwu/model/Llama-2-7b-hf")
if LOCAL_LLAMA_ROOT.is_dir():
    os.environ.setdefault("PRISMATIC_LLAMA2_7B_PATH", str(LOCAL_LLAMA_ROOT))

from prismatic import load


LOCAL_MODEL_ROOT = Path("/root/wxwu/model")


class VLM:
    def __init__(self, cfg):
        start_time = time.time()
        # Resolve the released Prismatic checkpoint from the shared local model
        # directory. This avoids falling back to a Hugging Face cache or Hub
        # download when the config contains the short model name.
        model_id = str(cfg.model_id)
        model_path = Path(model_id)
        if not model_path.is_absolute():
            local_candidate = LOCAL_MODEL_ROOT / model_path
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
        "Get unnormalized losses (negative logits) of the tokens"
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        losses = self.model.get_loss(
            image,
            prompt_text,
            return_string_probabilities=tokens,
        )[0]
        losses = np.array(losses)
        if get_smx:
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))
        return losses

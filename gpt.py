import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import requests


def _load_local_env():
    """Load simple KEY=value entries from the repository's untracked .env."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_local_env()
API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = "gpt-5.6-luna"


def _optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


_usage_lock = threading.Lock()
_usage = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
    "cost_known": True,
}
_usage_file = None
_input_price_usd_per_1m = _optional_float(os.environ.get("OPENAI_INPUT_USD_PER_1M"))
_output_price_usd_per_1m = _optional_float(os.environ.get("OPENAI_OUTPUT_USD_PER_1M"))
_current_question_ind = None


def set_current_question(question_ind):
    global _current_question_ind
    _current_question_ind = question_ind


def configure_usage(output_dir=None, rank=0, input_price_usd_per_1m=None,
                    output_price_usd_per_1m=None):
    """Configure per-process API accounting and its JSONL output file.

    Prices are deliberately configurable because OpenAI-compatible gateways
    generally return token usage but not the account's USD tariff.
    """
    global _usage_file, _input_price_usd_per_1m, _output_price_usd_per_1m
    if input_price_usd_per_1m is not None:
        _input_price_usd_per_1m = float(input_price_usd_per_1m)
    if output_price_usd_per_1m is not None:
        _output_price_usd_per_1m = float(output_price_usd_per_1m)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _usage_file = Path(output_dir) / f"api_usage.rank{rank}.jsonl"


def usage_snapshot():
    with _usage_lock:
        return {
            **_usage,
            "input_usd_per_1m": _input_price_usd_per_1m,
            "output_usd_per_1m": _output_price_usd_per_1m,
        }


def write_usage_summary(output_dir, rank=0):
    summary_path = Path(output_dir) / f"api_usage.rank{rank}.json"
    summary_path.write_text(
        json.dumps(usage_snapshot(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def _record_usage(response_payload):
    usage = response_payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    cost_usd = None
    if _input_price_usd_per_1m is not None and _output_price_usd_per_1m is not None:
        cost_usd = (
            prompt_tokens * _input_price_usd_per_1m
            + completion_tokens * _output_price_usd_per_1m
        ) / 1_000_000.0

    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": response_payload.get("model", OPENAI_MODEL),
        "request_id": response_payload.get("id"),
        "question_ind": _current_question_ind,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }
    with _usage_lock:
        _usage["calls"] += 1
        _usage["prompt_tokens"] += prompt_tokens
        _usage["completion_tokens"] += completion_tokens
        _usage["total_tokens"] += total_tokens
        if cost_usd is None:
            _usage["cost_known"] = False
        else:
            _usage["cost_usd"] += cost_usd
        if _usage_file is not None:
            with _usage_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def prompt_make(prompt_path, ex_prompt):
    with open(prompt_path, "r", encoding='utf-8') as f:
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        return prompt_system, prompt


def gpt_4o_mini(prompt_path, ex_prompt, img_path=None):
    api_key = API_KEY

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    content = [{"type": "text", "text": prompt}]
    if img_path:
        base64_image = encode_image(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": content}
        ]}
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty; add it to the repository .env file")
    response = requests.post(
        f"{OPENAI_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    output = response.json()
    _record_usage(output)
    return output["choices"][0]['message']["content"]

import base64
import os
from pathlib import Path
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
    return output["choices"][0]['message']["content"]

from pathlib import Path


def load_prompt(path, substitutions=None):
    prompt = Path(path).read_text(encoding="utf-8").strip()
    for name, value in (substitutions or {}).items():
        prompt = prompt.replace(f"{{{name}}}", str(value))
    return prompt

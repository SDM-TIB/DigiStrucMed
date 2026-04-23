"""Step 1 JSON I/O and logging (also used from step2/step3)."""
import json
from pathlib import Path
from datetime import datetime


def save_json(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    n = len(data) if isinstance(data, list) else "dict"
    print(f"  [saved] {path}  ({n})")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(step: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{step}] {ts} - {msg}")

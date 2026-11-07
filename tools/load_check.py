#!/usr/bin/env python3
"""Check that the nodes load in the latest ComfyUI.

Clones ComfyUI, installs its requirements into a virtual environment (CPU PyTorch on
Linux), links this repository into custom_nodes, starts the server, and checks that every
node in NODE_CLASS_MAPPINGS appears in /object_info. No API key is used, so the check
costs nothing.

Usage:
  load_check.py WORKDIR                  clone ComfyUI and create the environment in WORKDIR
  load_check.py WORKDIR --python PYTHON  use an existing environment and WORKDIR/ComfyUI
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PORT = 8199
STARTUP_TIMEOUT_SECONDS = 300


def node_names() -> list[str]:
    tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS" for t in node.targets
        ):
            return [key.value for key in node.value.keys]
    raise SystemExit("load-check: NODE_CLASS_MAPPINGS not found in nodes.py")


def run(*args: str | Path, cwd: Path | None = None) -> None:
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def prepare(workdir: Path, python: str | None) -> tuple[Path, str]:
    comfyui = workdir / "ComfyUI"
    if not comfyui.exists():
        run("git", "clone", "--depth", "1", COMFYUI_REPO, comfyui)
    if python is None:
        venv = workdir / ".venv"
        if not venv.exists():
            run(sys.executable, "-m", "venv", venv)
        python = str(venv / "bin" / "python")
        pip = [python, "-m", "pip", "install", "--quiet"]
        if sys.platform.startswith("linux"):
            run(*pip, "torch", "torchvision", "torchaudio", "--index-url", TORCH_CPU_INDEX)
        run(*pip, "-r", comfyui / "requirements.txt")
    link = comfyui / "custom_nodes" / ROOT.name
    if not link.exists():
        link.symlink_to(ROOT, target_is_directory=True)
    return comfyui, python


def object_info(name: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/object_info/{name}", timeout=5) as response:
            return json.loads(response.read()).get(name)
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--python")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    comfyui, python = prepare(args.workdir.resolve(), args.python)
    commit = subprocess.run(["git", "log", "-1", "--format=%h %cs"], cwd=comfyui, capture_output=True, text=True)
    print(f"load-check: ComfyUI {commit.stdout.strip()}")

    log_path = args.workdir / "comfyui.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [python, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(PORT)],
            cwd=comfyui,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        names = node_names()
        found: dict[str, dict] = {}
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while len(found) < len(names) and time.monotonic() < deadline and server.poll() is None:
            for name in names:
                if name not in found and (info := object_info(name)):
                    found[name] = info
            time.sleep(2)
    finally:
        server.terminate()
        server.wait(timeout=30)

    log_text = log_path.read_text(errors="replace")
    failed = "IMPORT FAILED" in log_text and ROOT.name in log_text
    for name in names:
        info = found.get(name)
        models = info["input"]["required"]["model"][0] if info else []
        print(f"load-check: {name}: {'loaded' if info else 'MISSING'} (models: {', '.join(models)})")
    if failed or len(found) < len(names):
        print(log_text[-4000:], file=sys.stderr)
        print("load-check: the nodes did not load", file=sys.stderr)
        return 1
    print("load-check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

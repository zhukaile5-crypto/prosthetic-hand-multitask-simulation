"""Portable launcher for the two original loopback-only web demos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


REPOSITORY = Path(__file__).resolve().parents[1]
DEMOS = {
    "five-finger": ("五指假肢抓握仿真", 8878, "pbl-five-finger-grasp-v1"),
    "two-finger": ("抓握交互仿真", 8877, "pbl-reflex-grasp-v1"),
}


def _open_when_ready(url: str, expected_app: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "/state", timeout=1) as response:
                state = json.load(response)
            if state.get("app") != expected_app:
                print("The selected port is occupied by another app.", file=sys.stderr)
                return
            if state.get("error"):
                print("The renderer could not start; see the server error above.", file=sys.stderr)
                return
            if not state.get("loading", False):
                webbrowser.open(url)
                return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(0.25)
    print(f"Browser opening timed out. Check server output, then open {url}", file=sys.stderr)


def main(demo: str) -> int:
    directory, default_port, app_id = DEMOS[demo]
    parser = argparse.ArgumentParser(description=f"Launch the original {demo} web demo.")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    parser.add_argument("--check", action="store_true", help="Load the model and render one initial frame, then exit; no experiment is run.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    project = REPOSITORY / "projects" / directory
    if not (project / "server.py").is_file():
        parser.error(f"Demo sources are missing: {project}")

    # A relative MJCF path also avoids native Windows/CJK path parsing problems.
    os.chdir(project)
    sys.path.insert(0, str(project))
    try:
        import mujoco
        from simulation import GraspSimulation
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Install requirements.txt with this Python interpreter.", file=sys.stderr)
        return 1

    if args.check:
        sim = GraspSimulation()
        with mujoco.Renderer(sim.model, height=240, width=320) as renderer:
            frame = sim.render(renderer)
        if frame.shape != (240, 320, 3):
            raise RuntimeError(f"Unexpected rendered frame shape: {frame.shape}")
        print(f"PASS: {demo}; model nq={sim.model.nq}, nu={sim.model.nu}; frame=320x240; simulation time={sim.data.time:.3f}s")
        return 0

    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting {demo}: {url}\nKeep this terminal open. Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url, app_id), daemon=True).start()
    sys.argv = [str(project / "server.py"), "--port", str(args.port)]
    runpy.run_path(str(project / "server.py"), run_name="__main__")
    return 0

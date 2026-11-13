"""Read-only rendering of recorded full-hand trials; never simulate a new step.

Example:
  python render_replay.py --left output/.../nominal__capsule__ratio__520101 \
      --right output/.../nominal__capsule__unified__520101 \
      --out evidence/replay_capsule_v1 --times 3 7.5 9 12.5 13.5 --gif

Contact-force labels come exclusively from the saved trial. mj_forward restores
geometry; its reconstructed contact force is intentionally never consulted.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from foundation_config import FoundationConfig
from hand_plant import HandPlant

ROOT = Path(__file__).resolve().parent
SHAPE_NAMES = {'capsule': '胶囊体', 'box': '圆角盒', 'sphere': '球体'}
METHOD_NAMES = {'ratio': '摩擦比基线', 'unified': '多任务融合策略'}
DIGIT_NAMES = ('拇指', '食指', '中指', '无名指', '小指')
BG = '#eaf0f5'
INK = '#1c3044'
MUTED = '#52677b'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def local_path(value: str | Path) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Replay inputs and outputs must stay in the independent integration directory')
    return path


def forbid_advancement(*args, **kwargs):
    raise RuntimeError('Replay prohibits physics advancement and execution commands')


def replay_guard() -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch.object(mujoco, 'mj_step', forbid_advancement))
    stack.enter_context(patch.object(mujoco, 'mj_step1', forbid_advancement))
    stack.enter_context(patch.object(mujoco, 'mj_step2', forbid_advancement))
    for name in ('advance', 'execute_delta', 'set_closure', 'set_motion', 'set_environment', 'begin_release'):
        stack.enter_context(patch.object(HandPlant, name, forbid_advancement))
    return stack


@lru_cache(maxsize=16)
def font(size: int):
    for filename in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if Path(filename).exists():
            return ImageFont.truetype(filename, size=size)
    return ImageFont.load_default(size=size)


@dataclass
class TrialReplay:
    path: Path

    def __post_init__(self):
        self.path = local_path(self.path)
        if self.path.is_dir():
            self.path = self.path / 'trace.npz'
        self.meta_path = self.path.with_name('metadata.json')
        trace_bytes, meta_bytes = self.path.read_bytes(), self.meta_path.read_bytes()
        self.initial_hashes = {
            str(self.path.relative_to(ROOT)): hashlib.sha256(trace_bytes).hexdigest(),
            str(self.meta_path.relative_to(ROOT)): hashlib.sha256(meta_bytes).hexdigest(),
        }
        self.meta = json.loads(meta_bytes.decode('utf-8'))
        with np.load(BytesIO(trace_bytes), allow_pickle=False) as archive:
            self.arrays = {key: archive[key].copy() for key in archive.files}
        embedded = json.loads(str(self.arrays.pop('metadata').item()))
        if any(self.meta.get(key) != value for key, value in embedded.items()):
            raise ValueError('External and embedded metadata disagree')
        if self.meta.get('schema') != 'shadow_hand_trial_v1':
            raise ValueError('This renderer accepts only new full-hand session trial schema')
        if self.meta.get('trace_sha256') != self.initial_hashes[str(self.path.relative_to(ROOT))]:
            raise ValueError('Trace SHA does not match saved metadata')
        required = ('time', 'qpos', 'qvel', 'ctrl', 'normal', 'position', 'rotation', 'phase')
        if any(key not in self.arrays for key in required):
            raise ValueError('Missing saved state, pose, or actual-force arrays')
        self.times = np.asarray(self.arrays['time'], float)
        if self.times.ndim != 1 or len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError('Trial needs at least two strictly increasing time samples')
        for key in ('time', 'qpos', 'qvel', 'ctrl', 'normal', 'position', 'rotation'):
            if len(self.arrays[key]) != len(self.times) or not np.isfinite(self.arrays[key]).all():
                raise ValueError(f'Non-finite or misaligned saved array: {key}')
        if self.arrays['normal'].shape != (len(self.times), 5) or np.min(self.arrays['normal']) < -1e-9:
            raise ValueError('Saved forces must be five finite nonnegative physical measurements')
        recorded_sources = self.meta.get('source_sha256', {})
        if not all(key in recorded_sources for key in ('hand_plant.py', 'foundation_config.py')):
            raise ValueError('Recorded physical source identity is missing')
        self.source_hashes = {}
        for name, expected in recorded_sources.items():
            path = local_path(name)
            actual = digest(path)
            if actual != expected:
                raise ValueError(f'Source identity changed since this trial: {name}')
            self.source_hashes[name] = actual
        self.plant = HandPlant(FoundationConfig(**self.meta['config']))
        if self.plant.model_sha256 != self.meta.get('model_sha256'):
            raise ValueError('Recompiled model identity does not match the recorded trial')
        if self.plant.model.na != 0 or self.plant.model.nmocap != 0:
            raise ValueError('This saved schema does not include actuator activation or mocap state')
        if self.arrays['qpos'].shape[1:] != (self.plant.model.nq,) or self.arrays['qvel'].shape[1:] != (self.plant.model.nv,):
            raise ValueError('Full saved joint state does not match the compiled model')
        self.ctrl_key = 'ctrl_before' if 'ctrl_before' in self.arrays else 'ctrl'
        if self.arrays[self.ctrl_key].shape[1:] != (self.plant.model.nu,):
            raise ValueError('Saved actuator controls do not match the compiled model')
        self.validation = self.validate_saved_geometry()

    def restore(self, index: int) -> None:
        data = self.plant.data
        data.qpos[:] = self.arrays['qpos'][index]
        data.qvel[:] = self.arrays['qvel'][index]
        # ctrl_before belongs to the measured qpos/force epoch. ctrl is the next
        # command just issued at that same sample; neither is re-executed here.
        data.ctrl[:] = self.arrays[self.ctrl_key][index]
        data.time = float(self.times[index])
        mujoco.mj_forward(self.plant.model, data)

    def validate_saved_geometry(self) -> dict:
        position_error, rotation_error, state_error = 0., 0., 0.
        for index in range(len(self.times)):
            self.restore(index)
            data = self.plant.data
            position_error = max(position_error, float(np.max(np.abs(data.xpos[self.plant.object] - self.arrays['position'][index]))))
            palm_rotation = data.xmat[self.plant.palm].reshape(3, 3)
            relative_rotation = palm_rotation.T @ data.xmat[self.plant.object].reshape(3, 3)
            rotation_error = max(rotation_error, float(np.max(np.abs(relative_rotation - self.arrays['rotation'][index]))))
            for key in ('qpos', 'qvel'):
                state_error = max(state_error, float(np.max(np.abs(getattr(data, key) - self.arrays[key][index]))))
            state_error = max(state_error, float(np.max(np.abs(data.ctrl - self.arrays[self.ctrl_key][index]))))
        passed = position_error <= 1e-10 and rotation_error <= 1e-10 and state_error <= 1e-12
        if not passed:
            raise ValueError(f'Saved geometry restoration failed: {position_error=}, {rotation_error=}, {state_error=}')
        return dict(passed=passed, samples_checked=len(self.times), object_position_max_absolute_error_m=position_error,
                    object_relative_rotation_max_absolute_error=rotation_error, saved_state_max_absolute_error=state_error,
                    control_field=self.ctrl_key, compiled_nq=self.plant.model.nq, compiled_nv=self.plant.model.nv,
                    compiled_nu=self.plant.model.nu, compiled_na=self.plant.model.na, compiled_nmocap=self.plant.model.nmocap)

    def nearest(self, time: float) -> int:
        if time < self.times[0] - 1e-8 or time > self.times[-1] + 1e-8:
            raise ValueError(f'Time {time} lies outside this actual recorded trajectory')
        return int(np.argmin(np.abs(self.times - time)))

    def render_panel(self, renderer, time: float, width: int, height: int) -> Image.Image:
        index = self.nearest(time)
        self.restore(index)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.045, 0., .36]
        cam.azimuth, cam.elevation, cam.distance = 150., -15., .68
        renderer.update_scene(self.plant.data, camera=cam)
        frame = Image.fromarray(renderer.render().copy())
        panel = Image.new('RGB', (width, height + 190), BG)
        panel.paste(frame, (0, 75))
        draw = ImageDraw.Draw(panel)
        method = self.meta['method']
        title = f"{METHOD_NAMES.get(method, method)}  [{method}]"
        draw.text((18, 9), title, font=font(22), fill=INK)
        label = f"{SHAPE_NAMES.get(self.meta['shape'], self.meta['shape'])}   t = {self.times[index]:.3f} s   {self.arrays['phase'][index]}"
        draw.text((18, 42), label, font=font(17), fill=MUTED)
        draw.text((18, height + 84), '保存的实际逐指接触力 / N', font=font(17), fill=INK)
        box_width = (width - 32) / 5
        for digit, name in enumerate(DIGIT_NAMES):
            x = int(16 + box_width * digit)
            draw.rounded_rectangle((x, height + 113, int(x + box_width - 7), height + 166), radius=5, fill='white')
            draw.text((x + 8, height + 116), name, font=font(15), fill=MUTED)
            draw.text((x + 8, height + 137), f"{self.arrays['normal'][index, digit]:.3f}", font=font(19), fill='#087e8b')
        palm = float(self.arrays.get('palm', np.zeros(len(self.times)))[index])
        draw.text((18, height + 170), f"掌部：{palm:.3f} N   ·   受力取自原始记录，不由回放重算", font=font(13), fill=MUTED)
        return panel

    def unchanged(self) -> dict:
        final = {name: digest(ROOT / name) for name in self.initial_hashes}
        if final != self.initial_hashes:
            raise RuntimeError('A replay input changed during rendering')
        if any(digest(ROOT / name) != value for name, value in self.source_hashes.items()):
            raise RuntimeError('A recorded source changed during rendering')
        return dict(input_sha256_before=self.initial_hashes, input_sha256_after=final, inputs_unchanged=True,
                    source_sha256=self.source_hashes, source_identity_passed=True, model_sha256=self.plant.model_sha256,
                    method=self.meta['method'], shape=self.meta['shape'], seed=self.meta['seed'],
                    run_id=self.meta['run_id'], geometry_validation=self.validation)


def asset_identity() -> dict:
    path = ROOT / 'review/source_provenance.json'
    provenance = json.loads(path.read_text(encoding='utf-8'))
    hashes = {}
    for relative, expected in provenance['upstream_assets'].items():
        asset = ROOT / 'assets/shadow' / relative
        hashes[str(asset.relative_to(ROOT))] = digest(asset)
        if hashes[str(asset.relative_to(ROOT))] != expected:
            raise ValueError(f'Copied upstream asset changed: {relative}')
    legacy = ROOT / 'assets/legacy_model.xml'
    hashes[str(legacy.relative_to(ROOT))] = digest(legacy)
    if digest(legacy) != provenance['legacy_scene_sha256']:
        raise ValueError('Copied parent model changed')
    return dict(provenance_path=str(path.relative_to(ROOT)), provenance_sha256=digest(path),
                copied_assets_match_original_provenance=True, asset_sha256=hashes)


def compose(trials, renderers, time: float, width: int, height: int) -> Image.Image:
    panels = [trial.render_panel(renderer, time, width, height) for trial, renderer in zip(trials, renderers)]
    image = Image.new('RGB', (len(panels) * width + (len(panels) - 1) * 8, height + 254), BG)
    draw = ImageDraw.Draw(image)
    draw.text((18, 8), '五指假肢 · 真实轨迹回放', font=font(23), fill=INK)
    scenario = trials[0].meta['scenario']
    scenario_name = scenario.get('name', str(scenario)) if isinstance(scenario, dict) else str(scenario)
    draw.text((18, 40), f"场景：{scenario_name}   种子：{trials[0].meta['seed']}   ·   仅显示保存状态", font=font(15), fill=MUTED)
    for column, panel in enumerate(panels):
        image.paste(panel, ((width + 8) * column, 64))
    return image


def main(argv=None):
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True, help='New session trial folder or trace.npz')
    parser.add_argument('--right', help='Optional matching trial of the other method')
    parser.add_argument('--out', required=True, help='Fresh evidence/replay_* output directory')
    parser.add_argument('--times', nargs='*', type=float, default=[3., 7.5, 9., 12.5, 13.5])
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float, default=16.)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--width', type=int, default=600)
    parser.add_argument('--height', type=int, default=460)
    parser.add_argument('--validate-only', action='store_true', help='Validate actual saved records without creating a Renderer')
    args = parser.parse_args(argv)
    out = local_path(args.out)
    if out.parent != ROOT / 'evidence' or not out.name.startswith('replay_'):
        raise ValueError('Output must be a new direct evidence/replay_* directory')
    if out.exists():
        raise FileExistsError('Replay never overwrites an existing output')
    if not 1 <= args.fps <= 30 or not 320 <= args.width <= 1200 or not 240 <= args.height <= 1000:
        raise ValueError('Choose fps 1..30 and ordinary renderer dimensions')
    source_hash = digest(Path(__file__))
    asset_check = asset_identity()
    with replay_guard():
        trials = [TrialReplay(Path(args.left))]
        if args.right:
            trials.append(TrialReplay(Path(args.right)))
            a, b = (trial.meta for trial in trials)
            for key in ('shape', 'seed', 'scenario', 'config', 'sensor_period', 'handoff_time', 'release_at'):
                if a.get(key) != b.get(key):
                    raise ValueError(f'Side-by-side trials do not share {key}')
            if a['method'] == b['method']:
                raise ValueError('Paired view requires two different saved methods')
        for time in args.times:
            for trial in trials:
                trial.nearest(time)
        if args.gif and (args.start > args.end or args.start < 0):
            raise ValueError('Invalid GIF time interval')
        out.mkdir(parents=True, exist_ok=False)
        outputs, frame_times, gif_info = {}, [], None
        if not args.validate_only:
            renderers = []
            try:
                renderers = [mujoco.Renderer(trial.plant.model, height=args.height, width=args.width) for trial in trials]
                for time in args.times:
                    name = f'frame_t{time:07.3f}.png'
                    compose(trials, renderers, time, args.width, args.height).save(out / name)
                    outputs[name] = digest(out / name)
                if args.gif:
                    frame_count = round((args.end - args.start) * args.fps)
                    if frame_count < 2 or abs(frame_count / args.fps - (args.end - args.start)) > 1e-8:
                        raise ValueError('GIF duration must contain an integer number of at least two frames')
                    frame_times = [args.start + index / args.fps for index in range(frame_count)]
                    # GIF stores centiseconds. Cumulative boundary rounding gives
                    # exact total time (e.g. 120/130 ms alternation at 8 fps).
                    boundaries = [round(index * 100 / args.fps) * 10 for index in range(frame_count + 1)]
                    durations = np.diff(boundaries).astype(int).tolist()
                    frames = [compose(trials, renderers, time, args.width, args.height).quantize(colors=192)
                              for time in frame_times]
                    frames[0].save(out / 'paired_replay.gif', save_all=True, append_images=frames[1:],
                                   duration=durations, loop=0, optimize=False, disposal=2)
                    with Image.open(out / 'paired_replay.gif') as saved_gif:
                        actual_durations = []
                        for index in range(saved_gif.n_frames):
                            saved_gif.seek(index)
                            actual_durations.append(int(saved_gif.info['duration']))
                    if sum(actual_durations) != boundaries[-1] or len(actual_durations) != frame_count:
                        raise RuntimeError('Saved GIF frame count or display duration disagrees with requested playback')
                    gif_info = dict(frame_count=frame_count, duration_ms=sum(actual_durations),
                                    frame_duration_ms=actual_durations, endpoint_excluded=True)
                    outputs['paired_replay.gif'] = digest(out / 'paired_replay.gif')
            finally:
                for renderer in renderers:
                    renderer.close()
        final_asset_check = asset_identity()
        if final_asset_check != asset_check or digest(Path(__file__)) != source_hash:
            raise RuntimeError('Replay code or copied visual assets changed during rendering')
        report = dict(schema='full_hand_read_only_replay_v1', passed=True,
                      created_utc=datetime.now(timezone.utc).isoformat(), renderer_sha256=source_hash,
                      physics_advancement_prohibited=True, control_invocation_prohibited=True,
                      state_restoration='full qpos/qvel and measurement-time ctrl_before; mj_forward only',
                      force_label_source='saved normal[time,5] object-contact and palm arrays; no reconstructed forces',
                      asset_identity=asset_check, trials=[trial.unchanged() for trial in trials],
                      snapshot_requested_times_s=args.times, gif_frame_requested_times_s=frame_times,
                      gif_display_timing=gif_info, elapsed_wall_seconds=perf_counter()-began,
                      render=dict(width_per_panel=args.width, image_height=args.height, fps=args.fps,
                                  camera=dict(lookat=[.045,0.,.36], azimuth=150.,elevation=-15.,distance=.68)),
                      output_sha256=outputs,
                      scope='display of saved new-platform trials; no new simulation or performance estimate')
        json_write(out / 'replay_validation.json', report)
        print(json.dumps(dict(passed=True, out=str(out), outputs=list(outputs),
                              restoration_samples=[len(trial.times) for trial in trials]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

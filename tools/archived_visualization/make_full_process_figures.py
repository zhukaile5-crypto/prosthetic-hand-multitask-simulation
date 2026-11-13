"""Publication-style full-process total-force figures, from recorded trials only."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
OUT = ROOT / '总指力大图与阶段拆分_2026-11-12'
BATCH = ROOT.parent / '多任务联合控制研究_2026-11-03/output/cold_start_take_pilot_v1_20261112'
os.environ['MPLCONFIGDIR'] = str(WORK / 'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from PIL import Image

SCENES = {
    'nominal': ('正常运输', '正常运输过程中的总指力', '两策略从 0 s 分别控制；观察至 16 s，四场景采用统一坐标范围。'),
    'friction_drop': ('摩擦变化', '摩擦变化条件下的总指力', '两策略从 0 s 分别控制；9.3 s 改变接触摩擦，四场景采用统一坐标范围。'),
    'fragile_conflict': ('易碎限力', '易碎限力条件下的总指力', '易碎限力条件从 0 s 生效，9.3 s 加入外载荷；四场景采用统一坐标范围。'),
    'collision': ('碰撞扰动', '碰撞扰动条件下的总指力', '9.3 s 是障碍物开始接近的指令时刻，实际接触随后发生；四场景采用统一坐标范围。'),
}
ARMS = ('baseline', 'macro')
COLORS = {'baseline':'#3478B5', 'macro':'#188A65'}
LABELS = {'baseline':'力比 PI 基线', 'macro':'宏观版'}
NEW_NAMES = [f'{i:02d}_{v[0]}_全程总指力_科研版.png' for i, v in enumerate(SCENES.values(), 13)]
NEW_NAMES.append('17_四种情况_全程总指力总览.png')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert read(BATCH / 'audit.json')['passed']
inputs = {str(BATCH / n):sha(BATCH / n) for n in ('audit.json', 'plan.json')}
preserved = {str(p):sha(p) for p in ROOT.rglob('*') if p.is_file()
    and not p.is_relative_to(ROOT / '.制作与校验')
    and not (p.parent == OUT and p.name in NEW_NAMES)}
rows = {}
for scene in SCENES:
    rows[scene] = {}
    for arm in ARMS:
        folder = BATCH / 'runs' / f'{arm}__{scene}__619741'
        for name in ('metadata.json', 'result.json', 'trace.npz'):
            inputs[str(folder / name)] = sha(folder / name)
        meta, result = read(folder / 'metadata.json'), read(folder / 'result.json')
        assert meta['control_start_s'] == 0 and meta['common_preparation_used'] is False
        for name, h in result['artifacts_sha256'].items():
            assert sha(folder / name) == h
        with np.load(folder / 'trace.npz', allow_pickle=False) as z:
            t = np.r_[0., z['substep_time']+z['substep_duration']]
            force = np.vstack([z['normal'][0], z['substep_normal']]).sum(axis=1)
        assert len(t) == len(force) == 32001 and np.all(np.isfinite(force))
        assert np.allclose(t, np.arange(32001)*.0005, atol=1e-8, rtol=0)
        assert abs(force.max()-result['total_force']['full']['raw_total_peak_N']) < 1e-10
        assert force.min() >= 0 and force.max() < 27.5
        rows[scene][arm] = dict(time=t, force=force, meta=meta, result=result)
    assert rows[scene]['baseline']['meta']['config'] == rows[scene]['macro']['meta']['config']
    assert np.array_equal(rows[scene]['baseline']['time'], rows[scene]['macro']['time'])

for p in (r'C:\Windows\Fonts\times.ttf', r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\msyhbd.ttc'):
    fm.fontManager.addfont(p)
TITLE = fm.FontProperties(fname=r'C:\Windows\Fonts\msyhbd.ttc')
plt.rcParams.update({'font.family':['Times New Roman','SimSun'], 'font.size':13,
    'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':'#78838D', 'axes.linewidth':.8, 'figure.facecolor':'white', 'savefig.facecolor':'white',
    'path.simplify':False, 'agg.path.chunksize':0})
legend = [Line2D([0], [0], color=COLORS[a], linewidth=1.4 if a == 'macro' else 1., label=LABELS[a]) for a in ARMS]

def plot(ax, scene, overview=False):
    series = []
    for arm in ARMS:
        r = rows[scene][arm]
        line, = ax.plot(r['time'], r['force'], color=COLORS[arm], linewidth=.7 if arm == 'baseline' else 1.05,
            alpha=.68 if arm == 'baseline' else 1., zorder=2 if arm == 'baseline' else 3)
        if arm == 'macro':
            line.set_path_effects([pe.Stroke(linewidth=1.65, foreground='white', alpha=.82), pe.Normal()])
        assert np.array_equal(line.get_xdata(), r['time'])
        assert np.array_equal(line.get_ydata(), r['force'])
        ax.axvline(r['result']['lift_request_start_s'], color=COLORS[arm], lw=.65, ls=':', alpha=.65, zorder=1)
        series.append(dict(arm=arm, plotted_samples=len(r['time']), time_range_s=[float(r['time'][0]), float(r['time'][-1])],
            force_peak_N=float(r['force'].max()), lift_request_s=float(r['result']['lift_request_start_s']),
            force_values_unchanged=True, time_values_unchanged=True, shift_s=0.))
    for x, style in ((7.5, '--'), (9.3, ':'), (12, '--')):
        ax.axvline(x, color='#66737D', lw=.75, ls=style, alpha=.6, zorder=1)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 27.5)
    ax.set_xticks([0, 4, 8, 12, 16] if overview else [0, 2, 4, 6, 7.5, 9.3, 12, 14, 16])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel('实验时间（s）')
    ax.set_ylabel('总指力（N）')
    ax.grid(axis='y', color='#E5E9EC', lw=.6, zorder=0)
    if not overview:
        for x, label in ((3.75, '抓握与抬升'), (9.75, '持稳与运输'), (14, '放下')):
            ax.text(x, .955, label, transform=ax.get_xaxis_transform(), ha='center', va='top', color='#4B5660', fontsize=12)
    return dict(scene=scene, x_range=[0,16], y_range=[0,27.5], series=series)

def save(fig, name, records, notes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt in fig.texts:
        b = txt.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, txt.get_text()
    p = OUT / name
    fig.savefig(p, dpi=240 if len(records) == 1 else 220)
    plt.close(fig)
    with Image.open(p) as im:
        dimensions = list(im.size)
        im.thumbnail((1700,1500), Image.Resampling.LANCZOS)
        im.save(WORK / '预览' / name)
    return dict(path=str(p), sha256=sha(p), dimensions=dimensions, panels=records, notes=notes)

figures = []
for i, (scene, (label, title, extra)) in enumerate(SCENES.items(), 13):
    fig, ax = plt.subplots(figsize=(16,7.5), dpi=240)
    fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.78)
    fig.suptitle(title, fontproperties=TITLE, fontsize=24, y=.975)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,.915), ncol=2, frameon=False, fontsize=14)
    record = plot(ax, scene)
    lifts = {a:rows[scene][a]['result']['lift_request_start_s'] for a in ARMS}
    notes = [
        '注：总指力为五指对目标物法向力之和；保留 0-16 s 的全部 2 kHz 原始记录，未平移、未平滑。',
        f"事件：7.5 s 拿取评价、9.3 s 运输、12 s 放下；蓝、绿虚线标示各自抬升时刻，分别为 {lifts['baseline']:.3f} s、{lifts['macro']:.3f} s。",
        extra,
    ]
    for y, words in zip((.115,.073,.031), notes):
        fig.text(.075, y, words, fontsize=11.5, color='#46545F')
    figures.append(save(fig, f'{i:02d}_{label}_全程总指力_科研版.png', [record], notes))

fig, axs = plt.subplots(2, 2, figsize=(16,12), dpi=220)
fig.subplots_adjust(left=.07, right=.98, bottom=.155, top=.86, hspace=.36, wspace=.18)
fig.suptitle('四种情况下的全程总指力', fontproperties=TITLE, fontsize=24, y=.975)
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,.94), ncol=2, frameon=False, fontsize=14)
records = []
for index, (ax, (scene, (label, _, _))) in enumerate(zip(axs.flat, SCENES.items())):
    records.append(plot(ax, scene, overview=True))
    ax.set_title(f'{chr(65+index)}  {label}', fontproperties=TITLE, fontsize=16, loc='left', pad=12)
notes = [
    '注：总指力为五指对目标物法向力之和；四图统一坐标范围，保留 0-16 s 原始时间，未平滑或平移。',
    '灰线依次标示 7.5 s 拿取评价、9.3 s 运输、12 s 放下；蓝、绿虚线标示各自抬升时刻。',
    '摩擦变化、外载荷与障碍物接近均按 9.3 s 日程发生；易碎限力从 0 s 生效；实际碰撞接触随后发生。',
]
for y, words in zip((.09,.055,.02), notes):
    fig.text(.07, y, words, fontsize=11.5, color='#46545F')
figures.append(save(fig, '17_四种情况_全程总指力总览.png', records, notes))

assert all(sha(p) == h for p, h in inputs.items())
assert all(sha(p) == h for p, h in preserved.items())
audit = dict(status='generated', source_batch=str(BATCH), input_sha256=inputs,
    inputs_unchanged=True, previous_report_files_unchanged=True, previous_file_count=len(preserved),
    simulation_rerun=False, filtering=False, interpolation=False, time_shift=False,
    source_sampling_hz=2000, figures=figures)
(WORK / '四场景全程_科研图核对.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(files=[f['path'] for f in figures], source_records_unchanged=True, previous_outputs_unchanged=True), ensure_ascii=False))

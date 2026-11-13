from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent / 'original_report_not_included'
PROJECT = Path(__file__).resolve().parents[2] / 'projects' / '多任务联合控制研究_2026-11-03'
BATCH = PROJECT / 'output/midterm_core_macro_ablation_v1_verified_20261111'
OUT = ROOT / '受力曲线对照'
WORK = ROOT / '.制作与校验/受力曲线_2026-11-12'
for folder in (OUT, WORK):
    folder.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(WORK / 'mplconfig')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from PIL import Image

for font in (r'C:\Windows\Fonts\times.ttf', r'C:\Windows\Fonts\simsun.ttc',
             r'C:\Windows\Fonts\msyhbd.ttc'):
    fm.fontManager.addfont(font)
TITLE = fm.FontProperties(fname=r'C:\Windows\Fonts\msyhbd.ttc')
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'], 'font.size': 12,
    'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#70808D', 'axes.linewidth': .8, 'axes.labelsize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'path.simplify': False, 'agg.path.chunksize': 0,
})
COLORS = {'pi': '#2767A8', 'core': '#128376'}
SCENES = {'nominal': '正常运输', 'friction_drop': '摩擦变化',
          'fragile_conflict': '易碎限力', 'collision': '碰撞扰动'}
# Same phase windows and scales for every scene and both policies.
PANELS = [
    dict(key='full', title='全程', bounds=(0., 16.), ymax=25., ticks=[0, 4, 8, 12, 16], yticks=[0, 5, 10, 15, 20, 25]),
    dict(key='pickup', title='拿取', bounds=(0., 7.5), ymax=12., ticks=[0, 1.5, 3, 4.5, 6, 7.5], yticks=[0, 3, 6, 9, 12]),
    dict(key='transport', title='运输（含持稳）', bounds=(7.5, 12.), ymax=25., ticks=[7.5, 8, 9, 10, 11, 12], yticks=[0, 5, 10, 15, 20, 25]),
    dict(key='putdown', title='放下', bounds=(12., 16.), ymax=10., ticks=[12, 13, 14, 15, 16], yticks=[0, 2, 4, 6, 8, 10]),
]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

plan = read(BATCH / 'plan.json')
ledger = read(BATCH / 'assignment_ledger.json')
assert read(BATCH / 'execution_complete.json')['execution_complete']
core_metrics = {row['job']['scene']: row['metrics']
                for row in ledger if row['job']['arm'] == 'core'}
protected = {}
for path in (BATCH/'plan.json', BATCH/'assignment_ledger.json', BATCH/'execution_complete.json'):
    protected[str(path)] = sha(path)

records = {}
for scene in SCENES:
    records[scene] = {}
    for role in ('pi', 'core'):
        folder = (Path(plan['references'][scene]['pi']['path']) if role == 'pi'
                  else BATCH / 'runs' / f'core__{scene}__619741')
        expected = (plan['references'][scene]['pi']['hashes'] if role == 'pi'
                    else core_metrics[scene]['input_sha256'])
        for name, expected_hash in expected.items():
            path = folder / name
            assert sha(path) == expected_hash, str(path)
            protected[str(path)] = expected_hash
        meta = read(folder / 'metadata.json')
        assert meta['seed'] == 619741 and meta['scenario']['name'] == scene
        assert meta['motion_mode'] == 'transport'
        assert meta['duration'] == 16. and meta['handoff_time'] == 7.5 and meta['release_at'] == 12.
        trajectory = meta['task_contract']['user_motion_trajectory']
        assert trajectory['start_s'] == 9.3 and trajectory['end_s'] == 12.
        if role == 'core':
            assert meta['experiment_arm'] == 'core'
        else:
            assert meta['task_policy'] == 'legacy'
        with np.load(folder / 'trace.npz', allow_pickle=False) as trace:
            substep_time = trace['substep_time']
            duration = trace['substep_duration']
            normals = np.vstack((trace['normal'][0], trace['substep_normal']))
        assert normals.shape == (32001, 5) and substep_time.shape == (32000,)
        assert np.isfinite(normals).all()
        assert np.allclose(substep_time, np.arange(32000)*.0005, atol=1e-8, rtol=0)
        assert np.array_equal(duration, np.full(32000, .0005))
        time = np.arange(32001)*.0005
        force = normals.sum(axis=1)
        assert force.min() >= 0
        records[scene][role] = dict(folder=str(folder), meta=meta, time=time, force=force)
    assert records[scene]['pi']['meta']['config'] == records[scene]['core']['meta']['config']
    assert records[scene]['pi']['meta']['model_sha256'] == records[scene]['core']['meta']['model_sha256']
    pickup = records[scene]['pi']['time'] <= 7.5
    assert np.array_equal(records[scene]['pi']['force'][pickup], records[scene]['core']['force'][pickup])

evidence = []
for number, (scene, scene_name) in enumerate(SCENES.items(), 1):
    pair = records[scene]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=240)
    fig.subplots_adjust(left=.073, right=.972, bottom=.19, top=.825, hspace=.42, wspace=.19)
    fig.text(.073, .953, scene_name + '  两种策略总指力对照',
             fontproperties=TITLE, fontsize=23, color='#142636', va='center')
    fig.text(.073, .900, '四个子图均以总指力为纵轴：五个手指对物体的法向接触力之和。',
             fontsize=13, color='#536572')
    handles = [
        Line2D([0], [0], color=COLORS['pi'], lw=1.8, ls=(0, (5, 3)), label='力比 PI 基线'),
        Line2D([0], [0], color=COLORS['core'], lw=1.8, label='中期宏观策略'),
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.975, .970),
               frameon=False, ncol=2, fontsize=13.5, handlelength=3.2, columnspacing=2.)
    panel_evidence = []
    for idx, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        start, end = panel['bounds']
        ax.set_xlim(start, end)
        ax.set_ylim(0, panel['ymax'])
        ax.set_xticks(panel['ticks'])
        ax.set_yticks(panel['yticks'])
        ax.set_xlabel('实验时间 (s)', labelpad=5)
        ax.set_ylabel('总指力 (N)', labelpad=8)
        ax.set_title(f"({'abcd'[idx]}) {panel['title']}", loc='left',
                     fontproperties=TITLE, fontsize=15, pad=12, color='#142636')
        ax.grid(axis='both', color='#DCE4EB', lw=.6)
        ax.set_axisbelow(True)
        if panel['key'] == 'full':
            for stage_boundary in (7.5, 12.):
                ax.axvline(stage_boundary, color='#84929D', lw=.9, ls=(0, (2, 3)), zorder=1)
            for centre, label in ((3.75, '拿取'), (9.75, '运输'), (14., '放下')):
                ax.text(centre, .96, label, transform=ax.get_xaxis_transform(),
                        fontsize=11, color='#667784', ha='center', va='top')
        elif panel['key'] == 'pickup':
            ax.text(.5, .95, '共同准备流程；两种策略曲线重合', transform=ax.transAxes,
                    fontsize=11, color='#667784', ha='center', va='top')
        elif panel['key'] == 'transport':
            ax.axvline(9.3, color='#84929D', lw=.9, ls=(0, (2, 3)), zorder=1)
            ax.annotate('运动开始', xy=(9.3, .96), xycoords=('data', 'axes fraction'),
                        xytext=(7, 0), textcoords='offset points', fontsize=11,
                        color='#667784', ha='left', va='top')
        bindings = {}
        for role in ('core', 'pi'):
            row = pair[role]
            mask = (row['time'] >= start-1e-8) & (row['time'] <= end+1e-8)
            xx, yy = row['time'][mask], row['force'][mask]
            assert yy.max() < panel['ymax']
            assert abs(xx[0]-start) < 1e-8 and abs(xx[-1]-end) < 1e-8
            line, = ax.plot(xx, yy, color=COLORS[role], lw=1.15,
                           ls=(0, (5, 3)) if role == 'pi' else '-',
                           solid_capstyle='round', dash_capstyle='butt',
                           zorder=3 if role == 'pi' else 2)
            assert np.array_equal(line.get_xdata(), xx)
            assert np.array_equal(line.get_ydata(), yy)
            bindings[role] = dict(samples=int(xx.size), max_N=float(yy.max()),
                                  source_field='sum(substep_normal, axis=1), with initial normal sample')
        panel_evidence.append(dict(key=panel['key'], title=panel['title'],
                                   interval_s=list(panel['bounds']), y_limits_N=[0., panel['ymax']],
                                   series=bindings))
    fig.text(.073, .092,
             '拿取：0–7.5 s；运输：7.5–12 s（含持稳，9.3 s 开始运动）；放下：12–16 s。',
             fontsize=11.5, color='#536572')
    fig.text(.073, .055,
             '原始 2 kHz 记录，未平滑；四场景同名子图刻度一致，不同阶段的纵轴范围见坐标。',
             fontsize=11.5, color='#536572')
    fig.text(.972, .022,
             '来源：冻结配对实验  ·  同一已见种子 619741  ·  本次仅绘图，未重新仿真',
             fontsize=10.5, color='#6B7984', ha='right')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for node in fig.findobj(matplotlib.text.Text):
        if node.get_visible() and node.get_text():
            box = node.get_window_extent(renderer)
            if box.width > 0 and box.height > 0 and (
                box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1
            ):
                clipped.append(node.get_text())
    assert not clipped, clipped
    # Ensure the bottom labels do not overlap the explanatory footer.
    footer_top = max(text.get_window_extent(renderer).y1 for text in fig.texts[-3:])
    assert min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes[1]) > footer_top+12
    path = OUT / f'{number:02d}_{scene_name}_两种策略受力变化.png'
    fig.savefig(path, dpi=240)
    assert Image.open(path).size == (3840, 2160)
    plt.close(fig)
    evidence.append(dict(scene=scene, file=str(path), sha256=sha(path), image_size=[3840, 2160],
                         quantity='sum of five finger normal forces', panels=panel_evidence,
                         sample_rate_hz=2000, smoothing=False, decimation=False,
                         all_plotted_arrays_match_source=True, pickup_curves_identical=True,
                         all_text_inside_canvas=True, footer_separation_verified=True,
                         sources={role: row['folder'] for role, row in pair.items()}))

for path, expected_hash in protected.items():
    assert sha(path) == expected_hash, path

audit = dict(status='passed', layout='full_pickup_transport_putdown_total_finger_force',
             figures=evidence, input_hashes=protected, source_files_unchanged=True,
             phase_semantics=dict(pickup='common preparation 0-7.5 s',
                                  transport='holding and transport 7.5-12 s; motion starts 9.3 s',
                                  putdown='release request through fixed end 12-16 s'),
             physics_or_controller_imported=False, new_simulations=0,
             font_roles={'Chinese titles': 'Microsoft YaHei', 'Chinese axes and prose': 'SimSun',
                         'Latin and numerals in axes and prose': 'Times New Roman'},
             visual_review=dict(status='pending'))
(WORK / '曲线核对.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / '读图说明.txt').write_text("""四场景总指力对照

正常运输、摩擦变化、易碎限力、碰撞扰动各一张PNG，3840×2160，可插入中期汇报PPT。
每张图按从左到右、从上到下排列：全程、拿取、运输（含持稳）、放下。
所有子图使用同一指标：总指力＝五个手指对物体的法向接触力之和，单位N。
蓝色虚线表示力比PI基线，绿色实线表示中期宏观策略。

时间划分依据保存的实验元数据：
全程：0–16秒。
拿取：0–7.5秒，包含共同抓取、抬起与准备；两种策略尚未接管，曲线完全重合。
运输：7.5–12秒，包含接管后的持稳和运输；实际运输运动从9.3秒开始，图中用灰色虚线标出。
放下：12–16秒，从统一释放请求到实验结束；保留两种策略实际卸力时间的差异。

所有横轴均为原实验时钟，不按各自触台事件平移。
四场景同名子图使用相同坐标范围：全程与运输为0–25N，拿取为0–12N，放下为0–10N。
阶段图对纵轴做适当放大，均从零开始，未截断任何力峰。
采用保存的原始2kHz物理子步记录，未平滑、未抽点。动画使用的200Hz展示记录粒度较低。
总指力之和不是空间矢量合力，也不能直接代替单指限力或逐指力波动指标。

四场景来自已有开发实验的同一种子619741。
未重新运行控制器或仿真，未修改原平台、实验记录或已有视频。
绘图与来源核对记录位于汇报文件夹 .制作与校验/受力曲线_2026-11-12。
""", encoding='utf-8')
print(json.dumps(dict(created=[row['file'] for row in evidence], input_files_verified=len(protected),
                      source_files_unchanged=True, new_simulations=0,
                      panels=['全程', '拿取', '运输（含持稳）', '放下'],
                      quantity='五指法向接触力之和'), ensure_ascii=False))

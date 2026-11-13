"""Render paired saved states without rerunning either controller.

The camera uses qpos only. Force labels always come from the original dense
trace, never from contact forces recalculated by mj_forward during replay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from physics import ROOT, Plant
from scenarios import Scenario, schedule


METHODS = ('ratio','coordinated')
LABELS = {'ratio':'摩擦比基线','coordinated':'事件触发协调'}
COLORS = {'ratio':'#45b7cf','coordinated':'#b497f4'}
CAMERA = dict(lookat=[.045,0.,.40],azimuth=150.,elevation=-15.,distance=.60)
PREDECLARED_SELECTION = dict(declared_at_utc='2026-10-13T11:40:41Z',
                            representative_seed=8001,
                            scenarios=['friction_drop','local_limit'],
                            requested_window_s=[0.,15.],representative_at_s=9.4,
                            selection_basis='Fixed before viewing formal results; no winning-case selection')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font_path(requested):
    choices = [requested] if requested else [Path(r'C:\Windows\Fonts\msyh.ttc'),
                                           Path(r'C:\Windows\Fonts\simhei.ttf')]
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError('Chinese font unavailable; pass --font with a .ttc/.ttf file')


def confirmed_drop_times(trace, columns, result, scenario):
    """Confirm loss from full-rate samples, without inventing a dropout."""
    get = lambda name:trace[:,columns.index(name)]
    times, shifts, heights = get('time'),get('position_error_m'),get('object_z')
    confirmed = np.zeros(len(times),dtype=bool)
    since, dropped = None, False
    for i, instant in enumerate(times):
        hold = scenario.release_at is None or instant<scenario.release_at-1e-8
        if instant>=8. and hold and result['setup_ok']:
            if shifts[i]>.080 or heights[i]<heights[0]+.020:
                if since is None:
                    since = instant
                if instant-since>=.050-1e-8:
                    dropped = True
            else:
                since = None
        confirmed[i] = dropped
    if scenario.release_at is None:
        assert bool(confirmed[-1]) == result['dropped'], 'Trace/drop metadata disagree'
    return confirmed


def read_trial(results_dir, result, scenario, manifest):
    trace_path = (results_dir/result['trace_file']).resolve()
    assert trace_path.is_relative_to(results_dir.resolve())
    with np.load(trace_path) as data:
        trace, columns = data['trace'].copy(),data['columns'].tolist()
        qpos, qtimes = data['qpos'].copy(),data['qpos_time'].copy()
        metadata = json.loads(str(data['metadata']))
    assert metadata['parameters'] == result['parameters']
    assert (metadata['scenario'],metadata['method'],metadata['seed']) == (
        result['scenario'],result['method'],result['seed'])
    assert len(qpos)==len(qtimes)>0 and np.all(np.diff(qtimes)>0.)
    plant = Plant(scenario,result['parameters'],manifest['physics_timestep'])
    assert qpos.shape[1] == plant.model.nq
    # The shared model parks its unused collision fixture outside the workspace.
    # Hide this decorative obstruction in scenes where it never acts; only the
    # in-memory rendering alpha changes, and the model XML stays untouched.
    if scenario.event != 'collision':
        plant.model.geom_rgba[plant.obstacle_geom,3] = 0.
    renderer = mujoco.Renderer(plant.model,height=450,width=600)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA['lookat']
    for key in ('azimuth','elevation','distance'):
        setattr(camera,key,CAMERA[key])
    return dict(result=result,path=trace_path,trace=trace,columns=columns,qpos=qpos,qtimes=qtimes,
                plant=plant,renderer=renderer,camera=camera,
                confirmed_drop=confirmed_drop_times(trace,columns,result,scenario))


def render_panel(trial, scenario, requested_time, fonts):
    qindex = int(np.argmin(abs(trial['qtimes']-requested_time)))
    instant = float(trial['qtimes'][qindex])
    trace = trial['trace']
    tindex = int(np.argmin(abs(trace[:,trial['columns'].index('time')]-instant)))
    assert abs(float(trace[tindex,0])-instant)<1e-8
    row = trace[tindex]
    normal = np.array([row[trial['columns'].index(f'normal_{i}')] for i in range(5)])
    plant = trial['plant']
    plant.data.qpos[:] = trial['qpos'][qindex]
    plant.data.qvel[:] = 0.
    plant.data.time = instant
    env = schedule(scenario,trial['result']['parameters'],instant)
    plant.set_friction(env['friction'])
    plant.data.mocap_pos[plant.obstacle_mocap] = env['obstacle']
    mujoco.mj_forward(plant.model,plant.data)
    trial['renderer'].update_scene(plant.data,camera=trial['camera'])
    pixels = trial['renderer'].render()
    panel = Image.new('RGB',(600,630),'#13202b')
    panel.paste(Image.fromarray(pixels),(0,42))
    draw = ImageDraw.Draw(panel)
    method = trial['result']['method']
    draw.rectangle((0,0,599,4),fill=COLORS[method])
    draw.text((14,10),LABELS[method],font=fonts['label'],fill=COLORS[method])
    draw.text((450,12),f't={instant:05.2f}s',font=fonts['small'],fill='#f4f7fa')
    total = float(normal.sum())
    over = bool(np.any(normal>np.array(scenario.limit)+.25) or total>scenario.budget+.5)
    draw.text((14,505),f'五指合力 {total:.2f} N / 预算 {scenario.budget:.2f} N',
              font=fonts['label'],fill='#f4f7fa')
    draw.text((14,536),'逐指 N/上限：'+' · '.join(f'{n:.1f}/{lim:.1f}' for n,lim in zip(normal,scenario.limit)),
              font=fonts['small'],fill='#c7d5df')
    dropped = bool(trial['confirmed_drop'][tindex])
    state = '已确认掉落' if dropped else '未确认掉落'
    if instant>=8. and not trial['result']['setup_ok']:
        state = '初始抓握未建立'
    status = f'{state} | 当前容差判据超限：'+('是' if over else '否')
    draw.text((14,566),status,font=fonts['small'],fill='#ffad98' if dropped or over else '#a8dfce')
    force_peak = float(trial['result']['physics_peak_force_N'])
    penetration_peak = float(trial['result']['physics_peak_penetration_mm'])
    assert np.isfinite(force_peak) and np.isfinite(penetration_peak)
    draw.text((14,597),f'整程物理子步力峰值 {force_peak:.2f} N · 穿透 {penetration_peak:.2f} mm',
              font=fonts['small'],fill='#a8bdcf')
    values = dict(time=instant,normal_N=normal.tolist(),normal_limits_N=list(scenario.limit),
                  five_finger_sum_N=total,budget_N=scenario.budget,
                  confirmed_drop=dropped,instantaneous_tolerance_violation=over,
                  friction=env['friction'],applied_force_N=env['force'].tolist(),
                  qpos_index=qindex,trace_index=tindex)
    return panel,values


def compose_frame(panels, scenario, seed, fonts, fps):
    frame = Image.new('RGB',(1232,766),'#0b141e')
    draw = ImageDraw.Draw(frame)
    draw.text((16,12),scenario.title,font=fonts['title'],fill='#f4f7fa')
    draw.text((16,47),f'固定代表种子 {seed} · 相同环境参数 · 保存状态回放',font=fonts['small'],fill='#a8bdcf')
    for index,panel in enumerate(panels):
        frame.paste(panel,(16+index*600,78))
    draw.text((16,716),'当前超限容差：逐指上限 +0.25 N / 总预算 +0.5 N；连续 30 ms 才记全程限力不合格。',
              font=fonts['note'],fill='#a8bdcf')
    draw.text((16,742),f'{fps:g} Hz 可视回放不呈现全部尖峰；峰值来自原始物理子步统计。单例不能代替全矩阵统计。',
              font=fonts['note'],fill='#a8bdcf')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scenarios',nargs='+',default=['friction_drop','local_limit'])
    parser.add_argument('--seed',type=int,default=8001)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--end',type=float,default=15.)
    parser.add_argument('--fps',type=float,default=5.)
    parser.add_argument('--representative-at',type=float,default=9.4)
    parser.add_argument('--font',type=Path)
    args = parser.parse_args()
    assert 0.<=args.start<=args.end and 0.<args.fps<=20.
    results_dir = args.results.resolve()
    manifest = json.loads((results_dir/'manifest.json').read_text(encoding='utf-8'))
    results = json.loads((results_dir/'trials.json').read_text(encoding='utf-8'))
    definitions = {s['name']:s for s in manifest['scenario_definitions']}
    assert sha256(ROOT.parent/'五指假肢抓握仿真/model.xml') == manifest['source_model_sha256']
    for name in ('physics.py','scenarios.py','controllers.py','run_experiments.py'):
        assert sha256(ROOT/name) == manifest['code_sha256'][name], f'Replay geometry/schedule changed: {name}'
    assert args.seed in manifest['seeds']
    for scenario_name in args.scenarios:
        assert scenario_name in manifest['scenarios'], scenario_name
        for method in METHODS:
            assert any(r['scenario']==scenario_name and r['method']==method and r['seed']==args.seed for r in results)
    args.out.mkdir(parents=True,exist_ok=True)
    font_file = font_path(args.font)
    fonts = {key:ImageFont.truetype(str(font_file),size) for key,size in
             [('title',25),('label',20),('small',17),('note',15)]}
    matches_predeclared = (args.seed == PREDECLARED_SELECTION['representative_seed'] and
                           args.scenarios == PREDECLARED_SELECTION['scenarios'] and
                           [args.start,args.end] == PREDECLARED_SELECTION['requested_window_s'] and
                           args.representative_at == PREDECLARED_SELECTION['representative_at_s'])
    report = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  source_results=str(results_dir),source_suite=manifest['suite'],
                  representative_seed=args.seed,
                  selection='Predeclared fixed example' if matches_predeclared else 'Manually requested alternative; not the predeclared example',
                  predeclared_selection=PREDECLARED_SELECTION,
                  matches_predeclared_selection=matches_predeclared,
                  compared_methods=list(METHODS),method_labels=LABELS,
                  camera=CAMERA,render_resolution_per_method=[600,450],fps=args.fps,
                  rendering_only_style='Hide inactive collision fixture outside collision scenarios',
                  instantaneous_threshold_margins_N={'per_finger':.25,'five_finger_budget':.5},
                  requested_window_s=[args.start,args.end],representative_at_s=args.representative_at,
                  script_sha256=sha256(__file__),font_path=str(font_file),font_sha256=sha256(font_file),
                  source_hashes={'manifest.json':sha256(results_dir/'manifest.json'),
                                 'trials.json':sha256(results_dir/'trials.json'),
                                 'physics.py':sha256(ROOT/'physics.py'),'scenarios.py':sha256(ROOT/'scenarios.py'),
                                 'controllers.py':sha256(ROOT/'controllers.py'),
                                 'run_experiments.py':sha256(ROOT/'run_experiments.py'),
                                 'model.xml':sha256(ROOT.parent/'五指假肢抓握仿真/model.xml')},
                  interpretation='Observational saved-state example, not full-matrix evidence; raw force labels are original samples. Sparse visualization (default 5 Hz) does not show every force spike; episode maxima are the original physical-substep summaries, never replay recomputation.',
                  scenarios=[])
    for scenario_name in args.scenarios:
        spec = {**definitions[scenario_name],'limit':tuple(definitions[scenario_name]['limit'])}
        scenario = Scenario(**spec)
        trials = []
        try:
            for method in METHODS:
                result = next(r for r in results if (r['scenario'],r['method'],r['seed'])==(scenario_name,method,args.seed))
                trials.append(read_trial(results_dir,result,scenario,manifest))
            assert trials[0]['result']['parameters'] == trials[1]['result']['parameters']
            final_time = min(args.end,*(float(t['qtimes'][-1]) for t in trials))
            assert final_time>=args.start
            targets = np.arange(args.start,final_time+1e-8,1./args.fps)
            frames,records = [],[]
            for instant in targets:
                rendered = [render_panel(trial,scenario,float(instant),fonts) for trial in trials]
                frames.append(compose_frame([item[0] for item in rendered],scenario,args.seed,fonts,args.fps))
                records.append({method:item[1] for method,item in zip(METHODS,rendered)})
            stem = f'{scenario_name}__ratio_vs_coordinated__{args.seed}'
            gif_path, png_path = args.out/f'{stem}.gif',args.out/f'{stem}.png'
            frames[0].save(gif_path,save_all=True,append_images=frames[1:],duration=round(1000./args.fps),loop=0,optimize=False)
            representative_index = int(np.argmin(abs(targets-args.representative_at)))
            frames[representative_index].save(png_path)
            entry = dict(name=scenario_name,title=scenario.title,frames=len(frames),
                         actual_window_s=[float(targets[0]),float(targets[-1])],
                         representative_frame=representative_index,representative_time_s=float(targets[representative_index]),
                         gif=str(gif_path.resolve()),png=str(png_path.resolve()),
                         output_sha256={'gif':sha256(gif_path),'png':sha256(png_path)},
                         trace_sha256={t['result']['method']:sha256(t['path']) for t in trials},
                         original_substep_peaks={t['result']['method']:{
                             'physics_peak_force_N':float(t['result']['physics_peak_force_N']),
                             'physics_peak_penetration_mm':float(t['result']['physics_peak_penetration_mm'])}
                             for t in trials},
                         original_parameters=trials[0]['result']['parameters'],samples=records)
            report['scenarios'].append(entry)
            print(json.dumps({k:entry[k] for k in ('name','frames','gif','png')},ensure_ascii=False),flush=True)
        finally:
            for trial in trials:
                trial['renderer'].close()
    path = args.out/'render_manifest.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Replay provenance: {path.resolve()}',flush=True)


if __name__=='__main__':
    main()

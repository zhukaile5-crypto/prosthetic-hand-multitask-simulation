"""Predeclared fixed release-window TV; no original metric or score changes."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'review/v21_small_plan_v1.json'
CONTRACT = ROOT/'review/v21_release_variation_contract_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def calculate(time, normal, skin):
    t = np.asarray(time, float)
    if t.ndim != 1 or not np.isfinite(t).all():
        return dict(status='unknown_clock', target_normal_TV_N=None, all_actor_skin_TV_N=None)
    mask = (t >= 12.-1e-9) & (t <= 16.+1e-9)
    selected = t[mask]
    expected = 12.+np.arange(801)*.005
    if selected.shape != (801,) or np.max(abs(selected-expected)) > 1e-9:
        return dict(status='unknown_incomplete_fixed_window', target_normal_TV_N=None, all_actor_skin_TV_N=None)
    out = dict(status='complete_fixed_200Hz_release_window', samples=801, window_s=[12., 16.])
    for name, values in [('target_normal_TV_N', normal), ('all_actor_skin_TV_N', skin)]:
        array = np.asarray(values, float)
        if array.shape != (len(t), 5) or not np.isfinite(array[mask]).all():
            out[name] = None
            out['status'] = 'unknown_force_field'
        else:
            out[name] = float(np.abs(np.diff(array[mask], axis=0)).sum())
    return out


def ready(plan):
    if plan.get('planned') != 9 or len(plan.get('stages', [])) != 9:
        raise ValueError('original registered9 assignments required')
    expected = {(variant, scene, motion, seed) for variant in ('v20', 'prefix', 'v21')
        for scene, motion, seed in [('nominal', 'transport', 619741),
            ('fragile_conflict', 'hold', 619741), ('friction_drop', 'transport', 619742)]}
    actual = []
    for stage in plan['stages']:
        a = stage['assignment']
        if (a['methods'] != ['ratio_effort'] or len(a['scenarios']) != 1 or len(a['seeds']) != 1
                or a['task_policy'] != 'integrated' or a['receiver_policy'] != 'coordinated'
                or not a['physical_randomization'] or a['release_variant'] != stage['variant']):
            raise ValueError('one registered same-information assignment required')
        actual.append((stage['variant'], a['scenarios'][0], a['motion_mode'], a['seeds'][0]))
    if len(set(actual)) != 9 or set(actual) != expected:
        raise ValueError('all3 variants x all3 original contexts exactly once required')
    markers = {}
    for s in plan['stages']:
        p = ROOT/s['output']/'execution_complete.json'
        if not p.is_file():
            return False, {}
        m = read(p)
        if not (m.get('execution_complete') and m.get('sources_unchanged')
                and m.get('runtime_failures') == [] and m.get('trials') == m.get('planned') == 1):
            return False, {}
        markers[s['name']] = sha(p)
    return True, markers


def collect(plan):
    valid, markers = ready(plan)
    if not valid:
        raise ValueError('all9 terminal before trace/result reads')
    if plan['release_variation_contract_sha256'] != sha(CONTRACT):
        raise ValueError('predeclared release variation contract changed')
    old = read(ROOT/'review/v17b_prospective80_plan_v1.json')
    rows, refs, issues = [], {}, []

    def row(run):
        meta = read(run/'metadata.json')
        if sha(run/'trace.npz') != meta['trace_sha256']:
            raise ValueError('real trace identity mismatch')
        with np.load(run/'trace.npz', allow_pickle=False) as z:
            value = calculate(z['time'], z['normal'], z['skin_normal'])
        return meta, value, sha(run/'trace.npz')

    for s in plan['stages']:
        a = s['assignment']; scene = a['scenarios'][0]; seed = a['seeds'][0]
        registration = read(ROOT/s['registration'])
        if registration['source_sha256'] != plan['source_sha256'] or registration['assignment'] != a:
            raise ValueError('registered source or assignment changed')
        run = ROOT/s['output']/'runs'/f'{scene}__ratio_effort__{seed}'
        meta, value, digest = row(run)
        if (meta['research_source_sha256'] != plan['source_sha256']
                or meta['release_variant'] != a['release_variant']
                or meta['seed'] != seed or meta['motion_mode'] != a['motion_mode']):
            raise ValueError('assigned variant/source/context differs')
        key = (scene, a['motion_mode'], seed)
        if key not in refs:
            stage = next(t for t in old['stages'] if t['name'] == 'baselines_'+a['motion_mode'])
            original = ROOT/stage['output']/'runs'/f'{scene}__ratio_effort__{seed}'
            rm, rv, rd = row(original)
            refs[key] = dict(context_key=list(key), run=str(original), metrics=rv,
                trace_sha256=rd, config=rm['config'], noise=rm['noise'], limits=rm['limits'], budget=rm['budget'])
        ref = refs[key]
        paired = all(meta[k] == ref[k] for k in ('config', 'noise', 'limits', 'budget'))
        if not paired or value['status'] != 'complete_fixed_200Hz_release_window':
            issues.append(s['name']+'/unpaired_or_unknown')
        rows.append(dict(stage=s['name'], variant=s['variant'], context_key=list(key),
            metrics=value, trace_sha256=digest, same_physical_noise_limits=paired,
            original_task_success=read(run/'result.json')['task_success'],
            reference_PI_metrics=ref['metrics']))
    return dict(schema='v21_release_force_variation_readout_v1', complete=not issues,
        records=rows, reference_PI_records=list(refs.values()), issues=issues,
        plan_sha256=sha(PLAN), contract_sha256=sha(CONTRACT), reader_sha256=sha(__file__),
        completion_marker_sha256=markers, new_metric_is_predeclared_diagnostic=True,
        original_metrics_changed=False, physics_steps=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--out', required=True)
    args = ap.parse_args(); out = ROOT/args.out
    if out.exists():
        raise FileExistsError(out)
    if args.self_test:
        t = 12.+np.arange(801)*.005; z = np.zeros((801, 5)); step = z.copy(); step[400:, 0] = 2.
        zig = z.copy(); zig[1::2, 2] = 1.
        bad = z.copy(); bad[4, 1] = np.nan
        tests = [calculate(t,z,z)['target_normal_TV_N'] == 0.,
            calculate(t,step,z)['target_normal_TV_N'] == 2.,
            calculate(t,z,zig)['all_actor_skin_TV_N'] == 800.,
            calculate(t,bad,z)['target_normal_TV_N'] is None,
            calculate(t[:-1],z[:-1],z[:-1])['target_normal_TV_N'] is None]
        value = dict(schema='v21_release_variation_reader_software_v1', passed=all(tests),
            checks=len(tests), reader_sha256=sha(__file__), contract_sha256=sha(CONTRACT),
            physics_steps=0, new_results_read=False)
    elif args.execute:
        value = collect(read(PLAN))
    else:
        raise ValueError('explicit self-test or execute required')
    with out.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(passed=value.get('passed'), complete=value.get('complete'), path=str(out))))
    return 1 if value.get('passed') is False or args.execute and not value.get('complete') else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Recompute foundation probe gates from raw arrays without probe helpers.

This reads existing physical development probes. It does not run policies or
advance a simulator, and a raw-data pass is not a complete physical certificate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()

def sustained_clearance(t,clear,origin,dwell):
    start=None
    for i in np.flatnonzero(t>=origin-1e-9):
        if not clear[i]:
            start=None
        else:
            if start is None:start=float(t[i])
            if t[i]-start>=dwell-1e-9:return float(t[i]-origin)
    return None

def read_probe(path,protocol):
    source=Path(path)
    if source.is_dir():source=source/'raw_probe.npz'
    with np.load(source,allow_pickle=False) as saved:
        a={key:saved[key].copy() for key in saved.files}
    meta=json.loads(str(a['metadata']))
    cfg=meta['config'];foundation=protocol['physical_foundation'];g=foundation['gates']
    t=np.asarray(a['time']);normal=np.asarray(a['normal']);dt=float(cfg['control_dt'])
    physical_dt=float(cfg['timestep']);duration=float(meta['duration'])
    origin=float(meta['release_at']);count=len(t)
    subtime=np.asarray(a['substep_time']);subduration=np.asarray(a['substep_duration'])
    subnormal=np.asarray(a['substep_normal'])
    required=['normal','skin_normal','all_hand_object','support','penetration','object_z','lift','alpha','warnings','qpos','qvel','ctrl',
              'substep_normal','substep_skin_normal','substep_all_hand_object','substep_support','substep_penetration']
    checks={'critical_arrays_finite':all(np.isfinite(a[k]).all() for k in required),
            'control_clock':bool(count==round(duration/dt)+1 and np.allclose(t,np.arange(count)*dt,atol=1e-8,rtol=0)),
            'physical_clock':bool(len(subtime)==round(duration/physical_dt) and
                np.allclose(subtime,np.arange(len(subtime))*physical_dt,atol=1e-8,rtol=0) and
                np.allclose(subduration,physical_dt,atol=1e-12,rtol=0)),
            'normal_shape':normal.shape==(count,5),
            'substep_force_shape':subnormal.shape==(len(subtime),5),
            'unfiltered_forces_nonnegative':bool(np.all(normal>=-1e-9) and np.all(subnormal>=-1e-9)),
            'zero_numeric_warnings':bool(np.max(a['warnings'])<=g['numerical_warnings_max']),
            'penetration_within_bound':bool(np.max(a['substep_penetration'])<=g['max_hand_object_penetration_m']),
            'no_hidden_control_clip':bool(float(meta.get('hidden_ctrl_clip',np.inf))<=1e-10)}
    start,end=foundation['static_window_s']
    selected=(t>=start-1e-9)&(t<=end+1e-9)
    expected=round((end-start)/dt)+1
    contact=normal>g['digit_contact_min_N']
    support=(contact.sum(axis=1)>=g['minimum_supported_digits'])&contact[:,g['required_digit']]&(
        a['all_hand_object']>g['digit_contact_min_N'])
    held=support&(a['support']<=g['table_support_max_N'])&(
        np.abs(a['lift']-foundation['target_lift_m'])<=g['lift_error_max_m'])&(
        a['object_z']-a['object_z'][0]>=g['minimum_object_lift_from_initial_m'])
    checks['static_window_full_count']=bool(selected.sum()==expected)
    coverage=float(np.mean(held[selected])) if selected.any() else 0.
    checks['static_supported_lift_coverage']=coverage>=g['static_support_coverage_min']
    at=int(np.argmin(np.abs(t-origin)))
    checks['release_command_endpoint_present']=bool(abs(t[at]-origin)<=1e-8)
    checks['supported_at_release_command']=bool(support[at] and a['support'][at]<=g['table_support_max_N'] and
        a['object_z'][at]-a['object_z'][0]>=g['minimum_object_lift_from_initial_m'])
    clear=a['all_hand_object']<=g['release_all_hand_object_force_max_N']
    latency=sustained_clearance(t,clear,origin,g['release_dwell_s'])
    checks['all_hand_release_confirmed']=bool(latency is not None and latency<=g['release_deadline_s']+1e-9)
    bindings=meta.get('source_sha256',meta.get('source_hashes',{}))
    binding_checks={}
    for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py'):
        binding_checks[name]=bool(name in bindings and (ROOT/name).is_file() and sha(ROOT/name)==bindings[name])
    row={'path':str(source.resolve()),'raw_sha256':sha(source),'config':cfg,'checks':checks,
         'raw_gates_passed':all(checks.values()),'held_coverage':coverage,'held_samples':int(selected.sum()),
         'held_mean_force':np.mean(normal[selected],axis=0).tolist(),
         'held_total_mean_force':float(np.mean(normal[selected].sum(axis=1))),
         'held_minimum_object_lift_m':float(np.min((a['object_z']-a['object_z'][0])[selected])),
         'held_maximum_lift_error_m':float(np.max(np.abs(a['lift'][selected]-foundation['target_lift_m']))),
         'finger_peak_force':np.max(subnormal,axis=0).tolist(),'total_peak_force':float(np.max(subnormal.sum(axis=1))),
         'max_hand_object_penetration_m':float(np.max(a['substep_penetration'])),
         'release_latency_s':latency,'independent_release_uses_endpoint_time_difference':True,
         'metadata_release_latency_s':meta.get('release_latency'),'model_sha256':meta.get('model_sha256'),
         'source_binding_checks':binding_checks,'sources_match_current':all(binding_checks.values()),
         'declared_source_sha256':bindings}
    # Diagnose the entire-lifecycle peak by phase without changing the gate.
    pre=subtime<origin-1e-9
    post=~pre
    row['peak_by_phase']={name:{'finger':np.max(subnormal[mask],axis=0).tolist(),
                               'total':float(np.max(subnormal[mask].sum(axis=1)))}
                          for name,mask in [('before_release',pre),('after_release',post)] if mask.any()}
    return row

def rel_diff(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.max(np.abs(a-b)/np.maximum(.1,np.maximum(np.abs(a),np.abs(b)))))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',nargs='+',required=True,type=Path)
    parser.add_argument('--protocol',type=Path,default=ROOT/'acceptance_protocol.json')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    records=[read_probe(p,protocol) for p in args.source]
    buckets={}
    for row in records:
        c=row['config'];key=(c['shape'],float(c['solref']),float(c.get('size',1.)),float(c.get('mass',.12)))
        buckets.setdefault(key,[]).append(row)
    pairs=[];g=protocol['physical_foundation']['gates']
    declared_steps=sorted(map(float,protocol['physical_foundation']['physical_timesteps_s']),reverse=True)
    if len(declared_steps)!=2:raise ValueError('This audit expects a declared coarse/fine physical-step pair.')
    coarse_dt,fine_dt=declared_steps
    for key,rows in buckets.items():
        by_dt={float(row['config']['timestep']):row for row in rows}
        if coarse_dt in by_dt and fine_dt in by_dt:
            coarse,fine=by_dt[coarse_dt],by_dt[fine_dt]
            mean_error=max(rel_diff(coarse['held_mean_force'],fine['held_mean_force']),rel_diff(coarse['held_total_mean_force'],fine['held_total_mean_force']))
            peak_error=max(rel_diff(coarse['finger_peak_force'],fine['finger_peak_force']),rel_diff(coarse['total_peak_force'],fine['total_peak_force']))
            coarse_config={k:v for k,v in coarse['config'].items() if k!='timestep'}
            fine_config={k:v for k,v in fine['config'].items() if k!='timestep'}
            checks={'same_configuration_except_timestep':coarse_config==fine_config,
                    'same_task_outcome':coarse['raw_gates_passed']==fine['raw_gates_passed'],
                    'both_probe_gates_passed':coarse['raw_gates_passed'] and fine['raw_gates_passed'],
                    'mean_force_step_sensitivity':mean_error<=g['step_halving_static_mean_force_relative_difference_max'],
                    'peak_force_step_sensitivity':peak_error<=g['step_halving_peak_force_relative_difference_max']}
            pairs.append({'key':list(key),'checks':checks,'passed':all(checks.values()),
                          'coarse_timestep':coarse_dt,'fine_timestep':fine_dt,
                          'max_relative_mean_difference':mean_error,'max_relative_peak_difference':peak_error,
                          'relative_denominator':'max(0.1 N, abs(coarse), abs(fine)), maximum across digits and total'})
    expected={(shape,float(solref),float(dt)) for shape in protocol['physical_foundation']['shapes']
              for solref in protocol['physical_foundation']['contact_time_constants_s'] for dt in protocol['physical_foundation']['physical_timesteps_s']}
    observed={(r['config']['shape'],float(r['config']['solref']),float(r['config']['timestep'])) for r in records}
    matrix_complete=observed==expected and len(records)==len(expected)
    result={'schema':'independent-foundation-raw-audit-v1','checked_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':sha(args.protocol),'source_sha256':sha(Path(__file__)),
            'declared_matrix_complete':matrix_complete,'probe_count':len(records),'step_pair_count':len(pairs),
            'all_observed_raw_gates_passed':all(r['raw_gates_passed'] for r in records),
            'all_raw_sources_match_current':all(r['sources_match_current'] for r in records),
            'all_observed_step_pairs_passed':bool(pairs) and all(p['passed'] for p in pairs),
            'full_raw_matrix_passed':matrix_complete and all(r['raw_gates_passed'] for r in records) and len(pairs)==6 and all(p['passed'] for p in pairs),
            'limitations':['No simulation, policy execution or probe helper was imported.',
                           'This is raw-record arithmetic only; free-object structure, finite actuation and model-fit acceptance need separate evidence.',
                           'A failed, missing or partial matrix is not an accepted three-shape foundation.'],
            'probes':records,'timestep_pairs':pairs}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=(args.out or ROOT/f'review/independent_foundation_raw_{stamp}.json').resolve()
    if not out.is_relative_to(ROOT.resolve()) or out.exists():raise SystemExit('A fresh output inside the integration directory is required.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'probe_count':len(records),'observed_raw_gates_passed':result['all_observed_raw_gates_passed'],
                      'matrix_complete':matrix_complete,'full_raw_matrix_passed':result['full_raw_matrix_passed'],'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()

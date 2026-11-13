"""Same eight synthetic helper scopes plus necessary cache/AST regression.
Never reads any real cohort output, cached score or NPZ.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import ast,hashlib,importlib.util,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'review/read_v15A_paired_prospective_v1.py'
EXPECTED='58e425e8713c5d83cb4477ac287143d135eeb3009651106f6177652e51b4d7fc'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
spec=importlib.util.spec_from_file_location('paired_helper_independent_v2',SOURCE);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
plan=json.loads((ROOT/'review/v15A_prospective_multiscene_plan_v2.json').read_text(encoding='utf-8'))
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
def valid_bundle(path):
    path=Path(path);scene,method,seed=path.name.split('__');candidate='candidate_A' in str(path);motion='transport' if 'transport' in str(path) else 'hold'
    identity=dict(method=method,seed=int(seed),shape='capsule',motion_mode=motion,task_policy='integrated' if candidate else 'legacy',receiver_policy='blind',scenario=dict(name=scene),run_id='synthetic-'+path.name,split='robustness')
    d=dict(status='measurements_computed_no_new_success_label',measurement_source_sha256=h.AUX_SHA,identity=identity,
        original_result=dict(task_success=False,record_complete=True,research_metrics=dict(record_complete=True)),
        measurements={w:dict(unselected_force=dict(available=False,reason='synthetic_unavailable'),physical_object=dict(available=False)) for w in ('static','dynamic','whole_holding')})
    return dict(status='read_from_stable_completed_or_retained_error_inputs',diagnostics=d,physical_context=dict(same=True),timing=None)
def fake_build(*,pending_transport=False,bad_one=False):
    calls=[]
    def read(p):
        if str(p)=='synthetic_plan':return deepcopy(plan)
        return dict(execution_complete=not(pending_transport and 'transport' in str(p)))
    def bundle(path,aux,cache):
        calls.append(str(path));b=valid_bundle(path)
        if bad_one and len(calls)==1:b['diagnostics']=dict(status='assigned_trace_missing_or_unexecuted',original_result=dict(task_success=True),measurements=None)
        return b
    with patch.object(h,'read',read),patch.object(h,'load_auxiliary',lambda:object()),patch.object(h,'checked_bundle',bundle),patch.object(h,'sha',lambda p:'synthetic-fingerprint'),patch.object(Path,'is_file',lambda p:True):return h.build_readout('synthetic_plan',Path('synthetic_cache'))

good=fake_build();ck('64_records_48_pairs_and_same_ratio_kept_distinct',good['complete'] and len(good['records'])==64 and len(good['pairs'])==48 and len({r['record_id'] for r in good['records']})==64 and sum(p['primary_same_preference_contrast'] for p in good['pairs'])==16)
part=fake_build(pending_transport=True);ck('complete_hold32_partial24pairs_without_overall_final',not part['complete'] and part['summaries'] is None and set(part['completed_strata_descriptive'])=={'hold'})
missing=fake_build(bad_one=True);ck('truthy_missing_trace_diagnostics_cannot_count_as_final_readable',not missing['complete'] and missing['completion']['readable_records']==63 and missing['summaries'] is None,dict(actual_complete=missing['complete'],readable_records=missing['completion']['readable_records']))
b=dict(diagnostics=dict(original_result=dict(research_metrics=dict(static_quality=dict(eligible=False,raw_200Hz=dict(finger_TV_rate_N_s=123.)))),measurements=dict(static=dict(unselected_force=dict(available=True,finger_TV_rate_N_s=0.),physical_object={}))))
v=h.metric_values(b);ck('eligible_NULL_separate_from_all_assigned_measured_zero',v['eligible.static.raw_200Hz.finger_TV_rate_N_s'] is None and v['raw.static.finger_TV_rate_N_s']==0.)
ck('relative_baseline_zero_and_missing_remain_unknown',h.difference(1.,0.)['relative_delta'] is None and not h.difference(None,0.)['available'])
timing=h.timing_stats(dict(control_ms=np.array([100.,1.,1.,1.,20.])),[0,4]);ck('actual_decision_indices_not_all5ms_dilution',timing['control_ms']['count']==2 and timing['control_ms']['median']==60.)
text=SOURCE.read_text(encoding='utf-8');ck('cache_key_static_input_aux_helper_binding',all(x in text for x in ('inputs=inputs,aux=AUX_SHA,helper=sha(__file__)','joint_decisions.jsonl.gz','delivery_intervals.npz','pending_inputs_changed_during_read')))
ck('two_seed_descriptive_no_CI_contract',good['statistical_scope']['physical_seed_clusters']==2 and good['statistical_scope']['ci_or_pvalues'] is False and good['statistical_scope']['independent_context_n16_claim'] is False)
old=ROOT/'review/revisions/v15A_paired_helper_746b_before_ready_fix.py'
def defs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
a,z=defs(old),defs(SOURCE);ck('three_computation_functions_AST_unchanged_from746',sha(old)==h.COMPATIBLE_CACHE_HELPER_SHA and all(a[k]==z[k] for k in ('bundle','timing_stats','metric_values')) and h.compatible_cache_functions())
valid=valid_bundle(Path('candidate_A_hold/nominal__ratio_effort__619731'))
bad=[]
for mutation in ('missing_identity','bad_source','record_incomplete','missing_window','pending_status'):
    q=deepcopy(valid)
    if mutation=='missing_identity':q['diagnostics']['identity'].pop('shape')
    elif mutation=='bad_source':q['diagnostics']['measurement_source_sha256']='other'
    elif mutation=='record_incomplete':q['diagnostics']['original_result']['research_metrics']['record_complete']=False
    elif mutation=='missing_window':q['diagnostics']['measurements'].pop('dynamic')
    else:q['status']='pending_unexecuted_or_result_not_finished'
    bad.append(not h.bundle_ready(q))
ck('actual_schema_failed_task_still_ready_unknown_or_incomplete_not_ready',h.bundle_ready(valid) and all(bad))

with tempfile.TemporaryDirectory(prefix='v15A_helper_pure_') as td:
    folder=Path(td)/'candidate_A_hold/nominal__ratio_effort__619731';folder.mkdir(parents=True);cache=Path(td)/'cache';cache.mkdir()
    names=('trace.npz','metadata.json','result.json','delivery_intervals.npz','delivery_summary.json','joint_decisions.jsonl.gz','runtime_error.json')
    for name in names:(folder/name).write_bytes(('synthetic '+name).encode())
    inputs={n:h.fingerprint(folder/n) for n in names}
    def key(helper):return hashlib.sha256(json.dumps(dict(path=str(folder.resolve()),inputs=inputs,aux=h.AUX_SHA,helper=helper),sort_keys=True).encode()).hexdigest()
    prior=deepcopy(valid);prior.update(inputs=inputs,cache_key=key(h.COMPATIBLE_CACHE_HELPER_SHA));oldcache=cache/(prior['cache_key']+'.json');oldcache.write_text(json.dumps(prior),encoding='utf-8')
    with patch.object(h,'bundle',side_effect=AssertionError('valid old cache should be reused')):
        reused=h.checked_bundle(folder,object(),cache)
    ck('old746_ready_cache_reused_only_with_exact_input_binding',h.bundle_ready(reused) and reused['compatible_cache_helper_sha256']==h.COMPATIBLE_CACHE_HELPER_SHA and reused['inputs']==inputs)
    oldcache.unlink();current=cache/(key(EXPECTED)+'.json');invalid=deepcopy(prior);invalid['diagnostics']['status']='assigned_trace_missing_or_unexecuted';current.write_text(json.dumps(invalid),encoding='utf-8');invalidsha=sha(current);calls=[]
    def fresh(path,aux,where):calls.append(str(where));return deepcopy(valid)
    with patch.object(h,'bundle',fresh):retried=h.checked_bundle(folder,object(),cache)
    ck('invalid_current_retained_and_fresh_read_requested',h.bundle_ready(retried) and len(calls)==1 and 'unavailable_retries' in calls[0] and any(invalidsha in p.name for p in (cache/'invalid_retained').iterdir()))
    current.unlink();oldcache.write_text(json.dumps(prior),encoding='utf-8');(folder/'trace.npz').write_bytes(b'changed synthetic input');calls.clear()
    with patch.object(h,'bundle',fresh):changed=h.checked_bundle(folder,object(),cache)
    ck('changed_input_cannot_borrow_old_cache',len(calls)==1 and 'compatible_cache_helper_sha256' not in changed)

report=dict(schema='v15A_paired_helper_independent_v2',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=EXPECTED,source_unchanged=sha(SOURCE)==EXPECTED,
    checker_sha256=sha(__file__),scope='Same eight synthetic scopes plus readiness schema/three unchanged ASTs and synthetic-tempfile cache admission. No real cohort outputs/cache/NPZ or scores read.',physics_steps=0,model_imported='mujoco' in sys.modules)
out=ROOT/'review/v15A_paired_helper_independent_v2.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

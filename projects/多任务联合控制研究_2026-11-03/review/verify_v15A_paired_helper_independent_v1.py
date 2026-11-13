"""Pure synthetic paired-helper review. Never reads cohort run outputs."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'review/read_v15A_paired_prospective_v1.py'
EXPECTED='746b16dcc99869325ec1ada087b1990b37cd6195c28ce4641b28ea4f791e7ca2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
spec=importlib.util.spec_from_file_location('paired_helper_readonly',SOURCE);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
plan=json.loads((ROOT/'review/v15A_prospective_multiscene_plan_v2.json').read_text(encoding='utf-8'))
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
def fake_build(*,pending_transport=False,bad_one=False):
    calls=[]
    def read(p):
        if str(p)=='synthetic_plan':return deepcopy(plan)
        return dict(execution_complete=not(pending_transport and 'transport' in str(p)))
    def bundle(path,aux,cache):
        path=Path(path);scene,method,seed=path.name.split('__');role='candidate_A' if 'candidate_A' in str(path) else 'baseline';motion='transport' if 'transport' in str(path) else 'hold'
        calls.append(str(path))
        if bad_one and len(calls)==1:return dict(status='read_from_stable_completed_or_retained_error_inputs',diagnostics=dict(status='assigned_trace_missing_or_unexecuted',original_result=dict(task_success=True),measurements=None),physical_context=dict(same=True),timing=None)
        identity=dict(method=method,seed=int(seed),motion_mode=motion,task_policy='integrated' if role=='candidate_A' else 'legacy',receiver_policy='blind',scenario=dict(name=scene))
        return dict(status='read_from_stable_completed_or_retained_error_inputs',diagnostics=dict(status='measurements_computed_no_new_success_label',identity=identity,original_result=dict(task_success=True,research_metrics={}),measurements={}),physical_context=dict(same=True),timing=None)
    with patch.object(h,'read',read),patch.object(h,'load_auxiliary',lambda:object()),patch.object(h,'bundle',bundle),patch.object(h,'sha',lambda p:'synthetic-fingerprint'),patch.object(Path,'is_file',lambda p:True):result=h.build_readout('synthetic_plan',Path('synthetic_cache'))
    return result
good=fake_build();ck('64_records_48_pairs_and_same_ratio_kept_distinct',good['complete'] and len(good['records'])==64 and len(good['pairs'])==48 and len({r['record_id'] for r in good['records']})==64 and len([p for p in good['pairs'] if p['primary_same_preference_contrast']])==16)
part=fake_build(pending_transport=True);ck('complete_hold32_partial24pairs_without_overall_final',not part['complete'] and part['summaries'] is None and set(part['completed_strata_descriptive'])=={'hold'})
missing=fake_build(bad_one=True);ck('truthy_missing_trace_diagnostics_cannot_count_as_final_readable',not missing['complete'],dict(actual_complete=missing['complete'],readable_records=missing['completion']['readable_records'],bad_diagnostic_status=missing['records'][0]['bundle']['diagnostics']['status']))
b=dict(diagnostics=dict(original_result=dict(research_metrics=dict(static_quality=dict(eligible=False,raw_200Hz=dict(finger_TV_rate_N_s=123.)))),measurements=dict(static=dict(unselected_force=dict(available=True,finger_TV_rate_N_s=0.),physical_object={}))))
v=h.metric_values(b);ck('eligible_NULL_separate_from_all_assigned_measured_zero',v['eligible.static.raw_200Hz.finger_TV_rate_N_s'] is None and v['raw.static.finger_TV_rate_N_s']==0.)
ck('relative_baseline_zero_and_missing_remain_unknown',h.difference(1.,0.)['relative_delta'] is None and not h.difference(None,0.)['available'])
timing=h.timing_stats(dict(control_ms=np.array([100.,1.,1.,1.,20.])),[0,4]);ck('actual_decision_indices_not_all5ms_dilution',timing['control_ms']['count']==2 and timing['control_ms']['median']==60.)
text=SOURCE.read_text(encoding='utf-8');ck('cache_key_static_input_aux_helper_binding',all(x in text for x in ('inputs=inputs,aux=AUX_SHA,helper=sha(__file__)','joint_decisions.jsonl.gz','delivery_intervals.npz','pending_inputs_changed_during_read')))
ck('two_seed_descriptive_no_CI_contract',good['statistical_scope']['physical_seed_clusters']==2 and good['statistical_scope']['ci_or_pvalues'] is False and good['statistical_scope']['independent_context_n16_claim'] is False)
report=dict(schema='v15A_paired_helper_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=EXPECTED,source_unchanged=sha(SOURCE)==EXPECTED,
    checker_sha256=sha(__file__),scope='Pure synthetic role/NULL/denominator/completion/timing-index and static cache review. No actual cohort branch, metric or packet read.',physics_steps=0,model_imported='mujoco' in sys.modules)
out=ROOT/'review/v15A_paired_helper_independent_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

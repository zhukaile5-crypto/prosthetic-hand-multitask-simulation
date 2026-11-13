"""All-assigned paired DESCRIPTIVE readout for the predeclared V15A64 cohort.

Uses frozen continuous_diagnostics_v1.read_run once per immutable run/cache.
No new success/score, CI, winner selection, controller or physical execution.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import argparse,gzip,hashlib,importlib.util,itertools,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='v15A_paired_prospective_readout_v1'
AUX_SHA='05e86a5869f144f990dbb9883add20d002c4384f402cfb07b15db333e9bd8cfa'
QUALITY_FIELDS=('finger_TV_N','finger_TV_rate_N_s','finger_net_drift_N','finger_net_drift_rate_N_s','finger_reversal_N','finger_reversal_rate_N_s','total_force_TV_N','total_force_TV_rate_N_s','finger_SD_mean_N','total_force_SD_N')
FORCE_FIELDS=('mean_total_force_N','mean_per_finger_N','finger_TV_N','finger_TV_rate_N_s','per_finger_TV_N','total_force_TV_N','total_force_TV_rate_N_s','per_finger_SD_N','mean_finger_SD_N','total_force_SD_N','per_finger_CV','total_force_CV','finger_net_drift_N','finger_reversal_N','finger_net_drift_rate_N_s','finger_reversal_rate_N_s')
BODY_FIELDS=('relative_handoff_max_displacement_m','relative_handoff_terminal_displacement_m','world_handoff_max_displacement_m','world_handoff_terminal_displacement_m','min_world_height_m','terminal_world_height_m','min_world_height_minus_handoff_m','relative_linear_speed_max_m_s','relative_linear_speed_terminal_m_s','relative_orientation_change_max_rad','relative_orientation_change_terminal_rad','relative_angular_speed_max_rad_s','relative_angular_speed_terminal_rad_s')
TIMING=('control_ms','full_control_loop_ms','observation_ms','actor_callback_ms','physics_advance_ms')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def fingerprint(path):return sha(path) if Path(path).is_file() else None
def plain(v):
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def flattened(prefix,value,target):
    if isinstance(value,dict):
        for k,x in value.items():flattened(prefix+'.'+k,x,target)
    elif isinstance(value,(list,tuple,np.ndarray)):
        for i,x in enumerate(value):flattened(prefix+'.'+str(i),x,target)
    elif value is None or isinstance(value,(bool,int,float,np.number)):target[prefix]=plain(value)

def difference(a,b):
    if a is None or b is None:return dict(candidate=a,baseline=b,available=False,delta=None,abs_delta=None,relative_delta=None,reason='one_or_both_unavailable')
    if isinstance(a,bool) and isinstance(b,bool):
        d=int(a)-int(b);return dict(candidate=a,baseline=b,available=True,delta=d,abs_delta=abs(d),relative_delta=None,reason='boolean_difference_relative_not_defined')
    if not isinstance(a,(int,float)) or isinstance(a,bool) or not isinstance(b,(int,float)) or isinstance(b,bool):return dict(candidate=a,baseline=b,available=False,delta=None,abs_delta=None,relative_delta=None,reason='inconsistent_scalar_types')
    d=a-b;r=None if b==0 else d/b
    return dict(candidate=a,baseline=b,available=True,delta=d,abs_delta=abs(d),relative_delta=r,reason='zero_baseline_relative_undefined' if b==0 else 'candidate_minus_baseline;relative_divides_signed_baseline')

def descriptive(values):
    known=[x for x in values if x is not None];base=dict(n_assigned=len(values),n_known=len(known),n_unknown=len(values)-len(known))
    if known and all(isinstance(x,bool) for x in known):return dict(**base,n_true=sum(known),n_false=len(known)-sum(known),true_fraction_known=sum(known)/len(known))
    nums=[x for x in known if isinstance(x,(int,float)) and not isinstance(x,bool)]
    if len(nums)!=len(known):return dict(**base,statistics=None,reason='mixed_scalar_types')
    return dict(**base,statistics=None if not nums else dict(mean=float(np.mean(nums)),median=float(np.median(nums)),min=float(min(nums)),max=float(max(nums))))

class ArrayArchive:
    def __init__(self,arrays):self.arrays=arrays;self.files=list(arrays)
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def __getitem__(self,key):return self.arrays[key]

def load_auxiliary():
    p=ROOT/'review/continuous_diagnostics_v1.py'
    if sha(p)!=AUX_SHA:raise ValueError('Frozen auxiliary reader changed')
    s=importlib.util.spec_from_file_location('frozen_v15A_auxiliary',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def timing_stats(a,indices):
    out={}
    for name in TIMING:
        vals=a.get(name);expected=len(indices)
        if vals is None or any(i>=len(vals) for i in indices):out[name]=dict(available=False,count=expected,median=None,p95=None,max=None);continue
        v=np.asarray(vals)[indices]
        ok=bool(len(v) and np.isfinite(v).all() and np.all(v>=0))
        out[name]=dict(available=ok,count=expected,median=float(np.median(v)) if ok else None,p95=float(np.percentile(v,95)) if ok else None,max=float(v.max()) if ok else None)
    return out

def bundle(folder,aux,cache_directory):
    """Cache key binds every input used, not the shared method label."""
    folder=Path(folder);names=('trace.npz','metadata.json','result.json','delivery_intervals.npz','delivery_summary.json','joint_decisions.jsonl.gz','runtime_error.json')
    inputs={n:fingerprint(folder/n) for n in names}
    if inputs['result.json'] is None and inputs['runtime_error.json'] is None:return dict(status='pending_unexecuted_or_result_not_finished',run_directory=str(folder),inputs=inputs,diagnostics=None,timing=None)
    key=hashlib.sha256(json.dumps(dict(path=str(folder.resolve()),inputs=inputs,aux=AUX_SHA,helper=sha(__file__)),sort_keys=True).encode()).hexdigest();cache=cache_directory/(key+'.json')
    if cache.is_file():
        b=read(cache);b['cache_read']=True;return b
    arrays={};ledger={};trace=folder/'trace.npz';lp=folder/'delivery_intervals.npz'
    if trace.is_file():
        wanted=('time','normal','position','relative','relative_velocity','rotation','angular_velocity','qpos','metadata','controller_active',*TIMING)
        with np.load(trace,allow_pickle=False) as z:arrays={k:z[k] for k in wanted if k in z.files}
    if lp.is_file():
        wanted=('interval_start_s','interval_end_s','contact_valid','object_state_valid','object_free_qpos7_end','object_free_qvel6_end','receiver_any_actor_top_normal_N',*('target_'+k+'_normal_N' for k in aux.CATEGORIES))
        with np.load(lp,allow_pickle=False) as z:ledger={k:z[k] for k in wanted if k in z.files}
    # Frozen read_run receives the very same arrays in memory. It is called
    # once, without a second compressed-array read, and retains its semantics.
    original=np.load
    def in_memory(path,*a,**k):
        if Path(path).resolve()==trace.resolve():return ArrayArchive(arrays)
        if Path(path).resolve()==lp.resolve():return ArrayArchive(ledger)
        raise ValueError('Unexpected additional archive '+str(path))
    with patch.object(np,'load',in_memory):diagnostics=aux.read_run(folder)
    times=[];modes=[]
    if (folder/'joint_decisions.jsonl.gz').is_file():
        with gzip.open(folder/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                d=json.loads(line);times.append(float(d['time_s']));modes.append(d['mode'])
    t=arrays.get('time',np.empty(0));indices=[];unaligned=[]
    for decision_time in times:
        j=int(np.argmin(abs(t-decision_time))) if len(t) else -1
        if j<0 or abs(t[j]-decision_time)>1e-9:unaligned.append(decision_time)
        else:indices.append(j)
    active=np.flatnonzero(arrays.get('controller_active',np.zeros(len(t),bool))).tolist()
    timing=dict(decision=timing_stats(arrays,indices),active_5ms=timing_stats(arrays,active),actual_decision_count=len(times),unaligned_decision_times=unaligned,
        source='all saved joint_decisions time_s matched to trace clock; never all900ticks diluted into planning cost',
        control_ms_scope='control branch after public sensing/task setup, includes state update/decision/execute; excludes outer sensing and physics integration',
        full_control_loop_ms_scope='public observation+task/control+trace path, excludes physics integration; separate physics_advance_ms is not summed as overlapping clock')
    for mode in ('holding','release'):
        selected=[indices[i] for i,m in enumerate(modes) if m==mode] if len(indices)==len(times) else []
        timing[mode+'_decision']=timing_stats(arrays,selected)
    if inputs!={n:fingerprint(folder/n) for n in names}:return dict(status='pending_inputs_changed_during_read',run_directory=str(folder),inputs=inputs,diagnostics=None,timing=None)
    metadata=read(folder/'metadata.json') if (folder/'metadata.json').is_file() else {}
    context={k:metadata.get(k) for k in ('config','noise','model_sha256','initial_position','limits','budget','sensor_period','physics_timestep')}
    b=plain(dict(status='read_from_stable_completed_or_retained_error_inputs',run_directory=str(folder),inputs=inputs,diagnostics=diagnostics,physical_context=context,timing=timing,cache_read=False,cache_key=key))
    cache_directory.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(b,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');return b

def metric_values(b):
    diag=b.get('diagnostics') or {};result=diag.get('original_result') or {};research=result.get('research_metrics') or {};values={}
    for name in ('task_success','record_complete','A_S','first_stable_after_handoff','release_exposed','release_success','release_latency'):values['original.legacy.'+name]=result.get(name)
    for name in ('mission_success','full_motion_mission_success','preparation_passed','policy_exposed','initial_stability_wait_s','failure_at'):values['original.'+name]=research.get(name)
    for name in ('safety','release','progress','wrist_tracking_guard'):flattened('original.'+name,research.get(name),values)
    for window,key in [('static','static_quality'),('dynamic','dynamic_quality'),('whole_holding','entire_holding_quality')]:
        q=research.get(key) or {};values['eligible.'+window+'.eligible']=q.get('eligible')
        for signal in ('bandlimited','raw_200Hz'):
            x=q.get(signal) or {}
            for field in QUALITY_FIELDS:values['eligible.'+window+'.'+signal+'.'+field]=x.get(field) if q.get('eligible') else None
        measurement=(diag.get('measurements') or {}).get(window) or {};raw=measurement.get('unselected_force') or {};body=measurement.get('physical_object') or {}
        values['raw.'+window+'.available']=raw.get('available');values['object.'+window+'.available']=body.get('available')
        for field in FORCE_FIELDS:flattened('raw.'+window+'.'+field,raw.get(field),values)
        for field in BODY_FIELDS:values['object.'+window+'.'+field]=body.get(field)
        flattened('object.'+window+'.raw_world',body.get('raw_integration_end_world'),values)
    for window,data in (diag.get('contact_windows') or {}).items():
        for category,measure in data.items():
            for field in ('available','any_positive_contact','peak_N','impulse_Ns','positive_duration_s'):values['contact.'+window+'.'+category+'.'+field]=measure.get(field)
    for stage,x in (diag.get('object_end_states') or {}).items():
        for field in ('available','position_m','linear_speed_m_s','angular_speed_rad_s'):flattened('endstate.'+stage+'.'+field,x.get(field),values)
    for stage,x in (b.get('timing') or {}).items():
        if isinstance(x,dict):flattened('timing.'+stage,x,values)
    return values

def summaries(records,pairs,metrics):
    arm_ids=sorted({x['arm_id'] for x in records});baselines=('ratio_effort','fixed_effort','fusion_effort')
    def group(seed=None):
        rr=[x for x in records if seed is None or x['context_key'][2]==seed];pp=[x for x in pairs if seed is None or x['context_key'][2]==seed]
        arms={arm:{m:descriptive([r['metrics'].get(m) for r in rr if r['arm_id']==arm]) for m in metrics} for arm in arm_ids}
        contrasts={method:{m:dict(delta=descriptive([p['differences'][m]['delta'] for p in pp if p['baseline_force_preference']==method]),relative_delta=descriptive([p['differences'][m]['relative_delta'] for p in pp if p['baseline_force_preference']==method])) for m in metrics} for method in baselines}
        return dict(arms=arms,contrasts=contrasts,unit_note='Context arithmetic summaries within the given physical seed(s); no CI/pvalue/independent-context inference')
    seeds=sorted({x['context_key'][2] for x in records});return dict(per_physical_seed={str(s):group(s) for s in seeds},all_assigned_descriptive=group(),physical_seed_clusters=len(seeds),contexts_are_not_independent_random_units=True)

def build_readout(plan_path,cache_directory):
    plan=read(plan_path);aux=load_auxiliary();records=[];stages=[];issues=[]
    for stage in plan['stages']:
        output=ROOT/stage['output'];finish=read(output/'execution_complete.json') if (output/'execution_complete.json').is_file() else None
        stages.append(dict(name=stage['name'],arm=stage['arm'],motion_mode=stage['motion_mode'],planned=stage['planned'],complete=bool(finish and finish.get('execution_complete')),execution_complete=finish))
        for seed,scene,method in itertools.product(stage['seeds'],stage['scenarios'],stage['methods']):
            directory=output/'runs'/f'{scene}__{method}__{seed}'
            try:b=bundle(directory,aux,cache_directory)
            except Exception as exc:b=dict(status='retained_read_error_or_pending_write',run_directory=str(directory),error=type(exc).__name__+': '+str(exc),diagnostics=None,timing=None)
            arm_id='/'.join((stage['arm'],method,stage['task_policy'],stage['receiver_policy']));diag=b.get('diagnostics') or {};identity=diag.get('identity') or {}
            expected=dict(method=method,seed=seed,motion_mode=stage['motion_mode'],task_policy=stage['task_policy'],receiver_policy=stage['receiver_policy'])
            mismatch=[k for k,v in expected.items() if identity and identity.get(k)!=v]
            if identity and (identity.get('scenario') or {}).get('name')!=scene:mismatch.append('scenario')
            if mismatch:issues.append(dict(path=str(directory),identity_mismatch=mismatch))
            records.append(dict(record_id=stage['name']+'/'+scene+'/'+str(seed)+'/'+method,arm_id=arm_id,role=stage['arm'],force_preference=method,context_key=[scene,stage['motion_mode'],seed],
                assigned=True,status=b['status'],identity_mismatch=mismatch,metrics=metric_values(b),bundle=b))
    all_metrics=sorted(set().union(*(r['metrics'] for r in records)));pairs=[]
    expected_contexts=sorted({tuple(x['context_key']) for x in records})
    for key in expected_contexts:
        aa=[r for r in records if tuple(r['context_key'])==key and r['role']=='candidate_A']
        if len(aa)!=1:issues.append(dict(context=key,candidate_assignment_count=len(aa)));continue
        A=aa[0]
        for method in ('ratio_effort','fixed_effort','fusion_effort'):
            bb=[r for r in records if tuple(r['context_key'])==key and r['role']=='baseline' and r['force_preference']==method]
            if len(bb)!=1:issues.append(dict(context=key,baseline=method,assignment_count=len(bb)));continue
            B=bb[0]
            ready=bool(A['bundle'].get('diagnostics') and B['bundle'].get('diagnostics'))
            pairing=None if not ready else A['bundle'].get('physical_context')==B['bundle'].get('physical_context')
            if ready and not pairing:issues.append(dict(context=key,baseline=method,error='actual_physical_config_or_noise_mismatch'))
            pairs.append(dict(context_key=list(key),candidate_record_id=A['record_id'],baseline_record_id=B['record_id'],candidate_arm=A['arm_id'],baseline_arm=B['arm_id'],baseline_force_preference=method,actual_physical_context_equal=pairing,
                primary_same_preference_contrast=method=='ratio_effort',differences={m:difference(A['metrics'].get(m),B['metrics'].get(m)) for m in all_metrics}))
    readable=sum(bool(r['bundle'].get('diagnostics')) for r in records)
    complete=all(s['complete'] for s in stages) and len(records)==readable==64 and len(pairs)==48 and not issues
    strata={}
    for motion in ('hold','transport'):
        rr=[r for r in records if r['context_key'][1]==motion];pp=[p for p in pairs if p['context_key'][1]==motion];ss=[s for s in stages if s['motion_mode']==motion]
        if len(rr)==32 and len(pp)==24 and all(s['complete'] for s in ss) and all(r['bundle'].get('diagnostics') for r in rr) and all(p['actual_physical_context_equal'] is True for p in pp):strata[motion]=summaries(rr,pp,all_metrics)
    return plain(dict(schema=SCHEMA,status='complete_descriptive_cohort' if complete else 'partial_no_overall_conclusion',complete=complete,completion=dict(stages=stages,assigned=len(records),pairs=len(pairs),readable_records=sum(bool(r['bundle'].get('diagnostics')) for r in records),pending_records=[r['record_id'] for r in records if not r['bundle'].get('diagnostics')],issues=issues),
        records=records,pairs=pairs,metric_registry=all_metrics,summaries=summaries(records,pairs,all_metrics) if complete else None,completed_strata_descriptive=strata,
        comparison_order=['candidate_A vs legacy ratio_effort (same prior, primary coordination contrast)','candidate_A vs legacy fixed_effort','candidate_A vs legacy fusion_effort (retain different force/effort tradeoffs)'],
        provenance=dict(plan=str(plan_path),plan_sha256=sha(plan_path),helper_sha256=sha(__file__),auxiliary_reader_sha256=AUX_SHA),
        statistical_scope=dict(physical_seed_clusters=2,paired_contexts=16,paired_method_contrasts=48,candidate_trajectories=16,total_assigned=64,independent_context_n16_claim=False,ci_or_pvalues=False,weighted_total_score=False,posthoc_winner_filter=False),
        difference_convention='delta=candidate−baseline; abs_delta=abs(delta); relative_delta=(candidate−baseline)/baseline for finite nonzero numeric baseline. Bool/zero-baseline/unavailable relative values remain null.',
        original_eligibility_scope='Eligible metrics remain NULL on either ineligible side. All-assigned raw0 is a real measurement, never substituted for eligible NULL. No method-specific success filtering.',physics_steps=0))

def self_test():
    tests={
        'same_method_distinct_arms':len({'candidate_A/ratio_effort/integrated/blind','baseline/ratio_effort/legacy/blind'})==2,
        'zero_baseline_no_epsilon':difference(1.,0.)['relative_delta'] is None,
        'unknown_not_zero':not difference(None,0.)['available'],
        'raw_zero_counted_known':descriptive([0.,None])['n_known']==1,
        'boolean_delta_not_relative':difference(True,False)['delta']==1 and difference(True,False)['relative_delta'] is None}
    return dict(passed=all(tests.values()),tests=tests,source_sha256=sha(__file__),physics_steps=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='review/v15A_prospective_multiscene_plan_v2.json');p.add_argument('--cache',default='review/v15A_paired_metric_cache_v1');p.add_argument('--out',required=True);p.add_argument('--allow-partial',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    out=ROOT/a.out
    if out.exists():raise FileExistsError(out)
    report=self_test() if a.self_test else build_readout(ROOT/a.plan,ROOT/a.cache)
    if not a.self_test and not report['complete'] and (not a.allow_partial or 'partial' not in out.name.lower()):raise RuntimeError('Incomplete cohort: use a fresh explicitly partial filename; final readout must await all64')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(output=str(out),source_sha256=sha(__file__),status=report.get('status'),complete=report.get('complete'),passed=report.get('passed'))))
if __name__=='__main__':main()

"""Terminal-only execution audit, distinct from performance/auxiliary readers.

--prepare reads only the registration and code. --execute first requires every
registered stage's execution_complete=True; pending is not failure. All80 rows
are retained. No step, plant, forward, contact query or physical replay exists.
One static model per saved config may identify native21 ports and lift bias.
Every logged motion M/target is checked against its attained residual; a
separate prespecified six-record public-robot reconstruction remains explicit.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from unittest.mock import patch
from contextlib import ExitStack
import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT.parent / '五指策略接入_2026-11-03'
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(FOUNDATION))
sys.dont_write_bytecode = True
PLAN = ROOT / 'review/v17b_prospective80_plan_v1.json'
PIN = 'b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
STOP = ROOT/'review/v17b_prospective80_progress_v1.json'
RESUME = ROOT/'review/v17b_prospective80_resume_plan_v1.json'
RESUME_PROGRESS = ROOT/'review/v17b_prospective80_resume_progress_v1.json'
CHAIN = ROOT/'review/v17b_prospective80_completion_chain_v1.json'
COLS = ('time','normal','skin_normal','shear','palm','hand_other','all_hand_object',
        'support','penetration','world_penetration','position','relative','rotation',
        'relative_velocity','angular_velocity','lift','qpos','qvel','wrist_position',
        'ctrl_before','ctrl','alpha_before','alpha','robot_actuator_target21',
        'executed_reference21','phase','command','observed_normal','validity',
        'observed_age','observed_auxiliary_wrench','observed_auxiliary_valid',
        'legacy_common_sensor_json','receiver_measurement_json','task_request_json',
        'environment_json','raw_patches_json')
MOTION_SELECTIONS = (('hold','fragile_conflict',619741),
                     ('transport','nominal',619741),
                     ('transport','friction_drop',619742))


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray): return plain(x.tolist())
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x
def module(name, relative):
    s=importlib.util.spec_from_file_location(name,ROOT/relative)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def source_path(k): return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
def jobs(plan):
    return [dict(stage=s['name'],arm=s['arm'],motion=s['motion_mode'],scenario=e,
                 seed=n,method=m,task_policy=s['task_policy'],receiver_policy=s['receiver_policy'],
                 run=str(ROOT/s['output']/'runs'/f'{e}__{m}__{n}'))
            for s in plan['stages'] for n in s['seeds'] for e in s['scenarios'] for m in s['methods']]
def key(r): return (r['scenario'],r['method'],int(r['seed']))
def gate(plan, reader=read):
    states=[]
    for s in plan['stages']:
        try: d=reader(ROOT/s['output']/'execution_complete.json')
        except FileNotFoundError: d=None
        states.append(dict(stage=s['name'],complete=d is not None and d.get('execution_complete') is True,completion=d))
    try: chain=reader(CHAIN)
    except FileNotFoundError: chain=None
    return dict(ready=all(r['complete'] for r in states) and chain is not None and chain.get('complete') is True,
                states=states,completion_chain=chain,pending_is_failure=False,branch_results_read=False)
def digest_array(a):
    a=np.ascontiguousarray(a)
    # Text payload is exact; archive-wide Unicode width is separately recorded.
    if a.dtype.kind in 'US':
        return sha_bytes(json.dumps(a.tolist(),ensure_ascii=False,separators=(',',':')).encode('utf-8'))
    return sha_bytes(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def fingerprints(a, cutoff):
    take=a['time']<cutoff-1e-9
    return {k:dict(payload_sha256=digest_array(a[k][take]),shape=list(a[k][take].shape),
                   dtype=str(a[k].dtype),literal_sha256=sha_bytes(np.ascontiguousarray(a[k][take]).tobytes())) for k in COLS}
def same_fingerprints(a,b):
    return [k for k in COLS if a[k]['shape']!=b[k]['shape'] or a[k]['payload_sha256']!=b[k]['payload_sha256']]


def motion_record(q):
    """Rebuild the recorded attained motion residual without solving any QP."""
    errors=[]; task=q.get('motion_task_public');available=bool(q['motion_task_available'])
    if 'motion_task_public' not in q or q.get('motion_evidence_role')!='already_created_public_task_copy_after_command_decision':
        errors.append('new_decision_missing_full_motion_logging_contract')
    if not available:
        if task is not None: errors.append('unavailable_motion_has_matrix')
        return errors,dict(available=False)
    if task is None: return ['available_motion_missing_matrix'],dict(available=True)
    M=np.asarray(task['matrix_Mx21'],float);b=np.asarray(task['target_M'],float)
    if (M.ndim!=2 or M.shape[1]!=21 or M.shape[0] not in (2,4,6,8,10)
        or b.shape!=(len(M),) or not np.isfinite(M).all() or not np.isfinite(b).all()):
        return ['invalid_full_motion_shape_or_values'],dict(available=True)
    meta=task['metadata']
    if (abs(task['time_s']-q['time_s'])>1e-9 or abs(meta['read_time_s']-q['time_s'])>1e-9
        or meta['acquisition_time_s']>q['time_s']+1e-9
        or meta['future_pressure_prediction_available'] or meta['future_motion_prediction_available']
        or meta['force_safety_certified'] or meta['placement_certified']): errors.append('clock_or_false_prediction_claim')
    from_projection=-.02*np.concatenate([r['measured_contact_average_projection_m_s'] for r in meta['rows']])
    if not np.array_equal(from_projection,b): errors.append('target_not_declared_observed_projection')
    evidence=dict(available=True,rows=len(M),matrix_sha256=sha_bytes(M.tobytes()),target_sha256=sha_bytes(b.tobytes()))
    if q['geometry_proposal_accepted']:
        scale=np.asarray(q['scale21']);x=np.asarray(q['proposed_reference_increment21'])/scale
        lo=np.asarray(q['actual_step_lower21'])/scale;hi=np.asarray(q['actual_step_upper21'])/scale
        ms=M*scale[None,:]
        divisor=max(float(np.max(np.sum(abs(ms)*np.maximum(abs(lo),abs(hi))[None,:],axis=1),initial=0.)),float(np.max(abs(b),initial=0.)))
        divisor=divisor if divisor else 1.
        residual=(ms/divisor)@x-b/divisor
        level=next((v for v in q['hierarchy_levels'] if v['name']=='observed_external_contact_motion_correction'),None)
        if level is None: errors.append('available_motion_missing_hierarchy_layer')
        else:
            saved=np.asarray(level['residual']);tol=level['numerical_atol']+level['numerical_rtol']*max(1.,float(np.max(abs(x))))
            err=float(np.max(abs(residual-saved),initial=0.))
            if residual.shape!=saved.shape or err>tol+128*np.finfo(float).eps*max(1.,float(np.max(abs(residual),initial=0.))):errors.append('fullM_attained_residual_mismatch')
            evidence.update(residual_error_inf=err,declared_freeze_tolerance=tol)
    return errors,evidence


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--self-test',action='store_true');ap.add_argument('--out',default='review/v17b_prospective80_execution_independent_v1.json');args=ap.parse_args()
    if args.self_test:
        seen=[]
        def absent(p):seen.append(str(p));raise FileNotFoundError(p)
        g=gate(dict(stages=[dict(name='pending',output='output/NO_RUN_READ')]),absent)
        assert not g['ready'] and not g['pending_is_failure'] and len(seen)==2
        a={'time':np.array([0.,.005,12.])};a.update({k:np.array([1.,2.,3.]) for k in COLS if k!='time'})
        b={k:v.copy() for k,v in a.items()};assert not same_fingerprints(fingerprints(a,12),fingerprints(b,12))
        b['normal'][1]=np.nextafter(2.,np.inf);assert same_fingerprints(fingerprints(a,12),fingerprints(b,12))==['normal']
        print(json.dumps(dict(passed=True,pure_checks=3,branch_results_read=False,physics_steps=0)));return 0
    plan=read(PLAN)
    if sha(PLAN)!=PIN: raise ValueError('Registered80 plan identity changed')
    assigned=jobs(plan)
    if len(assigned)!=80 or len({(j['stage'],j['scenario'],j['seed'],j['method']) for j in assigned})!=80:raise ValueError('Invalid80 unique assignment')
    if args.prepare:
        out=ROOT/'review/v17b_prospective80_execution_readiness_v2.json'
        if out.exists():raise FileExistsError(out)
        current={k:sha(source_path(k)) for k in plan['source_sha256']}
        result=dict(schema='v17b_execution_reader_readiness_v1',plan_sha256=PIN,assigned_count=80,
            prepared=True,execution_audit_run=False,all_assigned_checked=False,branch_results_read=False,
            current136_match=len(current)==136 and current==plan['source_sha256'],reader_sha256=sha(__file__),
            scope=['all6stage136snapshots/env42/80assignments/results/failures','all80 native21 5ms permission/clock/rate/receipt',
                   'immutable original stopped16 receipt + resume original64 plan/progress/completionchain; every retained original16 file SHA',
                   'all16contexts fivearm strictpre7.5 physical+publicinstrument; sameholding16fullA strictpre12 all37 fullJSON',
                   'all80 acquisition domain masks/no fabricated pressure forecasts','all logged M/target clock/shape/declared proxy/attained residual',
                   'prespecified six public robot reconstruction in separate appendix; no performance selection'],
            motion_six_prespecified_contexts=MOTION_SELECTIONS,
            motion_rule='first and last accepted motion-available CONTACT per listed context; absent strata stay absent, no replacement',
            physics_steps=0)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(path=str(out),sha256=sha(out),**{k:result[k] for k in ('prepared','current136_match','branch_results_read')})));return 0
    if not args.execute:raise ValueError('Use explicit --prepare or post-terminal --execute')
    terminal=gate(plan)
    if not terminal['ready']:
        print(json.dumps(dict(status='PENDING',terminal=terminal,assigned_count=80,all_assigned_checked=False)));return 2
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    base=module('v17b_saved_base_independent', 'review/read_v15_active9_independent_v1.py')
    v17=module('v17b_saved_domain_independent','review/v17_saved_reader_helpers_v1.py')
    checks=[];issues=[];rows=[];contexts=defaultdict(list);hashes={};forbidden=[]
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:issues.append(dict(name=name,detail=detail))
    def h(p):
        p=str(Path(p).resolve())
        if p not in hashes:hashes[p]=sha(p)
        return hashes[p]
    ck('source136 current exact',len(plan['source_sha256'])==136 and all(h(source_path(k))==v for k,v in plan['source_sha256'].items()))
    ck('environment42 and interpreter exact',len(plan['environment']['native_binaries_sha256'])==42 and all(h(k)==v for k,v in plan['environment']['native_binaries_sha256'].items()) and h(plan['environment']['executable'])==plan['environment']['executable_sha256'])
    parent=read(ROOT/'review/v17_transfer_small_plan_v1.json')['source_sha256']
    ck('old133 exactly retained; logging only3new',len(parent)==133 and all(plan['source_sha256'].get(k)==v for k,v in parent.items()) and len(set(plan['source_sha256'])-set(parent))==3)
    continuation={};retained_ok=False
    try:
        resume=read(RESUME);progress=read(RESUME_PROGRESS);stop=read(STOP);chain=terminal['completion_chain']
        incident=read(ROOT/'review/v17b_prefix_reader_correction_independent_v1.json')
        old_stop_pin=incident['retained_reader_incident_sha256'][str(STOP)]
        ck('original stopped16 progress exact and still failed-not-rewritten',h(STOP)==old_stop_pin==resume['stopped_progress_sha256']==chain['stopped_progress_sha256'] and stop['complete'] is False and stop['stopped_on_prefix_contract_failure'] is True and len(stop['records'])==2)
        ck('resume exactly original remaining64 assignments without repeated16',resume['original_plan_sha256']==PIN and resume['source_sha256']==plan['source_sha256'] and resume['environment']==plan['environment'] and resume['retained_stages']==plan['stages'][:2] and resume['remaining_stages']==plan['stages'][2:] and resume['retained_trials']==16 and resume['remaining_trials']==64 and resume['total_planned']==80 and resume['old16_repeated'] is False and resume['algorithm_model_gain_measurements_metrics_thresholds_assignments_changed'] is False)
        ck('resume and completion chain identities totals source environment',progress['complete'] is True and progress['physics_active'] is False and progress['resume_plan_sha256']==h(RESUME)==chain['resume_plan_sha256'] and chain['resume_progress_sha256']==h(RESUME_PROGRESS) and chain['original_plan_sha256']==PIN and chain['retained_trials']==16 and chain['continued_trials']==64 and chain['total_completed']==80 and chain['old16_repeated'] is False and chain['control_or_metric_changes'] is False and chain['source_sha256']==plan['source_sha256'] and chain['environment']==plan['environment'])
        ck('corrected fullJSON reader and independent correction gates pinned',h(ROOT/resume['corrected_reader'])==resume['corrected_reader_sha256']==chain['corrected_reader_sha256'] and h(ROOT/resume['reader_gate'])==resume['reader_gate_sha256']==chain['reader_gate_sha256'] and h(ROOT/resume['incident_gate'])==resume['incident_gate_sha256']==chain['incident_gate_sha256'] and h(ROOT/'review/continue_v17b_prospective80_v1.py')==resume['scheduler_sha256'])
        ck('all original stage registrations retained',all(h(ROOT/p)==digest for p,digest in resume['original_registration_sha256'].items()) and set(resume['original_registration_sha256'])=={s['registration'] for s in plan['stages']})
        retained_actual={}
        for stage in plan['stages'][:2]:
            folder=ROOT/stage['output']
            for p in folder.rglob('*'):
                if p.is_file() and 'sources' not in p.relative_to(folder).parts:
                    retained_actual[str(p.relative_to(ROOT))]=h(p)
        retained_ok=retained_actual==resume['retained_files_sha256']
        ck('every retained original16 file exact-byte identity and complete file set',retained_ok,dict(files=len(retained_actual)))
        ck('chain binds every6 completion and assignment ledger',set(chain['stage_completion_sha256'])=={s['name'] for s in plan['stages']} and all(chain['stage_completion_sha256'][s['name']]==h(ROOT/s['output']/'execution_complete.json') and chain['stage_assignment_ledger_sha256'][s['name']]==h(ROOT/s['output']/'assignment_ledger.json') for s in plan['stages']))
        ck('original scheduler and legacy comparator preserved not patched',h(ROOT/plan['scheduler'])==plan['scheduler_sha256'] and all(h(ROOT/p)==digest for p,digest in plan['prefix_helper_sha256'].items()))
        continuation=dict(original_stop_sha256=h(STOP),resume_plan_sha256=h(RESUME),resume_progress_sha256=h(RESUME_PROGRESS),completion_chain_sha256=h(CHAIN),retained_original16_files=len(retained_actual),retained_data_unchanged=retained_ok)
    except Exception as exc:
        issues.append(dict(name='continuation_chain_exception',error=repr(exc),traceback=traceback.format_exc()))
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden '+name)
        return f
    with ExitStack() as guards:
        for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant')))
        guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor.apply')))
        for stage in plan['stages']:
            folder=ROOT/stage['output'];tag=stage['name']
            stage_jobs=[j for j in assigned if j['stage']==tag]
            try:
                reg=read(ROOT/stage['registration']);done=read(folder/'execution_complete.json')
                manifest=read(folder/'manifest.json');execution=read(folder/'plan.json');trials=read(folder/'trials.json')
                failures=read(folder/'runtime_failures.json');ledger=read(folder/'assignment_ledger.json');snap=read(folder/'source_snapshot_manifest.json')
                expected=dict(methods=stage['methods'],scenarios=stage['scenarios'],seeds=stage['seeds'],motion_mode=stage['motion_mode'],physical_randomization=True,task_policy=stage['task_policy'],receiver_policy=stage['receiver_policy'])
                ck(tag+'/registration and plan identities',reg['assignment']==expected and reg['source_sha256']==plan['source_sha256'] and reg['environment']==plan['environment'] and reg['master_plan_sha256']==PIN)
                wanted={key(j) for j in stage_jobs}
                ck(tag+'/allassigned results and failures retained',done['sources_unchanged'] and not failures and done['planned']==done['trials']==stage['planned'] and len(trials)==len(ledger)==len(execution['jobs'])==stage['planned'] and {key(x) for x in trials}=={key(x['job']) for x in ledger}=={key(x) for x in execution['jobs']}==wanted)
                ck(tag+'/terminal file hashes',all(done[k+'_sha256']==h(folder/(k+'.json')) for k in ('trials','assignment_ledger','plan','manifest')))
                ck(tag+'/source environment manifest identities',manifest['research_source_sha256']==plan['source_sha256'] and read(folder/'environment.json')['pinned_environment']==plan['environment'] and set(snap['files'])==set(plan['source_sha256']) and manifest['source_snapshot_manifest_sha256']==h(folder/'source_snapshot_manifest.json'))
                bad=[]
                for k,digest in plan['source_sha256'].items():
                    rec=snap['files'][k];p=(folder/rec['snapshot_path']).resolve()
                    if not p.is_relative_to(folder.resolve()) or Path(rec['original_path']).resolve()!=source_path(k).resolve() or h(p)!=digest or rec['sha256']!=digest:bad.append(k)
                ck(tag+'/every136 actual source snapshot',not bad,bad)
            except Exception as exc:
                issues.append(dict(name=tag+'/stage_exception',error=repr(exc),traceback=traceback.format_exc()))
                rows.extend(dict(**j,status='unassessed_stage_error') for j in stage_jobs);continue
            for job in stage_jobs:
                run=Path(job['run']);label=tag+'/'+job['scenario']+'/'+str(job['seed'])+'/'+job['method'];row=dict(**job,status='unassessed');rows.append(row)
                try:
                    meta=read(run/'metadata.json');result=read(run/'result.json');path=run/'trace.npz'
                    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
                    t=a['time'];active=a['controller_active'].astype(bool)
                    ck(label+'/result-trials metadata exact',base.equal(result,next(v for v in trials if key(v)==key(job))) and result['trace_sha256']==meta['trace_sha256']==h(path) and meta['research_source_sha256']==plan['source_sha256'] and meta['task_policy']==job['task_policy'] and meta['receiver_policy']==job['receiver_policy'] and meta['motion_mode']==job['motion'])
                    ck(label+'/full16 clocks and32000raw',meta['completed'] and len(t)==3201 and np.array_equal(t,np.arange(3201)*.005) and len(a['substep_time'])==32000 and np.max(abs(a['substep_time']+a['substep_duration']-np.arange(1,32001)*.0005))<1e-8)
                    model,modelsha=hand_plant._model(FoundationConfig(**meta['config']));actor=ReferenceActor21(model)
                    ck(label+'/compiled model identity',modelsha==meta['model_sha256'])
                    ports=a['executed_reference21'].copy();ports[:,18]+=actor.lift_bias_m
                    ck(label+'/every200Hz21engine ports',ports.tobytes()==a['ctrl'][:,actor.indices21].tobytes() and a['robot_actuator_target21'].tobytes()==a['ctrl_before'][:,actor.indices21].tobytes())
                    obs=[json.loads(str(s)) for s in a['observation_json']];debug=[json.loads(str(s)) for s in a['debug_json']];graphs=[json.loads(str(s)) for s in a['receiver_release_tasks_json']]
                    domain=v17.domain_audit(a,[json.loads(str(s)) for s in a['legacy_common_sensor_json']])
                    ck(label+'/all3201 acquisition-domain masks metadata',domain['passed'],domain)
                    errors=[];last_version=None;last_gen=None
                    for i in np.flatnonzero(active):
                        f=obs[i];before=np.asarray(f['robot_reference21']);ex=a['executed_reference21'][i];mode=debug[i]['mode'];gen=int(a['mode_generation'][i])
                        if (np.max(abs(ex-before)-reference_rates(mode)*.005)>1e-12 or np.max(np.asarray(f['reference_lower21'])-ex)>1e-12 or np.max(ex-np.asarray(f['reference_upper21']))>1e-12):errors.append((int(i),'actual_permission_rate'))
                        if gen!=f['mode_generation'] or gen!=debug[i]['mode_generation'] or (ex-before).tobytes()!=a['reference_increment21'][i].tobytes() or abs(f['robot_control_time_s']-t[i])>1e-9:errors.append((int(i),'clock_gen_delta'))
                        changed=a['requested_reference21'][i].tobytes()!=ex.tobytes()
                        if changed!=bool(debug[i].get('execution_changed_plan')):errors.append((int(i),'receipt'))
                        if job['arm']=='full_V17' and t[i]>=12.-1e-9:
                            g=graphs[i]
                            if g is None or abs(g['time_s']-t[i])>1e-9 or g['holding_force_tasks_active'] or np.any(a['desired'][i]!=0) or g['placement_certified'] or g['placed'] or not g['no_regrip_after_release']:errors.append((int(i),'release_lifecycle'));continue
                            if g['phase']=='APPROACH_SUPPORT' and (not np.array_equal(ex[:18],f['release_entry_reference21'][:18]) or not np.array_equal(ex[19:],f['release_entry_reference21'][19:])):errors.append((int(i),'approach_entry_notheld'))
                            if last_version is not None and g['version']!=last_version and gen<=last_gen:errors.append((int(i),'generation_notinvalidated'))
                            last_version=g['version'];last_gen=gen
                    ck(label+'/everyactive5ms permission clock generation receipt',not errors,errors[:12])
                    ck(label+'/unknown future pressure remains NaN',meta['pressure_prediction_available'] is False and not np.any(a['prediction_available']) and np.isnan(a['prediction_at']).all() and np.isnan(a['predicted_normal']).all())
                    bad=[];counts=Counter();motion_summary=[]
                    for kind in ('joint','release'):
                        dm=read(run/(kind+'_decision_manifest.json'))
                        ck(label+'/'+kind+' decision hash',h(run/dm['file'])==dm['sha256'])
                        with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:records=[json.loads(s) for s in f]
                        if len(records)!=dm['records']:bad.append((kind,'manifest_count'))
                        for k,q in enumerate(records):
                            i=round(q['time_s']/.005);gen=q['generation'] if kind=='joint' else q['mode_generation'];counts[kind]+=1
                            if q['decision_id']!=k or not 0<=i<len(t) or abs(t[i]-q['time_s'])>1e-9 or gen!=int(a['mode_generation'][i]):bad.append((kind,k,'decision_clock_generation'))
                            if kind=='joint':
                                acc=q.get('proposal_acceptance') or {};native=q.get('native');accepted=bool(acc.get('accepted'))
                                if acc.get('different_force_feedback_problem_used') or (accepted and not acc.get('objective_bytes_equal')):bad.append((kind,k,'goal_substitution'))
                                if job['arm']=='full_V17' and q['mode']!='holding':bad.append((kind,k,'holding_controller_release_bypass'))
                                if native is not None and native.get('reference21') is not None:
                                    full=native['diagnostics'].get('prepared_value_final_full_recheck',{})
                                    if not accepted or not full.get('passed') or np.asarray(native['requested_delta_effort21']).tobytes()!=np.asarray(q['qp']['d21']).tobytes():bad.append((kind,k,'native_full_check'))
                            else:
                                accepted=q['geometry_proposal_accepted'];direct=q['phase']!='CONTACT_UNLOADING'
                                if accepted!=(q['hierarchy_status']=='CONVERGED_DECLARED_LINEAR_HIERARCHY') or (not accepted and not direct and not q['fallback']) or q['public_process']['holding_force_tasks_active']:bad.append((kind,k,'fallback_or_scope'))
                                e,_=v17.task_record(q);me,ev=motion_record(q)
                                if e or me:bad.append((kind,k,'recorded_task_evidence',e,me))
                                if ev['available']:motion_summary.append(dict(decision_id=k,time_s=q['time_s'],**ev))
                    ck(label+'/every saved decision native goal fullM frozen residual fallback',not bad,bad[:12])
                    lm=read(run/'delivery_manifest.json');ls=read(run/'delivery_summary.json')
                    with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:l={k:z[k] for k in z.files}
                    st,en=l['interval_start_s'],l['interval_end_s'];valid=l['contact_valid'].astype(bool)
                    ck(label+'/all32000 cached ledger original phase and hashes',len(en)==32000 and st.tobytes()==a['substep_time'].tobytes() and np.array_equal(st[1:],en[:-1]) and all(h(run/lm[k]['file'])==lm[k]['sha256'] for k in ('arrays','summary')) and ls['evaluation_only'] and not ls['controller_input_permitted'])
                    ii=np.rint(en/.005).astype(int);pick=np.flatnonzero((ii>=1)&(ii<=3200)&(abs(en-ii*.005)<1e-9)&l['object_state_valid'].astype(bool));qa,va=ls['object_qpos_address'],ls['object_qvel_address']
                    ck(label+'/ledger qv integration end exact',np.array_equal(l['object_free_qpos7_end'][pick],a['qpos'][ii[pick],qa:qa+7]) and np.array_equal(l['object_free_qvel6_end'][pick],a['qvel'][ii[pick],va:va+6]))
                    dt=np.maximum(0,np.minimum(en,16)-np.maximum(st,12));wm=dt>0;complete=bool(ls['window_coverage_complete'] and valid[wm].all())
                    floor=bool(np.any(l['target_floor_normal_N'][wm]>0)) if complete else None
                    ck(label+'/raw floor event and unknown retained',ls['floor_event'] is floor)
                    context_keys=('shape','scenario','seed','split','config','initial_position','limits','budget','noise','model_sha256','motion_mode','observer_warmup_s','receiver_instrument')
                    row.update(status='audited',trace_sha256=h(path),metadata_sha256=h(run/'metadata.json'),result_sha256=h(run/'result.json'),prefix75=fingerprints(a,7.5),prefix12=fingerprints(a,12.),physical_sensor_context={k:meta.get(k) for k in context_keys},decision_counts=dict(counts),motion_logged_records=motion_summary,domain=domain,active5ms_count=int(active.sum()),original_task_success=result['task_success'],original_full_motion=result['research_metrics']['full_motion_mission_success'],original_force_guard=result['research_metrics']['safety']['guard_pass'],floor_event=floor)
                    contexts[(job['motion'],job['scenario'],job['seed'])].append(row)
                    print(json.dumps(dict(audited=sum(r['status']=='audited' for r in rows),assigned=80)),flush=True)
                except Exception as exc:
                    row.update(status='reader_error',error=repr(exc));issues.append(dict(name=label+'/reader_exception',error=repr(exc),traceback=traceback.format_exc()))
    for context, group in contexts.items():
        label='/'.join(map(str,context));reference=group[0]
        ck(label+'/all5 context arms retained',len(group)==5 and Counter(r['arm'] for r in group)==Counter({'baselines':3,'holding_A':1,'full_V17':1}))
        ck(label+'/actual config noise receiver specification same',all(base.equal(r['physical_sensor_context'],reference['physical_sensor_context']) for r in group))
        mismatch=[dict(arm=r['arm'],method=r['method'],columns=same_fingerprints(r['prefix75'],reference['prefix75'])) for r in group]
        ck(label+'/all5 exact37 original physical and public prefix pre7.5',not any(v['columns'] for v in mismatch),mismatch)
        full=next((r for r in group if r['arm']=='full_V17'),None);holding=next((r for r in group if r['arm']=='holding_A'),None)
        ck(label+'/full versus A completeJSON and6metadata strictpre12',full is not None and holding is not None and not same_fingerprints(full['prefix12'],holding['prefix12']))
    ck('all80 individually assessed including task failures',len(rows)==80 and all(r['status']=='audited' for r in rows))
    ck('all16 context clusters checked',len(contexts)==16)
    ck('source136 unchanged after reading',all(sha(source_path(k))==v for k,v in plan['source_sha256'].items()))
    ck('zero forbidden physical pipeline calls',not forbidden,forbidden)
    all_checked=len(rows)==80 and all(r['status']=='audited' for r in rows)
    result=dict(schema='v17b_prospective80_execution_independent_v1',passed=not issues and all_checked,
        plan_sha256=PIN,assigned_count=80,all_assigned_checked=all_checked,created_utc=datetime.now(timezone.utc).isoformat(),
        rows=rows,checks=checks,issues=issues,terminal_gate=terminal,source_sha256=plan['source_sha256'],environment=plan['environment'],
        input_sha256=hashes,reader_sha256=sha(__file__),shared_helpers_sha256={str(ROOT/n):sha(ROOT/n) for n in ('review/read_v15_active9_independent_v1.py','review/v17_saved_reader_helpers_v1.py')},
        retained16_resume64_chain=continuation,
        scope=dict(physics_steps=0,plant_or_data_constructions=0,solver_replays=0,logged_fullM_residual_checked='all available saved matrices',
                   independent_robot_M_reconstruction='separate six-record appendix required; not implied by logged residual checks',
                   six_contexts_predeclared=MOTION_SELECTIONS),
        limitations=['Execution/source acceptance is separate from task success and comparative performance.',
            'Two physical seeds; no n80 independent population claim. Failures and quality NULLs not selected away.',
            'All200Hz21 ports checked; no nonexistent2kHz controller command telemetry invented.',
            'No all-packet RNG reconstruction: same code/specification plus actual full public prep and full/A pre12 exact; fixed draw recipe supported by prior constructor/logging gate.',
            'All saved M/target matrices checked for shape/time/observed target and achieved residual. This does not independently rederive every M from robot FK.',
            'No performance metrics recalculated here; parent/affine own raw and auxiliary measurements.'])
    out.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    out.with_suffix('.md').write_text(f"# V17b前瞻80执行来源独审\n\n{'PASS' if result['passed'] else 'FAIL'}；{sum(c['passed'] for c in checks)}/{len(checks)}。完整审查{sum(r['status']=='audited' for r in rows)}/80条；source136、二物理seed分母、六批全部来源/分配、实际5ms21ports、37列共同前缀、保存M/target残差与全部失败见JSON。\n\n本门不代替质量/稳定交付的主指标独审，不给胜负；完整M的机器人重建为另列六样本附录，不能把已保存残差检查当所有M独立重建。\n",encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],assigned_count=80,all_assigned_checked=all_checked,checks=len(checks),issues=issues[:8],out=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

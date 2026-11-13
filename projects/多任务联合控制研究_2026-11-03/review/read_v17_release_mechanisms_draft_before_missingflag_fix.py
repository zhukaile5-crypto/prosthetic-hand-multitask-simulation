"""Append three V17 trials to the frozen twelve-record release readout.

Offline only. No imports of runtime/controllers/model/solver. The frozen force,
ledger and event formulas remain unchanged. CONTACT denominators are separate
from direct stages and from post-clear lack of patches.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse,gzip,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'review/v16b_release_mechanisms_full12_v1.json'
PRIOR_SHA='5ea145c6da446b2fe581f798e0694184a018d06a41eb24a553ec3cfb2d2e2c9d'
BASE=ROOT/'review/read_v16_release_mechanisms_v1.py'
BASE_SHA='571fbdab19ec01f00b8923f68023be5db9774a4de6df1e5106e79d010b103c4d'
APPEND=ROOT/'review/read_v16b_release_mechanisms_v1.py'
APPEND_SHA='60e1699a0f362296e5d37127cc604bb56d1be1beaa5c9570f992482140c65e6f'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def dist(x):
    x=[float(v) for v in x if v is not None and np.isfinite(v)]
    return None if not x else dict(n=len(x),min=min(x),median=float(np.median(x)),max=max(x))


def phase_technical(ds):
    """Only CONTACT is a requested solver domain; direct motion is not failure."""
    stats={};transitions=[];previous=None
    for d in ds:
        if d['phase']!=previous:transitions.append(dict(time_s=d['time_s'],phase=d['phase']));previous=d['phase']
    for phase in sorted({d['phase'] for d in ds}):
        rows=[d for d in ds if d['phase']==phase];contact=phase=='CONTACT_UNLOADING'
        gaps=[];steps=[];goal_pending=[]
        for d in rows:
            cur=np.asarray(d['current_reference21']);nom=np.asarray(d['nominal_reference21']);u=np.asarray(d['proposed_reference_increment21']);scale=np.asarray(d['scale21'])
            wanted=(nom-cur)[:18]/scale[:18];gap=float(wanted@wanted)
            tol=128*np.finfo(float).eps*max(1.,gap)
            gaps.append(float(np.max(abs(nom[:18]-cur[:18]))));steps.append(bool(np.any(u!=0)))
            if gap>tol:goal_pending.append(dict(time_s=d['time_s'],projection=float((u[:18]/scale[:18])@wanted),roundoff=tol))
        diags=[d.get('hierarchy_diagnostics') or {} for d in rows]
        accepted=[d for d in rows if d.get('geometry_proposal_accepted')]
        motion=[d for d in rows if d.get('motion_task_available')]
        cop=[((d.get('public_process') or {}).get('transfer') or {}).get('continuous_support_evidence') or {} for d in rows]
        trends=[x.get('cop_trend') for x in cop if x.get('cop_trend') is not None]
        stats[phase]=dict(decision_count=len(rows),first_s=rows[0]['time_s'],last_s=rows[-1]['time_s'],solver_requested=contact,
            contact_solver_requested_count=len(rows) if contact else 0,
            geometry_available_count=sum(d.get('geometry_status')=='EXPERIMENTAL_GEOMETRY_TASKS' for d in rows) if contact else None,
            geometry_accepted_count=len(accepted) if contact else None,
            hierarchy_returned_count=sum(d.get('hierarchy_status') is not None for d in rows) if contact else None,
            hierarchy_status=dict(Counter(str(d.get('hierarchy_status')) for d in rows)) if contact else None,
            fallback=dict(Counter(str(d['fallback']) for d in rows if d.get('fallback') is not None)) if contact else None,
            numerical_error_count=sum(d.get('numerical_error') is not None for d in rows) if contact else None,
            motion_task_available_count=len(motion) if contact else None,
            accepted_motion_task_count=sum(d.get('motion_task_available') and d.get('geometry_proposal_accepted') for d in rows) if contact else None,
            normal_progress_achieved_m=dist(x.get('normal_progress_achieved_m') for x in diags) if contact else None,
            accepted_normal_progress_achieved_m=dist((d.get('hierarchy_diagnostics') or {}).get('normal_progress_achieved_m') for d in accepted) if contact else None,
            proposed_normal_virtual_work_J=dist(x.get('normal_virtual_unloading_work_J') for x in diags) if contact else None,
            normal_work_role='observed pressure times proposed old-plane displacement; NOT actual contact work or pressure prediction',
            nonzero_20ms_proposed_reference_count=sum(steps),zero_20ms_proposed_reference_count=len(rows)-sum(steps),
            canonical_finger_gap_inf_rad=dist(gaps),canonical_goal_pending_count=len(goal_pending),
            nearzero_canonical_projection_when_pending=sum(abs(x['projection'])<=x['roundoff'] for x in goal_pending),
            canonical_role='CONTACT posture is last-level preference, not primary progress; post-goal zeros are not a lock',
            cop_trend_known_count=len(trends),cop_trend_motion_resolved_count=sum(t.get('motion_resolved_against_engineering_proxy') is True for t in trends))
    return dict(available=True,decision_count=len(ds),stage_counts=dict(Counter(d['phase'] for d in ds)),stage_transitions=transitions,
        phases=stats,accepted_geometry_count=sum(d.get('geometry_proposal_accepted',False) for d in ds),
        first_coordinated_retreat_s=next((d['time_s'] for d in ds if d['phase']=='FREE_CLEAR_AND_RETREAT'),None),
        last_public_process=None if not ds else ds[-1].get('public_process'),
        direct_stage_unknown_geometry_not_solver_failure=True,no_new_success_label=True)


def physical_phases(row,arrays,ds):
    if not row.get('measurements_computed') or not ds:return None
    p=Path(row['run'])
    with arrays.load(p/'trace.npz',allow_pickle=False) as z:
        t=z['time'];ref=z['executed_reference21'];sf=z['substep_skin_normal'];st=z['substep_time'];sd=z['substep_duration'];sp=z['substep_palm']
    with arrays.load(p/'delivery_intervals.npz',allow_pickle=False) as z:
        ls=z['interval_start_s'];le=z['interval_end_s'];q=z['object_free_qpos7_end'];v=z['object_free_qvel6_end']
        force={k:z[k] for k in ('target_receiver_top_normal_N','target_floor_normal_N','target_hand_normal_N')}
    meta=read(p/'metadata.json');limits=np.asarray(meta['limits']);budget=float(meta['budget'])
    transitions=[]
    for d in ds:
        if not transitions or transitions[-1][1]!=d['phase']:transitions.append((float(d['time_s']),d['phase']))
    if transitions[0][0]>12+1e-9:return dict(available=False,reason='no_software_phase_at_release_entry')
    spans=[(max(12.,s),min(16.,transitions[i+1][0] if i+1<len(transitions) else 16.),phase) for i,(s,phase) in enumerate(transitions)]
    groups={}
    for a,b,phase in spans:groups.setdefault(phase,[]).append((a,b))
    stats={}
    for phase,windows in groups.items():
        sw=sum((np.maximum(0.,np.minimum(st+sd,b)-np.maximum(st,a)) for a,b in windows),np.zeros(len(st)))
        lw=sum((np.maximum(0.,np.minimum(le,b)-np.maximum(ls,a)) for a,b in windows),np.zeros(len(ls)))
        sm=sw>0;lm=lw>0
        tm=np.zeros(len(t)-1,bool)
        for a,b in windows:tm|=(t[1:]>=a-1e-9)&(t[1:]<b-1e-9)
        delta=np.diff(ref,axis=0)[tm]
        if not sm.any() or not lm.any():stats[phase]=dict(available=False,reason='missing_fixed_phase_coverage');continue
        mass_momentum_unknown=True
        stats[phase]=dict(available=True,windows_s=windows,duration_s=float(sw.sum()),raw_substeps=int(sm.sum()),
            peak_any_finger_N=float(sf[sm].max()),finger_peak5_N=sf[sm].max(axis=0),palm_peak_N=float(sp[sm].max()),
            local_excess_integral_Ns=float(np.sum(np.maximum(sf-limits,0)*sw[:,None])),
            total_excess_integral_Ns=float(np.maximum(sf.sum(axis=1)-budget,0)@sw),
            raw_positive_finger_count_time_mean=float(((sf>0).sum(axis=1)*sw).sum()/sw.sum()),
            raw_positive_finger_count_role='strict measured cached normal >0, descriptive; not a new contact qualification threshold',
            object_max_horizontal_speed_m_s=float(np.linalg.norm(v[lm,:2],axis=1).max()),
            object_max_linear_speed_m_s=float(np.linalg.norm(v[lm,:3],axis=1).max()),
            object_max_angular_speed_rad_s=float(np.linalg.norm(v[lm,3:],axis=1).max()),
            object_first_position_m=q[np.flatnonzero(lm)[0],:3],object_last_position_m=q[np.flatnonzero(lm)[-1],:3],
            object_first_velocity6=v[np.flatnonzero(lm)[0]],object_last_velocity6=v[np.flatnonzero(lm)[-1]],
            momentum_not_computed_without_separately_declared_mass=mass_momentum_unknown,
            contacts={k:dict(peak_N=float(x[lm].max()),impulse_Ns=float(x@lw),positive_duration_s=float(lw[x>0].sum())) for k,x in force.items()},
            executed_5ms_step_count=len(delta),executed_nonzero_5ms_step_count=int(np.count_nonzero(np.any(delta!=0,axis=1))),
            executed_reference_TV_finger18_rad=float(abs(delta[:,:18]).sum()),executed_reference_TV_lift_m=float(abs(delta[:,18]).sum()),
            clocks='phase from task decision clock; cached force labels interval start, object state labels integrated end; no instantaneous equality assumed')
    events={}
    for category,key in [('top','first_positive_state'),('hand','last_positive_state'),('floor','first_positive_state')]:
        event=(row['release_contact_ledger'].get(category) or {}).get(key)
        if event is None:events[category]=None;continue
        end=event['integration_end_time_s'];i=int(np.argmin(abs(st+sd-end)))
        events[category]=dict(event=event,matched_force_record_end_s=float(st[i]+sd[i]),time_match_error_s=float(abs(st[i]+sd[i]-end)),
            cached_finger_normal5_N=sf[i],cached_palm_N=float(sp[i]),strict_positive_finger_count=int(np.count_nonzero(sf[i]>0)),
            role='same cached integration end label; all-external finger pressure, target-only contact belongs exclusively to offline ledger')
    return dict(available=True,by_phase=stats,contact_event_hand_force=events,
        approach_preservation_evaluated_from_actual_pressures_and_object_velocities_not_reference_alone=True,
        no_low_momentum_threshold_or_success_label_added=True)


def self_test():
    def d(phase,accepted=False,status=None):
        return dict(time_s=12.,phase=phase,current_reference21=[0.]*21,nominal_reference21=[0.]*21,
            proposed_reference_increment21=[0.]*21,scale21=[1.]*21,geometry_status='NOT_REQUESTED_DIRECT_STAGE' if phase!='CONTACT_UNLOADING' else 'UNAVAILABLE_GEOMETRY_TASKS',
            geometry_proposal_accepted=accepted,hierarchy_status=status,fallback=None,public_process={})
    x=phase_technical([d('APPROACH_SUPPORT'),d('CONTACT_UNLOADING'),d('FREE_CLEAR_AND_RETREAT')]);p=x['phases']
    assert p['APPROACH_SUPPORT']['contact_solver_requested_count']==0
    assert p['FREE_CLEAR_AND_RETREAT']['geometry_accepted_count'] is None
    assert p['CONTACT_UNLOADING']['decision_count']==1 and p['CONTACT_UNLOADING']['geometry_available_count']==0
    assert p['CONTACT_UNLOADING']['canonical_goal_pending_count']==0 and p['CONTACT_UNLOADING']['zero_20ms_proposed_reference_count']==1
    assert p['FREE_CLEAR_AND_RETREAT']['nearzero_canonical_projection_when_pending']==0
    print(json.dumps(dict(self_test_passed=True,checks=5,physics_steps=0)))


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--out');parser.add_argument('--execute',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args(argv)
    if args.self_test:return self_test()
    if not args.execute or not args.out:raise ValueError('Explicit --execute --out only after root confirms complete V17 cohort')
    out=Path(args.out);out=out if out.is_absolute() else ROOT/out
    if out.exists():raise FileExistsError(out)
    for p,h in [(PRIOR,PRIOR_SHA),(BASE,BASE_SHA),(APPEND,APPEND_SHA)]:
        if sha(p)!=h:raise RuntimeError('Frozen input/helper identity changed: '+str(p))
    plan_path=ROOT/'review/v17_transfer_small_plan_v1.json';plan=read(plan_path)
    stages={s['motion']:ROOT/s['output'] for s in plan['stages'] if s['policy']=='coordinated'}
    for folder in set(stages.values()):
        done=read(folder/'execution_complete.json')
        if done.get('execution_complete') is not True or done.get('sources_unchanged') is not True:raise RuntimeError('Whole new cohort not terminal/source-stable')
    helper=module('v17_frozen_force_ledger',BASE);append=module('v17_frozen_event_alignment',APPEND)
    pubpath=ROOT/'review/read_v16_release_geometry_six_cases_v1.py';public=module('v17_frozen_public_geometry',pubpath)
    rows=deepcopy(read(PRIOR)['records']);cache_dir=ROOT/'review/v17_release_mechanism_cache_v1';cache_dir.mkdir(exist_ok=True)
    decision_cache={}
    def technical(folder):
        with gzip.open(Path(folder)/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:ds=[json.loads(line) for line in f]
        ds=[d for d in ds if 12-1e-9<=d['time_s']<=16+1e-9];decision_cache[str(folder)]=ds;return phase_technical(ds)
    helper.technical=technical
    for motion,scene in [('hold','fragile_conflict'),('transport','nominal'),('transport','friction_drop')]:
        path=stages[motion]/'runs'/f'{scene}__ratio_effort__619732';arrays=append.ArrayCache();helper.np=append.NumpyProxy(arrays)
        try:
            row=helper.run_record(path,'contact_V17',(motion,scene),cache_dir)
            if not row.get('measurements_computed'):raise RuntimeError('Missing assigned V17 run: '+str(path))
            row['role']='corrected_V16b';append.goal_diagnostics(row,helper);row['role']='contact_V17'
            if row.get('goal_comparison',{}).get('available') is not True:raise RuntimeError('Missing actual clearance goal')
            row['stage_alignment']=append.stage_alignment(row,arrays,public)
            if str(path) not in decision_cache:technical(path)
            row['release_decisions']=phase_technical(decision_cache[str(path)])
            row['phase_physical_diagnostics']=physical_phases(row,arrays,decision_cache[str(path)]);rows.append(row)
        finally:arrays.close();helper.np=np
    rows.sort(key=lambda r:(r['context']['motion_mode'],r['context']['scenario'],r['role']))
    complete=len(rows)==15 and all(r.get('measurements_computed') and all(r.get('coverage',{}).get(k) for k in ('full16','ledger_release_known','raw_force_complete')) for r in rows)
    if not complete:raise RuntimeError('All15 complete fixed records required')
    pairs=[]
    for new in [r for r in rows if r['role']=='contact_V17']:
        for old in rows:
            if old['context']==new['context'] and old['role']!='contact_V17':
                pairs.append(dict(context=new['context'],candidate_role=new['role'],reference_role=old['role'],
                    candidate_record_index=rows.index(new),reference_record_index=rows.index(old)))
    document=dict(schema='v17_contact_task_fifteen_release_mechanisms_v1',complete=True,planned=15,computed=15,records=rows,pairs=pairs,
        source_sha256={Path(__file__).name:sha(__file__),BASE.name:BASE_SHA,APPEND.name:APPEND_SHA,pubpath.name:sha(pubpath)},
        frozen_prior_twelve_sha256=PRIOR_SHA,new_plan_sha256=sha(plan_path),new_execution_source_sha256=plan['source_sha256'],
        scope=dict(physics_steps=0,models_constructed=0,controllers_run=0,old_labels_or_metrics_changed=False,
            candidate3_seen_contexts_onephysicalseed_not15independent_draws=True,normal_virtual_work_not_actual_contact_work=True,
            solver_denominator_only_CONTACT_not_direct_or_postclear=True,correct_goal_not_clearance_or_placement=True,
            continuous_receiver_load_or_no_floor_not_success=True))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(helper.plain(document),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(complete=True,computed=15,pairs=len(pairs),path=str(out),sha256=sha(out))))


if __name__=='__main__':main()

"""Read-only fixed nine-record V16 release mechanism comparison.

New3 vs old candidateA3 vs same-prior legacy ratio3. It does not run controllers,
models, evaluators, or change any original labels. Pending data remain pending.
"""
from pathlib import Path
from collections import Counter
import argparse,ast,gzip,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT.parent/'五指策略接入_2026-11-03'/'foundation_config.py'
OPEN18=np.concatenate(next(ast.literal_eval(n.value) for n in ast.parse(CONFIG.read_text(encoding='utf-8')).body
    if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='OPEN' for x in n.targets)))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def rotation_angle(q0,q):
    a=q0/np.linalg.norm(q0);b=q/np.linalg.norm(q,axis=1)[:,None]
    w=b[:,0]*a[0]+b[:,1:]@a[1:]
    v=a[0]*b[:,1:]-b[:,0,None]*a[1:]-np.cross(np.broadcast_to(a[1:],b[:,1:].shape),b[:,1:])
    return 2*np.arctan2(np.linalg.norm(v,axis=1),abs(w))


def technical(p):
    file=p/'release_decisions.jsonl.gz'
    if not file.is_file():return dict(available=False,reason='legacy has no V16 geometric decision telemetry; not zero failures')
    with gzip.open(file,'rt',encoding='utf-8') as f:ds=[json.loads(line) for line in f]
    ds=[d for d in ds if 12-1e-9<=d['time_s']<=16+1e-9]
    stats=dict(available=True,decision_count=len(ds),accepted_geometry_count=sum(d['geometry_proposal_accepted'] for d in ds),
        geometry_status=dict(Counter(d.get('geometry_status') for d in ds)),
        hierarchy_status=dict(Counter(str(d.get('hierarchy_status')) for d in ds)),
        fallback=dict(Counter(str(d.get('fallback')) for d in ds if d.get('fallback') is not None)),
        numerical_error_count=sum(d.get('numerical_error') is not None for d in ds),
        adapter_unknown_reason=dict(Counter(d.get('geometry',{}).get('adapter_reason') for d in ds if d.get('geometry',{}).get('adapter_reason'))),
        stage_counts=dict(Counter(d['phase'] for d in ds)),requested_OPEN_decisions=0,
        accepted_requested_OPEN_decisions=0,nearzero_OPEN_progress_count=0,
        accepted_nearzero_OPEN_progress_count=0,progress_ratios=[],
        nearzero_definition='abs(dot(delta/scale,wanted/scale)) <=128eps*max(1,||wanted/scale||^2); numerical algebraic zero, not a new physical threshold')
    transitions=[];previous=None;permission=Counter();status=Counter();copmoving=0;copknown=0;retreat=[];deadline=[]
    for d in ds:
        if d['phase']!=previous:transitions.append(dict(time_s=d['time_s'],phase=d['phase']));previous=d['phase']
        current=np.asarray(d['current_reference21']);nom=np.asarray(d['nominal_reference21']);delta=np.asarray(d['proposed_reference_increment21']);scale=np.asarray(d['scale21'])
        wanted=(nom-current)[:18]/scale[:18];u=delta[:18]/scale[:18];den=float(wanted@wanted)
        tol=128*np.finfo(float).eps*max(1.,den)
        if den>tol:
            stats['requested_OPEN_decisions']+=1
            accepted=d['geometry_proposal_accepted'];stats['accepted_requested_OPEN_decisions']+=int(accepted)
            progress=float(u@wanted);zero=abs(progress)<=tol
            stats['nearzero_OPEN_progress_count']+=int(zero)
            stats['accepted_nearzero_OPEN_progress_count']+=int(zero and accepted)
            stats['progress_ratios'].append(progress/den)
        pp=d.get('public_process') or {};tr=pp.get('transfer') or {};payload=tr.get('continuous_support_evidence') or {}
        permission[str(tr.get('stage_permission'))]+=1;status[str(tr.get('status'))]+=1
        trend=payload.get('cop_trend')
        if trend is not None:copknown+=1;copmoving+=int(trend.get('motion_resolved_against_engineering_proxy') is True)
        if pp.get('coordinated_retreat_started'):retreat.append(d['time_s'])
        if pp.get('deadline_without_coordinated_retreat'):deadline.append(d['time_s'])
    ratios=stats.pop('progress_ratios')
    stats.update(stage_transitions=transitions,transfer_permission=dict(permission),transfer_status=dict(status),
        zero_requested_OPEN_decisions=len(ds)-stats['requested_OPEN_decisions'],
        cop_trend_known_decisions=copknown,cop_trend_motion_resolved_decisions=copmoving,
        first_coordinated_retreat_s=min(retreat) if retreat else None,
        first_deadline_without_coordinated_retreat_s=min(deadline) if deadline else None,
        requested_OPEN_progress_ratio_summary=None if not ratios else dict(min=float(min(ratios)),median=float(np.median(ratios)),max=float(max(ratios))),
        nearzero_fraction_of_requested=None if not stats['requested_OPEN_decisions'] else stats['nearzero_OPEN_progress_count']/stats['requested_OPEN_decisions'],
        nearzero_fraction_of_accepted_requested=None if not stats['accepted_requested_OPEN_decisions'] else stats['accepted_nearzero_OPEN_progress_count']/stats['accepted_requested_OPEN_decisions'],
        last_public_process=None if not ds else ds[-1].get('public_process'),
        last_decision=None if not ds else {k:ds[-1].get(k) for k in ('time_s','phase','current_reference21','nominal_reference21','proposed_reference_increment21','geometry_proposal_accepted','hierarchy_status')})
    return stats


def run_record(p,role,case,cache_dir):
    required=('metadata.json','result.json','trace.npz','delivery_intervals.npz')
    missing=[n for n in required if not (p/n).is_file()]
    base=dict(role=role,context=dict(motion_mode=case[0],scenario=case[1],physical_seed=619732),run=str(p))
    if missing:return dict(**base,status='pending_or_missing',missing=missing,measurements_computed=False)
    identity={n:sha(p/n) for n in required}
    for n in ('release_decisions.jsonl.gz','delivery_summary.json'):
        if (p/n).is_file():identity[n]=sha(p/n)
    key=hashlib.sha256(json.dumps(dict(inputs=identity,reader=sha(__file__)),sort_keys=True).encode()).hexdigest()
    cache=cache_dir/(key+'.json')
    if cache.is_file():return read(cache)
    meta=read(p/'metadata.json');result=read(p/'result.json')
    # Each selected compressed column is fetched exactly once per run.
    with np.load(p/'trace.npz',allow_pickle=False) as z:
        t={k:z[k] for k in ('time','skin_normal','palm','lift','executed_reference21','substep_time','substep_duration','substep_skin_normal','substep_palm')}
    with np.load(p/'delivery_intervals.npz',allow_pickle=False) as z:
        l={k:z[k] for k in ('interval_start_s','interval_end_s','contact_valid','object_state_valid','object_free_qpos7_end','object_free_qvel6_end',
             'target_receiver_top_normal_N','target_receiver_otherface_normal_N','target_floor_normal_N','target_hand_normal_N','receiver_any_actor_top_normal_N')}
    clock=t['time'];starts=l['interval_start_s'];ends=l['interval_end_s'];dt=ends-starts
    full=bool(len(clock)==3201 and np.allclose(clock,np.arange(3201)*.005,atol=1e-8,rtol=0) and
        len(starts)>0 and abs(starts[0])<1e-8 and abs(ends[-1]-16)<1e-8 and np.all(dt>0) and
        np.allclose(starts[1:],ends[:-1],atol=1e-9,rtol=0))
    lm=(ends>12+1e-9)&(starts<16-1e-9);weights=np.maximum(0,np.minimum(ends,16)-np.maximum(starts,12))
    known=bool(full and l['contact_valid'][lm].all() and l['object_state_valid'][lm].all())
    q=l['object_free_qpos7_end'];v=l['object_free_qvel6_end'];entry=int(np.argmin(abs(ends-12)))
    angles=rotation_angle(q[entry,3:],q[:,3:]) if np.isfinite(q[:,3:]).all() and (np.linalg.norm(q[:,3:],axis=1)>0).all() else None
    def state(i):return dict(integration_end_time_s=float(ends[i]),position_m=q[i,:3].tolist(),quaternion_wxyz=q[i,3:].tolist(),
        linear_speed_m_s=float(np.linalg.norm(v[i,:3])),horizontal_speed_m_s=float(np.linalg.norm(v[i,:2])),angular_speed_norm_rad_s=float(np.linalg.norm(v[i,3:])),
        rotation_change_from_release_rad=None if angles is None else float(angles[i]),target_hand_normal_N=float(l['target_hand_normal_N'][i]),
        target_top_normal_N=float(l['target_receiver_top_normal_N'][i]),target_floor_normal_N=float(l['target_floor_normal_N'][i]))
    contacts={}
    for label,col in [('top','target_receiver_top_normal_N'),('other_receiver_face','target_receiver_otherface_normal_N'),('floor','target_floor_normal_N'),('hand','target_hand_normal_N'),('receiver_any_actor_top','receiver_any_actor_top_normal_N')]:
        force=l[col];hits=np.flatnonzero(lm&(force>0))
        contacts[label]=None if not known else dict(first_positive_end_s=None if not len(hits) else float(ends[hits[0]]),
            last_positive_end_s=None if not len(hits) else float(ends[hits[-1]]),peak_N=float(force[lm].max()),impulse_Ns=float(force@weights),
            positive_contact_duration_s=float(weights[force>0].sum()),end16_N=float(force[-1]),
            first_positive_state=None if not len(hits) else state(int(hits[0])),last_positive_state=None if not len(hits) else state(int(hits[-1])))
    st=t['substep_time'];sd=t['substep_duration'];sf=t['substep_skin_normal'];sm=(st>=12-1e-9)&(st<16-1e-9)
    subcomplete=bool(len(st)>0 and abs(st[0])<1e-8 and abs(st[-1]+sd[-1]-16)<1e-8 and np.allclose(st[1:],st[:-1]+sd[:-1],atol=1e-8,rtol=0) and np.isfinite(sf).all())
    limits=np.asarray(meta['limits']);budget=float(meta['budget'])
    force=None if not subcomplete else dict(finger_peak_N=sf[sm].max(axis=0).tolist(),peak_any_finger_N=float(sf[sm].max()),
        local_excess_integral_Ns=float(np.sum(np.maximum(sf[sm]-limits,0)*sd[sm,None])),
        total_excess_integral_Ns=float(np.maximum(sf[sm].sum(axis=1)-budget,0)@sd[sm]),palm_peak_N=float(t['substep_palm'][sm].max()),
        role='cached integral record force; same original limits; stage start>=12, no new success/guard label')
    rm=result.get('research_metrics',{})
    out=dict(**base,status='computed',measurements_computed=True,input_sha256=identity,
        coverage=dict(full16=full,ledger_release_known=known,raw_force_complete=subcomplete,public_samples=len(clock),intervals=len(starts),raw_force_substeps=len(st)),
        original=dict(legacy_task_success=result.get('task_success'),record_complete=result.get('record_complete'),mission_success=rm.get('mission_success'),
            full_motion_mission_success=rm.get('full_motion_mission_success'),failure_at=rm.get('failure_at'),failure_reason=meta.get('failure_reason'),
            release=rm.get('release'),safety=rm.get('safety'),quality=rm.get('quality_windows')),
        release_contact_ledger=contacts,release_force=force,
        release_object=None if not known else dict(entry=state(entry),end16=state(len(ends)-1),max_rotation_from_release_rad=None if angles is None else float(angles[lm].max()),
            max_horizontal_displacement_from_entry_m=float(np.linalg.norm(q[lm,:2]-q[entry,:2],axis=1).max()),
            max_horizontal_speed_m_s=float(np.linalg.norm(v[lm,:2],axis=1).max())),
        reference=dict(post_action_at12_reference21=t['executed_reference21'][int(np.argmin(abs(clock-12)))].tolist(),end21=t['executed_reference21'][-1].tolist(),
            open18_declared_initial_pose=OPEN18.tolist(),end18_distance_to_declared_OPEN_inf=float(np.max(abs(t['executed_reference21'][-1,:18]-OPEN18))),
            end_lift_encoder_m=float(t['lift'][-1]),
            end200Hz_skin_normal5_N=t['skin_normal'][-1].tolist(),end200Hz_palm_N=float(t['palm'][-1]),
            end_cached_skin_normal5_N=sf[-1].tolist(),end_cached_palm_N=float(t['substep_palm'][-1]),
            old_frozen_release_thumb_terminal_reference5=[-1.,.1,0.,0.,0.],
            role='stored t12 executed_reference is post-action, not the incoming actual release-entry snapshot'),
        release_decisions=technical(p),not_new_placement_label=True)
    cache.write_text(json.dumps(plain(out),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return plain(out)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--only-hold',action='store_true');args=ap.parse_args(argv)
    out=Path(args.out);out=out if out.is_absolute() else ROOT/out
    if out.exists():raise FileExistsError(out)
    plan=read(ROOT/'review/v16_release_small_plan_v1.json')
    stages={x['motion']:ROOT/x['output'] for x in plan['stages'] if x['policy']=='coordinated'}
    cases=[('hold','fragile_conflict'),('transport','nominal'),('transport','friction_drop')]
    cache=ROOT/'review/v16_release_mechanism_cache_v1';cache.mkdir(exist_ok=True)
    rows=[]
    for case in cases:
        motion,scene=case;name=f'{scene}__ratio_effort__619732'
        for role,path in [('new_V16',stages[motion]/'runs'/name),
            ('old_candidate_A',ROOT/'output'/f'v15A_prospective_candidate_A_{motion}_v2'/'runs'/name),
            ('same_prior_legacy_ratio',ROOT/'output'/f'v15A_prospective_baseline_{motion}_v2'/'runs'/name)]:
            if args.only_hold and motion!='hold':
                rows.append(dict(role=role,context=dict(motion_mode=motion,scenario=scene,physical_seed=619732),
                    run=str(path),status='deferred_until_root_terminal',measurements_computed=False))
            else:rows.append(run_record(path,role,case,cache))
    complete=all(x['measurements_computed'] and x.get('coverage',{}).get('full16') for x in rows)
    if not complete and not args.allow_partial:raise RuntimeError('all nine complete inputs required; use a fresh clearly partial output')
    doc=dict(schema='v16_release_fixed_nine_mechanism_readout_v1',complete=complete,planned=9,
        computed=sum(x['measurements_computed'] for x in rows),pending=sum(not x['measurements_computed'] for x in rows),
        reader_sha256=sha(__file__),plan_sha256=sha(ROOT/'review/v16_release_small_plan_v1.json'),
        scope=dict(physics_steps=0,models_imported=0,controllers_run=0,source_files_changed=False,old_labels_unchanged=True,
            fixed_window_s=[12,16],contact_clock='cached solve interval plus integrated end-state labels, not continuous-time/contact recomputation',
            positive_contact_definition='force>0; no new material/placement threshold',
            nearzero_OPEN='numerical projected OPEN-progress proxy only; original labels retained',
            sample_scope='three seen failure contexts, all physical seed619732; no population claim',
            old_V16_decision_telemetry_missing_is_not_zero_failure=True,
            continuous_receiver_load_not_placement=True),records=rows)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(plain(doc),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),complete=complete,computed=doc['computed'],pending=doc['pending']),ensure_ascii=False))


if __name__=='__main__':main()

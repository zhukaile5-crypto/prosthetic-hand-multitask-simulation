"""Posthoc two-case release mechanisms; saved data only, one load per array.

Target identity/object state are privileged OFFLINE evaluation, never input to
the controller. No solver, robot model, FK, contact function or step is called.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CASES=(('nominal_transport','nominal',619741),('friction_transport','friction_drop',619742))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def first(t,mask):
    ids=np.flatnonzero(mask)
    return None if not len(ids) else float(t[ids[0]])
def last(t,mask):
    ids=np.flatnonzero(mask)
    return None if not len(ids) else float(t[ids[-1]])
def stats(x):
    a=np.asarray(x,float)
    return None if not len(a) else dict(n=len(a),mean=float(np.mean(a)),min=float(np.min(a)),max=float(np.max(a)))
def angle(q,origin):return float(2*np.arccos(np.clip(abs(np.dot(q/np.linalg.norm(q),origin/np.linalg.norm(origin))),0,1)))

def run_case(stage,scene,seed,aux):
    p=ROOT/f'output/v18_small_{stage}_v1/runs/{scene}__ratio_effort__{seed}'
    m=read(p/'metadata.json');original=read(p/'result.json')
    needed=('time','normal','skin_normal','palm','hand_other','all_hand_object','support','lift',
            'executed_reference21','reference_increment21','observation_json','raw_patches_json',
            'substep_time','substep_duration','substep_skin_normal','substep_palm','substep_hand_other')
    with np.load(p/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in needed}
    with np.load(p/'delivery_intervals.npz',allow_pickle=False) as z:l={k:z[k] for k in z.files}
    with gzip.open(p/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:all_d=[json.loads(s) for s in f]
    ds=[d for d in all_d if 12-1e-9<=d['time_s']<16-1e-9]
    t=a['time'];st=a['substep_time'];dur=a['substep_duration'];end=st+dur
    lt=l['interval_end_s'];release=(lt>12+1e-9)&(lt<=16+1e-9)
    indices=np.flatnonzero((t>=12-1e-9)&(t<=16+1e-9))
    obs={int(i):json.loads(str(a['observation_json'][i])) for i in indices}
    raw={int(i):json.loads(str(a['raw_patches_json'][i])) for i in indices}
    changed=[dict(time_s=d['time_s'],phase=d['phase']) for i,d in enumerate(ds) if not i or d['phase']!=ds[i-1]['phase']]
    top=l['target_receiver_top_normal_N'];anytop=l['receiver_any_actor_top_normal_N'];floor=l['target_floor_normal_N'];hand=l['target_hand_normal_N']
    public=[]
    for i in indices:
        o=obs[int(i)];r=o['receiver_surface'];acq=r.get('acquisition_time_s')
        if r['valid'] and acq is not None and acq>=12-1e-9 and r['normal_load_N']>r['uncertainty6'][2]:public.append(int(i))
    firstR=None if not public else float(t[public[0]])
    events=dict(release_request_s=12.,first_target_top_interval_end_s=first(lt,release&(top>0)),
        first_receiver_any_actor_interval_end_s=first(lt,release&(anytop>0)),
        first_non_target_receiver_interval_end_s=first(lt,release&((anytop-top)>1e-9)),
        first_public_receiver_compression_time_s=firstR,
        first_public_receiver_acquisition_s=None if not public else obs[public[0]]['receiver_surface']['acquisition_time_s'],
        first_target_floor_interval_end_s=first(lt,release&(floor>0)),
        last_target_hand_positive_interval_end_s=last(lt,release&(hand>0)),
        original_frame_clearance_confirmation_s=None if original['release_latency'] is None else 12+original['release_latency'],
        phases=changed)
    entry_i=int(np.argmin(abs(lt-12.)));entry_q=l['object_free_qpos7_end'][entry_i]
    def snapshot(at):
        j=int(np.argmin(abs(lt-at)));i=int(np.searchsorted(t,at+1e-9)-1);i=max(int(indices[0]),min(int(indices[-1]),i))
        o=obs[i];q=l['object_free_qpos7_end'][j];v=l['object_free_qvel6_end'][j];pre=o['robot_reference21'];post=a['executed_reference21'][i]
        d=ds[min(max(int(round((float(t[i])-12)/.005)),0),len(ds)-1)]
        process=d['public_process'];r=o['receiver_surface'];H=process.get('raw_coacquired_loads')
        patches=raw[i];on_target=[x for x in patches if x.get('is_target')]
        raw_target_f=sum(np.asarray(x['force'],float) for x in on_target) if on_target else np.zeros(3)
        return dict(requested_event_s=at,ledger_interval_start_s=float(l['interval_start_s'][j]),ledger_interval_end_s=float(lt[j]),
            public_time_s=float(t[i]),phase=d['phase'],target_top_N=float(top[j]),receiver_any_actor_top_N=float(anytop[j]),
            target_floor_N=float(floor[j]),target_hand_N=float(hand[j]),object_position_world_m=q[:3].tolist(),
            object_linear_velocity_world_m_s=v[:3].tolist(),object_linear_speed_m_s=float(np.linalg.norm(v[:3])),
            object_horizontal_speed_m_s=float(np.linalg.norm(v[:2])),object_angular_speed_rad_s=float(np.linalg.norm(v[3:])),
            object_linear_momentum_world_Ns=(float(m['config']['mass'])*v[:3]).tolist(),
            object_orientation_change_from_release_rad=angle(q[3:],entry_q[3:]),
            public_lift_q_m=float(o['robot_position25'][24]),public_lift_v_m_s=float(o['robot_velocity25'][24]),
            incoming_lift_reference_m=float(pre[18]),post_action_lift_reference_m=float(post[18]),
            incoming_reference_debt_m=float(o['robot_position25'][24]-pre[18]),
            post_action_reference_debt_m=float(o['robot_position25'][24]-post[18]),
            public_normal5_N=o['normal'],public_aux_pressure_N=o['auxiliary_pressure'],hand_acquisition_s=o['sample_time'],
            receiver_acquisition_s=r.get('acquisition_time_s'),receiver_public_N=r.get('normal_load_N'),
            current_signed_H_R=H,unload_fraction=d.get('unload_fraction'),
            tangent_task_active_digits=d['contact_intents'].get('tangent_active_digits'),
            raw_target_contact_count=len(on_target),raw_target_force_palm_N=np.asarray(raw_target_f).tolist(),
            raw_target_slip_speed_max_m_s=max([float(np.linalg.norm(x['slip_velocity'])) for x in on_target]+[0.]),
            estimated_normal_intent_sum_N=None if d['qp'] is None else float(np.sum(d['qp'].get('zP',[]))),
            observed_normal_task_sum_N=None if d['qp_problem'] is None else float(np.sum(d['qp_problem']['y_p'])),
            true_target_geometry_is_offline_only=True)
    snapshots={}
    for label,event in events.items():
        if isinstance(event,(float,int)) and event is not None:snapshots[label]=snapshot(event)
    for e in changed:snapshots['phase_'+e['phase']]=snapshot(e['time_s'])
    snapshots['end']=snapshot(16.)
    if events['first_target_top_interval_end_s'] is not None:
        qtop=l['object_free_qpos7_end'][int(np.argmin(abs(lt-events['first_target_top_interval_end_s'])))];aftertop=release&(lt>=events['first_target_top_interval_end_s']-1e-9)
        displacement=l['object_free_qpos7_end'][aftertop,:3]-qtop[:3]
        trajectory=dict(max_horizontal_displacement_after_first_top_m=float(np.max(np.linalg.norm(displacement[:,:2],axis=1))),
            final_horizontal_displacement_after_first_top_m=float(np.linalg.norm(displacement[-1,:2])),
            max_horizontal_speed_after_first_top_m_s=float(np.max(np.linalg.norm(l['object_free_qvel6_end'][aftertop,:2],axis=1))),
            max_angular_speed_after_first_top_rad_s=float(np.max(np.linalg.norm(l['object_free_qvel6_end'][aftertop,3:],axis=1))))
    else:trajectory=None
    phase_stats=[]
    for k,e in enumerate(changed):
        start=e['time_s'];stop=changed[k+1]['time_s'] if k+1<len(changed) else 16.
        selected=[d for d in ds if start-1e-9<=d['time_s']<stop-1e-9]
        normal_robot=[];tangent_power=[];tangent_slip=[];hasP=0;zero_intent=0;debt=[];liftv=[];drift=[]
        for d in selected:
            oi=int(round(d['time_s']/.005));o=obs[oi];v=np.asarray(o['robot_velocity25'],float);p0=d['qp_problem']
            if p0 is not None and len(p0['y_p']):
                J=np.asarray(p0['Jpx25'],float);y=np.asarray(p0['y_p'],float);w=J@v
                normal_robot.append(float(np.sum(y*w)/np.sum(y)));hasP+=1
                if d['unload_fraction']==1.:zero_intent+=1
            records=d['contact_intents'].get('tangent_records',[])
            tau=sum((np.asarray(x['generalized_effort25'],float) for x in records),np.zeros(25))
            tangent_power.append(float(tau@v));tangent_slip.extend(float(np.linalg.norm(x['slip2_m_s'])) for x in records)
            debt.append(d['governor']['actual_reference_lag_m']);liftv.append(v[24])
            drift.append(float(np.linalg.norm(np.asarray(d['proposed_reference21'])[19:]-np.asarray(d['current_reference21'])[19:])))
        sm=(st>=start-1e-9)&(st<stop-1e-9)
        lm=(lt>start+1e-9)&(lt<=stop+1e-9)
        phase_stats.append(dict(phase=e['phase'],start_s=start,stop_s=stop,decisions=len(selected),normal_geometry_decisions=hasP,
            accepted_effort_goal_count=sum(d['effort_goal_realized'] for d in selected),
            qp_status_counts=dict(Counter('no_problem' if d['qp'] is None else d['qp']['status'] for d in selected)),
            fallback_counts=dict(Counter(str(d.get('fallback')) for d in selected)),
            unload_fraction_1_with_geometry_count=zero_intent,
            observed_pressure_weighted_robot_normal_velocity_m_s=stats(normal_robot),
            normal_velocity_role='positive=inward ROBOT motion along delayed public plane normal; not robot-external relative velocity',
            virtual_tangent_effort_current_robot_power_W=stats(tangent_power),
            tangent_power_role='signed device-plane EFFORT INTENT dot current robot velocity; not measured contact dissipation',
            device_slip_norm_m_s=stats(tangent_slip),lift_actual_velocity_m_s=stats(liftv),incoming_lift_reference_debt_m=stats(debt),
            wrist_proposal_step_norm_rad=stats(drift),
            raw_finger_peak_N=float(np.max(a['substep_skin_normal'][sm])) if np.any(sm) else None,
            raw_palm_peak_N=float(np.max(a['substep_palm'][sm])) if np.any(sm) else None,
            target_top_impulse_Ns=float(np.sum(top[lm]*l['interval_duration_s'][lm])),
            receiver_any_actor_top_impulse_Ns=float(np.sum(anytop[lm]*l['interval_duration_s'][lm])),
            target_hand_impulse_Ns=float(np.sum(hand[lm]*l['interval_duration_s'][lm])),
            target_floor_impulse_Ns=float(np.sum(floor[lm]*l['interval_duration_s'][lm]))))
    peakmask=(st>=12-1e-9)&(st<16-1e-9);ids=np.flatnonzero(peakmask);ri,fi=np.unravel_index(int(np.argmax(a['substep_skin_normal'][peakmask])),a['substep_skin_normal'][peakmask].shape);pi=int(ids[ri])
    peak=dict(finger=int(fi),force_N=float(a['substep_skin_normal'][pi,fi]),interval_start_s=float(st[pi]),interval_end_s=float(end[pi]))
    peak['state']=snapshot(peak['interval_end_s'])
    # Event-local public task windows are fixed +/-25ms, not selected for scores.
    near=[]
    for d in ds:
        if firstR is not None and firstR-.025-1e-9<=d['time_s']<=firstR+.05+1e-9:
            o=obs[int(round(d['time_s']/.005))]
            near.append(dict(time_s=d['time_s'],phase=d['phase'],H_R=d['public_process'].get('raw_coacquired_loads'),
                unload_fraction=d['unload_fraction'],normal_intent_p=d['qp_problem']['z_intent_p'] if d['qp_problem'] else None,
                requested_velocity_m_s=d['governor']['requested_velocity_m_s'],governed_velocity_m_s=d['governor']['governed_velocity_intent_m_s'],
                actual_lift_velocity_m_s=o['robot_velocity25'][24],current_ref_lift_m=d['current_reference21'][18],executed_ref_lift_m=d['proposed_reference21'][18],
                receiver_fresh_streak=d['public_process']['receiver_fresh_streak'],acceptance=d['effort_goal_realized']))
    selected_aux=next(x for x in aux['records'] if x['stage']==stage)
    return dict(context=dict(scenario=scene,motion_mode='transport',physical_seed=seed),run=str(p),events=events,event_states=snapshots,
        phase_statistics=phase_stats,release_raw_peak=peak,trajectory=trajectory,public_tasks_near_first_receiver=near,
        original_task_success=original['task_success'],original_release_success=original['release_success'],original_clearance_latency_s=original['release_latency'],
        original_safety=original['safety'],existing_auxiliary_metrics=selected_aux['metrics'],
        source_sha256=m['research_source_sha256'],input_sha256={f:sha(p/f) for f in ('trace.npz','delivery_intervals.npz','metadata.json','result.json','release_decisions.jsonl.gz')},
        coverage=dict(truth_samples=len(t),raw_intervals=len(st),ledger_intervals=len(lt),release_decisions=len(ds),end_decision_bookkeeping=len(all_d)-len(ds)),
        offline_target_truth_used_only_for_evaluation=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    auxpath=ROOT/'review/v18_small_auxiliary_v1.json';aux=read(auxpath)
    records=[run_case(*case,aux) for case in CASES]
    payload=dict(schema='v18_two_release_floor_mechanisms_posthoc_v1',complete=True,records=records,
        input_auxiliary_sha256=sha(auxpath),reader_sha256=sha(__file__),physics_steps=0,models_constructed=0,
        no_source_or_results_modified=True,
        timing_scope='cached force uses actual interval start/end; q/v object ledger end-state; public fresh-boundary observation has own10ms acquisition',
        task_scope='normal z and device tangential terms are INTENTS; robot J@v is old-plane robot motion, not relative external dynamics',
        causal_scope='chronology plus structural hypotheses; no isolated factor causality or new success labels')
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for r in records:print(json.dumps(dict(context=r['context'],events=r['events'],trajectory=r['trajectory'],
                                         peak={k:v for k,v in r['release_raw_peak'].items() if k!='state'},
                                         phase=[{k:x[k] for k in ('phase','start_s','stop_s','decisions','accepted_effort_goal_count','qp_status_counts','raw_finger_peak_N','unload_fraction_1_with_geometry_count')} for x in r['phase_statistics']]),ensure_ascii=False))
    print(json.dumps(dict(out=str(out),sha256=sha(out)),ensure_ascii=False))
if __name__=='__main__':main()

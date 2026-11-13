"""V10 terminal-only public policy mechanism readout; no physical stepping.

Selected-model forecasts are re-derived from PUBLIC q/v and pinned measured
K20. They are paired with actual future robot encoders, not object state.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from types import MethodType
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'review'))
import verify_v10_controller_public_dispatch_v1 as replay
from verify_robot_servo_response_v1 import AXES25, UNITS25, error_summary


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(x):return x['scenario'],x['method'],int(x['seed'])


def canonical(row):
    r=row['research_metrics'];s=r['safety'];release=r['release']
    return dict(legacy_task=r['mission_success'],full_motion_task=r['full_motion_mission_success'],
        failure_reason=r['failure_reason'],failure_at=r['failure_at'],wrist_failure_at=r['wrist_tracking_guard']['failure_at'],
        whole_force_guard=s['guard_pass'],whole_peak_by_finger_N=s['finger_peak_N'],whole_local_excess_Ns=s['local_excess_integral_Ns'],
        holding_safety=s['stages']['policy_holding_transport'],stage_safety=s['stages'],
        physical_release=release['physical_success'],task_qualified_release=release['task_qualified_success'],release_latency_s=release['physical_clearance_latency_s'],
        initial_stability_wait_s=r['initial_stability_wait_s'],
        static_quality=r['static_quality'],dynamic_quality=r['dynamic_quality'],entire_holding_quality=r['entire_holding_quality'],
        dynamic_motion_tracking=r['progress']['dynamic_motion_tracking'],holding_motion_tracking=r['progress']['holding_motion_tracking'])


def public_qv(value):
    q=np.r_[np.asarray(value['raw_joint'],float),float(value['lift'])]
    if q.shape!=(25,) or not np.isfinite(q).all():raise ValueError('Public robot25 positions missing')
    velocity=value.get('lift_velocity_m_s')
    if velocity is None:return q,None
    v=np.r_[np.asarray(value['raw_joint_velocity'],float),float(velocity)]
    return q,(v if v.shape==(25,) and np.isfinite(v).all() else None)


class Models:
    def __init__(self):self.cache={}
    def at(self,selection):
        f=selection['source_fingerprints'];cid=selection['context_id']
        if cid not in self.cache:
            for pathkey,digestkey in (('model','model_sha256'),('dense_kernel','dense_kernel_sha256'),('short_comparison','short_comparison_sha256')):
                if sha(f[pathkey])!=f[digestkey]:raise ValueError('Pinned selected servo data changed')
            model=read(f['model']);short=read(f['short_comparison'])
            if tuple(model['output_axes25'])!=AXES25:raise ValueError('Selected robot25 order mismatch')
            with np.load(f['dense_kernel'],allow_pickle=False) as z:R=z['response25x8'][4].copy()
            error=np.asarray(short['metrics_by_horizon']['0.02']['A']['max_abs_per_axis25'],float)
            self.cache[cid]=(R,error,deepcopy(f))
        return self.cache[cid]


def reconstruct_inner_stabilizer(metadata,times,observations,tasks,debug,requested,executed_motion):
    """Conditional software reconstruction on ACTUAL joint observations/plans.

    Not an alternate physical baseline rollout, and not the hypothetical
    uniform nominal intention used by the model optimizer.
    """
    c=replay.construct(metadata);captured={};original_step=c.nominal.step
    def capture(_self,*args,**kwargs):
        value=original_step(*args,**kwargs);captured['action']=deepcopy(value);return value
    c.nominal.step=MethodType(capture,c.nominal)
    tick_lookup={int(round(float(t)/.005)):i for i,t in enumerate(times)}
    def recorded_decision(self,before):
        i=tick_lookup[int(round(float(self.latest_frame.time)/.005))];d=debug[i]
        self.plan_active=d.get('status')=='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP';self.robot_predicted=None;self.debug=deepcopy(d)
        if self.plan_active:
            self.plan_start=float(self.latest_frame.time);self.plan_reference=np.r_[self.state.alpha,self.executed_motion]
            self.plan_delta=np.asarray(d['applied_input_increment_8'],float);self.reference_before=deepcopy(before)
    c._decide=MethodType(recorded_decision,c)
    reasons=Counter();rows=[];max_action=max_motion=0.;ticks=0
    for i in np.flatnonzero((times>=7.3-1e-9)&(times<12.-1e-9)):
        f=replay.frame(observations[i])
        if times[i]<7.5-1e-9:c.state.update(f,.005);continue
        c.set_task_request(tasks[i]);result=c.step(f);ticks+=1
        max_action=max(max_action,float(np.max(abs(result.delta_closure-requested[i]))))
        max_motion=max(max_motion,float(np.max(abs(c.motion_reference-executed_motion[i]))))
        action=captured['action'];pause=bool(action.pause_transport)
        if pause:
            reason=('observability' if c.nominal.label=='unobservable_pause' or np.count_nonzero(c.state.valid)<3 else str((action.debug or {}).get('qp_status','not_logged')))
            reasons[reason]+=1
            if result.debug.get('stabilizer_pause_replaced_by_directional_plan'):
                rows.append(dict(time_s=float(times[i]),nominal_label=c.nominal.label,qp_status=(action.debug or {}).get('qp_status'),
                    valid_count=int(np.count_nonzero(c.state.valid)),reason_class=reason))
    return dict(available=max_action<=1e-12 and max_motion<=1e-12,policy_ticks=ticks,maximum_action_replay_difference=max_action,
        maximum_motion_replay_difference=max_motion,nominal_pause_reason_tick_counts=dict(reasons),pause_override_records=rows,
        scope='actual V10 observations and recorded accepted plans; exact software-state reconstruction, not a counterfactual physical baseline or future uniform intention')


def joint_mechanisms(directory,models):
    metadata=read(directory/'metadata.json');trace=directory/'trace.npz'
    with np.load(trace,allow_pickle=False) as z:
        t=z['time'].copy();obs_strings=z['observation_json'].copy();task_strings=z['task_request_json'].copy();debug_strings=z['debug_json'].copy()
        alpha_before=z['alpha_before'].copy();alpha_after=z['alpha'].copy();action=z['requested_action'].copy();executed=z['executed_action'].copy();motion=z['executed_motion_reference'].copy()
        command=z['command'].copy();pressure_prediction=z['prediction_available'].copy()
    indices=np.flatnonzero((t>=7.3-1e-9)&(t<=12.+1e-9));obs={i:json.loads(str(obs_strings[i])) for i in indices}
    tasks={i:json.loads(str(task_strings[i])) for i in indices};debug={i:json.loads(str(debug_strings[i])) for i in indices}
    policy=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9));lookup={int(round(float(v)/.005)):i for i,v in enumerate(t)}
    decisions=[i for i in policy if abs((t[i]-7.5)/.02-round((t[i]-7.5)/.02))<1e-7]
    records=[];errors=[];errors_realized=[];empirical=[];bank=[]
    for i in decisions:
        d=debug[i]
        if 'servo_selection' in d:
            s=d['servo_selection'];bank.append(dict(time_s=float(t[i]),status=d.get('servo_status'),context_id=s.get('context_id'),
                normalized_reference_distance=s.get('normalized_reference_distance'),robot_state_distance=s.get('robot_state_distance'),robot_velocity_distance=s.get('robot_velocity_distance'),
                reference_outside_component_footprint=s.get('reference_outside_component_footprint'),nearest_recorded_mixed_point_normalized_distance=s.get('nearest_recorded_mixed_point_normalized_distance'),
                mode_scope=s.get('mode_scope'),sequence_support=s.get('sequence_support')))
        if d.get('status')!='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP':continue
        increment=np.asarray(d['applied_input_increment_8'],float);start=np.r_[obs[i]['alpha'],obs[i]['executed_motion_reference']]
        base_tick=int(round(float(t[i])/.005));quarter_indices=[lookup.get(base_tick+j) for j in range(4)]
        release_truncated=any(k is None or str(command[k])=='release' or t[k]>=12.-1e-9 for k in quarter_indices)
        actual=[];requested_vs_executed=[];late=[]
        if not release_truncated:
            actual=np.array([np.r_[alpha_after[k],motion[k]] for k in quarter_indices])
            expected=start+np.arange(1,5)[:,None]/4*increment
            quarter_error=np.max(abs(actual-expected),axis=0)
            requested_vs_executed=np.max(abs(action[quarter_indices]-executed[quarter_indices]),axis=0).tolist()
            late=[bool(debug[k].get('planned_finite_ramp_realization_changed',False)) for k in quarter_indices]
            realized=bool(np.max(quarter_error)<=1e-10)
        else:quarter_error=None;realized=False
        record=dict(start_time_s=float(t[i]),context_id=d.get('servo_selection',{}).get('context_id'),applied_increment8=increment.tolist(),
            four_quarters_available=not release_truncated,release_truncated=release_truncated,reference_ramp_realized=realized,
            max_actual_reference_quarter_deviation8=None if quarter_error is None else quarter_error.tolist(),
            requested_execution_difference5=requested_vs_executed,late_permission_or_domain_flags=late,
            rollback_flag_any=any(debug[k].get('reference_memory_rollback',False) for k in quarter_indices if k is not None),
            nominal_pause_overridden=any(debug[k].get('stabilizer_pause_replaced_by_directional_plan',False) for k in quarter_indices if k is not None),
            allocator_status=d.get('allocation',{}).get('status'),candidate_component_check=d.get('candidate_future_component_check'),
            exact_model_geometry_cost_before=d.get('exact_model_geometry_cost_before'),exact_model_geometry_cost_after=d.get('exact_model_geometry_cost_after'),
            geometry_cost_scope='predicted robot FK under declared finite-ramp intentions; not actual future force or physical baseline comparison')
        future=lookup.get(base_tick+4)
        if future is None or future not in obs or release_truncated:
            record['prediction_error_available']=False;record['unavailable_reason']='missing future public robot endpoint or release-truncated ramp'
        else:
            q,v=public_qv(obs[i]);actual_q,_=public_qv(obs[future])
            if v is None:
                record['prediction_error_available']=False;record['unavailable_reason']='public robot lift/angular velocity missing; not filled with zero'
            else:
                R,error,fingerprints=models.at(d['servo_selection']);prediction=q+.02*v+R@increment;delta=actual_q-prediction
                record.update(prediction_error_available=True,predicted_robot25=prediction.tolist(),actual_robot25_at_20ms=actual_q.tolist(),
                    robot_error25=delta.tolist(),ctx_empirical_error25=error.tolist(),exceeds_ctx_observed_error25=(abs(delta)>error+1e-12).tolist(),
                    ctx_error_excess25=np.maximum(0.,abs(delta)-error).tolist(),source_fingerprints=fingerprints,
                    error_comparison_scope='empirical observed context error, not a guaranteed or probabilistic bound')
                errors.append(delta);empirical.append(abs(delta)>error+1e-12)
                if realized and not any(late):errors_realized.append(delta)
        records.append(record)
    inner=reconstruct_inner_stabilizer(metadata,t,obs,tasks,debug,action,motion)
    statuses=Counter(debug[i].get('status','NO_V10_STATUS') for i in policy)
    decision_status=Counter(debug[i].get('status','NO_V10_STATUS') for i in decisions)
    result=dict(trace_sha256=sha(trace),policy_ticks=len(policy),public_decisions=len(decisions),tick_status_counts=dict(statuses),decision_status_counts=dict(decision_status),
        intervention_decisions=len(records),intervention_ticks=statuses.get('EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP',0),
        late_realization_flag_ticks=sum(bool(debug[i].get('planned_finite_ramp_realization_changed',False)) for i in policy),
        rollback_flag_ticks=sum(bool(debug[i].get('reference_memory_rollback',False)) for i in policy),
        rollback_count_semantics='flag occupancy and per-decision any; a sticky flag is not counted as distinct PI rollback calls',
        explicit_extra_freeze_ticks=sum(bool(debug[i].get('extra_motion_freeze',False)) for i in policy),
        stabilizer_pause_override_flag_ticks=sum(bool(debug[i].get('stabilizer_pause_replaced_by_directional_plan',False)) for i in policy),
        pressure_prediction_available_ticks=int(pressure_prediction[policy].sum()),bank_decisions=bank,accepted_plans=records,
        all_plan_robot_error_summary=None if not errors else error_summary(errors),
        realized_reference_plan_robot_error_summary=None if not errors_realized else error_summary(errors_realized),
        empirical_error_exceed_count25=None if not empirical else np.array(empirical).sum(axis=0).tolist(),
        empirical_error_comparison_denominator=len(empirical),inner_stabilizer_replay=inner,
        limitations=['no plant/physics executed by analysis','geometry costs are predicted finite FK, not pressure forecasts','q25 errors diagnose model migration only; they do not establish force causality','grouped patch-normal geometry may hide opposing individual patch displacements'])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hold',type=Path,default=ROOT/'output/development_v10_hold_v1');parser.add_argument('--transport',type=Path,default=ROOT/'output/development_v10_transport_v1')
    parser.add_argument('--hold-only',action='store_true',help='Read the complete terminal hold cohort, never partial trials')
    parser.add_argument('--out',type=Path,default=ROOT/'review/v10_factorial_mechanisms_v1');args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing mechanism evidence; choose a fresh output')
    batches=[]
    for motion,path in ((('hold',args.hold),) if args.hold_only else (('hold',args.hold),('transport',args.transport))):
        finish=read(path/'execution_complete.json')
        if not finish.get('execution_complete'):raise ValueError('Both full batches must be terminal before performance readout')
        if sha(path/'trials.json')!=finish['trials_sha256']:raise ValueError('Terminal trials identity mismatch')
        plan=read(path/'plan.json');manifest=read(path/'manifest.json');trials=read(path/'trials.json')
        if {key(j) for j in plan['jobs']}!={key(r) for r in trials}:raise ValueError('Complete assignment/result coverage required; no failed trial removal')
        batches.append((motion,path,finish,plan,manifest,trials))
    if len(batches)==2 and batches[0][4]['research_source_sha256']!=batches[1][4]['research_source_sha256']:raise ValueError('Hold/transport execution source versions differ')
    models=Models();pairs=[];all_results=[]
    for motion,path,finish,plan,manifest,trials in batches:
        indexed={key(r):r for r in trials}
        for row in trials:all_results.append(dict(motion=motion,scenario=row['scenario'],method=row['method'],seed=row['seed'],metrics=canonical(row)))
        for job in plan['jobs']:
            if job['method']!='joint':continue
            scenario,_,seed=key(job);joint=indexed[(scenario,'joint',seed)];baseline=indexed[(scenario,'ratio_preview',seed)]
            jdir=path/'runs'/f'{scenario}__joint__{seed}';bdir=path/'runs'/f'{scenario}__ratio_preview__{seed}'
            jm=canonical(joint);bm=canonical(baseline)
            metadata=read(jdir/'metadata.json');bmdata=read(bdir/'metadata.json')
            pair=dict(motion=motion,scenario=scenario,seed=seed,joint=jm,ratio_preview=bm,
                same_actual_physical_config=metadata['config']==bmdata['config'],
                jointly_eligible_quality_windows={w:bool(jm[w]['eligible'] and bm[w]['eligible']) for w in ('static_quality','dynamic_quality','entire_holding_quality')},
                mechanisms=joint_mechanisms(jdir,models))
            pairs.append(pair)
    args.out.mkdir(parents=True)
    result=dict(schema='v10_factorial_public_mechanism_readout_v1',complete=True,all_assigned_results=all_results,pairs=pairs,
        trial_count=len(all_results),pair_count=len(pairs),robot_axes25=AXES25,robot_units25=UNITS25,
        batch_identity=[dict(motion=m,path=str(p),trials_sha256=f['trials_sha256'],manifest_sha256=sha(p/'manifest.json'),sources_unchanged=f['sources_unchanged'],runtime_failures=f['runtime_failures']) for m,p,f,_,_,_ in batches],
        source_sha256=sha(__file__),analysis_helpers={str(Path(replay.__file__)):sha(replay.__file__)},new_physics=False,
        scope='one seen seed, four environments and two public motion modes; no population superiority or uniform-intention causal baseline claim')
    with (args.out/'readout.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    lines=[f'# V10完整{len(all_results)}条：公共策略机制读数','',
        '所有分配与结果均保留。按运动×环境逐对比较joint与同preview的ratio；不把一个已见种子的40局当成40次独立物理随机重复。','',
        '| 运动/环境 | joint/ratio完整任务 | joint/ratio全程限力 | joint/ratio持握超限 N·s | 新介入计划 | 真正兑现四quarter | pause覆盖tick |',
        '|---|---|---|---|---:|---:|---:|']
    for p in pairs:
        j=p['joint'];b=p['ratio_preview'];m=p['mechanisms'];realized=sum(x['reference_ramp_realized'] for x in m['accepted_plans'])
        lines.append(f"| {p['motion']}/{p['scenario']} | {int(j['full_motion_task'])}/{int(b['full_motion_task'])} | {int(j['whole_force_guard'])}/{int(b['whole_force_guard'])} | {j['holding_safety']['local_excess_integral_Ns']:.6f}/{b['holding_safety']['local_excess_integral_Ns']:.6f} | {m['intervention_decisions']} | {realized} | {m['stabilizer_pause_override_flag_ticks']} |")
    lines+=['','每个被接受的新决策单列实际公开q25/v25、选中完整K20和applied_increment8重算的20ms数值预测，并配对真实未来机器人编码器。四个参考quarter的实际兑现、权限晚变、请求与执行不一致、释放截断分别保留。经验error25超覆盖只说明超出该context既有观测误差，不是违反硬物理界。','',
        '原稳定器pause原因由实际joint观测与已记录介入计划进行软件状态重构，只有动作与motion重放一致时才使用。它不是另一条物理ratio轨迹；真实方法效果使用同条件ratio_preview独立实际结果。所有uniform名义意图的几何成本比较仍只是模型内意图比较。','',
        f'JSON包含本次完整{len(all_results)}局V3结果、共同品质窗资格、每场景status/Bank距离/fallback/rollback/extra-freeze、全部新计划20ms机器人误差与25轴经验超覆盖，以及实际追踪、释放与全程/持握安全。没有因为任务失败或品质缺窗删除分母。','',
        'No physics was run by this readout; force safety/global model admission is not granted.']
    (args.out/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,trials=len(all_results),pairs=len(pairs),output=str(args.out),readout_sha256=sha(args.out/'readout.json'))))


if __name__=='__main__':main()

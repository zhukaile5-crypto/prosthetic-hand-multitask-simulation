"""Independent terminal V3 pairing/raw audit and phase-specific outcome readout.

No producer module imports, no plant construction and no physical stepping.
Training/holdout superposition is analyzed separately, never used to select a
winning profile or overwrite the frozen020ms data.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from collections import Counter
import ast,base64,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/release_local_response_capsule_v3'
OUT=ROOT/'review/release_local_response_v3_terminal_audit_v1.json'
PURE=ROOT/'review/verify_release_basis_v2_independent.py'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
WALL={'qp_seconds','control_ms','wall_seconds'}
checks=[];issues=[]


def check(name,value,detail=None):
    row=dict(name=name,passed=bool(value),detail=detail);checks.append(row)
    if not value:issues.append(row)


def metric(name,a,b):
    same=a is b if a is None or b is None else a==b if isinstance(a,(bool,str)) or isinstance(b,(bool,str)) else bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))
    check(name,same,None if same else dict(actual=a,reported=b))


def raw_variation(t,force):
    f=np.asarray(force,float);duration=float(t[-1]-t[0]);tv=abs(np.diff(f,axis=0)).sum(axis=0);net=abs(f[-1]-f[0])
    reverse=np.maximum(0.,(tv-net)/2.)
    return dict(samples=len(t),duration_s=duration,finger_TV_N=float(tv.sum()),finger_TV_rate_N_s=float(tv.sum()/duration),
        net_drift_N=float(net.sum()),reversal_N=float(reverse.sum()),per_finger_TV_N=tv.tolist(),
        interpretation='raw release transition path variation, not stable-holding quality or a hard material guard')


def load_raw_arithmetic():
    tree=ast.parse(PURE.read_text(encoding='utf-8'))
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','digest','encode','strip_wall','mapping','at','dwell','variation','clearance_detail'}]
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(PURE)+'::pure_helpers','exec'),globals())
    jobloop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='job' and isinstance(n.iter,ast.Name) and n.iter.id=='jobs')
    start=next(i for i,n in enumerate(jobloop.body) if isinstance(n,ast.With) and 'np.load(trace' in ast.unparse(n))
    stop=next(i for i,n in enumerate(jobloop.body) if isinstance(n,ast.With) and 'decoder_commands.npz' in ast.unparse(n))
    body=[]
    for n in jobloop.body[start:stop]:
        if isinstance(n,ast.Expr) and any(isinstance(v,ast.Constant) and isinstance(v.value,str) and 'summary_' in v.value for v in ast.walk(n)):continue
        body.append(deepcopy(n))
    names=('safety','release_values','mission','x','t')
    body.append(ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in ('safety','release','mission','raw','time')],values=[ast.Name(id=k,ctx=ast.Load()) for k in names])))
    function=ast.FunctionDef(name='raw_check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ('jid','d','reported','r','m','trace')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[],type_params=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(PURE)+'::raw_arithmetic','exec'),globals())


def same_trace(zero,branch,fields,origin=None):
    mismatches=[]
    with np.load(zero,allow_pickle=False) as x,np.load(branch,allow_pickle=False) as y:
        for name in fields:
            a,b=x[name],y[name]
            if origin is not None:
                clock=x['substep_time'] if name.startswith('substep_') else x['time'];mask=clock<=origin+1e-9;a,b=a[mask],b[mask]
            equal=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
            if not equal:mismatches.append(name)
        if origin is None:observed=np.array_equal(x['observation_json'],y['observation_json'])
        else:
            mask=x['time']<=origin+1e-9;observed=np.array_equal(x['observation_json'][mask],y['observation_json'][mask])
    return not mismatches and observed,dict(physical_mismatches=mismatches,public_observations_exact=bool(observed))


def main():
    summary=read(RUN/'summary.json')
    if not summary.get('execution_complete'):raise RuntimeError('All14 terminal attempts required; no partial conclusions')
    if OUT.exists():raise FileExistsError(OUT)
    load_raw_arithmetic()
    manifest=read(RUN/'manifest.json');plan=read(RUN/'plan.json');snap=read(RUN/'source_snapshot_manifest.json');rows=read(RUN/'results.json');jobs=manifest['jobs'];ledger=read(RUN/'assignment_ledger.json')
    check('14_assigned_terminal_attempts_retained',len(rows)==len(jobs)==len(ledger)==14 and Counter(r['job_id'] for r in rows)==Counter(j['job_id'] for j in jobs)==Counter(r['job']['job_id'] for r in ledger) and rows==summary['results']==read(RUN/'results_partial.json'))
    check('manifest_plan_and_snapshot_identities',plan==manifest['plan'] and sha(RUN/'plan.json')==manifest['plan_sha256'] and sha(RUN/'source_snapshot_manifest.json')==manifest['source_snapshot_manifest_sha256'])
    expected_origins={'phi_150ms':30,'phi_350ms':70};variants={'zero':(0.,0.),'phi_slow':(1.,0.),'lift_slow':(0.,1.),'mixed_50_50':(.5,.5),'mixed_50_100':(.5,1.),'mixed_100_50':(1.,.5),'mixed_100_100':(1.,1.)}
    check('two_origins_independent_3train_4heldout',plan['planned']==14 and plan['training']==['zero','phi_slow','lift_slow'] and set(plan['strict_holdout'])==set(variants)-set(plan['training']) and {r['stage']:r['relative_control_tick'] for r in plan['probe_origins']}==expected_origins and all(sum(j['stage']==s and j['training_role']=='train' for j in jobs)==3 and sum(j['stage']==s and j['training_role']=='heldout_mixed' for j in jobs)==4 for s in expected_origins))
    source_rows=[]
    for name,h in manifest['sources'].items():
        e=snap['files'][name];current=Path(e['original_path']);copy=RUN/e['snapshot_path']
        source_rows.append(dict(name=name,match=sha(current)==sha(copy)==e['sha256']==h and current.stat().st_size==copy.stat().st_size==e['bytes']))
    check('244_source_snapshots_and_originals_exact',len(source_rows)==244 and all(r['match'] for r in source_rows) and set(snap['files'])==set(manifest['sources']))
    failures=[r for r in rows if r['status']!='complete'];check('terminal_failure_ledger_exact',failures==summary['failures']==read(RUN/'runtime_failures.json'))
    indexed={(r['stage'],r['variant']):r for r in rows};records={};profiles={};paths={};entries={};metas={}
    physical=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position','relative','rotation','relative_velocity','angular_velocity','lift','lift_ref','wrist_position','requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha','executed_action','warning_count','collision','actual_motion_target','executed_motion_reference','penetration','world_penetration','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
    for job in jobs:
        stage=job['stage'];v=job['variant'];pair=(stage,v);jid=job['job_id'];row=indexed[pair]
        if row['status']!='complete':records[jid]=dict(stage=stage,variant=v,status=row['status'],failure=row);continue
        d=Path(job['directory']);trace=d/'record/trace.npz';reported=read(d/'result.json');r=reported['research_metrics'];m=read(d/'record/metadata.json');metas[pair]=m;paths[pair]=trace
        check(jid+'/artifact_identity',sha(trace)==reported['trace_sha256']==r['trace_sha256']==m['trace_sha256']==row['trace_sha256'] and m['research_source_sha256']==manifest['sources'] and reported['stage']==stage and reported['training_role']==job['training_role'])
        check(jid+'/nominal_context_only',m['seed']==610120 and m['shape']=='capsule' and m['config']['mass']==.12 and m['config']['friction']==.6 and m['config']['timestep']==.0005 and m['config']['control_dt']==.005)
        record=raw_check(jid,d,reported,r,m,trace);x=record.pop('raw');t=record.pop('time')
        record.update(stage=stage,variant=v,training_role=job['training_role'],status='complete',stages=r['safety']['stages'],initial_stability_wait_s=r['initial_stability_wait_s'],full_motion_mission_success=r['full_motion_mission_success'],event_recovery=r['event_recovery'])
        mask=(t>=12.-1e-9)&(t<=16.+1e-9)
        record['release_raw_200Hz_target_TV_diagnostic']=raw_variation(t[mask],x['normal'][mask])
        record['release_raw_200Hz_all_external_skin_TV_diagnostic']=raw_variation(t[mask],x['skin_normal'][mask])
        with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
        profiles[pair]=p;origin_tick=expected_origins[stage];a,b=variants[v];tick=p['relative_control_tick'];weight=np.clip((tick-origin_tick)/4.,0.,1.);tau=p['control_time']-12.
        phi=tau if a==0 else (2400.+tick-4.*a*weight)*.005-12.;lclock=tau if b==0 else (2400.+tick-4.*b*weight)*.005-12.;lift=.1*(1.-np.clip(lclock/.6,0,1))
        check(jid+'/8801_held_coordinates_actual_denominator',len(tick)==8801 and np.array_equal(weight,p['ramp_weight']) and np.array_equal(phi,p['actual_phi_s']) and np.array_equal(lclock,p['lift_clock_s']) and np.array_equal(lift,p['actual_command21'][:,18]))
        check(jid+'/21references_to_actual22ctrl',np.array_equal(p['actual_command21'][:,:18],p['actual_ctrl'][:,2:20]) and np.array_equal(p['actual_command21'][:,19:],p['actual_ctrl'][:,:2]) and np.max(abs((p['actual_ctrl'][:,20]-lift)-(p['native_carrier_ctrl'][:,20]-p['native_carrier_command21'][:,18])))<1e-15)
        unique=np.r_[True,np.diff(tick)!=0];commands=p['actual_command21'][unique];phase=p['actual_phi_s'][unique];same=np.diff(tick)==0
        check(jid+'/held_phase_native_rates_and_floor',np.all(np.diff(phase)>=-1e-12) and np.all(np.diff(phase)<=.005+1e-12) and np.all(np.diff(p['actual_command21'],axis=0)[same]==0) and lift[-1]==0. and np.all((lift>=0)&(lift<=.1)))
        with np.load(d/'robot_physical_stream.npz',allow_pickle=False) as z:stream={k:z[k].copy() for k in z.files}
        clock=stream['physical_time_s']
        check(jid+'/32001_robot25_endclock_raw_pressure',len(clock)==32001 and np.allclose(clock,np.arange(32001)*.0005,atol=1e-7,rtol=0) and stream['position25'].shape==stream['velocity25'].shape==(32001,25) and np.array_equal(stream['external_finger_pressure5_N'][1:],x['substep_skin_normal']) and np.array_equal(stream['target_finger_pressure5_N'][1:],x['substep_normal']) and np.allclose(clock[1:],x['substep_time']+x['substep_duration'],atol=1e-12,rtol=0))
        with np.load(trace,allow_pickle=False) as z:
            q=np.column_stack((z['qpos'][:,m['raw_joint_qpos_indices']],z['lift']));velocity=z['qvel'][:,list(m['raw_joint_qpos_indices'])+[0]]
        check(jid+'/robot25_control_versus_physical_exact',np.array_equal(q,stream['position25'][::10]) and np.array_equal(velocity,stream['velocity25'][::10]))
        entry={}
        for name in ('release_entry','probe_entry'):
            ep=d/name;side=read(ep/'entry_side_state.json');signature=read(ep/'entry_signature.json');local=read(ep/'local_control_state.json')
            with np.load(ep/'entry_physics.npz',allow_pickle=False) as z:physics={k:z[k].item() if z[k].shape==() else z[k] for k in z.files}
            check(jid+'/'+name+'/checksums',dict(signature,local_control_state_sha256=digest(local))==row['entries'][name] and signature['physics_arrays_sha256']==digest(encode(physics)) and signature['integration_state_sha256']==digest(encode(physics['integration_state'])) and signature['python_causal_side_state_sha256']==digest(strip_wall(side)))
            check(jid+'/'+name+'/RNG_queue_FK_present',at(side,'sensors','rng').get('numpy_generator')=='PCG64' and len(at(side,'sensors','queue')['items'])>0 and {'integration_state','qpos','qvel','ctrl','qacc_warmstart'}<=set(mapping(at(side,'skin_geometry','fk_data'))))
            entry[name]=dict(physics=physics,side=strip_wall(side),local=local)
        entries[pair]=entry
        record['phase_contract']=dict(origin_control_s=(2400+origin_tick)*.005,full_input_control_s=(2400+origin_tick+4)*.005,first_full_input_physical_end_s=(2400+origin_tick+4)*.005+.0005,release_entry_lift_amplitude_m=.1/.6*.02)
        record['trace_sha256']=sha(trace);records[jid]=record
    pair_checks=[]
    for stage,origin_tick in expected_origins.items():
        zero=(stage,'zero')
        if zero not in profiles:continue
        z=profiles[zero];zcommand=z['actual_command21'];unique=np.r_[True,np.diff(z['relative_control_tick'])!=0];step_cap=abs(np.diff(zcommand[unique],axis=0)).max(axis=0);tv_cap=abs(np.diff(zcommand[unique],axis=0)).sum(axis=0)
        check(stage+'/zero22ctrl_native_exact',np.array_equal(z['actual_ctrl'],z['native_carrier_ctrl']) and np.array_equal(zcommand,z['native_carrier_command21']))
        for label,reference in [('oldnative',ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz'),('frozen020zero',ROOT/'output/release_local_response_capsule_v2/runs/capsule__zero/record/trace.npz')]:
            equal,detail=same_trace(paths[zero],reference,physical);check(stage+'/zero_full16s40fields_public_'+label,equal,detail)
        for v in variants:
            pair=(stage,v)
            if pair not in profiles:continue
            p=profiles[pair];origin=(2400+origin_tick)*.005
            equal,detail=same_trace(paths[zero],paths[pair],physical,origin);check(stage+'/'+v+'/complete_prefix_including_origin',equal,detail)
            check(stage+'/'+v+'/actual_inputs_executed_zero_not_requested',np.array_equal(p['control_time'],z['control_time']) and np.array_equal(p['physical_time'],z['physical_time']) and np.array_equal(p['actual_phi_s']-z['actual_phi_s'],p['actual_phi_offset_s']) and np.array_equal(p['actual_command21'][:,18]-zcommand[:,18],p['actual_lift_offset_m']) and np.array_equal(p['native_carrier_command21'],zcommand))
            same_entry=True
            for name in ('release_entry','probe_entry'):
                x,y=entries[zero][name],entries[pair][name]
                same_entry &= x['physics'].keys()==y['physics'].keys() and all(np.array_equal(x['physics'][k],y['physics'][k],equal_nan=True) for k in x['physics']) and x['side']==y['side'] and x['local']==y['local']
            check(stage+'/'+v+'/two_full_entries_exact_to_stage_zero',same_entry)
            unique=np.r_[True,np.diff(p['relative_control_tick'])!=0];steps=abs(np.diff(p['actual_command21'][unique],axis=0))
            check(stage+'/'+v+'/21actual_step_cumulative_authority',np.all(steps<=step_cap+1e-10) and np.all(steps.sum(axis=0)<=tv_cap+1e-9))
            pair_checks.append(dict(stage=stage,variant=v,prefix_exact=equal,entry_exact=bool(same_entry)))
    if all((s,'zero') in paths for s in expected_origins):
        a,b=expected_origins;equal,detail=same_trace(paths[(a,'zero')],paths[(b,'zero')],physical);check('new_two_zeros_full16s_each_other',equal,detail)
    for record in records.values():
        if record['status']!='complete':continue
        zero=records[f"capsule__{record['stage']}__zero"]
        record['delta_from_same_stage_executed_zero']=dict(whole_peak_N=max(record['safety']['finger_peak_N'])-max(zero['safety']['finger_peak_N']),
            whole_local_excess_Ns=record['safety']['local_excess_integral_Ns']-zero['safety']['local_excess_integral_Ns'],
            physical_clearance_latency_s=record['release']['physical_clearance_latency_s']-zero['release']['physical_clearance_latency_s'],
            release_raw200Hz_target_TV_rate_N_s=record['release_raw_200Hz_target_TV_diagnostic']['finger_TV_rate_N_s']-zero['release_raw_200Hz_target_TV_diagnostic']['finger_TV_rate_N_s'])
    result=dict(schema='release_local_response_v3_terminal_independent_audit_v1',checks=checks,issues=issues,passed=not issues,source_checks=source_rows,
        records=records,pairing=pair_checks,input_sha256={n:sha(RUN/n) for n in ('summary.json','manifest.json','plan.json','results.json','assignment_ledger.json','source_snapshot_manifest.json')},
        raw_arithmetic_source_sha256=sha(PURE),checker_sha256=sha(Path(__file__)),physics_steps=0,model_admitted=False,material_safety_certified=False,
        scope='all14 complete assigned records/raw/V3, source244, within-origin paired entries/prefixes, two new zero full16s comparisons; no new dynamics or model fit',
        interpretation='Peak, excess, TV and delay are separate. Local metric improvements may be real without all metrics dominating; no new release controller or global model admitted.')
    with OUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# V3两个入口14分支：独立完整指标与配对核查','',f'检查{sum(c["passed"] for c in checks)}/{len(checks)}；问题{len(issues)}。无物理重跑、拟合或旧数据修改。','',
        '| origin | 变体/角色 | 峰值 N | 全程超限 N·s | 释放延迟 s | raw release TV N/s | 力限 |', '|---|---|---:|---:|---:|---:|---:|']
    for record in records.values():
        if record['status']!='complete':continue
        lines.append(f"| {record['stage']} | {record['variant']}/{record['training_role']} | {max(record['safety']['finger_peak_N']):.6f} | {record['safety']['local_excess_integral_Ns']:.8f} | {record['release']['physical_clearance_latency_s']:.3f} | {record['release_raw_200Hz_target_TV_diagnostic']['finger_TV_rate_N_s']:.5f} | {int(record['safety']['guard_pass'])} |")
    lines+=['','峰值和超限积分的降低不等于TV更小；释放TV包含有意卸载及清离，不能直接称持握更平稳。与各stage实际zero的逐项增减已保存在JSON，无需所有指标支配才承认一项局部收益，也不能用该收益抹掉安全/任务/时间代价。','',
        '每origin的zero+两个单轴为训练参考，四mixed为固定留出；.020旧七仅是冻结参照，不回填新数据。实际输入分母在两个origin都来自release12.000锁存高度及其原生下降时钟，不能把probe当前高度重新当速度定义。','',
        '新V3实际零分支和entry/prefix是否通过只按本次原始记录核算，不借纯QA充当物理结果。模型叠加误差及20/40/100/200ms窗口由另一个明确角色的review分析器报告。']
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=not issues,checks=len(checks),issues=issues,output=str(OUT),sha256=sha(OUT)),ensure_ascii=False))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()

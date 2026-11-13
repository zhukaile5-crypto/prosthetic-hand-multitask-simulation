"""Actual-trajectory robot-only actuation diagnosis; no new dynamics or policy.

The .5ms contact records are evaluation data only. Actuation algebra receives
only named robot q25/v25 and actual21 control-before references at200Hz.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION))
sys.dont_write_bytecode=True
EXPECTED='f0671211d2e15cc96eeb8c87207a43bc1839d55512ab18bdc2cebff4ed0a052c'
GROUPS={'TH':[0,1,2,3,4],'FF':[5,6,7],'MF':[8,9,10],'RF':[11,12,13],
        'LF':[14,15,16,17],'lift':[18],'WRJ2':[19],'WRJ1':[20]}
DIGITS=['TH','FF','MF','RF','LF']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def summarize(a,ids,step=.005):
    p=a['effort'][ids];e=a['error'][ids];strain=a['strain_proxy'][ids];power=a['power'][ids]
    return dict(samples=len(ids),first_s=float(a['time'][ids[0]]),last_s=float(a['time'][ids[-1]]),
        effort_signed_mean21=p.mean(0).tolist(),effort_abs_mean21=abs(p).mean(0).tolist(),
        effort_abs_max21=abs(p).max(0).tolist(),servo_error_rms21=np.sqrt((e*e).mean(0)).tolist(),
        servo_error_abs_max21=abs(e).max(0).tolist(),strain_proxy_mean21_J=strain.mean(0).tolist(),
        strain_proxy_max21_J=strain.max(0).tolist(),power_mean21_W=power.mean(0).tolist(),
        positive_model_power_integral21_J=(np.maximum(power,0).sum(0)*step).tolist(),
        negative_model_power_integral21_J=(np.minimum(power,0).sum(0)*step).tolist(),
        ctrl_clipped_sample_counts21=a['ctrl_clipped'][ids].sum(0).tolist(),
        force_clipped_sample_counts21=a['force_clipped'][ids].sum(0).tolist(),
        lift_without_known_robot_gravity_bias_strain_mean_J=float(a['lift_unbiased_proxy'][ids].mean()),
        groups={name:dict(effort_signed_mean=float(p[:,g].sum(1).mean()),
            effort_abs_mean=float(abs(p[:,g]).sum(1).mean()),
            strain_proxy_mean_J=float(strain[:,g].sum(1).mean()),
            strain_proxy_max_J=float(strain[:,g].sum(1).max()),
            power_mean_W=float(power[:,g].sum(1).mean()),
            effort_unit='N' if name=='lift' else 'Nm',
            error_unit='m' if name=='lift' else 'rad_transmission_coordinate') for name,g in GROUPS.items()})

def first_diff(t,a,b):
    different=np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b
    ids=np.flatnonzero(different)
    return None if not len(ids) else dict(time_s=float(t[ids[0]]),index=int(ids[0]),max_abs_difference=float(np.max(abs(a[ids[0]]-b[ids[0]]))))

def material_metrics(t,force,limits,budget,start,end,dt):
    mask=(t>=start-1e-9)&(t<end-1e-9)
    f=force[mask];local=np.maximum(f-limits,0);total=np.maximum(f.sum(1)-budget,0)
    return dict(local_excess_integral_N_s=float(local.sum()*dt),
        local_excess_per_digit_N_s=(local.sum(0)*dt).tolist(),budget_excess_integral_N_s=float(total.sum()*dt),
        max5_N=f.max(0).tolist(),over_limit_substeps5=(f>limits).sum(0).tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out)
    for p in [out,out.with_suffix('.md'),out.with_suffix('.npz')]:
        if p.exists():raise FileExistsError('Fresh derived evidence required:'+str(p))
    core=ROOT/'v11/robot_servo_effort.py'
    if sha(core)!=EXPECTED:raise RuntimeError('Frozen V11 identity changed')
    import mujoco
    from foundation_config import FoundationConfig
    from hand_plant import _model
    from v11.robot_servo_effort import RobotServoEffort,ACTUATORS21
    # Compile only a default known robot model. No actual external mass,
    # friction, pose, object motion or contact state enters this adapter.
    model,_=_model(FoundationConfig())
    servo=RobotServoEffort(model)
    if servo.unavailable:raise RuntimeError(servo.unavailable)
    cohort=ROOT/'output/development_v10_hold_v1'
    complete=load_json(cohort/'execution_complete.json')
    if not complete['execution_complete']:raise RuntimeError('Complete cohort required')
    checks=[]
    def check(name,flag):
        checks.append(dict(name=name,passed=bool(flag)))
        if not flag:raise AssertionError(name)
    for name in ['hand_plant.py','foundation_config.py']:
        check('frozen foundation robot source '+name,sha(FOUNDATION/name)==sha(cohort/'sources/foundation'/name))
    aids=servo.aids;gain=model.actuator_gainprm[aids,:3];bias=model.actuator_biasprm[aids,:3]
    check('actual fixed position servo kp contract',np.all(gain[:,1:]==0) and np.all(bias[:,0]==0) and np.array_equal(gain[:,0],-bias[:,1]))
    kp=gain[:,0];kv=-bias[:,2]
    forearm=model.body('rh_forearm').id
    robot_weight=float(model.body_subtreemass[forearm]*9.81)
    lift_bias=robot_weight/float(kp[18])
    protected={name:getattr(mujoco,name) for name in ['mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'] if hasattr(mujoco,name)}
    calls={name:0 for name in protected}
    def forbidden(name):
        def reject(*a,**k):calls[name]+=1;raise AssertionError('No simulation/FK/contact pipeline:'+name)
        return reject
    for name in protected:setattr(mujoco,name,forbidden(name))
    try:
        data={};inputs={};derived={};decisions=None
        for method in ['joint','ratio_preview']:
            base=cohort/'runs'/f'collision__{method}__610212';path=base/'trace.npz';meta_path=base/'metadata.json';result_path=base/'result.json'
            meta=load_json(meta_path)
            inputs[method]={str(p):sha(p) for p in [path,meta_path,result_path]}
            check(method+' saved trace identity',sha(path)==meta['trace_sha256'])
            check(method+' named robot axis whitelist',tuple(meta['raw_joint_names'])==servo.axes25[:24] and np.array_equal(meta['raw_joint_qpos_indices'],servo.qadr[:24]))
            with np.load(path,allow_pickle=False) as z:
                # Full archived columns may be decompressed by NumPy, but only
                # selected25 robot coordinates and selected21 ports survive.
                t=z['time'];q=z['qpos'][:,servo.qadr];v=z['qvel'][:,servo.dadr]
                u=z['ctrl_before'][:,servo.ctrladr];post=z['ctrl'][:,servo.ctrladr]
                normal=z['normal'];skin=z['skin_normal'];shear=z['shear'];palm=z['palm'];other=z['hand_other'];collision=z['collision']
                rt=z['substep_time'];rd=z['substep_duration'];rawskin=z['substep_skin_normal'];debug=z['debug_json']
            check(method+' fixed200Hz clock',len(t)==3201 and np.max(abs(np.diff(t)-.005))<1e-9)
            check(method+' no absent lift velocity',v.shape==(3201,25) and np.isfinite(v).all())
            # Only holding7.5<=t<12. All state/ref samples are pre-action at t.
            ids=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9));n=len(ids)
            a=dict(time=t[ids],q=q[ids],v=v[ids],actual_before21=u[ids],actual_post21=post[ids],
                normal=normal[ids],skin=skin[ids],shear=shear[ids],palm=palm[ids],other=other[ids],collision=collision[ids])
            for key in ['length','speed','error','elastic_drive','unbounded','effort','strain_proxy','power','ctrl_clipped','force_clipped','ctrl_kink','force_kink']:
                a[key]=np.empty((n,21),bool if 'clipped' in key or 'kink' in key else float)
            a['lift_unbiased_proxy']=np.empty(n)
            for j,i in enumerate(ids):
                s=servo.evaluate(q[i],v[i],u[i],time_s=float(t[i]))
                check_name=method+' all robot effort samples available'
                if s.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise RuntimeError((check_name,s.reasons))
                a['length'][j]=s.actuator_length21;a['speed'][j]=s.actuator_velocity21;a['error'][j]=s.servo_error21
                a['elastic_drive'][j]=kp*s.servo_error21;a['unbounded'][j]=s.unbounded_effort21;a['effort'][j]=s.predicted_effort21
                a['strain_proxy'][j]=.5*kp*s.servo_error21**2;a['power'][j]=s.predicted_effort21*s.actuator_velocity21
                a['lift_unbiased_proxy'][j]=.5*kp[18]*(s.servo_error21[18]-lift_bias)**2
                for key,source in [('ctrl_clipped','control_clipped21'),('force_clipped','force_clipped21'),('ctrl_kink','control_at_clip_kink21'),('force_kink','force_at_clip_kink21')]:a[key][j]=s.diagnostics[source]
                if not np.allclose(a['unbounded'][j],a['elastic_drive'][j]-kv*s.actuator_velocity21,rtol=1e-12,atol=1e-12):raise AssertionError('position/damping algebra')
            check(method+' all900 robot effort samples available',n==900)
            check(method+' actual gravity bias included at lift port',np.max(abs(u[ids,18]-(.1+lift_bias)))<.002)
            # Data shape is explicit; no actual q_object/v_object or obstacle port.
            check(method+' exact25plus21 inputs',a['q'].shape==(900,25) and a['actual_before21'].shape==(900,21))
            data[method]=a
            for k,value in a.items():derived[method+'__'+k]=value
            data[method]['full_summary']=summarize(a,np.arange(n))
            data[method]['raw_material']=material_metrics(rt,rawskin,np.asarray(meta['limits']),float(meta['budget']),7.5,12.,.0005)
            data[method]['source_end_hashes']={str(p):sha(p) for p in [path,meta_path,result_path]}
            check(method+' source inputs unchanged',inputs[method]==data[method]['source_end_hashes'])
            if method=='joint':
                decision_ids=[]
                for i in ids:
                    d=json.loads(str(debug[i]))
                    if d.get('status')=='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP' and abs((t[i]-7.5)/.02-round((t[i]-7.5)/.02))<1e-7:decision_ids.append(i)
                decisions=t[decision_ids]
                check('exact20 executed plans',len(decisions)==20)
        j=data['joint'];b=data['ratio_preview'];t=j['time']
        check('shared pre-action clock',np.array_equal(t,b['time']))
        before=t<9.54-1e-9
        check('first active prefix physical/ref exact',all(np.array_equal(j[k][before],b[k][before]) for k in ['q','v','actual_before21','actual_post21','skin']))
        first={k:first_diff(t,j[k],b[k]) for k in ['actual_post21','actual_before21','q','v','skin','effort','strain_proxy']}
        windows=[]
        for at in decisions:
            k=int(np.argmin(abs(t-at)));finish=k+4;window=np.arange(k,finish)
            record=dict(start_s=float(at),end_s=float(t[finish]),
                joint=summarize(j,window),ratio_preview=summarize(b,window),
                observed_skin_before5={m:data[m]['skin'][k].tolist() for m in data},
                observed_skin_after20ms5={m:data[m]['skin'][finish].tolist() for m in data},
                observed_skin_joint_minus_baseline_at20ms5=(j['skin'][finish]-b['skin'][finish]).tolist(),
                internally_computed_effort_joint_minus_baseline_at20ms21=(j['effort'][finish]-b['effort'][finish]).tolist(),
                strain_proxy_joint_minus_baseline_at20ms21_J=(j['strain_proxy'][finish]-b['strain_proxy'][finish]).tolist(),
                actual_before_command_joint_minus_baseline_at20ms21=(j['actual_before21'][finish]-b['actual_before21'][finish]).tolist(),
                actual_before_target_minus_length_decomposition21=dict(
                    delta_target=(j['actual_before21'][finish]-b['actual_before21'][finish]).tolist(),
                    delta_length=(j['length'][finish]-b['length'][finish]).tolist(),
                    delta_error=(j['error'][finish]-b['error'][finish]).tolist()),
                causal_effect_claim=False)
            windows.append(record)
        active_ids=np.unique(np.concatenate([np.arange(int(np.argmin(abs(t-x))),int(np.argmin(abs(t-x)))+4) for x in decisions]))
        post_ids=np.flatnonzero(t>=9.54-1e-9)
        sections={}
        for name,indices in [('full_holding_7p5_to12',np.arange(900)),('post_first_reference_divergence',post_ids),('twenty_plan_windows_union',active_ids)]:
            sections[name]={m:summarize(data[m],indices) for m in data}
        k=int(np.argmin(abs(t-9.54)))
        immediate={}
        for m in data:
            a=data[m];s=servo.evaluate(a['q'][k],a['v'][k],a['actual_post21'][k],time_s=float(t[k]))
            immediate[m]=dict(actual_post21=a['actual_post21'][k].tolist(),effort21=s.predicted_effort21.tolist(),
                strain_proxy21_J=(.5*kp*s.servo_error21**2).tolist())
        check('first difference exact same physical state',np.array_equal(j['q'][k],b['q'][k]) and np.array_equal(j['v'][k],b['v'][k]))
        check('no forbidden dynamics/FK/contact calls',not any(calls.values()))
        check('frozenV11 core unchanged',sha(core)==EXPECTED)
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out.with_suffix('.npz'),**derived)
        sources=dict(v11=dict(path=str(core),sha256=sha(core)),script=dict(path=str(Path(__file__)),sha256=sha(__file__)),
            foundation_robot_sources={name:sha(FOUNDATION/name) for name in ['hand_plant.py','foundation_config.py']})
        summary=dict(schema='v10_hold_collision_actual_servo_effort_mechanism_v1',passed=all(x['passed'] for x in checks),checks=checks,
            input_sha256=inputs,sources=sources,derived_timeseries_sha256=sha(out.with_suffix('.npz')),
            scope=dict(actual_robot_inputs_only=True,world_or_object_observation_inputs=False,actual_external_parameters_used=False,
                model_compile='default publicly known robot structure; no actual randomized external parameters supplied',
                sampling='200Hz pre-action robot q/v with ctrl_before; same time and exact actual trajectories, not hypothetical future ramp',
                hold_window='7.5<=t<12; release excluded',new_physics_steps=0,forbidden_pipeline_calls=calls,
                modeled_effort_is_not_measured_current_or_contact_force=True,force_gradient_available=False),
            units=dict(axes25=['rad']*24+['m'],actuator_error21=['rad']*18+['m','rad','rad'],
                actuator_effort21=['Nm']*18+['N','Nm','Nm'],strain_proxy='J dimensional proxy only, NOT actual stored energy',power='W model-computed instantaneous robot actuation power; NOT motor electrical or contact power'),
            actual_actuator_names21=list(ACTUATORS21),robot_axes25=list(servo.axes25),robot_qpos_addresses25=servo.qadr.tolist(),robot_qvel_addresses25=servo.dadr.tolist(),
            kp21=kp.tolist(),kv21=kv.tolist(),control_limits21=servo.ctrl_range.tolist(),force_limits21=servo.force_range.tolist(),
            known_robot_lift_gravity_bias_reference_m=lift_bias,known_robot_weight_N=robot_weight,
            proxy_definition='0.5*kp*(effective_actual_target-transmission_length)^2; unlimited position-servo strain proxy. Actual effort additionally has damping and force clamp. Lift proxy also reported after removing known robot gravity-compensation command bias.',
            first_differences=first,first_actual_command_same_state=immediate,sections=sections,
            raw_holding_material={m:data[m]['raw_material'] for m in data},plan_windows=windows,
            interpretation='Paired actual trajectories share the pre-divergence state. Later effort/pressure associations are descriptive closed-loop consequences, not an identified reference-to-contact-force gradient. Fixed tendon/Jacobian and robot internal stiffness do not determine external ownership, geometry/compliance or force distribution.')
        with out.open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
        lines=['# V10持握碰撞：实际伺服顺应性特征核查','',
            '仅比较同seed610212的joint与ratio_preview完整实际轨迹，持握窗7.5≤t<12。机器人24角编码器+升降轴、速度及实际21控制前端口参考进入f067模型；不输入物体状态/真实质量/摩擦。无新动力学、无接触求解。','',
            '内部驱动力是已知机器人模型计算，不是新增力矩传感器读数。0.5kp·error²仅为未限幅伺服应变代理，不是实际储能；力限幅、速度阻尼与升降重力偏置另列。K/D不作为接触力梯度。','',
            f'首个实际参考分歧：{first["actual_post21"]}；控制前参考/状态分歧：{first["actual_before21"]} / {first["q"]}。',
            '', '|分组|全持握joint应变代理均值(J)|基线(J)|干预窗口joint(J)|基线(J)|','|---|---:|---:|---:|---:|']
        for name in GROUPS:
            full=sections['full_holding_7p5_to12'];win=sections['twenty_plan_windows_union']
            lines.append(f'|{name}|{full["joint"]["groups"][name]["strain_proxy_mean_J"]:.9g}|{full["ratio_preview"]["groups"][name]["strain_proxy_mean_J"]:.9g}|{win["joint"]["groups"][name]["strain_proxy_mean_J"]:.9g}|{win["ratio_preview"]["groups"][name]["strain_proxy_mean_J"]:.9g}|')
        lines+=['',f'完整检查{len(checks)}项通过。各实际20ms窗口、逐执行器有符号驱动力/误差/限幅/功率、力超限分指积分及全部来源指纹见JSON；200Hz派生时间序列见同名NPZ。',
            '', '因果边界：同一次接触既改变编码器误差，也改变接触压力，因此二者相关不能反推压力梯度。局部回撤几何满足不意味着压力量值下降；该日志只判断顺应性特征是否存在，不能单独证实下一控制器可安全预测接触力。']
        with out.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
        print(json.dumps(dict(passed=summary['passed'],checks=len(checks),out=str(out),sha256=sha(out),first_differences=first),ensure_ascii=False))
    finally:
        for name,value in protected.items():setattr(mujoco,name,value)

if __name__=='__main__':main()

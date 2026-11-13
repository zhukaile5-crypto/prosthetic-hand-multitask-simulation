"""One preselected public snapshot; timing excludes archive/model setup."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(F));sys.dont_write_bytecode=True
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    module=ROOT/'v13_joint/native_effort21.py';before=sha(module)
    path=ROOT/'output/development_v12_native_pilot_v1/runs/nominal__fusion_effort__617901/trace.npz'
    with np.load(path,allow_pickle=False) as z:
        t=z['time'];observations=z['observation_json']
    i=int(np.argmin(abs(t-8.5)));frame=json.loads(str(observations[i]));del observations
    import research_imports,mujoco,hand_plant
    forbidden=[]
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('Forbidden '+_name)
        setattr(mujoco,name,deny)
    hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No plant construction'))
    from foundation_config import FoundationConfig
    from v13_a1.codec import EnginePort21
    from v13_joint.native_effort21 import NativeEffort21
    model,_=hand_plant._model(FoundationConfig());core=NativeEffort21(model)
    q=np.r_[frame['raw_joint'],frame['lift']];v=np.r_[frame['raw_joint_velocity'],frame['lift_velocity_m_s']]
    actual=EnginePort21(frame['robot_actuator_target21'],float(t[i]),core.actor.lift_bias_m);ref=core.actor.decode(actual)
    rate=np.r_[np.full(18,1.2),.035,.12,.12];step=rate*.020
    calls=[0];native=core.servo.evaluate
    def counted(*a,**k):calls[0]+=1;return native(*a,**k)
    core.servo.evaluate=counted
    start=time.perf_counter_ns()
    d=core.build_domain(q25=q,v25=v,robot_axes25=core.robot_axes25,time_s=float(t[i]),current_reference=ref,
        nominal_reference=ref,actual_engine=actual,step_lower21=-step,step_upper21=step)
    build_ms=(time.perf_counter_ns()-start)/1e6;build_calls=calls[0]
    start=time.perf_counter_ns()
    result=core.solve(d,p0_effort21=d.p0_effort21,delta_effort21=np.zeros(21),correction25=np.zeros(25),
        residual_scale25=np.ones(25),delta_scale21=np.ones(21))
    solve_ms=(time.perf_counter_ns()-start)/1e6
    report=dict(schema='v13_native_effort21_single_public_snapshot_profile_v1',source_sha256=before,
        module_unchanged=sha(module)==before,snapshot=dict(source=str(path),trace_sha256=sha(path),time_s=float(t[i]),index=i,
            scenario='nominal',method='fusion_effort',seed=617901,motion='hold'),
        domain_available=d.available,domain_status=d.status,domain_reason=d.reason,proposal_available=result.available,status=result.status,reason=result.reason,
        build_domain_ms=build_ms,solve_ms=solve_ms,build_native_evaluation_calls=build_calls,
        solve_native_evaluation_calls=calls[0]-build_calls,reported_solve_native_evaluation_calls=result.diagnostics.get('native_evaluations'),
        candidate_count=sum(len(s['candidates']) for s in result.diagnostics.get('finite_candidate_solutions',[])),
        per_axis_candidate_counts=[len(s['candidates']) for s in result.diagnostics.get('finite_candidate_solutions',[])],
        input_scales='All-one positive multipliers, exactly the QP scale convention',
        workload='One fixed actual public q/v/current engine snapshot, nominal=current, d request0/correction0, software20ms reference envelope; no QP or policy executed',
        setup_excluded='Archive IO, source hashing, model compilation, imports and construction excluded from measured build/solve',
        timing_scope='One cold-inverse call on one host; no repeat-average, hardware real-time conclusion or policy throughput claim',
        forbidden_calls=forbidden,writer_calls=core.actor.write_count,physics_steps=0)
    out=ROOT/'review/v13_native_effort21_public_profile_v1.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(output=str(out),sha256=sha(out),build_ms=build_ms,solve_ms=solve_ms,domain_available=d.available,proposal_available=result.available)))
if __name__=='__main__':main()

"""Independent development-only read-only runtime audit; no formal trials."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import mujoco
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import controllers as c
from evaluation import evaluate_trial

def digest():
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
            for f in ['controllers.py','physics.py','evaluation.py','run_experiments.py','scenarios.py','settings.json']}

def frame(t,n,valid=None,velocity=0.):
    return SimpleNamespace(time=t,dt=.005,normal=np.array(n,float),
        tangential=np.tile([0.,.20],(5,1)),slip_local=np.tile([velocity,0.],(5,1)),
        contact_pos=np.array([[-.025,0,0],[.025,-.024,0],[.025,-.008,0],[.025,.008,0],[.025,.024,0]]),
        contact_normal=np.array([[1.,0,0]]+[[-1.,0,0]]*4),
        tangent_basis=np.tile([[0.,1.,0],[0.,0.,1.]],(5,1,1)),
        valid=np.ones(5,bool) if valid is None else np.asarray(valid,bool),age=0.,
        joint=np.full(5,.55),joint_velocity=np.zeros(5),alpha=np.full(5,.55),
        support_axis=np.array([0.,0.,1.]),phase='HOLDING',closure_permission=np.ones(5,bool))

def qp_checks():
    rng=np.random.default_rng(7238); cases=[]; real=c.solve_qp
    for name in ['ratio','unified','unified_no_continuity','unified_no_geometry','unified_fixed_stage']:
        for k in range(5):
            limits=rng.uniform(1.3,5.,5);normal=rng.uniform(.2,.7,5)*(limits-.12)
            budget=float(normal.sum()+rng.uniform(.5,2.))
            controller=c.make_controller(name,limits,budget)
            for j in range(60):controller.state.update(frame(j*.005,normal),.005)
            captured={}
            def capture(Q,q,A,b,passes,tolerance,**kwargs):
                captured.update(Q=Q.copy(),q=q.copy(),A=A.copy(),b=b.copy(),kwargs=kwargs)
                return real(Q,q,A,b,passes,tolerance,**kwargs)
            c.solve_qp=capture
            try: controller._decide(frame(1.,normal),'HOLDING')
            finally:c.solve_qp=real
            Q,q,A,b=[captured[z] for z in ['Q','q','A','b']]
            kwargs=captured.get('kwargs',{})
            x,r,it=real(Q,q,A,b,20,2e-5,**kwargs)
            xx,rr,iit=real(Q,q,A,b,5000,1e-9,**kwargs)
            fun=lambda z:float(.5*z@Q@z+q@z)
            cases.append(dict(method=name,k=k,normal=normal.tolist(),limits=limits.tolist(),budget=budget,
                residual_20=r,residual_5000=rr,iterations_20=it,iterations_5000=iit,
                objective_20=fun(x),objective_5000=fun(xx),
                relative_objective_gap=(fun(x)-fun(xx))/max(1.,abs(fun(xx))),
                first_action_max_difference=float(np.max(abs(x[:5]-xx[:5]))),
                Q_condition=float(np.linalg.cond(Q))))
    return cases

def force_and_velocity():
    xml='<mujoco><option gravity="0 0 -9.81" timestep=".0005"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body name="ball" pos="0 0 .095"><freejoint/><geom name="sphere" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'sphere');forces=[]
    for i in range(d.ncon):
        ct=d.contact[i];f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        sign=1. if ct.geom2==obj else -1.
        world=sign*ct.frame.reshape(3,3).T@f[:3]
        forces.append(dict(geom1=int(ct.geom1),geom2=int(ct.geom2),normal=f[0],on_object_world=world.tolist()))
    xml='<mujoco><option gravity="0 0 0"/><worldbody><body name="body" pos=".2 .3 .4"><freejoint/><inertial pos=".10 .05 .02" mass="1" diaginertia=".03 .04 .05"/><geom type="box" size=".2 .1 .1"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    d.qvel[:]=[.3,-.2,.1,.6,.4,-.2];mujoco.mj_forward(m,d)
    v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,1,v,0)
    local=np.array([.05,-.03,.01]);point=d.xpos[1]+d.xmat[1].reshape(3,3)@local
    predicted=v[3:]+np.cross(v[:3],point-d.xipos[1])
    eps=1e-7;mujoco.mj_integratePos(m,d.qpos,d.qvel,eps);mujoco.mj_forward(m,d)
    point2=d.xpos[1]+d.xmat[1].reshape(3,3)@local
    actual=(point2-point)/eps
    return dict(contact_force_cases=forces,force_on_sphere_is_upward=all(x['on_object_world'][2]>0 for x in forces),
        com_point_velocity_predicted=predicted.tolist(),finite_difference_velocity=actual.tolist(),
        point_velocity_error=float(np.linalg.norm(predicted-actual)))

def synthetic_task():
    t=np.arange(0.,10.0001,.005);n=len(t);force=np.tile([2.,2.,2.,0.,0.],(n,1));force[t>=1.]=0.
    trace=dict(time=t,normal=force,position=np.tile([0.,0.,1.],(n,1)),relative=np.zeros((n,3)),
        rotation=np.tile(np.eye(3),(n,1,1)),relative_velocity=np.zeros((n,3)),angular_velocity=np.zeros((n,3)),
        support=np.zeros(n),palm=np.where(t>=1.,10.,0.))
    meta=dict(scenario=dict(name='audit_synthetic',duration=10.,limit=[5.]*5,budget=22.),
        reference=dict(relative=[0.,0.,0.],rotation=np.eye(3).tolist(),absolute_z=1.,orientation_mode='full'),
        command_at=0.,failure_at=None,method='synthetic',seed=None)
    result=evaluate_trial(trace,meta,dict(minimum_contacts=3,required_fingers=[0]))
    keys=['task_success','end_reason','established_at','first_stable_time','stable_hold_fraction','static_window']
    gap=trace.copy();keep=(t<2.)|(t>3.)
    gap={key:val[keep] for key,val in gap.items()};gap['normal'][:]=[2.,2.,2.,0.,0.];gap['palm'][:]=0.
    gr=evaluate_trial(gap,meta,dict(minimum_contacts=3,required_fingers=[0]))
    tr=trace.copy();tr['normal']=np.tile([2.,2.,2.,0.,0.],(n,1));tr['palm']=np.zeros(n)
    tr.update(substep_time=np.arange(2000)*.0005,substep_duration=np.full(2000,.0005),
        substep_normal=np.tile([2.,2.,2.,0.,0.],(2000,1)))
    sr=evaluate_trial(tr,meta,dict(minimum_contacts=3,required_fingers=[0]))
    return dict(holding_contact_loss={key:result[key] for key in keys},
        truth_gap=dict(task_success=gr['task_success'],truth_record_complete=gr['truth_record_complete']),
        truncated_substeps=sr['force_safety'])

if __name__=='__main__':
    before=digest();result=dict(scope='development-only synthetic and isolated primitive checks; no formal trials',
        source_sha256=before,contact_convention=force_and_velocity(),qp_cases=qp_checks(),
        task_completion_contract=synthetic_task())
    result['source_unchanged']=before==digest()
    (ROOT/'review'/'independent_preflight_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(source_unchanged=result['source_unchanged'],contact_convention=result['contact_convention'],
        synthetic_success=result['task_completion_contract']['holding_contact_loss']['task_success'],
        qp_infeasible_20=sum(z['residual_20']>1e-3 for z in result['qp_cases']),
        qp_infeasible_5000=sum(z['residual_5000']>1e-3 for z in result['qp_cases']),
        largest_feasible_objective_gap=max((z['relative_objective_gap'] for z in result['qp_cases'] if z['residual_20']<=1e-3 and z['residual_5000']<=1e-3),default=None)),ensure_ascii=False))

"""Eight independent analytical scopes; fake known robot only, no physics.

Reviewer did not implement governor.py. Reviewer authored the separate effort
QP component; this is cross-component review, not independent whole-system QA.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import copy,hashlib,importlib.util,json,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SOURCE=ROOT/'v18_effort_transfer/governor.py'
PIN='ed03151f974a81d24e24ce8b8d7378e5d7c775f9b14e851e5bd832a1f11d4782'
STEM=ROOT/'review/v18_governor_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def main():
    if any(STEM.with_suffix(s).exists() for s in ('.json','.md')):raise FileExistsError(STEM)
    if sha(SOURCE)!=PIN:raise ValueError('Final governor source pin changed')
    import mujoco
    from v18_effort_transfer.governor import LiftProgressGovernor
    forbidden=[];checks=[];scopes=[]
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden '+name)
        return f
    class Model:
        def __init__(self):
            self.jnt_dofadr=np.array([0]);self.jnt_type=np.array([mujoco.mjtJoint.mjJNT_SLIDE])
            self.actuator_trntype=np.array([mujoco.mjtTrn.mjTRN_JOINT]);self.actuator_trnid=np.array([[0,-1]])
            self.actuator_gear=np.array([[1.,0.,0.,0.,0.,0.]])
            self.actuator_dyntype=np.array([mujoco.mjtDyn.mjDYN_NONE]);self.actuator_actnum=np.array([0]);self.actuator_delay=np.array([0.])
            self.actuator_ctrlnum=np.array([1]);self.actuator_outnum=np.array([1]);self.actuator_plugin=np.array([-1])
            self.actuator_gaintype=np.array([mujoco.mjtGain.mjGAIN_FIXED]);self.actuator_biastype=np.array([mujoco.mjtBias.mjBIAS_AFFINE])
            self.actuator_gainprm=np.array([[1500.,0.,0.]])
            self.actuator_biasprm=np.array([[0.,-1500.,-9.]])
            self.dof_damping=np.array([6.]);self.body_subtreemass=np.array([1.])
            self.actuator_ctrladr=np.array([0]);self.actuator_outadr=np.array([0]);self.actuator_group=np.array([0])
            self.actuator_ctrllimited=np.array([True]);self.actuator_ctrlrange=np.array([[-.2,.2]])
            self.actuator_forcelimited=np.array([True]);self.actuator_forcerange=np.array([[-100.,100.]])
            self.jnt_actfrclimited=np.array([False]);self.jnt_actgravcomp=np.array([False])
            self.opt=SimpleNamespace(disableflags=0,disableactuator=0,enableflags=0)
        def actuator(self,name):
            if name!='lift_motor':raise KeyError(name)
            return SimpleNamespace(id=0)
        def joint(self,name):
            if name!='transport_lift':raise KeyError(name)
            return SimpleNamespace(id=0)
        def body(self,name):
            if name!='rh_forearm':raise KeyError(name)
            return SimpleNamespace(id=0)
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def call(g,**changes):
        values=dict(time_s=12.,q_m=.05,v_m_s=0.,current_reference_m=.05,
            reference_lower_m=0.,reference_upper_m=.1,reference_rate_lower_m_s=-.035,
            reference_rate_upper_m_s=.035,requested_velocity_m_s=-.03,dt_s=.005,
            brake_requested=False,deadline_s=None)
        values.update(changes);return g.step(**values)
    def check_interval(r,lo,hi,u):
        ck('actual global/rate intersection',r.available and lo<=r.reference_m<=hi and r.increment_m==r.reference_m-u)
    def run(name,fn):
        first=len(checks)
        try:fn()
        except Exception as exc:ck(name+'/exception',False,dict(error=repr(exc),traceback=traceback.format_exc()))
        scopes.append(dict(name=name,passed=all(c['passed'] for c in checks[first:]),check_count=len(checks)-first))
    with ExitStack() as guards:
        for name in ('MjModel','MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce','mj_tendon','mj_transmission'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        def ordinary():
            g=LiftProgressGovernor(Model());r=call(g)
            ck('tau=(9+6)/1500 exactly .01',g.tau_s==.01)
            ck('velocity target .0497 rate projects to .049825',r.available and abs(r.reference_m-.049825)<2e-17 and abs(r.diagnostics['unprojected_velocity_reference_m']-.0497)<2e-17)
            check_interval(r,.05-.035*.005,.05+.035*.005,.05)
            ck('no actual stopping/prediction certificate',not r.diagnostics['stopping_time_or_distance_certified'] and not r.diagnostics['future_robot_response_predicted'])
        run('ordinary debt-free analytic velocity target',ordinary)
        def debt():
            g=LiftProgressGovernor(Model());r=call(g,current_reference_m=.048)
            ck('existing debt .002 exceeds .0007 budget',r.diagnostics['current_debt_exceeds_budget'] and abs(r.diagnostics['recovery_reference_budget_m']-.0007)<1e-18)
            ck('down request becomes q recovery and increases ref',r.diagnostics['governed_velocity_intent_m_s']==0 and abs(r.reference_m-.048175)<2e-17)
            ck('old overbudget not falsely signed reachable',not r.diagnostics['current_q_brake_reference_reachable_within_budget'] and r.diagnostics['minimum_reference_return_time_at_current_q_s']>.02)
            r=call(g,current_reference_m=.052,requested_velocity_m_s=.03)
            ck('opposite upward debt recovers by lowering ref',r.diagnostics['current_upward_debt_exceeds_budget'] and abs(r.reference_m-.051825)<2e-17)
        run('both existing reference debt directions',debt)
        def overspeed():
            g=LiftProgressGovernor(Model());r=call(g,v_m_s=-.2)
            ck('measured descent blocks more descent without fake stop',r.diagnostics['measured_descent_exceeds_envelope'] and r.reference_m==.05 and r.diagnostics['actual_velocity_m_s']==-.2)
            r=call(g,v_m_s=-.2,requested_velocity_m_s=.03)
            ck('positive retreat preserved during descent overspeed',r.reference_m>.05 and r.diagnostics['governed_velocity_intent_m_s']>0)
            r=call(g,v_m_s=.2,requested_velocity_m_s=.03)
            ck('ascent overspeed blocks more upward progress',r.diagnostics['measured_ascent_exceeds_envelope'] and r.reference_m==.05)
            r=call(g,v_m_s=.2,requested_velocity_m_s=-.03)
            ck('downward unload preserved during ascent overspeed',r.reference_m<.05)
        run('actual overspeed and opposite recovery',overspeed)
        def authority():
            g=LiftProgressGovernor(Model());r=call(g,reference_upper_m=.0501,requested_velocity_m_s=.03)
            ck('hard ceiling dominates velocity intention',r.reference_m==.0501);check_interval(r,.049825,.0501,.05)
            r=call(g,q_m=.051,reference_rate_lower_m_s=0.,reference_rate_upper_m_s=0.)
            ck('zero authority legal hold is not recoverable',r.available and r.reference_m==.05 and r.diagnostics['minimum_reference_return_time_at_current_q_s'] is None and not r.diagnostics['current_q_brake_reference_reachable_within_budget'])
            r=call(g,reference_lower_m=.051)
            ck('empty genuine rate/global intersection refused',not r.available and r.reference_m is None and r.status=='UNAVAILABLE_EMPTY_ACTUAL_AUTHORITY')
        run('actual authority ceiling zero and empty set',authority)
        def admission():
            model=Model();model.actuator_gainprm[0,0]=1400.;model.actuator_biasprm[0,1]=-1400.
            r=call(LiftProgressGovernor(model));ck('nonstandard kp cannot borrow1500 actor bias',not r.available and r.status=='UNAVAILABLE_ROBOT_SERVO')
            model=Model();model.actuator_gear[0,0]=-1.;ck('negative unit transmission not silently normalized',not call(LiftProgressGovernor(model)).available)
            model=Model();g=LiftProgressGovernor(model);model.actuator_group[0]=1
            ck('routing changed after binding refused',call(g).status=='UNAVAILABLE_MODEL_CHANGED')
        run('nonstandard servo and model mutation',admission)
        def clocks():
            g=LiftProgressGovernor(Model());a=call(g,time_s=12.1);b=call(g,time_s=12.1)
            ck('identical duplicate deterministic exact',json.dumps(plain(a.__dict__),sort_keys=True)==json.dumps(plain(b.__dict__),sort_keys=True))
            ck('stateless offline earlier query permitted not runtime authentication',call(g,time_s=12.).available)
            for dt in (0.,-.005,.02,True,'0.005'):
                r=call(g,dt_s=dt);ck('invalid dt rejected '+repr(dt),not r.available and r.reference_m is None)
            ck('negative clock rejected',not call(g,time_s=-.005).available)
        run('strict execution dt and stateless temporal boundary',clocks)
        def deadline():
            g=LiftProgressGovernor(Model());r=call(g,time_s=12.495,deadline_s=12.5)
            ck('before deadline downward request still available',r.reference_m<.05 and not r.diagnostics['deadline_exceeded'])
            r=call(g,time_s=12.5,deadline_s=12.5)
            ck('at deadline q brake and supervisor exit required',r.reference_m==.05 and r.diagnostics['supervisor_exit_required'] and r.status=='BRAKE_RECOVER')
            r=call(g,requested_velocity_m_s=.03,brake_requested=True)
            ck('explicit brake overrides positive request to q',r.reference_m==.05 and r.diagnostics['unprojected_velocity_reference_m']==.05)
        run('deadline and explicit braking',deadline)
        def native():
            model=Model();model.actuator_forcerange[0]=[-5.,5.];g=LiftProgressGovernor(model);r=call(g)
            d=r.diagnostics['current_native_effort_of_emitted_reference'];raw=1500.*(r.reference_m+9.81/1500.)-1500.*.05
            ck('engine bias exactly once native raw order',d['engine_target_m']==r.reference_m+9.81/1500. and d['raw_effort_N']==raw)
            ck('known force clip reported not pseudo inverse',d['force_clipped'] and d['effort_N']==5. and r.available)
            ck('reference does not include engine gravity offset',abs(r.reference_m-.049825)<2e-17 and not r.diagnostics['actor_gravity_bias_applied_here'])
        run('known native clipping and unbiased reference',native)
    ck('no forbidden model/pipeline calls',not forbidden,forbidden)
    ck('final source pin unchanged',sha(SOURCE)==PIN)
    report=dict(schema='v18_governor_independent_v1',role='cross_component_independent_governor_review; reviewer did not implement governor, authored separate effort_task',
        passed=all(c['passed'] for c in checks),source_sha256=PIN,checker_sha256=sha(__file__),checks=checks,scopes=scopes,forbidden_calls=forbidden,
        model_constructions=0,physics_steps=0,scope_count=8,clock_boundary='Governor is stateless; rewind/different duplicate payload/public qvu authority belong to supervisor. Strict dt=.005 is enforced here.',
        limitations=['Fake known robot, analytical arrays only; no F067/plant/native model evaluation or physical braking validation.','Recovery budget is return of reference toward CURRENT q; no future stopping bound/certificate.','Known force clipping reports modeled effort and does not certify authority to change pressure or actual velocity.'])
    STEM.with_suffix('.json').write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    STEM.with_suffix('.md').write_text(f"# V18升降进度组件独立解析门\n\n{'PASS' if report['passed'] else 'FAIL'}；8个scope，{sum(c['passed'] for c in checks)}/{len(checks)}项。审者未参与governor实现，是另一个effort_task组件作者的交叉独审。绑定源码 `{PIN}`。\n\n覆盖q−u正反债务、实际正反超速与相反恢复、真实reference/rate交集、零权限及空域、非标准servo/路由变更、严格5ms及重复幂等/离线回查边界、deadline和显式q制动、native重力偏置一次与已知forceclip。\n\n所有模型均独立fake数组；禁止Model/Data/积分/forward/FK/接触/传动入口，调用0。governor无历史clock身份认证，rewind等运行资格由supervisor负责，不能把本门称运行层时钟通过。恢复预算亦非实际停速/停距证书。\n\n[完整证据]({STEM.name}.json)。\n",encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],scope_count=8,checks=len(checks),failed=[c for c in checks if not c['passed']],sha256=sha(STEM.with_suffix('.json')),md_sha256=sha(STEM.with_suffix('.md'))),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

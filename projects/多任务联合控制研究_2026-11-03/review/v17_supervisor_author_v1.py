"""Lifecycle-only independent fixtures; kernel algebra is separately reviewed.

No simulator model, sensors, integration or controller plant is constructed.
The root kernel call is replaced by a declared finite normalized proposal only
to test dispatch, domains, scaling, receipts and phase semantics.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
forbidden=[]
def no(*a,**kw):forbidden.append('forbidden');raise RuntimeError('no physics/model/writer')
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
    if hasattr(mujoco,name):setattr(mujoco,name,no)
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry
from v15_support.receiver_sensor import ReceiverMeasurement
from v16_release_geometry.transfer_observer import TransferObserver,HandSupportSample,PreReleaseLoadPrior
import v17_contact_transfer.supervisor as module


def main(out):
    checks=[]
    def ck(name,value,details=None):checks.append(dict(name=name,passed=bool(value),details=details))
    def rejects(name,fn):
        try:fn();ck(name,False)
        except (ValueError,TypeError):ck(name,True)
    start=np.r_[np.full(18,.4),.1,.02,-.01];goal=np.r_[[-1.,.1,0.,0.,0.],np.zeros(9),.1,np.zeros(3)]
    lower=np.r_[np.full(18,-2.),0.,-.5,-.5];upper=np.r_[np.full(18,2.),.1,.5,.5]
    def graph(lift=.1):
        s=start.copy();s[18]=lift
        entry=ReleaseEntry(1.,Reference21(s,1.),1)
        observer=TransferObserver(request_time_s=1.,receiver_half_size_xy_m=[.2,.2],receiver_up_axis3=[0,0,1],prior=PreReleaseLoadPrior(1.,.1,.995))
        return module.ReleaseSupervisor(entry,open18=goal,observer=observer),s
    def inputs(t,*,pressure=1.,R=1.,receiver_valid=True,hand_valid=True,aux_valid=True):
        acq=t-.01
        f=SimpleNamespace(time=t,dt=.005,normal=np.full(5,pressure),valid=np.full(5,hand_valid,bool),age=.01,sample_time=acq,
            auxiliary_pressure=0.,auxiliary_valid=aux_valid,auxiliary_age=.01)
        hand=HandSupportSample(t,acq,hand_valid,.8 if hand_valid else None,.1 if hand_valid else None,None if hand_valid else 'fixture_unknown')
        receiver=ReceiverMeasurement(t,acq,.01,receiver_valid,np.array([0,0,-R,0,0,0]) if receiver_valid else None,
            max(0.,R) if receiver_valid else None,np.full(6,.01),np.full(6,.03),np.array([0.,0.]) if receiver_valid and R>.03 else None,
            'boundary_after_forward',None if receiver_valid else 'random_dropout')
        return f,dict(time_s=t,hand=hand,receiver=receiver,hand_receiver_confounded=None,geometry_acquisition_time_s=None)
    def observe(g,t,cur,**kwargs):
        f,ti=inputs(t,**kwargs);state=g.observe(frame=f,transfer_inputs=ti,current21=cur);return f,state
    class Adapter:
        def __init__(self,motion=None):self.motion_task=motion;self.calls=[]
        def current_geometry_tasks(self,frame,current,desired,lo,hi):
            self.calls.append((np.array(lo),np.array(hi)))
            return SimpleNamespace(status='EXPERIMENTAL_GEOMETRY_TASKS',diagnostics=dict(pressure_prediction_available=False),normal_matrix_m_per_reference=np.zeros((1,21)),tangent_matrix_m_per_reference=np.zeros((3,21)),step_lower21=np.array(lo),step_upper21=np.array(hi))
    kernel=[]
    def solved(tasks,*,reference_step_scale21,motion_task):
        x=np.zeros(21);x[0]=.1;x[18]=.1
        x=np.minimum(tasks.step_upper21/reference_step_scale21,np.maximum(tasks.step_lower21/reference_step_scale21,x))
        kernel.append(dict(scale=reference_step_scale21.copy(),motion=motion_task))
        return SimpleNamespace(available=True,x=x,status='fixture_finite_proposal',levels=(),diagnostics=dict(normal_work_role='geometric proposal, not force safety'))
    original=module._contact_solve;module._contact_solve=solved
    try:
        g,cur=graph();adapter=Adapter()
        f,state=observe(g,1.,cur,R=1.)
        r,d=g.action(frame=f,adapter=adapter,current21=cur,lower21=lower,upper21=upper,generation=1)
        ck('pre-request receiver data cannot trigger contact',g.phase=='APPROACH_SUPPORT')
        ck('approach holds18 and wrists exactly, lift descends actual5ms rate',np.array_equal(r.values[:18],cur[:18]) and np.array_equal(r.values[19:],cur[19:]) and abs(r.values[18]-(cur[18]-.18*.005))<1e-15)
        ck('approach bypasses geometry/kernel and keeps holding force inactive',not kernel and not adapter.calls and not state['holding_force_tasks_active'])
        observe(g,1.005,cur,R=1.)
        f,state=observe(g,1.01,cur,R=1.)
        ck('first fresh postrequest receiver sample not sufficient',g.phase=='APPROACH_SUPPORT')
        f,state=observe(g,1.015,cur,R=1.)
        ck('twofresh anyactor receiver candidates enable contact, not placed',g.phase=='CONTACT_UNLOADING' and state['first_receiver_event_latched'] and not state['placed'])
        ck('unknown attribution is not relabelled clear',state['transfer']['stage_permission']=='OBSERVATION_ONLY')
        r,d=g.action(frame=f,adapter=adapter,current21=cur,lower21=lower,upper21=upper,generation=1)
        ck('CONTACT kernel runs despite lack of transfer certification',len(kernel)==1 and d['geometry_proposal_accepted'])
        ck('motion unavailable passed as None, never zero measurement',kernel[-1]['motion'] is None and g.decision_evidence[-1]['motion_task_role']=='unknown_not_a_zero_motion_measurement')
        lo,hi=adapter.calls[-1]
        ck('CONTACT no artificial thumb OPEN cone and allows positive lift within global',hi[0]>0 and hi[18]>=0 and lo[18]<0)
        ck('neutral lift interval does not freeze or onlyallow descent',g.lift_interval(Reference21(cur,f.time))[1]>cur[18])
        ck('normalized output restored once then firstquarter issued',abs(r.values[0]-cur[0]-.1*3*.02*.25)<1e-15 and abs(r.values[18]-cur[18])<1e-15)
        # Positive lift from the test proposal is clipped only by actual global
        # upper=.1, not a hidden stage lock; repeat below from lift=.05.
        ccur=cur.copy();ccur[18]=.05
        f,_=observe(g,1.02,ccur,R=1.)
        before=len(kernel)
        r,_=g.action(frame=f,adapter=adapter,current21=ccur,lower21=lower,upper21=upper,generation=2)
        ck('mechanical generation forces a newdecision not relabelled oldplan',len(kernel)==before+1 and g.plan_generation==2)
        ck('positive CONTACT lift proposal truly can execute',r.values[18]>ccur[18])
        receipt=SimpleNamespace(execution_changed=True,stale_plan_replaced=False,mode_generation=2,executed=Reference21(r.values,f.time))
        g.reconcile(receipt);ck('changed actualreceipt invalidates unused20ms plan',g.last_decision==-np.inf)
        # R becomes missing but valid low hand-pressure dwell continues.
        for j in range(45):f,state=observe(g,1.025+j*.005,ccur,R=0.,receiver_valid=False,pressure=.01)
        ck('hand low dwell independent of receiver dropout',g.was_clear and g.phase=='FREE_CLEAR_AND_RETREAT' and state['low_hand_pressure_confirmed'])
        before=len(kernel);r,d=g.action(frame=f,adapter=adapter,current21=ccur,lower21=lower,upper21=upper,generation=2)
        ck('freeclear direct legalclearance bypasses hierarchy certificate',len(kernel)==before and not d['geometry_proposal_accepted'] and d['geometry_status']=='NOT_REQUESTED_DIRECT_STAGE')
        ck('freeclear directstep preserves realbounds/rates/currentclock',np.all(r.values>=lower)&np.all(r.values<=upper)&np.all(abs(r.values-ccur)<=np.r_[np.full(18,3.),.18,.12,.12]*.005+1e-15) and r.time_s==f.time)
        f,state=observe(g,f.time+.005,ccur,R=0.,receiver_valid=False,pressure=.5)
        ck('recontact monitoring never resumes approach or holding',state['recontact_episode_count']==1 and g.phase=='FREE_CLEAR_AND_RETREAT' and not state['holding_force_tasks_active'])
        f,state=observe(g,f.time+.005,ccur,R=0.,receiver_valid=False,pressure=.5,aux_valid=False)
        ck('unknown afterclear not manufactured recontact episode',state['recontact_episode_count']==1 and state['low_hand_pressure_N'] is None)
        u,z=graph(lift=0.);f,state=observe(u,1.,z,R=0.,receiver_valid=False)
        ck('no receiver at zero lift explicitly unsupported freeclear',u.phase=='UNSUPPORTED_FREE_CLEAR' and state['unsupported_clearance'] and not state['placed'])
        for tt in (1.005,1.01,1.015):f,state=observe(u,tt,z,R=1.)
        ck('late receiver event cannot regress unsupported freeclear to gripmode',u.phase=='UNSUPPORTED_FREE_CLEAR')
        b,bc=graph()
        for tt in (1.,1.005,1.01,1.015):f,state=observe(b,tt,bc,R=1.,hand_valid=False)
        ck('receiver candidate does not require full hand wrench availability',b.phase=='CONTACT_UNLOADING' and not state['low_hand_pressure_available'])
        # Nonavailable / malformed kernel output preserves explicit legal fallback.
        module._contact_solve=lambda *a,**kw:None
        r,d=b.action(frame=f,adapter=adapter,current21=bc,lower21=lower,upper21=upper,generation=1)
        ck('unavailable kernel explicit legal nominal fallback without certificate',d['fallback'] is not None and not d['geometry_proposal_accepted'] and np.all(r.values>=lower)&np.all(r.values<=upper))
        module._contact_solve=lambda *a,**kw:SimpleNamespace(available=True,x=np.full(21,np.nan),status='badfixture',levels=(),diagnostics={})
        b.last_decision=-np.inf;r,d=b.action(frame=f,adapter=adapter,current21=bc,lower21=lower,upper21=upper,generation=1)
        ck('nonfinite accepted-kernel output rejected honestly',not d['geometry_proposal_accepted'] and d['numerical_error']=='invalid_normalized_kernel_proposal')
        f,state=observe(b,4.5,bc,R=0.,receiver_valid=False)
        ck('original3.5deadline recorded with unclearedhand, no extension/placed',state['deadline_reached'] and state['deadline_without_sampled_hand_clearance'] and not state['placed'])
        rejects('bool generation rejected',lambda:b.action(frame=f,adapter=adapter,current21=bc,lower21=lower,upper21=upper,generation=True))
        ck('all forbidden pipelines0',not forbidden)
    finally:module._contact_solve=original
    source=ROOT/'v17_contact_transfer/supervisor.py'
    report=dict(schema='v17_supervisor_author_pure_v1',passed=all(c['passed'] for c in checks),checks=checks,check_count=len(checks),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope=dict(physics_steps=0,models_constructed=0,sensor_samples=0,writer_calls=0,root_kernel_math_not_certified_by_fixture=True,
            lifecycle_dispatch_only=True,normal_work_is_reference_proposal_not_material_safety=True))
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),source=report['source_sha256'],path=str(out),failed=[c['name'] for c in checks if not c['passed']]),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

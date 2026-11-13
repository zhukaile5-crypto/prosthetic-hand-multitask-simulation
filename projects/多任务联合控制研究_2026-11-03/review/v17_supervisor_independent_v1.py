"""Eight independent lifecycle/interface counterexamples, no numerical QP claim.

No model, plant, public archive, renderer, sensor or physics is constructed.
Only the contact solver call is replaced by a declared normalized proposal.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry
from v15_support.receiver_sensor import ReceiverMeasurement
from v16_release_geometry.transfer_observer import TransferObserver,HandSupportSample
import v17_contact_transfer.supervisor as subject

EXPECTED='b5eb741004ae763ffe604223b0c7e64d0c6954500a116ef401d406d2f200834b'
CHECKS=[]
def ck(name,condition,details=None):CHECKS.append(dict(name=name,passed=bool(condition),details=details))
def rejects(call):
    try:call()
    except (ValueError,TypeError):return True
    return False

START=np.r_[.3+np.arange(18)*.01,.05,.03,-.02]
OPEN=np.full(18,-.2)
LO=np.r_[np.full(18,-2.),0.,-.5,-.5]
HI=np.r_[np.full(18,2.),.1,.5,.5]
RATES=np.r_[np.full(18,3.),.18,.12,.12]

def graph(lift=.05):
    ref=START.copy();ref[18]=lift
    entry=ReleaseEntry(1.,Reference21(ref,1.),1)
    observer=TransferObserver(request_time_s=1.,receiver_half_size_xy_m=[.2,.1],receiver_up_axis3=[0,0,1])
    return subject.ReleaseSupervisor(entry,open18=OPEN,observer=observer),ref

def packet(now,*,pressure=1.,hand_valid=True,receiver=True,compression=1.,aux_valid=True):
    acq=now-.01
    frame=SimpleNamespace(time=now,dt=.005,normal=np.full(5,pressure),valid=np.full(5,hand_valid,bool),
        sample_time=acq,age=.01,auxiliary_pressure=0.,auxiliary_valid=aux_valid,auxiliary_age=.01)
    hand=HandSupportSample(now,acq,hand_valid,.7 if hand_valid else None,.1 if hand_valid else None,
        None if hand_valid else 'independent_unknown_hand')
    r=ReceiverMeasurement(now,acq,.01,receiver,
        np.array([0.,0.,-compression,0.,0.,0.]) if receiver else None,
        compression if receiver else None,np.full(6,.01),np.full(6,.03),None,
        'boundary_after_forward',None if receiver else 'random_dropout')
    return frame,dict(time_s=now,hand=hand,receiver=r,hand_receiver_confounded=None,geometry_acquisition_time_s=None)

def observe(g,now,cur,**kwargs):
    f,ti=packet(now,**kwargs)
    return f,g.observe(frame=f,transfer_inputs=ti,current21=cur)

class Adapter:
    motion_task=None
    def __init__(self):self.calls=[]
    def current_geometry_tasks(self,frame,current,desired,lo,hi):
        self.calls.append(dict(time=frame.time,lo=lo.copy(),hi=hi.copy()))
        return SimpleNamespace(status='TEST_GEOMETRY_ONLY',diagnostics={'current_time_s':frame.time},
            normal_matrix_m_per_reference=np.zeros((1,21)),tangent_matrix_m_per_reference=np.zeros((2,21)),
            step_lower21=lo.copy(),step_upper21=hi.copy())

def main(out):
    path=ROOT/'v17_contact_transfer/supervisor.py';actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=EXPECTED:raise RuntimeError('Unexpected production source; no tests run')
    forbidden=[]
    def no(*a,**k):forbidden.append('forbidden');raise RuntimeError('No model/physics/writer in this gate')
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):setattr(mujoco,name,no)
    calls=[];control={'available':True,'x':np.r_[.25,np.zeros(17),.5,0.,0.]}
    def kernel(tasks,*,reference_step_scale21,motion_task):
        calls.append(dict(scale=reference_step_scale21.copy(),motion=motion_task))
        return SimpleNamespace(available=control['available'],x=control['x'].copy(),
            status='INDEPENDENT_INTERFACE_STUB_NOT_QP_EVIDENCE',levels=(),diagnostics={})
    original=subject._contact_solve;subject._contact_solve=kernel
    def act(g,f,cur,a,generation=1):
        return g.action(frame=f,adapter=a,current21=cur,lower21=LO,upper21=HI,generation=generation)
    try:
        # 1. No unrequested shape opening/hidden force stabilization in approach.
        g,cur=graph();a=Adapter();f,state=observe(g,1.,cur,hand_valid=False)
        native=g.nominal(1.02,LO,HI);command,debug=act(g,f,cur,a)
        ck('approach holds entry finger/wrist, requests lift0 and executes only actual5ms descent',
            np.array_equal(native.values[:18],cur[:18]) and np.array_equal(native.values[19:],cur[19:]) and native.values[18]==0
            and np.array_equal(command.values[:18],cur[:18]) and np.array_equal(command.values[19:],cur[19:])
            and abs(command.values[18]-(cur[18]-.18*.005))<1e-15 and command.time_s==1.
            and not calls and not a.calls and not state['holding_force_tasks_active'])

        # 2. Pre-request and repeated packets do not make a two-fresh event.
        observe(g,1.005,cur,hand_valid=False);f,_=observe(g,1.01,cur,hand_valid=False)
        same=observe(g,1.01,cur,hand_valid=False)[1]
        before=g.phase;streak=g.receiver_streak
        f,state=observe(g,1.015,cur,hand_valid=False)
        ck('two distinct postrequest receiver samples suffice despite unknown handH, without placement attribution',
            before=='APPROACH_SUPPORT' and streak==1 and same['receiver_fresh_streak']==1
            and g.phase=='CONTACT_UNLOADING' and state['first_receiver_event_latched']
            and state['receiver_event_is_any_actor_candidate_not_placement'] and not state['placed']
            and state['transfer']['stage_permission']=='OBSERVATION_ONLY')

        # 3. A test contact proposal opposing OPEN proves no hidden all18 cone.
        command,_=act(g,f,cur,a);lo=a.calls[-1]['lo'];hi=a.calls[-1]['hi']
        expected_delta=control['x']*(RATES*.02)*.25
        ck('CONTACT permits actual bidirectional thumb/lift and restores normalized kernel once',
            hi[0]>0 and lo[0]<0 and hi[18]>0 and lo[18]<0
            and np.allclose(command.values-cur,expected_delta,atol=2e-17,rtol=0)
            and g.lift_interval(Reference21(cur,f.time))[0]==0 and g.lift_interval(Reference21(cur,f.time))[1]>HI[18]
            and command.values[0]>cur[0] and command.values[18]>cur[18],
            dict(issued_thumb_delta=float(command.values[0]-cur[0]),issued_lift_delta=float(command.values[18]-cur[18]),
                 scale_restoration='x * 20ms rate scale then first-quarter command; no actual solver tested'))

        # 4. Valid sampled pressure continuity has its own clock, not R/H.
        clear,cc=graph();low_before=None
        for j in range(41):
            ff,ss=observe(clear,1.01+j*.005,cc,pressure=.01,receiver=False)
            if j==39:low_before=ss['low_hand_pressure_confirmed']
        uninterrupted=clear.was_clear and clear.phase=='FREE_CLEAR_AND_RETREAT' and not low_before
        interrupted,ic=graph()
        for j in range(21):observe(interrupted,1.01+j*.005,ic,pressure=.01,receiver=False)
        observe(interrupted,1.115,ic,pressure=.01,receiver=False,aux_valid=False)
        for j in range(40):ff,ss=observe(interrupted,1.12+j*.005,ic,pressure=.01,receiver=False)
        notyet=not ss['low_hand_pressure_confirmed']
        ff,ss=observe(interrupted,1.32,ic,pressure=.01,receiver=False)
        ck('handlow200ms is independent of receiver dropout and restarts after invalid aux packet',
            uninterrupted and notyet and ss['low_hand_pressure_confirmed'] and interrupted.phase=='FREE_CLEAR_AND_RETREAT')

        # 5. Clearing does not later restart a grip or approach phase.
        oldn=len(calls);ff,ss=observe(clear,1.215,cc,pressure=.8,receiver=False)
        command,dbg=act(clear,ff,cc,Adapter())
        free_kept=(clear.phase=='FREE_CLEAR_AND_RETREAT' and ss['recontact_episode_count']==1 and len(calls)==oldn
            and not ss['holding_force_tasks_active'] and dbg['geometry_status']=='NOT_REQUESTED_DIRECT_STAGE')
        unsupported,uc=graph(lift=0.);observe(unsupported,1.,uc,receiver=False)
        for t in (1.005,1.01,1.015):ff,ss=observe(unsupported,t,uc)
        ck('FREE recontact and UNSUPPORTED late receiver never return to gripping or CONTACT',
            free_kept and unsupported.phase=='UNSUPPORTED_FREE_CLEAR' and not ss['placed'] and ss['no_regrip_after_release'])

        # 6. Existing cached20ms plan must be discarded on generation/receipt.
        gg,gc=graph();aa=Adapter()
        for t in (1.,1.005,1.01,1.015):ff,_=observe(gg,t,gc)
        act(gg,ff,gc,aa);n=len(calls)
        ff,_=observe(gg,1.02,gc);act(gg,ff,gc,aa,generation=2);gen_refresh=len(calls)==n+1 and gg.plan_generation==2
        gg.reconcile(SimpleNamespace(execution_changed=True,stale_plan_replaced=False,mode_generation=2,executed=Reference21(gc,ff.time)))
        invalid=gg.last_decision==-np.inf
        ff,_=observe(gg,1.025,gc);n=len(calls);act(gg,ff,gc,aa,generation=2)
        ck('generation change and actual changed receipt each force a fresh decision before20ms',
            gen_refresh and invalid and len(calls)==n+1)

        # 7. Model/solver unknown is a bounded, honestly uncertified nominal.
        control['available']=False;ff,_=observe(gg,1.03,gc);gg.reconcile(SimpleNamespace(execution_changed=False,stale_plan_replaced=True,mode_generation=2,executed=Reference21(gc,ff.time)))
        command,dbg=act(gg,ff,gc,aa,generation=2)
        ck('unknown contact proposal fallback remains actual bounded without force/geometry certificate',
            not dbg['geometry_proposal_accepted'] and dbg['fallback']=='legal_canonical_nominal_no_geometry_or_motion_certificate'
            and np.all(command.values>=LO) and np.all(command.values<=HI) and np.all(abs(command.values-gc)<=RATES*.005+1e-15)
            and not dbg['pressure_prediction_available'])

        # 8. Deadline and typed5ms clock remain observational; never set placed.
        deadline,dc=graph()
        _,before=observe(deadline,4.495,dc,receiver=False)
        ff,after=observe(deadline,4.5,dc,receiver=False)
        bad=SimpleNamespace(**vars(ff));bad.dt=.02
        ck('original3.5s deadline retained with honest uncleared outcome and strict5ms action',
            not before['deadline_reached'] and after['deadline_reached'] and after['deadline_without_sampled_hand_clearance']
            and not after['placed'] and not after['placement_certified']
            and rejects(lambda:act(deadline,bad,dc,Adapter())) and not forbidden)
    finally:subject._contact_solve=original
    report=dict(schema='v17_supervisor_independent_v1',passed=all(c['passed'] for c in CHECKS),checks=CHECKS,
        check_count=len(CHECKS),source_sha256=actual,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=dict(models_constructed=0,physics_steps=0,public_archives_read=0,sensor_samples=0,writer_calls=0,
            kernel='Declared normalized stub only; no certification of hierarchy mathematics, physical authority or actual performance',
            CONTACT_lift_permission='Neutral0..finiteMAX task interval; actual caller box/rate remains authoritative; unlike V16 no receiver-event lift freeze'),
        forbidden_calls=forbidden)
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),passed=report['passed'],count=len(CHECKS),failures=[c['name'] for c in CHECKS if not c['passed']],sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

"""Small release execution-contract amendment gate, no model/physics."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v16_release_geometry.supervisor import ReleaseSupervisor
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry
from v16_release_geometry.transfer_observer import TransferObserver,PreReleaseLoadPrior,HandSupportSample
from v16_release_geometry.geometry_tasks import GeometryTasks
from v15_support.receiver_sensor import ReceiverMeasurement
from v14_coord.design import reference_rates,CONTROL_PERIOD_S
SOURCE=ROOT/'v16_release_geometry/supervisor.py';EXPECTED='d3f9a66a832907c300520972581dead25c3286c4c1712f85b2952521197b3628'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
checks=[]
def ck(name,value,detail=None):checks.append(dict(name=name,passed=bool(value),detail=detail))
class Adapter:
    def __init__(self):self.calls=0;self.domains=[]
    def current_geometry_tasks(self,frame,current,nominal,lo,hi):
        self.calls+=1;self.domains.append((lo.copy(),hi.copy()))
        return GeometryTasks('UNAVAILABLE_GEOMETRY_TASKS',None,None,None,None,None,nominal-current,lo,hi,dict(unknown_reasons=['synthetic_missing_geometry']))
def new():
    current=np.r_[np.full(18,.001),.1,0.,0.]
    g=ReleaseSupervisor(ReleaseEntry(12.,Reference21(current,12.),1),open18=np.zeros(18),observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=[.2,.2],receiver_up_axis3=[0.,0.,1.],prior=PreReleaseLoadPrior(1.2,.01,11.995)))
    return g,current,np.r_[np.full(18,-2.),0.,-2.,-2.],np.r_[np.full(18,2.),.1,2.,2.]
def observe(g,i,current,*,low=False,valid=True):
    t=12.01+i*.005;acq=t-.01;H,R=(1.,.2) if i==0 else (0.,1.2)
    frame=SimpleNamespace(time=t,normal=np.full(5,.01 if low else .2),valid=np.full(5,valid),auxiliary_valid=valid,auxiliary_pressure=0.,auxiliary_age=.01,sample_time=acq,age=.01)
    sigma=np.array([.005]*3+[.0001]*3);w=np.array([0.,0.,-R,-.005*R,.01*R,0.])
    receiver=ReceiverMeasurement(t,acq,.01,True,w,R,sigma,3*sigma,None,'boundary_after_forward',None)
    hand=HandSupportSample(t,acq,True,H,.001) if valid else HandSupportSample(t,None,False,None,None,'missing')
    inp=dict(time_s=t,hand=hand,receiver=receiver,hand_receiver_confounded=False,geometry_acquisition_time_s=acq)
    state=g.observe(frame=frame,transfer_inputs=inp,current21=current);return frame,state
def prepared():
    g,current,lo,hi=new()
    for i in range(3):frame,_=observe(g,i,current)
    a=Adapter();first,_=g.action(frame=frame,adapter=a,current21=current,lower21=lo,upper21=hi,generation=1)
    return g,a,first,lo,hi
g,a,first,lo,hi=prepared();ahead=first.values.copy();ahead[0]=g.plan_current[0]+.6*g.plan_delta[0]
frame,_=observe(g,3,ahead);second,debug=g.action(frame=frame,adapter=a,current21=ahead,lower21=lo,upper21=hi,generation=1)
ck('old_late_current_thumb_counterexample_now_nonreversing',second.values[0]<=ahead[0] and np.all(second.values[:18]>=0.) and np.all(second.values[:18]<=ahead[:18]) and np.all(abs(second.values-ahead)<=reference_rates('release')*CONTROL_PERIOD_S+1e-15),dict(current_thumb=float(ahead[0]),returned_thumb=float(second.values[0])))
g,a,first,lo,hi=prepared();frame,_=observe(g,3,first.values);second,debug=g.action(frame=frame,adapter=a,current21=first.values,lower21=lo,upper21=hi,generation=2)
ck('old_generation_counterexample_now_replans_before_retagging',a.calls==2 and g.decision_evidence[-1]['mode_generation']==2 and g.plan_generation==2)
g,current,lo,hi=new();_,one=observe(g,0,current);_,two=observe(g,1,current);a=Adapter();frame,_=observe(g,2,current);ref,dbg=g.action(frame=frame,adapter=a,current21=current,lower21=lo,upper21=hi,generation=1)
ck('two_fresh_receiver_packets_latch_and_action_itself_freezes_lift',not one['first_receiver_event_latched'] and two['first_receiver_event_latched'] and a.domains[0][0][18]==0. and a.domains[0][1][18]==0. and ref.values[18]==current[18] and g.retreat_start_s is None)
g,a,first,lo,hi=prepared();receipt=SimpleNamespace(execution_changed=True,stale_plan_replaced=False,executed=first);g.reconcile(receipt);frame,_=observe(g,3,first.values);g.action(frame=frame,adapter=a,current21=first.values,lower21=lo,upper21=hi,generation=1)
ck('changed_actual_receipt_discards_remaining20ms_plan',a.calls==2 and g.decision_evidence[-1]['time_s']==frame.time)
g,current,lo,hi=new();states=[]
for i in range(62):frame,state=observe(g,i,current,low=i>=2,valid=i!=20);states.append(state)
ck('pressure_window_interruption_requires_new_full200ms_and_transfer_evidence',states[60]['low_hand_pressure_confirmed'] is False and states[60]['coordinated_retreat_started'] is False and states[61]['low_hand_pressure_confirmed'] is True and states[61]['coordinated_retreat_started'] is True and states[61]['transfer']['stage_permission']=='EXPERIMENTAL_COORDINATION_EVIDENCE')
_,lost=observe(g,62,current,low=True,valid=False);_,deadline=observe(g,698,current,low=True,valid=False)
ck('latched_low_observation_unknown_later_never_means_placed_and_deadline_unchanged',lost['sampled_hand_low_was_observed'] is True and lost['low_hand_pressure_confirmed'] is False and not lost['placement_certified'] and not lost['delivery_confirmed'] and not lost['holding_force_tasks_active'] and deadline['deadline_reached'] is True and not deadline['placement_certified'])
prior=ROOT/'review/v16_supervisor_independent_v1.json';assert json.loads(prior.read_text(encoding='utf-8'))['passed'] is False
report=dict(schema='v16_supervisor_independent_v2',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=EXPECTED,source_unchanged=sha(SOURCE)==EXPECTED,prior_failed_evidence=dict(path=str(prior),sha256=sha(prior)),hierarchy_sha256=sha(ROOT/'v16_release_geometry/hierarchy.py'),
    scope='Six limited synthetic execution/process gates: two preserved079e counterexamples, first receiver freeze, receipt invalidation, pressure gap and no-placement/deadline. Existing hierarchy numerical audit is separate; no solver-family rerun.',models_created=0,physics_steps=0,raw_records_read=0)
out=ROOT/'review/v16_supervisor_independent_v2.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

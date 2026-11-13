"""Two narrow079e execution-contract counterexamples; no model/physics."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
EXPECTED='079e3e9a7b9639cf4c52c1fc6b550edfc921f627c75b51513c66d3f4cae0bee4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=ROOT/'v16_release_geometry/supervisor.py';snapshot=ROOT/'review/revisions/v16_supervisor_079e_independent_snapshot.py'
if not snapshot.exists():
    assert sha(source)==EXPECTED;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(source.read_bytes())
assert sha(snapshot)==EXPECTED
h=ModuleType('v16_release_geometry._supervisor_independent079e');h.__package__='v16_release_geometry';h.__file__=str(snapshot);sys.modules[h.__name__]=h;exec(compile(snapshot.read_bytes(),str(snapshot),'exec'),h.__dict__)
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry
from v16_release_geometry.transfer_observer import TransferObserver,PreReleaseLoadPrior,HandSupportSample
from v16_release_geometry.geometry_tasks import GeometryTasks
from v15_support.receiver_sensor import ReceiverMeasurement
from v14_coord.design import reference_rates,CONTROL_PERIOD_S
class Adapter:
    def __init__(self):self.calls=0
    def current_geometry_tasks(self,frame,current,nominal,lo,hi):
        self.calls+=1
        return GeometryTasks('UNAVAILABLE_GEOMETRY_TASKS',None,None,None,None,None,nominal-current,lo,hi,dict(unknown_reasons=['synthetic_missing_geometry']))
def observe(g,i,current):
    t=12.01+i*.005;acq=t-.01;H,R=(1.,.2) if i==0 else (.2,1.)
    frame=SimpleNamespace(time=t,normal=np.full(5,.2),valid=np.ones(5,bool),auxiliary_valid=True,auxiliary_pressure=0.,auxiliary_age=.01,sample_time=acq,age=.01)
    sigma=np.array([.005]*3+[.0001]*3);w=np.array([0.,0.,-R,-.005*R,.01*R,0.])
    receiver=ReceiverMeasurement(t,acq,.01,True,w,R,sigma,3*sigma,None,'boundary_after_forward',None)
    inputs=dict(time_s=t,hand=HandSupportSample(t,acq,True,H,.001),receiver=receiver,hand_receiver_confounded=False,geometry_acquisition_time_s=acq)
    g.observe(frame=frame,transfer_inputs=inputs,current21=current);return frame
def make():
    current=np.r_[np.full(18,.001),.1,0.,0.];entry=ReleaseEntry(12.,Reference21(current,12.),1)
    g=h.ReleaseSupervisor(entry,open18=np.zeros(18),observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=[.2,.2],receiver_up_axis3=[0.,0.,1.],prior=PreReleaseLoadPrior(1.2,.01,11.995)))
    for i in range(3):frame=observe(g,i,current)
    lower=np.r_[np.full(18,-2.),0.,-2.,-2.];upper=np.r_[np.full(18,2.),.1,2.,2.]
    adapter=Adapter();first,_=g.action(frame=frame,adapter=adapter,current21=current,lower21=lower,upper21=upper,generation=1)
    return g,adapter,first,lower,upper
g,a,first,lo,hi=make();ahead=first.values.copy();ahead[0]=g.plan_current[0]+.6*g.plan_delta[0]
frame=observe(g,3,ahead);second,debug=g.action(frame=frame,adapter=a,current21=ahead,lower21=lo,upper21=hi,generation=1)
legal_change=bool(np.all(abs(ahead-first.values)<=reference_rates('release')*CONTROL_PERIOD_S+1e-15))
thumb=dict(current_reference=float(ahead[0]),returned_reference=float(second.values[0]),closing_increment=float(second.values[0]-ahead[0]),same_generation=True,actual_change_within_5ms_rate=legal_change,fallback=debug['fallback'])
g2,a2,first2,lo,hi=make();frame2=observe(g2,3,first2.values);second2,debug2=g2.action(frame=frame2,adapter=a2,current21=first2.values,lower21=lo,upper21=hi,generation=2)
generation=dict(adapter_calls=a2.calls,recorded_plan_generation=g2.decision_evidence[-1]['mode_generation'],returned_debug_generation=debug2['mode_generation'],process_version_unchanged=g2.version==g2.plan_version)
checks=[dict(name='late_actual_reference_cannot_reverse_thumb_toward_grip',passed=bool(second.values[0]<=ahead[0]),detail=thumb),dict(name='generation_change_must_discard_and_replan',passed=bool(a2.calls==2 and g2.decision_evidence[-1]['mode_generation']==2),detail=generation)]
report=dict(schema='v16_supervisor_independent_v1',passed=all(x['passed'] for x in checks),checks=checks,source_sha256=EXPECTED,source_snapshot=str(snapshot),snapshot_sha256=sha(snapshot),hierarchy_sha256=sha(ROOT/'v16_release_geometry/hierarchy.py'),transfer_sha256=sha(ROOT/'v16_release_geometry/transfer_observer.py'),scope='Two execution-contract synthetic counterexamples only; unavailable geometry fallback is explicit. No live runtime or physical failure claim.',models_created=0,physics_steps=0)
out=ROOT/'review/v16_supervisor_independent_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],path=str(out),sha256=sha(out),checks=checks),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

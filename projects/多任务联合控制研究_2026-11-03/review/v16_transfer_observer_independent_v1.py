"""Independent public-sample oracle, including preserved395e failure.
No model, physics, real records, evaluator or production writes.
"""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
EXPECTED='395eb8f325f03d5127dafe744dbd298cab31eed11d016e8fbe8c190156e24bd9'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[ROOT/'v16_release_geometry/transfer_observer.py',*(ROOT/'review/revisions').glob('*transfer*.py')]
SOURCE=next((p for p in paths if p.is_file() and sha(p)==EXPECTED),None)
if SOURCE is None:raise RuntimeError('Pinned395e original/snapshot must remain available')
source_bytes=SOURCE.read_bytes();h=ModuleType('v16_transfer_independent395e');h.__file__=str(SOURCE);sys.modules[h.__name__]=h;exec(compile(source_bytes,str(SOURCE),'exec'),h.__dict__)
from v15_support.receiver_sensor import ReceiverMeasurement
checks=[]
def ck(name,okay,detail=None):checks.append(dict(name=name,passed=bool(okay),detail=detail))
def rejects(call):
    try:call();return False
    except (ValueError,TypeError):return True
def observer(request=12.,prior=True):return h.TransferObserver(request_time_s=request,receiver_half_size_xy_m=[.2,.2],receiver_up_axis3=[0.,0.,1.],prior=h.PreReleaseLoadPrior(1.2,.01,request-.005) if prior else None)
def receiver(t,R=1.,x=.01,y=.005,*,zero_sigma=False,valid=True):
    sigma=np.zeros(6) if zero_sigma else np.array([.005]*3+[.0001]*3)
    if not valid:return ReceiverMeasurement(t,None,None,False,None,None,sigma,3*sigma,None,None,'acquisition_unavailable')
    w=np.array([0.,0.,-R,-y*R,x*R,0.])
    return ReceiverMeasurement(t,t-.01,.01,True,w,R,sigma,3*sigma,None,'boundary_after_forward',None)
def put(o,i,*,valid=True,geometry=False,request=12.,H=None,R=None,x=.01,zero_sigma=False):
    t=request+.01+i*.005;H=(1. if i==0 else .2) if H is None else H;R=(.2 if i==0 else 1.) if R is None else R
    hand=h.HandSupportSample(t,t-.01,True,H,.001) if valid else h.HandSupportSample(t,None,False,None,None,'missing')
    return o.update(time_s=t,hand=hand,receiver=receiver(t,R,x,zero_sigma=zero_sigma,valid=valid),hand_receiver_confounded=geometry,geometry_acquisition_time_s=t-.01 if geometry is not None else None)

# Signed projection and engineering width from distinct force components.
M=np.zeros((6,5));M[2,:2]=[1.,-2.];N=np.array([1.,1.,0,0,0]);F=np.zeros((5,6));F[0,2]=.2;B=np.tile(np.eye(3)[:2],(5,1,1));aux=np.array([0.,0.,.1,0.,0.,0.])
kw=dict(read_time_s=12.01,acquisition_time_s=12.,up_at_acquisition3=[0,0,1],axis_time_s=12.,normal5=N,normal_wrench_map6x5=M,non_normal5x6=F,tangent_basis5x2x3=B,auxiliary_wrench6=aux,valid5=[True]*5,distributed_valid5=[True]*5,auxiliary_valid=True,auxiliary_acquisition_s=12.,force_sigma_N=.01,tangential_sigma_N=.02,auxiliary_uncertainty_N=.1)
signed=h.HandSupportSample.from_components(**kw)
ck('signed_H_not_pressure_sum_and_L1_engineering_width',signed.valid and np.isclose(signed.signed_support_N,-.7,rtol=0,atol=1e-15) and np.isclose(signed.uncertainty_N,.19,rtol=0,atol=1e-15))
wrong=h.HandSupportSample.from_components(**dict(kw,axis_time_s=12.005));missing=h.HandSupportSample.from_components(**dict(kw,auxiliary_valid=False))
ck('mismatched_axis_or_missing_auxiliary_remains_unknown',not wrong.valid and wrong.signed_support_N is None and not missing.valid and missing.uncertainty_N is None)
o=observer();pre=[]
for t in (12.,12.005):pre.append(o.update(time_s=t,hand=h.HandSupportSample(t,t-.01,True,1.,.01),receiver=receiver(t,.2)))
post=put(o,0)
ck('prerequest_delayed_samples_not_in_postrelease_window',all(e.status=='UNKNOWN' and 'not_yet_postrequest' in e.unknown_reasons for e in pre) and post.continuous_support_evidence['window_samples']==1)

o=observer()
for i in range(26):put(o,i)
lost=put(o,26,valid=False)
after=[put(o,i) for i in range(27,68)]
ck('one_missing_tick_breaks_dwell_no_zero_or_sigma_shrink',lost.status=='UNKNOWN' and lost.continuous_support_evidence['receiver_support_N'] is None and not after[-2].continuous_support_evidence['window_complete'] and after[-1].continuous_support_evidence['window_complete'] and abs(after[-1].continuous_support_evidence['receiver_engineering_width_N']-.015)<1e-15)
last=after[-1];identical=put(o,67)
ck('idempotence_changed_same_tick_and_clock_rewind',identical is last and rejects(lambda:put(o,67,H=.3)) and rejects(lambda:put(o,66)))
o=observer();put(o,0);gap=put(o,2)
ck('skipped_public_tick_cannot_accumulate_window',gap.continuous_support_evidence['window_samples']==1 and 'public_tick_gap' in gap.unknown_reasons)
o=observer()
for i in range(41):e=put(o,i,geometry=None if i==20 else False)
clear=put(o,41)
ck('one_unknown_attribution_breaks_whole_window_clearance',e.stage_permission=='OBSERVATION_ONLY' and clear.stage_permission=='OBSERVATION_ONLY' and not clear.continuous_support_evidence['geometry_clear_throughout_window'])
o=observer(prior=False)
for i in range(42):e=put(o,i)
ck('missing_prior_fraction_None_and_no_placement_certificate',e.continuous_support_evidence['coupled_transfer_fraction'] is None and e.stage_permission=='OBSERVATION_ONLY' and not e.to_public()['placement_certified'] and not e.to_public()['holding_support_may_be_requested'])

failures=[]
for request in (1.,12.):
    o=observer(request)
    for i in range(43):e=put(o,i,request=request,zero_sigma=True)
    p=e.continuous_support_evidence;failures.append(dict(request_time_s=request,cop_constant_m=[.01,.005],samples=p['window_samples'],window_complete=p['window_complete'],geometry_clear=p['geometry_clear_throughout_window'],balanced=p['load_sum_compatible_with_prior_proxy'],fraction=p['coupled_transfer_fraction'],cop_trend=p['cop_trend'],actual_status=e.status,actual_permission=e.stage_permission,expected_motion=False,expected_permission='EXPERIMENTAL_COORDINATION_EVIDENCE'))
ck('constant_nonzero_CoP_zero_sigma_must_not_resolve_motion',all(not f['cop_trend']['motion_resolved_against_engineering_proxy'] and f['actual_permission']==f['expected_permission'] for f in failures),failures)

ck('strict_public_contract_no_truth_extra_and_three_sigma_not_hard_bound',rejects(lambda:h.HandSupportSample(12.01,12.,True,.2,.01,target_id=7)) and rejects(lambda:h.HandSupportSample(12.01,12.,True,.2,True)) and 'engineering' in h.__doc__.lower() and 'not gaussian hard bounds' in h.__doc__.lower() and 'mujoco' not in sys.modules)
report=dict(schema='v16_transfer_observer_independent_v1',passed=all(x['passed'] for x in checks),checks=checks,source_sha256=EXPECTED,source_snapshot_path=str(SOURCE),source_snapshot_unchanged=sha(SOURCE)==EXPECTED,checker_sha256=sha(__file__),receiver_sha256=sha(ROOT/'v15_support/receiver_sensor.py'),
    scope='Independent signed force/clock/freshness/missing-window/attribution/idempotence and zero-sigma constant-CoP fixtures. No author30 rerun, actual raw/model or physical performance validation.',
    issues=['Constant nonzero CoP with legitimate zero engineering width is falsely classified moving due to OLS cancellation; fix numerical regression invariance, not sensor sigma or physical threshold.'],model_created=False,raw_records_read=0,physics_steps=0)
out=ROOT/'review/v16_transfer_observer_independent_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

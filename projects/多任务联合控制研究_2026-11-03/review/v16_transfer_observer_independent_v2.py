"""Limited numerical-amendment review; old failed395e report is retained."""
from pathlib import Path
from types import ModuleType
import ast,copy,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'v16_release_geometry/transfer_observer.py';OLD=ROOT/'review/revisions/v16_transfer_observer_395e_before_centered_cop.py'
EXPECTED='700e840b57471f1caa0c1dfee8e66bd8718d359665e2eccbaa00acc3d6b6c705'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED and sha(OLD)=='395eb8f325f03d5127dafe744dbd298cab31eed11d016e8fbe8c190156e24bd9'
h=ModuleType('v16_transfer_independent700e');h.__file__=str(SOURCE);sys.modules[h.__name__]=h;exec(compile(SOURCE.read_bytes(),str(SOURCE),'exec'),h.__dict__)
from v15_support.receiver_sensor import ReceiverMeasurement
checks=[]
def ck(name,okay,detail=None):checks.append(dict(name=name,passed=bool(okay),detail=detail))
before=ast.parse(OLD.read_text(encoding='utf-8'));after=ast.parse(SOURCE.read_text(encoding='utf-8'))
def cop_node(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test).startswith("complete and all((s['cop'] is not None"))
# Different Python AST unparse versions may parenthesize the generator.
def find_cop(tree):
    for n in ast.walk(tree):
        if isinstance(n,ast.If) and "complete and all" in ast.unparse(n.test) and "s['cop']" in ast.unparse(n.test):return n
    raise AssertionError('CoP regression branch not found')
restored=copy.deepcopy(after);find_cop(restored).body=copy.deepcopy(find_cop(before).body)
ck('only_CoP_regression_branch_changed_AST',ast.dump(restored,include_attributes=False)==ast.dump(before,include_attributes=False))

def run(request,*,rate=0.,noise=False):
    o=h.TransferObserver(request_time_s=request,receiver_half_size_xy_m=[.2,.2],receiver_up_axis3=[0,0,1],prior=h.PreReleaseLoadPrior(1.2,.01,request-.005));samples=[]
    for i in range(43):
        acq=request+i*.005;t=acq+.01;H,R=(1.,.2) if i==0 else (.2,1.);x=.01+rate*(i*.005);y=.005
        sigma=np.array([.001*(1+i/100)]*3+[.00003*(1+i/50)]*3) if noise else np.zeros(6)
        w=np.array([0.,0.,-R,-y*R,x*R,0.]);unc=3*sigma
        receiver=ReceiverMeasurement(t,acq,.01,True,w,R,sigma,unc,None,'boundary_after_forward',None)
        e=o.update(time_s=t,hand=h.HandSupportSample(t,acq,True,H,.001),receiver=receiver,hand_receiver_confounded=False,geometry_acquisition_time_s=acq)
        samples.append((w,unc))
    return e,samples[-41:]

constants=[run(t)[0] for t in (1.,12.)]
ck('both_origins_constant_nonzero_CoP_exact_zero_and_permission_restored',all(e.stage_permission=='EXPERIMENTAL_COORDINATION_EVIDENCE' and e.continuous_support_evidence['cop_trend']['velocity_m_s']==[0.,0.] and not e.continuous_support_evidence['cop_trend']['motion_resolved_against_engineering_proxy'] for e in constants))
linears=[run(t,rate=.1)[0] for t in (1.,12.)];trends=[e.continuous_support_evidence['cop_trend']['velocity_m_s'] for e in linears]
ck('true_linear_trend_survives_and_time_origin_is_irrelevant',trends[0]==trends[1] and np.allclose(trends[0],[.1,0.],rtol=0,atol=1e-15) and all(e.stage_permission=='OBSERVATION_ONLY' and e.continuous_support_evidence['cop_trend']['motion_resolved_against_engineering_proxy'] for e in linears),dict(trends=trends))

e,samples=run(12.,rate=.03,noise=True);width=[]
for w,u in samples:
    candidates_x=[-mm/ff for mm in (w[4]-u[4],w[4]+u[4]) for ff in (w[2]-u[2],w[2]+u[2])]
    candidates_y=[mm/ff for mm in (w[3]-u[3],w[3]+u[3]) for ff in (w[2]-u[2],w[2]+u[2])]
    point=np.array([-w[4]/w[2],w[3]/w[2]])
    width.append([max(point[0]-min(candidates_x),max(candidates_x)-point[0]),max(point[1]-min(candidates_y),max(candidates_y)-point[1])])
width=np.array(width);ticks=np.arange(41,dtype=float)-20;coeff=ticks/(.005*float(ticks@ticks))
expected=abs(coeff)@width;actual=np.array(e.continuous_support_evidence['cop_trend']['engineering_width_m_s'])
wrong_independent_anchor=np.abs(coeff[1:])@(width[1:]+width[0])
bound=128*np.finfo(float).eps*max(1.,float(expected.max()))
ck('shared_anchor_interval_matches_independent_full_coefficient_L1_oracle',np.max(abs(actual-expected))<=bound and np.all(actual<wrong_independent_anchor),dict(actual=actual.tolist(),independent_L1_oracle=expected.tolist(),incorrect_independent_anchor=wrong_independent_anchor.tolist(),roundoff_bound=bound))
public=e.to_public()
ck('numerical_fix_does_not_claim_Gaussian_or_placement_certification',not public['placement_certified'] and not public['physical_safety_certified'] and not public['holding_support_may_be_requested'] and 'engineering proxies' in h.__doc__.lower() and 'not gaussian hard bounds' in h.__doc__.lower() and 'mujoco' not in sys.modules)
prior=ROOT/'review/v16_transfer_observer_independent_v1.json';v1=json.loads(prior.read_text(encoding='utf-8'));assert v1['passed'] is False
report=dict(schema='v16_transfer_observer_independent_v2',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=EXPECTED,source_unchanged=sha(SOURCE)==EXPECTED,old_failed_evidence=dict(path=str(prior),sha256=sha(prior),passed=False),checker_sha256=sha(__file__),
    scope='Only395e→700e regression-branch AST diff, zero-sigma constants/two clock origins, genuine linear trend and shared-anchor coefficient uncertainty. Prior9 passing scopes not rerun. No actual records/model/physics.',model_created=False,raw_records_read=0,physics_steps=0)
out=ROOT/'review/v16_transfer_observer_independent_v2.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

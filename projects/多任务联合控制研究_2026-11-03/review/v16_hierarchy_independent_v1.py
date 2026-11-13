"""Independent finite 2D active-face enumeration and analytical hierarchy gate.

No optimizer library, model, FK, plant, runtime, physics or actual run input.
The reference solver enumerates all zero/one/two active inequality subsets,
solves their equality-constrained KKT systems, and picks the feasible minimum.
This is a small well-conditioned oracle, not a general-purpose replacement.
"""
from pathlib import Path
from itertools import combinations
from types import SimpleNamespace
import sys,importlib.util,hashlib,json,datetime,warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'v16_release_geometry/hierarchy.py'
EXPECTED='5564c7165279e46b11642719461737418bf31be23a66acf3c9f672982b4401bf'
CHECKS=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def check(name,passed,details=None):CHECKS.append(dict(name=name,passed=bool(passed),details=details))
def full_constraints(lo,hi,G=None,h=None):
    return np.vstack((np.eye(2),-np.eye(2),np.empty((0,2)) if G is None else G)),np.r_[hi,-lo,np.empty(0) if h is None else h]
def face_oracle(A,b,G,h,E,f):
    """Every optimum of a 2D polygon is on a face with <=2 independent rows."""
    H=2*A.T@A;c=2*A.T@b;options=[]
    for count in range(3):
        for subset in combinations(range(len(G)),count):
            ids=list(subset);C=np.vstack((E,G[ids]));d=np.r_[f,h[ids]]
            K=np.block([[H,C.T],[C,np.zeros((len(C),len(C)))]])
            rhs=np.r_[c,d];v=np.linalg.lstsq(K,rhs,rcond=None)[0];x=v[:2]
            if np.max(abs(K@v-rhs),initial=0)>2e-9 or np.max(G@x-h,initial=0)>2e-9:continue
            options.append((float(np.square(A@x-b).sum()),x))
    if not options:raise AssertionError('Independent face oracle found no feasible stationary face')
    return min(options,key=lambda p:p[0])
def oracle_levels(lo,hi,levels,G=None,h=None):
    GG,hh=full_constraints(lo,hi,G,h);E=np.empty((0,2));f=np.empty(0);records=[]
    for A,b in levels:
        loss,x=face_oracle(A,b,GG,hh,E,f);records.append(dict(x=x.copy(),residual=A@x-b,loss=loss,E=E.copy(),f=f.copy()))
        E=np.vstack((E,A));f=np.r_[f,A@x]
    return x,records,GG,hh
def direct_kkt(A,b,x,G,h,E):
    g=2*A.T@(A@x-b);active=np.flatnonzero(abs(G@x-h)<2e-8);best=np.inf
    for count in range(3):
        for subset in combinations(active,count):
            ids=list(subset);C=np.vstack((E,G[ids]))
            if len(C):
                lam=np.linalg.lstsq(C.T,-g,rcond=None)[0]
                if np.min(lam[len(E):],initial=0)<-2e-8:continue
                r=float(np.max(abs(g+C.T@lam),initial=0))
                comp=float(np.max(abs(lam[len(E):]*(h[ids]-G[ids]@x)),initial=0))
            else:r=float(np.max(abs(g),initial=0));comp=0.
            best=min(best,max(r,comp))
    return best
def run_case(m,name,lo,hi,levels,G=None,h=None):
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    ls=[(np.asarray(A,float).reshape((-1,2)),np.asarray(b,float)) for A,b in levels]
    expected,stages,GG,hh=oracle_levels(lo,hi,ls,G,h)
    r=m.solve_hierarchy(lower=lo,upper=hi,levels=tuple(m.Level(str(i),A,b) for i,(A,b) in enumerate(ls)),inequality_matrix=G,inequality_upper=h)
    differences=[];kkts=[]
    for (A,b),s in zip(ls,stages):
        differences.append(float(np.max(abs(A@r.x-b-s['residual']),initial=0)))
        kkts.append(direct_kkt(A,b,r.x,GG,hh,s['E']))
    passed=r.available and np.max(GG@r.x-hh,initial=0)<2e-8 and max(differences,default=0)<2e-8 and max(kkts,default=0)<2e-7
    check(name,passed,dict(actual=r.x.tolist(),oracle=expected.tolist(),residual_differences=differences,independent_kkt_inf=kkts,status=r.status))
    return r

def main():
    before=sha(SOURCE)
    if before!=EXPECTED:raise RuntimeError('Source changed; bind a reviewed revision before executing')
    m=load(SOURCE,'_v16_hierarchy_independent_candidate')
    # Preserve the independently observed false certificate from the old source.
    oldpath=ROOT/'review/revisions/v16_hierarchy_a4f233cc_independent_snapshot.py'
    old=load(oldpath,'_v16_hierarchy_independent_old_a4f')
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        oldresult=old.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=(old.Level('goal',np.eye(2),np.array([1.,0.])),),inequality_matrix=[[1e200,0.]],inequality_upper=[0.])
    old_failure=dict(source_sha256=sha(oldpath),available=oldresult.available,x=oldresult.x.tolist(),reported_violation=oldresult.diagnostics['hard_constraint_violation'],original_row_value=float(np.array([1e200,0.])@oldresult.x),warnings=[str(w.message) for w in ws])
    for scale in (1e-200,1.,1e200):
        r=m.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=(m.Level('goal',np.eye(2),np.array([1.,0.])),),inequality_matrix=[[scale,0.]],inequality_upper=[0.])
        check('hard_halfspace_positive_rescaling_'+str(scale),r.available and abs(r.x[0])<1e-12 and abs(r.x[1])<1e-12,dict(x=r.x.tolist(),status=r.status))
    run_case(m,'analytic_compatible_tasks',[-1,-1],[1,1],[([[1,0]],[.25]),([[0,1]],[.75])])
    r=run_case(m,'conflicting_lower_tasks_cannot_trade_higher_residual',[-1,-1],[1,1],[([[1,1]],[1]),(np.eye(2),[-1,1]),([[1,0]],[.9])])
    check('analytic_conflict_solution',np.allclose(r.x,[0,1],atol=2e-8,rtol=0))
    run_case(m,'boundary_dependent_duplicates',[-1,-1],[1,1],[([[1,0]],[2]),([[0,1]],[.3])],np.array([[1.,0.],[2.,0.],[1.,1.]]),np.array([.4,.8,.7]))
    run_case(m,'fixed_coordinate_and_impossible_target',[0,-1],[0,1],[([[1,0]],[2]),([[0,1]],[.3])])
    run_case(m,'empty_primary_rows_leave_secondary_free',[-1,-1],[1,1],[(np.empty((0,2)),[]),(np.eye(2),[.2,-.3])])
    G=np.array([[1.,-1.]]);h=np.zeros(1)
    op=run_case(m,'OPEN_before_tangent_avoids_trivial_zero',[0,0],[1,1],[([[1,0]],[1]),([[0,1]],[0])],G,h)
    tp=run_case(m,'reverse_order_demonstrates_zero_lock',[0,0],[1,1],[([[0,1]],[0]),([[1,0]],[1])],G,h)
    check('priority_is_substantive_not_weight_blend',np.allclose(op.x,[1,1],atol=2e-8,rtol=0) and np.allclose(tp.x,[0,0],atol=2e-8,rtol=0))
    # A fixed-seed finite set: enumerate every active subset for each level.
    rng=np.random.default_rng(491673)
    for k in range(12):
        angle=rng.uniform(-np.pi,np.pi);A=np.array([[np.cos(angle),np.sin(angle)]])
        C=rng.normal(size=(3,2));d=rng.uniform(.1,1.,size=3)
        run_case(m,'enumerated_polygon_'+str(k),[-1,-1],[1,1],[(A,rng.uniform(-1.5,1.5,size=1)),(np.eye(2),rng.uniform(-2,2,size=2))],C,d)
    # Representation changes preserve the exact finite linear problem.
    A=np.array([[1.,2.]]);b=np.array([.7]);base=[(A,b),(np.eye(2),np.array([-.5,.8]))]
    G=np.array([[1.,1.],[-1.,2.]]);h=np.array([.8,.9])
    x=run_case(m,'representation_base',[-1,-1],[1,1],base,G,h).x
    split=[(np.vstack((A,A))/np.sqrt(2),np.r_[b,b]/np.sqrt(2)),base[1]]
    y=run_case(m,'representation_proportional_split',[-1,-1],[1,1],split,G[::-1],h[::-1]).x
    perm=[(aa[:,::-1],bb) for aa,bb in base]
    z=run_case(m,'representation_coordinate_permutation',[-1,-1],[1,1],perm,G[:,::-1],h).x[::-1]
    check('split_and_permutation_preserve_solution',np.allclose(x,y,atol=2e-8,rtol=0) and np.allclose(x,z,atol=2e-8,rtol=0))
    # Real geometry wrapper, only 2 of its 21 reference coordinates move.
    lo=np.zeros(21);hi=np.zeros(21);hi[[0,18]]=1;nom=np.zeros(21);nom[0]=1
    normal=np.zeros((1,21));normal[0,[0,18]]=[1,-1]
    tangent=np.zeros((1,21));tangent[0,18]=1
    tasks=SimpleNamespace(step_lower21=lo,step_upper21=hi,open_target_increment21=nom,normal_matrix_m_per_reference=normal,normal_offset_m=np.zeros(1),tangent_matrix_m_per_reference=tangent,tangent_target_m=np.zeros(1))
    wr=m.solve_geometry_release(tasks,reference_step_scale21=np.ones(21))
    check('geometry_wrapper_finger_OPEN_lift_coordination',wr.available and np.allclose(wr.x[[0,18]],[1,1],atol=2e-8,rtol=0) and [r['name'] for r in wr.levels]==['OPEN_scalar_progress','minimum_tangent_at_retained_OPEN_progress','closest_declared_nominal'],dict(x=wr.x.tolist(),status=wr.status))
    def rejection(**kw):
        try:m.solve_hierarchy(**kw)
        except (ValueError,FloatingPointError):return True
        return False
    args=dict(lower=[-1.,-1.],upper=[1.,1.],levels=(m.Level('x',np.eye(2),np.ones(2)),))
    check('infeasible_start_and_impossible_constant_rejected',rejection(**{**args,'initial':[2.,0.]}) and rejection(**{**args,'inequality_matrix':[[0.,0.]],'inequality_upper':[-1.]}))
    budget=m.solve_hierarchy(**{**args,'max_iterations':1,'inequality_matrix':[[1.,1.]],'inequality_upper':[.1]})
    check('iteration_limit_not_falsely_available',not budget.available and budget.status=='UNAVAILABLE_UNCERTIFIED_HIERARCHY')
    check('unrepresentable_objective_not_falsely_available',rejection(**{**args,'levels':(m.Level('overflow',np.eye(2)*1e200,np.ones(2)*1e200),)}))
    after=sha(SOURCE)
    report=dict(schema='v16-hierarchy-independent-v1',role='independent_finite_pure_gate',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(SOURCE),source_sha256=before,source_after_sha256=after,test_source_sha256=sha(Path(__file__)),old_false_certificate=old_failure,scope=dict(two_dimensional_active_subset_enumeration=12,analytical_cases=True,real_21_axis_wrapper_case=True,model_constructions=0,physics_steps=0,FK_calls=0,scipy_used=False,global_numeric_guarantee=False),checks=CHECKS,passed=before==after and all(c['passed'] for c in CHECKS),counts=dict(total=len(CHECKS),passed=sum(c['passed'] for c in CHECKS),failed=sum(not c['passed'] for c in CHECKS)))
    out=ROOT/'review/v16_hierarchy_independent_v1.json'
    if out.exists():raise FileExistsError('Preserve previous evidence')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(report=str(out),sha256=sha(out),source_sha256=before,passed=report['passed'],counts=report['counts']),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

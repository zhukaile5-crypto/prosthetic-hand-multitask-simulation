"""Eight bounded independent numeric gates; no model, FK, physics or data fit."""
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v17_contact_transfer import hierarchy as kernel
from v17_contact_transfer.task_solver import solve_contact_unloading

EXPECTED={'v17_contact_transfer/hierarchy.py':'a3c9f090a5cf4368d7d1ab36344b0aac4cfd76aac3c0367a298a38fdf2932457',
          'v17_contact_transfer/task_solver.py':'951d3555c93fe798a31fda6f2143b15ddd08571b85a46dbfced414a2a9e73a42'}
CHECKS=[]
def ck(name,value,details=None):CHECKS.append(dict(name=name,passed=bool(value),details=details))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rejects(call):
    try:call()
    except (ValueError,TypeError,FloatingPointError):return True
    return False
def pad(rows,k=3):
    a=np.asarray(rows,float).reshape(-1,k);out=np.zeros((len(a),21));out[:,:k]=a;return out
def task(C,p,canonical,*,tangent=(),time_s=2.):
    k=len(canonical);lo=np.zeros(21);hi=np.zeros(21);lo[:k]=-1;hi[:k]=1
    nominal=np.zeros(21);nominal[:k]=canonical;T=pad(tangent,k)
    return SimpleNamespace(normal_matrix_m_per_reference=pad(C,k),normal_offset_m=np.zeros(len(C)),
        step_lower21=lo,step_upper21=hi,open_target_increment21=nominal,
        tangent_matrix_m_per_reference=T,tangent_target_m=np.zeros(len(T)),
        diagnostics=dict(current_time_s=time_s,patches=[dict(observed_normal_N=float(v)) for v in p]))
def motion(rows,target,*,k=3,stamp=2.):
    return SimpleNamespace(time_s=stamp,matrix_Mx21=pad(rows,k),target_M=np.asarray(target,float),metadata={'role':'independent declared projection fixture'})
def original():
    t=task([[-1.,-1.,0.]],[8.],[-1.,1.,-.7],tangent=[[0,0,1]])
    return t,motion([[1,0,0],[0,0,1]],[-1.,.25])

def polygon_lex_oracle(levels,G,h):
    """Exhaustive2D feasible active-face LS, independent of kernel directions.

    Includes interior, every edge and every vertex, then freezes each attained
    vector. Convex least squares over a finite polygon attains one such face.
    This oracle neither imports nor calls any production nullspace/KKT helper.
    """
    E=np.empty((0,2));f=np.empty(0);x=np.zeros(2);losses=[]
    for A,b in levels:
        A=np.asarray(A,float).reshape(-1,2);b=np.asarray(b,float);candidates=[]
        for count in range(3):
            for active in combinations(range(len(G)),count):
                W=np.vstack((E,G[list(active)]));rhs=np.r_[f,h[list(active)]]
                Q=2*A.T@A;linear=2*A.T@b
                K=np.block([[Q,W.T],[W,np.zeros((len(W),len(W)))]])
                sol=np.linalg.lstsq(K,np.r_[linear,rhs],rcond=None)[0][:2]
                if np.max(abs(W@sol-rhs),initial=0)>1e-10 or np.max(G@sol-h,initial=0)>1e-10:continue
                candidates.append((float(np.sum((A@sol-b)**2)),sol))
        if not candidates:raise AssertionError('Independent polygon oracle has no feasible face')
        loss,x=min(candidates,key=lambda pair:pair[0]);losses.append(loss)
        E=np.vstack((E,A));f=np.r_[f,A@x]
    return x,losses

def main(out):
    before={p:sha(ROOT/p) for p in EXPECTED}
    if before!=EXPECTED:raise RuntimeError('Unexpected candidate source, do not run')
    t,m=original();r=solve_contact_unloading(t,reference_step_scale21=np.ones(21),motion_task=m)
    notes=[v for level in r.levels for v in level['degenerate_face_repairs']]
    independent_hard=float(np.max(t.normal_matrix_m_per_reference@r.x+t.normal_offset_m,initial=0))
    ck('original21D failed face now attains [1,1,.25] with original hard/KKT diagnostics',
        r.available and np.allclose(r.x[:3],[1,1,.25],atol=2e-12,rtol=0)
        and np.array_equal(r.x[3:],np.zeros(18)) and independent_hard<=1e-12
        and all(z['status']=='KKT_CONVERGED' and z['stationarity_inf']<1e-10 and z['original_hard_primal_violation']<1e-10 for z in r.levels)
        and r.diagnostics['all_original_hard_rows_retained'] and any(z['corrected'] for z in notes)
        and not r.diagnostics['force_prediction_available'],
        dict(status=r.status,x=r.x[:3].tolist(),levels=[(z['name'],z['status'],z['stationarity_inf']) for z in r.levels],
             repair_events=r.diagnostics['degenerate_face_events'],virtual_work_J=r.diagnostics['normal_virtual_unloading_work_J']))

    # Independent global face oracle on3 predetermined small convex problems.
    generic=[
        ('two_coupled_equalities',[(np.array([[1.,2.]]),np.array([1.5])),(np.array([[1.,-1.]]),np.array([.25]))],[],[]),
        ('unreachable_sum_then_cross',[(np.array([[1.,1.]]),np.array([3.])),(np.array([[1.,-1.]]),np.array([-2.]))],[],[]),
        ('oblique_polygon',[(np.array([[1.,-.25]]),np.array([1.125])),(np.eye(2),np.array([-.5,.75]))],[[2.,-1.],[-1.,-1.]],[.5,.75])]
    cases=[];all_ok=True
    for name,levels,extra,bounds in generic:
        G=np.vstack((np.asarray(extra,float).reshape(-1,2),np.eye(2),-np.eye(2)))
        h=np.r_[bounds,np.ones(4)];oracle,losses=polygon_lex_oracle(levels,G,h)
        rr=kernel.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],
            levels=tuple(kernel.Level(str(i),A,b) for i,(A,b) in enumerate(levels)),
            inequality_matrix=np.asarray(extra,float).reshape(-1,2),inequality_upper=np.asarray(bounds,float))
        recomputed=[float(np.sum((A@rr.x-b)**2)) for A,b in levels]
        ok=rr.available and np.max(G@rr.x-h)<=1e-10 and np.allclose(rr.x,oracle,atol=5e-10,rtol=0)
        ok &= np.allclose(recomputed,losses,atol=5e-10,rtol=0)
        ok &= all(abs(rec['squared_loss']-loss)<5e-10 for rec,loss in zip(rr.levels,recomputed))
        all_ok &= ok;cases.append(dict(name=name,available=rr.available,x=rr.x.tolist(),oracle=oracle.tolist(),losses=recomputed,oracle_losses=losses))
    ck('three2D global active-face oracles agree on final point and every original loss',all_ok,cases)

    tiny=kernel.solve_hierarchy(lower=[0.,-1.],upper=[0.,1.],
        levels=(kernel.Level('y',np.array([[0.,1.]]),np.array([1.])),),
        inequality_matrix=[[0.,1e-240]],inequality_upper=[0.])
    near=kernel.solve_hierarchy(lower=[0.,-1.],upper=[0.,1.],
        levels=(kernel.Level('y',np.array([[0.,1.]]),np.array([1.])),),
        inequality_matrix=[[1.,2.**-60]],inequality_upper=[0.])
    near_notes=[v for z in near.levels for v in z['degenerate_face_repairs']]
    ck('real tiny independent blockers never become dependent permissions',
        tiny.available and tiny.x[1]==0 and tiny.diagnostics['original_hard_constraint_violation']==0
        and not near.available and near.x[1]==0 and all(not v['certified_dependent_rows'] for v in near_notes),
        dict(tiny_status=tiny.status,near_status=near.status,near_x=near.x.tolist(),near_proof=near_notes))

    no_motion=task([[-1.,0.,0.]],[1.],[0.,.6,-.7])
    unknown=solve_contact_unloading(no_motion,reference_step_scale21=np.ones(21))
    measured=solve_contact_unloading(no_motion,reference_step_scale21=np.ones(21),motion_task=motion([[0,1,0]],[0.]))
    bad=motion([[0,1,0]],[0.],stamp=2.005)
    missing_clock=task([[-1.,0.,0.]],[1.],[0.,.6,-.7],time_s=None)
    ck('unknown motion omitted, observed zero is real layer, unknown/mismatched clock rejected',
        unknown.available and measured.available and abs(unknown.x[1]-.6)<2e-12 and abs(measured.x[1])<2e-12
        and not unknown.diagnostics['observed_motion_task']['available'] and measured.diagnostics['observed_motion_task']['available']
        and rejects(lambda:solve_contact_unloading(no_motion,reference_step_scale21=np.ones(21),motion_task=bad))
        and rejects(lambda:solve_contact_unloading(missing_clock,reference_step_scale21=np.ones(21),motion_task=motion([[0,1,0]],[0.]))))

    merged=task([[-1,0,0],[0,-1,0]],[2.,6.],[-1.,-.5,-.7],tangent=[[0,0,1]])
    split=task([[-1,0,0],[0,-1,0],[-1,0,0]],[.5,6.,1.5],[-1.,-.5,-.7],tangent=[[0,0,1]])
    a=solve_contact_unloading(merged,reference_step_scale21=np.ones(21),motion_task=m)
    b=solve_contact_unloading(split,reference_step_scale21=np.ones(21),motion_task=m)
    ck('pressure-conserving samepoint subdivision preserves primary work and actual optimized proposal',
        a.available and b.available and np.allclose(a.x,b.x,atol=2e-12,rtol=0)
        and abs(a.diagnostics['normal_virtual_unloading_work_J']-b.diagnostics['normal_virtual_unloading_work_J'])<2e-12,
        dict(merged=a.x[:3].tolist(),split=b.x[:3].tolist(),virtual_work_J=a.diagnostics['normal_virtual_unloading_work_J']))

    strong=kernel.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=(
        kernel.Level('priority_x',np.array([[1.,0.]]),np.array([.375])),
        kernel.Level('priority_y',np.array([[0.,1.]]),np.array([-.625])),
        kernel.Level('strong_incompatible_last',1e4*np.eye(2),np.array([-1e7,1e7]))))
    ck('strong lower task cannot trade frozen attained higher residuals',
        strong.available and np.allclose(strong.x,[.375,-.625],atol=5e-12,rtol=0)
        and max(v['preceding_residual_drift_inf'] for v in strong.levels)<5e-12
        and all(v['numerical_atol']==1e-9 and v['numerical_rtol']==1e-8 for v in strong.levels),
        dict(x=strong.x.tolist(),higher_drift=[v['preceding_residual_drift_inf'] for v in strong.levels]))

    units=task([[-1.,0.]],[3.],[-.2,-.5],tangent=[[0,1]])
    scale=np.ones(21);scale[:2]=[.2,.5];units.step_lower21[:2]=[-.2,-.5];units.step_upper21[:2]=[.2,.5]
    uu=solve_contact_unloading(units,reference_step_scale21=scale,motion_task=motion([[0,1]],[.15],k=2))
    zero=task([[-1.,0.]],[1.],[0.,0.]);zero.step_lower21[:]=0;zero.step_upper21[:]=0
    zz=solve_contact_unloading(zero,reference_step_scale21=np.ones(21))
    ck('actual reference scale applied once, genuine fixed domain zero remains honest',
        uu.available and np.allclose((uu.x*scale)[:2],[.2,.15],atol=2e-12,rtol=0)
        and zz.available and np.array_equal(zz.x,np.zeros(21)) and zz.diagnostics['normal_progress_achieved_m']==0
        and not zz.diagnostics['physical_safety_certified'])

    with patch.object(kernel,'SPARSE_VISIT_BUDGET',0):
        budget=solve_contact_unloading(t,reference_step_scale21=np.ones(21),motion_task=m)
    budget_notes=[v for z in budget.levels for v in z['degenerate_face_repairs']]
    ck('exhausted exact-face proof budget rejects proposal without dropping original row',
        not budget.available and any('budget_exceeded' in str(v.get('reason')) for v in budget_notes)
        and budget.diagnostics['all_original_hard_rows_retained']
        and budget.diagnostics['original_hard_constraint_violation']<1e-10,
        dict(status=budget.status,levels=[z['status'] for z in budget.levels],repair_notes=budget_notes))

    after={p:sha(ROOT/p) for p in EXPECTED}
    report=dict(schema='v17_hierarchy_independent_v1',passed=all(c['passed'] for c in CHECKS),check_count=len(CHECKS),checks=CHECKS,
        source_sha256=before,source_unchanged=after==before,checker_sha256=sha(Path(__file__)),
        scope=dict(models=0,physics_steps=0,FK_calls=0,physical_archives_read=0,mujoco_imported='mujoco' in sys.modules,
            independent_oracle='2D exhaustive feasible active-face least squares, freeze each attained vector; no production KKT/nullspace helper reused',
            limits='Finite analytic regression and failure semantics only; no global numerical, physical contact, force or placement certificate'))
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),passed=report['passed'],count=len(CHECKS),failures=[c['name'] for c in CHECKS if not c['passed']],sha256=sha(out))))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

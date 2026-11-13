"""Nontrivial high-weight task-row witnesses; no tolerance relaxation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'review/v15_large_task_rows_v1.json'
    if out.exists():raise FileExistsError('Fresh witness evidence required')
    from v15_support.task_qp import solve_intent_qp
    from review.verify_v13_joint_intent_qp_independent_v1 import fixture,gradient,objective
    expected='6d48d44ad544af323a11a33a29d73ea5757c4cb35bd183a989b6a81433c817e4'
    if sha(ROOT/'v15_support/task_qp.py')!=expected:raise ValueError('Frozen math identity differs')
    rows=[]
    for scale in (1e3,1e5,1e8):
        k=fixture();E=np.zeros((1,22));E[0,[0,21]]=scale;b=np.array([2*scale]);r=solve_intent_qp(**k,extra_task_rows={'matrix':E,'target':b})
        x=np.r_[r.d21,r.zP];res=E@x-b;g=gradient(x,k)+2*E.T@res;a=scale**2/(1+2*scale**2)
        kd=float(np.max(abs(g[:21])));kz=float(max(abs(g[21:])));honest=not r.available or (kd<=r.diagnostics['numerical_tolerance_d21'] and kz<=r.diagnostics['numerical_tolerance_zP'])
        rows.append(dict(scale=scale,status=r.status,reason=r.reason,iterations=r.iterations,d0=float(r.d21[0]),z0=float(r.zP[0]),analytic_real_d0=a,analytic_real_z0=1+a,
            original_objective=r.objective,independent_objective=objective(x,k)+float(res@res),task_residual=float(res[0]),independent_raw_dKKT=kd,independent_raw_zKKT=kz,
            reported_dKKT=r.diagnostics['direct_KKT_inf_d21'],reported_zKKT=r.diagnostics['direct_KKT_inf_zP'],no_false_available=honest))
    result=dict(schema='v15_nontrivial_large_row_witnesses_v1',passed=all(r['no_false_available'] and abs(r['original_objective']-r['independent_objective'])<1e-10 for r in rows),rows=rows,source_sha256=expected,checker_sha256=sha(__file__),
        scope='One nonzero cold-start coupled d/z soft row; real-number optimum known. High-scale representability can honestly be unavailable even with a tiny objective gap; default original-unit KKT is never relaxed.',physics_steps=0,simulator_imported='mujoco' in sys.modules)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

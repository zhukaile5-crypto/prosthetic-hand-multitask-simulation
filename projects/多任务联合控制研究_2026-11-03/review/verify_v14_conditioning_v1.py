"""Unit/conditioning diagnosis and additional scope checks; pure NumPy only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EXPECTED='1f56174b10af062dea161513874607e6971788d1a9589b2e2c414a5694eeeebc'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'review/v14_conditioning_author_v1.json'
    if out.exists():raise FileExistsError('Fresh scope evidence required')
    source=ROOT/'v14_coord/intent_qp.py'
    if sha(source)!=EXPECTED:raise ValueError('Frozen math candidate changed')
    from v14_coord.intent_qp import solve_intent_qp
    from review.verify_v13_joint_intent_qp_independent_v1 import fixture,objective,gradient
    data=json.loads((ROOT/'review/v14_archived_fusion225_queries_v1.json').read_text('utf-8'))
    records=[];checks=[]
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    for case in data['queries']:
        k=case['kwargs'];A=np.asarray(k['A21x25']);J=np.asarray(k['Jpx25']);c=np.asarray(k['support_coeff_p']);s=k['scales'];sr=np.asarray(s['residual25']);sf=np.asarray(s['intentP']);sd=np.asarray(s['delta21']);sl=s['load']
        T=sr[:,None]*np.c_[A.T,-J.T];q=np.r_[np.zeros(21),c];D=np.r_[sd*sd,sf*sf]
        H=2*(T.T@T+np.diag(D)+sl*sl*np.outer(q,q))
        units=np.r_[sd,sf];Hbar=H/units[:,None]/units[None,:]
        eig=np.linalg.eigvalsh(H);scaled=np.linalg.eigvalsh(Hbar)
        # The original projected method uses a common step <=1/L in raw mixed
        # Nm/N effort and N patch coordinates; this compares the same Hessian,
        # not alternate objective weights or a new response model.
        records.append(dict(decision_id=case['decision_id'],time_s=case['time_s'],original_eigen_min=float(eig[0]),original_eigen_max=float(eig[-1]),original_condition=float(eig[-1]/eig[0]),dimensionless_condition=float(scaled[-1]/scaled[0]),delta_scale_min=float(min(sd)),delta_scale_max=float(max(sd)),patch_scale_min=float(min(sf)),patch_scale_max=float(max(sf))))
    k=fixture(2);k['patch_digit_p']=np.array([0,0]);k['caps5'][0]=1.;k['total_budget']=.8
    r=solve_intent_qp(**k);w=solve_intent_qp(**k,warm_start={'d21':r.d21,'zP':r.zP})
    check('warm_start_same_original_problem',r.available and w.available and abs(r.objective-w.objective)<1e-12)
    check('warm_start_does_not_change_p0_or_target',w.diagnostics['fixed_original_p0_21']==k['p0_21'].tolist() and w.diagnostics['original_z_intent_p']==k['z_intent_p'].tolist())
    bad=solve_intent_qp(**k,warm_start={'d21':np.full(21,np.nan),'zP':np.zeros(2)})
    check('unknown_warm_start_not_fabricated',bad.status=='unknown' and bad.d21 is None and bad.zP is None)
    # Equality/degenerate/hinge boundary examples with exact unretargeted goals.
    for index in range(12):
        x=fixture(index%4);P=len(x['y_p']);x['support_coeff_p']=np.full(P,(-1 if index%2 else 1.));x['load_proxy_N']=float(index%3)
        x['effort_lower21'][:index%6]=x['effort_upper21'][:index%6]=.25
        if index%3==0:x['caps5'][:]=0.
        if index%4==0:x['total_budget']=0.
        rr=solve_intent_qp(**x)
        check('degenerate_'+str(index)+'_original_KKT_available',rr.available)
        if rr.d21 is not None:
            xx=np.r_[rr.d21,rr.zP]
            check('degenerate_'+str(index)+'_original_objective',abs(objective(xx,x)-rr.objective)<1e-12)
            check('degenerate_'+str(index)+'_expanded_hinge_equivalence',abs(rr.objective-rr.diagnostics['expanded_least_squares_objective'])<1e-12)
    check('no_simulator_import','mujoco' not in sys.modules)
    check('frozen_source_unchanged',sha(source)==EXPECTED)
    def summarize(key):
        a=np.asarray([r[key] for r in records]);return dict(minimum=float(min(a)),median=float(np.median(a)),maximum=float(max(a)))
    result=dict(schema='v14_unit_conditioning_author_v1',passed=all(c['passed'] for c in checks),checks=checks,failures=[c for c in checks if not c['passed']],
        source_sha256=EXPECTED,query_fixture_sha256=sha(ROOT/'review/v14_archived_fusion225_queries_v1.json'),records=records,
        summary={key:summarize(key) for key in ('original_condition','dimensionless_condition','original_eigen_min','original_eigen_max','delta_scale_min','delta_scale_max')},
        interpretation='Hessian condition number depends on units. Congruence is only a variable substitution; all costs/scales/directKKT remain in original units. The old Euclidean grouped projection is independently correct, not blamed as an invalid projection. No closed-loop pressure result.',no_physics=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=len(result['failures']),summary=result['summary']),ensure_ascii=False))
if __name__=='__main__':main()

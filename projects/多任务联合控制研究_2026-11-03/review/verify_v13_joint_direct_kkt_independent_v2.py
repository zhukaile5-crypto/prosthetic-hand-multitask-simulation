"""Prepared final-SHA direct KKT regressions. Pure NumPy, never a simulator.

Must be invoked only after kernel owner declares the final SHA. This extends,
not replaces, the saved b115 exhaustive objective/projection oracle suite.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kernel',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    path=Path(args.kernel).resolve();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh independent direct-KKT evidence required')
    if sha(path)!=args.expected_sha:raise ValueError('Final kernel SHA differs')
    helper=ROOT/'review/verify_v13_joint_intent_qp_independent_v1.py'
    if sha(helper)!='9c53e7e95435221d087a3a84c532bb7c135c4588228e2b7b286cf157c90ff737':raise ValueError('Independent math definition/oracle changed')
    oracle=module_at(helper,'independent_qp_oracle_v1')
    kernel=module_at(path,'independent_direct_kkt_'+args.expected_sha[:12])
    checks=[];records=[]
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=clean(detail)))
    check('no_simulator_import','mujoco' not in sys.modules)
    def run(name,k):
        result=kernel.solve_intent_qp(**k)
        row=dict(name=name,status=result.status,available=result.available,iterations=result.iterations,objective=result.objective,diagnostics=result.diagnostics)
        records.append(row)
        if result.d21 is not None:
            x=np.r_[result.d21,result.zP];g=oracle.gradient(x,k)
            row.update(d21=result.d21,zP=result.zP,independent_gradient=g,
                       independent_objective=oracle.objective(x,k))
            check(name+'_original_objective',abs(row['independent_objective']-result.objective)<1e-9*max(1.,abs(result.objective)))
            check(name+'_original_residual',np.max(abs(result.residual25-(k['A21x25'].T@result.d21-k['Jpx25'].T@(result.zP-k['y_p']))),initial=0)<1e-12)
        else:x=g=None
        return result,x,g,row
    # Prior substantive cancellation witness. Honest unavailability is allowed;
    # available=True requires the actual analytic optimum / interior gradient.
    k=oracle.fixture();k['A21x25'][0,0]=1.;k['Jpx25'][0,1]=1.;k['scales']['residual25'][0]=2e8
    r,x,g,row=run('large_L_roundoff',k)
    check('large_L_no_false_available',not r.available or (x is not None and abs(r.zP[0]-.5)<1e-6 and abs(g[21])<1e-6),row)
    # All scales1; projected displacement is tiny because the domain is tiny.
    k=oracle.fixture();k['z_intent_p']=np.array([5e-13]);k['caps5'][0]=1e-12;k['total_budget']=1e-12;k['support_coeff_p']=np.ones(1);k['load_proxy_N']=1.
    r,x,g,row=run('narrow_z_cap_budget',k)
    check('narrow_z_no_strict_interior_false_stationarity',not r.available or (x is not None and (r.zP[0]==1e-12 or abs(g[21])<1e-7)),row)
    if x is not None:
        row['unit_step_mapping']=float(r.zP[0]-np.clip(r.zP[0]-g[21],0.,1e-12))
        row['objective_gap_to_upper']=float(r.objective-oracle.objective(np.r_[r.d21,1e-12],k))
    # Same narrow-domain issue on the effort box; positive smooth gradient2
    # drives a tiny negative lower bound, not an interior point.
    k=oracle.fixture();k['A21x25'][0,0]=1.;k['Jpx25'][0,0]=1.;k['y_p']=np.ones(1);k['z_intent_p']=np.zeros(1);k['caps5'][0]=0.;k['total_budget']=0.
    k['effort_lower21'][0]=-1e-12;k['effort_upper21'][0]=1e-12
    r,x,g,row=run('narrow_d_box',k)
    check('narrow_d_no_strict_interior_false_stationarity',not r.available or (x is not None and (r.d21[0]==-1e-12 or abs(g[0])<1e-7)),row)
    # A closed form with TWO positive group multipliers, active global budget,
    # active nonnegativity, both signs of effort-box bound and fixed originalp0.
    k=oracle.fixture(6);k['z_intent_p']=np.array([10.,9.,8.,7.,6.,0.]);k['patch_digit_p']=np.array([0,0,1,1,2,2]);k['caps5']=np.array([4.,4.,5.,10.,10.]);k['total_budget']=10.
    k['p0_21'][0]=5.;k['effort_lower21'][0]=3.;k['effort_upper21'][0]=4.
    k['p0_21'][1]=-5.;k['effort_lower21'][1]=-4.;k['effort_upper21'][1]=-3.
    r,x,g,row=run('multi_active_dual_complementarity',k)
    expected_z=np.array([2.5,1.5,2.5,1.5,2.,0.]);expected_d=np.zeros(21);expected_d[:2]=[-1.,1.]
    check('multi_active_exact_closed_form',r.available and np.max(abs(r.zP-expected_z))<1e-10 and np.max(abs(r.d21-expected_d))<1e-10)
    if x is not None:
        lower_dual=np.zeros(21);upper_dual=np.zeros(21);lower_dual[1]=2.;upper_dual[0]=2.
        group_dual=np.array([7.,3.,0.,0.,0.]);budget_dual=8.;nonnegative_dual=np.array([0.,0.,0.,0.,0.,8.])
        stat=np.r_[g[:21]-lower_dual+upper_dual,g[21:]+group_dual[k['patch_digit_p']]+budget_dual-nonnegative_dual]
        lo=k['effort_lower21']-k['p0_21'];hi=k['effort_upper21']-k['p0_21'];group_sums=np.bincount(k['patch_digit_p'],weights=r.zP,minlength=5)
        comp=np.r_[lower_dual*(r.d21-lo),upper_dual*(hi-r.d21),group_dual*(k['caps5']-group_sums),budget_dual*(k['total_budget']-r.zP.sum()),nonnegative_dual*r.zP]
        row['independent_known_dual']=dict(lower21=lower_dual,upper21=upper_dual,group5=group_dual,budget=budget_dual,nonnegativeP=nonnegative_dual)
        row['independent_stationarity_inf']=float(np.max(abs(stat)));row['independent_complementarity_inf']=float(np.max(abs(comp)))
        check('multi_active_independent_stationarity',np.max(abs(stat))<1e-9,row['independent_stationarity_inf'])
        check('multi_active_independent_complementarity',np.max(abs(comp))<1e-9,row['independent_complementarity_inf'])
        check('multi_active_fixed_p0_not_retargeted',np.array_equal(r.p21[:2],np.array([4.,-4.])))
    for name,mode in (('zero_group_cap','group'),('zero_total_budget','budget'),('fixed_effort_box','effort')):
        k=oracle.fixture(2);k['z_intent_p']=np.array([2.,1.]);k['patch_digit_p']=np.array([0,0])
        if mode=='group':k['caps5'][0]=0.
        elif mode=='budget':k['total_budget']=0.
        else:k['effort_lower21'][0]=k['effort_upper21'][0]=1.
        r,x,g,row=run(name,k)
        check(name+'_honest_available',r.available)
        if mode in ('group','budget'):check(name+'_zero_normal_exact',x is not None and np.array_equal(r.zP,np.zeros(2)))
        else:check(name+'_fixed_effort_exact',x is not None and r.d21[0]==1.)
    check('no_simulator_import_after_all','mujoco' not in sys.modules)
    if sha(path)!=args.expected_sha:raise ValueError('Kernel changed during final-SHA audit')
    output=dict(schema='v13_joint_direct_kkt_independent_v2',passed=all(c['passed'] for c in checks),checks=checks,records=records,
        failures=[c for c in checks if not c['passed']],kernel_path=str(path),kernel_sha256=args.expected_sha,checker_sha256=sha(__file__),
        physics_steps=0,simulator_imported='mujoco' in sys.modules,
        note='Complements the unchanged268check objective/projection suite. No tolerance/goal/physical threshold change; analytical dual values are independent test oracles, not production multipliers.')
    with out.open('x',encoding='utf-8') as stream:json.dump(clean(output),stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=output['passed'],checks=len(checks),failures=len(output['failures']),out=str(out),sha256=sha(out),physics_steps=0),ensure_ascii=False))

if __name__=='__main__':main()

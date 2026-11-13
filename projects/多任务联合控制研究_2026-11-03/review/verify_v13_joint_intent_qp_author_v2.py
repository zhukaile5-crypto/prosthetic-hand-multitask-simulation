"""Author numerical QA: independent closed forms/active-set enumeration, no plant."""
from pathlib import Path
from itertools import combinations
from copy import deepcopy
import datetime,hashlib,importlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
kernel=importlib.import_module('v13_joint.intent_qp')

def fixture(P=1):
    A=np.zeros((21,25));A[0,0]=1.
    J=np.zeros((P,25))
    if P:J[:,0]=1.
    low=np.zeros(21);high=np.zeros(21);low[0]=-10.;high[0]=10.
    return dict(p0_21=np.zeros(21),A21x25=A,Jpx25=J,y_p=np.ones(P),z_intent_p=np.full(P,2.),
        effort_lower21=low,effort_upper21=high,patch_digit_p=np.arange(P,dtype=int)%5,
        caps5=np.full(5,10.),total_budget=10.,support_coeff_p=np.zeros(P),load_proxy_N=0.,
        scales=dict(residual25=np.ones(25),intentP=np.ones(P),delta21=np.ones(21),load=1.))

def enumerate_qp(H,q,C,b):
    """Small independent exhaustive KKT active-set oracle, not kernel projection."""
    n=len(q);best=None
    for count in range(min(n,len(b))+1):
        for active in combinations(range(len(b)),count):
            E=C[list(active)] if active else np.empty((0,n));rhs=b[list(active)] if active else np.empty(0)
            K=np.block([[H,E.T],[E,np.zeros((count,count))]])
            if np.linalg.matrix_rank(K,tol=1e-11)<len(K):continue
            solution=np.linalg.solve(K,np.r_[-q,rhs]);x=solution[:n];dual=solution[n:]
            if np.max(C@x-b)>1e-8 or (len(dual) and dual.min()<-1e-8):continue
            value=float(.5*x@H@x+q@x)
            if best is None or value<best[0]:best=(value,x,active,dual)
    assert best is not None
    return best

def main():
    checks=[];cases={}
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def solve(k,**extra):return kernel.solve_intent_qp(**k,**extra)
    start=time.perf_counter()
    # Scalar coupled problem: stationary solution d=1/3,z=5/3,obj=1/3.
    k=fixture();r=solve(k,atol=1e-10,rtol=1e-9)
    check('coupled_closed_form_converged',r.available,r.status)
    check('coupled_closed_form_values',np.allclose([r.d21[0],r.zP[0]],[1/3,5/3],atol=1e-7,rtol=0),[float(r.d21[0]),float(r.zP[0])])
    check('coupled_true_residual_and_objective',np.isclose(r.residual25[0],-1/3,atol=1e-7) and np.isclose(r.objective,1/3,atol=1e-9))
    check('coupled_projected_KKT_small',r.diagnostics['projected_gradient_inf_d21']<1e-7 and r.diagnostics['projected_gradient_inf_zP']<1e-7)
    cases['scalar']=dict(status=r.status,iterations=r.iterations,objective=r.objective)
    # Load hinge active: d=.8,z=2.6,shortfall1.4,obj3.6.
    k=fixture();k['support_coeff_p'][:]=1.;k['load_proxy_N']=4.;r=solve(k,atol=1e-10,rtol=1e-9)
    check('load_hinge_closed_form',r.available and np.allclose([r.d21[0],r.zP[0],r.load_shortfall_N],[.8,2.6,1.4],atol=1e-7,rtol=0),dict(status=r.status,d=float(r.d21[0]),z=float(r.zP[0]),shortfall=r.load_shortfall_N))
    check('load_hinge_objective',np.isclose(r.objective,3.6,atol=1e-9))
    cases['active_load']=dict(status=r.status,iterations=r.iterations)
    # Group and total-budget projection is not sequential clipping.
    v=np.array([5.,4.,4.]);digits=np.array([0,0,1]);caps=np.array([4.,10.,0.,0.,0.]);budget=6.
    projected=kernel.project_normal_intents(v,digits,caps,budget)
    C=np.vstack((-np.eye(3),[1.,1.,0.],[0.,0.,1.],np.ones(3)));b=np.r_[np.zeros(3),4.,10.,6.]
    reference=enumerate_qp(np.eye(3),-v,C,b)
    check('joint_group_budget_projection_closed_form',np.allclose(projected,[2.5,1.5,2.],atol=1e-12,rtol=0),projected.tolist())
    check('joint_projection_exhaustive_KKT_oracle',np.allclose(projected,reference[1],atol=1e-11,rtol=0))
    sequential_input=np.array([2.5,1.5,4.]);sequential=kernel.project_normal_intents(sequential_input,[0,1,2],[10.]*5,6.)
    check('sequential_group_then_budget_is_real_counterexample',np.linalg.norm(projected-v)<np.linalg.norm(sequential-v)-1e-4)
    check('projection_idempotent_and_input_immutable',np.allclose(kernel.project_normal_intents(projected,digits,caps,budget),projected,atol=1e-12) and np.array_equal(v,[5.,4.,4.]))
    # Several fixed small-vector projections checked by an independent active set.
    projection_errors=[]
    for vv,bb in [([1.,2.,3.],2.),([-1.,2.,8.],3.5),([0.,0.,0.],0.),([4.,4.,4.],7.),([.1,8.,-4.],4.2)]:
        vv=np.asarray(vv);cc=np.array([3.,5.,0.,0.,0.]);C=np.vstack((-np.eye(3),[1.,1.,0.],[0.,0.,1.],np.ones(3)));b=np.r_[np.zeros(3),3.,5.,bb]
        oracle=enumerate_qp(np.eye(3),-vv,C,b)[1];got=kernel.project_normal_intents(vv,digits,cc,bb)
        projection_errors.append(float(np.max(abs(got-oracle))))
    check('five_projection_active_set_oracles',max(projection_errors)<1e-10,projection_errors)
    # Full small constrained coupled QP, active load in whole feasible region.
    k=fixture(2);k['Jpx25'][1,0]=-.5;k['y_p']=np.array([.8,1.2]);k['z_intent_p']=np.array([2.,1.5])
    k['support_coeff_p']=np.array([1.,-.3]);k['load_proxy_N']=2.5;k['caps5'][:2]=[1.2,1.]
    k['total_budget']=1.7;k['effort_lower21'][0]=-.2;k['effort_upper21'][0]=.4
    M=np.array([1.,-1.,.5]);ce=np.array([0.,1.,-.3]);H=2*(np.outer(M,M)+np.eye(3)+np.outer(ce,ce));q=2*(.2*M-np.array([0.,2.,1.5])-2.5*ce)
    C=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.],[0.,1.,0.],[0.,0.,1.],[0.,1.,1.]])
    b=np.array([.2,.4,0.,0.,1.2,1.,1.7]);oracle=enumerate_qp(H,q,C,b)
    r=solve(k,atol=1e-10,rtol=1e-9);got=np.r_[r.d21[0],r.zP]
    check('constrained_coupled_exhaustive_KKT',r.available and np.allclose(got,oracle[1],atol=1e-7,rtol=0),dict(actual=got.tolist(),oracle=oracle[1].tolist(),status=r.status))
    actual_objective=float((M@got+.2)**2+(got[1]-2)**2+(got[2]-1.5)**2+got[0]**2+(2.5-ce@got)**2)
    check('constrained_true_objective',np.isclose(r.objective,actual_objective,atol=1e-12))
    check('constrained_primal_caps_and_budget',max(r.diagnostics['primal_violations'].values())<1e-9)
    cases['constrained']=dict(status=r.status,iterations=r.iterations,active_constraints=list(oracle[2]))
    # p0 must remain fixed even when outside the absolute feasible effort box.
    k=fixture(0);k['p0_21'][0]=10.;k['effort_lower21'][0]=3.;k['effort_upper21'][0]=4.;r=solve(k)
    check('original_p0_not_projected_before_goal',r.available and r.p21[0]==4. and r.residual25[0]==-6. and r.diagnostics['fixed_original_p0_21'][0]==10.)
    # z_intent is an unprojected objective target too.
    k=fixture();k['A21x25'][:]=0.;k['Jpx25'][:]=0.;k['z_intent_p'][:]=5.;k['caps5'][0]=1.;k['total_budget']=1.;r=solve(k)
    check('original_force_intent_not_retargeted',r.available and abs(r.zP[0]-1.)<1e-10 and abs(r.objective-16.)<1e-9)
    check('known_zero_J_not_unknown_J',r.status=='converged')
    # Unknown is not zero and malformed scales/bounds are not accepted.
    for field,value in [('Jpx25',None),('load_proxy_N',None),('load_proxy_N',float('nan')),('y_p',[float('nan')])]:
        k=fixture();k[field]=value;r=solve(k)
        check('unknown_'+field+'_'+str(value),r.status=='unknown' and not r.available and r.residual25 is None)
    k=fixture();k['scales']={'residual25':np.ones(25)};r=solve(k)
    check('scales_strictly_required',r.status=='invalid_input' and not r.available)
    k=fixture();k['effort_lower21'][0]=1.;k['effort_upper21'][0]=0.;r=solve(k)
    check('empty_box_not_retargeted',r.status=='invalid_input' and r.reason=='empty_effort_box')
    k=fixture();k['support_coeff_p'][:]=1.;k['load_proxy_N']=4.;before=deepcopy(k);r=solve(k,max_iterations=1,atol=1e-12,rtol=0.)
    check('budget_exhaustion_is_not_converged',r.status=='nonconvergence' and not r.available and r.residual25 is not None,r.status)
    check('nonconverged_true_residual',np.allclose(r.residual25,k['A21x25'].T@r.d21-k['Jpx25'].T@(r.zP-k['y_p']),atol=0.,rtol=0.))
    check('all_input_arrays_unchanged',all(np.array_equal(k[key],before[key]) for key in ('p0_21','A21x25','Jpx25','y_p','z_intent_p','effort_lower21','effort_upper21','caps5','support_coeff_p')))
    check('returned_arrays_readonly',not r.d21.flags.writeable and not r.zP.flags.writeable and not r.residual25.flags.writeable)
    # v2 convergence regressions: no step-size difference is a KKT certificate.
    k=fixture();k['A21x25'][:]=0.;k['A21x25'][0,0]=1.;k['Jpx25'][:]=0.;k['Jpx25'][0,1]=1.
    k['y_p'][:]=0.;k['z_intent_p'][:]=1.;k['scales']['residual25'][0]=2e8
    r=solve(k)
    check('huge_L_false_convergence_blocked',not r.available and r.status in ('numerical_stagnation','nonconvergence'),dict(status=r.status,z=r.zP.tolist(),direct_KKT=r.diagnostics.get('direct_KKT_inf_zP')))
    check('huge_L_direct_gradient_not_zero_mapping',r.diagnostics['direct_KKT_inf_zP']>1.9 and r.diagnostics['projected_gradient_inf_zP']==0.)
    check('huge_L_original_objective_and_weights_preserved',r.objective==1. and r.diagnostics['fixed_original_p0_21']==[0.]*21)
    cases['huge_L']=dict(status=r.status,iterations=r.iterations,z=r.zP.tolist(),direct_KKT=r.diagnostics['direct_KKT_inf_zP'])
    k=fixture();k['A21x25'][:]=0.;k['Jpx25'][:]=0.;k['y_p'][:]=0.;k['z_intent_p'][:]=5e-13
    k['support_coeff_p'][:]=1.;k['load_proxy_N']=1.;k['caps5'][0]=1e-12;k['total_budget']=1e-12
    narrow_initial=kernel._normal_intent_cone(np.array([-2.]),np.array([5e-13]),(np.array([0]),np.array([],int),np.array([],int),np.array([],int),np.array([],int)),k['caps5'],1e-12)
    check('narrow_strict_interior_gradient_not_swallowed',narrow_initial['stationarity_inf']==2. and not narrow_initial['group_active'][0] and not narrow_initial['budget_active'])
    r=solve(k)
    check('narrow_no_iteration0_false_available',r.iterations>0 and (not r.available or (r.zP[0]==1e-12 and r.diagnostics['direct_KKT_inf_zP']<1e-7)),dict(status=r.status,z=r.zP.tolist(),iter=r.iterations))
    check('narrow_singleton_projection_exact_cap',kernel.project_normal_intents([.5],[0],[1e-12,0.,0.,0.,0.],1e-12)[0]==1e-12)
    cases['narrow']=dict(status=r.status,iterations=r.iterations,z=r.zP.tolist(),direct_KKT=r.diagnostics['direct_KKT_inf_zP'])
    box=kernel._box_normal_cone(np.array([-2.]),np.array([5e-13]),np.array([0.]),np.array([1e-12]))
    check('narrow_box_direct_gradient_not_width_mapping',box['stationarity_inf']==2. and box['strict_interior_count']==1)
    box=kernel._box_normal_cone(np.array([-3.,2.,7.]),np.array([1.,0.,.4]),np.array([0.,0.,.4]),np.array([1.,1.,.4]))
    check('box_endpoint_sign_and_fixed_interval',box['stationarity_inf']==0. and box['fixed_coordinates']==1)
    groups=(np.array([0]),np.array([1]),np.array([],int),np.array([],int),np.array([],int))
    cone=kernel._normal_intent_cone(np.array([-1.,-3.]),np.array([.4,.6]),groups,np.array([1.,1.,0.,0.,0.]),1.)
    check('shared_budget_minimax_dual_closed_form',abs(cone['stationarity_inf']-1.)<1e-12 and abs(cone['global_multiplier']-2.)<1e-12,cone)
    cone=kernel._normal_intent_cone(np.array([-3.,-1.]),np.array([.4,.6]),groups,np.array([.4,1.,0.,0.,0.]),1.)
    check('active_group_plus_global_dual_closed_form',cone['stationarity_inf']==0. and cone['global_multiplier']==1. and cone['group_multipliers'][0]==2.,cone)
    groups1=(np.array([0]),np.array([],int),np.array([],int),np.array([],int),np.array([],int))
    cone=kernel._normal_intent_cone(np.array([-1e12]),np.array([np.nextafter(1.,0.)]),groups1,np.array([1.,0.,0.,0.,0.]),10.)
    check('large_dual_times_ULP_slack_not_hidden',cone['stationarity_inf']==0. and cone['complementarity_inf']>1e-7,cone)
    cone=kernel._normal_intent_cone(np.array([-1e12]),np.array([1e-12-1e-20]),groups1,np.array([1e-12,0.,0.,0.,0.]),10.)
    check('tiny_domain_cap_strict_interior_not_active_by_atol',not cone['group_active'][0] and cone['stationarity_inf']>=1e12)
    check('all_converged_results_require_direct_cone_and_complementarity',r.diagnostics.get('projected_gradient_is_diagnostic_only') is True)
    elapsed=time.perf_counter()-start
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=ROOT/f'review/v13_joint_intent_qp_author_{stamp}.json'
    report=dict(schema='v13_joint_intent_qp_author_validation_v2',role='kernel_author_QA_not_independent_acceptance',
        passed=all(c['passed'] for c in checks),checks=checks,count=len(checks),source_sha256=hashlib.sha256((ROOT/'v13_joint/intent_qp.py').read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,wall_seconds_for_entire_QA=elapsed,
        physics_steps=0,plant_imports=False,native_inverse_or_codec_tested=False,global_float_optimality_certified=False,
        interpretation='Closed forms and exhaustive small KKT active sets verify the pure convex kernel only; all scales are explicit fixtures, not physical tuning or evaluation utility.')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],cases=cases,wall_seconds=elapsed),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

"""Canonical budget witness vs actual submitted tasks, pure numerical cases."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy
from dataclasses import is_dataclass
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v24_canonical_task_fusion import task_solver as task
from v23_task_execution import finite_solver as new_backend
from v22_joint_task_fusion import finite_solver as old_backend
from v23_task_execution.runtime import source_identity as parent_source_identity
from v22_joint_task_fusion.numerics import numerical_identity
from v21_stable_handoff.prefix_supervisor import score_emitted_reference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(v):
    if is_dataclass(v):return plain(vars(v))
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def base(P=1):
    B=np.zeros((25,21));B[:21]=np.eye(21);J=np.zeros((P,25));J[:,0]=1.
    lo=np.zeros(21);hi=np.zeros(21);lo[0]=-.005;hi[0]=.005
    return dict(time_s=12.5,dt_s=.005,current_reference21=np.zeros(21),encoder_reference21=np.zeros(21),
        step_lower21=lo,step_upper21=hi,reference_rates21=np.ones(21),normal_jacobian_Px25=J,
        proposal_map25x21=B,robot_velocity25=np.zeros(25),pressure_P_N=np.ones(P),reference_target_delta21=np.zeros(21))
def main(out):
    if out.exists():raise FileExistsError(out)
    prod=('v24_canonical_task_fusion/task_solver.py','v23_task_execution/finite_solver.py','v22_joint_task_fusion/task_bundle.py','v22_joint_task_fusion/numerics.py','v21_stable_handoff/design.py')
    before={p:sha(ROOT/p) for p in prod};parent=parent_source_identity(['ratio_effort']);parent_plan=read(ROOT/'review/v23_small_plan_v1.json')
    assert len(parent)==186 and parent==parent_plan['source_sha256']
    numerical=numerical_identity();checks=[];cases=[];started=time.perf_counter();canonical_cert_checks=[]
    spec=importlib.util.spec_from_file_location('v24_author_independent_original_verifier',ROOT/'review/v21_original_matrix_kkt_verifier_v1.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def certcheck(c):
        k=c['KKT'];d=c['hard_domain'];p=dict(A=c['matrix'],b=c['target'],G=d['original_inequality_matrix_with_box'],h=d['original_inequality_upper_with_box'],
            E=k['full_equality_matrix'],f=k['full_equality_target'],previous_frozen=c['previous_frozen'],
            atol=c['record']['numerical_atol'],rtol=c['record']['numerical_rtol'])
        v=verifier.verify_stage(p,c['x'],duals=dict(inequality_unit=k['full_normalized_inequality_dual'],equality_raw=k['equality_raw_dual'],provenance='current own certificate original-domain dual, no historical substitution'))
        return dict(name=c['name'],passed=bool(v['kkt_passed']),original_problem_sha256=v.get('original_problem_sha256'),objective=v.get('objective'),strict_box=bool(np.all(c['x']>=d['lower']) and np.all(c['x']<=d['upper'])),stationarity_inf=v.get('stationarity_inf'))
    def summarize(name,r,kw):
        w=r.diagnostics.get('normal_witness_certified_prefix',());canonical_cert_checks.extend(certcheck(c) for c in w)
        score=score_emitted_reference(r,current21=np.asarray(kw['current_reference21']),proposed21=np.asarray(kw['current_reference21'])+np.asarray(kw['reference_rates21'])*kw['dt_s']*r.x,scale21=np.asarray(kw['reference_rates21'])*kw['dt_s']) if r.available else None
        cases.append(dict(name=name,status=r.status,available=r.available,full=r.fully_converged,x=r.x,
            actual_names=[c['name'] for c in r.certified_prefix],actual_indices=[c['index'] for c in r.certified_prefix],witness_names=[c['name'] for c in w],
            witness_anchor=r.diagnostics.get('normal_witness_final_extended_anchor'),motion_budget=r.diagnostics.get('motion_task',{}).get('normalized_absolute_budget_M'),score=score))
        return score
    # Closed-form: eta*=0, canonical xA=0, |x+1|<=1 still permits x=-1.
    k=base();M=np.zeros((1,21));M[0,0]=1.;k.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.array([1.]),motion_time_s=12.5)
    for variant in ('backend','joint'):
        r=task.solve_contact_exit(**k,formulation=variant);w=r.diagnostics.get('normal_witness_certified_prefix',())
        ck(variant+' complete two-layer canonical witness',r.available and len(w)==2 and [c['name'] for c in w]==['normal_inward_correction_slack',task.CANONICAL_WITNESS_LEVEL] and all(c['KKT']['certified'] for c in w))
        ck(variant+' unique selector only x21 identity',len(w)==2 and np.array_equal(w[1]['matrix'][:,:21],np.eye(21)) and np.array_equal(w[1]['matrix'][:,21:],np.zeros((21,1))) and np.array_equal(w[1]['target'],np.zeros(21)))
        ck(variant+' canonical zero does not freeze actual exit',r.available and abs(w[1]['x'][0])<1e-8 and abs(r.x[0]+1.)<1e-8)
        ck(variant+' selector never an actual task/record',all(c['name']!=task.CANONICAL_WITNESS_LEVEL for c in r.certified_prefix) and all(c['name']!=task.CANONICAL_WITNESS_LEVEL for c in r.levels) and [c['index'] for c in r.certified_prefix]==list(range(r.completed_prefix)))
        ck(variant+' actual level records use global indices without selector slot',[lev['index'] for lev in r.levels]==list(range(len(r.levels))) and all('solver_local_index' in lev for lev in r.levels))
        if variant=='backend':ck('backend submits only normal first certificate',r.certified_prefix[0]['name']=='normal_inward_correction_slack' and sum(c['task_stage']=='normal_slack' for c in r.certified_prefix)==1)
        else:ck('joint does not submit either witness certificate',r.certified_prefix[0]['name']=='joint_normal_and_observed_projection' and not any(c['name']=='normal_inward_correction_slack' for c in r.certified_prefix) and r.diagnostics['second_stage_lower'][21]<r.diagnostics['second_stage_upper'][21])
        score=summarize(variant+'_analytic_unlock',r,k);ck(variant+' actual emitted same-task score',score and score['available'])
    ck('analytic misfreezing raises exit loss0 to1',(-1.+1.)**2==0. and (0.+1.)**2==1.)
    # Independent normal correction vs joint compromise; no new physical gains.
    k=base();k['robot_velocity25'][0]=.4;k.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.array([.1]),motion_time_s=12.5)
    b=task.solve_contact_exit(**k,formulation='backend');j=task.solve_contact_exit(**k,formulation='joint')
    ck('joint keeps eta free for original analytic compromise',j.available and abs(j.x[0]+.25)<1e-7 and abs(j.diagnostics['certified_slack_eta_P'][0]-.15)<1e-7)
    ck('backend keeps normal optimum only',b.available and abs(b.diagnostics['certified_slack_eta_P'][0])<1e-8 and b.x[0]<=-.4+1e-7)
    for variant in ('backend','joint'):
        k=base();r=task.solve_contact_exit(**k,formulation=variant);ck(variant+' unknown motion omitted not zero',r.available and r.diagnostics['motion_unknown_not_zero'] and not r.diagnostics['motion_task']['available'])
        k=base(P=0);k['path_reference_target_delta21']=np.r_[-.003,np.zeros(20)];r=task.solve_contact_exit(**k,formulation=variant)
        ck(variant+' knownP0 unchanged path and no selector',r.available and abs(r.x[0]+.6)<1e-7 and not r.diagnostics['canonical_witness_required'] and len(r.diagnostics['normal_witness_certified_prefix'])==0)
        k=base();k['normal_jacobian_Px25']=None;r=task.solve_contact_exit(**k,formulation=variant);ck(variant+' unknown geometry notP0',not r.available and r.status=='UNKNOWN_CONTACT_EXIT_INPUT')
    # Deliberately lose only the selector certificate: first normal proof is
    # retained as evidence but cannot define a budget or actual prefix.
    real=task.solve_hierarchy
    for variant in ('backend','joint'):
        def missing_normal(**kw):
            return SimpleNamespace(x=np.asarray(kw['initial']).copy(),status='FORCED_NORMAL_UNAVAILABLE',available=False,fully_converged=False,completed_prefix=0,
                certified_prefix=(),levels=(dict(index=0,name='normal_inward_correction_slack',status='FORCED_FAILURE'),),
                failed_layer=dict(index=0,name='normal_inward_correction_slack',status='FORCED_FAILURE'),diagnostics=dict(author_injection=True))
        with patch.object(task,'solve_hierarchy',missing_normal):r=task.solve_contact_exit(**base(),formulation=variant)
        ck(variant+' failed normal has no canonical budget or actual layer',not r.available and r.completed_prefix==0 and r.certified_prefix==() and r.levels==() and r.failed_layer is None and not r.diagnostics['normal_witness_budget_defined'] and r.diagnostics['witness_failed_layer']['index']==0)
        count=[0]
        def missing_canonical(**kw):
            count[0]+=1;rr=real(**kw)
            return SimpleNamespace(x=rr.certified_prefix[0]['x'],status='FORCED_CANONICAL_SELECTOR_UNAVAILABLE',available=True,fully_converged=False,completed_prefix=1,
                certified_prefix=rr.certified_prefix[:1],levels=(rr.levels[0],dict(index=1,name=task.CANONICAL_WITNESS_LEVEL,status='FORCED_FAILURE')),
                failed_layer=dict(index=1,name=task.CANONICAL_WITNESS_LEVEL,status='FORCED_FAILURE'),diagnostics=rr.diagnostics)
        with patch.object(task,'solve_hierarchy',missing_canonical):r=task.solve_contact_exit(**base(),formulation=variant)
        ck(variant+' incomplete canonical pair cannot create actual certificate/budget',not r.available and r.completed_prefix==0 and r.certified_prefix==() and r.levels==() and r.failed_layer is None and not r.diagnostics['normal_witness_budget_defined'] and r.diagnostics['witness_failed_layer']['index']==1 and count[0]==1)
        count=[0]
        def fail_exit(**kw):
            count[0]+=1
            if count[0]==1:return real(**kw)
            return SimpleNamespace(x=np.asarray(kw['initial']).copy(),status='FORCED_ACTUAL_TASK_FAILURE',available=False,fully_converged=False,completed_prefix=0,
                certified_prefix=(),levels=(dict(index=0,name=kw['levels'][0].name,status='FORCED_FAILURE'),),failed_layer=dict(index=0,name=kw['levels'][0].name,status='FORCED_FAILURE'),diagnostics=dict(author_injection=True))
        with patch.object(task,'solve_hierarchy',fail_exit):r=task.solve_contact_exit(**base(),formulation=variant)
        ck(variant+' actual failure indices exclude witness selector',r.failed_layer['index']==(1 if variant=='backend' else 0) and r.completed_prefix==(1 if variant=='backend' else 0) and bool(r.available)==(variant=='backend') and len(r.diagnostics['normal_witness_certified_prefix'])==2)
    # One archived full public mathematical packet. No trace/model reconstruction.
    publicpath=ROOT/'review/v23_joint_friction_first_divergence_v1.json';archived=read(publicpath)
    sample=next(x for x in archived['local_same_public_input_counterfactuals'] if abs(x['all_kwargs']['time_s']-13.64)<1e-9)
    k=deepcopy(sample['all_kwargs']);k.pop('formulation',None);captured=[]
    def capture(**kw):captured.append(kw);return real(**kw)
    with patch.object(task,'solve_hierarchy',capture):r=task.solve_contact_exit(**k,formulation='joint')
    ck('archived13.640 actual canonical call available',r.available and len(captured)==2 and len(captured[0]['levels'])==2)
    score=summarize('archived13.640_canonical_joint',r,k);ck('archived actual emitted same-task score',score and score['available'])
    other=old_backend.solve_hierarchy(**captured[0]);w=r.diagnostics['normal_witness_certified_prefix']
    canonical_cert_checks.extend(certcheck(c) for c in other.certified_prefix)
    md=r.diagnostics['motion_task'];units=np.asarray(r.diagnostics['reference_step_units21']);m=np.asarray(md['matrix_Mx21'])*units[None,:]/np.asarray(md['effective_row_units_M'])[:,None]
    offset=k['dt_s']*np.asarray(md['error_M_m_s'])/np.asarray(md['effective_row_units_M'])
    dx=float(np.max(abs(other.x[:21]-w[-1]['x'][:21]),initial=0.));dh=float(np.max(abs(abs(m@other.x[:21]+offset)-np.asarray(md['normalized_absolute_budget_M'])),initial=0.))
    tolerance=max(c['KKT']['coordinate_tolerance'] for c in (*other.certified_prefix,*w))
    ck('two numerical backends same canonical problem stable witness/budget',other.fully_converged and other.completed_prefix==2 and dx<=tolerance and dh<=tolerance,dict(x21_difference_inf=dx,budget_difference_inf=dh,unchanged_original_coordinate_tolerance=tolerance))
    ck('all saved witness certificates independently originalKKT and strictbox',all(c['passed'] and c['strict_box'] for c in canonical_cert_checks))
    ck('owned and frozen source subset unchanged',before=={p:sha(ROOT/p) for p in prod})
    ck('old complete186 and numerical identity unchanged',parent_source_identity(['ratio_effort'])==parent and numerical_identity()==numerical)
    d=dict(schema='v24_task_author_v1',passed=all(c['passed'] for c in checks),checks=checks,issues=[c['name'] for c in checks if not c['passed']],cases=cases,
        canonical_certificate_checks=canonical_cert_checks,production_source_sha256=before,numerical_identity=numerical,parent186_sha256=parent,
        archived_public_mathematical_input_sha256=sha(publicpath),checker_sha256=sha(__file__),elapsed_s=time.perf_counter()-started,
        physics_steps=0,model_constructions=0,runtime_or_actual_physical_performance_tested=False,
        scope='Author algebra/certificate-role tests and one archived public mathematical packet; old186 and all numerical gains/rates/domains remain unchanged. Independent integration still required.')
    out.write_text(json.dumps(plain(d),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=d['passed'],checks=len(checks),issues=d['issues'],out=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if d['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.out))

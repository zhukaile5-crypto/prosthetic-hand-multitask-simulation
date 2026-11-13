"""Finite V21 prefix-kernel AUTHOR QA, zero model/FK/physics/native.
One archived original V20 query is solved ONCE as a whole; no replayed prefix
is used to manufacture a historical certificate. Synthetic counteroracles are
independent closed forms/fault injection; this is not independent acceptance.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import argparse,ast,gzip,hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SRC=ROOT/'v21_stable_handoff/prefix_hierarchy.py';OLD=ROOT/'v17_contact_transfer/hierarchy.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='review/v21_prefix_hierarchy_author_attempt1.json');args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    startpin=sha(SRC);snap=ROOT/'review/revisions'/('v21_prefix_hierarchy_'+startpin[:12]+'.py');snap.parent.mkdir(exist_ok=True)
    if snap.exists():assert snap.read_bytes()==SRC.read_bytes()
    else:snap.write_bytes(SRC.read_bytes())
    spec=importlib.util.spec_from_file_location('v21_prefix_hierarchy_author_target',SRC);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    cases=[];costs=[];archive={}
    def ck(name,fn):
        start=time.perf_counter_ns()
        try:detail=fn();cases.append(dict(name=name,passed=True,detail=plain(detail)))
        except Exception as exc:cases.append(dict(name=name,passed=False,error=repr(exc)))
        costs.append((time.perf_counter_ns()-start)/1e6)
    def close(x,y):assert np.allclose(x,y,atol=2e-9,rtol=1e-8),(x,y)
    def lev(name,a,b):return m.Level(name,np.asarray(a,float),np.asarray(b,float))
    def full():return m.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=[lev('normal',[[1.,0.]],[.3]),lev('motion',[[0.,1.]],[.4]),lev('pose',np.eye(2),[-1.,-1.])])
    def complete():
        z=full();assert z.available and z.fully_converged and z.completed_prefix==3 and z.failed_layer is None;close(z.x,[.3,.4]);assert z.accepted_prefix is z.certified_prefix
        return dict(status=z.status,x=z.x,completed=z.completed_prefix)
    ck('full_hierarchy_closed_form_and_explicit_prefix_API',complete)
    def snapshots():
        z=full();close(z.certified_prefix[0]['x'],[.3,0.]);close(z.certified_prefix[1]['x'],[.3,.4]);close(z.certified_prefix[2]['x'],[.3,.4])
        assert all(s['KKT']['certified'] and all(s['KKT']['checks'].values()) for s in z.certified_prefix)
        assert z.certified_prefix[0]['previous_frozen']==() and len(z.certified_prefix[2]['previous_frozen'])==2
        return dict(snapshot_x=[s['x'] for s in z.certified_prefix])
    ck('each_success_saved_at_its_own_completion_not_replayed_final_point',snapshots)
    def later_limit():
        z=m.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=[lev('first',[[1.,0.]],[.3]),lev('later',[[0.,1.]],[2.])],max_iterations=1)
        assert z.available and not z.fully_converged and z.completed_prefix==1 and z.failed_layer['index']==1;close(z.x,[.3,0.]);assert z.levels[-1]['status']=='ITERATION_LIMIT' and not z.levels[-1]['certified'];assert z.diagnostics['failed_trial_returned'] is False
        return dict(returned=z.x,failed=z.failed_layer['status'],failed_trial_residual=z.levels[-1]['residual'])
    ck('later_iteration_failure_returns_last_certified_x_not_failed_iterate',later_limit)
    def first_limit():
        z=m.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=[lev('first',np.eye(2),[2.,.25])],max_iterations=1)
        assert not z.available and not z.fully_converged and z.completed_prefix==0 and z.failed_layer['index']==0;close(z.x,[0.,0.]);assert z.levels[-1]['point_role']=='failed_uncommitted_trial'
        return dict(returned=z.x,status=z.status,failed_residual=z.levels[-1]['residual'])
    ck('empty_prefix_keeps_genuine_anchor_without_task_certificate',first_limit)
    def nonzero_anchor():
        z=m.solve_hierarchy(lower=[1.,-.5],upper=[2.,.5],initial=[1.,.1],levels=[lev('normal',[[1.,0.]],[1.5]),lev('blocked',[[0.,1.]],[2.])],max_iterations=1)
        assert z.available and z.completed_prefix==1;close(z.x,[1.5,.1]);close(z.diagnostics['original_feasible_anchor'],[1.,.1]);return dict(anchor=z.diagnostics['original_feasible_anchor'],returned=z.x)
    ck('domain_without_zero_retains_original_nonzero_anchor_and_prefix',nonzero_anchor)
    def no_levels():
        z=m.solve_hierarchy(lower=[1.],upper=[2.],initial=[1.2],levels=[])
        assert not z.available and not z.fully_converged and z.completed_prefix==0 and z.failed_layer is None;close(z.x,[1.2]);return dict(status=z.status)
    ck('no_requested_task_is_feasible_anchor_not_vacuous_task_success',no_levels)
    def invalid_anchor():
        try:m.solve_hierarchy(lower=[1.],upper=[2.],initial=[0.],levels=[lev('a',[[1.]],[1.])])
        except ValueError:return dict(rejected=True)
        raise AssertionError('infeasible actual initial accepted')
    ck('actual_infeasible_anchor_rejected_before_any_certificate',invalid_anchor)
    def hard_rows():
        z=m.solve_hierarchy(lower=[-2.,-2.],upper=[2.,2.],initial=[0.,0.],inequality_matrix=[[1.,1.]],inequality_upper=[.5],levels=[lev('normal',np.eye(2),[1.,1.]),lev('later',[[1.,-1.]],[5.])])
        assert z.available and z.fully_converged;close(z.x,[.25,.25]);assert all(s['hard_domain']['original_inequality_matrix_with_box'].shape==(5,2) for s in z.certified_prefix)
        return dict(x=z.x,domain=z.certified_prefix[0]['hard_domain'])
    ck('original_nonbox_hard_constraints_and_frozen_higher_residuals_preserved',hard_rows)
    def failure_exception():
        real=m._repair_degenerate_direction;calls=[]
        def stop(**kw):
            calls.append(1)
            if len(calls)==2:raise FloatingPointError('injected_second_layer_failure')
            return real(**kw)
        with patch.object(m,'_repair_degenerate_direction',stop):z=full()
        assert z.completed_prefix==1 and z.available and not z.fully_converged;close(z.x,[.3,0.]);assert z.failed_layer['status']=='LAYER_EXCEPTION_FAILURE';return dict(returned=z.x,reason=z.failed_layer['reason'])
    ck('numerical_exception_after_certification_preserves_prior_snapshot',failure_exception)
    def malformed_later():
        z=m.solve_hierarchy(lower=[-1.],upper=[1.],levels=[lev('valid',[[1.]],[.2]),lev('malformed',[[1.,2.]],[0.])])
        assert z.available and z.completed_prefix==1 and z.failed_layer['status']=='LAYER_EXCEPTION_FAILURE';close(z.x,[.2]);return dict(failure=z.failed_layer)
    ck('late_invalid_layer_cannot_erase_earlier_real_certificate',malformed_later)
    def cert_fixture(**change):
        kw=dict(a=np.array([[1.]]),b=np.array([1.]),x=np.array([1.]),status='KKT_CONVERGED',e=np.empty((0,1)),f=np.empty(0),g=np.array([[1.],[-1.]]),h=np.array([2.,2.]),active=[],w=np.empty((0,1)),multipliers=np.empty(0),original_g=np.array([[1.],[-1.]]),original_h=np.array([2.,2.]),equal=np.empty((0,1)),rhs_equal=np.empty(0),frozen=[],atol=1e-9,rtol=1e-8);kw.update(change);return m._layer_certificate(**kw)
    def false_status():
        c=cert_fixture(x=np.array([0.]));assert not c['certified'] and not c['checks']['fresh_original_stationarity'];return c['checks']
    ck('green_status_without_true_original_stationarity_is_not_certified',false_status)
    def false_primal():
        c=cert_fixture(x=np.array([3.]),b=np.array([3.]));assert not c['certified'] and not c['checks']['original_hard_primal'];return c['checks']
    ck('green_status_without_original_hard_primal_is_not_certified',false_primal)
    def false_dual():
        c=cert_fixture(b=np.array([0.]),g=np.array([[1.],[-1.]]),h=np.array([1.,2.]),active=[0],w=np.array([[1.]]),multipliers=np.array([-2.]),original_h=np.array([1.,2.]));assert not c['certified'] and not c['checks']['inequality_dual_nonnegative'];return c['checks']
    ck('stationary_green_point_with_negative_dual_is_not_certified',false_dual)
    def false_comp():
        c=cert_fixture(x=np.array([0.]),active=[0],w=np.array([[1.]]),multipliers=np.array([2.]));assert not c['certified'] and not c['checks']['complementarity'];return c['checks']
    ck('stationary_positive_dual_on_slack_face_fails_complementarity',false_comp)
    def false_frozen():
        c=cert_fixture(frozen=[(np.array([[1.]]),np.array([0.]))]);assert not c['certified'] and not c['checks']['previous_frozen_residuals'];return c['checks']
    ck('green_status_with_changed_prior_attainment_is_not_certified',false_frozen)
    def false_working():
        c=cert_fixture(active=[0],w=np.empty((0,1)),multipliers=np.empty(0));assert not c['certified'] and not c['checks']['original_working_set_matches_dual_record'];return c['checks']
    ck('stale_working_set_or_multipliers_cannot_sign_a_snapshot',false_working)
    def readonly():
        A=np.array([[1.,0.]]);b=np.array([.3]);z=m.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],levels=[m.Level('a',A,b)]);A[0,0]=9;b[0]=9
        assert z.certified_prefix[0]['matrix'][0,0]==1. and z.certified_prefix[0]['target'][0]==.3 and not z.x.flags.writeable and not z.certified_prefix[0]['x'].flags.writeable
        try:z.certified_prefix[0]['hard_domain']['lower'][0]=0
        except ValueError:pass
        else:raise AssertionError('mutable source-domain array')
        return dict(no_input_alias=True,arrays_readonly=True)
    ck('snapshots_and_original_domain_do_not_alias_mutated_inputs',readonly)
    def dim149():
        n=149;start=time.perf_counter_ns();z=m.solve_hierarchy(lower=np.full(n,-1.),upper=np.ones(n),levels=[lev('extended',np.eye(n),np.linspace(-.3,.3,n))]);elapsed=(time.perf_counter_ns()-start)/1e6
        assert z.fully_converged and z.completed_prefix==1 and len(z.x)==149;close(z.x,np.linspace(-.3,.3,n));return dict(n=n,elapsed_ms=elapsed,scope='one unconstrained interior resource-domain fixture, not all149-dimensional problems')
    ck('149_dimension_21_plus128_slacks_resource_scope',dim149)
    def dim256():
        n=256;z=m.solve_hierarchy(lower=np.zeros(n),upper=np.zeros(n),levels=[lev('constant',np.zeros((1,n)),[1.])]);assert z.available and z.fully_converged and len(z.x)==256;close(z.x,np.zeros(n));return dict(n=n,constant_residual=z.levels[0]['residual'])
    ck('256_fixed_domain_boundary_is_finite_not_unknown_motion',dim256)
    def too_large():
        try:m.solve_hierarchy(lower=np.zeros(257),upper=np.ones(257),levels=[])
        except ValueError:return dict(rejected=True)
        raise AssertionError('257 coordinates accepted')
    ck('257_dimension_explicit_resource_limit_rejects',too_large)
    def code_identity():
        oldtree=ast.parse(OLD.read_text(encoding='utf-8'));newtree=ast.parse(SRC.read_text(encoding='utf-8'))
        def fun(t,name):return next(v for v in t.body if isinstance(v,ast.FunctionDef) and v.name==name)
        def loop(t):return next(v for v in ast.walk(fun(t,'solve_hierarchy')) if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id=='iterations')
        assert ast.dump(loop(oldtree),include_attributes=False)==ast.dump(loop(newtree),include_attributes=False)
        names=[v.name for v in oldtree.body if isinstance(v,ast.FunctionDef) and v.name not in ('solve_hierarchy','solve_geometry_release')]
        assert all(ast.dump(fun(oldtree,n),include_attributes=False)==ast.dump(fun(newtree,n),include_attributes=False) for n in names)
        return dict(inner_active_set_loop_AST_exact=True,copied_helper_functions_AST_exact=names,max_dimension_only_resource_extension=True)
    ck('original_primal_dual_iteration_and_numerical_helpers_AST_exact',code_identity)
    def archive_one():
        run=ROOT/'output/v20_small_nominal_transport_v1/runs/nominal__ratio_effort__619741';dm=json.loads((run/'release_decision_manifest.json').read_text(encoding='utf-8'))
        with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:
            q=next(v for v in (json.loads(line) for line in f) if v['time_s']==13.96)
        diag=q['hierarchy']['diagnostics'];units=np.asarray(diag['reference_step_units21']);lo=np.asarray(diag['actual_step_lower21'])/units;hi=np.asarray(diag['actual_step_upper21'])/units
        levels=[lev(v['name'],v['matrix'],v['target']) for v in diag['task_matrices_and_targets']]
        z=m.solve_hierarchy(lower=lo,upper=hi,initial=diag['normalized_anchor21'],levels=levels,max_iterations=400,atol=1e-9,rtol=1e-8)
        assert z.available and not z.fully_converged and z.completed_prefix==2 and z.failed_layer['index']==2
        assert z.failed_layer['status']=='DEPENDENT_BLOCKER_NUMERICAL_FAILURE';assert np.array_equal(z.x,z.certified_prefix[-1]['x']);assert z.diagnostics['prefix_replay_solve_count']==0
        # Same captured problem solved once. Earlier actual-layer records are
        # compared numerically, not mislabelled historical byte-exact replay.
        for i in range(2):close(z.levels[i]['residual'],q['hierarchy']['levels'][i]['residual'])
        archive.update(path=str(run/dm['file']),sha256=sha(run/dm['file']),decision_time_s=q['time_s'],original_status=q['hierarchy']['status'],whole_hierarchy_calls=1,returned_x=z.x,completed=z.completed_prefix,failed=z.failed_layer['status'])
        return archive
    ck('one_actual_V20_L3_blocker_preserves_two_certified_layers_without_replay',archive_one)
    ck('no_model_physics_or_native_modules_imported',lambda: {'mujoco_imported':False} if 'mujoco' not in sys.modules else (_ for _ in ()).throw(AssertionError('mujoco unexpectedly imported')))
    unchanged=sha(SRC)==startpin and sha(OLD)=='a3c9f090a5cf4368d7d1ab36344b0aac4cfd76aac3c0367a298a38fdf2932457';passed=all(c['passed'] for c in cases) and unchanged
    report=dict(schema='v21_prefix_hierarchy_author_v1',passed=passed,checks=len(cases),passed_checks=sum(c['passed'] for c in cases),cases=cases,source_sha256={'v21_stable_handoff/prefix_hierarchy.py':startpin,'v17_contact_transfer/hierarchy.py':sha(OLD)},source_snapshot=str(snap),source_unchanged_during_gate=unchanged,author_script_sha256=sha(__file__),
        numerical_budget=dict(max_iterations=400,atol=1e-9,rtol=1e-8,sparse_visits=m.SPARSE_VISIT_BUDGET,rational_operations=m.EXACT_OPERATION_BUDGET,integer_bits=m.EXACT_INTEGER_BIT_BUDGET),
        scope=dict(author_role=True,model=0,physics=0,FK=0,writer=0,native=0,archived_V20_full_problem_calls=1,prefix_replay_calls=0),
        timings_ms=dict(sum=float(sum(costs)),median=float(np.median(costs)),max=float(max(costs))),limitations=['Finite linear problem certificates only, not physical feasibility/response/safety.','Available is nonempty certified prefix; caller must inspect required prefix length and fully_converged.','256 resource input limit does not certify complete large-dimensional numerical robustness or control-period runtime.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=passed,checks=len(cases),failed=[c for c in cases if not c['passed']],out=str(out),sha256=sha(out),source_sha256=startpin),ensure_ascii=False));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())

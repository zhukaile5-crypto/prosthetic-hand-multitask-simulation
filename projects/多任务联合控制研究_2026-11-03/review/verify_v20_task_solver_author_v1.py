"""Finite pure author gate for V20 task_solver; not independent acceptance.
No model/physics/FK/native/writer modules imported. Original152 only hashed.
"""
from pathlib import Path
from copy import deepcopy
from itertools import product
import argparse,ast,hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SRC=ROOT/'v20_contact_exit/task_solver.py';F=ROOT.parent/'五指策略接入_2026-11-03'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def base(P=1):
    H=np.zeros((25,21));H[:21]=np.eye(21);J=np.zeros((P,25));J[:,0]=1
    lo=np.zeros(21);hi=np.zeros(21);lo[:2]=-.01;hi[:2]=.01
    return dict(time_s=12.5,dt_s=.005,current_reference21=np.zeros(21),encoder_reference21=np.zeros(21),
        step_lower21=lo,step_upper21=hi,reference_rates21=np.full(21,2.),normal_jacobian_Px25=J,
        proposal_map25x21=H,robot_velocity25=np.zeros(25),pressure_P_N=np.ones(P),
        reference_target_delta21=np.r_[.01,.006,np.zeros(19)])

def box_oracle(A,b,lo,hi):
    """Independent exhaustive2D active subsets,9 faces, no hierarchy call."""
    best=None
    for state in product((-1,0,1),repeat=2):
        x=np.zeros(2);fixed=[i for i,s in enumerate(state) if s];free=[i for i,s in enumerate(state) if not s]
        for j in fixed:x[j]=lo[j] if state[j]<0 else hi[j]
        if free:x[free]=np.linalg.lstsq(A[:,free],b-A[:,fixed]@x[fixed] if fixed else b,rcond=None)[0]
        if np.any(x<lo-1e-12) or np.any(x>hi+1e-12):continue
        loss=float(np.square(A@x-b).sum())
        if best is None or loss<best[0]:best=(loss,x.copy(),state)
    return best

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='review/v20_task_solver_author_attempt1.json');args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    sourcehash=sha(SRC);snapshot=ROOT/'review/revisions'/('v20_task_solver_'+sourcehash[:12]+'.py');snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists():assert snapshot.read_bytes()==SRC.read_bytes()
    else:snapshot.write_bytes(SRC.read_bytes())
    plan=json.loads((ROOT/'review/v19_small_plan_v1.json').read_text(encoding='utf-8'))
    old={k:sha(F/k[11:] if k.startswith('foundation/') else ROOT/k) for k in plan['source_sha256']}
    spec=importlib.util.spec_from_file_location('v20_contact_exit.task_solver',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cases=[];timings=[]
    def case(name,fn):
        began=time.perf_counter_ns()
        try:detail=fn();cases.append(dict(name=name,passed=True,detail=plain(detail)))
        except Exception as exc:cases.append(dict(name=name,passed=False,error=repr(exc)))
        timings.append((time.perf_counter_ns()-began)/1e6)
    def solve(p):
        z=m.solve_contact_exit(**p)
        assert z.available,(z.status,z.diagnostics)
        return z,z.x*np.asarray(p['reference_rates21'])*p['dt_s']
    def close(x,y):assert np.allclose(x,y,atol=2e-10,rtol=1e-8),(x,y)
    def normal_open_conflict():
        p=base();z,d=solve(p);close(d[:2],[-.01,.006]);assert z.diagnostics['task_priority']==('pressure_share_current_contact_exit','canonical_reference_intent_and_minimal_wrist_action')
        return dict(delta=d[:2],levels=z.levels)
    case('normal_exit_overrides_opposite_OPEN_lower_priority',normal_open_conflict)
    def inward():
        p=base();p['robot_velocity25'][0]=.03;p['current_reference21'][0]=.002;z,d=solve(p)
        close(z.diagnostics['normal_target_b_P_m'],[-.01065]);close(d[0],-.01)
        return dict(target=z.diagnostics['normal_target_b_P_m'],velocity=z.diagnostics['normal_current_velocity_P_m_s'],debt=z.diagnostics['normal_reference_debt_P_m'])
    case('inward_current_velocity_and_debt_add_explicit_rhs',inward)
    def outward():
        p=base();p['robot_velocity25'][0]=-100.;p['encoder_reference21'][0]=100.;z,d=solve(p);close(z.diagnostics['normal_target_b_P_m'],[-.01]);close(d[0],-.01)
        return dict(delta=d[0],target=z.diagnostics['normal_target_b_P_m'])
    case('already_outward_velocity_or_virtual_debt_never_cancel_new_exit',outward)
    def shifted():
        p=base();p['current_reference21'][:]=3.;p['encoder_reference21'][:]=3.;z,d=solve(p);close(d[0],-.01);close(z.diagnostics['reference21'],p['current_reference21']+d)
        return dict(nonzero_current=z.diagnostics['current_reference21'][0],actual_delta=d[0])
    case('absolute_reference_origin_is_preserved_not_reanchored',shifted)
    def nonzero_anchor():
        p=base();p['step_lower21'][0]=.003;p['step_upper21'][0]=.005;z,d=solve(p)
        close(z.diagnostics['anchor_delta21'][0],.003);close(z.diagnostics['normal_outward_box_capacity_P_m_s'],[0]);close(d[0],.003);assert not z.diagnostics['zero_increment_in_actual_box']
        return dict(delta=d[0],anchor=z.diagnostics['anchor_delta21'][0],normal_residual=z.diagnostics['normal_task_residual_P_m'])
    case('true_inward_only_box_uses_nonzero_anchor_without_fake_infeasibility',nonzero_anchor)
    def negative_anchor():
        p=base();p['step_lower21'][0]=-.006;p['step_upper21'][0]=-.002;z,d=solve(p);close(d[0],-.006);close(z.diagnostics['anchor_delta21'][0],-.002)
        return dict(delta=d[0],anchor=z.diagnostics['anchor_delta21'][0])
    case('negative_only_box_anchor_and_outward_capacity_are_original_domain',negative_anchor)
    def opposing():
        p=base(2);p['normal_jacobian_Px25'][1,0]=-1;p['pressure_P_N']=np.array([3.,1.]);z,d=solve(p);close(d[0],-.005)
        return dict(delta=d[0],original_rows=z.diagnostics['normal_matrix_N_Px21'][:,0],residuals=z.diagnostics['normal_task_residual_P_m'])
    case('opposed_patch_normals_stay_separate_weighted_closed_form',opposing)
    def passive():
        p=base();p['normal_jacobian_Px25'][0,24]=1.;p['robot_velocity25'][24]=.02;z,d=solve(p);close(z.diagnostics['normal_current_velocity_P_m_s'],[.02]);close(z.diagnostics['normal_target_b_P_m'],[-.0101])
        return dict(vnormal=z.diagnostics['normal_current_velocity_P_m_s'],nullspace_velocity=p['robot_velocity25'][24])
    case('full25_velocity_keeps_uncontrolled_tendon_motion',passive)
    def zero_authority():
        p=base();p['proposal_map25x21'][0,:]=0;p['robot_velocity25'][0]=.1;z,d=solve(p)
        assert z.diagnostics['zero_local_normal_authority_rows']==[0];close(z.diagnostics['normal_target_b_P_m'],[-.0005]);close(z.diagnostics['normal_effective_report_units_P_m'],[1.]);close(d[:2],p['reference_target_delta21'][:2]);assert z.levels[0]['squared_loss']>0
        return dict(constant_loss=z.levels[0]['squared_loss'],status=z.status)
    case('known_zero_local_authority_retains_nonzero_constant_residual',zero_authority)
    def motion_priority():
        p=base();M=np.zeros((1,21));M[0,0]=1.;M[0,1]=1.;p.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.array([-.4]),motion_time_s=p['time_s'])
        z,d=solve(p);close(d[:2],[-.01,.01]);assert z.levels[1]['preceding_residual_drift_inf']<1e-8
        return dict(delta=d[:2],levels=z.levels)
    case('motion_layer_cannot_sacrifice_attained_normal_exit',motion_priority)
    def motion_null():
        p=base();M=np.zeros((1,21));M[0,1]=1.;p.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.array([-.6]),motion_time_s=p['time_s']);z,d=solve(p);close(d[:2],[-.01,.003]);return dict(delta=d[:2])
    case('motion_nullspace_solution_survives_conflicting_OPEN',motion_null)
    def motion_zero():
        p=base();p.update(motion_matrix_Mx21=np.zeros((1,21)),motion_error_M_m_s=np.array([.2]),motion_time_s=p['time_s']);z,d=solve(p);assert z.diagnostics['motion_task']['zero_local_authority_rows']==[0];close(z.levels[1]['squared_loss'],1e-6);return dict(constant_loss=z.levels[1]['squared_loss'])
    case('known_zero_motion_row_retains_unknown_independent_constant_error',motion_zero)
    def absent_motion():
        z,d=solve(base());assert not z.diagnostics['motion_task']['available'] and z.diagnostics['motion_task']['matrix_Mx21'] is None;return z.diagnostics['motion_task']
    case('unknown_motion_omitted_not_fabricated_zero_measurement',absent_motion)
    def no_contact():
        p=base(0);z,d=solve(p);close(d[:2],p['reference_target_delta21'][:2]);assert len(z.levels)==1 and z.diagnostics['normal_geometry_role']=='KNOWN_NO_ACTIVE_POSITIVE_PRESSURE_ROWS';return dict(levels=len(z.levels))
    case('explicit_P0_is_no_active_contact_not_unknown_geometry',no_contact)
    def split():
        p=base(2);p['normal_jacobian_Px25'][1,0]=-1;p['pressure_P_N']=np.array([3.,1.]);z,d=solve(p)
        q=base(3);q['normal_jacobian_Px25'][2,0]=-1;q['pressure_P_N']=np.array([1.,2.,1.]);zz,dd=solve(q);close(dd,d)
        for x in (-1.,.25,1.):
            xx=np.zeros(21);xx[0]=x
            c=z.diagnostics['task_matrices_and_targets'][0];cc=zz.diagnostics['task_matrices_and_targets'][0]
            close(np.square(c['matrix']@xx-c['target']).sum(),np.square(cc['matrix']@xx-cc['target']).sum())
        return dict(original=d[0],split=dd[0])
    case('same_point_pressure_split_preserves_objective_and_solution',split)
    def permutation():
        p=base(2);p['normal_jacobian_Px25'][1,0]=-1;p['pressure_P_N']=np.array([3.,1.]);z,d=solve(p)
        q=deepcopy(p);q['normal_jacobian_Px25']=q['normal_jacobian_Px25'][::-1];q['pressure_P_N']=q['pressure_P_N'][::-1];zz,dd=solve(q);close(d,dd);return dict(original=d[0],permuted=dd[0])
    case('patch_permutation_keeps_same_task_solution',permutation)
    def extreme_pressure():
        p=base(2);p['normal_jacobian_Px25'][1,0]=-1;p['pressure_P_N']=np.array([1e300,1e300]);z,d=solve(p);close(z.diagnostics['pressure_share_P'],[.5,.5]);close(d[0],0.);return dict(shares=z.diagnostics['pressure_share_P'])
    case('extreme_finite_pressure_normalization_does_not_overflow_sum',extreme_pressure)
    def units_once():
        p=base();p['reference_rates21'][0]=1.;p['step_lower21'][0]=-.005;p['step_upper21'][0]=.005;z,d=solve(p);close(d[0],-.005);close(z.x[0],-1.);return dict(normalized_x=z.x[0],physical_delta=d[0])
    case('normalized_result_restores_reference_units_exactly_once',units_once)
    rng=np.random.default_rng(20261106)
    for k in range(6):
        def oracle_case(k=k):
            p=base(3);J=np.zeros((3,25));J[:,:2]=rng.uniform(-2,2,(3,2));p['normal_jacobian_Px25']=J;p['pressure_P_N']=rng.uniform(.2,3,3);p['robot_velocity25'][:2]=rng.uniform(-.05,.05,2);p['current_reference21'][:2]=rng.uniform(-.003,.003,2)
            p['step_lower21'][:2]=rng.uniform(-.01,-.002,2);p['step_upper21'][:2]=rng.uniform(.002,.01,2);z,d=solve(p)
            lev=z.diagnostics['task_matrices_and_targets'][0];A=lev['matrix'][:,:2];b=lev['target'];lo=p['step_lower21'][:2]/.01;hi=p['step_upper21'][:2]/.01;best=box_oracle(A,b,lo,hi)
            assert best is not None;close(z.x[:2],best[1]);close(lev['squared_loss'],best[0]);return dict(oracle=best[1],actual=z.x[:2],loss=best[0],face=best[2])
        case('independent_2D_nine_active_faces_'+str(k),oracle_case)
    def readonly():
        p=base();save=deepcopy(p);z,d=solve(p)
        assert all(np.array_equal(p[k],v) if isinstance(v,np.ndarray) else p[k]==v for k,v in save.items());assert not z.x.flags.writeable and not z.diagnostics['normal_matrix_N_Px21'].flags.writeable
        try:z.diagnostics['normal_matrix_N_Px21'][0,0]=3
        except ValueError:pass
        else:raise AssertionError('mutable task evidence')
        return dict(inputs_unchanged=True,readonly=True)
    case('inputs_preserved_and_result_matrices_readonly',readonly)
    def invalid(name,mutate,status):
        def fn():
            p=base();mutate(p);z=m.solve_contact_exit(**p);assert not z.available and z.status==status and np.isnan(z.x).all();return dict(status=z.status,reason=z.diagnostics['reason'])
        case(name,fn)
    invalid('unknown_geometry_is_not_no_contact',lambda p:p.update(normal_jacobian_Px25=None),'UNKNOWN_CONTACT_EXIT_INPUT')
    invalid('unknown_encoder_debt_not_filled_zero',lambda p:p.update(encoder_reference21=None),'UNKNOWN_CONTACT_EXIT_INPUT')
    invalid('invalid_dt_rejects_non5ms',lambda p:p.update(dt_s=.02),'INVALID_CONTACT_EXIT_INPUT')
    invalid('pressure_bool_rejected',lambda p:p.update(pressure_P_N=[True]),'INVALID_CONTACT_EXIT_INPUT')
    invalid('numeric_string_velocity_rejected',lambda p:p.update(robot_velocity25=['0']*25),'INVALID_CONTACT_EXIT_INPUT')
    invalid('empty_actual_box_not_model_box_fallback',lambda p:p['step_lower21'].__setitem__(0,.02),'INVALID_CONTACT_EXIT_INPUT')
    invalid('partial_motion_unknown_not_zero_row',lambda p:p.update(motion_matrix_Mx21=np.zeros((1,21))),'UNKNOWN_CONTACT_EXIT_INPUT')
    invalid('future_motion_task_clock_rejected',lambda p:p.update(motion_matrix_Mx21=np.zeros((1,21)),motion_error_M_m_s=[0.],motion_time_s=12.51),'INVALID_CONTACT_EXIT_INPUT')
    invalid('positive_pressure_share_underflow_explicit_unavailable',lambda p:p.update(normal_jacobian_Px25=np.r_[p['normal_jacobian_Px25'],p['normal_jacobian_Px25']],pressure_P_N=[1e308,5e-324]),'UNAVAILABLE_NUMERICAL_CONTACT_EXIT')
    def units_underflow():
        p=base();p['normal_jacobian_Px25'][0,0]=5e-324;z=m.solve_contact_exit(**p);assert not z.available and z.status=='UNAVAILABLE_NUMERICAL_CONTACT_EXIT';return z.diagnostics
    case('tiny_nonzero_row_units_not_mislabelled_exact_zero_authority',units_underflow)
    def stateless():
        p=base();z,d=solve(p);zz,dd=solve(p);assert np.array_equal(z.x,zz.x);return dict(same_input_exact=True,clock_lifecycle_guard_owned_by_caller=True)
    case('stateless_same_input_exact_without_hidden_clock_state',stateless)
    sourceunchanged=sha(SRC)==sourcehash;oldafter={k:sha(F/k[11:] if k.startswith('foundation/') else ROOT/k) for k in old}
    assert len(cases)<=40
    passed=all(x['passed'] for x in cases) and sourceunchanged and old==oldafter==plan['source_sha256'] and 'mujoco' not in sys.modules
    result=dict(schema='v20_contact_exit_task_solver_author_v1',passed=passed,checks=len(cases),passed_checks=sum(x['passed'] for x in cases),cases=cases,
        source_sha256={'v20_contact_exit/task_solver.py':sourcehash,'v20_contact_exit/design.py':sha(ROOT/'v20_contact_exit/design.py'),'v17_contact_transfer/hierarchy.py':sha(ROOT/'v17_contact_transfer/hierarchy.py')},source_snapshot=str(snapshot),frozen_hierarchy_sha256=sha(ROOT/'v17_contact_transfer/hierarchy.py'),
        parent152_unchanged=old==oldafter==plan['source_sha256'],reviewer_role='component author mathematical QA, not independent acceptance',
        scope=dict(models=0,physics=0,FK=0,native_evaluations=0,writer=0,component_fixtures=len(cases)),
        software_wall_ms=dict(sum=float(sum(timings)),median=float(np.median(timings)),max=float(max(timings))),
        numerical_budget=dict(max_iterations=400,atol=1e-9,rtol=1e-8),limitations=['Declared geometric reference intents, not servo response/pressure prediction.','No physical safety/stopping/clearance/placement or complete finite-input domain certificate.'])
    out.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=passed,checks=len(cases),failed=[c for c in cases if not c['passed']],out=str(out),sha256=sha(out),source_sha256=sourcehash),ensure_ascii=False));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())

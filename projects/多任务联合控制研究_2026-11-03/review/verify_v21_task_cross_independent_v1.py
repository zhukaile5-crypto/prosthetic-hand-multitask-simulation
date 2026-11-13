"""Finite cross-author V21 task/ablation/emission tests, pure NumPy only.
Reviewer authored prefix_hierarchy, NOT the task_solver/prefix_task/helper.
This is cross-component math/meaning verification, not independent acceptance
of the reviewer's own prefix kernel. No model/native/physical pipeline.
"""
from pathlib import Path
from copy import deepcopy
import argparse,ast,hashlib,json,sys,warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v21_stable_handoff import task_solver as T,prefix_task as P
from v20_contact_exit import task_solver as V20

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def helper(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_emitted_reference');env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env);return env['score_emitted_reference']

def fixture(P=1,axes=(0,1)):
    B=np.zeros((25,21));B[:21]=np.eye(21);J=np.zeros((P,25));J[:,0]=1.;lo=np.zeros(21);hi=np.zeros(21)
    for j in axes:lo[j]=-.005;hi[j]=.005
    return dict(time_s=12.5,dt_s=.005,current_reference21=np.zeros(21),encoder_reference21=np.zeros(21),step_lower21=lo,step_upper21=hi,reference_rates21=np.ones(21),normal_jacobian_Px25=J,proposal_map25x21=B,robot_velocity25=np.zeros(25),pressure_P_N=np.ones(P),reference_target_delta21=np.zeros(21))

def close(a,b):assert np.allclose(a,b,atol=3e-9,rtol=1e-8),(a,b)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='review/v21_task_cross_independent_v1.json');args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    files=('v21_stable_handoff/task_solver.py','v21_stable_handoff/prefix_task.py','v21_stable_handoff/prefix_supervisor.py','v21_stable_handoff/prefix_hierarchy.py','v21_stable_handoff/design.py')
    pins={f:sha(ROOT/f) for f in files};oldpath=ROOT/'review/revisions/v21_prefix_supervisor_91ac_before_row_norm_guard.py';score=helper(ROOT/'v21_stable_handoff/prefix_supervisor.py');oldscore=helper(oldpath);cases=[];details={}
    def ck(name,fn):
        try:value=fn();cases.append(dict(name=name,passed=True,detail=plain(value)))
        except Exception as exc:cases.append(dict(name=name,passed=False,error=repr(exc)))
    def complete_case(p):
        z=T.solve_contact_exit(**p);assert z.available,(z.status,z.diagnostics);return z
    def validate_extended(z,p):
        d=z.diagnostics;ex=np.asarray(d['extended_normalized_x']);n=21+len(p['pressure_P_N']);assert ex.shape==(n,) and z.x.shape==(21,)
        assert np.array_equal(z.x,ex[:21]);close(d['delta_reference21'],z.x*np.asarray(d['reference_step_units21']))
        G=np.asarray(d['second_stage_hard_matrix']);h=np.asarray(d['second_stage_hard_upper']);assert np.max(G@ex-h,initial=0.)<=2e-8
        close(d['final_slack_P_m'],ex[21:]*np.asarray(d['normal_effective_report_units_P_m']))
        for c in z.certified_prefix:
            assert c['matrix'].shape[1]==n;assert c['KKT']['certified'];assert all(c['KKT']['checks'].values());assert np.max(abs(c['matrix']@ex-c['attained']),initial=0.)<=c['KKT']['coordinate_tolerance']
        return dict(internal_n=n,external_n=len(z.x),max_raw_hard_violation=float(np.max(G@ex-h,initial=0.)))
    def slack_direction():
        p=fixture();p['normal_jacobian_Px25'][0,1]=1.;p['reference_target_delta21'][:2]=.005;z=complete_case(p);d=z.diagnostics;c=z.certified_prefix[0]
        assert np.count_nonzero(c['matrix'][:,:21])==0;close(d['certified_slack_eta_P'],[0.]);assert (d['normal_matrix_N_Px21']@d['delta_reference21']).item()<(d['normal_matrix_N_Px21']@(d['certified_L1_extended_anchor'][:21]*d['reference_step_units21'])).item()-.001
        assert np.array_equal(d['second_stage_lower'][21:],d['second_stage_upper'][21:]);close(z.x[:2],[-1.,-1.]);return validate_extended(z,p)
    ck('one_sided_L1_freezes_slack_and_leaves_strictly_outward_reference_freedom',slack_direction)
    def nonzero_witness():
        p=fixture();p['robot_velocity25'][0]=.2;M=np.zeros((1,21));M[0,0]=1.;p.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.zeros(1),motion_time_s=12.5);z=complete_case(p);d=z.diagnostics;xa=np.asarray(d['certified_L1_extended_anchor']);units=np.asarray(d['reference_step_units21']);motion=d['motion_task'];own=(M@(xa[:21]*units)+.005*p['motion_error_M_m_s'])/motion['effective_row_units_M']
        assert abs(xa[0])>.1;close(own,motion['witness_normalized_residual_M']);close(abs(own),motion['normalized_absolute_budget_M']);assert np.max(np.asarray(motion['budget_inequality_matrix'])@xa-motion['budget_inequality_upper'])<=2e-8
        close(z.x[0],-.2);return dict(anchor=xa,final=z.x,own_witness=own,reset_zero_would_be_different=True,extended=validate_extended(z,p))
    ck('motion_budget_is_witnessed_by_actual_nonzero_L1_anchor_not_reset_zero',nonzero_witness)
    def forced_nonzero_box():
        p=fixture();p['step_lower21'][0]=.001;p['step_upper21'][0]=.004;z=complete_case(p);d=z.diagnostics;close(z.x[0],.2);close(d['certified_slack_P_m'],[.001]);assert d['current_reference_outside_task_box'];return validate_extended(z,p)
    ck('actual_box_excluding_zero_preserves_physical_slack_and_nonzero_anchor',forced_nonzero_box)
    def split():
        p=fixture();p['normal_jacobian_Px25'][0,1]=1.;p['pressure_P_N'][:]=3.;a=complete_case(p)
        q=fixture(2);q['normal_jacobian_Px25'][:,1]=1.;q['pressure_P_N']=np.array([.75,2.25]);b=complete_case(q);close(a.x,b.x)
        # Independent comparison of the L1 weighted physical-identical slack
        # cost for a non-optimal shared eta, not just selected solutions.
        ca=a.certified_prefix[0];cb=b.certified_prefix[0];xa=np.zeros(22);xb=np.zeros(23);xa[21:]=.17;xb[21:]=.17
        costA=float(np.square(ca['matrix']@xa-ca['target']).sum());costB=float(np.square(cb['matrix']@xb-cb['target']).sum());close(costA,costB)
        return dict(reference_solution=a.x,shared_eta_cost=[costA,costB],internal_dimensions=[22,23])
    ck('same_point_pressure_split_preserves_cost_and_reference_with_distinct_slack_dimension',split)
    def known_P0():
        p=fixture(0);p['reference_target_delta21'][0]=.004;p['path_reference_target_delta21']=np.r_[-.002,np.zeros(20)];z=complete_case(p);assert z.fully_converged and len(z.diagnostics['extended_normalized_x'])==21 and len(z.diagnostics['certified_slack_eta_P'])==0;close(z.x[0],-.4);assert z.diagnostics['original_canonical_target_delta21'][0]==.004
        sc=score(z,current21=p['current_reference21'],proposed21=z.diagnostics['reference21'],scale21=np.full(21,.005));assert sc['available'];return dict(internal_n=21,posture=z.x[0],fullscore=sc['available'])
    ck('P0_has21_dimensions_and_explicit_path_only_changes_last_goal',known_P0)
    def posture_failure():
        p=fixture(1,axes=(0,1,2));p['step_upper21'][0]=-.001;p['reference_target_delta21'][:3]=[-.005,.01,.015];p['max_iterations']=2;z=T.solve_contact_exit(**p)
        assert z.available and not z.fully_converged and z.completed_prefix==2,(z.status,z.completed_prefix,z.failed_layer);assert z.failed_layer['name']=='canonical_or_explicit_path_reference';assert z.failed_layer['status']=='ITERATION_LIMIT';close(z.x[:3],[-1.,0.,0.]);assert z.certified_prefix[-1]['name']=='scalar_pressure_share_normal_withdrawal'
        sc=score(z,current21=p['current_reference21'],proposed21=z.diagnostics['reference21'],scale21=np.full(21,.005));assert sc['available'] and not sc['fully_converged'];return dict(returned=z.x,failed=z.failed_layer['status'],score_available=sc['available'],extended=validate_extended(z,p))
    ck('genuine_last_posture_iteration_failure_returns_prior_certified_action_not_failed_trial',posture_failure)
    def ablation_identity():
        def fn(path):return next(x for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name=='solve_contact_exit')
        assert ast.dump(fn(ROOT/'v20_contact_exit/task_solver.py'),include_attributes=False)==ast.dump(fn(ROOT/'v21_stable_handoff/prefix_task.py'),include_attributes=False)
        p=fixture(1,axes=(0,1,2));p['reference_target_delta21'][1:3]=[.01,.015];p['max_iterations']=2;new=P.solve_contact_exit(**p);old=V20.solve_contact_exit(**p)
        assert not old.available and new.available and not new.fully_converged and new.completed_prefix==1;close(new.x[:3],[-1.,0.,0.]);assert old.levels[-1]['status']==new.levels[-1]['status']=='ITERATION_LIMIT'
        for key in ('normal_matrix_N_Px21','normal_target_b_P_m','raw_reference_target_delta21'):
            assert np.array_equal(old.diagnostics[key],new.diagnostics[key])
        return dict(task_constructor_AST_exact=True,original_status=old.status,new_status=new.status,certified_prefix=new.completed_prefix)
    ck('V20_prefix_ablation_preserves_objective_constructor_and_changes_only_failure_return',ablation_identity)
    def helper_actual_du():
        p=fixture();z=complete_case(p);u=np.asarray(z.diagnostics['reference21']);sc=score(z,current21=p['current_reference21'],proposed21=u,scale21=np.full(21,.005));assert sc['available'];x=np.r_[u/.005,z.diagnostics['extended_normalized_x'][21:]]
        for c,v in zip(z.certified_prefix,sc['levels']):close(v['residual'],c['matrix']@x-c['target']);close(v['squared_loss'],np.square(c['matrix']@x-c['target']).sum())
        bad=u.copy();bad[0]+=.001;badscore=score(z,current21=p['current_reference21'],proposed21=bad,scale21=np.full(21,.005));assert not badscore['available'];return dict(internal_n=len(x),original_available=sc['available'],changed_reference_available=badscore['available'])
    ck('emission_uses_real_du_plus_original_eta_and_scores_every_certified_layer',helper_actual_du)
    def historical_overflow():
        p=fixture();M=np.zeros((1,21));M[0,1]=1.;p.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.zeros(1),motion_row_scale_M=np.array([1e-200]),motion_time_s=12.5,max_iterations=1);z=T.solve_contact_exit(**p)
        assert z.available and z.completed_prefix==1 and not z.fully_converged
        u=np.zeros(21);u[1]=.0025;extended=np.asarray(z.diagnostics['extended_normalized_x']).copy();extended[:21]=u/.005;G=z.diagnostics['second_stage_hard_matrix'];h=z.diagnostics['second_stage_hard_upper'];violation=float(np.max(G@extended-h))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');before=oldscore(z,current21=np.zeros(21),proposed21=u,scale21=np.full(21,.005))
        after=score(z,current21=np.zeros(21),proposed21=u,scale21=np.full(21,.005));good=score(z,current21=np.zeros(21),proposed21=np.asarray(z.diagnostics['reference21']),scale21=np.full(21,.005))
        assert before['available'] is True and violation>1e190 and after['available'] is False and good['available'] is True
        details['preserved_actual_old_failure']=dict(source_sha256=sha(oldpath),source_snapshot=str(oldpath),old_available=True,new_available=False,unmodified_reference_new_available=good['available'],true_raw_hard_violation=violation,finite_motion_row_scale=1e-200,completed_prefix=z.completed_prefix,old_overflow_warnings=[str(w.message) for w in ws],changed_reference21=u.tolist())
        return details['preserved_actual_old_failure']
    ck('actual_finite_row_norm_overflow_failure_preserved_and_fresh_helper_rejects_same_motion_violation',historical_overflow)
    def zero_row_guard():
        p=fixture();z=complete_case(p);z=deepcopy(z);n=len(z.diagnostics['extended_normalized_x']);z.diagnostics['second_stage_hard_matrix']=np.zeros((1,n));z.diagnostics['second_stage_hard_upper']=np.array([-1.]);sc=score(z,current21=p['current_reference21'],proposed21=z.diagnostics['reference21'],scale21=np.full(21,.005));assert not sc['available'];return dict(known_zero_inconsistent_domain_rejected=True)
    ck('new_extra_domain_guard_rejects_impossible_constant_row',zero_row_guard)
    def unknown_motion():
        p=fixture();p['motion_error_M_m_s']=np.zeros(1);z=T.solve_contact_exit(**p);assert not z.available and z.status=='UNKNOWN_CONTACT_EXIT_INPUT' and np.isnan(z.x).all();return dict(status=z.status)
    ck('incomplete_motion_problem_is_unknown_not_fabricated_zero',unknown_motion)
    after={f:sha(ROOT/f) for f in files};ck('review_bound_source_unchanged',lambda: {'source_unchanged':True} if pins==after else (_ for _ in ()).throw(AssertionError((pins,after))))
    assert 'mujoco' not in sys.modules
    report=dict(schema='v21_task_cross_independent_v1',passed=all(c['passed'] for c in cases),checks=len(cases),cases=cases,source_sha256=pins,source_after_sha256=after,reader_sha256=sha(__file__),historical_helper_sha256=sha(oldpath),historical_failure=details.get('preserved_actual_old_failure'),
        role='prefix_hierarchy author cross-review of independently authored task_solver/prefix_task/emitted helper; not own-kernel independent acceptance',
        scope=dict(robot_model=0,physics=0,FK=0,native=0,writer=0,resource149_fixture_repeated=False),
        limitations=['Declared reference hierarchy and witnessed TASK budget only; not actual motion, contact pressure or safe handoff.','No complete runtime/lifecycle admission here.','Prefix availability is not full-goal convergence; every failed layer remains labelled.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');out.with_suffix('.md').write_text('# V21任务层与发出参考跨作者审查\n\n'+('PASS' if report['passed'] else 'FAIL')+f"；{sum(c['passed'] for c in cases)}/{len(cases)}有限纯检查。绑定源、解析期望、单侧slack/真实L1 witness/21与扩展变量/后层失败前缀见JSON。\n\n发现并实际复现旧91ac helper的finite-row norm overflow：运动预算violation2.5e197却available=True。旧完整源经SHA核对保留；新helper用相同反例拒绝，未改变正常锚点可用性。没有修改本审查对象源码。\n\n审者为prefix内核作者，未参与任务层/消融包装/ROOT helper。零模型、物理、FK或writer，不重跑149维资源门，不签真实运动/清离或安全。\n",encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(cases),failed=[c for c in cases if not c['passed']],out=str(out),sha256=sha(out),source_sha256=pins),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

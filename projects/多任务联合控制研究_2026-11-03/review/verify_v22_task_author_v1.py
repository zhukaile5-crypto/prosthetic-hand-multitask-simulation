"""Analytic task conflict, unknown/P0 semantics, and actual emitted scoring."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v22_joint_task_fusion import task_solver as task
from v21_stable_handoff.prefix_supervisor import score_emitted_reference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def base(P=1):
    B=np.zeros((25,21));B[:21]=np.eye(21);J=np.zeros((P,25));J[:,0]=1.
    lo=np.zeros(21);hi=np.zeros(21);lo[0]=-.005;hi[0]=.005
    return dict(time_s=12.5,dt_s=.005,current_reference21=np.zeros(21),encoder_reference21=np.zeros(21),
        step_lower21=lo,step_upper21=hi,reference_rates21=np.ones(21),normal_jacobian_Px25=J,
        proposal_map25x21=B,robot_velocity25=np.zeros(25),pressure_P_N=np.ones(P),reference_target_delta21=np.zeros(21))
def main(out):
    assert not out.exists();checks=[];cases=[]
    sources=['v22_joint_task_fusion/task_solver.py','v22_joint_task_fusion/task_bundle.py','v22_joint_task_fusion/finite_solver.py','v22_joint_task_fusion/numerics.py']
    before={k:sha(ROOT/k) for k in sources}
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    k=base();k['robot_velocity25'][0]=.4;M=np.zeros((1,21));M[0,0]=1.
    k.update(motion_matrix_Mx21=M,motion_error_M_m_s=np.array([.1]),motion_time_s=12.5)
    b=task.solve_contact_exit(**k,formulation='backend');j=task.solve_contact_exit(**k,formulation='joint')
    ck('both actual contact formulations available',b.available and j.available)
    ck('joint analytic compromise x=-.25 eta=.15',abs(j.x[0]+.25)<1e-7 and abs(j.diagnostics['certified_slack_eta_P'][0]-.15)<1e-7)
    ck('backend keeps prior correction optimum',abs(b.diagnostics['certified_slack_eta_P'][0])<1e-8 and b.x[0]<=-.4+1e-7)
    ck('joint witness is not submitted old normal prefix',not j.diagnostics['normal_witness_is_submitted_prefix'] and j.certified_prefix[0]['name']=='joint_normal_and_observed_projection' and all(c['name']!='normal_inward_correction_slack' for c in j.certified_prefix))
    ck('joint eta original bounds remain free',j.diagnostics['second_stage_lower'][21]<j.diagnostics['second_stage_upper'][21])
    md=j.diagnostics['motion_task'];units=np.asarray(j.diagnostics['reference_step_units21']);effective=np.asarray(md['effective_row_units_M'])
    witness=(M*units[None,:]/effective[:,None])@np.asarray(md['witness_normalized_x21'])+.005*k['motion_error_M_m_s']/effective
    violation=np.max(md['budget_inequality_matrix']@j.diagnostics['extended_normalized_x']-md['budget_inequality_upper'],initial=0.)
    tolerance=max(c['KKT']['coordinate_tolerance'] for c in j.certified_prefix)
    ck('motion budget witnessed same L1 point',md['available'] and np.allclose(abs(witness),md['normalized_absolute_budget_M'],atol=1e-14,rtol=0) and violation<=tolerance)
    for name,r,kw in [('backend_conflict',b,k),('joint_conflict',j,k)]:
        score=score_emitted_reference(r,current21=kw['current_reference21'],proposed21=kw['current_reference21']+.005*r.x,scale21=np.full(21,.005))
        ck(name+' actual emitted task score',score['available']);cases.append(dict(name=name,status=r.status,x0=float(r.x[0]),completed=r.completed_prefix,full=r.fully_converged,score=score))
    for variant in ('backend','joint'):
        k=base();r=task.solve_contact_exit(**k,formulation=variant)
        ck(variant+' unknown motion omitted explicitly',r.available and r.diagnostics['motion_unknown_not_zero'] and not r.diagnostics['motion_task']['available'])
        k=base(P=0);k['path_reference_target_delta21']=np.r_[-.003,np.zeros(20)];r=task.solve_contact_exit(**k,formulation=variant)
        ck(variant+' known P0 retains explicit path',r.available and abs(r.x[0]+.6)<1e-7 and len(r.diagnostics['certified_slack_eta_P'])==0)
        k=base();k['normal_jacobian_Px25']=None;r=task.solve_contact_exit(**k,formulation=variant)
        ck(variant+' unknown geometry not P0',not r.available and r.status=='UNKNOWN_CONTACT_EXIT_INPUT')
    # A witness for another objective cannot become a false joint certificate.
    original=task.solve_hierarchy
    for variant in ('backend','joint'):
        count=[0]
        def fail_second(**kw):
            count[0]+=1
            if count[0]==1:return original(**kw)
            return SimpleNamespace(x=np.asarray(kw['initial']).copy(),status='FORCED_NO_CERT',available=False,
                certified_prefix=(),fully_converged=False,failed_layer=dict(index=0,name=kw['levels'][0].name,status='FORCED_NO_CERT'),levels=(),diagnostics={'test_injection':True})
        with patch.object(task,'solve_hierarchy',fail_second):r=task.solve_contact_exit(**base(),formulation=variant)
        ck(variant+' second-stage failure honest prefix',r.available if variant=='backend' else not r.available and r.completed_prefix==0)
    after={k:sha(ROOT/k) for k in sources};ck('production sources unchanged',before==after)
    result=dict(schema='v22_task_author_v1',passed=all(c['passed'] for c in checks),checks=checks,cases=cases,
        production_source_sha256=after,physics_steps=0,solver_is_actual=True,issues=[c['name'] for c in checks if not c['passed']])
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'out':str(out),'passed':result['passed'],'checks':len(checks),'issues':result['issues']},ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();raise SystemExit(main(ROOT/a.out))

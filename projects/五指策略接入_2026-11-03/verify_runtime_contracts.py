"""Socket-free API and mathematical run/evaluation contracts; no dynamics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from io import BytesIO
import json,hashlib
import numpy as np

import session
import server
from evaluate_runs import first_dwell,summarize
from run_comparison import jobs_from_plan

ROOT=Path(__file__).resolve().parent

def request(viewer,path,method='GET',payload=None,origin=None):
    h=server.Handler.__new__(server.Handler)
    h.server=SimpleNamespace(viewer=viewer,server_port=8880)
    h.path=path;result={}
    data=json.dumps(payload).encode() if payload is not None else b''
    h.headers={'Content-Length':str(len(data))}
    if origin is not None:h.headers['Origin']=origin
    h.rfile=BytesIO(data)
    def reply(value,mime='application/json; charset=utf-8',status=200):
        result.update(value=value,mime=mime,status=status)
    h.reply=reply
    if method=='GET':h.do_GET()
    else:h.do_POST()
    return result

def main():
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    blocked=dict(passed=False,problems=['test physical gate not passed'])
    with patch('session.check_admission',return_value=blocked),patch('session.HandPlant',side_effect=AssertionError('physics constructed too early')):
        rejected=False
        try:session.TrialSession()
        except RuntimeError:rejected=True
        check('admission_rejects_before_physics_and_policy',rejected)
    viewer=server.Viewer()
    with patch('server.check_admission',return_value=blocked):
        state=request(viewer,'/state');frame=request(viewer,'/frame.png?snapshot=0')
        check('state_has_integration_app_identity',state['value']['app']==session.APP_ID)
        check('state_frame_snapshot_identity',state['value']['snapshot_id']==0 and frame['status']==200 and frame['value'].startswith(b'\x89PNG'))
        check('expired_snapshot_never_returns_different_image',request(viewer,'/frame.png?snapshot=999')['status']==410)
        capabilities=request(viewer,'/capabilities')['value']
        check('blocked_capabilities_are_explicit',not capabilities['admitted'] and len(capabilities['scenarios'])==16)
        result=request(viewer,'/command','POST',{'action':'new_run','value':{'method':'ratio'}})
        check('new_trial_rejected_without_physical_gate',result['status']==409 and viewer.commands.empty())
        result=request(viewer,'/command','POST',{'action':'camera','value':{}},origin='https://foreign.example')
        check('cross_origin_command_rejected',result['status']==403 and viewer.commands.empty())
        check('unknown_command_rejected',request(viewer,'/command','POST',{'action':'unknown'})['status']==400)
        check('invalid_snapshot_id_rejected',request(viewer,'/frame.png?snapshot=nope')['status']==400)
    with patch('server.check_admission',return_value={'passed':True,'problems':[]}):
        result=request(viewer,'/command','POST',{'action':'new_run','value':{'method':'unified'}})
        check('admitted_command_is_queued_not_claimed_executed',result['status']==200 and result['value']['status']=='queued')
        sequence,payload=viewer.commands.get_nowait()
        check('command_sequence_is_monotonic',sequence==result['value']['sequence']==1 and payload['value']['method']=='unified')
    check('viewer_worker_never_started_in_contract_tests',not viewer.thread.is_alive())
    check('fixed_scenario_identifiers_are_unique',len(session.SCENE_BY_NAME)==16)
    env,lift,wrist=session.event_schedule('collision',9.5,.6,.1)
    check('collision_fixture_reaches_hand_region',env['obstacle_x']==-.30)
    check('events_do_not_occur_in_quality_window',session.event_schedule('three_pulses',8.5,.6,.1)[0]['force']==[0.,0.,0.])
    check('friction_event_uses_simulation_time',session.event_schedule('friction_drop',9.3,.6,.1)[0]['friction']==.33)
    check('release_scene_pull_straddles_release',session.event_schedule('release_slip',12.1,.6,.1)[0]['force'][1]>0.)
    s,a=session.physical_parameters('nominal',530101,'confirmation')
    s,b=session.physical_parameters('nominal',530101,'confirmation')
    check('paired_parameters_do_not_depend_on_method',a==b)
    t=np.arange(0.,1.40001,.005);mask=np.ones(len(t),bool)
    check('stability_requires_fresh_dwell',abs(first_dwell(t,mask,0.,1.4)-.2)<1e-12)
    mask[:]=False;mask[10:30]=True
    check('short_stable_burst_does_not_pass',first_dwell(t,mask,0.,1.4) is None)
    plan={'methods':['ratio','unified'],'phases':{'confirmation':{'seeds':list(range(530101,530109)),'scenarios':list(session.SCENE_BY_NAME)}}}
    jobs=jobs_from_plan(plan,'confirmation')
    check('confirmation_cross_product_has_256_trials',len(jobs)==256)
    check('confirmation_pairs_share_scene_shape_seed',len({(j['scenario'],j['shape'],j['seed']) for j in jobs})==128)
    check('strict_json_converts_missing_values',session.jsonable({'x':np.nan,'y':np.array([np.inf,1.])})=={'x':None,'y':[None,1.]})
    fixture={'passed':True,'source_path':'verify_runtime_contracts.py',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'summary_path':'verify_runtime_contracts.py',
        'summary_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'force_rmse_N':0.,'force_p95_N':0.}
    for radius in (.02,.04):
        evidence={**fixture,'radius':radius}
        check(f'validated_radius_{radius}_accepted_by_identity_contract',
            session.validate_radius_evidence({'selected_radius':radius,'selected_holdout':evidence})==radius)
    rejected=False
    try:session.validate_radius_evidence({'selected_radius':.05,'selected_holdout':{**fixture,'radius':.05}})
    except ValueError:rejected=True
    check('unvalidated_radius_expansion_rejected',rejected)
    rejected=False
    try:session.validate_radius_evidence({'selected_radius':.04,'selected_holdout':{**fixture,'source_sha256':'wrong','radius':.04}})
    except ValueError:rejected=True
    check('holdout_source_mismatch_rejected',rejected)
    # A tiny paired mathematical readout exercises cohort exclusion, not physics.
    rows=[]
    for seed in (1,2):
        for method,value in [('ratio',.2),('unified',.4)]:
            rows.append(dict(split='synthetic',mode='scheduled',scenario='nominal',shape='capsule',seed=seed,
                method=method,boundary=False,task_success=True,A_S=.2,preparation_passed=True,
                static_eligible=True,mean_force=2.,finger_tv_rate=value,safety={'force_protection_pass':True,'total_excess_integral':0.,'local_excess_integral':0.},
                release_planned=True,release_success=True,failure_reason=None))
    result=summarize(rows,bootstrap=20)
    tv=next(r for r in result['comparisons'] if r['endpoint']=='finger_tv_rate')
    check('paired_readout_uses_signed_candidate_minus_baseline',abs(tv['difference']-.2)<1e-12 and tv['pairs']==2)
    rows[-1]['static_eligible']=False
    result=summarize(rows,bootstrap=20);tv=next(r for r in result['comparisons'] if r['endpoint']=='finger_tv_rate')
    check('missing_window_excluded_from_quality_not_task_denominator',tv['pairs']==1 and sum(r['trials'] for r in result['rows'])==4)
    files=('session.py','server.py','run_comparison.py','evaluate_runs.py','verify_runtime_contracts.py')
    report=dict(passed=all(checks.values()),checks=checks,check_count=len(checks),
        scope='Socket-free API, admission ordering, synthetic radius-evidence identity, schedule and mathematical readout tests only; no policy instantiated, no MuJoCo dynamics or server worker run. Radius identity fixture is this test source, never an admission certificate.',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    out=ROOT/'review/runtime_static_contract.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'certificate':str(out)},ensure_ascii=False))

if __name__=='__main__':main()

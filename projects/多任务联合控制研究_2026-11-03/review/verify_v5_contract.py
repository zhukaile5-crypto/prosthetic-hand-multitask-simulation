"""Independent task-contract and injectable interface checks, no dynamics.

Self-tests exercise rejection witnesses for the checker. They do not certify a
controller. Real API/adapter checks require an explicitly supplied candidate.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse,ast,hashlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={'time_s','mode','motion_target','executed_motion_reference','motion_error',
    'tracking_tolerances','limits','budget','deadline_s','remaining_time_s',
    'deadline_exceeded','force_scope'}
FORBIDDEN={'scenario','scenario_name','object_id','target_id','is_target','object_pose',
    'object_position','true_mass','true_friction','true_mu','future_disturbance',
    'future_external_force','future_friction','event_schedule','environment_schedule'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b,tol=1e-10):
    try:return bool(np.allclose(a,b,atol=tol,rtol=0,equal_nan=False))
    except (ValueError,TypeError):return False
def verify_request(request,inputs):
    checks={'ordinary_dict':type(request)is dict,'required_fields':isinstance(request,dict) and REQUIRED<=set(request)}
    if not checks['required_fields']:return dict(passed=False,checks=checks)
    def finite_vector(key,size):
        x=np.asarray(request[key],float);return x.shape==(size,) and np.isfinite(x).all()
    checks['public_fields_only']=not(FORBIDDEN&set(request))
    checks['finite_dimensions']=all(finite_vector(k,n) for k,n in [('motion_target',3),('executed_motion_reference',3),('motion_error',3),('tracking_tolerances',3),('limits',5)])
    checks['finite_clock_deadline_budget']=all(np.isfinite(float(request[k])) for k in ['time_s','deadline_s','remaining_time_s','budget'])
    checks['nominal_task_retained']=close(request['motion_target'],inputs['motion_target']) and close(request['executed_motion_reference'],inputs['executed_motion_reference'])
    checks['signed_motion_error']=close(request['motion_error'],np.asarray(inputs['motion_target'])-np.asarray(inputs['executed_motion_reference']))
    checks['limits_and_tolerances_retained']=close(request['limits'],inputs['limits']) and close(request['tracking_tolerances'],inputs['tracking_tolerances']) and close(request['budget'],inputs['budget'])
    checks['remaining_time']=close(request['remaining_time_s'],max(0.,inputs['deadline_s']-inputs['time_s']))
    checks['time_and_mode']=close(request['time_s'],inputs['time_s']) and request['mode']==inputs['mode'] and close(request['deadline_s'],inputs['deadline_s'])
    if inputs['time_s']<inputs['deadline_s']-1e-9:checks['deadline_not_early']=request['deadline_exceeded'] is False
    if inputs['time_s']>inputs['deadline_s']+1e-9:checks['deadline_not_stale']=request['deadline_exceeded'] is True
    checks['all_external_force_scope']=request['force_scope'] in {'external_all','external_all_instrumented_fingers'}
    try:json.dumps(request,allow_nan=False);checks['ordinary_json_serializable']=True
    except (ValueError,TypeError):checks['ordinary_json_serializable']=False
    return dict(passed=all(checks.values()),checks=checks)

def contract_cases():
    common=dict(mode='hold',deadline_s=12.,limits=[5.]*5,budget=18.,tracking_tolerances=[.01,.015,.015])
    return [dict(common,time_s=8.,motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.]),
        dict(common,time_s=9.8,motion_target=[.094,.021,-.014],executed_motion_reference=[.1,0.,0.]),
        dict(common,time_s=12.1,motion_target=[.094,.021,-.014],executed_motion_reference=[.1,.01,-.005]),
        dict(common,mode='release',time_s=10.,motion_target=[.094,.021,-.014],executed_motion_reference=[.098,.01,0.])]

def verify_real_builder(builder):
    results=[]
    for inputs in contract_cases():results.append(verify_request(builder(**deepcopy(inputs)),inputs))
    invalid=[]
    for key,value in [('motion_target',[0.,0.]),('executed_motion_reference',[0.,float('nan'),0.]),('tracking_tolerances',[.01,-.015,.015]),('limits',[5.,5.,5.,5.]),('budget',float('inf')),('time_s',float('nan')),('time_s',True)]:
        inputs=deepcopy(contract_cases()[1]);inputs[key]=value
        try:builder(**inputs);invalid.append(dict(field=key,rejected=False))
        except (ValueError,TypeError):invalid.append(dict(field=key,rejected=True))
    inputs=deepcopy(contract_cases()[0]);inputs.pop('tracking_tolerances');unknown=builder(**inputs)
    default_unknown=unknown['tracking_tolerances']==[.01,None,None]
    return dict(passed=all(r['passed'] for r in results) and all(r['rejected'] for r in invalid) and default_unknown,valid_cases=results,invalid_inputs=invalid,unregistered_default_wrist_tolerances_remain_unknown=default_unknown,scope='Real task builder API only, not motion/controller/physics admission')

def verify_real_modes(module):
    checks={};base=dict(command='hold',preparation_complete=True,motion_target=[.1,.02,0.],executed_motion_reference=[.1,0.,0.])
    checks['moving_user_task_is_transport']=module.select_task_mode(**base)=='transport'
    checks['motion_does_not_skip_preparation']=module.select_task_mode(**dict(base,preparation_complete=False))=='preparation'
    checks['stationary_task_is_hold']=module.select_task_mode(**dict(base,executed_motion_reference=[.1,.02,0.]))=='hold'
    checks['current_public_sliding_can_request_recovery']=module.select_task_mode(**base,online_contact_proxy='sliding')=='recovery'
    checks['release_preempts_preparation']=module.select_task_mode(**dict(base,command='release',preparation_complete=False))=='release'
    checks['clear_release_is_monitored']=module.select_task_mode(**base,release_active=True,clear_confirmed=True)=='monitor'
    request=module.build_task_request(**contract_cases()[1]);checks['valid_contract_roundtrip']=module.validate_task_request(request)==request
    for field,value in [('scenario','hidden_future'),('object_state_available',True),('remaining_time_s',99.),('deadline_exceeded',0)]:
        bad=deepcopy(request);bad[field]=value
        try:module.validate_task_request(bad);checks['reject_poison/'+field]=False
        except (ValueError,TypeError):checks['reject_poison/'+field]=True
    return dict(passed=all(checks.values()),checks=checks,scope='Real public contract mode/validation API only')

def verify_nominal_passthrough(outputs,tol=1e-9):
    """Only for fixtures independently declaring safe/executable nominal input.

    Fields: nominal_input8, actual_input8, correction_input8, safe_nominal,
    nominal_feasible, motion_active. This checks actual controller outputs; the
    fixture's physical/synthetic premises must be separately justified.
    """
    checks={'safe_nonzero_motion_fixture_exercised':any(r.get('safe_nominal') and r.get('nominal_feasible') and r.get('motion_active') for r in outputs)}
    for i,r in enumerate(outputs):
        nominal=np.asarray(r['nominal_input8']);actual=np.asarray(r['actual_input8']);correction=np.asarray(r['correction_input8'])
        valid=nominal.shape==actual.shape==correction.shape==(8,) and np.isfinite(np.r_[nominal,actual,correction]).all()
        checks[f'{i}/finite_input8']=bool(valid)
        checks[f'{i}/actual_equals_nominal_plus_correction']=valid and close(actual,nominal+correction,tol)
        if r['safe_nominal'] and r['nominal_feasible']:
            checks[f'{i}/safe_nominal_not_vetoed']=valid and close(actual,nominal,tol) and close(correction,np.zeros(8),tol)
        if r.get('motion_active'):
            checks[f'{i}/genuine_nonzero_motion_fixture']=bool(np.linalg.norm(nominal[5:])>1e-12)
    return dict(passed=bool(outputs) and all(checks.values()),checks=checks)

def verify_clear_latch(outputs):
    first=next((i for i,r in enumerate(outputs) if r.get('clear_confirmed')),None)
    checks={'clear_was_really_exercised':first is not None}
    if first is not None:
        tail=outputs[first:]
        checks['release_authority_retained']=all(r['release_authority'] and not r['holding_called'] for r in tail)
        checks['monitoring_retained']=all(r['monitoring_active'] for r in tail)
        checks['no_held_or_preclear_reclose_after_clear']=all(r['posture_mode'] not in {'hold','held_grasp','public_thumb_clearance','contact_clearance_pose'} for r in tail)
        for previous,current in zip(tail,tail[1:]):
            if current.get('observation_unknown'):
                checks['unknown_does_not_create_recontact']=checks.get('unknown_does_not_create_recontact',True) and current['recontact_count']==previous['recontact_count']
    return dict(passed=all(checks.values()),checks=checks)

def compare_baseline_outputs(reference,candidate,tol=1e-12):
    keys=('delta_closure','predicted_normal','pause_transport','motion_reference')
    checks={'same_sample_count':len(reference)==len(candidate),'nonempty_archived_sequence':bool(reference)}
    for key in keys:
        present=[key in r for r in reference];other=[key in r for r in candidate]
        checks[key+'/same_presence']=present==other
        if any(present):checks[key+'/numeric_equal']=all(bool(np.allclose(a[key],b[key],atol=tol,rtol=0,equal_nan=key=='predicted_normal')) for a,b,p in zip(reference,candidate,present) if p)
    return dict(passed=all(checks.values()),checks=checks,scope='Same archived inputs only; new closed-loop equivalence remains a separate physical-run check')

def static_boundary(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));hits=[]
    for n in ast.walk(tree):
        value=n.attr if isinstance(n,ast.Attribute) else n.id if isinstance(n,ast.Name) else n.value if isinstance(n,ast.Constant) and isinstance(n.value,str) else n.name if isinstance(n,ast.alias) else None
        if value in FORBIDDEN:hits.append(dict(line=n.lineno,field=value))
    return dict(no_forbidden_read_spelling=not hits,hits=hits,scope='Static spelling scan only; real input-poison invariance and runtime dataflow are still required')

def checker_self_test():
    inputs=contract_cases()[1]
    request={**inputs,'motion_error':[-.006,.021,-.014],'remaining_time_s':2.2,'deadline_exceeded':False,'force_scope':'external_all'}
    checks={'valid_public_request_accepted':verify_request(request,inputs)['passed']}
    bad=deepcopy(request);bad['motion_error']=[0.,0.,0.];checks['lost_motion_request_rejected']=not verify_request(bad,inputs)['passed']
    bad=deepcopy(request);bad['scenario']='future_known_scene';checks['scene_leak_rejected']=not verify_request(bad,inputs)['passed']
    good=dict(nominal_input8=np.r_[np.zeros(5),-.001,.002,0.],actual_input8=np.r_[np.zeros(5),-.001,.002,0.],correction_input8=np.zeros(8),safe_nominal=True,nominal_feasible=True,motion_active=True)
    checks['safe_nonzero_nominal_motion_accepted']=verify_nominal_passthrough([good])['passed']
    bad=deepcopy(good);bad['actual_input8']=np.zeros(8);bad['correction_input8']=-bad['nominal_input8'];checks['old_anchor_motion_veto_rejected']=not verify_nominal_passthrough([bad])['passed']
    released=dict(clear_confirmed=True,release_authority=True,holding_called=False,monitoring_active=True,posture_mode='fully_open_after_clear',recontact_count=0)
    unknown=dict(released,observation_unknown=True);checks['unknown_monitor_latched_release_accepted']=verify_clear_latch([released,unknown])['passed']
    reclose=dict(released,clear_confirmed=False,posture_mode='public_thumb_clearance',recontact_count=1);checks['postclear_reclose_rejected']=not verify_clear_latch([released,reclose])['passed']
    return dict(checker_self_tests_passed=all(checks.values()),checks=checks,production_verified=False,physics_steps_executed=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path);a=p.parse_args();out=checker_self_test()
    if a.contract:
        path=a.contract.resolve();before=sha(path);spec=importlib.util.spec_from_file_location('independent_candidate_task_contract',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        out['real_builder']=verify_real_builder(m.build_task_request);out['real_modes']=verify_real_modes(m);out['source_sha256']=before;out['source_unchanged']=sha(path)==before;out['static_boundary']=static_boundary(path)
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()

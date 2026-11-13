"""Independent pure/saved-stream audit; no plant is constructed or advanced."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from types import SimpleNamespace
import json,hashlib,importlib.util,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from task_contract import validate_task_request,build_task_request
from task_trajectory import validate_preview,unknown_future_point,public_transport_trajectory
from evaluation import force_band,read_protocol
from research_imports import source_identity
from design_validation import synthetic


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def rejects(value):
    try:validate_preview(value);return False
    except ValueError:return True


def main():
    files=['task_contract.py','task_trajectory.py','runtime.py','research_imports.py','evaluation.py','protocol.json']
    before={f:sha(ROOT/f) for f in files};p=read_protocol();checks={};details={};issues=[]
    base=ROOT/'evidence/runtime_v6_baseline_equivalence'
    cert=json.loads((base/'certificate.json').read_text(encoding='utf-8'))
    current=source_identity(['ratio','unified']);saved=cert['source_sha256']
    details['physical_run_source_vs_current']={f:dict(executed=saved.get(f),current=current.get(f))
        for f in sorted(set(saved)|set(current)) if saved.get(f)!=current.get(f)}
    checks['physical_certificate_passes_73_checks']=cert['passed'] and len(cert['checks'])==73
    old_eval=ROOT/'output/development_v5_pilot/sources/evaluation.py'
    spec=importlib.util.spec_from_file_location('saved_v2_evaluation',old_eval)
    v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
    count=0;task_rows={};input_hashes={}
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():continue
        trace_path=folder/'trace.npz';old_hash=sha(trace_path)
        with np.load(trace_path,allow_pickle=False) as z:
            requests=z['task_request_json'];times=z['time'];checked=0
            for raw in requests:
                request=json.loads(str(raw));rebuilt=validate_task_request(request)
                if request!=rebuilt:raise AssertionError('Saved public contract changed')
                checked+=1
            checks[folder.name+'/every_saved_task_dict_unchanged']=checked==3201
            count+=checked;task_rows[folder.name]=checked
            raw={f:z[f] for f in ['time','normal','substep_time','substep_duration','substep_normal','substep_skin_normal']}
            metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            b3,c3=force_band(raw,metadata,p);b2,c2=v2.force_band(raw,metadata,p)
            checks[folder.name+'/v3_complete_force_coverage']=c3['valid'] and np.isfinite(b3).all()
            for window,(lo,hi) in {'static':(8.2,8.9),'dynamic':(9.3,11.7)}.items():
                mask=(times>=lo-1e-9)&(times<=hi+1e-9)
                checks[folder.name+'/'+window+'_filter_bitwise_v2']=np.array_equal(b3[mask],b2[mask])
        input_hashes[str(trace_path.relative_to(ROOT))]=old_hash
        checks[folder.name+'/trace_file_unchanged']=old_hash==sha(trace_path)
    details['saved_task_dicts_validated']=count;details['per_trace']=task_rows
    checks['all_12804_saved_task_dicts_validated']=count==12804
    u=unknown_future_point(10.,[.1,0.,0.]);v=deepcopy(u)
    v.update(requested_times_s=[10.,10.02],trajectory_times_s=[10.,10.],
        motion_targets=[[.1,0.,0.],[.09,.1,.1]],active_user_motion=[True,True])
    checks['unknown_future_cannot_fabricate_waypoints']=rejects(v)
    v=public_transport_trajectory().preview(11.98);v['motion_targets'][-1]=[.08,.12,.03]
    checks['known_preview_tail_keeps_continuous_endpoint']=rejects(v)
    v=deepcopy(u);v['available_through_s']=-1.;v['trajectory_times_s']=[-1.]
    checks['preview_end_cannot_be_negative']=rejects(v)
    v=public_transport_trajectory().preview(12.);v['future_reference_available']=True
    checks['completed_stage_cannot_claim_available_future']=rejects(v)
    for label,x in [('nan',float('nan')),('inf',float('inf')),('boolean',True)]:
        v=deepcopy(u);v['motion_targets'][0][0]=x;checks['preview_rejects_'+label]=rejects(v)
    for field in ['object_pose','future_force','true_friction','dropout_schedule']:
        v=deepcopy(u);v[field]=0.;checks['preview_rejects_hidden_'+field]=rejects(v)
    r=build_task_request(10.,'hold',[1.,0.,0.],[0.,0.,0.],12.,[5.]*5,18.);r['motion_error'][0]=True
    try:validate_task_request(r);accept_bool=True
    except ValueError:accept_bool=False
    checks['derived_motion_error_rejects_boolean']=not accept_bool
    if accept_bool:issues.append(dict(id='derived-vector-bool',scope='Strict schema only; generated runtime contracts use floats',
        description='True in motion_error is accepted when expected value is 1.0 due Python equality'))
    v=deepcopy(u);v['motion_targets'][0][0]=np.bool_(True)
    checks['numeric_preview_rejects_numpy_boolean']=rejects(v)
    if not rejects(v):issues.append(dict(id='numpy-bool-numeric-input',scope='Non-JSON Python caller only',
        description='np.bool_ passes scalar conversion and becomes 1.0'))
    synt,meta=synthetic();a,_=force_band(synt,meta,p)
    late=deepcopy(synt);late['substep_normal'][synt['substep_time']>=12.,0]=100.
    b,fc=force_band(late,meta,p);mask=(synt['time']>=7.5-1e-9)&(synt['time']<=12.+1e-9)
    checks['scheduled_release_future_force_cannot_change_holding']=np.array_equal(a[mask],b[mask])
    checks['holding_retains_all_901_points_and_preaction_endpoint']=int(mask.sum())==901 and fc['holding_endpoint'].startswith('pre-action')
    early_meta=deepcopy(meta);early_meta['release_at']=8.
    ea,_=force_band(synt,early_meta,p);el=deepcopy(synt);el['substep_normal'][synt['substep_time']>=8.,0]=100.
    eb,ec=force_band(el,early_meta,p);mask=(synt['time']>=7.5-1e-9)&(synt['time']<=8.+1e-9)
    early_diff=float(np.max(abs(ea[mask]-eb[mask])))
    checks['early_manual_release_force_cannot_enter_prior_holding']=early_diff==0.
    checks['early_manual_release_clock_is_actual_phase_boundary']=8. in ec['phase_boundaries_s']
    if early_diff>0.:issues.append(dict(id='manual-early-release-filter-boundary',scope='Manual early release; scheduled 12s matrices unaffected',
        description='FIR uses protocol.release_s instead of metadata.release_at; early-release force enters actual holding',
        counterexample_release_at_s=8.,maximum_filtered_difference_N=early_diff,declared_boundaries_s=ec['phase_boundaries_s']))
    for release in (2.,7.5,8.,12.,16.):
        public=deepcopy(meta);public['release_at']=release
        aa,ca=force_band(synt,public,p);counter=deepcopy(synt)
        counter['substep_normal'][synt['substep_time']>=release,0]=100.
        bb,cb=force_band(counter,public,p);prior=synt['time']<=release+1e-9
        checks[f'actual_release_{release}_filters_finite_with_exact_boundary']=ca['valid'] and np.isfinite(aa).all() and release in ca['phase_boundaries_s']
        checks[f'actual_release_{release}_future_does_not_enter_prior_stage']=np.array_equal(aa[prior],bb[prior])
    device=dict(force_sigma_N=.01,tangential_sigma_N=.005,position_sigma_m=.0001,slip_sigma_m_s=.0002,effort_sigma=.02,tactile_delay_s=.01)
    old_request=build_task_request(10.,'hold',[.1,0.,0.],[.1,0.,0.],12.,[5.]*5,18.,sensor_characteristics=device)
    checks['legacy_six_device_keys_remain_exact_without_injected_defaults']=old_request['sensor_characteristics']==device and validate_task_request(old_request)==old_request
    extended=dict(device,auxiliary_force_sigma_N=.03,auxiliary_moment_sigma_Nm=.004)
    new_request=build_task_request(10.,'hold',[.1,0.,0.],[.1,0.,0.],12.,[5.]*5,18.,sensor_characteristics=extended)
    checks['explicit_effective_auxiliary_device_sigmas_roundtrip_exactly']=new_request['sensor_characteristics']==extended and validate_task_request(new_request)==new_request
    for field in ('auxiliary_force_sigma_N','auxiliary_moment_sigma_Nm'):
        bad=dict(extended);bad[field]=float('nan')
        try:build_task_request(10.,'hold',[.1,0.,0.],[.1,0.,0.],12.,[5.]*5,18.,sensor_characteristics=bad);rejected=False
        except ValueError:rejected=True
        checks[field+'_rejects_nonfinite']=rejected
    for label,x in [('nan',float('nan')),('boolean',True),('numpy_boolean',np.bool_(False)),('string','0.1')]:
        sample=dict(device,force_sigma_N=x)
        try:build_task_request(10.,'hold',[.1,0.,0.],[.1,0.,0.],12.,[5.]*5,18.,sensor_characteristics=sample);rejected=False
        except ValueError:rejected=True
        checks['public_sensor_spec_rejects_'+label]=rejected
    from runtime import public_motion_reference_bounds
    class RobotModel:
        body_subtreemass=np.array([3.]);jnt_limited=np.ones(3,bool);actuator_ctrllimited=np.ones(3,bool)
        jnt_range=np.array([[0.,.12],[-.5,.2],[-.7,.5]])
        actuator_ctrlrange=np.array([[-.02,.11],[-.4,.15],[-.6,.4]])
        def joint(self,name):return SimpleNamespace(id=['transport_lift','rh_WRJ2','rh_WRJ1'].index(name))
        def actuator(self,name):return SimpleNamespace(id=['lift_motor','rh_A_WRJ2','rh_A_WRJ1'].index(name))
    bound=public_motion_reference_bounds(SimpleNamespace(model=RobotModel(),hand=0,config=SimpleNamespace(lift_target=.1)))
    checks['robot_motion_bounds_intersect_joint_ctrl_and_gravity_offset']=np.allclose(bound['lower'],[0.,-.4,-.6],atol=1e-15,rtol=0.) and np.allclose(bound['upper'],[.11-3.*9.81/1500.,.15,.4],atol=1e-15,rtol=0.)
    checks['reviewed_sources_stable_during_audit']=before=={f:sha(ROOT/f) for f in files}
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(schema='independent-v6-public-runtime-evaluation-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed_checks=all(checks.values()),checks=checks,check_count=len(checks),unresolved_findings=issues,details=details,
        reviewed_source_sha256=before,input_trace_sha256=input_hashes,old_v2_evaluator_sha256=sha(old_eval),
        scope='Pure functions and saved trace validation only; no new plant/physical steps/performance trials. Four physical runs preceded stricter validation; their executed source certificate is retained, never replaced with current hashes.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'review'/f'independent_v6_public_runtime_evaluation_{stamp}.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed_checks=result['passed_checks'],checks=len(checks),task_dicts=count,
        issues=issues,source_changes=details['physical_run_source_vs_current'],output=str(out)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

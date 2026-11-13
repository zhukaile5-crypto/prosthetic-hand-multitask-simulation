"""Independent saved-data arithmetic audit; no MuJoCo/plant imports or stepping."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'calibration/phase3_path_validation_v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-9):return bool(np.allclose(a,b,atol=tol,rtol=0,equal_nan=False))
def serial(a):
    if isinstance(a,np.ndarray):return a.tolist()
    if isinstance(a,np.generic):return a.item()
    if isinstance(a,dict):return {k:serial(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)):return [serial(v) for v in a]
    return a

def effect(kernel,kt,times,values,queries):
    """Independent vectorized known-command convolution, no output inputs."""
    delta=np.diff(values,axis=0);starts=np.asarray(times)[:-1]
    ages=np.asarray(queries)[:,None]-starts[None,:];active=ages>1e-10
    result=np.zeros((len(queries),16))
    for axis in range(8):
        use=np.flatnonzero(delta[:,axis]!=0.)
        if not len(use):continue
        age=ages[:,use];mask=active[:,use]
        for output in range(16):
            response=np.interp(age,kt,kernel[:,output,axis],left=0.,right=kernel[-1,output,axis])*mask
            result[:,output]+=response@delta[use,axis]
    return result

def rolling_errors(times,measured,history_times,history_y,modeled,history_modeled,plan):
    grouped=defaultdict(list);details=[]
    for origin,now in enumerate(times):
        tt=np.r_[history_times,times[:origin+1]]
        yy=np.vstack((history_y-history_modeled,measured[:origin+1]-modeled[:origin+1]))
        keep=(tt>=now-plan['causal_window_s']-1e-10)&(tt<=now+1e-10);tt=tt[keep];yy=yy[keep]
        if len(tt)>=4:
            center=tt-tt.mean();slope=np.sum(center[:,None]*(yy-yy.mean(0)),axis=0)/max(1e-12,float(center@center))
            level=yy.mean(0)+slope*(now-tt.mean())
        else:slope=np.zeros(16);level=yy[-1]
        for h in plan['forecast_horizons_s']:
            j=origin+round(h/plan['interval_s'])
            if j<len(times):
                e=level+h*slope+modeled[j]-measured[j]
                grouped[str(float(h))].append(e);details.append((float(now),float(h),e))
    return grouped,details

def score(errors,plan):
    e=np.asarray(errors);rmse=np.sqrt(np.mean(e*e,axis=0));p95=np.quantile(abs(e),.95,axis=0);g=plan['model_gates']
    limit=np.r_[np.full(5,g['pressure_rmse_N']),np.full(5,g['shear_rmse_N']),np.full(3,g['force_axis_rmse_N']),np.full(3,g['moment_axis_rmse_Nm'])]
    passed=rmse<=limit;passed[:5]&=p95[:5]<=g['pressure_p95_N'];passed[5:10]&=p95[5:10]<=g['shear_p95_N']
    return dict(vectors=len(e),rmse_per_output=rmse,p95_abs_per_output=p95,max_abs_per_output=abs(e).max(0),engineering_gate_per_output=passed,all_engineering_gates_passed=bool(passed.all()))

def score_matches(own,saved):
    return all(close(value,saved[key],1e-9) if isinstance(value,np.ndarray) else value==saved[key] for key,value in own.items())

def raw_wrench(patches):
    pressure=np.zeros(5);shear=np.zeros(5);w=np.zeros(6)
    for p in patches:
        n=max(0.,float(p['normal']))
        if n<=1e-10:continue
        f=np.asarray(p['force']);x=np.asarray(p['position']);d=np.asarray(p['normal_dir']);d=d/np.linalg.norm(d)
        w[:3]+=f;w[3:]+=np.cross(x,f)+np.asarray(p['torque'])
        digit=int(p['digit'])
        if 0<=digit<5:pressure[digit]+=n;shear[digit]+=np.linalg.norm(f-n*d)
    return np.r_[pressure,shear,w]

def main():
    plan=read(DATA/'plan.json');summary=read(DATA/'summary.json');finish=read(DATA/'execution_complete.json');sources=read(DATA/'source_snapshot_manifest.json')['files'];checks={};records=[];unavailable=[];worst=np.zeros(16);worst_p95=np.zeros(16);differential_fail=[];ranges={};replay=[]
    def check(k,v):checks[k]=bool(v)
    check('execution_identity_and_development_scope',summary==finish and finish['execution_complete'] and finish['sources_unchanged'] and not finish['changed_sources'] and finish['model_admitted'] is False and finish['input_kernel_modified'] is False and finish['calibration_not_performance'])
    check('nominal_three_shape_frozen_plan',plan['status']=='frozen_before_phase3_physical_data' and plan['mass_setup_kg']==.12 and plan['physical_randomization'] is False and plan['model_admitted'] is False and len(plan['shapes'])==3)
    check('declared_plan_identity',sha(DATA/'plan.json')==read(DATA/'environment.json')['plan_sha256'])
    check('all65_source_keys',len(sources)==len(plan['sources'])==65 and set(sources)==set(plan['sources']))
    for key,item in sources.items():
        snapshot=DATA/item['snapshot_path'];declared=plan['sources'][key]
        check('source/'+key,snapshot.is_file() and sha(snapshot)==item['sha256']==item['snapshot_sha256']==declared['sha256'] and snapshot.stat().st_size==declared['bytes'] and sha(Path(declared['path']))==declared['sha256'])
    model_path=DATA/sources['research/calibration/public_contact_model_phase2_v1.json']['snapshot_path'];model=read(model_path)
    check('kernel_identity_unchanged',sha(model_path)==plan['input_model']['sha256']=='af29469115036d673a395d105306a42fd232fe278c8fb5194e07fcfbd148baee')
    previous=plan['revision_disclosure'];check('v2_failed_evidence_retained',sha(previous['previous_plan'])==previous['previous_plan_sha256'] and sha(Path(previous['previous_execution'])/'summary.json')==previous['previous_execution_summary_sha256'] and read(Path(previous['previous_execution'])/'summary.json')['execution_complete'] is False and sha(previous['archived_v2_script'])==previous['archived_v2_script_sha256'])
    expected_labels={'zero','zero_replay'}|{f'micro_{a}_{s}' for a in range(8) for s in ['plus','minus']}|{f'mixed_{i:02d}' for i in range(8)}
    for shape in plan['shapes']:
        setup=read(DATA/shape/'setup.json');anchor=read(DATA/shape/'independent_height_gate_reference.json');config=setup['config'];marker=np.asarray(anchor['initial_object_in_palm_marker_m']);minimum=np.full(8,np.inf);maximum=np.full(8,-np.inf);actual_min=np.full(3,np.inf);actual_max=np.full(3,-np.inf)
        check(shape+'/actual_standard_config',config['mass']==.12 and config['friction']==.6 and config['seed']==plan['seed'] and config['shape']==shape and setup['preparation_controller_acted'] is False and sha(DATA/shape/'preparation_trace.npz')==setup['preparation_trace_sha256'])
        entry=next(x for x in model['shapes'] if x['shape']==shape)['coordinates']['current_palm'];kernel=np.asarray(entry['response_to_20ms_interval_increment']);kt=np.asarray(entry['response_kernel_times_s'])
        for cp in sorted((DATA/shape).glob('checkpoint_*')):
            checkpoint=read(cp/'checkpoint.json');cp_summary=read(cp/'summary.json');key=shape+'/'+cp.name
            branches={p.name:p for p in cp.iterdir() if p.is_dir()};check(key+'/all26_labels_retained',set(branches)==expected_labels and cp_summary['planned_branches']==26)
            zero_path=cp/'zero/trace.npz';repeat_path=cp/'zero_replay/trace.npz'
            with np.load(zero_path,allow_pickle=False) as a,np.load(repeat_path,allow_pickle=False) as b:
                exact=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
                zero_y=a['y_palm'][::4];check(key+'/zero_full_saved_values_bitwise',exact);replay.append(dict(shape=shape,checkpoint=cp.name,exact=exact,control_samples=len(a['time']),physical_substeps=len(a['physical_time'])))
            prefix_t=np.asarray(checkpoint['reference_history_times']);prefix_v=np.asarray(checkpoint['reference_history_values']);history=checkpoint['causal_history'];ht=np.array([x['time'] for x in history if x['time']<checkpoint['time_s']-1e-10]);hy=np.array([x['y_palm'] for x in history if x['time']<checkpoint['time_s']-1e-10]).reshape(-1,16)
            valid_labels=[];checkpoint_unavailable=[];summary_by_branch={x['branch']:x for x in cp_summary['branch_scores']}
            for label,branch in branches.items():
                m=read(branch/'metadata.json');bk=key+'/'+label;requested=np.asarray(m['requested_path']);lower=np.asarray(m['lower']);upper=np.asarray(m['upper']);outside=np.flatnonzero(np.any((requested<lower-1e-12)|(requested>upper+1e-12),axis=0))
                if not m['available']:
                    check(bk+'/unavailable_no_physics_or_zero_imputation',not m['attempted_physics'] and not (branch/'trace.npz').exists() and len(outside)>0 and np.array_equal(outside,m['unavailable_inputs']))
                    unavailable.append(dict(shape=shape,checkpoint=cp.name,branch=label,axes=outside.tolist(),reason=m['unavailable_reason']));checkpoint_unavailable.append(label);continue
                valid_labels.append(label);check(bk+'/available_complete_within_bounds',m['attempted_physics'] and m['complete'] and m['failure'] is None and not len(outside))
                with np.load(branch/'trace.npz',allow_pickle=False) as a:
                    z={k:a[k] for k in a.files if k not in ['metadata','raw_patches','physical_raw_patches']}
                    if label in ['zero','mixed_00']:
                        residual=max(np.max(abs(raw_wrench(json.loads(str(v)))-y)) for field,yfield in [('raw_patches','y_palm'),('physical_raw_patches','physical_y_palm')] for v,y in zip(a[field],z[yfield]))
                        check(bk+'/all_patch16_reconstruction',residual<=1e-9)
                check(bk+'/trace_identity',sha(branch/'trace.npz')==m['trace_sha256'])
                t=z['time'];pt=z['physical_time'];dt=np.asarray(z['physical_duration']);check(bk+'/complete_end_clocks',len(t)==101 and len(pt)==1000 and close(t-t[0],np.arange(101)*.005,1e-8) and close(pt-t[0],np.arange(1,1001)*.0005,1e-8) and close(z['physical_interval_start']+dt,pt,1e-12) and close(dt,.0005,1e-12))
                check(bk+'/finite_saved_numeric_states',all(np.isfinite(v).all() for v in z.values() if v.dtype.kind in 'fiu'))
                executed=np.asarray(m['executed_path']);interp=np.vstack([requested[0],*[requested[i]+j/4*(requested[i+1]-requested[i]) for i in range(25) for j in range(1,5)]])
                check(bk+'/actual_4tick_input_history',close(z['input_reference'],executed,1e-12) and close(executed,interp,1e-10))
                minimum=np.minimum(minimum,z['input_reference'].min(0));maximum=np.maximum(maximum,z['input_reference'].max(0));actual_min=np.minimum(actual_min,z['physical_gate_actual_robot_motion_end'].min(0));actual_max=np.maximum(actual_max,z['physical_gate_actual_robot_motion_end'].max(0))
                for prefix in ['', 'physical_']:
                    press=z[prefix+'pressure'];sy=z[prefix+'shear_magnitude'];wrench=z[prefix+'wrench_palm'];R=z[prefix+'gate_palm_rotation_end'];palm=z[prefix+'gate_palm_position_end'];obj=z[prefix+'gate_object_position_end']
                    rel=np.einsum('nji,nj->ni',R,obj-palm);drift=np.linalg.norm(rel-marker,axis=1);delta=(palm+np.einsum('nij,j->ni',R,marker))[:,2]-anchor['actual_anchor_marker_z_m'];floor=np.maximum(.025,.07+delta)
                    check(bk+'/'+prefix+'component_outputs',close(z[prefix+'y_palm'],np.column_stack([press,sy,wrench]),1e-12))
                    check(bk+'/'+prefix+'height_relative_gate_recalculated',close(rel,z[prefix+'gate_object_relative_position_end'],1e-12) and close(drift,z[prefix+'gate_relative_drift_m'],1e-12) and close(delta,z[prefix+'gate_actual_robot_delta_z_m'],1e-12) and close(floor,z[prefix+'gate_minimum_object_lift_m'],1e-12))
                    actual=z[prefix+'gate_actual_robot_motion_end'];ref=z[prefix+'input_reference'];error=actual-ref[:,5:];check(bk+'/'+prefix+'robot_tracking_separate',close(error,z[prefix+'gate_robot_tracking_error'],1e-12) and close(actual,np.column_stack([z[prefix+'robot_lift_position_m'],z[prefix+'robot_wrist_position_rad']]),1e-12) and np.array_equal(abs(error)>np.array([.01,.015,.015]),z[prefix+'gate_robot_tracking_outside']) and close(z[prefix+'gate_geometry_time_s'],z[prefix+'time'],1e-12))
                    check(bk+'/'+prefix+'independent_raw_engineering_guard',np.all(press<=5.02) and np.all(press.sum(1)<=18.02) and np.all(z[prefix+'penetration']<=.002) and np.all(z[prefix+'table_support_N']<=.05) and np.all(z[prefix+'object_lift_m']>=floor) and np.all(drift<=.005) and np.all((press>.2).sum(1)>=3) and np.all(press[:,0]>.2) and not np.any(z[prefix+'gate_robot_tracking_failed']) and not np.any(z[prefix+'warning_count']))
                # Absolute-time independent tracking dwell, restored per branch.
                onset=list(checkpoint['independent_gate_timer_checkpoint']['tracking']['outside_since']);expected=[]
                for now,e in zip(np.r_[t[0],pt],np.vstack([z['gate_robot_tracking_error'][0],z['physical_gate_robot_tracking_error']])):
                    duration=np.zeros(3)
                    for axis,bad in enumerate(abs(e)>np.array([.01,.015,.015])):
                        if bad:
                            if onset[axis] is None:onset[axis]=now
                            duration[axis]=max(0.,now-onset[axis])
                        else:onset[axis]=None
                    expected.append(duration)
                expected=np.asarray(expected);check(bk+'/tracking_dwell_actual_time',close(expected[1:],z['physical_gate_robot_tracking_dwell_s'],1e-10) and close(np.vstack([expected[0],expected[10::10]]),z['gate_robot_tracking_dwell_s'],1e-10))
                if label not in ['zero','zero_replay']:
                    validation=read(branch/'validation.json');selected_t=t[::4];selected_y=z['y_palm'][::4];bt=np.r_[prefix_t[-1],t[0]+np.arange(1,26)*.02];queries=np.r_[ht,selected_t]
                    baseline=effect(kernel,kt,prefix_t,prefix_v,queries);branch_effect=effect(kernel,kt,bt,requested,queries);modeled=baseline+branch_effect
                    errors,details=rolling_errors(selected_t,selected_y,ht,hy,modeled[len(ht):],modeled[:len(ht)],plan)
                    check(bk+'/all_causal_raw_errors_recomputed',len(details)==len(validation['error_vectors']) and all(abs(now-v['origin_time_s'])<1e-10 and h==v['horizon_s'] and close(e,v['error'],1e-9) for (now,h,e),v in zip(details,validation['error_vectors'])))
                    for horizon,e in errors.items():
                        own=score(e,plan);saved=validation['complete_causal_by_horizon'][horizon]
                        check(bk+'/causal_metric/'+horizon,score_matches(own,saved) and score_matches(own,summary_by_branch[label]['complete_causal_by_horizon'][horizon]))
                        worst=np.maximum(worst,own['rmse_per_output']);worst_p95=np.maximum(worst_p95,own['p95_abs_per_output'])
                    de=selected_y-zero_y-branch_effect[len(ht):];ds=score(de,plan);check(bk+'/differential_metric_separate',score_matches(ds,validation['differential_response_error']) and validation['differential_uses_future_zero_offline_only'] is True and validation['model_admitted'] is False)
                    if not ds['all_engineering_gates_passed']:differential_fail.append(dict(shape=shape,checkpoint=cp.name,branch=label,failed_outputs=np.flatnonzero(~ds['engineering_gate_per_output']).tolist()))
                records.append(dict(shape=shape,checkpoint=cp.name,branch=label,trace_sha256=m['trace_sha256'],min_height_margin_m=float(np.min(z['physical_object_lift_m']-z['physical_gate_minimum_object_lift_m'])),max_relative_drift_m=float(z['physical_gate_relative_drift_m'].max()),max_pressure_N=float(z['physical_pressure'].max()),max_penetration_m=float(z['physical_penetration'].max())))
            check(key+'/checkpoint_summary_counts',cp_summary['complete'] and cp_summary['executed_branches']==len(valid_labels) and sorted(cp_summary['unavailable_branches'])==sorted(checkpoint_unavailable) and set(summary_by_branch)==set(valid_labels)-{'zero','zero_replay'})
            print(key,'raw_branches',len(valid_labels),'recomputed',flush=True)
        ranges[shape]=dict(reference_min=minimum,reference_max=maximum,actual_robot_min=actual_min,actual_robot_max=actual_max)
    check('global390_382_8_count',len(records)==sum(c['executed_branches'] for c in summary['checkpoints'])==382 and len(unavailable)==8 and len(replay)==15 and len(records)+len(unavailable)==plan['planned_branches']==390)
    result=dict(schema='independent_phase3_v3_saved_data_audit_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(DATA/'plan.json'),summary_sha256=sha(DATA/'summary.json'),source_manifest_sha256=sha(DATA/'source_snapshot_manifest.json'),reviewer_sha256=sha(__file__),physics_steps_executed=0,model_admitted=False,records=records,zero_replays=replay,unavailable=unavailable,ranges=ranges,worst_branch_causal_rmse_per_output=worst,worst_branch_causal_p95_per_output=worst_p95,differential_failed_branches=differential_fail,limitations=['Standard .12kg/.6 nominal morphology-specific local response after one declared path; not broad random physics, unknown shapes, collision, sliding or release evidence.','All 352 scored branch causal errors and metrics recomputed from saved control outputs/history and unchanged kernel; raw all-patch reconstruction additionally covers zero and mixed_00 for every checkpoint.','The accepted estimator combines frozen kernel with causal past-residual level/slope; a passing summary is not admission of the bare kernel or a global hard error bound.','V3 transportation height uses actual robot end-FK with 25mm floor and5mm relative retention, plus separate user tracking; v1/v2 failures and semantic revisions remain preserved.'])
    out=ROOT/'review'/('independent_phase3_v3_saved_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(serial(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),worst_rmse=worst,worst_p95=worst_p95,differential_failures=len(differential_fail),ranges=ranges)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

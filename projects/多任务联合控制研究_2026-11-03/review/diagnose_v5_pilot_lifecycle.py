"""Offline diagnosis of the completed 36-run single-seed V5 development pilot.

No MuJoCo stepping, controller optimization, source changes or model fitting.
Only a fresh review artifact is written by the explicit main entry point.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/development_v5_pilot'
SCENES=('nominal','sphere','sensor_noise','dynamic_transport','fragile_conflict','contact_robustness')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def first_clear(times,normal,threshold=.2,dwell=.2):
    start=None
    for t,n in zip(times,normal):
        if n<=threshold:
            if start is None:start=float(t)
            if t-start>=dwell-1e-10:return float(t)
        else:start=None
    return None


def analyze():
    completion=json.loads((DATA/'execution_complete.json').read_text(encoding='utf-8'))
    assert completion['execution_complete'] and completion['trials']==36 and completion['sources_unchanged']
    production=('coordinator.py','coordinator_model.py','coordinator_contacts.py','runtime.py','evaluation.py','task_contract.py','protocol.json',
                'lifecycle_release.py','lifecycle_plant.py','calibration/public_contact_model_phase2_v1.json')
    sources={name:sha(ROOT/name) for name in production};inputs={};paired_release=[];holding=[];tracking=[]
    def load(scene,method):
        run=DATA/'runs'/f'{scene}__{method}__610201'
        paths={name:run/name for name in ('trace.npz','metadata.json','result.json')}
        result=json.loads(paths['result.json'].read_text(encoding='utf-8'));assert result['record_complete']
        for path in paths.values():inputs[str(path)]=sha(path)
        return np.load(paths['trace.npz'],allow_pickle=False),json.loads(paths['metadata.json'].read_text(encoding='utf-8')),result,str(run)
    for scene in SCENES:
        for method in ('ratio','passive','joint'):
            a,m,r,run=load(scene,method);quality=r['research_metrics']['dynamic_quality']
            holding.append(dict(scene=scene,method=method,mission_success=r['task_success'],
                static_TV_N_s=r['finger_tv_rate'],dynamic_TV_N_s=quality['bandlimited']['finger_TV_rate_N_s'],
                dynamic_mean_force_N=quality['mean_actual_finger_force_N'],
                holding_excess_Ns=r['safety']['stages']['policy_holding']['local_excess_integral_Ns'],source=run))
            if scene=='dynamic_transport':
                t=a['time'];w=(t>=9.3-1e-9)&(t<12.-1e-9);target=a['actual_motion_target'][w]
                ref=a['executed_motion_reference'][w];actual=np.c_[a['lift'][w],a['wrist_position'][w]]
                tracking.append(dict(method=method,window_s=[9.3,12.],right_endpoint_excluded=True,samples=int(w.sum()),
                    coordinate_units=['m','rad','rad'],reference_RMSE=np.sqrt(np.mean((ref-target)**2,axis=0)).tolist(),
                    actual_encoder_RMSE=np.sqrt(np.mean((actual-target)**2,axis=0)).tolist(),
                    actual_encoder_max_abs=np.max(np.abs(actual-target),axis=0).tolist(),
                    full_motion_mission_success=r['research_metrics']['full_motion_mission_success'],
                    wrist_guard=r['research_metrics']['wrist_tracking_guard']))
        for method in ('ratio_release','joint_release'):
            a,m,r,run=load(scene,method);t=a['time'];st=a['substep_time'];dt=a['substep_duration'];skin=a['substep_skin_normal']
            mask=st>=12.-1e-9;indices=np.flatnonzero(mask);v=skin[mask];caps=np.asarray(m['limits'])
            ij=np.unravel_index(np.argmax(v),v.shape);tick=indices[ij[0]];support=a['substep_support']
            table=indices[support[mask]>.02];k=int(np.argmin(np.abs(t-12.)));debug=a['debug_json']
            d=json.loads(str(debug[k]));obs=json.loads(str(a['observation_json'][k]));patches=a['raw_patches_json']
            rt=np.flatnonzero(t>=12.-1e-9);full_skin=[];non_target_aux=[]
            for row in rt:
                pp=json.loads(str(patches[row]));full_skin.append(sum(float(x['normal']) for x in pp))
                non_target_aux.append(sum(float(x['normal']) for x in pp if int(x['digit'])<0 and not x.get('is_target',False)))
            physical_clear=first_clear(t[rt],full_skin)
            recontact=None
            if physical_clear is not None:
                hit=[float(t[row]) for row,n in zip(rt,full_skin) if t[row]>physical_clear+1e-9 and n>.2]
                if hit:recontact=hit[0]
            released=[json.loads(str(debug[row])) for row in rt]
            public_clear=next((float(x['first_clear_confirmation']) for x in released if x.get('first_clear_confirmation') is not None),None)
            qids=np.asarray(m['raw_joint_qpos_indices'],int)
            paired_release.append(dict(scene=scene,method=method,source=run,primitive=m['release_primitive_selection']['selected'],
                material_limits_N=caps.tolist(),budget_N=m['budget'],whole_force_guard_pass=r['safety']['force_protection_pass'],
                peak_N=float(v[ij]),peak_finger=int(ij[1]),peak_time_end_s=float(st[tick]),
                target_pressure_at_peak_N=float(a['substep_normal'][tick,ij[1]]),table_support_at_peak_N=float(support[tick]),
                first_table_support_time_end_s=None if not len(table) else float(st[table[0]]),table_threshold_N=.02,
                per_finger_peak_N=v.max(axis=0).tolist(),
                release_local_excess_Ns=float(np.sum(np.maximum(v-caps,0.)*dt[mask,None])),
                release_total_budget_excess_Ns=float(np.sum(np.maximum(v.sum(axis=1)-m['budget'],0.)*dt[mask])),
                above_any_local_limit_duration_s=float(dt[mask][np.any(v>caps,axis=1)].sum()),
                full_raw_patch_clear_confirmation_200Hz_s=physical_clear,full_raw_patch_recontact_after_confirmation_200Hz_s=recontact,
                maximum_non_target_aux_pressure_200Hz_N=max(non_target_aux),public_sensor_clear_confirmation_s=public_clear,
                official_release=r['research_metrics']['release'],entry=dict(time_s=float(t[k]),alpha=a['alpha_before'][k].tolist(),
                    raw_robot_joint_position=a['qpos'][k,qids].tolist(),actuator_before=d['actuator_before'],
                    lift_reference=float(obs['lift_ref']),wrist_reference=a['executed_motion_reference'][k,1:].tolist(),
                    observed_pressure=obs['normal'],observed_shear=obs['shear_magnitude'],sample_time_s=obs['sample_time'],
                    true_finger_pressure_for_evaluation=a['skin_normal'][k].tolist(),
                    relative_speed_for_evaluation_m_s=float(np.linalg.norm(a['relative_velocity'][k])),
                    labels='Public observations for possible online entry checks; true force/relative speed are evaluation only')))
    # Existing forecast logs permit exact decomposition without fitting or
    # rerunning the policy. The first trajectory difference is invertible.
    a,m,r,run=load('fragile_conflict','joint');t=a['time'];debug=a['debug_json'];skin=a['skin_normal'];fragile=[]
    for when in (9.32,9.36,9.50,9.54,9.56,9.60,10.,10.4):
        k=int(np.argmin(np.abs(t-when)));d=json.loads(str(debug[k]));nom=np.asarray(d['nominal_force_prediction']);h=2;i=4
        parts={name:float(np.asarray(d[key])[h,i]) for name,key in (('calibration','pressure_calibration_error_N'),
            ('residual_fit','pressure_residual_fit_error_N'),('slope','pressure_slope_error_N'),
            ('memory_change','pressure_past_input_memory_change_N'),('drift_change','pressure_residual_drift_change_N'))}
        parts['fixed_margin']=float(d['forecast_margin_N']);unc=sum(parts[x] for x in ('calibration','residual_fit','slope','fixed_margin'))
        chosen=nom+np.cumsum(np.asarray(d['predicted_extra_pressure_variation_N']),axis=0)
        assert np.allclose(chosen[0],d['force_prediction'],atol=1e-8)
        fragile.append(dict(time=float(t[k]),true_LF_skin_N=float(skin[k,4]),estimated_LF_now_N=d['pressure_now'][4],
            nominal_LF_60ms_N=float(nom[h,i]),nominal_LF_upper60_N=float(nom[h,i]+unc),
            chosen_LF_60ms_N=float(chosen[h,i]),uncertainty_total60_N=unc,parts60_N=parts,
            nominal_failed_checks=d['nominal_failed_checks'],correction_alpha=d['correction_8'][:5],
            force_scope='all instrumented finger external pressure; no true-state input to controller'))
    assert sources=={name:sha(ROOT/name) for name in production}
    assert all(sha(Path(path))==value for path,value in inputs.items())
    return dict(schema='v5_pilot_lifecycle_diagnosis_v1',phase='development',physical_noise_seed=610201,
        generalization_claim_allowed=False,source_sha256=sources,input_sha256=inputs,sources_and_inputs_unchanged=True,
        original_completion=completion,holding=holding,dynamic_tracking=tracking,releases=paired_release,fragile_forecast_chain=fragile,
        interpretation=['Single development seed and seen cases; no closed-loop counterfactual or new physics.',
            'Substep skin_normal covers all finger external contacts. Substep palm/other are target-specific.',
            'Full raw-patch clearance is checked only at200Hz. No claim of all-auxiliary2000Hz clearance is made.',
            'Object-only substep pressure equals the reported peak finger pressure in these release cases.',
            'Small one-substep guard failures are preserved; no matched dt check from these joint terminal states exists.'])


if __name__=='__main__':
    path=ROOT/'review/v5_pilot_lifecycle_diagnosis_v1.json'
    if path.exists():raise FileExistsError(path)
    report=analyze();path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(path),sha256=sha(path),sources_and_inputs_unchanged=True,
        release_rows=[{key:x[key] for key in ('scene','method','whole_force_guard_pass','peak_N','peak_finger','peak_time_end_s','above_any_local_limit_duration_s','full_raw_patch_clear_confirmation_200Hz_s','public_sensor_clear_confirmation_s')} for x in report['releases']]),ensure_ascii=False,indent=2))

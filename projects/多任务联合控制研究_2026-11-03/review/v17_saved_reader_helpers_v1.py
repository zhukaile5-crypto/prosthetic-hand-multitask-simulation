"""Saved public metadata/domain/virtual-task checks. No model or solver import."""
import json,math
import numpy as np
ADDED_FIELDS=('slip_valid5','slip_domain_acquisition_time_s','slip_domain_time_s','slip_domain_role','slip_domain_available','slip_domain_reason')
ROLE='VALIDITY_OF_EXISTING_INSTRUMENT_SLIP_CHANNEL'
def canonical(x):return json.dumps(x,sort_keys=True,allow_nan=True,separators=(',',':'))
def compare_legacy(comparison,a,b,t,oldtime,pre):
    ma=t<12.-1e-9 if pre else np.ones(len(t),bool);mb=oldtime<12.-1e-9 if pre else np.ones(len(oldtime),bool)
    aa=a['legacy_common_sensor_json'][ma];bb=b['legacy_common_sensor_json'][mb];bad=[];extras=set()
    for i,(x,y) in enumerate(zip(aa,bb)):
        x=json.loads(str(x));y=json.loads(str(y));extras.update(set(x)-set(y))
        if set(x)!=set(y)|set(ADDED_FIELDS):bad.append((i,'extra_or_missing_field'));continue
        for k in ADDED_FIELDS:x.pop(k)
        if canonical(x)!=canonical(y):bad.append((i,'original_value_changed'))
    item=next(x for x in comparison if x['column']=='legacy_common_sensor_json')
    item.update(exact_payload_equal=aa.shape==bb.shape and not bad,rule='remove_only_exact_ADDED_FIELDS6_then_exact_original_JSON',added_fields=sorted(extras),original_field_errors=bad[:8],original_field_error_count=len(bad),literal_storage_change_expected_for_declared_new_metadata=True)
    return comparison
def domain_audit(a,frames):
    raw=np.zeros((len(frames),5));times=np.asarray(a['time']);errors=[];available=0;bits=0
    for i,s in enumerate(a['raw_patches_json']):
        for patch in json.loads(str(s)):
            d=int(patch['digit'])
            if 0<=d<5:raw[i,d]+=max(0.,float(patch['normal']))
    for i,f in enumerate(frames):
        if not all(k in f for k in ADDED_FIELDS):errors.append((i,'missing_fields'));continue
        acq=float(f['sample_time']);j=round(acq/.005);retained=bool(0<=j<=i and j>=i-101 and abs(times[j]-acq)<1e-9)
        mask=f['slip_valid5'];valid=f['valid']
        strict=len(mask)==len(valid)==5 and all(type(v) is bool for v in mask+valid)
        expected=((raw[j]>.02)&np.asarray(valid,bool)) if retained else np.zeros(5,bool)
        if not strict or not np.array_equal(mask,expected):errors.append((i,'mask_mismatch',mask,expected.tolist()))
        if (f['slip_domain_role']!=ROLE or f['slip_domain_available'] is not retained or abs(float(f['slip_domain_time_s'])-float(f['time']))>1e-9 or abs(float(f['slip_domain_acquisition_time_s'])-acq)>1e-9 or (retained and f['slip_domain_reason'] is not None) or (not retained and f['slip_domain_reason']!='instrument_acquisition_domain_history_missing')):errors.append((i,'provenance_clock_or_reason'))
        available+=int(retained);bits+=int(np.sum(expected))
    return dict(passed=not errors,error_count=len(errors),errors=errors[:12],frames=len(frames),digit_bits_checked=len(frames)*5,available_frames=available,valid_slip_bits=bits,rule='sum positive raw pressure by digit at acquisition; >.02 AND original valid5; any actor; no target filtering',threshold_N=.02)
def task_record(q):
    """Independent normal/available residual checks, NOT full motion-M replay."""
    errors=[];limits=[]
    if not q['geometry_proposal_accepted']:return errors,dict(accepted=False,motion_matrix_independently_recomputed=False)
    C=np.asarray(q['normal_rows'],float);T=np.asarray(q['tangent_rows'],float);d=np.asarray(q['proposed_reference_increment21'],float);scale=np.asarray(q['scale21'],float);x=d/scale
    lo=np.asarray(q['actual_step_lower21'])/scale;hi=np.asarray(q['actual_step_upper21'])/scale
    levels=q['hierarchy_levels'];diag=q['hierarchy_diagnostics'];tol=levels[0]['numerical_atol']+levels[0]['numerical_rtol']*max(1.,float(np.max(abs(x))))
    inward=C@d;rowmax=np.max(abs(C*scale),axis=1);rowlen=np.array([np.linalg.norm(v/m) if m else 1. for v,m in zip(C*scale,rowmax)]);norm=np.maximum(rowmax,np.finfo(float).tiny)*rowlen
    hard=max(0.,float(np.max(inward/norm,initial=0.)),float(np.max(lo-x,initial=0.)),float(np.max(x-hi,initial=0.)))
    if hard>tol:errors.append('raw_declared_hard_domain')
    pressure=np.array([p['observed_normal_N'] for p in q['geometry']['patches']],float);relative=pressure/pressure.max();weights=relative/relative.sum();out=-weights@(C*scale)
    if weights.tobytes()!=np.asarray(diag['observed_pressure_weights'],float).tobytes():errors.append('observed_pressure_weights')
    if abs(q['geometry']['current_time_s']-q['time_s'])>1e-9:errors.append('geometry_snapshot_clock')
    upper=float(np.sum(np.where(out>=0,out*hi,out*lo)));ps=max(float(np.max(abs(out),initial=0.)),abs(upper));ps=ps if ps else 1.
    primary=np.array([(float(out@x)-upper)/ps]);size=float(np.max(np.sum(abs(T*scale)*np.maximum(abs(lo),abs(hi))[None,:],axis=1),initial=0.));ts=size if size else 1.;tangent=(T*scale/ts)@x
    current=np.asarray(q['current_reference21']);nominal=np.asarray(q['nominal_reference21']);step_lo=np.asarray(q['actual_step_lower21']);step_hi=np.asarray(q['actual_step_upper21'])
    wanted=np.clip(nominal-current,step_lo,step_hi);producer_nominal=np.clip((current+wanted)-current,step_lo,step_hi);canonical_res=x-np.clip(producer_nominal/scale,lo,hi)
    by={r['name']:r for r in levels}
    for name,computed in [('pressure_weighted_normal_withdrawal',primary),('low_unnecessary_tangent_increment',tangent),('canonical_clearance_posture',canonical_res)]:
        saved=np.asarray(by[name]['residual']);err=float(np.max(abs(computed-saved),initial=0.))
        if saved.shape!=computed.shape or err>tol+128*np.finfo(float).eps*max(1.,float(np.max(abs(computed),initial=0.))):errors.append(name+'_final_frozen_residual')
        limits.append(dict(level=name,final_reconstruction_error_inf=err,declared_freeze_tolerance=tol))
    for level in levels:
        if level['status']!='KKT_CONVERGED' or level['primal_violation']>tol or level['preceding_residual_drift_inf']>tol:errors.append('saved_primal_or_frozen_status')
        r=np.asarray(level['residual'],float)
        if not np.isfinite(r).all() or abs(float(r@r)-level['squared_loss'])>128*np.finfo(float).eps*max(1.,abs(level['squared_loss'])):errors.append('residual_loss_inconsistency')
    motion='observed_external_contact_motion_correction' in by
    if motion!=q['motion_task_available']:errors.append('motion_layer_presence')
    if motion:
        meta=q['geometry']['motion_domain'];digits=meta['selected_public_digits'];expected_rows=2*len(digits)
        if len(set(digits))!=len(digits) or any(not 0<=i<5 for i in digits) or len(by['observed_external_contact_motion_correction']['residual'])!=expected_rows or abs(meta['read_time_s']-q['time_s'])>1e-9:errors.append('motion_metadata_rows_clock')
    if np.max(abs(np.asarray(diag['actual_inward_proposals_m'])-inward),initial=0)>128*np.finfo(float).eps*max(1.,float(np.max(abs(inward),initial=0.))):errors.append('recorded_normal_proposal')
    return errors,dict(accepted=True,original_hard_domain_normalized_violation=hard,declared_tolerance=tol,reconstructed_layers=limits,motion_matrix_independently_recomputed=False,limitation='Original MotionTask matrix not logged; saved residual/loss checked here, selected separate robot-only replay required.')

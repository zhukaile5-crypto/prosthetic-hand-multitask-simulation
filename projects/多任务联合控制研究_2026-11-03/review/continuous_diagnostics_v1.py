"""FROZEN auxiliary continuous measurement contract, offline and read-only.

All assigned fixed windows, with no success/contact filtering. Never a new
success rule, utility score, evaluator replacement, controller or simulation.
"""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json
import numpy as np

SCHEMA='all-assigned-continuous-diagnostics-v1'
WINDOWS={'static':(8.2,8.9),'dynamic':(9.3,11.7),'whole_holding':(7.5,12.)}
CATEGORIES=('receiver_top','receiver_otherface','floor','other_nonhand_world','hand','other_nonworld')
SENSOR_PERIOD_S=.005
CONTACT_CLOCK_TOLERANCE_S=1e-7

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def window_mask(t,lo,hi):return (t>=lo-1e-9)&(t<=hi+1e-9)
def _coverage(t,lo,hi):
    expected=round((hi-lo)/SENSOR_PERIOD_S)+1
    return len(t)==expected and len(t)>1 and abs(t[0]-lo)<1e-9 and abs(t[-1]-hi)<1e-9 and np.allclose(np.diff(t),SENSOR_PERIOD_S,rtol=0,atol=1e-9)
def _cv(sd,mean):
    # No epsilon or fabricated zero CV for zero mean.
    if mean==0.:return None,'undefined_zero_mean'
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):v=float(sd/mean)
    return (v,'defined') if np.isfinite(v) else (None,'unrepresentable_ratio')

def force_window(t,normal,lo,hi):
    use=window_mask(t,lo,hi);tt=t[use];f=normal[use] if normal is not None and normal.shape==(len(t),5) else np.empty((0,5))
    base=dict(window_s=[lo,hi],samples=len(tt),expected_samples=round((hi-lo)/SENSOR_PERIOD_S)+1,selection='all_assigned_fixed_window_no_contact_or_success_filter',signal='actual_target_finger_normal_N_unfiltered200Hz')
    if not _coverage(tt,lo,hi) or f.shape!=(len(tt),5) or not np.isfinite(f).all() or np.any(f<0.):return dict(**base,available=False,reason='incomplete_or_unknown_raw_force',metrics=None)
    seconds=hi-lo;tv=np.abs(np.diff(f,axis=0)).sum(axis=0);drift=np.abs(f[-1]-f[0]);reversal=np.maximum(0.,.5*(tv-drift));total=f.sum(axis=1)
    mean=f.mean(axis=0);sd=f.std(axis=0,ddof=0);total_mean=float(total.mean());total_sd=float(total.std(ddof=0));cv=[_cv(x,y) for x,y in zip(sd,mean)];tcv,tcvstatus=_cv(total_sd,total_mean)
    return dict(**base,available=True,duration_s=seconds,mean_total_force_N=total_mean,mean_per_finger_N=mean,
        finger_TV_N=float(tv.sum()),finger_TV_rate_N_s=float(tv.sum()/seconds),per_finger_TV_N=tv,
        total_force_TV_N=float(abs(np.diff(total)).sum()),total_force_TV_rate_N_s=float(abs(np.diff(total)).sum()/seconds),
        per_finger_SD_N=sd,mean_finger_SD_N=float(sd.mean()),total_force_SD_N=total_sd,per_finger_CV=[x[0] for x in cv],per_finger_CV_status=[x[1] for x in cv],total_force_CV=tcv,total_force_CV_status=tcvstatus,
        per_finger_net_drift_N=drift,per_finger_reversal_N=reversal,finger_net_drift_N=float(drift.sum()),finger_reversal_N=float(reversal.sum()),
        finger_net_drift_rate_N_s=float(drift.sum()/seconds),finger_reversal_rate_N_s=float(reversal.sum()/seconds),
        force_at_window_start_N=f[0],force_at_window_end_N=f[-1])

def _relative_angle(R0,R):
    delta=np.einsum('ij,njk->nik',R0.T,R);c=np.clip((np.trace(delta,axis1=1,axis2=2)-1.)/2.,-1.,1.)
    vee=np.c_[delta[:,2,1]-delta[:,1,2],delta[:,0,2]-delta[:,2,0],delta[:,1,0]-delta[:,0,1]]*.5
    return np.arctan2(np.linalg.norm(vee,axis=1),c)
def _quaternion_angle(q0,q):
    q0=np.asarray(q0,float);q0=q0/np.linalg.norm(q0);q=q/np.linalg.norm(q,axis=1)[:,None]
    w=q[:,0]*q0[0]+q[:,1:]@q0[1:]
    v=q0[0]*q[:,1:]-q[:,0,None]*q0[1:]-np.cross(np.broadcast_to(q0[1:],q[:,1:].shape),q[:,1:])
    return 2.*np.arctan2(np.linalg.norm(v,axis=1),abs(w))

def object_window(t,a,lo,hi,ledger,delivery_metadata):
    use=window_mask(t,lo,hi);ix=np.flatnonzero(use);h=np.flatnonzero(abs(t-7.5)<1e-9)
    fields={'position':(len(t),3),'relative':(len(t),3),'relative_velocity':(len(t),3),'rotation':(len(t),3,3),'angular_velocity':(len(t),3)}
    if not _coverage(t[use],lo,hi) or len(h)!=1 or any(k not in a or a[k].shape!=shape or not np.isfinite(a[k][use]).all() or not np.isfinite(a[k][h[0]]).all() for k,shape in fields.items()):return dict(available=False,reason='incomplete_or_unknown_physical_truth',window_s=[lo,hi])
    h=int(h[0]);drift=np.linalg.norm(a['relative'][use]-a['relative'][h],axis=1);world=np.linalg.norm(a['position'][use]-a['position'][h],axis=1)
    rv=np.linalg.norm(a['relative_velocity'][use],axis=1);omega=np.linalg.norm(a['angular_velocity'][use],axis=1);angle=_relative_angle(a['rotation'][h],a['rotation'][use])
    result=dict(available=True,window_s=[lo,hi],clock='200Hz truth before current action, includes12s pre-release-action endpoint',
        relative_handoff_max_displacement_m=float(drift.max()),relative_handoff_terminal_displacement_m=float(drift[-1]),world_handoff_max_displacement_m=float(world.max()),world_handoff_terminal_displacement_m=float(world[-1]),
        min_world_height_m=float(a['position'][use,2].min()),terminal_world_height_m=float(a['position'][use,2][-1]),min_world_height_minus_handoff_m=float(a['position'][use,2].min()-a['position'][h,2]),
        relative_linear_speed_max_m_s=float(rv.max()),relative_linear_speed_terminal_m_s=float(rv[-1]),relative_orientation_change_max_rad=float(angle.max()),relative_orientation_change_terminal_rad=float(angle[-1]),
        relative_angular_speed_max_rad_s=float(omega.max()),relative_angular_speed_terminal_rad_s=float(omega[-1]))
    raw=dict(available=False,reason='missing_or_unknown_object_end_state')
    required=('interval_end_s','object_state_valid','object_free_qpos7_end')
    qa=delivery_metadata.get('object_qpos_address')
    if all(k in ledger for k in required) and isinstance(qa,int) and not isinstance(qa,bool) and qa>=0 and 'qpos' in a and a['qpos'].ndim==2 and qa+7<=a['qpos'].shape[1]:
        use_raw=window_mask(ledger['interval_end_s'],lo,hi);q=ledger['object_free_qpos7_end'][use_raw];q0=a['qpos'][h,qa:qa+7]
        if len(q) and ledger['object_state_valid'][use_raw].all() and q.shape==(len(q),7) and np.isfinite(q).all() and np.isfinite(q0).all() and np.linalg.norm(q0[3:])>0 and np.all(np.linalg.norm(q[:,3:],axis=1)>0):
            angles=_quaternion_angle(q0[3:],q[:,3:]);raw=dict(available=True,samples=len(q),min_world_height_m=float(q[:,2].min()),terminal_world_height_m=float(q[-1,2]),max_world_displacement_from_handoff_m=float(np.linalg.norm(q[:,:3]-q0[:3],axis=1).max()),world_orientation_change_max_rad=float(angles.max()),world_orientation_change_terminal_rad=float(angles[-1]),clock='actual integration END; quaternion orientation in world, not palm-relative')
    result['raw_integration_end_world']=raw
    return result

def contact_window(ledger,field,lo,hi):
    base=dict(window_s=[lo,hi],threshold_N=0.,clock='cached constraint-solve force labelled by actual integration start/end; not continuous collision-time certificate')
    if any(k not in ledger for k in ('interval_start_s','interval_end_s','contact_valid',field)):return dict(**base,available=False,reason='missing_per_surface_ledger',any_positive_contact=None,peak_N=None,impulse_Ns=None,positive_duration_s=None,first_cached_interval_s=None)
    start=ledger['interval_start_s'];end=ledger['interval_end_s'];f=ledger[field];valid=ledger['contact_valid'];n=len(start)
    if any(x.shape!=(n,) for x in (end,f,valid)) or not np.isfinite(start).all() or not np.isfinite(end).all() or np.any(end<=start) or np.any(start[1:]<end[:-1]):return dict(**base,available=False,reason='invalid_or_overlapping_interval_coordinates',any_positive_contact=None,peak_N=None,impulse_Ns=None,positive_duration_s=None,first_cached_interval_s=None)
    dt=np.maximum(0.,np.minimum(end,hi)-np.maximum(start,lo));use=dt>0.;known=use&valid&np.isfinite(f)&(f>=0.);hits=known&(f>0.)
    complete=bool(use.any() and abs(dt.sum()-(hi-lo))<=CONTACT_CLOCK_TOLERANCE_S and known[use].all())
    first=int(np.flatnonzero(hits)[0]) if hits.any() else None;last=int(np.flatnonzero(hits)[-1]) if hits.any() else None
    return dict(**base,available=complete,any_positive_contact=True if first is not None else False if complete else None,peak_N=float(f[use].max()) if complete else None,impulse_Ns=float(np.sum(f[use]*dt[use])) if complete else None,positive_duration_s=float(dt[hits].sum()) if complete else None,
        first_cached_interval_s=None if first is None else [float(start[first]),float(end[first])],last_cached_interval_s=None if last is None else [float(start[last]),float(end[last])],
        observed_valid_duration_s=float(dt[known].sum()),observed_valid_peak_N=float(f[known].max()) if known.any() else None,observed_valid_impulse_Ns=float(f[known]@dt[known]) if known.any() else None)

def object_end_state(ledger,at_s):
    required=('interval_end_s','object_state_valid','object_free_qpos7_end','object_free_qvel6_end')
    base=dict(query_time_s=at_s,available=False,reason='missing_exact_end_state',qpos7=None,qvel6=None,position_m=None,linear_speed_m_s=None,angular_speed_rad_s=None)
    if any(k not in ledger for k in required):return base
    indices=np.flatnonzero(abs(ledger['interval_end_s']-at_s)<=1e-9)
    if len(indices)!=1:return base
    i=int(indices[0]);q=ledger['object_free_qpos7_end'][i];v=ledger['object_free_qvel6_end'][i]
    if not ledger['object_state_valid'][i] or q.shape!=(7,) or v.shape!=(6,) or not np.isfinite(np.r_[q,v]).all():return base
    return dict(query_time_s=at_s,available=True,end_time_s=float(ledger['interval_end_s'][i]),qpos7=q,qvel6=v,position_m=q[:3],linear_speed_m_s=float(np.hypot.reduce(v[:3])),angular_speed_rad_s=float(np.hypot.reduce(v[3:])),velocity_role='native free-joint velocity, not hand-relative')

def measure_arrays(trace,ledger,metadata,*,original_result=None,delivery_metadata=None):
    """Pure measurement API. Does not mutate inputs or infer new success."""
    delivery_metadata={} if delivery_metadata is None else delivery_metadata;t=np.asarray(trace.get('time',[]),float)
    if t.ndim!=1 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):return dict(schema=SCHEMA,status='invalid_or_missing_trace_clock',original_result=deepcopy(original_result),measurements=None)
    normal=np.asarray(trace['normal'],float) if 'normal' in trace else None
    measurements={name:dict(unselected_force=force_window(t,normal,lo,hi),physical_object=object_window(t,trace,lo,hi,ledger,delivery_metadata)) for name,(lo,hi) in WINDOWS.items()}
    floor=dict(before_release_0_to12=contact_window(ledger,'target_floor_normal_N',0.,12.),policy_holding_7p5_to12=contact_window(ledger,'target_floor_normal_N',7.5,12.),release_and_after12to16=contact_window(ledger,'target_floor_normal_N',12.,16.))
    contacts={}
    for name,(lo,hi) in {'before12':(0.,12.),'policy_holding':(7.5,12.),'release_and_after':(12.,16.)}.items():
        contacts[name]={k:contact_window(ledger,'target_'+k+'_normal_N',lo,hi) for k in CATEGORIES}
        contacts[name]['receiver_any_actor_top']=contact_window(ledger,'receiver_any_actor_top_normal_N',lo,hi)
    release=metadata.get('release_at')
    return dict(schema=SCHEMA,status='measurements_computed_no_new_success_label',measurements=measurements,floor=floor,contact_windows=contacts,
        object_end_states={'scheduled_handoff':object_end_state(ledger,7.5),'scheduled_holding_end':object_end_state(ledger,12.),'record_end16':object_end_state(ledger,16.)},
        original_result=deepcopy(original_result),identity={k:metadata.get(k) for k in ('method','seed','shape','scenario','motion_mode','task_policy','receiver_policy','run_id','split')},
        execution_context=dict(handoff_passed=metadata.get('handoff_passed'),preparation_failure=metadata.get('preparation_failure'),actual_release_at=release,
            scheduled_holding_interval_contains_actual_early_release=None if release is None else bool(release<12.),
            window_labels_are_scheduled_not_actual_phase=True),
        interpretation='All-assigned auxiliary fixed-window raw measurements; no success/contact filtering, no new threshold, utility score or replacement of original eligible NULLs. Early safety release remains visible in execution_context.')

def read_run(folder,*,record_label=None):
    folder=Path(folder).resolve();trace_path=folder/'trace.npz';result_path=folder/'result.json';metadata_path=folder/'metadata.json';result=read_json(result_path) if result_path.is_file() else None
    if not trace_path.is_file():return dict(schema=SCHEMA,status='assigned_trace_missing_or_unexecuted',run_directory=str(folder),record_label=record_label,original_result=result,measurements=None,runtime_error=read_json(folder/'runtime_error.json') if (folder/'runtime_error.json').is_file() else None)
    wanted=('time','normal','position','relative','relative_velocity','rotation','angular_velocity','qpos')
    with np.load(trace_path,allow_pickle=False) as z:
        trace={k:z[k] for k in wanted if k in z.files};embedded=json.loads(str(z['metadata'])) if 'metadata' in z.files else {}
    metadata=read_json(metadata_path) if metadata_path.is_file() else embedded
    ledger_path=folder/'delivery_intervals.npz';ledger={}
    if ledger_path.is_file():
        names=('interval_start_s','interval_end_s','contact_valid','object_state_valid','object_free_qpos7_end','object_free_qvel6_end','receiver_any_actor_top_normal_N',*('target_'+k+'_normal_N' for k in CATEGORIES))
        with np.load(ledger_path,allow_pickle=False) as z:ledger={k:z[k] for k in names if k in z.files}
    delivery=read_json(folder/'delivery_summary.json') if (folder/'delivery_summary.json').is_file() else {}
    report=measure_arrays(trace,ledger,metadata,original_result=result,delivery_metadata=delivery)
    report.update(run_directory=str(folder),record_label=record_label,measurement_source_sha256=sha(__file__),
        input_sha256=dict(trace=sha(trace_path),metadata=sha(metadata_path) if metadata_path.is_file() else None,delivery=sha(ledger_path) if ledger_path.is_file() else None,result=sha(result_path) if result_path.is_file() else None),
        npz_access_policy='Each required array loaded once per run; no observation JSON/controller/physical replay')
    return plain(report)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='append',default=[]);ap.add_argument('--batch',action='append',default=[]);ap.add_argument('--out',required=True);a=ap.parse_args()
    if not a.run and not a.batch:ap.error('At least one explicit run or assigned batch required')
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    assigned=[dict(path=str(Path(p).resolve()),source='explicit_run') for p in a.run]
    for value in a.batch:
        p=Path(value).resolve();manifest=read_json(p/'manifest.json')
        for j in manifest['jobs']:assigned.append(dict(path=str(Path(j['destination']).resolve()),source=str(p/'manifest.json'),job=j))
    records=[];seen={};duplicates=[]
    for item in assigned:
        path=item['path'];seen[path]=seen.get(path,0)+1
        if seen[path]>1:duplicates.append(path)
        try:record=read_run(path)
        except Exception as exc:
            original=None
            try:
                if (Path(path)/'result.json').is_file():original=read_json(Path(path)/'result.json')
            except (ValueError,OSError):pass
            record=dict(schema=SCHEMA,status='retained_reader_or_input_error',run_directory=path,error=type(exc).__name__+': '+str(exc),original_result=original,measurements=None)
        records.append(dict(assignment=item,diagnostics=record))
    output=dict(schema=SCHEMA,assignment_count=len(assigned),records=records,duplicate_assigned_paths=duplicates,measurement_source_sha256=sha(__file__),physics_steps=0,
        inferential_claims_allowed=False,no_new_success_or_weighted_score=True,scope='Descriptive all-assigned continuous metrics; original results and unavailable windows retained. A duplicate path is reported, never counted as a new independent trajectory.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(plain(output),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(assignment_count=len(assigned),output=str(out),source_sha256=sha(__file__))))
if __name__=='__main__':main()

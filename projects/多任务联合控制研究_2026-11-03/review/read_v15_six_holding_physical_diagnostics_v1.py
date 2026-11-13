"""Post-hoc unselected diagnostic metrics, NEVER replacement success labels.

Six unique observed holding trajectories only. No controller/physics/model.
Each trace/ledger NPZ column is loaded once, no filtering or sample deletion.
"""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=('fixed_effort','ratio_effort','fusion_effort')
WINDOWS={'static':(8.2,8.9),'dynamic':(9.3,11.7),'whole_holding':(7.5,12.)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def mask(t,a,b):return (t>=a-1e-9)&(t<=b+1e-9)
def angle_relative(R0,R):
    delta=np.einsum('ij,njk->nik',R0.T,R)
    c=np.clip((np.trace(delta,axis1=1,axis2=2)-1.)/2.,-1.,1.)
    vee=np.c_[delta[:,2,1]-delta[:,1,2],delta[:,0,2]-delta[:,2,0],delta[:,1,0]-delta[:,0,1]]*.5
    return np.arctan2(np.linalg.norm(vee,axis=1),c)
def quat_angle(q0,q):
    q0=np.asarray(q0,float);q0=q0/np.linalg.norm(q0);q=q/np.linalg.norm(q,axis=1)[:,None]
    w=q[:,0]*q0[0]+q[:,1:]@q0[1:]
    v=q0[0]*q[:,1:]-q[:,0,None]*q0[1:]-np.cross(np.broadcast_to(q0[1:],q[:,1:].shape),q[:,1:])
    return 2.*np.arctan2(np.linalg.norm(v,axis=1),abs(w))
def force(t,f,lo,hi):
    use=mask(t,lo,hi);tt=t[use];ff=f[use];expected=round((hi-lo)/.005)+1
    complete=len(tt)==expected and len(tt)>1 and abs(tt[0]-lo)<1e-9 and abs(tt[-1]-hi)<1e-9 and np.allclose(np.diff(tt),.005,rtol=0,atol=1e-9)
    if not complete or ff.shape!=(expected,5) or not np.isfinite(ff).all():return dict(available=False,reason='incomplete_or_unknown_raw_force',window_s=[lo,hi],expected_samples=expected,samples=len(tt),metrics=None)
    tv=np.abs(np.diff(ff,axis=0)).sum(axis=0);drift=np.abs(ff[-1]-ff[0]);rev=np.maximum(0.,.5*(tv-drift));total=ff.sum(axis=1);seconds=hi-lo
    return dict(available=True,window_s=[lo,hi],samples=len(tt),duration_s=seconds,signal='actual_target_finger_normal_N_unfiltered200Hz',selection='all_assigned_fixed_window_no_contact_or_success_filter',
        mean_total_force_N=float(total.mean()),mean_per_finger_N=ff.mean(axis=0),finger_TV_N=float(tv.sum()),finger_TV_rate_N_s=float(tv.sum()/seconds),per_finger_TV_N=tv,
        total_force_TV_N=float(abs(np.diff(total)).sum()),total_force_TV_rate_N_s=float(abs(np.diff(total)).sum()/seconds),
        per_finger_SD_N=ff.std(axis=0,ddof=0),mean_finger_SD_N=float(ff.std(axis=0,ddof=0).mean()),total_force_SD_N=float(total.std(ddof=0)),
        finger_net_drift_N=float(drift.sum()),finger_reversal_N=float(rev.sum()),force_at_window_start_N=ff[0],force_at_window_end_N=ff[-1])
def physical(t,a,lo,hi,handoff,ld):
    use=mask(t,lo,hi);ix=np.flatnonzero(use);fields=('position','relative','relative_velocity','rotation','angular_velocity')
    if not len(ix) or any(not np.isfinite(a[k][use]).all() for k in fields):return dict(available=False,reason='unknown_physical_truth')
    drift=np.linalg.norm(a['relative'][use]-a['relative'][handoff],axis=1);world=np.linalg.norm(a['position'][use]-a['position'][handoff],axis=1)
    rv=np.linalg.norm(a['relative_velocity'][use],axis=1);omega=np.linalg.norm(a['angular_velocity'][use],axis=1);angles=angle_relative(a['rotation'][handoff],a['rotation'][use])
    result=dict(available=True,window_s=[lo,hi],clock='200Hz truth before current action, includes12s pre-release-action endpoint',
        relative_handoff_max_displacement_m=float(drift.max()),relative_handoff_terminal_displacement_m=float(drift[-1]),
        world_handoff_max_displacement_m=float(world.max()),world_handoff_terminal_displacement_m=float(world[-1]),min_world_height_m=float(a['position'][use,2].min()),terminal_world_height_m=float(a['position'][use,2][-1]),
        min_world_height_minus_handoff_m=float(a['position'][use,2].min()-a['position'][handoff,2]),relative_linear_speed_max_m_s=float(rv.max()),relative_linear_speed_terminal_m_s=float(rv[-1]),
        relative_orientation_change_max_rad=float(angles.max()),relative_orientation_change_terminal_rad=float(angles[-1]),relative_angular_speed_max_rad_s=float(omega.max()),relative_angular_speed_terminal_rad_s=float(omega[-1]))
    # Additional native integration-end world state; distinct from200Hz
    # relative-to-hand quantities and cached contact-solve phase.
    rawmask=mask(ld['interval_end_s'],lo,hi);valid=ld['object_state_valid'][rawmask]
    if rawmask.any() and valid.all() and np.isfinite(ld['object_free_qpos7_end'][rawmask]).all():
        q=ld['object_free_qpos7_end'][rawmask];q0=a['qpos'][handoff,ld['_qpos_address']:ld['_qpos_address']+7]
        qa=quat_angle(q0[3:],q[:,3:]);result['raw_integration_end_world']=dict(available=True,samples=len(q),min_world_height_m=float(q[:,2].min()),terminal_world_height_m=float(q[-1,2]),max_world_displacement_from_handoff_m=float(np.linalg.norm(q[:,:3]-q0[:3],axis=1).max()),
            world_orientation_change_max_rad=float(qa.max()),world_orientation_change_terminal_rad=float(qa[-1]),clock='actual integration END; quaternion orientation in world, not palm-relative')
    else:result['raw_integration_end_world']=dict(available=False,reason='unknown_object_end_state')
    return result
def floor_event(ld,lo,hi):
    t0=ld['interval_start_s'];t1=ld['interval_end_s'];dt=np.maximum(0.,np.minimum(t1,hi)-np.maximum(t0,lo));use=dt>0
    valid=ld['contact_valid'][use];f=ld['target_floor_normal_N'];complete=bool(use.any() and abs(dt.sum()-(hi-lo))<=1e-7 and valid.all() and np.isfinite(f[use]).all())
    hits=use&ld['contact_valid']&(f>0.)
    first=int(np.flatnonzero(hits)[0]) if hits.any() else None
    return dict(window_s=[lo,hi],available=complete,any_positive_contact=True if first is not None else False if complete else None,peak_N=float(f[use].max()) if complete else None,
        impulse_Ns=float(np.sum(f[use]*dt[use])) if complete else None,positive_duration_s=float(dt[hits].sum()) if complete else None,
        first_cached_interval_s=None if first is None else [float(t0[first]),float(t1[first])],threshold_N=0.,clock='cached constraint-solve force labelled by actual integration start/end; not continuous collision-time certificate')
def main():
    out=ROOT/'review/v15_six_unique_holding_physical_diagnostics_v1.json'
    if out.exists():raise FileExistsError(out)
    rows=[];checks=[]
    for cell,folder in [('legacy_blind','v15_passive_blind_bridge_v1'),('integrated_holding_blind','v15_holding_task_ablation_v1')]:
        for method in METHODS:
            p=ROOT/'output'/folder/'runs'/f'nominal__{method}__617901';meta=read(p/'metadata.json');saved=read(p/'result.json');summary=read(p/'delivery_summary.json')
            fields=('time','normal','position','relative','relative_velocity','rotation','angular_velocity','qpos')
            with np.load(p/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in fields}
            with np.load(p/'delivery_intervals.npz',allow_pickle=False) as z:ld={k:z[k] for k in ('interval_start_s','interval_end_s','contact_valid','object_state_valid','object_free_qpos7_end','target_floor_normal_N')}
            ld['_qpos_address']=summary['object_qpos_address'];t=a['time'];handoff=int(np.argmin(abs(t-7.5)))
            measurements={name:dict(unselected_force=force(t,a['normal'],lo,hi),physical_object=physical(t,a,lo,hi,handoff,ld)) for name,(lo,hi) in WINDOWS.items()}
            floor={'before_release_0_to12':floor_event(ld,0.,12.),'policy_holding_7p5_to12':floor_event(ld,7.5,12.)}
            checks.append(dict(name=cell+'/'+method+'/all_unselected_windows_available',passed=all(v['unselected_force']['available'] and v['physical_object']['available'] for v in measurements.values())))
            rows.append(dict(cell=cell,method=method,seed=meta['seed'],shape=meta['shape'],scenario=meta['scenario']['name'],motion_mode=meta['motion_mode'],
                task_policy=meta['task_policy'],receiver_policy=meta['receiver_policy'],measurements=measurements,floor=floor,
                original_results_unchanged=dict(task_success=saved['task_success'],mission_success=saved['research_metrics']['mission_success'],full_motion_mission_success=saved['research_metrics']['full_motion_mission_success'],
                    failure_reason=saved['research_metrics']['failure_reason'],failure_at=saved['research_metrics']['failure_at'],static_quality=saved['research_metrics']['static_quality'],dynamic_quality=saved['research_metrics']['dynamic_quality']),
                handoff_position_m=a['position'][handoff],trace_sha256=sha(p/'trace.npz'),delivery_sha256=sha(p/'delivery_intervals.npz')))
    report=dict(schema='v15_six_unique_holding_unselected_physical_diagnostics_v1',passed=all(x['passed'] for x in checks),checks=checks,records=plain(rows),physics_steps=0,
        unique_trajectories=6,physical_contexts=1,seen_development_seed=617901,excluded_duplicate_cells=['release_only pre12 duplicate','combined pre12 duplicate'],
        interpretation='Post-hoc mechanism diagnosis only. Raw200Hz all-assigned windows do not replace original eligible quality metrics or success labels. No threshold/weight/composite score/new success definition, no superiority inference.',
        relative_coordinate_semantics='position is world; relative and rotation are palm-frame object transforms; relative_velocity subtracts palm linear and angular lever velocity, angular_velocity subtracts palm angular velocity. Relative transform change is not pure world translation/sliding; world diagnostics are separate.',
        quaternion_angle='2*atan2(norm(vector(conj(q_handoff)*q_t)),abs(scalar)); hand-relative matrix angle uses atan2(skew norm/2,(trace-1)/2)',checker_sha256=sha(__file__))
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf-8')
    print('path',str(out),'sha256',sha(out),'passed',report['passed'])
    for row in rows:
        if row['method']!='fusion_effort':continue
        print(row['cell'])
        for name,v in row['measurements'].items():
            f=v['unselected_force'];p=v['physical_object'];print(name,json.dumps(plain(dict(meanN=f['mean_total_force_N'],TV_Ns=f['finger_TV_rate_N_s'],totalSD=f['total_force_SD_N'],meanFingerSD=f['mean_finger_SD_N'],driftmm=1000*p['relative_handoff_max_displacement_m'],minheight=p['min_world_height_m'],relMaxSpeed=p['relative_linear_speed_max_m_s'],relEndSpeed=p['relative_linear_speed_terminal_m_s'],relAngleMaxDeg=np.degrees(p['relative_orientation_change_max_rad']),relAngleEndDeg=np.degrees(p['relative_orientation_change_terminal_rad_s']) if 'relative_orientation_change_terminal_rad_s' in p else np.degrees(p['relative_orientation_change_terminal_rad']),angularMax=p['relative_angular_speed_max_rad_s'])),ensure_ascii=False))
        print('floor',row['floor'])
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

"""Six archived release diagnostics; public signals and offline object state separate.

Reads each needed NPZ column once. No model/FK/physics or controller replay.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'review/v15A_prospective64_independent_v2.json'
MAIN_SHA='14467f1639574aa063d911a4a336f2f95919ccd463358ee210cbbc00f9a9ef44'
CASES={('hold','fragile_conflict',619732),('transport','nominal',619732),('transport','friction_drop',619732)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(v):
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def quat_angle(q0,q):
    a=q0/np.linalg.norm(q0);b=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w=b[:,0]*a[0]+b[:,1:]@a[1:];v=a[0]*b[:,1:]-b[:,0,None]*a[1:]-np.cross(np.broadcast_to(a[1:],b[:,1:].shape),b[:,1:])
    return 2*np.arctan2(np.linalg.norm(v,axis=1),abs(w))
def project_public_hand(o,history):
    """No object quantities: signed measured hand->external force along past up."""
    now=float(o['time']);acq=float(o['sample_time']);j=int(round(acq/.005));axis=history.get(j,{}).get('support_axis')
    if axis is None:return None
    up=np.asarray(axis,float);up/=np.linalg.norm(up)
    n=np.asarray(o['normal'],float);M=np.asarray(o['normal_wrench_map'],float);F=np.asarray(o['finger_non_normal_by_finger'],float);a=np.asarray(o['auxiliary_wrench'],float);B=np.asarray(o['tangent_basis'],float)
    valid=np.asarray(o['valid'],bool)&np.asarray(o['distributed_valid'],bool)
    specs=o['sensor_characteristics'];sf=float(specs['force_sigma_N']);st=float(specs['tangential_sigma_N']);ua=float(o['auxiliary_uncertainty'])
    if not(valid.all() and o['auxiliary_valid'] and np.isfinite(np.r_[n,M.ravel(),F.ravel(),a,B.ravel(),up,sf,st,ua]).all() and abs(acq-(now-float(o['auxiliary_age'])))<1e-9):return None
    force=M[:3]@np.maximum(n,0)+F[:,:3].sum(0)+a[:3]
    width=3*sf*float(abs(up@M[:3]).sum())+3*st*float(abs(np.einsum('ijk,k->ij',B,up)).sum())+ua
    return dict(signed_support_N=float(force@up),engineering_uncertainty_N=width,acquisition_s=acq,force3_hand_frame=force,up_at_acquisition=up,
        up_at_delivery=np.asarray(o['support_axis'],float),clamp_normal_sum_N=float(n.sum()),auxiliary_wrench6=a,
        uncertainty_role='device3sigma_L1_finger_projection_plus_declared_auxiliary_width; not a hard bound; normal-direction error not bounded')
def cop_proxy(r):
    if not r['valid']:return None
    w=np.asarray(r['wrench6'],float);u=np.asarray(r['uncertainty6'],float)
    if not np.isfinite(np.r_[w,u]).all() or -w[2]<=u[2]:return None
    x=[-m/f for m in (w[4]-u[4],w[4]+u[4]) for f in (w[2]-u[2],w[2]+u[2])]
    y=[m/f for m in (w[3]-u[3],w[3]+u[3]) for f in (w[2]-u[2],w[2]+u[2])]
    return dict(xy_m=np.array([-w[4]/w[2],w[3]/w[2]]),lower_m=np.array([min(x),min(y)]),upper_m=np.array([max(x),max(y)]),role='equivalent net-wrench point, not object CoM/identity or pressure-distribution centroid')

def main():
    assert sha(MAIN)==MAIN_SHA
    out=ROOT/'review/v16_release_geometry_six_cases_v1.json';directory=ROOT/'review/v16_release_geometry_six_cases_v1'
    if out.exists() or directory.exists():raise FileExistsError('Fresh review evidence required')
    directory.mkdir();allrows=read(MAIN)['rows'];selected=[r for r in allrows if (r['motion'],r['scenario'],r['seed']) in CASES and r['method']=='ratio_effort'];rows=[]
    for r in selected:
        d=Path(r['run']);meta=read(d/'metadata.json');ls=read(d/'delivery_summary.json')
        names=('time','normal','skin_normal','all_hand_object','lift','executed_reference21','receiver_measurement_json','receiver_release_tasks_json','observation_json','estimated_load','estimated_load_available','phase')
        with np.load(d/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in names}
        names=('interval_start_s','interval_end_s','contact_valid','object_state_valid','object_free_qpos7_end','object_free_qvel6_end','target_receiver_top_normal_N','target_receiver_otherface_normal_N','target_floor_normal_N','target_hand_normal_N','receiver_any_actor_top_normal_N')
        with np.load(d/'delivery_intervals.npz',allow_pickle=False) as z:L={k:z[k] for k in names}
        t=a['time'];indices=np.flatnonzero(t>=11.75-1e-9);history={int(i):json.loads(str(a['observation_json'][i])) for i in indices};use=np.flatnonzero(t>=11.8-1e-9)
        public=[]
        for i in use:
            o=history[int(i)];receiver=json.loads(str(a['receiver_measurement_json'][i]));hand=project_public_hand(o,history);cop=cop_proxy(receiver)
            public.append(dict(time_s=float(t[i]),hand=hand,receiver=receiver,cop=cop,motor_load_estimate_N=o['load_estimate'],motor_load_valid=o['load_valid'],motor_uncertainty_N=o['load_uncertainty'],state_load_N=float(a['estimated_load'][i]) if a['estimated_load_available'][i] else None,
                actual_reference21=a['executed_reference21'][i],lift_encoder_m=float(a['lift'][i]),target_finger_truth5_N=a['normal'][i],all_external_finger_truth5_N=a['skin_normal'][i],hand_target_truth_N=float(a['all_hand_object'][i])))
        prior=[p for p in public if 11.8-1e-9<=p['time_s']<12.-1e-9 and p['motor_load_valid']]
        pv=np.array([p['motor_load_estimate_N'] for p in prior],float);pu=np.array([p['motor_uncertainty_N'] for p in prior],float)
        reference=None if not len(prior) else dict(samples=len(prior),last_read_time_s=prior[-1]['time_s'],mean_proxy_N=float(pv.mean()),engineering_width_N=float(pu.mean()+max(abs(pv-pv.mean()))),source='pre-request public motor estimate; a prior, not fresh post-request evidence or true weight')
        starts=L['interval_start_s'];ends=L['interval_end_s'];q=L['object_free_qpos7_end'];v=L['object_free_qvel6_end'];mask=(ends>=12.-1e-9)&(ends<=16.+1e-9)
        refi=int(np.argmin(abs(ends-12.)));angle=quat_angle(q[refi,3:],q[:,3:]);horizontal=np.linalg.norm(v[:,:2],axis=1);omega=np.linalg.norm(v[:,3:],axis=1)
        top=L['target_receiver_top_normal_N'];floor=L['target_floor_normal_N'];anytop=L['receiver_any_actor_top_normal_N'];handf=L['target_hand_normal_N']
        hit=lambda x:np.flatnonzero(mask&(x>0));ti=hit(top);fi=hit(floor);hi=hit(handf)
        def state(k):
            j=min(len(t)-1,int(np.searchsorted(t,ends[k]-1e-9)));p=public[j-int(use[0])]
            return plain(dict(start_s=starts[k],end_s=ends[k],object_qpos7_end=q[k],object_qvel6_end=v[k],horizontal_speed_m_s=horizontal[k],angular_speed_norm_rad_s=omega[k],horizontal_speed_over_angular_speed_m=None if omega[k]==0 else horizontal[k]/omega[k],rotation_change_from_release_rad=angle[k],target_top_N=top[k],anyactor_top_N=anytop[k],floor_N=floor[k],target_hand_N=handf[k],first_public_delivery_at_or_after_end=p,
                clock_note='Cached contact solve and integrated end q/v share interval labels, not a recomputed end contact geometry. Public delivery can be up to5ms later and measures10ms-old acquisition.'))
        event_idx={'release_end_state':refi,'end16':len(ends)-1}
        if len(ti):event_idx.update(first_target_top=int(ti[0]),last_target_top=int(ti[-1]),peak_target_top=int(np.argmax(np.where(mask,top,-1))))
        if len(fi):event_idx['first_floor']=int(fi[0])
        if len(hi):event_idx['last_hand_target_contact']=int(hi[-1])
        for age in (.020,.150,.200,.350,.600,.800,1.,1.4,2.2,3.):event_idx['release_age_'+str(age)]=int(np.argmin(abs(ends-(12.+age))))
        events={name:state(k) for name,k in event_idx.items()}
        series=dict(time_s=np.array([p['time_s'] for p in public]),hand_support_N=np.array([np.nan if p['hand'] is None else p['hand']['signed_support_N'] for p in public]),hand_uncertainty_N=np.array([np.nan if p['hand'] is None else p['hand']['engineering_uncertainty_N'] for p in public]),receiver_load_N=np.array([np.nan if not p['receiver']['valid'] else p['receiver']['normal_load_N'] for p in public]),receiver_acquisition_s=np.array([np.nan if p['receiver']['acquisition_time_s'] is None else p['receiver']['acquisition_time_s'] for p in public]),cop_xy_m=np.array([[np.nan,np.nan] if p['cop'] is None else p['cop']['xy_m'] for p in public]),lift_reference_m=np.array([p['actual_reference21'][18] for p in public]),lift_encoder_m=np.array([p['lift_encoder_m'] for p in public]),target_finger_truth5_N=np.array([p['target_finger_truth5_N'] for p in public]))
        basename=f"{r['arm']}__{r['motion']}__{r['scenario']}__{r['seed']}";path=directory/(basename+'.npz');np.savez_compressed(path,**series)
        milestones={}
        for tm in (11.8,11.995,12.,12.2,12.4,12.6,12.8,13.,13.4,14.,15.,16.):
            p=min(public,key=lambda p:abs(p['time_s']-tm));milestones[str(tm)]=plain(p)
        # No new physical thresholds: compare fixed-window means and endpoint
        # trends rather than deciding a "safe/stable" label.
        windows={}
        for lo,hiw in ((11.8,12.),(12.,12.2),(12.2,12.4),(12.4,12.6),(12.6,12.8),(13.,13.2),(14.,14.2),(15.,15.2)):
            pp=[p for p in public if lo-1e-9<=p['time_s']<hiw-1e-9];good=[p for p in pp if p['hand'] is not None and p['receiver']['valid']];cc=[p for p in good if p['cop'] is not None]
            windows[f'{lo}:{hiw}']=plain(dict(samples=len(pp),aligned_hand_receiver_known=len(good),motor_valid=sum(p['motor_load_valid'] for p in pp),hand_support_mean_N=None if not good else np.mean([p['hand']['signed_support_N'] for p in good]),receiver_mean_N=None if not good else np.mean([p['receiver']['normal_load_N'] for p in good]),sum_support_proxy_mean_N=None if not good else np.mean([p['hand']['signed_support_N']+p['receiver']['normal_load_N'] for p in good]),clamp_sum_mean_N=None if not good else np.mean([p['hand']['clamp_normal_sum_N'] for p in good]),cop_known=len(cc),cop_change_m=None if len(cc)<2 else cc[-1]['cop']['xy_m']-cc[0]['cop']['xy_m']))
        release_public=[p for p in public if p['time_s']>=12.-1e-9];lift=np.array([p['actual_reference21'][18] for p in release_public]);updiff=np.diff(lift);numeric=64*np.finfo(float).eps
        zero=np.flatnonzero(abs(lift)<=1e-12);up=np.flatnonzero(updiff>numeric)
        rows.append(plain(dict(arm=r['arm'],motion=r['motion'],scenario=r['scenario'],seed=r['seed'],run=str(d),input_sha256={name:sha(d/name) for name in ('trace.npz','metadata.json','delivery_intervals.npz','delivery_summary.json')},reference_prior=reference,
            events=events,public_milestones=milestones,public_windows=windows,series_file=path.relative_to(ROOT).as_posix(),series_sha256=sha(path),
            anyactor_target_top_exact=anytop.tobytes()==top.tobytes(),receiver_graph_present=any(json.loads(str(x)) is not None for x in a['receiver_release_tasks_json']),receiver_policy=meta['receiver_policy'],
            lift_command=dict(first_near_zero_time_s=None if not len(zero) else release_public[int(zero[0])]['time_s'],post_request_positive_change_count=int(len(up)),max_positive_increment_m=float(max(0.,updiff.max())),positive_total_variation_m=float(updiff[updiff>0].sum()),max_reference_m=float(lift.max()),min_reference_m=float(lift.min()),withdraw_phase=None,role='actual committed reference, not user-target lift_ref; blind cohort has no receiver WITHDRAW phase'),
            original_result=r['result']['research_metrics']['release'],all_object_state_valid=bool(L['object_state_valid'].all()),all_contact_valid=bool(L['contact_valid'].all()))))
        print(json.dumps(dict(case=basename,first_top=None if not len(ti) else float(ends[ti[0]]),last_top=None if not len(ti) else float(ends[ti[-1]]),floor=None if not len(fi) else float(ends[fi[0]]),prior=reference),ensure_ascii=False))
    report=dict(schema='v16_release_geometry_six_cases_offline_diagnostic_v1',rows=rows,input64_audit_sha256=MAIN_SHA,source_sha256=sha(__file__),physics_steps=0,model_imports=0,
        roles=dict(online_observable='Public hand wrenches/normal maps and tangent components, recorded up-axis history, receiver delayed6-axis, motor estimate/valid/uncertainty, actual21 command and encoders.',offline_only='Target top/floor/hand contact classes and object free q7/v6; never proposed as observer inputs.'),
        limits=['Six selected failure/baseline records are mechanism examples, not new performance sample or population inference.','Anyactor-minus-target is not named hand; equality is recorded separately.','CoP is equivalent net-wrench point, not mass center/identity/support certificate.','Load sum conservation is only a quasi-static proxy; signed hand-to-receiver forces or external dynamics can confound transfer.','All six are blind receiver, so no active WITHDRAW graph exists; positive scalar lift adjustments are not a withdraw phase.'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(rows=len(rows),out=str(out),sha256=sha(out)),ensure_ascii=False))
if __name__=='__main__':main()

"""Saved cumulative reference-path coverage; no simulator imports or execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'calibration/phase3_path_validation_v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-9):return bool(np.allclose(a,b,atol=tol,rtol=0))
def main():
    mainpath=sorted((ROOT/'review').glob('independent_phase3_v3_saved_*.json'))[-1];audit=read(mainpath);checks={};records=[];ranges={}
    def check(k,v):checks[k]=bool(v)
    for shape in ['capsule','box','sphere']:
        case=DATA/shape;anchor=read(case/'independent_height_gate_reference.json');marker=np.asarray(anchor['initial_object_in_palm_marker_m']);minimum=np.asarray(audit['ranges'][shape]['reference_min']);maximum=np.asarray(audit['ranges'][shape]['reference_max']);amin=np.asarray(audit['ranges'][shape]['actual_robot_min']);amax=np.asarray(audit['ranges'][shape]['actual_robot_max']);last=None
        folders=sorted(case.glob('reference_segment_*'));check(shape+'/all_five_reference_segments',len(folders)==5)
        for folder in folders:
            m=read(folder/'metadata.json');key=shape+'/'+folder.name
            with np.load(folder/'trace.npz',allow_pickle=False) as a:z={k:a[k] for k in a.files if k not in ['metadata','raw_patches','physical_raw_patches']}
            check(key+'/complete_hash',m['available'] and m['complete'] and m['failure'] is None and sha(folder/'trace.npz')==m['trace_sha256'])
            t=z['time'];pt=z['physical_time'];dt=z['physical_duration'];duration=float(t[-1]-t[0]);check(key+'/full_contiguous_physical_end_clock',close(np.diff(t),.005,1e-8) and close(pt-t[0],np.arange(1,len(pt)+1)*.0005,1e-8) and close(z['physical_interval_start']+dt,pt,1e-12) and abs(dt.sum()-duration)<1e-8 and (last is None or abs(t[0]-last)<1e-8));last=float(t[-1])
            minimum=np.minimum(minimum,z['input_reference'].min(0));maximum=np.maximum(maximum,z['input_reference'].max(0));amin=np.minimum(amin,z['physical_gate_actual_robot_motion_end'].min(0));amax=np.maximum(amax,z['physical_gate_actual_robot_motion_end'].max(0))
            for prefix in ['', 'physical_']:
                p=z[prefix+'pressure'];R=z[prefix+'gate_palm_rotation_end'];palm=z[prefix+'gate_palm_position_end'];obj=z[prefix+'gate_object_position_end'];rel=np.einsum('nji,nj->ni',R,obj-palm);drift=np.linalg.norm(rel-marker,axis=1);delta=(palm+np.einsum('nij,j->ni',R,marker))[:,2]-anchor['actual_anchor_marker_z_m'];floor=np.maximum(.025,.07+delta)
                check(key+'/'+prefix+'actual_height_and_relative',close(rel,z[prefix+'gate_object_relative_position_end'],1e-12) and close(drift,z[prefix+'gate_relative_drift_m'],1e-12) and close(floor,z[prefix+'gate_minimum_object_lift_m'],1e-12))
                check(key+'/'+prefix+'raw_engineering_guards',np.all(p<=5.02) and np.all(p.sum(1)<=18.02) and np.all(p[:,0]>.2) and np.all((p>.2).sum(1)>=3) and np.all(z[prefix+'penetration']<=.002) and np.all(z[prefix+'table_support_N']<=.05) and np.all(z[prefix+'object_lift_m']>=floor) and np.all(drift<=.005) and not np.any(z[prefix+'warning_count']) and not np.any(z[prefix+'gate_robot_tracking_failed']))
                check(key+'/'+prefix+'tracking_separate_from_retention',close(z[prefix+'gate_robot_tracking_error'],z[prefix+'gate_actual_robot_motion_end']-z[prefix+'input_reference'][:,5:],1e-12) and close(z[prefix+'gate_geometry_time_s'],z[prefix+'time'],1e-12))
            check(key+'/reference_matches_public_transport_only',close(z['input_reference'][:,:5],read(case/'setup.json')['handoff_reference'][:5],1e-12) and not np.any(z['non_target_contact_pressure_N']))
            records.append(dict(shape=shape,segment=folder.name,start_s=float(t[0]),end_s=float(t[-1]),physical_substeps=len(pt),minimum_height_margin_m=float(np.min(z['physical_object_lift_m']-z['physical_gate_minimum_object_lift_m'])),maximum_relative_drift_m=float(z['physical_gate_relative_drift_m'].max()),maximum_tracking_error=np.max(abs(z['physical_gate_robot_tracking_error']),axis=0).tolist(),peak_pressure_N=float(z['physical_pressure'].max())))
        check(shape+'/reference_path_7_5_to_11_7',abs(records[-5]['start_s']-7.5)<1e-8 and abs(records[-1]['end_s']-11.7)<1e-8)
        ranges[shape]=dict(combined_reference_min=minimum.tolist(),combined_reference_max=maximum.tolist(),combined_actual_robot_min=amin.tolist(),combined_actual_robot_max=amax.tolist())
    result=dict(schema='independent_phase3_v3_cumulative_reference_path_supplement_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),main_audit_sha256=sha(mainpath),reviewer_sha256=sha(__file__),records=records,combined_path_and_branch_ranges=ranges,physics_steps_executed=0,model_admitted=False,scope='Saved standard .12kg/.6 three-shape reference segments and local checkpoint branches only; no new dynamics or general model admission.')
    out=ROOT/'review'/('independent_phase3_v3_reference_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),ranges=ranges),ensure_ascii=False))
if __name__=='__main__':main()

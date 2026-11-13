"""Six archived release paths; public robot/skin FK only, zero integration.

Object/receiver/floor diagnostics are a separate independent reader. This one
does not read object qpos, contact IDs, object mass/friction or true velocities.
"""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
from hand_plant import _model
from foundation_config import FoundationConfig
from v10.robot_contact_space import RobotContactSpace,RobotJointSample,PublicSkinPatch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=ROOT/'review/v16_robot_patch_motion_six_cases_v1.json'
    if out.exists():raise FileExistsError(out)
    forbidden=[]
    def no_call(*args,**kwargs):
        forbidden.append('forbidden_pipeline_call')
        raise RuntimeError('No physics/forward/contact solve permitted in this reader')
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):setattr(mujoco,name,no_call)
    model,_=_model(FoundationConfig())
    geom=RobotContactSpace(model)
    cases=[('hold','fragile_conflict'),('transport','nominal'),('transport','friction_drop')]
    results=[]
    for motion,scene in cases:
      for role in ('candidate_A','baseline'):
        stage=f'v15A_prospective_{role}_{motion}_v2'
        p=ROOT/'output'/stage/'runs'/f'{scene}__ratio_effort__619732'
        # Every compressed column fetched once. No full simulator qpos/qvel read.
        with np.load(p/'trace.npz',allow_pickle=False) as z:
            time=z['time']; observation=z['observation_json']; reference=z['executed_reference21']
        rows=[];frames=[];unavailable=[]
        indices=np.flatnonzero((time>=12-1e-9)&(time<=16+1e-9))
        decoded={int(k):json.loads(str(observation[k])) for k in indices}
        for k in indices:
            d=decoded[int(k)];t=float(time[k]);patches=d.get('skin_patches',[])
            q=np.asarray(d['robot_position25'],float);v=np.asarray(d['robot_velocity25'],float)
            if q.shape!=(25,) or v.shape!=(25,):raise ValueError('complete public robot25 required')
            sample=RobotJointSample(t,geom.raw_joint_names,q[:24],q[24],v[:24],v[24])
            counted=0
            for j,raw in enumerate(patches):
                if not np.isfinite(raw.get('normal',np.nan)) or raw['normal']<=0:
                    unavailable.append(dict(time_s=t,index=j,reason='positive public pressure missing'));continue
                patch=PublicSkinPatch.from_public(raw)
                if patch.reference_normal is None or patch.reference_position is None:
                    unavailable.append(dict(time_s=t,index=j,reason='acquisition plane missing'));continue
                kin=geom.point_kinematics(sample,patch)
                n=patch.reference_normal;vel=kin.point_velocity_world
                vn=float(n@vel);vt=float(np.linalg.norm(vel-n*vn))
                # Offline actual 20ms displacement of THIS ROBOT material point,
                # not the next frame's possibly different contact point.
                future=k+4;delta_n=None;delta_t=None
                if future<len(time) and abs(float(time[future]-time[k])-.02)<1e-9:
                    fd=decoded[int(future)]
                    fq=np.asarray(fd['robot_position25'],float)
                    fs=RobotJointSample(float(time[future]),geom.raw_joint_names,fq[:24],fq[24])
                    fk=geom.point_kinematics(fs,patch)
                    delta=fk.point_world-kin.point_world
                    delta_n=float(n@delta);delta_t=float(np.linalg.norm(delta-n*delta_n))
                rows.append(dict(time_s=t,patch_index=j,digit=patch.digit,robot_body=patch.robot_body,
                    pressure_N=float(raw['normal']),acquisition_time_s=patch.sample_time_s,age_s=t-patch.sample_time_s,
                    normal_robot_velocity_m_s=vn,tangent_robot_speed_m_s=vt,
                    actual_same_robot_point_20ms_inward_m=delta_n,
                    actual_same_robot_point_20ms_tangent_m=delta_t,
                    current_old_plane_offset_m=float(n@(kin.point_world-patch.reference_position))))
                counted+=1
            frames.append(dict(time_s=t,positive_public_patch_count=counted,
                total_public_patch_pressure_N=float(sum(x.get('normal',0) for x in patches)),
                authoritative_reference_lift_m=float(d['robot_reference21'][18]),
                actual_robot_lift_m=float(q[24]),skin_acquisition_time_s=d.get('sample_time')))
        def summarize(a,b):
            sub=[x for x in rows if a-1e-9<=x['time_s']<=b+1e-9]
            present=[x for x in frames if a-1e-9<=x['time_s']<=b+1e-9]
            if not sub:return dict(patch_rows=0,frames_with_observed_positive_patches=0,total_frames=len(present),not_proof_of_clearance=True)
            w=np.asarray([x['pressure_N'] for x in sub]); vn=np.asarray([x['normal_robot_velocity_m_s'] for x in sub]);vt=np.asarray([x['tangent_robot_speed_m_s'] for x in sub])
            future=[x for x in sub if x['actual_same_robot_point_20ms_inward_m'] is not None]
            fw=np.asarray([x['pressure_N'] for x in future]);dn=np.asarray([x['actual_same_robot_point_20ms_inward_m'] for x in future]);dt=np.asarray([x['actual_same_robot_point_20ms_tangent_m'] for x in future])
            return dict(patch_rows=len(sub),frames_with_observed_positive_patches=sum(x['positive_public_patch_count']>0 for x in present),total_frames=len(present),
                pressure_weighted_robot_normal_velocity_m_s=float(w@vn/w.sum()),
                pressure_weighted_robot_inward_velocity_positive_part_m_s=float(w@np.maximum(vn,0)/w.sum()),
                pressure_weighted_robot_tangent_speed_m_s=float(w@vt/w.sum()),
                max_robot_inward_velocity_m_s=float(max(vn)),min_robot_normal_velocity_m_s=float(min(vn)),
                max_robot_tangent_speed_m_s=float(max(vt)),
                pressure_weighted_fraction_with_positive_robot_inward_velocity=float(w@(vn>0)/w.sum()),
                actual_same_robot_point20ms_pressure_weighted_inward_m=None if not len(future) else float(fw@dn/fw.sum()),
                actual_same_robot_point20ms_pressure_weighted_tangent_m=None if not len(future) else float(fw@dt/fw.sum()))
        with gzip.open(p/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as f:
            release=next(json.loads(line) for line in f if json.loads(line)['time_s']>=12-1e-9)
        A=np.asarray(release['qp_problem']['A21x25'])
        # This rank is current kinematic transmission, NOT a servo model.
        s=np.linalg.svd(A,compute_uv=False)
        rank=int(np.linalg.matrix_rank(A))
        weighted=sorted(rows,key=lambda x:x['pressure_N']*max(x['normal_robot_velocity_m_s'],0),reverse=True)
        result=dict(role=role,motion_mode=motion,scenario=scene,physical_seed=619732,
            run_path=str(p),trace_sha256=sha(p/'trace.npz'),decision_sha256=sha(p/'joint_decisions.jsonl.gz'),
            public_robot_axis_names=list(geom.axes25),release_frame_count=len(indices),
            full_release=summarize(12,16),first_half_second=summarize(12,12.5),
            per_digit={str(d):None for d in range(5)},first_observed_inward=None,
            largest_pressure_times_positive_inward_velocity=weighted[:12],unavailable_rows=unavailable,
            first_release_transmission=dict(rank=rank,nullity=25-rank,singular_values=s.tolist(),
                role='known dl/dq and effort dual; not reference response'),
            authoritative_reference_lift_12_to16_m=[frames[0]['authoritative_reference_lift_m'],frames[-1]['authoritative_reference_lift_m']],
            last_observed_positive_patch_time_s=max((x['time_s'] for x in rows),default=None))
        for d in range(5):
            sub=[x for x in rows if x['digit']==d]
            if sub:
                w=np.asarray([x['pressure_N'] for x in sub]);vn=np.asarray([x['normal_robot_velocity_m_s'] for x in sub]);vt=np.asarray([x['tangent_robot_speed_m_s'] for x in sub])
                result['per_digit'][str(d)]=dict(patch_rows=len(sub),max_inward_robot_velocity_m_s=float(max(vn)),pressure_weighted_inward_positive_m_s=float(w@np.maximum(vn,0)/w.sum()),pressure_weighted_tangent_speed_m_s=float(w@vt/w.sum()))
        result['first_observed_inward']=next((x for x in rows if x['normal_robot_velocity_m_s']>0),None)
        results.append(result)
    doc=dict(schema='v16_public_robot_skin_motion_archived_six_v1',complete=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'v10/robot_contact_space.py']},
        scope=dict(physics_steps=0,forward_calls=0,contact_solve_calls=0,forbidden_pipeline_attempts=forbidden,
            robot_model='private nominal definition; no per-run object mass/friction used',
            observed_normal='hand-to-external at delayed acquisition, no propagated external plane',
            velocity='robot material point J_current v_current, NOT relative contact slip/object velocity',
            finite_displacement='archived q(t+20ms) same robot material point; evaluation only, never an online input',
            absent_public_patches='unknown geometry/no observed positive finger geometry; not all-hand clearance',
            actual_servo_map_or_pressure_accuracy_certified=False),records=results)
    out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),records=len(results),sha256=sha(out),forbidden_calls=len(forbidden)),ensure_ascii=False))


if __name__=='__main__':main()

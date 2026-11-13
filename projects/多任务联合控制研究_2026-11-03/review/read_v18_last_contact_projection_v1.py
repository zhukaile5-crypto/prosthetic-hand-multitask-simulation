"""Six fixed public packets plus pre-floor velocity decomposition; no step.

Observed device-plane absolute velocity = robot-point projected velocity plus
the same-acquisition relative-slip pair. This is not a full external twist or
future motion estimator and is never passed to a controller by this reader.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from hand_plant import _model
from foundation_config import FoundationConfig
from v10.robot_contact_space import RobotContactSpace,RobotJointSample,PublicSkinPatch
from v17_contact_transfer.motion_domain_sensors import SLIP_DOMAIN_ROLE

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    source=ROOT/'review/v18_release_two_case_mechanisms_v1.json';base=read(source)
    model=_model(FoundationConfig())[0];geometry=RobotContactSpace(model);results=[]
    guards=[patch('mujoco.'+n,side_effect=RuntimeError('forbidden integration/force pipeline')) for n in ('mj_step','mj_forward','mj_collision','mj_contactForce')]
    for g in guards:g.start()
    try:
        for record in base['records']:
            p=Path(record['run']);last=record['events']['last_target_hand_positive_interval_end_s'];floor=record['events']['first_target_floor_interval_end_s']
            with np.load(p/'trace.npz',allow_pickle=False) as z:
                times=z['time'];encoded=z['observation_json']
            # This supplement needs only six ledger columns, loaded once each.
            with np.load(p/'delivery_intervals.npz',allow_pickle=False) as z:
                lt=z['interval_end_s'];qp=z['object_free_qpos7_end'];qv=z['object_free_qvel6_end'];top=z['target_receiver_top_normal_N'];valid=z['object_state_valid'];start=z['interval_start_s']
            public={int(i):json.loads(str(encoded[i])) for i in np.flatnonzero((times>=12-1e-9)&(times<=16+1e-9))}
            packets=[]
            for offset in (.100,.050,.015):
                desired=last-offset;i=int(np.searchsorted(times,desired+1e-9)-1);o=public[i];acq=o['sample_time']
                ai=int(round(acq/.005));old=public[ai]
                q=np.asarray(old['robot_position25']);v=np.asarray(old['robot_velocity25'])
                sample=RobotJointSample(acq,geometry.raw_joint_names,q[:24],q[24],v[:24],v[24])
                palm=int(geometry.model.body('rh_palm').id)
                geometry.point_kinematics(sample,PublicSkinPatch(palm,np.zeros(3),np.array([0.,0.,1.]),acq))
                R=geometry.data.xmat[palm].reshape(3,3).copy();digits=[];unknown=[]
                for digit in range(5):
                    if not (o['valid'][digit] and o['slip_valid5'][digit] and o['slip_domain_available'] and o['slip_domain_role']==SLIP_DOMAIN_ROLE):
                        unknown.append(dict(digit=digit,reason='existing sensor validity/domain missing'));continue
                    ps=[x for x in o['skin_patches'] if x['digit']==digit]
                    if not ps or any(x['normal']<=0 or abs(x['sample_time']-acq)>1e-9 for x in ps):
                        unknown.append(dict(digit=digit,reason='coacquired positive patch geometry missing'));continue
                    weights=np.asarray([x['normal'] for x in ps]);weights=weights/weights.sum()
                    velocities=np.asarray([geometry.point_kinematics(sample,PublicSkinPatch.from_public(x)).point_velocity_world for x in ps])
                    mean=weights@velocities;B=np.asarray(o['tangent_basis'][digit]);s=np.asarray(o['slip_local'][digit]);hand=B@R.T@mean;e=hand+s
                    digits.append(dict(digit=digit,public_pressure_N=o['normal'][digit],patch_count=len(ps),pressure_shares=weights.tolist(),
                        acquired_robot_point_velocity_world_m_s=mean.tolist(),acquired_robot_plane_velocity_m_s=hand.tolist(),
                        observed_relative_slip2_m_s=s.tolist(),contact_average_absolute_projection2_m_s=e.tolist(),
                        robot_plane_speed_m_s=float(np.linalg.norm(hand)),relative_slip_speed_m_s=float(np.linalg.norm(s)),
                        external_plane_speed_m_s=float(np.linalg.norm(e)),
                        relative_slip_fraction_of_absolute=(float(np.linalg.norm(s)/np.linalg.norm(e)) if np.linalg.norm(e)>0 else None)))
                packets.append(dict(request_offset_before_last_target_hand_s=offset,public_time_s=float(times[i]),acquisition_s=acq,
                    current_robot_time_s=old['robot_control_time_s'],digits=digits,unknown=unknown))
            contact=np.flatnonzero((lt>12+1e-9)&(lt<=16+1e-9)&(top>0));lasttop=None if not len(contact) else float(lt[contact[-1]])
            def state(at,strict_before=False):
                j=int(np.searchsorted(lt,at-1e-9 if strict_before else at+1e-9,side='right')-1);j=max(0,j)
                return dict(interval_start_s=float(start[j]),interval_end_s=float(lt[j]),valid=bool(valid[j]),position_world_m=qp[j,:3].tolist(),
                    velocity_world_m_s=qv[j,:3].tolist(),horizontal_speed_m_s=float(np.linalg.norm(qv[j,:2])),vertical_speed_m_s=float(qv[j,2]),
                    angular_speed_rad_s=float(np.linalg.norm(qv[j,3:])),target_top_N=float(top[j]))
            events=dict(first_top=state(record['events']['first_target_top_interval_end_s']),last_hand=state(last),
                first_frame_clear=state(record['events']['original_frame_clearance_confirmation_s']),
                last_top=None if lasttop is None else state(lasttop),
                immediately_before_floor=state(floor,True),first_floor=state(floor),end=state(16.))
            segments=[]
            boundaries=[('top_to_last_hand',record['events']['first_target_top_interval_end_s'],last),
                        ('after_last_hand_before_floor',last,floor),('floor_and_after',floor,16.)]
            for name,a,b in boundaries:
                mask=(lt>=a-1e-9)&(lt<b-1e-9 if name!='floor_and_after' else lt<=b+1e-9)&valid
                ids=np.flatnonzero(mask)
                if not len(ids):segments.append(dict(name=name,available=False));continue
                vel=qv[ids];pos=qp[ids];initial=pos[0,:3];final=pos[-1,:3]
                segments.append(dict(name=name,start_s=float(lt[ids[0]]),end_s=float(lt[ids[-1]]),samples=len(ids),available=True,
                    horizontal_displacement_m=float(np.linalg.norm(final[:2]-initial[:2])),vertical_displacement_m=float(final[2]-initial[2]),
                    horizontal_speed_start_m_s=float(np.linalg.norm(vel[0,:2])),horizontal_speed_end_m_s=float(np.linalg.norm(vel[-1,:2])),
                    horizontal_speed_max_m_s=float(np.max(np.linalg.norm(vel[:,:2],axis=1))),
                    vertical_speed_start_m_s=float(vel[0,2]),vertical_speed_end_m_s=float(vel[-1,2]),
                    vertical_speed_min_m_s=float(np.min(vel[:,2])),vertical_speed_max_m_s=float(np.max(vel[:,2])),
                    angular_speed_start_rad_s=float(np.linalg.norm(vel[0,3:])),angular_speed_end_rad_s=float(np.linalg.norm(vel[-1,3:])),
                    angular_speed_max_rad_s=float(np.max(np.linalg.norm(vel[:,3:],axis=1)))))
            results.append(dict(context=record['context'],fixed_public_packets=packets,object_event_states=events,object_motion_segments=segments,
                last_target_top_interval_end_s=lasttop,input_sha256={n:sha(p/n) for n in ('trace.npz','delivery_intervals.npz')}))
    finally:
        for g in reversed(guards):g.stop()
    payload=dict(schema='v18_last_contact_projection_posthoc_v1',complete=True,records=results,reader_sha256=sha(__file__),base_sha256=sha(source),
        scope=dict(public_packets_per_case=3,offsets_before_last_offline_target_hand_s=[.1,.05,.015],
            offline_event_selection_not_online_actor_knowledge=True,known_robot_FK_only=True,physics_steps=0,contact_reads=0,global_forward_calls=0,
            device_projection_is_not_full_external_twist=True,no_controller_or_source_modified=True,
            no_floor_bounce_total_speed_attribution=True))
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for r in results:
        print(json.dumps(dict(context=r['context'],object_states=r['object_event_states'],segments=r['object_motion_segments'],
            public=[dict(time_s=x['public_time_s'],acq=x['acquisition_s'],digits=[{k:d[k] for k in ('digit','robot_plane_speed_m_s','relative_slip_speed_m_s','external_plane_speed_m_s')} for d in x['digits']]) for x in r['fixed_public_packets']]),ensure_ascii=False))
    print(json.dumps(dict(out=str(out),sha256=sha(out)),ensure_ascii=False))
if __name__=='__main__':main()

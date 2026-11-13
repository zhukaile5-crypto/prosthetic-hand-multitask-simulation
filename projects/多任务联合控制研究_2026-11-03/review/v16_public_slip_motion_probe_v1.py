"""Limited archived PUBLIC slip-twist identifiability probe, never control.

fit_tangent_twist consumes only per-finger public geometry/velocity channels.
Truth is accepted only by the separate offline_comparison function, after fit.
No actual object geometry/identity/material, future sample, model fit or gate
tuning is used in estimation. Numerical rank is not physical observability.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
TIMES=(11.5,12.05,12.25,12.45,12.65,12.85,13.25,13.5,14.)
RUNS=(('fragile_conflict','hold','output/v16b_goal_corrected_hold_v1/runs/fragile_conflict__ratio_effort__619732'),
      ('nominal','transport','output/v16b_goal_corrected_transport_v1/runs/nominal__ratio_effort__619732'),
      ('friction_drop','transport','output/v16b_goal_corrected_transport_v1/runs/friction_drop__ratio_effort__619732'))
LENGTH_M=.05  # Existing public wrench/lever normalization; no data selection.
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return plain(v.tolist())
 if isinstance(v,np.generic):return plain(v.item())
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def skew(p):
 x,y,z=p;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def fit_tangent_twist(rows,*,slip_sigma_m_s=None):
 """eta=[relative-origin-velocity, L*relative-angular-velocity] in m/s.

 One row pair per finger. Geometry and robot relative point velocities are
 co-acquired inputs; minimum norm is a representative, NOT a full state.
 """
 if len({r['digit'] for r in rows})!=len(rows):raise ValueError('One correlated velocity row pair per finger only')
 A=[];b=[]
 for r in rows:
  B=np.asarray(r['basis2x3']);p=np.asarray(r['contact_pos_palm_m']);vr=np.asarray(r['robot_relative_point_velocity_palm_m_s']);s=np.asarray(r['slip2_m_s'])
  if B.shape!=(2,3) or p.shape!=(3,) or vr.shape!=(3,) or s.shape!=(2,) or not np.isfinite(np.r_[B.ravel(),p,vr,s]).all():raise ValueError('Finite public row fields required')
  A.append(B@np.c_[np.eye(3),-skew(p)/LENGTH_M]);b.append(B@vr+s)
 A=np.vstack(A) if A else np.empty((0,6));b=np.concatenate(b) if b else np.empty(0)
 if not len(rows):return dict(status='UNKNOWN_NO_PUBLIC_CONTACT_VELOCITY_ROWS',rank=0,rows=0,singular_values=[],nullspace_basis6_by_nullity=np.eye(6),minimum_norm_representative_not_state=None,full_external_twist_state_available=False,fit_residual_rmse_m_s=None,design_matrix=A,measurement=b)
 U,s,Vt=np.linalg.svd(A,full_matrices=True);cutoff=max(A.shape)*np.finfo(float).eps*s[0];rank=int(np.sum(s>cutoff));V=Vt[:rank].T
 coeff=U[:,:rank].T@b/s[:rank];representative=V@coeff;residual=A@representative-b
 sigma=None if slip_sigma_m_s is None else float(slip_sigma_m_s)
 return dict(status='CONDITIONAL_UNVERIFIED_SINGLE_RIGID_ACTOR_LS',rank=rank,rows=len(b),finger_groups=len(rows),singular_values=s,numerical_rank_cutoff=cutoff,
  nullspace_basis6_by_nullity=Vt[rank:].T,observable_basis6_by_rank=V,observable_coefficients_m_s=coeff,
  minimum_norm_representative_not_state=representative,full_external_twist_state_available=False,
  scaled_translation_representative_m_s=representative[:3],angular_representative_rad_s=representative[3:]/LENGTH_M,
  fit_residual_rmse_m_s=float(np.linalg.norm(residual)/np.sqrt(len(residual))),per_finger_residual_norm_m_s=np.linalg.norm(residual.reshape(-1,2),axis=1),
  slip_noise_only_std_observable_modes_m_s=None if sigma is None else sigma/s[:rank],
  slip_noise_only_covariance_scope='Conditional on exact p/J/robot velocity and one actor; excludes correlated position errors-in-variables, model/actor mismatch and unknown nullspace. Not a posterior or hard bound.',
  unverified_single_actor_assumption=True,design_matrix=A,measurement=b)

def self_test():
 p=[np.array(x) for x in ((.02,0,.01),(-.01,.025,-.02),(.015,-.02,.03),(-.03,-.01,.02),(0,.02,.025))]
 normals=[np.array(x,float) for x in ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(0,1,1))]
 eta=np.array([.01,-.02,.03,.004,-.002,.003]);vp=np.array([.02,-.01,.005]);wp=np.array([.1,.2,-.1]);rows=[]
 for i,(x,n) in enumerate(zip(p,normals)):
  n/=np.linalg.norm(n);seed=np.eye(3)[np.argmin(abs(n))];b1=np.cross(n,seed);b1/=np.linalg.norm(b1);B=np.array([b1,np.cross(n,b1)]);vr=np.array([.003*i,-.001*i,.002*i]);vo=vp+np.cross(wp,x)+eta[:3]+np.cross(eta[3:]/LENGTH_M,x)
  rows.append(dict(digit=i,basis2x3=B,contact_pos_palm_m=x,robot_relative_point_velocity_palm_m_s=vr-vp-np.cross(wp,x),slip2_m_s=B@(vo-vr)))
 fit=fit_tangent_twist(rows);few=fit_tangent_twist(rows[:1]);empty=fit_tangent_twist([])
 return [dict(name='same-frame_relative_twist_sign_and_palm_motion_oracle',passed=fit['rank']==6 and np.max(abs(fit['minimum_norm_representative_not_state']-eta))<1e-14),
  dict(name='rank_deficiency_keeps_nullspace_and_no_full_state',passed=few['rank']==2 and few['nullspace_basis6_by_nullity'].shape==(6,4) and not few['full_external_twist_state_available']),
  dict(name='no_contacts_unknown_not_external_stationary_zero',passed=empty['minimum_norm_representative_not_state'] is None and empty['rank']==0)]

def public_rows(frame,acquired_robot_frame,geometry,mujoco):
 """Robot-only FK at tactile acquisition; no object/body identity used."""
 from v10.robot_contact_space import RobotJointSample,PublicSkinPatch
 a=float(frame['sample_time']);now=float(frame['time'])
 if abs(float(acquired_robot_frame['time'])-a)>1e-9 or a>now+1e-9:raise ValueError('No coacquired public robot history')
 q=np.asarray(acquired_robot_frame['robot_position25'],float);v=np.asarray(acquired_robot_frame['robot_velocity25'],float)
 sample=RobotJointSample(a,geometry.raw_joint_names,q[:24],q[24],v[:24],v[24]);palm=int(geometry.model.body('rh_palm').id)
 origin=geometry.point_kinematics(sample,PublicSkinPatch(palm,np.zeros(3),np.array([0.,0.,1.]),a))
 R=geometry.data.xmat[palm].reshape(3,3).copy();vp=origin.point_velocity_world;wp=origin.angular_jacobian3x25@v
 com=geometry.point_kinematics(sample,PublicSkinPatch(palm,np.asarray(geometry.model.body_ipos[palm]),np.array([0.,0.,1.]),a))
 n=np.asarray(frame['normal'],float);valid=np.asarray(frame['valid'],bool);basis=np.asarray(frame['tangent_basis'],float);slip=np.asarray(frame['slip_local'],float);pos=np.asarray(frame['contact_pos'],float)
 rows=[];excluded=[]
 for digit in range(5):
  patches=[p for p in frame.get('skin_patches',[]) if p.get('digit')==digit]
  if not valid[digit] or not np.isfinite(np.r_[n[digit],basis[digit].ravel(),slip[digit],pos[digit]]).all() or not patches:
   excluded.append(dict(digit=digit,reason='invalid_or_no_public_skin_geometry'));continue
  pressures=np.array([p['normal'] for p in patches],float)
  if not np.isfinite(pressures).all() or np.any(pressures<0) or pressures.sum()<=0:
   excluded.append(dict(digit=digit,reason='positive_public_patch_share_unavailable'));continue
  weights=pressures/pressures.sum();vel=[];points=[]
  for p in patches:
   if abs(float(p['sample_time'])-a)>1e-9:raise ValueError('Patch/velocity packet acquisition mismatch')
   k=geometry.point_kinematics(sample,PublicSkinPatch.from_public(p));vel.append(k.point_velocity_world);points.append(k.point_world)
  meanv=weights@np.array(vel);meanp=weights@np.array(points);vrp=R.T@(meanv-vp)-np.cross(R.T@wp,pos[digit])
  rows.append(dict(digit=digit,basis2x3=basis[digit],contact_pos_palm_m=pos[digit],robot_relative_point_velocity_palm_m_s=vrp,slip2_m_s=slip[digit],public_pressure_N=float(n[digit]),public_patch_count=len(patches),pressure_share_weights=weights,
   reported_vs_patch_centroid_m=float(np.linalg.norm(R.T@(meanp-origin.point_world)-pos[digit]))))
 return rows,excluded,dict(R=R,palm_origin_velocity_world=vp,palm_COM_velocity_world=com.point_velocity_world,palm_angular_world=wp)

def offline_comparison(fit,truth_relative,truth_velocity,truth_angular,kin):
 """AFTER fitting only. Historical truth convention is COM/body-origin mixed.
 Object origin=COM for these centered capsule archives; public palm COM/origin
 velocity difference is applied explicitly. No truth chooses rows/rank/gates.
 """
 R=kin['R'];omega=np.asarray(truth_angular);lever=np.asarray(truth_relative)
 rel0=np.asarray(truth_velocity)+R.T@(kin['palm_COM_velocity_world']-kin['palm_origin_velocity_world'])-np.cross(omega,lever)
 eta=np.r_[rel0,LENGTH_M*omega];A=fit['design_matrix'];b=fit['measurement'];rank=fit['rank']
 # Evaluate absolute object-COM speed independently from the legacy record.
 abs_v_COM=R@np.asarray(truth_velocity)+kin['palm_COM_velocity_world']+np.cross(kin['palm_angular_world'],R@lever)
 out=dict(offline_only=True,truth_eta_relative_at_palm_origin_m_s=eta,actual_object_COM_speed_world_m_s=float(np.linalg.norm(abs_v_COM)),actual_relative_angular_speed_rad_s=float(np.linalg.norm(omega)),
  truth_equation_rmse_m_s=float(np.linalg.norm(A@eta-b)/np.sqrt(len(b))) if len(b) else None,
  observable_mode_error_norm_m_s=None,unobservable_truth_component_norm_m_s=float(np.linalg.norm(eta)),
  truth_not_estimator_input_or_selection=True)
 if rank:
  V=fit['observable_basis6_by_rank'];out['observable_mode_error_norm_m_s']=float(np.linalg.norm(V.T@(fit['minimum_norm_representative_not_state']-eta)));out['unobservable_truth_component_norm_m_s']=float(np.linalg.norm(eta-V@(V.T@eta)))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--out',default='review/v16_public_slip_motion_probe_v1.json');args=p.parse_args();tests=self_test()
 if not args.execute:print(json.dumps(plain(dict(pure_checks=tests,actual_records_read=0))));return
 import research_imports,mujoco
 from hand_plant import _model
 from foundation_config import FoundationConfig
 from v10.robot_contact_space import RobotContactSpace
 forbidden=[]
 def no(*a,**k):forbidden.append('forbidden_pipeline');raise RuntimeError('No physical pipeline in archived public probe')
 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):setattr(mujoco,name,no)
 model,default_identity=_model(FoundationConfig());geometry=RobotContactSpace(model)
 rows=[];inputs=[]
 for scenario,motion,path in RUNS:
  folder=ROOT/path;meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'));archive=folder/'trace.npz'
  with np.load(archive,allow_pickle=False) as z:a={k:z[k] for k in ('time','observation_json','relative','relative_velocity','angular_velocity')}
  inputs.append(dict(run_directory=str(folder),recorded_trace_sha256=meta.get('trace_sha256'),metadata_sha256=sha(folder/'metadata.json'),whole_trace_rehashed=False))
  cache={}
  def packet(i):
   if i not in cache:cache[i]=json.loads(str(a['observation_json'][i]))
   return cache[i]
  for requested in TIMES:
   i=int(np.argmin(abs(a['time']-requested)));frame=packet(i);acq=float(frame['sample_time']);j=int(np.argmin(abs(a['time']-acq)))
   if abs(a['time'][j]-acq)>1e-9:raise ValueError('Exact coacquired recorded robot sample missing')
   public,excluded,kin=public_rows(frame,packet(j),geometry,mujoco);fit=fit_tangent_twist(public,slip_sigma_m_s=frame['sensor_characteristics']['slip_sigma_m_s'])
   evaluation=offline_comparison(fit,a['relative'][j],a['relative_velocity'][j],a['angular_velocity'][j],kin)
   rows.append(dict(scenario=scenario,motion=motion,requested_read_time_s=requested,actual_read_time_s=float(a['time'][i]),acquisition_time_s=acq,public_fingers=[r['digit'] for r in public],excluded=excluded,public_rows=public,fit=fit,offline_truth_evaluation=evaluation,
    position_sigma_m=frame['sensor_characteristics']['position_sigma_m'],actor_selection='All publicly valid finger groups; no target/actor IDs or template truth. Single actor assumption unverified.'))
 report=dict(schema='v16_public_slip_motion_probe_v1',pure_checks=tests,all_pure_checks_passed=all(c['passed'] for c in tests),fixed_requested_times_s=TIMES,packets=len(rows),runs=3,rows=rows,inputs=inputs,
  estimator_inputs='Public delayed slip_local/tangent_basis/contact_pos/skin_patches plus same-acquisition robot encoders/velocity and declared robot geometry only',
  estimator_twist_definition='Relative velocity field in coacquired palm frame: eta=[v_ext_at_palm_origin-v_palm_origin, L*(omega_ext-omega_palm)]',normalization_length_m=LENGTH_M,
  no_full_state_claim=True,one_actor_attribution_verified=False,new_model_fit=False,production_source_changed=False,physics_steps=0,forbidden_pipeline_calls=forbidden,
  model_scope='One declared default model used only for robot FK/Jacobian; actual external configuration not estimator input',
  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'skin_geometry.py',ROOT/'v10/robot_contact_space.py')},
  primary_velocity_reference='https://raw.githubusercontent.com/google-deepmind/mujoco/3.15.0/src/engine/engine_core_util.c#L778-L829')
 out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(dict(out=str(out),sha256=sha(out),packets=len(rows),ranks=[x['fit']['rank'] for x in rows],physics_steps=0,forbidden_pipeline_calls=forbidden),ensure_ascii=False))
if __name__=='__main__':main()

"""Read-only conditional rigid-motion geometry rank; no twist estimate/truth/model."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def skew(p):
 x,y,z=p;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def rank_record(C):
 if len(C):
  U,s,Vt=np.linalg.svd(C,full_matrices=True);tol=np.finfo(float).eps*max(C.shape)*s[0];rank=int(np.count_nonzero(s>tol))
 else:s=np.empty(0);tol=0.;rank=0;Vt=np.eye(6)
 return dict(conditional_C_rank=rank,numerical_rank_threshold=tol,singular_values=s.tolist(),nullspace_basis6_by_nullity=Vt[rank:].T.tolist(),full_rigid_state_available=False,actor_single_rigid_hypothesis_verified=False,rank_is_geometric_numerical_not_statistical_admission=True)
run=ROOT/'output/v23_small_joint_friction_drop_transport_619742_v1/runs/friction_drop__ratio_effort__619742';times=[11.5,13.63,13.64,14.,14.5,15.5,16.];L=.05
with np.load(run/'trace.npz',allow_pickle=False) as z:
 t=z['time'];obs=z['observation_json']
 rows=[]
 for req in times:
  i=int(np.argmin(abs(t-req)));f=json.loads(str(obs[i]));acq=float(f['sample_time']);mask=np.asarray(f['valid'],bool)&np.asarray(f['distributed_valid'],bool)&np.asarray(f['slip_valid5'],bool);T=np.asarray(f['tangent_basis']);pos=np.asarray(f['contact_pos']);public_patches=f['skin_patches'];groups=[];blocks=[]
  for digit in range(5):
   patches=[p for p in public_patches if p.get('digit')==digit and p.get('normal',0.)>0 and abs(p['sample_time']-acq)<=1e-9 and 0<=float(t[i])-p['sample_time']<=.12+1e-9]
   if not mask[digit] or not patches:continue
   C=T[digit]@np.column_stack((np.eye(3),-skew(pos[digit])/L));blocks.append(C);groups.append(dict(digit=digit,public_patch_count=len(patches),same_digit_rows=2,position_palm_m=pos[digit].tolist(),basis2x3=T[digit].tolist(),actor_identity='UNKNOWN_ALL_EXTERNAL_MIXTURE'))
  C=np.vstack(blocks) if blocks else np.empty((0,6));rr=rank_record(C);rows.append(dict(requested_time_s=req,actual_read_time_s=float(t[i]),acquisition_time_s=acq,age_s=float(f['age']),public_group_count=len(groups),public_groups=groups,conditional_C_scaled=C.tolist(),state_coordinates='instantaneous external velocity field at acquisition palm origin, expressed in its basis: [v_origin_palm_m_s,0.05*omega_palm_rad_s]; NOT relative derivative/CoM state',no_rows_status='UNKNOWN_NOT_STATIONARY' if not groups else None,**rr))
# Analytic planar example: all x/y projections on z=0 cannot see vz,wx,wy.
T=np.array([[1.,0.,0.],[0.,1.,0.]])
C=np.vstack([T@np.column_stack((np.eye(3),-skew(p))) for p in ([0.,0.,0.],[1.,0.,0.],[0.,1.,0.])]);pr=rank_record(C);assert pr['conditional_C_rank']==3 and np.array_equal(C@np.eye(6)[:,[2,3,4]],np.zeros((6,3)))
prior=ROOT/'review/v16_public_slip_motion_probe_v1.json';pj=json.loads(prior.read_text('utf8'));historical=[]
for z in pj['rows']:historical.append(dict(scenario=z['scenario'],requested_time_s=z['requested_read_time_s'],acquisition_time_s=z['acquisition_time_s'],fingers=z['public_fingers'],rank=z['fit'].get('rank'),status=z['fit'].get('status'),full_state_available=z['fit'].get('full_external_twist_state_available'),scope='Existing historical diagnostic only; no raw re-fit or truth use. Its twist is palm-relative, different from current absolute e.'))
cf=ROOT/'review/v23_joint_friction_first_divergence_v1.json';cj=json.loads(cf.read_text('utf8'));mapranks=[]
for z in cj['local_same_public_input_counterfactuals']:
 k=z['all_kwargs'];M=np.asarray(k['motion_matrix_Mx21']);B=np.asarray(k['proposal_map25x21']);mapranks.append(dict(input_trajectory=z['input_trajectory'],time_s=k['time_s'],M_reference_shape=list(M.shape),M_reference_rank=int(np.linalg.matrix_rank(M)),B_reference_rank=int(np.linalg.matrix_rank(B)),these_are_robot_reference_geometry_not_external_twist_observation_or_response=True))
inputs={'current_trace_sha256':sha(run/'trace.npz'),'existing_v16_probe_sha256':sha(prior),'saved_math_input_sha256':sha(cf)};out=ROOT/'review/v24_motion_observation_geometry_scope_v1.json';assert not out.exists();j=dict(schema='v24_motion_observation_geometry_scope_v1',complete=True,normalization_length_m=L,normalization_role='existing WRENCH_LENGTH=.05 for comparable v,Lomega columns; not a new physical threshold',fixed_public_rows=rows,planar_rank_deficiency_example=pr,existing_v16_conditional_ranks=historical,current_reference_map_ranks=mapranks,input_sha256=inputs,reader_sha256=sha(__file__),physics_steps=0,model_constructions=0,twist_estimates_produced=0,object_truth_inputs=False,controller_inputs_changed=False,scope='One completed seen friction trajectory at seven fixed read times; geometric rank under unverified common rigid actor assumption only. No model/engine/FK/step, no statistical observability certificate or physical response fit. No contact is UNKNOWN, never zero twist.')
out.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({'out':str(out),'sha256':sha(out),'fixed_ranks':[(z['actual_read_time_s'],z['public_group_count'],z['conditional_C_rank']) for z in rows],'planar_rank':pr['conditional_C_rank'],'M_ranks':mapranks},ensure_ascii=False))

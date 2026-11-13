"""Existing archive support and finite identifiability counterexamples, no model fit."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def svd_info(X):
 X=np.asarray(X,float)
 if not len(X):return dict(rows=0,columns=int(X.shape[1]),rank=0,singular_values=[],rank_threshold=0.)
 s=np.linalg.svd(X,compute_uv=False);tol=np.finfo(float).eps*max(X.shape)*s[0] if len(s) else 0.
 return dict(rows=len(X),columns=X.shape[1],rank=int(np.count_nonzero(s>tol)),singular_values=s.tolist(),rank_threshold=tol,rank_not_causal_identifiability_or_persistent_excitation_certificate=True)
# Exact finite closed-loop nonidentifiability (unknown drift and input response).
eps=np.array([.1,-.2,.15,.05,-.1,.2,0.]);ys=[]
for a,g in [(.8,.2),(1.,.4)]:
 y=[1.]
 for e in eps:y.append((a-g)*y[-1]+e)
 ys.append(np.asarray(y))
assert np.max(abs(ys[0]-ys[1]))<5e-16
# Same closed-loop coefficient .6, but a controlled u=1 at y=0 differs .2 vs .4.
# Low-rank input cannot identify the unexcited complement.
D=np.array([[1.],[0.]]);G1=np.array([[1.,0.]]);G2=np.array([[1.,10.]])
assert np.array_equal(G1@D,G2@D) and (G1@np.array([0.,1.]))[0]!=(G2@np.array([0.,1.]))[0]
# Moving projection plane can create apparent motion suppression without velocity change.
v=np.array([1.,0.,0.]);T0=np.array([[1.,0.,0.],[0.,1.,0.]]);T1=np.array([[0.,1.,0.],[0.,0.,1.]])
assert np.linalg.norm(T0@v)==1. and np.linalg.norm(T1@v)==0.
run=ROOT/'output/v23_small_joint_friction_drop_transport_619742_v1/runs/friction_drop__ratio_effort__619742'
with np.load(run/'trace.npz',allow_pickle=False) as z:t=z['time'];u=z['executed_reference21']
prior=ROOT/'review/v23_joint_friction_first_divergence_v1.json';rates=np.asarray(json.loads(prior.read_text('utf8'))['local_same_public_input_counterfactuals'][0]['all_kwargs']['reference_rates21']);units=rates*.005
motion={};phase={};available=0;clocks=[]
with gzip.open(run/'release_decisions.jsonl.gz','rt',encoding='utf8') as f:
 for line in f:
  z=json.loads(line);tick=round(z['time_s']/.005);phase[tick]=z['phase'];ct=z.get('contact_tasks')
  if not isinstance(ct,dict) or ct.get('motion_matrix_Mx21') is None:continue
  md=ct['diagnostics']['motion'];groups=tuple(x['digit'] for x in md['digit_details']);acq=md['acquisition_time_s'];motion[tick]=dict(groups=groups,acquisition_time_s=acq);available+=1;clocks.append(z['time_s']-acq)
# Full801 output records, but t=16 has no following physical interval.
first=round(12./.005);last=round(16./.005);X=(u[first:last]-u[first-1:last-1])/units
windows=Counter();groupX=defaultdict(list);groupX84=defaultdict(list);mode_changes=0
# At control t, newest y is acquired t-.01. Target acquisition t+.02 read at t+.03.
for i in range(first,last-6+1):
 if i not in motion or i+6 not in motion:windows['missing_endpoint_saved_projection']+=1;continue
 windows['both_saved_endpoints_available']+=1
 span=[motion.get(k) for k in range(i,i+7)]
 if any(v is None for v in span):windows['interior_saved_projection_gap']+=1;continue
 if any(v['groups']!=span[0]['groups'] for v in span):windows['digit_mask_changes_in30ms_observation_span']+=1;continue
 windows['same_saved_digit_group_all7_read_packets']+=1
 mode=phase[i];key=(mode,span[0]['groups']);groupX[key].append((u[i]-u[i-1])/units);groupX84[key].append(((u[i:i+4]-u[i-1])/units).reshape(-1))
 if any(phase[k]!=mode for k in range(i,i+5)):mode_changes+=1
support=[]
for key in sorted(groupX):support.append(dict(start_public_phase=key[0],saved_digit_group=list(key[1]),initial_21_increment_design=svd_info(groupX[key]),actual_declared_future4tick_command_design=svd_info(groupX84[key]),postselected_support_diagnostic_not_a_causal_training_cohort=True))
examples=dict(closed_loop={'policy':'u=-y','plants':[{'a':.8,'g':.2},{'a':1.,'g':.4}],'same_closed_loop_coefficient':.6,'shared_noise_sequence':eps.tolist(),'maximum_observed_trajectory_difference':float(np.max(abs(ys[0]-ys[1]))),'controlled_u1_y0_response':[.2,.4],'interpretation':'Without extra assumptions/excitation, feedback archive can identify a-g rather than separate drift a and causal input gain g.'},low_rank_input={'D':D.tolist(),'G1':G1.tolist(),'G2':G2.tolist(),'GD_same':(G1@D).tolist(),'unexcited_u':[0.,1.],'predictions_outside_support':[0.,10.],'interpretation':'Learning GD does not identify G on the full action domain.'},changing_projection={'world_velocity_same':v.tolist(),'T0':T0.tolist(),'T1':T1.tolist(),'projection0':(T0@v).tolist(),'projection1':(T1@v).tolist(),'interpretation':'Observed projection energy can fall solely because its plane changed; do not treat cross-time device components as one fixed velocity vector.'})
payload=dict(schema='causal_projection_response_identifiability_probe_v1',complete=True,pure_examples=examples,archive=dict(run=str(run),release_decision_count=len(phase),saved_projection_count=available,observation_delay_s=dict(minimum=min(clocks),maximum=max(clocks)),actual_executed_reference_increment_design_all800=svd_info(X),endpoint_window_counts=dict(windows),stable_saved_support_groups=support,public_phase_changed_within_conditioned_windows=mode_changes,randomized_exogenous_reference_assignment_logged=False,scope='One completed seen V23 trajectory; saved task projection availability, not all possible hardware observation availability. Input rank is a descriptive finite matrix rank, not causal identification. Same-mask window counts are explicitly postselected diagnostics, never survival-selected training admission.'),causal_time_contract=dict(control_interval_s=.005,device_delay_s=.01,model_future_acquisition_horizon_s=.02,last_measurement_available_at_control_t='acquisition t-.01',future_response_label='acquisition t+.02, read t+.03',future_zero_not_online=True),input_sha256={'trace':sha(run/'trace.npz'),'release_decisions':sha(run/'release_decisions.jsonl.gz'),'prior_math_inputs':sha(prior)},reader_sha256=sha(__file__),object_truth_inputs=False,model_fit=False,optimization_calls=0,physics_steps=0,model_constructions=0,production_changes=0)
out=ROOT/'review/causal_projection_response_identifiability_probe_v1.json';assert not out.exists();out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({'out':str(out),'sha256':sha(out),'saved_projection_count':available,'all800_input_rank':payload['archive']['actual_executed_reference_increment_design_all800']['rank'],'window_counts':dict(windows),'groups':[(z['start_public_phase'],z['saved_digit_group'],z['initial_21_increment_design']['rows'],z['initial_21_increment_design']['rank'],z['actual_declared_future4tick_command_design']['rank']) for z in support]},ensure_ascii=False))

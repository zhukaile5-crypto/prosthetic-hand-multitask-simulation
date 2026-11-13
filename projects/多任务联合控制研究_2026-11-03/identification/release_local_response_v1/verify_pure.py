"""Pure retained-data arithmetic only. Does not import MuJoCo or construct plant."""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
from datetime import datetime,timezone
import argparse,ast,json,sys,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from identification.release_local_response_v1.decoder import NativeEntry,NativePath2D,ReleaseCoordinate,ProbeAuthority,assess_increment,Robot25Authority,assess_actual_robot25
from identification.release_local_response_v1.snapshot_gate import SavedEntry,RestoreUnavailable,restore_zero_step_gate,require_physical_replay_certificate,verify_parent_sources,sha,mapping
PACKAGE=Path(__file__).resolve().parent
RUN=ROOT/'output/release_basis_minimal_v2'
def verify():
 checks={};plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'))
 for key in ['lock','independent_raw_audit','robot_geometry_source','robot_geometry_limits_review']:
  checks['input_'+key+'_identity']=sha(ROOT/plan['retained_inputs'][key])==plan['retained_inputs'][key+'_sha256']
 manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8-sig'));site=Path(manifest['environment']['executable']).parent.parent/'Lib/site-packages'
 def current(key):
  if key.startswith('research/'):return ROOT/key.removeprefix('research/')
  if key.startswith('foundation/'):return ROOT.parent/'五指策略接入_2026-11-03'/key.removeprefix('foundation/')
  if key.startswith('external/mujoco/'):return site/'mujoco'/key.removeprefix('external/mujoco/')
  if key.startswith('external/mujoco_distribution/'):return site/key.removeprefix('external/mujoco_distribution/')
  raise ValueError(key)
 parent=verify_parent_sources(RUN,current);checks['full230_parent_snapshot_current']=len(parent)==230
 source=RUN/'sources/foundation/foundation_config.py';values={}
 for node in ast.parse(source.read_text(encoding='utf-8')).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('OPEN','CLOSED') for t in node.targets):values[node.targets[0].id]=ast.literal_eval(node.value)
 xml=ET.parse(RUN/'sources/foundation/assets/legacy_model.xml').getroot();actuators=list(xml.find('actuator'));actuator_names=[a.get('name') for a in actuators]
 bounds={d.get('class'):np.fromstring(d.find('position').get('ctrlrange'),sep=' ') for d in xml.iter('default') if d.find('position') is not None and d.find('position').get('ctrlrange')}
 lower=np.array([bounds[a.get('class')][0] for a in actuators[2:20]]);upper=np.array([bounds[a.get('class')][1] for a in actuators[2:20]])
 def unarray(v):return np.frombuffer(__import__('base64').b64decode(v['bytes_b64']),dtype=np.dtype(v['array_dtype'])).reshape(v['shape']).copy()
 snapshots=[];total=0;exact=True
 for shape in ['capsule','box','sphere']:
  for path in ['native2d','fixed_height']:
   d=RUN/'runs'/f'{shape}__{path}';snapshot=SavedEntry.load_v2(d,RUN);snapshots.append(snapshot);side=mapping(snapshot.encoded_side);plant=mapping(side['plant_side']);metadata=json.loads((d/'record/metadata.json').read_text(encoding='utf-8-sig'))
   entry=NativeEntry(unarray(plant['alpha']),snapshot.physics['ctrl'][2:7],float.fromhex(plant['lift_ref']['float_hex']),unarray(plant['wrist_ref']),metadata['model_sha256'])
   decoder=NativePath2D(entry,open_pose=values['OPEN'],closed_pose=values['CLOSED'],finger_lower18=lower,finger_upper18=upper,release_open_rate=metadata['config']['release_open_rate'])
   with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:
    phi=z['path_time'];command=z['actual_command21'];native=z['native_command21']
    for i in range(len(phi)):
     q=ReleaseCoordinate(float(phi[i]),float(command[i,18]));candidate=decoder.decode(q)
     exact &= bool(np.array_equal(candidate.command21,command[i]))
     if path=='native2d':exact &= bool(np.array_equal(decoder.decode(decoder.native_coordinate(float(phi[i]))).command21,native[i]))
    total+=len(phi)
 checks['all52806_actual_release_command21_rows_bitexact']=exact and total==52806
 checks['six_complete_persisted_snapshot_checksums']=len(snapshots)==6
 checks['all3_full_causal_entries_pair_exact']=all(snapshots[i].causal_digest==snapshots[i+1].causal_digest for i in range(0,6,2))
 def rejects(call):
  try:call()
  except (ValueError,RestoreUnavailable,TypeError):return True
  return False
 checks['missing_real_hydration_adapter_refuses']=rejects(lambda:restore_zero_step_gate(snapshots[0]))
 checks['missing_real_replay_certificate_refuses']=rejects(require_physical_replay_certificate)
 checks['no_fake_certificate_is_accepted']=rejects(lambda:require_physical_replay_certificate({'passed':True}))
 class EchoOnlyMathStub:
  def source_identity(self):return dict(snapshots[0].source_identity)
  def restore_without_sampling_or_advancing(self,s):return s
  def capture_without_sampling_or_advancing(self,s):return s
 checks['synthetic_echo_integrity_interface_only']=restore_zero_step_gate(snapshots[0],EchoOnlyMathStub()).causal_digest==snapshots[0].causal_digest
 class BadIdentity(EchoOnlyMathStub):
  def source_identity(self):return {}
 checks['wrong_implementation_source_refuses']=rejects(lambda:restore_zero_step_gate(snapshots[0],BadIdentity()))
 class BadState(EchoOnlyMathStub):
  def restore_without_sampling_or_advancing(self,s):
   p=dict(s.physics);p['qpos']=p['qpos'].copy();p['qpos'][0]+=.001
   return replace(s,physics=p)
 checks['qpos_only_or_changed_state_roundtrip_refuses']=rejects(lambda:restore_zero_step_gate(snapshots[0],BadState()))
 checks['booleans_strings_nonfinite_progress_refused']=all(rejects(lambda x=x:ReleaseCoordinate(x,.1)) for x in [True,'0',np.nan,np.inf,-.1])
 checks['extra_hidden_finger_group_refused']=rejects(lambda:NativePath2D(entry,open_pose=values['OPEN']+[values['OPEN'][0]],closed_pose=values['CLOSED'],finger_lower18=lower,finger_upper18=upper,release_open_rate=.8))
 a=decoder.decode(ReleaseCoordinate(0.,entry.executed_lift_reference_m));b=decoder.decode(ReleaseCoordinate(.005,entry.executed_lift_reference_m));c=decoder.decode(ReleaseCoordinate(.004,entry.executed_lift_reference_m))
 # Synthetic in-memory envelope exercises validator mechanics; it is NOT a
 # measured robot rate/basis/force artifact and is never written into plan.
 auth=ProbeAuthority(np.r_[lower,0.,-.523599,-.698132],np.r_[upper,.1,.174533,.488692],entry.robot_model_sha256)
 kwargs=dict(anchor_command21=a.command21,total_variation_so_far21=np.zeros(21),dt_s=.005,authority=auth,expected_model_sha256=entry.robot_model_sha256,anchor_phi_s=0.)
 checks['missing_rate_cumulative_authority_not_zero']=not assess_increment(a,b,**kwargs).mechanically_eligible
 complete=replace(auth,declared_rate21=np.full(21,1e6),declared_increment21=np.full(21,1e6),declared_total_variation21=np.full(21,1e6),declared_progress_rate_s_s=2.,declared_total_progress_s=.02,local_phi_interval_s=[0.,.8],source_note='synthetic validator fixture only')
 kwargs['authority']=complete
 result=assess_increment(a,b,**kwargs);checks['mechanical_math_never_certifies_force']=result.mechanically_eligible and not result.force_response_available and not result.material_safety_certified
 checks['no_actual_progress_reversal']=not assess_increment(b,c,**kwargs).mechanically_eligible
 kwargs['authority']=replace(complete,declared_rate21=np.full(21,1e-9));checks['actual21_rate_enforced']=not assess_increment(a,b,**kwargs).mechanically_eligible
 kwargs['authority']=replace(complete,declared_increment21=np.full(21,1e-9));checks['actual21_finite_envelope_enforced']=not assess_increment(a,b,**kwargs).mechanically_eligible
 kwargs['authority']=replace(complete,declared_total_variation21=np.full(21,1e-9));checks['actual21_total_variation_enforced']=not assess_increment(a,b,**kwargs).mechanically_eligible
 axes=tuple('rh_fixture_'+str(i) for i in range(24))+('transport_lift',);robot=Robot25Authority(axes,np.ones(25,bool),np.full(25,-1.),np.full(25,1.),entry.robot_model_sha256)
 checks['actual_joint_rate_missing_does_not_pass']=not assess_actual_robot25(np.zeros(25),None,np.zeros(25),axes25=axes,authority=robot).mechanically_eligible
 checks['negative_actual_cumulative_travel_refused']=rejects(lambda:assess_actual_robot25(np.zeros(25),None,-np.ones(25),axes25=axes,authority=robot))
 checks['no_mujoco_or_physics_import']='mujoco' not in sys.modules
 checks['plan_draft_and_no_model_artifact']=plan['physics_runs_authorized_by_this_package'] is False and all(v=='missing' for v in plan['current_artifacts'].values())
 if not all(checks.values()):raise AssertionError(checks)
 return dict(schema='release_local_response_draft_pure_validation_v1',checks=checks,passed=True,physics_steps=0,retained_decoder_rows_compared=total,persisted_entries_checked=6,actual_restore_or_physical_replay_performed=False,synthetic_fixture_scope='Echo object and large envelopes only exercise pure validation; they are not real restoration/rate/response data.',execution_allowed=False,status=plan['status'],source_sha256={p.name:sha(p) for p in PACKAGE.iterdir() if p.is_file()},parent230_source_identity=parent)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out');args=p.parse_args();result=verify();out=Path(args.out) if args.out else PACKAGE/'qa'/('pure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
 if out.exists():raise FileExistsError(out)
 if not out.resolve().is_relative_to(ROOT):raise ValueError('New project review/QA output only')
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(output=str(out),sha256=sha(out),passed=result['passed'],checks=len(result['checks']),physics_steps=0,execution_allowed=False)))
if __name__=='__main__':main()

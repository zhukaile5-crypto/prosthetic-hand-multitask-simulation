"""Retained native decoder and tick algebra only. No plant/MuJoCo import."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from identification.release_local_response_v2.decoder import coordinates,native_envelope,VARIANTS,TRAIN,ReleasePath2D
PACKAGE=Path(__file__).resolve().parent;RUN=ROOT/'output/release_basis_minimal_v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'));d=RUN/'runs/capsule__native2d';side=json.loads((d/'entry_side_state.json').read_text(encoding='utf-8'));mapping=lambda x:dict(x.get('state',x)['mapping_items']);plant=mapping(mapping(side)['plant_side'])
 array=lambda x:np.frombuffer(base64.b64decode(x['bytes_b64']),dtype=x['array_dtype']).reshape(x['shape']).copy()
 values={}
 for n in ast.parse((RUN/'sources/foundation/foundation_config.py').read_text(encoding='utf-8')).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('OPEN','CLOSED') for t in n.targets):values[n.targets[0].id]=ast.literal_eval(n.value)
 xml=ET.parse(RUN/'sources/foundation/assets/legacy_model.xml').getroot();bounds={x.get('class'):np.fromstring(x.find('position').get('ctrlrange'),sep=' ') for x in xml.iter('default') if x.find('position') is not None and x.find('position').get('ctrlrange')};a=list(xml.find('actuator'))[2:20]
 with np.load(d/'entry_physics.npz',allow_pickle=False) as z:thumb=z['ctrl'][2:7]
 decoder=ReleasePath2D(entry_alpha=array(plant['alpha']),entry_thumb=thumb,entry_lift=float.fromhex(plant['lift_ref']['float_hex']),entry_wrist=array(plant['wrist_ref']),open_pose=values['OPEN'],closed_pose=values['CLOSED'],finger_lower=[bounds[x.get('class')][0] for x in a],finger_upper=[bounds[x.get('class')][1] for x in a],release_open_rate=.8)
 envelope=native_envelope(decoder,2400);checks={};records={}
 with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:
  tau=z['path_time'];tick=np.rint(z['control_time']/.005).astype(int)-2400;old=z['actual_command21'];physical=z['physical_time']
  zero=np.array([decoder.decode(**dict(path_time_s=coordinates(decoder,float(t),int(k),'zero',release_control_tick=2400,release_time_s=12.)['phi_s'],lift_reference=coordinates(decoder,float(t),int(k),'zero',release_control_tick=2400,release_time_s=12.)['lift_reference_m'])).command21 for t,k in zip(tau,tick)])
  checks['all8801_zero_command21_old_native_bitexact']=np.array_equal(zero,old)
  checks['old_control_held_clock_empirically_confirmed']=int(np.sum(np.diff(tick)==0))==8000 and int(np.sum(np.diff(tick)==1))==800 and int(np.sum(abs(np.diff(physical)-.0005)<1e-8))==8000
 for variant in VARIANTS:
  ticks=np.arange(801);tau=(2400+ticks)*.005-12.;info=[coordinates(decoder,float(t),int(k),variant,release_control_tick=2400,release_time_s=12.) for t,k in zip(tau,ticks)];phi=np.array([x['phi_s'] for x in info]);lift=np.array([x['lift_reference_m'] for x in info]);command=np.array([decoder.decode(x['phi_s'],x['lift_reference_m']).command21 for x in info]);zero=np.array([decoder.at(t,'native2d').command21 for t in tau])
  checks[variant+'_monotone_slow_only_progress']=bool(np.all(np.diff(phi)>=-1e-12) and np.all(np.diff(phi)<=.005+1e-12))
  checks[variant+'_monotone_native_lift_speed']=bool(np.all(np.diff(lift)<=1e-12) and np.all(-np.diff(lift)<=decoder.entry_lift/.6*.005+1e-12))
  checks[variant+'_no_new21_reference_envelope']=bool(np.all(abs(np.diff(command,axis=0))<=envelope['maximum_held_tick_increment21']+1e-10) and np.all(abs(np.diff(command,axis=0)).sum(axis=0)<=envelope['total_variation21']+1e-9))
  checks[variant+'_native_through_origin_inclusive']=np.array_equal(command[:5],zero[:5])
  checks[variant+'_actual_offset_matches_realzero']=bool(np.allclose(phi-tau,[x['actual_phi_offset_s'] for x in info],rtol=0,atol=0) and np.array_equal(lift-zero[:,18],np.array([x['actual_lift_offset_m'] for x in info])))
  checks[variant+'_lift_final_floor_not_positive_offset_forever']=lift[-1]==0.
  checks[variant+'_wrist_unchanged']=np.array_equal(command[:,19:],zero[:,19:])
  records[variant]=dict(at_origin=info[4],at_first_quarter=info[5],at_full20ms=info[8],last=info[-1],maximum_actual_phi_offset_s=float(abs(phi-tau).max()),maximum_actual_lift_offset_m=float((lift-zero[:,18]).max()),lift_clipped_application_ticks=sum(x['lift_clipped'] for x in info))
 checks['ramp_four_quarters_after_origin']=[coordinates(decoder,(2400+k)*.005-12.,k,'phi_slow',release_control_tick=2400,release_time_s=12.)['ramp_weight'] for k in range(4,9)]==[0.,.25,.5,.75,1.]
 checks['three_axis_training_four_mixed_holdout']=len(VARIANTS)==7 and tuple(plan['training'])==TRAIN and set(plan['strict_holdout'])==set(VARIANTS)-set(TRAIN)
 checks['no_mujoco_or_plant_import']='mujoco' not in sys.modules and 'hand_plant' not in sys.modules
 checks['no_force_model_or_safety_certificate']=plan['force_response_model_available'] is False and plan['material_safety_certified'] is False
 checks={k:bool(v) for k,v in checks.items()}
 if not all(checks.values()):raise AssertionError(checks)
 return dict(schema='release_local_response_v2_pure_v1',passed=True,checks=checks,physics_steps=0,plant_constructed=False,old_native_applications_compared=8801,records=records,source_sha256={p.name:sha(p) for p in PACKAGE.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ('plan.json','README.md'))},scope='Tick algebra/actual native reference equivalence only. Full7 physics/entry/prefix/zero16s acceptance must be executed by root; not claimed here.')
if __name__=='__main__':
 report=verify();out=PACKAGE/'qa'/('pure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(output=str(out),sha256=sha(out),checks=len(report['checks']),passed=True,physics_steps=0)))

"""Native held-clock/reference and failure-ledger algebra; no plant/MuJoCo import."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from identification.release_local_response_v3.decoder import coordinates,native_envelope,ORIGINS,VARIANTS,TRAIN,ReleasePath2D
PACKAGE=Path(__file__).resolve().parent;RUN=ROOT/'output/release_basis_minimal_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def verify():
 plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'));d=RUN/'runs/capsule__native2d';side=json.loads((d/'entry_side_state.json').read_text(encoding='utf-8'))
 mapping=lambda x:dict(x.get('state',x)['mapping_items']);plant=mapping(mapping(side)['plant_side'])
 array=lambda x:np.frombuffer(base64.b64decode(x['bytes_b64']),dtype=x['array_dtype']).reshape(x['shape']).copy()
 values={}
 for n in ast.parse((RUN/'sources/foundation/foundation_config.py').read_text(encoding='utf-8')).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('OPEN','CLOSED') for t in n.targets):values[n.targets[0].id]=ast.literal_eval(n.value)
 xml=ET.parse(RUN/'sources/foundation/assets/legacy_model.xml').getroot()
 bounds={x.get('class'):np.fromstring(x.find('position').get('ctrlrange'),sep=' ') for x in xml.iter('default') if x.find('position') is not None and x.find('position').get('ctrlrange')};actuators=list(xml.find('actuator'))[2:20]
 with np.load(d/'entry_physics.npz',allow_pickle=False) as z:thumb=z['ctrl'][2:7]
 decoder=ReleasePath2D(entry_alpha=array(plant['alpha']),entry_thumb=thumb,entry_lift=float.fromhex(plant['lift_ref']['float_hex']),entry_wrist=array(plant['wrist_ref']),open_pose=values['OPEN'],closed_pose=values['CLOSED'],finger_lower=[bounds[x.get('class')][0] for x in actuators],finger_upper=[bounds[x.get('class')][1] for x in actuators],release_open_rate=.8)
 envelope=native_envelope(decoder,2400);checks={};records={};zeros=[]
 with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:
  oldtau=z['path_time'];oldtick=np.rint(z['control_time']/.005).astype(int)-2400;old=z['actual_command21'];physical=z['physical_time']
 checks['real_native_has8801_applications8000_substeps']=len(old)==8801 and int(np.sum(np.diff(oldtick)==0))==8000 and int(np.sum(np.diff(oldtick)==1))==800 and int(np.sum(abs(np.diff(physical)-.0005)<1e-8))==8000
 for stage,origin in ORIGINS.items():
  coord=lambda tau,tick,variant:coordinates(decoder,float(tau),int(tick),variant,release_control_tick=2400,release_time_s=12.,probe_origin_tick=origin)
  zero=np.array([decoder.decode(x['phi_s'],x['lift_reference_m']).command21 for x in [coord(t,k,'zero') for t,k in zip(oldtau,oldtick)]])
  checks[stage+'_all8801_zero_commands_old_native_bitexact']=np.array_equal(zero,old);zeros.append(zero)
  ticks=np.arange(801);tau=(2400+ticks)*.005-12.;native=np.array([decoder.at(t,'native2d').command21 for t in tau])
  stages={}
  for variant,(a,b) in VARIANTS.items():
   infos=[coord(t,k,variant) for t,k in zip(tau,ticks)];phi=np.array([x['phi_s'] for x in infos]);lift=np.array([x['lift_reference_m'] for x in infos]);command=np.array([decoder.decode(x['phi_s'],x['lift_reference_m']).command21 for x in infos]);prefix=stage+'_'+variant
   checks[prefix+'_native_through_origin_inclusive']=np.array_equal(command[:origin+1],native[:origin+1])
   checks[prefix+'_monotonic_phi_rate0to1']=bool(np.all(np.diff(phi)>=-1e-12) and np.all(np.diff(phi)<=.005+1e-12))
   checks[prefix+'_nonincreasing_lift_native_speed']=bool(np.all(np.diff(lift)<=1e-12) and np.all(-np.diff(lift)<=decoder.entry_lift/.6*.005+1e-12))
   checks[prefix+'_actual21_step_and_total_variation_native_envelope']=bool(np.all(abs(np.diff(command,axis=0))<=envelope['maximum_held_tick_increment21']+1e-10) and np.all(abs(np.diff(command,axis=0)).sum(axis=0)<=envelope['total_variation21']+1e-9))
   checks[prefix+'_actual_input_against_zero_exact']=bool(np.array_equal(phi-tau,[x['actual_phi_offset_s'] for x in infos]) and np.array_equal(lift-native[:,18],[x['actual_lift_offset_m'] for x in infos]))
   checks[prefix+'_four_quarter_weights']=np.array_equal([x['ramp_weight'] for x in infos[origin:origin+5]],[0,.25,.5,.75,1.])
   checks[prefix+'_full_input_actual_phi_amplitude']=abs((phi[origin+4]-tau[origin+4])-(-a*.02))<1e-12
   checks[prefix+'_full_input_actual_lift_amplitude']=abs((lift[origin+4]-native[origin+4,18])-b*decoder.entry_lift/.6*.02)<1e-12
   checks[prefix+'_constant_entry_wrists']=np.array_equal(command[:,19:],native[:,19:])
   floor_tick=int(np.flatnonzero(lift==0.)[0]);floor_tolerance_tick=int(np.flatnonzero(lift<=1e-12)[0]);complete=np.max(abs(command[:,:18]-native[-1,:18]),axis=1)<=1e-12;final_tick=int(np.flatnonzero(complete)[0])
   checks[prefix+'_lift_within1e12_floor_by12p620_strictzero_by12p625']=bool((2400+floor_tolerance_tick)*.005<=12.620+1e-12 and (2400+floor_tick)*.005<=12.625+1e-12 and lift[-1]==0.)
   checks[prefix+'_phi_offset_retained_no_catchup']=abs((phi[-1]-tau[-1])-(-a*.02))<1e-12
   checks[prefix+'_all_finger_terminal_by14p225']=bool((2400+final_tick)*.005<=14.225+1e-12 and np.array_equal(command[-1,:18],native[-1,:18]))
   checks[prefix+'_not_falsely_completed_at13']=not bool(complete[200])
   stages[variant]=dict(at_origin=infos[origin],first_quarter=infos[origin+1],at_full20ms=infos[origin+4],last=infos[-1],full_control_time_s=(2400+origin+4)*.005,first_full_input_physical_end_s=(2400+origin+4)*.005+.0005,first_lift_floor_within1e12m_control_time_s=(2400+floor_tolerance_tick)*.005,first_lift_strict_bitzero_control_time_s=(2400+floor_tick)*.005,lift_reference_at12p620_m=float(lift[124]),first_finger_terminal_with1e12_tolerance_control_time_s=(2400+final_tick)*.005,maximum_actual_phi_offset_s=float(abs(phi-tau).max()),maximum_actual_lift_offset_m=float((lift-native[:,18]).max()))
  records[stage]=stages
 checks['new_origins_zero_references_identical']=np.array_equal(*zeros)
 checks['training_and_holdout_per_origin']=plan['planned']==14 and tuple(plan['training'])==TRAIN and set(plan['strict_holdout'])==set(VARIANTS)-set(TRAIN) and {x['relative_control_tick'] for x in plan['probe_origins']}==set(ORIGINS.values())
 # Exercise the real runner's job and failure-summary functions without its imports/plant.
 tree=ast.parse((PACKAGE/'run_pilot.py').read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('planned_jobs','summarize')]
 env=dict(Path=Path,ORIGINS=ORIGINS,VARIANTS=VARIANTS,TRAIN=TRAIN,identity=lambda:{'frozen':'unit'},ROOT=ROOT)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PACKAGE/'run_pilot.py'),'exec'),env)
 jobs=env['planned_jobs'](Path('pure_no_output'))
 checks['fourteen_unique_jobs_two_stages_independent_roles']=len(jobs)==14 and len({j['job_id'] for j in jobs})==14 and all(sum(j['stage']==s and j['training_role']=='train' for j in jobs)==3 and sum(j['stage']==s and j['training_role']=='heldout_mixed' for j in jobs)==4 for s in ORIGINS)
 failed=[dict(job_id=j['job_id'],stage=j['stage'],variant=j['variant'],status='failed',error='pure retained failure fixture') for j in jobs]
 summary=env['summarize'](Path('pure_no_output'),failed,{'frozen':'unit'})
 checks['all14_failed_retained_terminal_not_passed']=summary['execution_complete'] and summary['completed']==0 and len(summary['failures'])==14 and len(summary['pairs'])==14 and not summary['passed']
 for name,kwargs in [('unregistered020',dict(probe_origin_tick=4)),('boolean_origin',dict(probe_origin_tick=True)),('nonheld_clock',dict(native_tau=.1505))]:
  args=dict(native_tau=.150,relative_control_tick=30,variant='zero',release_control_tick=2400,release_time_s=12.,probe_origin_tick=30);args.update(kwargs)
  try:coordinates(decoder,**args);checks['reject_'+name]=False
  except ValueError:checks['reject_'+name]=True
 # Use frozen manifest paths to fingerprint all actual sources without importing the physics API.
 v2=ROOT/'output/release_local_response_capsule_v2';snap=json.loads((v2/'source_snapshot_manifest.json').read_text(encoding='utf-8'))['files'];lock=json.loads((ROOT/'identification/release_local_response_v2/dependency_lock.json').read_text(encoding='utf-8'))['sources']
 source_paths={k:Path(v['original_path']) for k,v in snap.items()};sources={k:sha(p) for k,p in source_paths.items()}
 checks['old237_originals_equal_frozen_lock']=len(sources)==237 and sources==lock
 for p in PACKAGE.iterdir():
  if p.is_file() and (p.suffix=='.py' or p.name in ('plan.json','README.md')):
   key='research/identification/release_local_response_v3/'+p.name;source_paths[key]=p;sources[key]=sha(p)
 checks['exact244_source_closure']=len(sources)==244
 checks['no_mujoco_plant_or_runner_import']='mujoco' not in sys.modules and 'hand_plant' not in sys.modules and 'identification.release_local_response_v3.run_pilot' not in sys.modules
 checks['no_force_model_or_certificate']=not plan['force_response_model_available'] and not plan['material_safety_certified']
 checks={k:bool(v) for k,v in checks.items()}
 if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
 return dict(schema='release_local_response_v3_pure_v1',passed=True,checks=checks,physics_steps=0,plant_constructed=False,old_native_applications_compared=2*8801,records=records,source_identity=sources,source_paths={k:str(p) for k,p in source_paths.items()},source_count=len(sources),frozen_v2_lock_sha256=sha(ROOT/'identification/release_local_response_v2/dependency_lock.json'),old_native_entry_decoder_source_sha256=sha(RUN/'sources/research/identification/release_basis_v2/decoder.py') if (RUN/'sources/research/identification/release_basis_v2/decoder.py').exists() else sha(ROOT/'identification/release_basis_v2/decoder.py'),scope='Pure native reference algebra and declared source identity only; no actual V3 physics, zero-step constructor, new prefix/entry pairing or force outcomes claimed.')

if __name__=='__main__':
 report=verify();out=PACKAGE/'qa'/('pure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.parent.mkdir(exist_ok=True)
 with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(dict(output=str(out),sha256=sha(out),checks=len(report['checks']),sources=report['source_count'],passed=True,physics_steps=0)))

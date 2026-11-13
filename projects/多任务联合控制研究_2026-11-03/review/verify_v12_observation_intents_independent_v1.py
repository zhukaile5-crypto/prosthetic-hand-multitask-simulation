"""Small independent, in-memory snapshots of V12 source; no plant imports."""
from pathlib import Path
from datetime import datetime,timezone
from types import ModuleType,SimpleNamespace
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
G=dict(base=2.,ratio_floor=1.8,ratio_kp=.8,ratio_ki=.2,slip_boost_rate=3.,baseline_tracking_gain=.25,release_rate=.8)
def packet(time=0.,valid=True,slip=.006):
 W=np.zeros((6,5));W[2]=1.
 return SimpleNamespace(time=time,sample_time=time,age=0.,normal=np.full(5,2.),tangential=np.zeros((5,2)),slip_local=np.zeros((5,2)),valid=np.full(5,valid,dtype=bool),shear_magnitude=np.full(5,.4),slip_magnitude=np.full(5,slip),alpha=np.full(5,.8),joint_velocity=np.zeros(5),phase='HOLDING',nominal_force_weights=np.ones(5),support_axis=np.array([0.,0.,1.]),contact_present=np.ones(5,bool),normal_wrench_map=W,finger_non_normal_by_finger=np.zeros((5,6)),distributed_valid=np.ones(5,bool),normal_support_coeff=np.ones(5),tangent_support_coeff=np.zeros(5),auxiliary_wrench=np.zeros(6),auxiliary_valid=False,auxiliary_pressure=0.,auxiliary_age=0.,auxiliary_uncertainty=.01,load_valid=False,sensor_characteristics=dict(force_sigma_N=.01))
def main(out):
 if out.exists():raise FileExistsError(out)
 contents={n:(ROOT/'v12'/n).read_bytes() for n in ['observation.py','intents.py']}
 pkg=ModuleType('peer_v12');pkg.__path__=[];sys.modules['peer_v12']=pkg
 mods={}
 for name,body in contents.items():
  key='peer_v12.'+name[:-3];m=ModuleType(key);m.__file__=str(ROOT/'v12'/name);m.__package__='peer_v12';sys.modules[key]=m;exec(compile(body.decode('utf-8-sig'),m.__file__,'exec'),m.__dict__);mods[name]=m
 State=mods['observation.py'].PublicTactileState;Fixed=mods['intents.py'].FixedIntent;Ratio=mods['intents.py'].RatioIntent
 checks=[]
 def check(name,value,details=None):checks.append(dict(name=name,passed=bool(value),details=details))
 s=State(G).update(packet(slip=0.),.005)
 f=Fixed(G).step(s,'RELEASING',np.full(5,5.),18.)
 check('native_RELEASING_phase_zero_fixed_intent',np.array_equal(f['desired_normal5'],np.zeros(5)),f['desired_normal5'].tolist())
 r=Ratio(G);z=r.step(s,'RELEASING',np.full(5,5.),18.)
 check('native_RELEASING_phase_zero_and_frozen_ratio_memory',np.array_equal(z['desired_normal5'],np.zeros(5)) and r.boost==0. and r.stable_since is None and len(r.samples)==0,dict(target=z['desired_normal5'].tolist(),boost=r.boost,stable_since=r.stable_since,samples=len(r.samples)))
 s=State(G);s.update(packet(0.),.005);first=s.mu_valid.copy();s.update(packet(.005,False),.005);missing=s.sliding_since.copy();s.update(packet(.205),.005)
 check('mu_requires_new_contiguous80ms_after_explicit_dropout',not s.mu_valid.any(),dict(first_mu_valid=first.tolist(),sliding_since_after_dropout=missing.tolist(),first_recovered_sliding_since=s.sliding_since.tolist(),mu_valid=s.mu_valid.tolist(),mu=s.mu.tolist(),valid_samples='only t0 and t.205, with explicit invalid update t.005'))
 # A finite measurement filter is not a current-force prediction.
 p=packet(1.,slip=0.);p.sample_time=.99;p.age=.01;s=State(G).update(p,.005)
 check('pressure_keeps_acquisition_age_and_unknown_current',np.allclose(s.normal_time,.99) and np.allclose(s.observation_age,.01) and not s.pressure_prediction_available and np.isnan(s.predicted_normal).all())
 # Distributed downward components must cancel upward finger components before
 # adding external auxiliary support; no object/target selector is supplied.
 p=packet(slip=0.);p.normal_wrench_map[2]=[-1.,.25,.25,.25,.25];p.auxiliary_valid=True;p.auxiliary_pressure=2.;p.auxiliary_wrench[2]=2.;s=State(G).update(p,.005)
 check('all_external_signed_support_net_not_sum_of_positive_parts',s.load_valid and abs(s.load-2.)<1e-12,dict(load=s.load,sources=s.load_sources))
 result=dict(schema='v12_observation_intents_independent_small_oracles_v1',created_utc=datetime.now(timezone.utc).isoformat(),role='source snapshots and independent minimal mathematical packets, not old-stream replay or plant validation',source_sha256={n:hashlib.sha256(body).hexdigest() for n,body in contents.items()},source_still_same={n:(ROOT/'v12'/n).read_bytes()==body for n,body in contents.items()},checks=checks,passed=all(c['passed'] for c in checks),physics_steps=0,plant_constructed=False,physics_api_imported='mujoco' in sys.modules,limitations=['Five narrow interface oracles only; author21/other evidence not rerun here.','Failed mu fixture isolates an explicit missing interval and first valid recovery, not a pressure response model or friction identity.'])
 with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),passed=result['passed'],failed=[c['name'] for c in checks if not c['passed']],sources=result['source_sha256'],source_still_same=result['source_still_same'],physics_steps=0)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();main(a.out)

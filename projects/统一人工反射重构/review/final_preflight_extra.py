"""Independent final audit of evidence contracts; reads development evidence only."""
from pathlib import Path
import sys,json,hashlib,copy
from dataclasses import fields
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physics import Plant,SensorPipeline
from scenarios import SCENARIOS,parameters
from evaluation import evaluate_trial
from run_experiments import hashes

def sensor_boundary():
    s=next(x for x in SCENARIOS if x.name=='nominal');p=parameters(s,410181)
    plant=Plant(s,p);truth=plant.truth();changed=copy.deepcopy(truth)
    for key in ('position','relative','rotation','relvel','relomega','relative_velocity','angular_velocity'):
        changed[key]=np.full_like(changed[key],987.)
    changed.update(object_mass=98.,actual_mu=.001,future_events=[{'onset':.1,'force':1000.}])
    alternate={**p,'mass':100.,'friction':.0001}
    a=SensorPipeline(s,p,410181).sample(truth,0.)
    b=SensorPipeline(s,alternate,410181).sample(changed,0.)
    unequal=[]
    for field in fields(a):
        x,y=getattr(a,field.name),getattr(b,field.name)
        numeric=np.asarray(x).dtype.kind in 'biufc'
        if not np.array_equal(np.asarray(x),np.asarray(y),equal_nan=numeric):unequal.append(field.name)
    return dict(passed=not unequal,unequal_fields=unequal,
        modified_hidden_fields=['position','relative','rotation','relvel','relomega','relative_velocity','angular_velocity','object_mass','actual_mu','future_events','parameter_mass','parameter_friction'],
        allowed_sources='unchanged local tactile skin plus hand encoders, effort and known hand calibration')

def formal_missing_record():
    t=np.arange(2001)*.005;n=len(t);force=np.tile([2.,2.,2.,0.,0.],(n,1))
    tr=dict(time=t,normal=force,skin_normal=force,position=np.tile([0.,0.,1.],(n,1)),relative=np.zeros((n,3)),
        rotation=np.tile(np.eye(3),(n,1,1)),relative_velocity=np.zeros((n,3)),angular_velocity=np.zeros((n,3)),
        support=np.zeros(n),palm=np.zeros(n),all_hand_object=force.sum(axis=1),motion_attained=np.ones(n,bool),
        substep_time=np.arange(2000)*.0005,substep_duration=np.full(2000,.0005),
        substep_normal=np.tile([2.,2.,2.,0.,0.],(2000,1)),substep_skin_normal=np.tile([2.,2.,2.,0.,0.],(2000,1)))
    meta=dict(scenario=dict(name='synthetic_final',duration=10.,limit=[5.]*5,budget=22.),
        reference=dict(relative=[0.,0.,0.],rotation=np.eye(3).tolist(),absolute_z=1.,orientation_mode='free'),
        command_at=0.,physics_timestep=.0005)
    cfg=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))['evaluation']
    r=evaluate_trial(tr,meta,cfg)
    return dict(task_success=r['task_success'],force_valid=r['force_safety']['valid'],
        force_pass=r['force_safety']['force_protection_pass'],full_substeps=r['force_safety']['full_substeps'],
        force_quality_valid=r['force_quality_contract']['valid'],
        passed=not any([r['task_success'],r['force_safety']['valid'],r['force_safety']['force_protection_pass'],
                       r['force_safety']['full_substeps'],r['force_quality_contract']['valid']]))

def smoke_evidence():
    folder=ROOT/'output/common_model_smoke_v1';manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((folder/'trials.json').read_text(encoding='utf-8'));checks=[]
    required=['substep_normal','substep_skin_normal','substep_shear','substep_palm','substep_support','substep_hand_other',
        'substep_all_hand_object','substep_collision','substep_penetration','substep_world_penetration','substep_duration']
    for row in rows:
        filename=folder/row['trace_file'];npz=np.load(filename);trace={k:npz[k] for k in npz.files if k!='metadata'}
        meta=json.loads(str(npz['metadata']));times=trace['time'];subt=trace['substep_time'];dur=trace['substep_duration']
        limits=np.asarray(meta['scenario']['limit']);total=trace['substep_skin_normal'].sum(axis=1)
        local_int=float(np.sum(np.maximum(trace['substep_skin_normal']-limits,0.)*dur[:,None]))
        total_int=float(np.sum(np.maximum(total-meta['scenario']['budget'],0.)*dur))
        peak=np.maximum(trace['substep_skin_normal'].max(axis=0),trace['skin_normal'][-1])
        duration=meta['scenario']['duration'];size=len(subt)
        checks.append(dict(key=filename.stem,
            trace_hash_matches=hashlib.sha256(filename.read_bytes()).hexdigest()==row['trace_sha256'],
            full_fields=all(k in trace for k in required),
            truth_clock=bool(len(times)==round(duration/.005)+1 and np.allclose(np.diff(times),.005,atol=1e-9,rtol=0)),
            interval_clock=bool(size==round(duration/.0005) and abs(subt[0])<1e-8 and
                abs(subt[-1]+dur[-1]-duration)<1e-8 and np.allclose(subt[:-1]+dur[:-1],subt[1:],atol=1e-8,rtol=0)),
            actual_force_shapes=all(trace[k].shape==(size,5) for k in ['substep_normal','substep_skin_normal','substep_shear']),
            force_peak_independent_match=bool(np.allclose(peak,row['force_safety']['finger_peak'],atol=1e-9,rtol=0)),
            force_exposure_independent_match=bool(abs(local_int-row['force_safety']['local_excess_integral'])<1e-9 and
                abs(total_int-row['force_safety']['total_excess_integral'])<1e-9),
            next_decision_timestamps=bool(np.allclose(trace['prediction_at'],(np.arange(len(times))//4+1)*.020,atol=1e-8,rtol=0)),
            evaluation_contract=bool(row['truth_record_complete'] and row['force_safety']['full_substeps'] and row['force_quality_contract']['valid']),
            model_identity=meta['benchmark_id']=='mujoco_finite_mass_five_compliant_contacts_v1' and not meta['model_metadata']['full_shadow_dynamics']))
    return dict(count=len(checks),source_matches_current=manifest['code_sha256']==hashes(),
        all_checks_pass=all(all(v for k,v in row.items() if k!='key') for row in checks),checks=checks)

def tuning_budget():
    plan=json.loads((ROOT/'controller_candidates.json').read_text(encoding='utf-8'));manifests=[]
    for j in range(3):
        path=ROOT/f'output/candidates_v1_{j}/manifest.json';m=json.loads(path.read_text(encoding='utf-8'));manifests.append(m)
    expected=30
    return dict(candidates_per_primary={k:len(v) for k,v in plan['candidates'].items()},
        cases_per_candidate=len(plan['development_seeds'])*len(plan['development_scenarios']),
        expected_total=360,
        same_matrix=all(m['seeds']==plan['development_seeds'] and m['scenarios']==plan['development_scenarios'] and
            m['methods']==['fixed','ratio','independent','unified'] and m['expected_trials']==4*expected and
            m['code_sha256']==hashes() for m in manifests),
        selection_rule=plan['selection_rule'],scope='checks planned tuning inputs only, does not select or inspect results')

if __name__=='__main__':
    before=hashes();r=dict(sensor_boundary=sensor_boundary(),formal_missing_record=formal_missing_record(),
        smoke_evidence=smoke_evidence(),tuning_budget=tuning_budget())
    r['source_sha256']=before;r['source_unchanged']=before==hashes()
    r['passed']=(r['source_unchanged'] and r['sensor_boundary']['passed'] and r['formal_missing_record']['passed'] and
        r['smoke_evidence']['all_checks_pass'] and r['smoke_evidence']['source_matches_current'] and r['tuning_budget']['same_matrix'])
    (ROOT/'review/final_preflight_extra.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],source_unchanged=r['source_unchanged'],sensor_boundary=r['sensor_boundary']['passed'],
        formal_missing_record=r['formal_missing_record'],smoke_count=r['smoke_evidence']['count'],
        smoke_all_checks_pass=r['smoke_evidence']['all_checks_pass'],same_tuning_matrix=r['tuning_budget']['same_matrix']),ensure_ascii=False))

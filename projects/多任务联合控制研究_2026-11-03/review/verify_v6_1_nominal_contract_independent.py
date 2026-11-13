"""Independent actual zero-step factory and pure nominal-statistic checks."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,sys,importlib,difflib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
EXPECTED={'coordinator.py':'9efec0b1f824c3da129b40b6cb75aaa4bfac2ca28a79facfce74b39f9141071f','coordinator_model.py':'2f3c3de3c081042116f026246e5e72e05bde77c660b8303d6701ef156e1a4bd6','coordinator_contacts.py':'b6398c2a0d8169065dca74488e6e8e192aafcd3990a93aea7b83fc0b9cb3153b'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def main():
    checks={};step_calls=[];source_before={k:sha(ROOT/k) for k in [*EXPECTED,'runtime.py','task_contract.py']}
    def check(k,v):checks[k]=bool(v)
    check('final_core_identity',all(source_before[k]==v for k,v in EXPECTED.items()))
    import mujoco
    originals={k:getattr(mujoco,k) for k in ['mj_step','mj_step1','mj_step2']}
    def forbidden(*a,**k):step_calls.append(True);raise RuntimeError('Independent zero-step audit forbids physical integration')
    for k in originals:setattr(mujoco,k,forbidden)
    try:
        core=importlib.import_module('coordinator');runtime=importlib.import_module('runtime');contacts=importlib.import_module('coordinator_contacts');sensors=importlib.import_module('sensors');policy=importlib.import_module('policy_sources')
        captured=[];factory=core.create_joint
        def capture(calibration,limits,budget,gains=None):
            captured.append(dict(gains=gains.copy() if isinstance(gains,dict) else gains,limits=np.asarray(limits).copy(),budget=budget));return factory(calibration,limits,budget,gains=gains)
        core.create_joint=capture
        ratio=runtime.ResearchSession(method='ratio',scenario='nominal',seed=610212,split='robustness')
        joint=runtime.ResearchSession(method='joint',scenario='nominal',seed=610212,split='robustness')
        core.create_joint=factory;fields=core.NOMINAL_GAIN_FIELDS;values={k:float(getattr(ratio.controller.g,k)) for k in fields};contract=joint.controller.nominal_feedback_contract
        check('runtime_passes_whole_actual_ratio_configuration',len(captured)==1 and captured[0]['gains']==ratio.gains and sorted(captured[0]['gains'])==contract['supplied_keys'])
        check('all_resolved_gains_exact_actual_baseline',asdict(joint.controller.g)==asdict(ratio.controller.g))
        check('seven_explicit_fields_are_resolved_not_hand_selected',contract['values']==values and contract['production_explicit'] and {k:float(getattr(joint.controller.g,k)) for k in fields}==values)
        check('zero_public_physical_clock_and_no_substeps',all(s.time==0. and s.plant.data.time==0. and not s.plant.substep_records for s in [ratio,joint]) and not step_calls)
        check('same_actual_physical_initialization',asdict(ratio.config)==asdict(joint.config) and np.array_equal(ratio.plant.data.qpos,joint.plant.data.qpos) and np.array_equal(ratio.plant.data.qvel,joint.plant.data.qvel))
        check('nominal_contract_hash',contract['sha256']==hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        rejected=[]
        bads={'missing':None,'base_below_floor':{**ratio.gains,'base':ratio.gains['ratio_floor']-.01},'nan':{**ratio.gains,'ratio_kp':float('nan')},'release_mismatch':{**ratio.gains,'release_rate':.123}}
        for key in fields:bads['missing_'+key]={k:v for k,v in ratio.gains.items() if k!=key}
        for label,g in bads.items():
            try:factory(joint.calibration,joint.limits,joint.budget,gains=g);rejected.append((label,False))
            except ValueError:rejected.append((label,True))
        check('all_missing_invalid_or_inconsistent_factory_contracts_rejected',all(ok for _,ok in rejected))
        weights=np.asarray(joint.calibration['nominal_force_weights']);membership=SimpleNamespace(mode=np.full(5,'STICK_CONFIRMED'));mask=np.ones(5,bool)
        state=SimpleNamespace(time=7.5,normal=2.*weights,delayed_normal=2.*weights,tangential=np.zeros((5,2)),shear_magnitude=.4*(2.*weights),valid=mask,contact_model_active=mask,contact=mask,slip_speed=0.,closure_permission=mask,alpha=np.zeros(5),model_reference_alpha=np.zeros(5),nominal_weights=weights,force_model=SimpleNamespace(inverse=lambda active:np.eye(5)))
        def run_transition(g):
            task=contacts.ForceQualityTask(g);targets=[];ratios=[];times=[]
            for i in range(81):
                state.time=7.5+i*.02;_,target,debug=task.propose(state,membership,mask,np.full(5,5.),18.,noise_sigma=np.zeros(5));task.reconcile(None)
                targets.append(target.copy());ratios.append(task.target_ratio);times.append(state.time)
            return task,np.asarray(targets),ratios,times
        task,targets,ratios,times=run_transition(joint.controller.g)
        check('constant_public_observation_crosses_learning_boundary',any(x is None for x in ratios) and any(x is not None for x in ratios))
        check('one_second_transition_no_force_target_jump',np.max(abs(np.diff(targets,axis=0)))<1e-12 and np.allclose(targets,2.*weights,atol=1e-12,rtol=0))
        from vendor.frozen_controllers import make_gains
        legacy=make_gains();state.normal=legacy.base*weights;state.delayed_normal=state.normal.copy();state.shear_magnitude=.4*state.normal
        old_task,old_targets,old_ratios,_=run_transition(legacy);old_jump=float(np.max(abs(np.diff(old_targets,axis=0))))
        check('generic_default_floor_jump_counterexample_exposed',legacy.base<legacy.ratio_floor and old_jump>.1)
        patches=[]
        for finger in range(5):
            for sign in [-1.,1.]:patches.append(dict(digit=finger,normal=1.,force=[sign,0.,1.],normal_dir=[0.,0.,1.],position=[.01*sign,0.,0.],torque=[0.,0.,0.],slip_velocity=[0.,0.,0.]))
        measured,diag=sensors.aggregate_patches(patches);state.normal=np.asarray(measured['normal']);state.delayed_normal=state.normal.copy();state.tangential=np.asarray(measured['tangential']);state.shear_magnitude=np.asarray(diag['shear_magnitude']);state.time=9.
        check('opposed_patch_tractions_cancel_vector_not_shear',np.allclose(state.tangential,0.) and np.allclose(state.shear_magnitude,2.))
        ours=contacts.ForceQualityTask(joint.controller.g);_,_,debug=ours.propose(state,membership,mask,np.full(5,5.),18.)
        baseline=policy.MigratedRatio('ratio',np.full(5,5.),18.,ratio.gains);baseline.state=state;baseline._ratio_target('HOLDING')
        check('new_ratio_statistic_matches_actual_migrated_baseline',debug['measured_ratio']==baseline.debug['ratio']==1.)
        check('public_statistic_definition_is_explicit','distributed' in debug['ratio_measurement_definition'])
        baseline_file=ROOT.parent/'五指策略接入_2026-11-03/vendor/frozen_settings.json';frozen=read(baseline_file);settings=frozen.get('methods',frozen.get('gains',{})).get('ratio',frozen.get('controllers',{}).get('ratio',{}))
        check('gains_match_frozen_settings_and_shared_release',all(ratio.gains[k]==v for k,v in settings.items()) and ratio.gains['release_rate']==ratio.config.release_open_rate)
        prior=ROOT/'output/development_v6_known_counterexamples_v1';check('prior_failed_gate_and_source_snapshots_retained',(prior/'trials.json').is_file() and (prior/'sources/coordinator.py').is_file())
        check('no_production_source_changed_by_audit',source_before=={k:sha(ROOT/k) for k in source_before})
        out=dict(schema='independent_v6_1_nominal_feedback_contract_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_before,reviewer_sha256=sha(__file__),actual_runtime_contract=contract,resolved_baseline_values=values,rejected=rejected,initialization=dict(times=times,targets=targets,max_target_step_N=float(np.max(abs(np.diff(targets,axis=0)))),target_ratio=ratios,generic_unverified_default_jump_N=old_jump),cancellation=dict(patch_count=len(patches),aggregate_tangent_norm_N=float(np.linalg.norm(state.tangential,axis=1).sum()),distributed_shear_N=float(state.shear_magnitude.sum()),ratio=debug['measured_ratio'],baseline_ratio=baseline.debug['ratio']),physics_steps_executed=len(step_calls),qp_calls=0,prior_trials_sha256=sha(prior/'trials.json'),limitations=['Zero-step configuration and constant-input/statistic contracts only; no closed-loop task, safety or quality claim.','Parameter/statistic corrections resolve a generic-default inconsistency and align a public channel with the already migrated baseline; no scene/seed condition or performance threshold is selected here.','Resolved parameter equality does not mean joint controller actions or algorithms are identical to ratio. Both seen12 gates must retain their independent source/result identities.'])
    finally:
        for k,v in originals.items():setattr(mujoco,k,v)
    path=ROOT/'review'/('independent_v6_1_nominal_contract_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');path.write_text(json.dumps(serial(out),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=out['failed'],path=str(path),sha256=sha(path),values=values,max_target_step_N=out['initialization']['max_target_step_N'],old_default_jump_N=old_jump,physics_steps_executed=len(step_calls)),ensure_ascii=False))
if __name__=='__main__':main()

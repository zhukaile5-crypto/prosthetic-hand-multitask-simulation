"""Production nominal-contract checks. Construction only, no physical step."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from coordinator import create_joint,self_test,NOMINAL_GAIN_FIELDS
from coordinator_contacts import ForceQualityTask
from runtime import ResearchSession
from policy_sources import MigratedRatio


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    names=('coordinator.py','coordinator_model.py','coordinator_contacts.py','runtime.py','research_imports.py')
    sources={name:sha(ROOT/name) for name in names};checks={}
    synthetic=self_test();checks['author_44_algebra_checks']=bool(synthetic['passed'] and len(synthetic['checks'])==44)
    ratio=ResearchSession(method='ratio',scenario='nominal',seed=610212,split='robustness')
    joint=ResearchSession(method='joint',scenario='nominal',seed=610212,split='robustness')
    supplied={key:float(ratio.gains[key]) for key in NOMINAL_GAIN_FIELDS}
    contract=joint.controller.nominal_feedback_contract
    checks['runtime_forwards_entire_actual_nominal_contract']=contract['values']==supplied and contract['supplied_keys']==sorted(ratio.gains)
    checks['all_resolved_gains_equal_actual_admitted_ratio']=asdict(joint.controller.g)==asdict(ratio.controller.g)
    checks['physical_configs_equal']=asdict(ratio.config)==asdict(joint.config)
    checks['construction_never_advanced_public_or_physical_time']=all(s.time==0. and s.plant.data.time==0. and not s.plant.substep_records for s in (ratio,joint))
    checks['production_contract_is_explicit']=bool(contract['production_explicit'])
    cal=joint.calibration;limits=joint.limits;budget=joint.budget
    rejected={}
    for label,values in (('missing',None),('partial',{k:v for k,v in ratio.gains.items() if k!='base'}),
        ('contradictory_initialization',{**ratio.gains,'base':ratio.gains['ratio_floor']-.1})):
        try:create_joint(cal,limits,budget,gains=values)
        except ValueError as exc:rejected[label]=str(exc)
        else:raise AssertionError(f'Incomplete production configuration accepted: {label}')
    checks['no_silent_defaults_or_invalid_initial_scalar']=len(rejected)==3
    # A constant public input crosses the one-second ratio-learning transition.
    # This is a scalar/statistical interface check, not simulated feedback.
    p=np.asarray(joint.config.holding_force,float);shear=.6*p
    state=SimpleNamespace(time=7.5,normal=p.copy(),delayed_normal=p.copy(),shear_magnitude=shear.copy(),
        tangential=np.column_stack((shear,np.zeros(5))),valid=np.ones(5,bool),contact_model_active=np.ones(5,bool),
        closure_permission=np.ones(5,bool),alpha=np.asarray(cal['reference_alpha']),model_reference_alpha=np.asarray(cal['reference_alpha']),
        nominal_weights=np.asarray(cal['nominal_force_weights']),contact=np.ones(5,bool),slip_speed=0.,
        force_model=joint.controller.state.force_model if joint.controller.state.force_matrix is not None else None)
    from force_model import ForceResponse
    state.force_model=ForceResponse(np.asarray(cal['force_matrix']),float(cal['force_tau']))
    membership=SimpleNamespace(mode=np.full(5,'STICK_CONFIRMED'))
    task=ForceQualityTask(joint.controller.g);targets=[];records=[]
    for tick in range(61):
        state.time=7.5+tick*joint.controller.g.decision_period
        action,target,debug=task.propose(state,membership,np.ones(5,bool),limits,budget)
        # Do not claim nominal requests were physically executed in this check.
        task.reconcile(None);targets.append(target.copy())
        records.append(dict(time_s=state.time,ratio_target=task.target_ratio,scalar=debug['ratio_scalar'],target=target.tolist()))
    targets=np.asarray(targets)
    checks['one_second_learning_was_reached']=task.target_ratio is not None
    checks['constant_observation_initialization_has_no_target_step']=bool(np.max(np.abs(np.diff(targets,axis=0)))<1e-12)
    # Opposed physical patch tractions can have zero aggregate tangent vector.
    state.tangential[:]=0.;state.shear_magnitude[:]=2.;state.delayed_normal[:]=2.;state.normal[:]=2.
    left=ForceQualityTask(joint.controller.g)
    _,_,ours=left.propose(state,membership,np.ones(5,bool),limits,budget)
    baseline=MigratedRatio('ratio',limits,budget,ratio.gains);baseline.state=state
    baseline._ratio_target('HOLDING')
    checks['distributed_shear_statistic_matches_actual_migrated_ratio']=bool(np.isclose(ours['measured_ratio'],1.) and np.isclose(ours['measured_ratio'],baseline.debug['ratio']))
    checks['opposed_vector_sum_does_not_erase_shear']=bool(np.linalg.norm(state.tangential,axis=1).sum()==0. and ours['measured_ratio']>0.)
    checks['source_identity_unchanged']=sources=={name:sha(ROOT/name) for name in names}
    if not all(checks.values()):raise AssertionError(checks)
    return dict(schema='v6_1_explicit_nominal_feedback_contract_validation_v1',passed=True,checks=checks,
        source_sha256=sources,validator_sha256=sha(Path(__file__)),contract=contract,
        actual_ratio_gains=ratio.gains,all_resolved_gains_match=True,rejected_missing_or_invalid_contracts=rejected,
        constant_input_transition=records,maximum_target_step_N=float(np.max(abs(np.diff(targets,axis=0)))),
        distributed_shear_counterexample=dict(aggregate_vector_norm_sum=0.,distributed_shear_sum_N=10.,ratio=ours['measured_ratio'],baseline_ratio=baseline.debug['ratio']),
        no_physical_steps=True,no_performance_claim=True,old_gate_retained='output/development_v6_known_counterexamples_v1')


if __name__=='__main__':
    output=ROOT/'review/nominal_feedback_contract_v6_1_validation_v1.json'
    if output.exists():raise FileExistsError('Existing contract evidence must be retained')
    report=run()
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),sources=report['source_sha256'],
        nominal_contract=report['contract'],maximum_target_step_N=report['maximum_target_step_N'],artifact=str(output)),ensure_ascii=False,indent=2))

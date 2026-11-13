"""Independent v2 mathematical regression review; no simulator construction."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from coordinator import self_test


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={n:sha(ROOT/n) for n in ('coordinator.py','coordinator_model.py','runtime.py','evaluation.py','protocol.json','research_imports.py','run_development.py')}
author=self_test()
# Execute the exact earlier independent QP fixture up to (but not including)
# its file-output statements. This reuses its mathematical state, not a hand.
fixture=ROOT/'review/independent_joint_budget_counterexample.py'
tree=ast.parse(fixture.read_text(encoding='utf-8'));nodes=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in node.targets):break
    nodes.append(node)
ns={'__file__':str(fixture),'__name__':'independent_v2_fixture'}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fixture),'exec'),ns)
r=ns['result'];joint=ns['joint'];state=ns['state']
P,c=state.force_model.action_prediction(ns['normal'],joint.g.horizon,joint.g.decision_period,
    state.contact_model_active,alpha=state.alpha,latent=state.latent_alpha,execution_dt=.005)
raw_all=P@joint.previous_plan[:5*joint.g.horizon]+c
checks=dict(expected_v2_1_source=before['coordinator.py']=='e194d7a4a7efd2512f654f099daeea463cc8d0a41a86da6d19683030ff05720f' and before['coordinator_model.py']=='acdfdcb618f82742454e8d6b06701938c130513b29bc3d961bdd914ca92f120a' and before['runtime.py']=='da96128b7ba3f724301e8dc00812232acc137c659296ed29f258f7d231e31e4c',
    all_author_math_checks=author['passed'],all_horizon_unilateral_pressure=bool(raw_all.min()>=-1e-8),
    displayed_prediction_matches_raw=bool(np.allclose(r['raw_affine_pressure'],r['displayed_clipped_pressure'],atol=1e-8,rtol=0)),
    negative_pressure_budget_credit_removed=abs(r['raw_sum']-r['clipped_sum'])<1e-8,
    first_horizon_budget_violation_not_understated=r['reported_budget_violation']+1e-7>=r['clipped_sum']+.35-r['budget'],
    model_invalid_is_explicitly_noncertified=author['checks']['unreachable_nonnegative_model_not_hidden_by_output_clip'],
    full_public_error_retained=author['checks']['full_public_error_not_truncated'],
    all_invalid_clears_stale_forecast=author['checks']['all_invalid_clears_previous_prediction'],
    success_marks_available=author['checks']['normal_solve_marks_prediction_available'],
    shared_source_identity_helper_frozen=before['research_imports.py']=='8338138430cbf63f1455fd2633289e5a6aa23ea79fb34445d9d465274858fec8' and before['run_development.py']=='86052f4a6c9a825461f246a0f1cda7cdf3852f6a78534a206d44fef93dcdad55',
    frozen_evaluation_and_protocol_unchanged=before['evaluation.py']=='6000ddf6754662c322cb5c290f23364fa58ef5b444e206ae34783eb955a9e891' and before['protocol.json']=='a03294f7ef35d173c43386631d2661db85b52a3c6529fe5f2c9fe1ac8c78c2a7')
for name,h in before.items():checks['source_unchanged_during_review/'+name]=sha(ROOT/name)==h
old_fixture=(ROOT/'output/development_v1/sources/coordinator.py').read_text(encoding='utf-8')
checks['withdraw_stale_prefreeze_fixture_finding']='SimpleNamespace(normal=' in old_fixture
runtime=(ROOT/'runtime.py').read_text(encoding='utf-8')
checks['runtime_invalid_availability_and_timestamp_gate']=all(word in runtime for word in ('prediction_available=prediction_available','model_invalid=model_invalid','observation_unavailable=observation_unavailable','if prediction_available else np.nan'))
checks['runtime_separate_predictor_coverage']=all(word in runtime for word in ('prediction_coverage','model_invalid_ticks','observation_unavailable_ticks'))
result=dict(revision='v2.1 mathematics and forecast-availability interface',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
    created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=before,author_math_checks=author,
    v2_same_independent_qp_fixture=r,raw_all_horizon_pressure_min=float(raw_all.min()),
    correctness_conclusion='Public error truncation and negative-pressure cancellation are resolved for the reviewed v2 equations and synthetic fixture; no physical safety or performance acceptance is implied.',
    invalid_model_logging=['Runtime accepts NaN forecast while retaining finite complete physical truth; this is correct separation, not a physics failure.',
        'v2.1 runtime timestamps a hold forecast only when prediction_available, and separately logs model-invalid/observation-unavailable coverage.',
        'v2.1 all-invalid sensor branch clears stale force prediction to NaN with an explicit unavailable reason.',
        'Zero-progress fallback may still show transport_paused=False; actual requested/executed progress and unavailable-forecast coverage must remain alongside that Boolean.'],
    remaining_model_limits=['motion_response_identified remains false; force-motion coupling is via engineering inequality envelope, not a measured motion Jacobian.',
        'drift_error_rate is still capped at5N/s by model.update, so it remains an engineering proxy rather than a guaranteed upper bound.',
        'Finite-pass two-stage quadratic slack control is not a proof of a global lexicographic physical safety optimum.',
        'The initial pre-freeze self_test-normal finding is withdrawn for the actual frozen v1/v2 source; both contain the normal field.'],
    scope='Pure algebra, QP and source-interface review only; no new physics, controller performance run, or production source changes.')
out=ROOT/('review/independent_coordinator_v2_1_math_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out)),ensure_ascii=False))

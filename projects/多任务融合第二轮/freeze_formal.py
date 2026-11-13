"""Record the second-round evaluation plan before any fresh evaluation runs."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from run_experiments import code_hashes, SECOND_ROUND_METHODS
from physics import ROOT
from analyze_results import CONFLICT_SCENARIOS, BASELINES, BOOTSTRAP_REPS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    destination=ROOT/'review/formal_plan.json'
    if destination.exists():
        raise RuntimeError('A formal plan is already recorded; preserve it.')
    for directory in ('test_holdout_v2','robustness_holdout_v2'):
        if (ROOT/'output'/directory).exists():
            raise RuntimeError('Fresh evaluation output exists before plan freeze.')
    gains=json.loads((ROOT/'selected_gains.json').read_text(encoding='utf-8'))
    assert set(SECOND_ROUND_METHODS)<=set(gains)
    assert gains['coordinated']==gains['coordinated_no_guard']==gains['coordinated_no_redistribution']
    for key,value in gains['ratio'].items():
        assert gains['coordinated'][key]==value,(key,value,gains['coordinated'][key])
    validation=json.loads((ROOT/'output/development_selected_v3_final/audit.json').read_text(encoding='utf-8'))
    assert validation['passed']
    files=('selected_gains.json','analyze_results.py','实验协议.md','validate.py',
           'validate_coordination.py','calibrate.py','render_comparison.py','plot_round2.py',
           'review/baseline_kernel_equivalence.json','review/physics_model_v2.json')
    plan=dict(created=datetime.now(timezone.utc).isoformat(),code_sha256=code_hashes(),
              supporting_sha256={name:sha(ROOT/name) for name in files},gains=gains,
              methods=list(SECOND_ROUND_METHODS),test_seeds=list(range(8001,8021)),
              robustness_seeds=list(range(9001,9011)),physics_timestep=.0005,
              primary_domain='15 nonboundary scenarios, including combined disturbance failures',
              primary_metric='safe_completed: uninterrupted 30ms force/budget proxy across acquisition and hold',
              strict_secondary='peak_limit_completed: every solver sample/endpoint within force/budget limits',
              primary_comparators=list(BASELINES),primary_test='two-sided whole-seed label exchange, Holm two comparisons',
              bootstrap_reps=BOOTSTRAP_REPS,confidence='95% seed-cluster percentile; marginal intervals',
              exploratory_conflict_scenarios=list(CONFLICT_SCENARIOS),
              replay=dict(seed=8001,scenarios=['friction_drop','local_limit'],methods=['ratio','coordinated']),
              tuning='Initial tuning: 648 baseline + 432 coordination; final clock: 1080 baseline on 3 seeds + 600 coordination on 5 seeds, including 360 reused evaluations. 2760 unique parameter trials; no heldout tuning.',
              reused_first_round_evaluation='diagnostics only; excluded from fresh evaluation',
              physical_model='common simplified fingertip collision proxies; no hardware calibration',
              no_post_test_tuning=True)
    destination.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(frozen=str(destination),methods=plan['methods']),ensure_ascii=False))

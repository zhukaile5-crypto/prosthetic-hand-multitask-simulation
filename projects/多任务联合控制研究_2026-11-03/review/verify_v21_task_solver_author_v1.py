"""Finite V21 task-author analytic/counterexample checks. No robot/physics."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime, timezone
import argparse, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from v21_stable_handoff import task_solver as task


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v, np.ndarray): return plain(v.tolist())
    if isinstance(v, np.generic): return plain(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, dict): return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [plain(x) for x in v]
    return v
def base(P=1, movable=(0,)):
    B = np.zeros((25, 21)); B[:21] = np.eye(21)
    J = np.zeros((P, 25)); J[:, 0] = 1.
    lo = np.zeros(21); hi = np.zeros(21)
    for i in movable: lo[i] = -.005; hi[i] = .005
    return dict(time_s=12.5, dt_s=.005, current_reference21=np.zeros(21), encoder_reference21=np.zeros(21),
        step_lower21=lo, step_upper21=hi, reference_rates21=np.ones(21),
        normal_jacobian_Px25=J, proposal_map25x21=B, robot_velocity25=np.zeros(25),
        pressure_P_N=np.ones(P), reference_target_delta21=np.zeros(21))


def main(out):
    if out.exists(): raise FileExistsError('Fresh evidence required')
    sources = ['v21_stable_handoff/task_solver.py', 'v21_stable_handoff/prefix_hierarchy.py', 'v21_stable_handoff/design.py']
    before = {p: sha(ROOT/p) for p in sources}; checks = []; cases = {}
    def ck(name, condition, details=None): checks.append(dict(name=name, passed=bool(condition), details=details))
    def solve(name, kwargs):
        r = task.solve_contact_exit(**kwargs)
        cases[name] = dict(status=r.status, available=r.available, fully_converged=r.fully_converged,
            completed_prefix=r.completed_prefix, x=plain(r.x), failed_layer=plain(r.failed_layer), diagnostics=plain(r.diagnostics))
        return r
    k = base(); k['reference_target_delta21'][0] = .005
    r = solve('outward_after_zero_slack', k)
    ck('L1 zero-slack does not freeze Ndu and L2 exits against OPEN', r.available and abs(r.x[0]+1) < 1e-8 and abs(r.diagnostics['certified_slack_P_m'][0]) < 1e-10)
    k = base(); k['robot_velocity25'][0] = .2; k['current_reference21'][0] = .001
    r = solve('inward_velocity_and_debt', k)
    ck('inward-only correction target excludes capacity term', r.available and np.allclose(r.diagnostics['normal_correction_target_P_m'], [-.00125], atol=1e-15, rtol=0))
    k = base(); k['robot_velocity25'][0] = -.2; k['current_reference21'][0] = -.001
    r = solve('outward_not_credited', k)
    ck('outward velocity and debt do not replace new withdrawal', r.available and np.array_equal(r.diagnostics['normal_correction_target_P_m'], [0.]) and r.x[0] < -.999999)
    k = base(); k['step_lower21'][0] = .001; k['step_upper21'][0] = .003
    r = solve('actual_box_excludes_zero', k)
    ck('nonzero anchor retains unavoidable positive slack and original domain', r.available and abs(r.x[0]-.2) < 1e-8 and abs(r.diagnostics['certified_slack_P_m'][0]-.001) < 1e-10 and r.diagnostics['current_reference_outside_task_box'])
    k = base(movable=(0, 1)); M = np.zeros((1, 21)); M[0, 1] = 1.
    k.update(motion_matrix_Mx21=M, motion_error_M_m_s=np.zeros(1), motion_time_s=12.5)
    k['reference_target_delta21'][1] = .005
    r = solve('orthogonal_motion_budget', k)
    ck('witnessed zero-motion budget permits orthogonal normal exit', r.available and abs(r.x[0]+1) < 1e-8 and abs(r.x[1]) < 1e-8)
    k = base(); M = np.zeros((1, 21)); M[0, 0] = 1.
    k.update(motion_matrix_Mx21=M, motion_error_M_m_s=np.array([.1]), motion_time_s=12.5)
    r = solve('signed_motion_absolute_budget', k)
    d = r.diagnostics; witness = d.get('motion_task', {}).get('witness_normalized_residual_M')
    final_residual = (M@(.005*r.x)+.005*np.array([.1]))/.005
    ck('motion budget uses certified L1 witness and permits residual sign change', r.available and np.max(abs(final_residual)-abs(witness)) < 1e-8 and abs(r.x[0]+.2) < 1e-7)
    k = base(); M = np.zeros((1, 21)); k.update(motion_matrix_Mx21=M, motion_error_M_m_s=np.array([.2]), motion_time_s=12.5)
    r = solve('known_constant_motion_row', k)
    ck('constant motion row remains observed nonzero constant, not missing', r.available and r.diagnostics['motion_task']['zero_local_authority_rows'] == [0] and r.diagnostics['motion_task']['error_M_m_s'][0] == .2)
    k1 = base(); r1 = solve('original_one_patch', k1)
    k2 = base(P=2); k2['pressure_P_N'][:] = .5; r2 = solve('split_two_patches', k2)
    ck('physical identical patch split preserves selected exit', r1.available and r2.available and np.allclose(r1.x, r2.x, atol=1e-8, rtol=0))
    k = base(P=2); k['normal_jacobian_Px25'][1, 0] = -1.
    r = solve('opposing_normals', k)
    ck('opposing normals keep one-sided residuals and honest blocked progress', r.available and abs(r.x[0]) < 1e-8 and np.array_equal(r.diagnostics['normal_matrix_N_Px21'][:, 0], [1., -1.]))
    k = base(); k['normal_jacobian_Px25'][:] = 0.; k['normal_jacobian_Px25'][0, 24] = 1.; k['robot_velocity25'][24] = .3
    r = solve('zero_reference_authority_with_robot_velocity', k)
    ck('uncontrolled25 velocity produces unavoidable slack without fake control', r.available and r.diagnostics['zero_local_normal_authority_rows'] == [0] and abs(r.diagnostics['certified_slack_P_m'][0]-.0015) < 1e-10)
    k = base(P=0); k['reference_target_delta21'][0] = .002
    r = solve('knownP0', k)
    ck('knownP0 omits normal slack but retains actual posture task', r.available and abs(r.x[0]-.4) < 1e-8 and len(r.diagnostics['certified_slack_P_m']) == 0)
    k = base(P=0); k['reference_target_delta21'][0] = .004; path = np.zeros(21); path[0] = -.003
    k['path_reference_target_delta21'] = path; r = solve('explicit_path_last_layer', k)
    ck('path changes only explicit last target and preserves original canonical', r.available and abs(r.x[0]+.6) < 1e-8 and r.diagnostics['original_canonical_target_delta21'][0] == .004 and r.diagnostics['effective_posture_target_delta21'][0] == -.003)
    k = base(); k['normal_jacobian_Px25'] = None; r = solve('unknown_normal', k)
    ck('unknown normal is unavailable not knownP0', not r.available and r.status == 'UNKNOWN_CONTACT_EXIT_INPUT' and np.isnan(r.x).all())
    k = base(); k['motion_error_M_m_s'] = np.zeros(1); r = solve('partial_motion_fields', k)
    ck('partial motion tuple is unknown rather than fabricated zero matrix', not r.available and r.status == 'UNKNOWN_CONTACT_EXIT_INPUT')
    original = task.solve_hierarchy; call = [0]
    def later_fail(**kwargs):
        call[0] += 1
        if call[0] == 1: return original(**kwargs)
        return SimpleNamespace(x=np.asarray(kwargs['initial']).copy(), status='FORCED_LATER_FAILURE', available=False,
            certified_prefix=(), fully_converged=False, failed_layer=dict(index=0, name=kwargs['levels'][0].name, status='FORCED_LATER_FAILURE'),
            levels=(dict(name=kwargs['levels'][0].name, status='FORCED_LATER_FAILURE'),), diagnostics=dict(author_injection=True))
    with patch.object(task, 'solve_hierarchy', later_fail): r = solve('forced_later_failure', base())
    ck('later failure retains certified L1 action and is not full convergence', r.available and r.completed_prefix == 1 and not r.fully_converged and r.status == 'CERTIFIED_CONTACT_EXIT_PREFIX' and r.failed_layer['index'] == 1)
    ck('no pressure/actual-motion/placement certificate', all(not r.diagnostics[k] for k in ('pressure_prediction_available', 'future_robot_response_predicted', 'physical_safety_certified', 'placement_certified')))
    after = {p: sha(ROOT/p) for p in sources}; ck('bound source unchanged during pure author checks', before == after)
    report = dict(schema='v21_task_solver_author_v1', passed=all(c['passed'] for c in checks), created_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks, issues=[c['name'] for c in checks if not c['passed']], cases=cases, source_sha256=after,
        scope='Finite analytic tasks and one explicit post-L1 failure injection; zero robot models, simulation, contact, writer or device',
        physics_steps=0, robot_models_constructed=0, source_unchanged=before == after)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=len(checks), issues=report['issues'], path=str(out), sha256=sha(out)), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args(); raise SystemExit(main(args.out))

"""Synthetic interface/constraint checks, never formal trial tuning."""
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace
import json
import time
import numpy as np
from controllers import METHODS, Gains, make_controller, solve_qp, project_force

ROOT = Path(__file__).resolve().parent


def frame(t=1., normal=None, slip=0., valid=None, alpha=.55):
    n = np.array([2., 1., 1., 1., 1.]) if normal is None else np.asarray(normal, float)
    direction = np.array([[1., 0., 0.]]+[[-1., 0., 0.]]*4)
    pos = np.array([[-.025, 0., 0.], [.025, -.024, 0.], [.025, -.008, 0.],
                    [.025, .008, 0.], [.025, .024, 0.]])
    basis = np.tile([[0., 0., 1.], [0., 1., 0.]], (5, 1, 1))
    return SimpleNamespace(time=t, dt=.005, normal=n, tangential=np.tile([.20, 0.], (5, 1)),
                           contact_pos=pos, contact_normal=direction, tangent_basis=basis,
                           slip_local=np.tile([slip, 0.], (5, 1)),
                           joint=np.full(5, alpha), joint_velocity=np.zeros(5),
                           alpha=np.full(5, alpha), valid=np.ones(5, bool) if valid is None else np.array(valid),
                           age=0., support_axis=np.array([0., 0., 1.]), phase='HOLDING')


def run_checks():
    checks = {}
    x, residual, _ = solve_qp(np.eye(2), np.array([-2., -1.]),
                             np.array([[-1., 0.], [0., -1.], [1., 1.]]),
                             np.array([0., 0., 1.]), 120, 1e-9)
    checks['qp_known_solution'] = bool(np.allclose(x, [1., 0.], atol=1e-6) and residual < 1e-7)
    weighted = solve_qp(np.diag([2., 8.]), np.array([-4., -8.]),
                        np.array([[-1., 0.], [0., -1.], [1., 1.]]),
                        np.array([0., 0., 1.]), 120, 1e-9)
    checks['qp_weighted_toy_optimum'] = bool(weighted.converged
        and np.allclose(weighted.x, [.4, .6], atol=1e-7) and weighted.residual < 1e-8)
    lower_bound = solve_qp(np.diag([2., 8.]), np.array([-4., -8.]),
                           np.array([[-1., 0.], [0., -1.], [1., 1.]]),
                           np.array([-.5, -.25, 1.]), 120, 1e-9)
    checks['qp_nonzero_feasible_start_search'] = bool(lower_bound.converged
        and np.allclose(lower_bound.x, [.5, .5], atol=1e-7) and lower_bound.residual < 1e-8)
    limited = solve_qp(np.eye(2), np.array([-2., -1.]),
                       np.array([[-1., 0.], [0., -1.], [1., 1.]]),
                       np.array([0., 0., 1.]), 1, 1e-9, feasible_initial=np.zeros(2))
    checks['iteration_limit_is_feasible_numerical_failure'] = bool(
        not limited.converged and limited.residual <= 1e-8
        and limited.status == 'numerical_iteration_limit_feasible')
    target = project_force([10., 1., 1., 1., 1.], [3.]*5, 5.)
    checks['force_projection'] = bool(np.all(target >= 0.) and np.all(target <= 3.)
                                     and abs(target.sum()-5.) < 1e-6)
    estimator = make_controller('fixed', [7.]*5, 24.).state
    for k in range(100):
        f = frame(k*.005, slip=.02)
        f.phase = 'CLOSING'
        estimator.update(f, .005)
    closing_retains_prior = np.allclose(estimator.mu, Gains().nominal_mu)
    for k in range(100):
        f = frame(1.+k*.005, slip=.02)
        f.joint_velocity[:] = .2
        estimator.update(f, .005)
    moving_retains_prior = np.allclose(estimator.mu, Gains().nominal_mu)
    for k in range(100):
        estimator.update(frame(2.+k*.005, slip=.02), .005)
    holding_adapts = np.all(estimator.mu < Gains().nominal_mu) and np.all(estimator.mu >= .10)
    checks['friction_identifiability_boundary'] = bool(
        closing_retains_prior and moving_retains_prior and holding_adapts)
    fused_state = make_controller('fixed', [7.]*5, 24.).state
    for k in range(100):
        f = frame(k*.005)
        f.motor_effort, f.load_estimate, f.load_uncertainty = 30., 3., .25
        f.load_valid, f.load_confidence = True, 1.
        fused_state.update(f, .005)
    valid_fusion = fused_state.load > fused_state.tactile_load
    valid_fusion &= fused_state.load <= 3. and fused_state.load_uncertainty >= .25
    f = frame(1.)
    f.load_estimate, f.load_uncertainty, f.load_valid = 9999., .01, False
    fused_state.update(f, .005)
    invalid_excluded = abs(fused_state.load-fused_state.tactile_load) < 1e-8
    checks['load_channels_confidence_and_uncertainty'] = bool(valid_fusion and invalid_excluded)
    methods = {}
    for name in METHODS:
        controller = make_controller(name, [7.]*5, 24.)
        f = frame()
        original_up = f.support_axis.copy()
        a = controller.step(f, 'HOLDING', 'hold')
        finite = np.isfinite(a.delta_closure).all() and np.isfinite(a.predicted_normal).all()
        bounds = np.max(a.delta_closure) <= controller.g.close_rate*.005+1e-8
        bounds &= np.min(a.delta_closure) >= -controller.g.open_rate*.005-1e-8
        observational_purity = np.array_equal(f.support_axis, original_up)
        release = controller.step(frame(1.005, valid=[False]*5), 'HOLDING', 'release')
        obedience = (np.all(release.delta_closure < 0.) and release.status == 'user_release'
                     and not release.pause_transport and np.all(controller.desired == 0.))
        invalid = make_controller(name, [7.]*5, 24.).step(frame(valid=[False]*5), 'HOLDING')
        unobservable = invalid.pause_transport and np.all(invalid.delta_closure == 0.)
        permission_frame = frame(normal=[0.]*5)
        permission_frame.closure_permission = np.array([False, True, True, True, True])
        permitted = make_controller(name, [7.]*5, 24.).step(permission_frame, 'CLOSING')
        respects_permission = permitted.delta_closure[0] <= 1e-10
        maximum_frame = frame(normal=[0.]*5, alpha=1.)
        maximum_frame.closure_max = np.array([1., 1.55, 1.55, 1.55, 1.55])
        maximum_action = make_controller(name, [7.]*5, 24.).step(maximum_frame, 'CLOSING')
        respects_maximum = maximum_action.delta_closure[0] <= 1e-10
        respects_maximum &= np.all(maximum_action.delta_closure[1:] > 0.)
        negative_frame = frame(normal=[6., 1., 1., 1., 1.], alpha=0.)
        negative_frame.closure_min = np.array([-1., 0., 0., 0., 0.])
        negative_frame.closure_max = np.array([1., 1.55, 1.55, 1.55, 1.55])
        negative_controller = make_controller(name, [2.]*5, 9.)
        for _ in range(80):
            negative_controller.state.update(negative_frame, .005)
        negative_action = negative_controller.step(negative_frame, 'HOLDING')
        negative_unloading = negative_action.delta_closure[0] < 0.
        negative_frame.time += .005
        negative_frame.alpha[0] = -.4
        negative_frame.joint[0] = -.4
        continuing_release = negative_controller.step(negative_frame, 'HOLDING', 'release')
        negative_release = continuing_release.delta_closure[0] < 0.
        near_lower = frame(2., alpha=0.)
        near_lower.alpha[0] = near_lower.joint[0] = -.999
        near_lower.closure_min = np.array([-1., 0., 0., 0., 0.])
        near_lower.closure_max = negative_frame.closure_max.copy()
        terminal_release = negative_controller.step(near_lower, 'RELEASING', 'release')
        release_endpoint = abs(terminal_release.delta_closure[0]+.001) < 1e-9
        methods[name] = dict(finite=bool(finite), action_bounds=bool(bounds),
                             observation_not_mutated=bool(observational_purity),
                             release_obedience=bool(obedience),
                             invalid_observation_pause=bool(unobservable),
                             public_closure_permission=bool(respects_permission),
                             per_finger_reachable_maximum=bool(respects_maximum),
                             negative_thumb_force_unloading=bool(negative_unloading),
                             negative_thumb_continued_release=bool(negative_release),
                             release_true_lower_endpoint=bool(release_endpoint),
                             action=a.delta_closure.tolist(), debug=a.debug)
    checks['all_method_interface_release_constraints'] = all(
        all(value for key, value in result.items() if key not in ('action', 'debug'))
        for result in methods.values())
    supported_controller = make_controller('unified', [7.]*5, 24.)
    for k in range(80):
        f = frame(k*.005)
        supported_controller.state.update(f, .005)
    supported_controller._decide(frame(1.), 'HOLDING')
    selected = supported_controller.debug['selected_support_contacts']
    checks['predicted_support_set_floor'] = bool(len(selected) == 3 and selected[0] == 0
        and all(supported_controller.predicted[i] >= Gains().support_contact_floor-1e-8 for i in selected)
        and not supported_controller.debug['contact_reestablishment'])
    for k in range(100):
        f = frame(2.+k*.005, normal=[.005, 1., 1., 1., 1.])
        supported_controller.step(f, 'HOLDING')
    checks['contact_loss_same_qp_reestablishment'] = bool(
        supported_controller.debug['contact_reestablishment']
        and supported_controller.label == 'contact_reestablishment'
        and supported_controller.pause_transport
        and supported_controller.interval_action[0] >= -1e-8)
    impossible = make_controller('unified', [.25]*5, 1.)
    for k in range(30):
        imp_action = impossible.step(frame(1.+k*.005, normal=[4.]*5, slip=.03), 'HOLDING')
    checks['infeasible_is_explicit'] = bool(imp_action.pause_transport
                                           and imp_action.debug['infeasible_decisions'] > 0)
    # All methods see the same noisy observation stream. This only checks
    # causal finite responses and execution rates, not physical performance.
    rng = np.random.default_rng(1337)
    controllers = [make_controller(name, [7.]*5, 24.) for name in METHODS]
    start = time.perf_counter()
    finite_all = True
    qp_seconds = []
    for k in range(800):
        n = np.array([2., 1., 1., 1., 1.])+.15*rng.standard_normal(5)
        f = frame(k*.005, normal=n, slip=.01 if 300 <= k < 380 else 0.)
        for c in controllers:
            a = c.step(f, 'CLOSING' if k < 100 else 'HOLDING')
            finite_all &= bool(np.isfinite(a.delta_closure).all())
            if k % 4 == 0 and 'qp_seconds' in a.debug:
                qp_seconds.append(a.debug['qp_seconds'])
    checks['noisy_causal_stream_finite'] = finite_all
    timing = dict(stream_800_ticks_7_methods_seconds=time.perf_counter()-start,
                  unified_decision_median_ms=float(np.median(qp_seconds)*1000.),
                  unified_decision_p95_ms=float(np.percentile(qp_seconds, 95)*1000.))
    return dict(passed=all(checks.values()), checks=checks, methods=methods, timing=timing,
                scope='Synthetic observability/action/solver checks only; no MuJoCo claim.',
                formal_seeds_used=False)


if __name__ == '__main__':
    result = run_checks()
    (ROOT/'controller_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'checks': result['checks'], 'timing': result['timing']}, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)

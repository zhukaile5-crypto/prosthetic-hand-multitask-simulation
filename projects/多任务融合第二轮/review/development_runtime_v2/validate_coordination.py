"""Reproducible mechanism checks for coordination, independent of MuJoCo trials.

These checks establish command behavior and bounds under supplied tactile frames.
They do not establish actual contact-force safety, physical validity, or a task
success advantage. Run directly to write review/controller_checks.json.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import traceback

import numpy as np

from controllers import Gains, METHODS, METHOD_TITLES, make_controller


ROOT = Path(__file__).resolve().parent
DT = .005
RANDOM_SEED = 44001
# The unchanged baseline uses 28-step bisection for its budget projection.
# Preserve its output exactly in unconflicted cases, including the small
# floating-point budget residual rather than silently changing the baseline.
REQUEST_TOLERANCE_N = 1e-6
COORDINATED_METHODS = ('coordinated', 'coordinated_no_guard',
                       'coordinated_no_redistribution')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sensor_frame(t, normal=None, shear=None, slip=0., valid=None, age=.01):
    return SimpleNamespace(
        time=float(t),
        normal=np.full(5, 2.) if normal is None else np.asarray(normal, dtype=float),
        shear=np.full(5, .8) if shear is None else np.asarray(shear, dtype=float),
        slip=float(slip),
        valid=np.ones(5, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        age=float(age), collision=0.,
    )


def ratio_core(**changes):
    parameters = dict(base=1.2, ratio_kp=1., ratio_ki=.3, ratio_floor=1.8)
    parameters.update(changes)
    return Gains(**parameters)


def initialize(controller, start=8., ticks=250):
    for k in range(ticks):
        controller.step(sensor_frame(start+k*DT), 'HOLDING', 'hold')
    return start+ticks*DT


def run_checks():
    results = []

    archived_path = ROOT/'review'/'development_runtime_v1'/'controllers.py'
    archived_source = archived_path.read_text(encoding='utf-8')
    current_source = (ROOT/'controllers.py').read_text(encoding='utf-8')
    archived_ast, current_ast = ast.parse(archived_source), ast.parse(current_source)
    preserved_names = ('Common','EventBaseline','RatioBaseline','AdaptiveConeBaseline',
                       'Fusion','box_budget_projection')
    for name in preserved_names:
        archived_node = next(node for node in archived_ast.body
                             if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name)
        current_node = next(node for node in current_ast.body
                            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name)
        require(ast.dump(archived_node,include_attributes=False)==ast.dump(current_node,include_attributes=False),
                f'{name}: baseline implementation AST changed')
        require(ast.get_source_segment(archived_source,archived_node)==ast.get_source_segment(current_source,current_node),
                f'{name}: baseline source text changed')
    results.append(dict(check='baseline_source_and_ast_preservation',
                        reference_file=str(archived_path.resolve()),
                        reference_sha256=sha256(archived_path),
                        preserved_definitions=list(preserved_names),
                        source_text_equal=True,ast_equal=True))

    # Normal frames vary after initialization to exercise the PI, rather than
    # checking only the constant acquisition request or the first sample.
    for gains in (ratio_core(), ratio_core(base=1.45, ratio_kp=2.,
                                          ratio_ki=.5, ratio_floor=1.6)):
        baseline = make_controller('ratio', [5.]*5, 22., gains)
        coordinated = make_controller('coordinated', [5.]*5, 22., gains)
        require(coordinated.g == gains, 'Explicit gains differ from runtime gains')
        maximum_difference = 0.
        for k in range(1000):
            normal = 2.+.1*np.sin(k*.025+np.arange(5)*.2)
            shear = normal*(.35+.03*np.sin(k*.04))
            frame = sensor_frame(8.+k*DT, normal=normal, shear=shear)
            before = baseline.step(frame, 'HOLDING', 'hold')
            after = coordinated.step(frame, 'HOLDING', 'hold')
            maximum_difference = max(maximum_difference,
                                     float(np.max(np.abs(before-after))))
        require(maximum_difference == 0., 'Unnecessary coordination changed the baseline')
        results.append(dict(check='no_conflict_baseline_identity', ticks=1000,
                            core=asdict(gains), max_absolute_difference=maximum_difference))

    default = make_controller('coordinated', [5.]*5, 22.)
    require((default.g.base, default.g.ratio_kp, default.g.ratio_ki,
             default.g.ratio_floor) == (1.2, 1., .3, 1.8),
            'Default coordinator core is not the declared low-gain ratio core')

    acquisition_identity_frames = 0
    for gains in (ratio_core(), ratio_core(base=1.45, ratio_kp=2.,
                                          ratio_ki=.5, ratio_floor=1.6)):
        for name in COORDINATED_METHODS:
            for phase, start in (('CLOSING', .1), ('LIFTING', 8.1), ('HOLDING', 4.1)):
                baseline = make_controller('ratio', [5.]*5, 22., gains)
                controller = make_controller(name, [5.]*5, 22., gains)
                for k in range(200):
                    # High acquisition slip is deliberate: it must not add
                    # force or run the ratio PI before an established hold.
                    frame = sensor_frame(start+k*DT, slip=.03)
                    expected = baseline.step(frame, phase, 'hold')
                    actual = controller.step(frame, phase, 'hold')
                    require(np.array_equal(actual, expected),
                            f'{name}: unneeded acquisition supervision changed the baseline')
                    require(controller.integral == 0. and controller.target_ratio is None,
                            f'{name}: acquisition ran the ratio PI/initialization')
                    require(controller.slip_boost == 0. and controller.coordination_boost == 0.,
                            f'{name}: acquisition responded to slip with added force')
                    require(controller.debug['redistributed_force'] == 0.,
                            f'{name}: acquisition redistributed force')
                    acquisition_identity_frames += 1
    results.append(dict(check='acquisition_baseline_identity_and_no_slip_action',
                        frames=acquisition_identity_frames,
                        phases=['CLOSING','LIFTING','time_before_8s'],
                        methods=list(COORDINATED_METHODS), supplied_slip=.03))

    acquisition_guarded = make_controller('coordinated', [5.]*5, 22., ratio_core())
    acquisition_baseline = make_controller('ratio', [5.]*5, 22., ratio_core())
    local_frame = sensor_frame(.3, normal=[6.,2.,2.,2.,2.], slip=.03)
    expected = acquisition_baseline.step(local_frame, 'CLOSING', 'hold')
    actual = acquisition_guarded.step(local_frame, 'CLOSING', 'hold')
    require(0. < actual[0] < expected[0], 'Acquisition guard cancelled rather than gently reducing a modest overload')
    require(np.array_equal(actual[1:],expected[1:]), 'A local acquisition fault changed unaffected requests')
    previous = actual.copy()
    invalid_frame = sensor_frame(.305, normal=[np.nan,2.,2.,2.,2.],
                                shear=[np.nan,.8,.8,.8,.8],
                                valid=[False,True,True,True,True])
    actual = acquisition_guarded.step(invalid_frame, 'LIFTING', 'hold')
    require(actual[0] <= previous[0]+1e-8, 'Invalid acquisition channel gained force')
    global_guarded = make_controller('coordinated', [5.]*5, 10., ratio_core())
    global_baseline = make_controller('ratio', [5.]*5, 10., ratio_core())
    global_frame = sensor_frame(.4, normal=[4.]*5, shear=[.8]*5)
    expected_total = float(np.sum(global_baseline.step(global_frame,'LIFTING','hold')))
    actual_total = float(np.sum(global_guarded.step(global_frame,'LIFTING','hold')))
    require(0. < actual_total < expected_total, 'Acquisition total-pressure guard failed to reduce the request')
    results.append(dict(check='acquisition_local_global_guard_and_invalid_freeze',
                        local_before=previous.tolist(),local_guard_request=actual.tolist(),
                        global_baseline_sum=expected_total,global_guarded_sum=actual_total))

    cross_phase = make_controller('coordinated', [5.]*5, 22., ratio_core())
    acquired = cross_phase.step(sensor_frame(7.9,normal=[6.,2.,2.,2.,2.]),'LIFTING','hold')
    holding = cross_phase.step(sensor_frame(8.,normal=[2.]*5),'HOLDING','hold')
    require(holding[0] <= acquired[0]+1e-8, 'Acquisition pressure memory was cleared at the hold boundary')
    for k in range(600):
        cross_phase.step(sensor_frame(8.005+k*DT),'HOLDING','hold')
    require(np.all(np.isinf(cross_phase.guard_caps)), 'Old pressure impulse permanently locked local caps')
    require(np.isinf(cross_phase.total_cap), 'Old pressure impulse permanently locked total cap')
    easing = make_controller('coordinated', [5.]*5,22.,ratio_core())
    first = easing.step(sensor_frame(.1,normal=[10.,2.,2.,2.,2.]),'CLOSING','hold')
    for k in range(100):
        last = easing.step(sensor_frame(.105+k*DT,normal=[5.,2.,2.,2.,2.]),'CLOSING','hold')
    require(last[0] > first[0], 'A past strong impulse remained the minimum cap despite easing pressure')
    results.append(dict(check='cross_phase_guard_continuity_and_recovery',
                        acquisition_thumb=float(acquired[0]),hold_boundary_thumb=float(holding[0]),
                        severe_impulse_request=float(first[0]),easing_pressure_request=float(last[0]),
                        recovered_local_caps=True))

    for slip_from_start in (False, True):
        baseline = make_controller('ratio', [5.]*5,22.,ratio_core())
        coordinated = make_controller('coordinated', [5.]*5,22.,ratio_core())
        maximum_difference = 0.
        for k in range(350):
            slip = .02 if slip_from_start or k >= 250 else 0.
            frame = sensor_frame(8.+k*DT,slip=slip,shear=[.25]*5 if slip else None)
            expected = baseline.step(frame,'HOLDING','hold')
            actual = coordinated.step(frame,'HOLDING','hold')
            maximum_difference = max(maximum_difference,float(np.max(np.abs(expected-actual))))
            require(coordinated.coordination_boost == 0., 'Slip alone created additional coordination force')
        require(maximum_difference == 0., 'Slip-only inputs changed the unconflicted ratio baseline')
        results.append(dict(check='slip_alone_retains_baseline',slip_from_initialization=slip_from_start,
                            ticks=350,max_absolute_difference=maximum_difference))

    controller = make_controller('coordinated', [5.]*5, 22., ratio_core())
    start = initialize(controller)
    target_before_slip = float(controller.target_ratio)
    maximum_boost = 0.
    for k in range(100):
        controller.step(sensor_frame(start+k*DT, shear=[.25]*5, slip=.02),
                        'HOLDING', 'hold')
        maximum_boost = max(maximum_boost, float(controller.coordination_boost))
    require(.25/2. < target_before_slip, 'Slip check failed to lower the friction ratio')
    require(controller.slip_confirmed, 'Sustained slip with falling ratio was missed')
    require(maximum_boost == 0., 'Falling ratio slip without pressure caused extra force')
    require(maximum_boost <= controller.g.coord_boost_max+1e-8,
            'Slip boost exceeded the declared bound')
    results.append(dict(check='falling_ratio_slip_confirmation', ticks=100,
                        target_ratio_before=target_before_slip,
                        supplied_ratio_during=.25/2., max_boost=maximum_boost,
                        boost_bound=controller.g.coord_boost_max))

    maximum_lost_channel_increment = 0.
    for k in range(100):
        output = controller.step(sensor_frame(start+.5+k*DT,
                                             normal=[6., 0., 2., 2., 2.],
                                             shear=[.75, 0., .25, .25, .25],
                                             slip=.02), 'HOLDING', 'hold')
        maximum_lost_channel_increment = max(maximum_lost_channel_increment,
                                              float(output[1]-controller.baseline_desired[1]))
    require(maximum_lost_channel_increment <= 1e-8,
            'A lost contact received coordination force above its reacquisition baseline')
    results.append(dict(check='lost_contact_receives_no_addition', ticks=100,
                        max_increment_over_baseline=maximum_lost_channel_increment))

    guarded_output = None
    unguarded_output = None
    for name in COORDINATED_METHODS:
        controller = make_controller(name, [5.]*5, 22., ratio_core())
        start = initialize(controller)
        output = controller.step(sensor_frame(start, normal=[6., 2., 2., 2., 2.]),
                                 'HOLDING', 'hold')
        if name == 'coordinated':
            guarded_output = output.copy()
            require(output[0] < controller.baseline_desired[0],
                    'Measured local overload did not reduce the affected request')
        elif name == 'coordinated_no_guard':
            unguarded_output = output.copy()
        previous = output.copy()
        output = controller.step(sensor_frame(start+DT,
                                             normal=[np.nan, 2., 2., 2., 2.],
                                             shear=[np.nan, .8, .8, .8, .8],
                                             valid=[False, True, True, True, True]),
                                 'HOLDING', 'hold')
        require(np.all(np.isfinite(output)), 'Invalid channel produced a nonfinite output')
        require(output[0] <= previous[0]+1e-8, 'Invalid channel received an increased request')
        results.append(dict(check='invalid_channel_cannot_increase', method=name,
                            before=float(previous[0]), after=float(output[0])))
    require(guarded_output[0] < unguarded_output[0],
            'Guard ablation did not remove selective overload reduction')
    results.append(dict(check='force_guard_has_selective_effect',
                        guarded_request=guarded_output.tolist(),
                        without_guard_request=unguarded_output.tolist()))

    redistributed = {}
    for name in ('coordinated', 'coordinated_no_redistribution'):
        controller = make_controller(name, [5.]*5, 22., ratio_core())
        start = initialize(controller)
        max_reassigned = 0.
        maximum_over_baseline_sum = 0.
        for k in range(120):
            output = controller.step(sensor_frame(start+k*DT,
                                                 normal=[2., 5., 2., 2., 2.],
                                                 shear=[.5, 1.25, .5, .5, .5],
                                                 slip=.02), 'HOLDING', 'hold')
            max_reassigned = max(max_reassigned,
                                 float(controller.debug['redistributed_force']))
            maximum_over_baseline_sum = max(maximum_over_baseline_sum,
                                             float(np.sum(output)-np.sum(controller.baseline_desired)))
            require(controller.coordination_boost <= controller.debug['pressure_removed_request']+1e-7,
                    'Compensation exceeded the baseline request actually removed by pressure protection')
        require(maximum_over_baseline_sum <= 1e-7,
                'Redistribution created extra total requested force beyond the baseline')
        redistributed[name] = dict(final_total_request=float(np.sum(output)),
                                   max_reassigned=max_reassigned,
                                   max_sum_over_baseline=maximum_over_baseline_sum)
    require(redistributed['coordinated']['max_reassigned'] > 0.,
            'Available valid-contact redistribution was never exercised')
    require(redistributed['coordinated_no_redistribution']['max_reassigned'] == 0.,
            'Redistribution persisted in its ablation')
    results.append(dict(check='redistribution_component_is_exercised', methods=redistributed))

    # Release authority is tested on every registered method after its active
    # internal memories have received a sustained-slip sequence.
    for name in METHODS:
        controller = make_controller(name, [5.]*5, 22., ratio_core())
        start = initialize(controller)
        for k in range(80):
            controller.step(sensor_frame(start+k*DT, slip=.02), 'HOLDING', 'hold')
        by_phase = controller.step(sensor_frame(start+.4), 'RELEASING', 'hold')
        by_command = controller.step(sensor_frame(start+.405), 'HOLDING', 'release')
        require(np.array_equal(by_phase, np.zeros(5)), f'{name}: release phase did not unload')
        require(np.array_equal(by_command, np.zeros(5)), f'{name}: release command did not unload')
    require(set(METHODS) == set(METHOD_TITLES), 'Method titles and registry differ')
    results.append(dict(check='all_methods_release_authority', methods=list(METHODS),
                        command_and_phase=True))

    rng = np.random.default_rng(RANDOM_SEED)
    checked_frames = 0
    max_budget_excess = 0.
    max_baseline_sum_excess = 0.
    regimes = (([5.]*5, 22.), ([4.2, 1.7, 3.5, 3.5, 3.5], 14.),
               ([3., 2., 2., 2., 2.], 10.))
    for name in COORDINATED_METHODS:
        for limits, budget in regimes:
            controller = make_controller(name, limits, budget, ratio_core())
            for k in range(3000):
                valid = rng.random(5) > .15
                normal = rng.exponential(2., 5)
                normal[rng.random(5) < .1] = 0.
                shear = rng.uniform(.01, .6, 5)*normal
                normal[~valid] = np.nan
                shear[~valid] = np.nan
                previous = controller.desired.copy()
                output = controller.step(sensor_frame(8.+k*DT, normal=normal,
                                                       shear=shear,
                                                       slip=rng.uniform(0., .018),
                                                       valid=valid, age=.06),
                                         'HOLDING', 'hold')
                require(np.all(np.isfinite(output)), f'{name}: nonfinite randomized output')
                require(np.all(output >= -1e-8), f'{name}: negative randomized request')
                require(np.all(output <= np.asarray(limits)+1e-7),
                        f'{name}: per-finger requested-force bound exceeded')
                max_budget_excess = max(max_budget_excess,float(np.sum(output)-budget))
                max_baseline_sum_excess = max(max_baseline_sum_excess,
                                               float(np.sum(output)-np.sum(controller.baseline_desired)))
                require(np.sum(output) <= budget+REQUEST_TOLERANCE_N, f'{name}: total requested-force bound exceeded')
                require(np.sum(output) <= np.sum(controller.baseline_desired)+REQUEST_TOLERANCE_N,
                        f'{name}: compensation created force beyond the baseline total')
                require(np.all(output[~valid] <= previous[~valid]+1e-7),
                        f'{name}: invalid channel gained force')
                unsupported = valid & ((normal <= .15) | (controller.normal <= .25))
                require(np.all(output[unsupported] <= controller.baseline_desired[unsupported]+1e-7),
                        f'{name}: lost contact received a positive addition')
                require(0. <= controller.coordination_boost <= controller.g.coord_boost_max+1e-7,
                        f'{name}: randomized boost exceeded its bound')
                require(controller.coordination_boost <= controller.debug['pressure_removed_request']+1e-7,
                        f'{name}: compensation exceeded the pressure-removed request')
                checked_frames += 1
    results.append(dict(check='randomized_command_feasibility', seed=RANDOM_SEED,
                        frames=checked_frames, methods=list(COORDINATED_METHODS),
                        projection_residual_tolerance_N=REQUEST_TOLERANCE_N,
                        maximum_budget_residual_N=max_budget_excess,
                        maximum_sum_over_baseline_N=max_baseline_sum_excess,
                        limit_regimes=[dict(limits=l, budget=b) for l, b in regimes]))
    return results


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'review'/'controller_checks.json')
    args = parser.parse_args()
    report = dict(
        status='running', generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Synthetic tactile-frame controller mechanisms and requested-force bounds only',
        limitations=['Actual contact-force safety is not proved by these checks.',
                     'Physical model validity is not tested here.',
                     'Task success or advantage over a baseline is not proved here.'],
        sources={'controllers.py': sha256(ROOT/'controllers.py'),
                 'validate_coordination.py': sha256(Path(__file__).resolve())},
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
    )
    try:
        report['checks'] = run_checks()
        report['status'] = 'passed'
        exit_code = 0
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(args.output.resolve()),
                         check_groups=len(report.get('checks', [])),
                         error=report.get('error')), ensure_ascii=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

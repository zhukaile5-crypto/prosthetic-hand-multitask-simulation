"""Read the terminal V26 five-branch record using saved public measurements and receipts."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'v26_five_handover_v1'
RUNNER = 'identification/v26_small_reference_response/run_identification.py'
ARMS = ('zero', 'zero_repeat', 'lift_plus', 'finger_single', 'mixed_holdout')
PREDICTOR_ARMS = ('zero', 'lift_plus', 'finger_single', 'mixed_holdout')
COMPONENTS = {'zero': (), 'zero_repeat': (), 'lift_plus': ((18, 1),),
              'finger_single': ((7, -1),), 'mixed_holdout': ((18, 1), (7, -1))}
AXIS_NAMES = {7: 'rh_A_FFJ0', 18: 'lift_reference_m'}
RATES = {7: 3., 18: .18}
ORIGIN, TRIGGER, DT, DELAY, HOLD = 13.940, 13.935, .005, .010, .200
MAIN_HORIZONS = (.020, .040)
OTHER_HORIZONS = (.005, .010, .100, .200)
HORIZONS = tuple(sorted(MAIN_HORIZONS + OTHER_HORIZONS))
INPUT_TOL, CLOCK_TOL, FORCE_SCALE = 1e-12, 1e-9, .030
TIMING_COLUMNS = {'control_ms', 'actor_callback_ms', 'full_control_loop_ms', 'observation_ms', 'physics_advance_ms'}
DEBUG_TIMING_KEYS = ('qp', 'native_inverse', 'whole_decision')
OBS = 'response_observations.jsonl.gz'
REC = 'response_receipts.jsonl.gz'
INT = 'response_input_intervals.jsonl.gz'
PROBE = 'probe_decisions.jsonl.gz'
RECEIVER = 'response_receiver.jsonl.gz'
STREAMS = (OBS, REC, INT, PROBE, RECEIVER)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(row, *, allow_nan=False):
    return json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=allow_nan).encode('utf-8')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def array(value, shape):
    raw = np.asarray(value, dtype=object)
    require(raw.shape == shape and all(finite(item) for item in raw.flat), 'invalid_numeric_shape_or_value')
    return np.asarray(raw, float)


def same_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def near(left, right, tolerance=INPUT_TOL):
    return bool(np.all(np.abs(np.asarray(left) - np.asarray(right)) <= tolerance))


def tick(time_s):
    require(finite(time_s), 'invalid_clock_type')
    index = round(time_s / DT)
    require(abs(time_s - index * DT) <= CLOCK_TOL, 'off_grid_clock')
    return index


def in_root(base, value):
    result = (base / value).resolve()
    require(result.is_relative_to(base.resolve()), 'artifact_path_outside_declared_root')
    return result


def validate_run(run, registration_path):
    complete = read(run / 'execution_complete.json')
    require(complete['schema'] == SCHEMA and complete['mode'] == 'identify', 'wrong_execution_schema')
    rows = complete['rows']
    require(complete['planned'] == len(rows) == 5 and [r['key'] for r in rows] == list(ARMS)
            and complete['all_assigned_accounted'] is True, 'terminal_assignment_not_exact_five')
    require(complete['sources_unchanged'] is True and complete['numerical_unchanged'] is True,
            'source_or_numeric_identity_changed')
    for name in ('plan', 'manifest', 'assignment_ledger'):
        require(sha(run / (name + '.json')) == complete[name + '_sha256'], 'completion_pin_mismatch:' + name)
    require(read(run / 'assignment_ledger.json') == rows and read(run / 'runtime_failures.json') == complete['runtime_failures'],
            'terminal_ledger_mismatch')
    plan = read(run / 'plan.json')
    manifest = read(run / 'manifest.json')
    reg_path = in_root(ROOT, registration_path or plan['registration'])
    require(sha(reg_path) == plan['registration_sha256'], 'registration_pin_mismatch')
    reg = read(reg_path)
    source = plan['source_sha256']
    require(len(source) == 199 and RUNNER in source and manifest['frozen_v25_source_sha256'] ==
            {key: value for key, value in source.items() if key != RUNNER}, 'wrong_199_source_chain')
    for key in ('source_sha256', 'numerical_identity', 'environment'):
        require(complete[key] == plan[key] == reg[key], 'registered_identity_mismatch:' + key)
    require(manifest['source_sha256'] == source and manifest['numerical_identity'] == plan['numerical_identity']
            and manifest['plan_sha256'] == complete['plan_sha256'], 'manifest_identity_mismatch')
    snapshots = run / 'source_snapshot_manifest.json'
    require(sha(snapshots) == manifest['source_snapshot_manifest_sha256'], 'snapshot_manifest_pin_mismatch')
    files = read(snapshots)['files']
    require(set(files) == set(source), 'source_snapshot_keys_mismatch')
    for name, record in files.items():
        path = in_root(run, record['snapshot_path'])
        require(sha(path) == record['sha256'] == source[name], 'source_snapshot_mismatch:' + name)
    require(plan['assignment'] == reg['assignment'] and reg['software_gate_passed'] is True,
            'registered_assignment_mismatch')
    protocol = plan['protocol']
    require(protocol == reg['identification_contract'] and protocol['arms'] == list(ARMS)
            and protocol['origin_s'] == ORIGIN and protocol['trigger_read_time_s'] == TRIGGER
            and protocol['hold_s'] == HOLD and protocol['delay_s'] == DELAY and protocol['extra_flush_s'] == DT
            and protocol['amplitude_fraction'] == .5 and protocol['acquisition_horizons_s'] == list(HORIZONS),
            'fixed_protocol_mismatch')
    axes = protocol['reference_axes21']
    require(len(axes) == len(set(axes)) == 21 and all(axes[i] == name for i, name in AXIS_NAMES.items()),
            'reference_axis_order_mismatch')
    for arm, parts in COMPONENTS.items():
        require(protocol['components'][arm] == [dict(axis=i, name=axes[i], sign=s) for i, s in parts],
                'registered_components_mismatch:' + arm)
    spec = reg['preregistered_readout']
    expected = dict(main_horizons_s=list(MAIN_HORIZONS), other_horizons_s=list(OTHER_HORIZONS),
                    input_numeric_tolerance=INPUT_TOL, force_scale_N=FORCE_SCALE,
                    finger_force_sigma_N=.01, receiver_force_sigma_N=.01)
    require(all(spec.get(key) == value for key, value in expected.items()), 'readout_parameters_mismatch')
    require(sha(in_root(ROOT, spec['contract_path'])) == spec['contract_sha256'], 'readout_contract_pin_mismatch')
    origin = read(run / 'origin_public.json')
    require(origin['available'] is True and origin['phase'] == 'TRANSFER' and origin['trigger'] == 'receiver_transfer'
            and origin['next_PRE_OBSERVE_boundary'] is True and tick(origin['time_s']) == tick(ORIGIN)
            and tick(origin['trigger_read_time_s']) == tick(TRIGGER), 'public_origin_mismatch')
    return complete, plan, axes, reg_path, spec, origin


def scan_stream(path, manifest, arm, last_tick):
    pin = manifest[path.name]
    digest = sha(path)
    require(pin['sha256'] == digest, 'stream_sha_mismatch:' + path.name)
    payload, prefix = hashlib.sha256(), hashlib.sha256()
    selected, count, previous = {}, 0, None
    first = tick(ORIGIN)
    targets = {first + round(h / DT) for h in HORIZONS}
    clock_key = {OBS: 'read_time_s', REC: 'issue_time_s', INT: 'start_s',
                 PROBE: 'time_s', RECEIVER: 'read_time_s'}[path.name]
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            key = tick(row[clock_key])
            require(previous is None or key == previous + 1, 'nonconsecutive_read_or_input_clock:' + path.name)
            previous = key
            encoded = row
            if path.name == PROBE:
                require(row['arm'] == arm, 'probe_arm_label_mismatch')
                if arm in ARMS[:2]:
                    encoded = {k: v for k, v in row.items() if k != 'arm'}
            raw = canonical(encoded) + b'\n'
            payload.update(raw)
            if path.name in (OBS, RECEIVER):
                acq = row['acquisition_time_s']
                require(tick(row['available_at_s']) == key, 'delivery_clock_mismatch:' + path.name)
                if acq is not None:
                    acquisition = tick(acq)
                    require(acquisition <= key and finite(row['age_s'])
                            and abs(row['age_s'] - (row[clock_key] - acq)) <= CLOCK_TOL, 'acquisition_clock_mismatch:' + path.name)
                    if acquisition <= first:
                        prefix.update(raw)
                    if acquisition in targets and key == acquisition + round(DELAY / DT):
                        require(acquisition not in selected, 'duplicate_exact_acquisition_label:' + path.name)
                        selected[acquisition] = row
            else:
                if key < first:
                    prefix.update(raw)
                if first - 1 <= key <= last_tick:
                    selected[key] = row
    require(count == pin['records'], 'stream_count_mismatch:' + path.name)
    expected_end = last_tick - 1 if path.name == INT else last_tick
    require(previous == expected_end, 'stream_does_not_reach_saved_boundary:' + path.name)
    if path.name in (PROBE, RECEIVER):
        require(count == last_tick - first + 1, 'short_program_or_receiver_window_mismatch')
    return selected, dict(file_sha256=digest, payload_sha256=payload.hexdigest(),
                          prefix_sha256=prefix.hexdigest(), rows=count)


def scan_branch(folder, row, plan, arm):
    meta = read(folder / 'metadata.json')
    fork = read(folder / 'opaque_fork_metadata.json')
    manifest = read(folder / 'response_manifest.json')['streams']
    require(meta['research_source_sha256'] == plan['source_sha256'] and meta['numerical_identity'] == plan['numerical_identity'],
            'branch_identity_mismatch')
    require(fork['passed'] is True and fork['physics_steps'] == fork['sensor_samples_added'] == 0
            and tick(fork['time_s']) == tick(ORIGIN), 'branch_fork_proof_mismatch')
    last_tick = tick(row['recorded_until_s']) - 1
    streams, evidence = {}, {}
    for name in STREAMS:
        streams[name], evidence[name] = scan_stream(folder / name, manifest, arm, last_tick)
    return dict(available=True, folder=folder, streams=streams, evidence=evidence, meta=meta,
                fork=fork['opaque_integration_sha256'], last_read_tick=last_tick)


def trace_debug(text):
    row = json.loads(str(text))
    timing = row.get('solver_timing_ms')
    if timing is not None:
        require(isinstance(timing, dict), 'invalid_debug_walltime_shape')
        for key in DEBUG_TIMING_KEYS:
            if key in timing:
                require(finite(timing[key]) and timing[key] >= 0, 'invalid_debug_walltime_value')
                timing[key] = '__validated_wall_time__'
    return canonical(row, allow_nan=True)


def compare_zero_traces(left, right):
    checks = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for name in sorted(set(a.files) | set(b.files)):
            if name not in a or name not in b:
                checks[name] = False
                continue
            if name == 'metadata':
                continue
            x, y = a[name], b[name]
            if name in TIMING_COLUMNS:
                checks[name] = bool(x.shape == y.shape and x.dtype == y.dtype and x.dtype.kind in 'fi'
                                    and np.isfinite(x).all() and np.isfinite(y).all() and (x >= 0).all() and (y >= 0).all())
            elif name == 'debug_json':
                checks[name] = bool(x.shape == y.shape and x.dtype.kind == y.dtype.kind == 'U'
                                    and all(trace_debug(u) == trace_debug(v) for u, v in zip(x.flat, y.flat)))
            elif x.dtype.kind == y.dtype.kind == 'U':
                checks[name] = bool(x.shape == y.shape and all(str(u).encode('utf-8') == str(v).encode('utf-8')
                                                              for u, v in zip(x.flat, y.flat)))
            else:
                checks[name] = same_bits(x, y)
    return checks


def verify_zeros(branches):
    zero, repeat = (branches[arm] for arm in ARMS[:2])
    if not zero['available'] or not repeat['available']:
        return dict(passed=False, reason='zero_or_repeat_record_unavailable')
    paths = [branch['folder'] / 'trace.npz' for branch in (zero, repeat)]
    traces = compare_zero_traces(*paths)
    streams = {name: all(zero['evidence'][name][k] == repeat['evidence'][name][k]
                         for k in ('rows', 'payload_sha256')) for name in STREAMS}
    return dict(passed=bool(traces) and all(traces.values()) and all(streams.values()), trace_columns=traces,
                streams=streams, trace_sha256=[sha(p) for p in paths],
                normalized_probe_provenance_field='arm', trace_provenance_field='metadata',
                timing_columns=sorted(TIMING_COLUMNS), debug_timing_keys=list(DEBUG_TIMING_KEYS))


def declared_target(branch, arm):
    row = branch['streams'][PROBE][tick(ORIGIN)]
    carrier = array(row['carrier_at_origin21'], (21,))
    lower = array(row['actual_TASK_lower21'], (21,))
    upper = array(row['actual_TASK_upper21'], (21,))
    require(np.all(lower <= upper) and np.all(lower - INPUT_TOL <= carrier)
            and np.all(carrier <= upper + INPUT_TOL), 'origin_carrier_outside_authority')
    target = carrier.copy()
    parts = row['component_support']
    require(len(parts) == len(COMPONENTS[arm]), 'origin_component_count_mismatch')
    for part, (axis, sign) in zip(parts, COMPONENTS[arm]):
        margin = max(0., float(upper[axis] - carrier[axis] if sign > 0 else carrier[axis] - lower[axis]))
        amplitude = .5 * min(RATES[axis] * DT, margin)
        target[axis] = carrier[axis] + sign * amplitude
        increment = target[axis] - carrier[axis]
        supported = bool(amplitude > 0 and increment != 0)
        require((part['axis'], part['name'], part['sign']) == (axis, AXIS_NAMES[axis], sign)
                and all(finite(part[name]) for name in ('origin_direction_margin', 'raw_amplitude', 'target_increment'))
                and near([margin, amplitude, increment], [part['origin_direction_margin'], part['raw_amplitude'], part['target_increment']])
                and part['supported'] is supported, 'origin_component_formula_mismatch')
    require(same_bits(target, array(row['raw_probe_target21'], (21,))), 'origin_fixed_target_mismatch')
    supported_count = sum(part['supported'] for part in parts)
    expected_exposure = ('ZERO_CONTROL' if not parts else 'UNEXPOSED' if supported_count == 0 else
                         'FULLY_EXPOSED' if supported_count == len(parts) else 'PARTIAL_UNSUPPORTED')
    require(row['origin_exposure_status'] == expected_exposure, 'origin_exposure_label_mismatch')
    return dict(carrier=carrier, target=target, parts=parts, exposure=row['origin_exposure_status'])


def actual_tick(branch, program, k, axes):
    rec = branch['streams'][REC][k]
    row = branch['streams'][PROBE][k]
    interval = branch['streams'][INT][k]
    actual = array(rec['executed_reference21'], (21,))
    pre = array(rec['precommit_reference21'], (21,))
    engine = array(rec['executed_engine21'], (21,))
    encoded = actual.copy()
    bias = rec['diagnostics']['known_actor_lift_bias_m']
    require(finite(bias) and rec['diagnostics']['reference_axes21'] == axes, 'receipt_axis_or_bias_mismatch')
    encoded[18] += bias
    require(same_bits(encoded, engine) and same_bits(actual - pre, array(rec['actual_reference_increment21'], (21,))),
            'actual_engine_or_increment_mismatch')
    prior = branch['streams'][REC][k - 1]
    require(same_bits(pre, array(prior['executed_reference21'], (21,))), 'precommit_history_discontinuity')
    require(tick(rec['issue_time_s']) == tick(rec['available_at_s']) == k and tick(interval['start_s']) == k
            and tick(interval['end_s']) == tick(interval['available_at_s']) == k + 1
            and interval['consecutive_5ms'] is True and same_bits(actual, array(interval['executed_reference21'], (21,))),
            'actual_receipt_interval_mismatch')
    receipt = row['actual_receipt']
    require(same_bits(actual, array(receipt['actual_reference21'], (21,)))
            and same_bits(engine, array(receipt['actual_engine21'], (21,)))
            and rec['mode_generation'] == receipt['mode_generation'] == row['current_generation']
            and rec['mode'] == receipt['mode'] == 'release' and rec['stale_plan_replaced'] is receipt['stale_plan_replaced'],
            'probe_decision_actual_receipt_mismatch')
    require(same_bits(program['target'], array(row['raw_probe_target21'], (21,)))
            and same_bits(program['carrier'], array(row['carrier_at_origin21'], (21,))), 'fixed_target_changed_during_program')
    lower, upper = (array(row[key], (21,)) for key in ('actual_TASK_lower21', 'actual_TASK_upper21'))
    require(np.all(lower - INPUT_TOL <= actual) and np.all(actual <= upper + INPUT_TOL), 'actual_outside_reported_task_authority')
    return actual, dict(carrier_hold_active=row['carrier_hold_active'] is True,
                        fixed_target_realized=near(actual, program['target']),
                        logged_target_clip=row['requested_target_clipped_by_fresh_TASK'] is True,
                        stale=rec['stale_plan_replaced'] is True)


def dose_history(branches, programs, axes):
    rows = []
    for k in range(tick(ORIGIN), tick(ORIGIN + HOLD)):
        row = dict(issue_time_s=k * DT, end_s=(k + 1) * DT, arms={}, matched=None, fixed=None, signed=None)
        actual = {}
        for arm in PREDICTOR_ARMS:
            try:
                actual[arm], evidence = actual_tick(branches[arm], programs[arm], k, axes)
                row['arms'][arm] = dict(available=True, executed_reference21=actual[arm].tolist(), **evidence)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                row['arms'][arm] = dict(available=False, reason=str(exc))
        if len(actual) == len(PREDICTOR_ARMS):
            deltas = {arm: value - actual['zero'] for arm, value in actual.items() if arm != 'zero'}
            residual = deltas['mixed_holdout'] - deltas['lift_plus'] - deltas['finger_single']
            signed, nontarget = {}, {}
            for arm, delta in deltas.items():
                axes_for_arm = {i for i, _ in COMPONENTS[arm]}
                nontarget[arm] = max(abs(delta[i]) for i in range(21) if i not in axes_for_arm)
                signed[arm] = {axes[i]: float(sign * delta[i]) for i, sign in COMPONENTS[arm]}
                row['arms'][arm]['delta_from_zero21'] = delta.tolist()
            row.update(matched=near(residual, 0.) and all(x <= INPUT_TOL for x in nontarget.values()),
                       fixed=all(x['fixed_target_realized'] for x in row['arms'].values()),
                       held=all(x['carrier_hold_active'] and not x['stale'] for x in row['arms'].values()),
                       signed=signed, combination_residual21=residual.tolist(), non_target_max_abs=nontarget)
        rows.append(row)
    return rows


def dose_at(history, h):
    n = round(h / DT)
    rows = history[:n]
    complete = len(rows) == n and all(r['matched'] is not None for r in rows)
    matched = complete and all(r['matched'] for r in rows)
    held = complete and all(r['held'] for r in rows)
    signed = {}
    for arm in PREDICTOR_ARMS[1:]:
        signed[arm] = {}
        for i, _ in COMPONENTS[arm]:
            key = AXIS_NAMES[i]
            values = [r['signed'][arm][key] for r in rows if r['signed'] is not None]
            signed[arm][key] = dict(ticks=len(values), positive_ticks=sum(x > INPUT_TOL for x in values),
                                    reversed_ticks=sum(x < -INPUT_TOL for x in values),
                                    signed_min=min(values, default=None), signed_max=max(values, default=None),
                                    signed_reference_seconds=sum(values) * DT if len(values) == n else None)
    parts = [x for per_arm in signed.values() for x in per_arm.values()]
    nonzero = complete and all(x['positive_ticks'] > 0 and x['reversed_ticks'] == 0 for x in parts)
    full = matched and held and nonzero and all(x['positive_ticks'] == n for x in parts) and all(r['fixed'] for r in rows)
    reason = (None if full else 'incomplete_actual_receipt_history' if not complete else
              'actual_non_target_or_combination_mismatch' if not matched else 'holding_interrupted_or_stale' if not held else
              'required_actual_component_unexposed_or_reversed' if not nonzero else 'fixed_registered_dose_not_fully_realized')
    return dict(expected_intervals=n, complete_intervals=sum(r['matched'] is not None for r in rows),
                matched_superposition=matched, carrier_hold_through_window=held, full_dose=full,
                matched_clipped_exploration=matched and held and nonzero and not full,
                full_dose_unavailable_reason=reason, actual_component_dose=signed,
                clipped_issue_times_s=[r['issue_time_s'] for r in rows if any(a.get('logged_target_clip') for a in r['arms'].values())])


def get_label(branch, stream, k, digit=None):
    if not branch['available']:
        return None, None, branch['reason']
    row = branch['streams'][stream].get(k)
    if row is None:
        return None, None, 'exact_acquisition_and_delivery_label_missing'
    clocks = {key: row[key] for key in ('acquisition_time_s', 'read_time_s', 'available_at_s', 'age_s')}
    if digit is None:
        return row, clocks, None
    matches = [d for d in row['digits'] if d['digit'] == digit]
    if len(matches) != 1:
        return None, clocks, 'digit_mask_missing_or_duplicated'
    record = matches[0]
    if any(not finite(record[key]) or abs(record[key] - row[key]) > CLOCK_TOL
           for key in ('acquisition_time_s', 'read_time_s', 'age_s')):
        return None, clocks, 'digit_acquisition_clock_mismatch'
    return {**record, 'external_rows_available': row['row_keys'].count([digit, 0]) == row['row_keys'].count([digit, 1]) == 1}, clocks, None


def force_label(branch, k, digit):
    row, clocks, reason = get_label(branch, RECEIVER if digit is None else OBS, k, digit)
    if row is None:
        return dict(value=None, reason=reason, clocks=clocks)
    valid_key, value_key = ('valid', 'normal_load_N') if digit is None else ('sensor_valid', 'observed_normal_N')
    value = row[value_key]
    valid = row[valid_key] is True and finite(value) and value >= 0
    result = dict(value=value if valid else None, reason=None if valid else 'force_measurement_invalid_or_unknown',
                  clocks=clocks, sensor_valid=row[valid_key])
    if digit is not None:
        result.update(contact_present=row['contact_present'], distributed_valid=row['distributed_valid'])
    else:
        result.update(normal_uncertainty_N=row['normal_uncertainty_N'], invalid_reason=row['invalid_reason'],
                      sampling_phase=row['sampling_phase'], phase=row['phase'])
    return result


def motion_label(branch, k, digit, kind, component):
    row, clocks, reason = get_label(branch, OBS, k, digit)
    if row is None:
        return dict(value=None, reason=reason, clocks=clocks)
    field, basis_key = (('raw_relative_slip2_m_s', 'basis_palm2x3') if kind == 'relative_slip'
                        else ('external_projection2_m_s', 'projection_world2x3'))
    valid = row['instrument_valid'] is True and row['sensor_valid'] is True
    if kind == 'external_projection':
        valid = valid and row['external_rows_available']
    try:
        basis = array(row[basis_key], (2, 3))
        values = array(row[field], (2,))
        valid = valid and bool(np.allclose(basis @ basis.T, np.eye(2), atol=1e-8, rtol=0.))
    except ValueError:
        valid = False
    return dict(value=float(values[component]) if valid else None, clocks=clocks,
                reason=None if valid else 'device_row_or_basis_unavailable', sensor_valid=row['sensor_valid'],
                instrument_valid=row['instrument_valid'], row_key=[digit, component],
                basis_palm2x3=row['basis_palm2x3'], projection_world2x3=row['projection_world2x3'],
                external_row_mask=row['external_rows_available'])


def predictions(values, reason, force):
    mix = values['mixed_holdout']
    output = {}
    for name, required in (('zero', ('zero',)), ('lift', ('lift_plus',)), ('finger', ('finger_single',)),
                           ('additive', ('lift_plus', 'finger_single', 'zero'))):
        missing = [arm for arm in ('mixed_holdout', *required) if values[arm] is None]
        why = reason or ('missing_labels:' + ','.join(missing) if missing else None)
        predicted = None if why else (values['lift_plus'] + values['finger_single'] - values['zero'] if name == 'additive' else values[required[0]])
        error = None if why else mix - predicted
        output[name] = dict(prediction=predicted, signed_error=error, absolute_error=None if error is None else abs(error),
                            unavailable_reason=why)
        if force:
            output[name]['within_engineering_scale'] = None if error is None else abs(error) <= FORCE_SCALE
    return output


def channel(labels, dose, common_reason, force):
    values = {arm: record['value'] for arm, record in labels.items()}
    reason = common_reason or dose['full_dose_unavailable_reason']
    primary = predictions(values, reason, force)
    exploratory = None
    if common_reason is None and dose['matched_clipped_exploration']:
        exploratory = predictions(values, None, force)
    effects = {arm: None if reason or values['zero'] is None or values[arm] is None else values[arm] - values['zero']
               for arm in PREDICTOR_ARMS[1:]}
    result = dict(labels=labels, errors=primary, matched_clipped_errors=exploratory, effects_vs_zero=effects,
                  common_predictor_support=all(row['signed_error'] is not None for row in primary.values()))
    if force:
        result['effect_exceeds_engineering_scale'] = {arm: None if v is None else abs(v) > FORCE_SCALE for arm, v in effects.items()}
    return result


def coverage(rows):
    return dict(channels=len(rows), common_predictor_support=sum(row['common_predictor_support'] for row in rows),
                available_by_predictor={p: sum(row['errors'][p]['signed_error'] is not None for row in rows)
                                        for p in ('zero', 'lift', 'finger', 'additive')},
                available_measurements_by_arm={a: sum(row['labels'][a]['value'] is not None for row in rows) for a in ARMS})


def collect(run, registration_path=None):
    marker = run / 'execution_complete.json'
    if not marker.exists():
        return dict(schema='v26_five_handover_readout_v1', status='AWAITING_TERMINAL', assigned=list(ARMS),
                    outcomes_read=False, horizons=[])
    complete, plan, axes, reg_path, spec, origin = validate_run(run, registration_path)
    branches, programs = {}, {}
    for arm, assignment in zip(ARMS, complete['rows']):
        if assignment['status'] != 'short_window_recorded':
            branches[arm] = dict(available=False, reason='terminal_arm_status:' + assignment['status'])
            continue
        try:
            branches[arm] = scan_branch(run / 'branches' / arm, assignment, plan, arm)
            programs[arm] = declared_target(branches[arm], arm)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            branches[arm] = dict(available=False, reason='branch_evidence_failed:' + str(exc))
    reproduction = verify_zeros(branches)
    available = all(b['available'] for b in branches.values())
    shared_prefix = available and all(
        b['fork'] == branches['zero']['fork'] and all(b['evidence'][name]['prefix_sha256'] == branches['zero']['evidence'][name]['prefix_sha256']
                                                    for name in (OBS, REC, INT, RECEIVER))
        and same_bits(programs[arm]['carrier'], programs['zero']['carrier'])
        and all(b['meta'].get(key) == branches['zero']['meta'].get(key) for key in ('config', 'noise', 'limits', 'budget', 'gains'))
        for arm, b in branches.items())
    reason = ('zero_reproduction_failed_or_unavailable' if not reproduction['passed'] else
              'common_origin_or_prefix_unavailable' if not shared_prefix else None)
    history = dose_history(branches, programs, axes)
    horizons = []
    for h in HORIZONS:
        k = tick(ORIGIN) + round(h / DT)
        dose = dose_at(history, h)
        pressure = []
        for digit in range(5):
            labels = {arm: force_label(b, k, digit) for arm, b in branches.items()}
            pressure.append(dict(digit=digit, **channel(labels, dose, reason, True)))
        receiver = channel({arm: force_label(b, k, None) for arm, b in branches.items()}, dose, reason, True)
        motion = []
        for kind in ('relative_slip', 'external_projection'):
            for digit in range(5):
                for component in range(2):
                    labels = {arm: motion_label(b, k, digit, kind, component) for arm, b in branches.items()}
                    basis = [labels[a].get('basis_palm2x3') for a in PREDICTOR_ARMS]
                    world = [labels[a].get('projection_world2x3') for a in PREDICTOR_ARMS]
                    motion.append(dict(kind=kind, digit=digit, component=component, coordinate_scope='device_named_row',
                                       basis_equal=all(x is not None for x in basis) and all(same_bits(basis[0], x) for x in basis[1:]),
                                       world_projection_equal=all(x is not None for x in world) and all(same_bits(world[0], x) for x in world[1:]),
                                       **channel(labels, dose, reason, False)))
        horizons.append(dict(horizon_s=h, primary=h in MAIN_HORIZONS, acquisition_time_s=k * DT,
                             expected_public_read_s=(k + round(DELAY / DT)) * DT, dose=dose,
                             pressure_N=pressure, receiver_normal_N=receiver, device_motion_m_s=motion,
                             coverage=dict(pressure=coverage(pressure), receiver=coverage([receiver]), motion=coverage(motion))))
    return dict(schema='v26_five_handover_readout_v1', status='TERMINAL_READ', outcomes_read=True,
                assigned=list(ARMS), assignment_ledger=complete['rows'], runtime_failures=complete['runtime_failures'],
                execution_complete=complete['execution_complete'], source_count=len(plan['source_sha256']),
                origin_public=origin, shared_origin_and_prefix_verified=shared_prefix, zero_reproduction=reproduction,
                physical_seed_clusters=1, analysis_preregistration=spec,
                branch_evidence={a: b['evidence'] if b['available'] else dict(reason=b['reason']) for a, b in branches.items()},
                registered_programs={a: dict(components=p['parts'], exposure=p['exposure'], target21=p['target'].tolist()) for a, p in programs.items()},
                actual_hold_history=history, horizons=horizons, input_sha256=dict(execution_complete=sha(marker),
                    registration=sha(reg_path), reader=sha(__file__)), physics_steps=0, parameters_fitted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--registration')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run = in_root(ROOT, args.run)
    output = in_root(ROOT, args.out)
    require(run.is_relative_to(ROOT / 'output') and run != ROOT / 'output', 'expected_registered_output_folder')
    require(not output.exists(), 'fresh_readout_path_required')
    try:
        result = collect(run, args.registration)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        result = dict(schema='v26_five_handover_readout_v1', status='EVIDENCE_INTEGRITY_FAILED',
                      assigned=list(ARMS), error=str(exc), physics_steps=0)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(output), sha256=sha(output), status=result['status']), ensure_ascii=False))
    return 0 if result['status'] == 'TERMINAL_READ' else 1


if __name__ == '__main__':
    raise SystemExit(main())

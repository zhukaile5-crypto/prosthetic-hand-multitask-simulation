"""Fixed five-fork experiment using the frozen V25 single-owner lifecycle."""
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import traceback

import numpy as np

from v13_a1.codec import REFERENCE_AXES21, Reference21, finite_array
from v14_coord.design import reference_rates
from v25_response_identification import run_identification as parent
from v25_response_identification.design import CONTROL_PERIOD_S, PROBE_HOLD_S, AMPLITUDE_FRACTION
from v25_response_identification.forks import _integration_digest, fork_session
from v25_response_identification.runtime import ProbeCoordinator, plain, source_files as parent_files

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'identification/v26_small_reference_response/run_identification.py'
SCHEMA = 'v26_five_handover_v1'
ASSIGNMENT = dict(mode='identify', method='ratio_effort', scenario='nominal', seed=619741,
                  motion_mode='transport', split='robustness', origin_trigger='receiver_transfer')
ORIGIN_S = 13.940
ORIGIN_READ_S = 13.935
LIFT_AXIS = REFERENCE_AXES21.index('lift_reference_m')
FINGER_AXIS = REFERENCE_AXES21.index('rh_A_FFJ0')
COMPONENTS = {'zero': (), 'zero_repeat': (), 'lift_plus': ((LIFT_AXIS, 1),),
              'finger_single': ((FINGER_AXIS, -1),), 'mixed_holdout': ((LIFT_AXIS, 1), (FINGER_AXIS, -1))}
ARMS = tuple(COMPONENTS)
HORIZONS_S = (.005, .010, .020, .040, .100, .200)
RECEIVER_STREAM = 'response_receiver.jsonl.gz'
EPS = 1e-9


def protocol():
    return dict(schema=SCHEMA, arms=list(ARMS), reference_axes21=list(REFERENCE_AXES21),
                components={arm: [dict(axis=axis, name=REFERENCE_AXES21[axis], sign=sign)
                                  for axis, sign in parts] for arm, parts in COMPONENTS.items()},
                origin_s=ORIGIN_S, trigger_read_time_s=ORIGIN_READ_S, hold_s=PROBE_HOLD_S,
                amplitude_fraction=AMPLITUDE_FRACTION, acquisition_horizons_s=list(HORIZONS_S),
                delay_s=.010, extra_flush_s=CONTROL_PERIOD_S, zero_repeat_independent_fork=True,
                unexposed_uses_zero_carrier=True, mixed_used_for_training=False)


def source_files():
    return {**parent_files(), SOURCE: ROOT / SOURCE}


def source_identity():
    return {name: parent.sha(path) for name, path in source_files().items()}


@dataclass(frozen=True)
class Program:
    arm: str
    origin_s: float = ORIGIN_S
    hold_s: float = PROBE_HOLD_S

    def __post_init__(self):
        if self.arm not in COMPONENTS or self.origin_s != ORIGIN_S or self.hold_s != PROBE_HOLD_S:
            raise ValueError('Only the preregistered five programs and fixed origin are permitted')


class SmallProbeCoordinator(ProbeCoordinator):
    """Change only the proposal before the frozen lifecycle's single commit."""

    def __init__(self, nominal, spec):
        self.nominal_supervisor = nominal
        self.spec = spec
        self.records = []
        self.carrier21 = self.target21 = self.origin_generation = self.terminal_status = None
        self.component_support = ()
        self.origin_exposure_status = None

    def _set_origin(self, proposed, lower, upper, generation):
        self.carrier21 = proposed.values.copy()
        self.target21 = self.carrier21.copy()
        self.origin_generation = generation
        step21 = reference_rates('release') * CONTROL_PERIOD_S
        parts = []
        for axis, sign in COMPONENTS[self.spec.arm]:
            carrier = self.carrier21[axis]
            margin = max(0., float(upper[axis]-carrier if sign > 0 else carrier-lower[axis]))
            amplitude = float(AMPLITUDE_FRACTION * min(step21[axis], margin))
            requested = carrier + sign*amplitude
            supported = amplitude > 0 and requested != carrier
            if supported:
                self.target21[axis] = requested
            parts.append(dict(axis=axis, name=REFERENCE_AXES21[axis], sign=sign,
                              origin_direction_margin=margin, raw_amplitude=amplitude,
                              target_increment=float(self.target21[axis]-carrier), supported=bool(supported),
                              reason=None if supported else 'ZERO_DIRECTION_AUTHORITY_OR_REFERENCE_ROUNDING'))
        self.component_support = tuple(parts)
        supported_count = sum(p['supported'] for p in parts)
        if not parts:
            self.origin_exposure_status = 'ZERO_CONTROL'
        elif supported_count == 0:
            self.origin_exposure_status = 'UNEXPOSED'
        elif supported_count == len(parts):
            self.origin_exposure_status = 'FULLY_EXPOSED'
        else:
            self.origin_exposure_status = 'PARTIAL_UNSUPPORTED'

    def action(self, **kwargs):
        proposed, debug = self.nominal_supervisor.action(**kwargs)
        now = float(kwargs['frame'].time)
        nominal = self.nominal_supervisor.decision_evidence[-1]
        lower = finite_array(nominal['actual_reference_lower21'], (21,), 'actual TASK lower')
        upper = finite_array(nominal['actual_reference_upper21'], (21,), 'actual TASK upper')
        if np.any(lower > upper):
            raise ValueError('Empty reference authority')
        generation = int(kwargs['generation'])
        if self.carrier21 is None:
            if abs(now-self.spec.origin_s) > EPS:
                raise ValueError('First program action must use the registered origin')
            self._set_origin(proposed, lower, upper, generation)
        if self.terminal_status is None:
            if generation != self.origin_generation:
                self.terminal_status = 'CANCELLED_MODE_GENERATION_CHANGED'
            elif now >= self.spec.origin_s+self.spec.hold_s-EPS:
                self.terminal_status = 'PROBE_WINDOW_COMPLETED_'+self.origin_exposure_status
        holding = self.terminal_status is None
        values = np.clip(self.target21, lower, upper) if holding else proposed.values.copy()
        changed = values.tobytes() != proposed.values.tobytes()
        certified = nominal.get('geometry_proposal_accepted') is True and (nominal.get('emitted_hierarchy_score') or {}).get('available') is True
        is_zero = not COMPONENTS[self.spec.arm]
        row = dict(time_s=now, arm=self.spec.arm, role='zero_control' if is_zero else
                   'mixed_holdout' if self.spec.arm=='mixed_holdout' else 'single_axis_training',
                   origin_s=self.spec.origin_s, active=holding and self.origin_exposure_status!='UNEXPOSED',
                   carrier_hold_active=holding, status=self.terminal_status or 'CARRIER_HOLD_'+self.origin_exposure_status,
                   origin_exposure_status=self.origin_exposure_status, component_support=plain(self.component_support),
                   origin_generation=self.origin_generation, current_generation=generation,
                   carrier_at_origin21=self.carrier21.tolist(), raw_probe_target21=self.target21.tolist(),
                   raw_probe_increment21=(self.target21-self.carrier21).tolist(),
                   nominal_proposal_reference21=proposed.values.tolist(), action_emitted_reference21=values.tolist(),
                   actual_TASK_lower21=lower.tolist(), actual_TASK_upper21=upper.tolist(),
                   requested_target_clipped_by_fresh_TASK=holding and not np.array_equal(values,self.target21),
                   nominal_reference_unchanged=not changed, nominal_task_certificate_available=bool(certified),
                   nominal_task_certificate_applies_to_emitted_reference=bool(certified and not changed),
                   probe_future_response_prediction_available=False, actual_receipt=None)
        self.records.append(row)
        result = dict(debug)
        result['response_probe'] = {key: val for key, val in row.items()
                                    if key not in ('actual_TASK_lower21', 'actual_TASK_upper21')}
        if is_zero:
            result['response_probe']['arm'] = 'zero'
        return Reference21(values, now), result


def arm_program(session, program):
    graph = session.v16_supervisor
    if abs(session.time-program.origin_s) > EPS or abs(session.sensors._last_time-(session.time-CONTROL_PERIOD_S)) > EPS:
        raise ValueError('Program requires the registered PRE_OBSERVE boundary')
    if graph is None or session.plant.joint_release_entry is None or session.probe_coordinator is not None:
        raise ValueError('Program requires one bound release lifecycle and a fresh fork')
    if session.release_driver is not graph or session.plant.v16_supervisor is not graph:
        raise ValueError('Frozen lifecycle owner differs')
    session.probe_coordinator = SmallProbeCoordinator(graph, program)


def receiver_row(session):
    frame = session.last_obs
    receiver = frame.receiver_surface
    read_time = float(frame.time)
    if abs(float(receiver['time_s'])-read_time) > EPS:
        raise ValueError('Receiver delivery clock differs from the already-read public frame')
    acquisition = receiver['acquisition_time_s']
    if acquisition is not None and acquisition > read_time+EPS:
        raise ValueError('Receiver acquisition cannot follow delivery')
    return dict(read_time_s=read_time, available_at_s=read_time, acquisition_time_s=acquisition,
                age_s=receiver['age_s'], valid=receiver['valid'], normal_load_N=receiver['normal_load_N'],
                normal_uncertainty_N=receiver['uncertainty6'][2], invalid_reason=receiver['invalid_reason'],
                sampling_phase=receiver['sampling_phase'], phase=session.v16_supervisor.public_state.get('phase'))


def save_receiver(destination, rows):
    path = destination / RECEIVER_STREAM
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    manifest_path = destination / 'response_manifest.json'
    manifest = parent.read(manifest_path)
    manifest['streams'][RECEIVER_STREAM] = dict(records=len(rows), sha256=parent.sha(path),
                                                source='existing_public_frame_no_additional_sample')
    parent.write(manifest_path, manifest)


def branch_actual_rows(branch, receiver):
    records = branch.probe_coordinator.records
    flush_ticks = round((PROBE_HOLD_S+.010+CONTROL_PERIOD_S)/CONTROL_PERIOD_S)
    expected = [(round(ORIGIN_S/CONTROL_PERIOD_S)+i)*CONTROL_PERIOD_S for i in range(flush_ticks)]
    receipts = [row for row in branch.response_observer.receipts() if row.issue_time_s >= ORIGIN_S-EPS]
    if not len(records) == len(receipts) == len(receiver) == flush_ticks:
        raise ValueError('Incomplete program, executed receipt or receiver window')
    actual = []
    for stamp, decision, event, measurement in zip(expected, records, receipts, receiver):
        if any(abs(t-stamp)>EPS for t in (decision['time_s'], event.issue_time_s, measurement['read_time_s'])):
            raise ValueError('Program/receipt/receiver tick disagreement')
        recorded = decision['actual_receipt']
        executed = event.executed_reference21
        if (recorded is None or not np.array_equal(executed, recorded['actual_reference21'])
                or not np.array_equal(event.executed_engine21, recorded['actual_engine21'])):
            raise ValueError('Decision log does not match the sole actual observer receipt')
        actual.append(executed.tolist())
    return actual


def execute_branches(base, out, expected_digest, failures):
    rows = []
    zero_carrier = zero_actual = None
    required_end = ORIGIN_S + PROBE_HOLD_S + .010 + CONTROL_PERIOD_S
    for arm in ARMS:
        print(json.dumps(dict(event='branch_start', arm=arm)), flush=True)
        branch = None
        destination = out/'branches'/arm
        try:
            branch, fork_meta = fork_session(base)
            if branch is base or fork_meta['opaque_integration_sha256'] != expected_digest:
                raise ValueError('Independent fork does not match the verified checkpoint')
            if abs(float(branch.sensor_characteristics['tactile_delay_s'])-.010) > EPS:
                raise ValueError('Registered tactile delay differs')
            arm_program(branch, Program(arm))
            receiver = []
            while branch.running and branch.time < required_end-EPS:
                branch.step_one_sensor_tick()
                receiver.append(receiver_row(branch))
            branch.save(destination)
            parent.write(destination/'opaque_fork_metadata.json', fork_meta)
            save_receiver(destination, receiver)
            if abs(branch.time-required_end)>EPS or _integration_digest(base)!=expected_digest:
                raise ValueError('Window endpoint or donor integration checkpoint differs')
            actual = branch_actual_rows(branch, receiver)
            probe = branch.probe_coordinator
            if arm=='zero':
                zero_carrier, zero_actual = probe.carrier21.copy(), actual
            elif zero_carrier is not None and not np.array_equal(probe.carrier21, zero_carrier):
                raise ValueError('Nominal origin carrier differs across forks')
            hold_ticks = round(PROBE_HOLD_S/CONTROL_PERIOD_S)
            if (zero_actual is not None and probe.origin_exposure_status=='UNEXPOSED'
                    and not np.array_equal(actual[:hold_ticks], zero_actual[:hold_ticks])):
                raise ValueError('Unexposed arm diverged from the actual zero carrier during hold')
            components = []
            for component in probe.component_support:
                axis = component['axis']
                delta = None if zero_actual is None else actual[0][axis]-zero_actual[0][axis]
                exposed = component['supported'] and delta is not None and delta*component['sign']>0
                components.append(dict(**component, actual_initial_delta_from_zero=delta,
                                       actual_initial_exposed=bool(exposed)))
            rows.append(dict(key=arm, status='short_window_recorded', reference_origin_s=ORIGIN_S,
                             terminal_probe_status=probe.terminal_status, origin_exposure_status=probe.origin_exposure_status,
                             recorded_until_s=branch.time, requested_last_read_s=required_end,
                             full_requested_window_recorded=True, actual_receipt_required=True,
                             zero_control=not COMPONENTS[arm], exposed=any(c['actual_initial_exposed'] for c in components),
                             actual_origin_component_support=components,
                             original_full16_task_metrics_available=False, response_fit_or_model_admission=False))
        except Exception as exc:
            failure = dict(key=arm, error=repr(exc), traceback=traceback.format_exc())
            failures.append(failure)
            rows.append(dict(key=arm, status='runtime_failed'))
            try:
                partial = None if (destination/'trace.npz').exists() else branch
                parent.retain_failure(destination, partial, failure)
            except Exception as save_error:
                failure['retention_error'] = repr(save_error)
        print(json.dumps(dict(event='branch_finished', arm=arm, status=rows[-1]['status'],
                              origin_exposure_status=rows[-1].get('origin_exposure_status'))), flush=True)
    return rows


def _zero_probe_payload(path, arm):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.pop('arm') != arm:
                raise ValueError('Incorrect zero branch label')
            yield row


def compare_zeros(out):
    zero, repeated = (out/'branches'/arm for arm in ARMS[:2])
    trace = parent.compare_trace(repeated/'trace.npz', zero/'trace.npz')
    public = parent.compare_public_logs(zero, repeated)
    receiver_equal = all(a == b for a, b in zip_longest(parent._gzip_payload(zero/RECEIVER_STREAM),
                                                       parent._gzip_payload(repeated/RECEIVER_STREAM)))
    program_equal = all(a == b for a, b in zip_longest(_zero_probe_payload(zero/'probe_decisions.jsonl.gz', 'zero'),
                                                      _zero_probe_payload(repeated/'probe_decisions.jsonl.gz', 'zero_repeat')))
    exact_trace = trace['same_column_keys'] and all(trace['checks'].values())
    return dict(passed=exact_trace and public['passed'] and receiver_equal and program_equal,
                exact_short_trace=exact_trace, frozen_trace_field_checks=trace, public_logs=public,
                exact_receiver_payload=receiver_equal, exact_probe_payload_excluding_arm_label=program_equal,
                short_window_only=True)


def validate_registration(reg, identity, numeric, environment):
    if (reg['assignment'] != ASSIGNMENT or reg['identification_contract'] != protocol()
            or reg['source_sha256'] != identity or reg['numerical_identity'] != numeric
            or reg['environment'] != environment or reg['software_gate_passed'] is not True):
        raise ValueError('Fixed experiment registration differs')
    parent_identity = {key: value for key,value in identity.items() if key != SOURCE}
    pin = reg['required_bridge_gate']
    bridge_path = ROOT/pin['path']
    pins = reg['required_bridge_artifact_sha256']
    origin_path = bridge_path.parent/'origin_public.json'
    checkpoint_path = bridge_path.parent/'opaque_fork_metadata.json'
    for path in (bridge_path, origin_path, checkpoint_path):
        if path.resolve().relative_to(ROOT.resolve()).as_posix() not in pins:
            raise ValueError('Bridge evidence pin missing: '+path.name)
    for name, digest in pins.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or parent.sha(path)!=digest:
            raise ValueError('Pinned V25 bridge artifact changed: '+name)
    bridge = parent.read(bridge_path)
    if (parent.sha(bridge_path)!=pin['sha256'] or bridge.get('schema')!='v25_full_remaining_fork_bridge_v1'
            or bridge.get('passed') is not True or bridge.get('full_remaining_episode_compared') is not True
            or bridge.get('source_sha256')!=parent_identity or bridge.get('numerical_identity')!=numeric):
        raise ValueError('Matching completed V25 full fork bridge is required')
    origin = parent.read(origin_path)
    if not is_registered_origin(origin):
        raise ValueError('V25 bridge origin differs')
    checkpoint = parent.read(checkpoint_path)
    digest = checkpoint['opaque_integration_sha256']
    if checkpoint.get('passed') is not True or len(digest)!=64:
        raise ValueError('Invalid existing checkpoint evidence')
    return digest


def is_registered_origin(origin):
    return (origin.get('available') is True and origin.get('trigger')=='receiver_transfer'
            and origin.get('phase')=='TRANSFER' and origin.get('next_PRE_OBSERVE_boundary') is True
            and math.isclose(origin['time_s'], ORIGIN_S, abs_tol=EPS, rel_tol=0.)
            and math.isclose(origin['trigger_read_time_s'], ORIGIN_READ_S, abs_tol=EPS, rel_tol=0.))


def snapshot_sources(out, identity):
    files = {}
    for name, original in source_files().items():
        payload = original.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest!=identity[name]:
            raise ValueError('Source changed before snapshot: '+name)
        destination = out/'sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        files[name] = dict(original_path=str(original), snapshot_path=destination.relative_to(out).as_posix(), sha256=digest)
    parent.write(out/'source_snapshot_manifest.json', dict(schema=SCHEMA, files=files))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identify', action='store_true', required=True)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = parent.fresh_output(args.out)
    registration = (ROOT/args.registration).resolve()
    registration_sha = parent.sha(registration)
    identity, numeric, environment = source_identity(), parent.numerical_identity(), parent.environment()
    if len(identity)!=199:
        raise ValueError('Expected parent198 plus one new source')
    expected_digest = validate_registration(parent.read(registration), identity, numeric, environment)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, failures = [], []
    stage = 'source_snapshot'
    try:
        snapshot_sources(out, identity)
        parent.write(out/'plan.json', dict(schema=SCHEMA, mode='identify', assignment=ASSIGNMENT,
                    protocol=protocol(), registration=str(registration), registration_sha256=registration_sha,
                    source_sha256=identity, numerical_identity=numeric, environment=environment))
        parent.write(out/'environment.json', dict(pinned_environment=environment, numerical_identity=numeric,
                    executable=sys.executable))
        parent.write(out/'manifest.json', dict(schema=SCHEMA, source_sha256=identity,
                    numerical_identity=numeric, plan_sha256=parent.sha(out/'plan.json'),
                    source_snapshot_manifest_sha256=parent.sha(out/'source_snapshot_manifest.json'),
                    frozen_v25_source_sha256={k:v for k,v in identity.items() if k!=SOURCE}))
        stage = 'verified_public_origin'
        base = parent.construct(ASSIGNMENT)
        base.research_sources = identity.copy()
        origin = parent.advance_to_origin(base, ASSIGNMENT)
        parent.write(out/'origin_public.json', origin)
        if not is_registered_origin(origin) or _integration_digest(base)!=expected_digest:
            raise ValueError('Reconstructed origin does not match the verified V25 checkpoint')
        stage = 'five_forks'
        rows = execute_branches(base, out, expected_digest, failures)
        stage = 'zero_reproduction'
        if all(row['status']=='short_window_recorded' for row in rows[:2]):
            reproduction = compare_zeros(out)
            parent.write(out/'zero_reproduction.json', reproduction)
            if not reproduction['passed']:
                failures.append(dict(stage=stage, error='Independent zero programs differ'))
        else:
            parent.write(out/'zero_reproduction.json', dict(passed=False, reason='zero_branch_incomplete'))
    except Exception as exc:
        failures.append(dict(stage=stage, error=repr(exc), traceback=traceback.format_exc()))
    by_arm = {row['key']: row for row in rows}
    rows = [by_arm.get(arm, dict(key=arm, status='not_executed', reason=stage)) for arm in ARMS]
    try:
        sources_unchanged = source_identity()==identity and parent.sha(registration)==registration_sha
        numerical_unchanged = parent.numerical_identity()==numeric and parent.environment()==environment
    except Exception as exc:
        sources_unchanged = numerical_unchanged = False
        failures.append(dict(stage='final_identity', error=repr(exc)))
    completed = sum(row['status']=='short_window_recorded' for row in rows)
    parent.write(out/'assignment_ledger.json', rows)
    parent.write(out/'runtime_failures.json', failures)
    passed = completed==len(ARMS) and not failures and sources_unchanged and numerical_unchanged
    complete = dict(schema=SCHEMA, mode='identify', execution_complete=passed, passed=passed,
                    planned=len(ARMS), completed=completed, all_assigned_accounted=True, rows=rows,
                    runtime_failures=failures, sources_unchanged=sources_unchanged, numerical_unchanged=numerical_unchanged,
                    source_sha256=identity, numerical_identity=numeric, environment=environment,
                    wall_seconds=time.perf_counter()-started, identification_response_validated=False,
                    assignment_ledger_sha256=parent.sha(out/'assignment_ledger.json'))
    for name in ('plan', 'manifest', 'zero_reproduction'):
        path = out/(name+'.json')
        complete[name+'_sha256'] = parent.sha(path) if path.exists() else None
    parent.write(out/'execution_complete.json', complete)
    print(json.dumps(dict(event='execution_complete', passed=passed, completed=completed, planned=len(ARMS),
                          failures=len(failures))), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

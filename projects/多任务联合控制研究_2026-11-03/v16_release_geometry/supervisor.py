"""One experimental release process: transfer, OPEN/clearance, then monitoring.

No truth, writer, force prior, simulator or object-state input. Holding is
preempted immediately. Low pressure and aggregate receiver evidence remain
explicit sampled proxies; no state in this graph means certified placement.
"""
from copy import deepcopy
import time
import numpy as np
from v13_a1.codec import Reference21, finite_array
from v13_joint.lifecycle import ReleaseEntry
from v13_joint.design import NONTHUMB_CLEAR_TIME_S, LIFT_DESCENT_TIME_S
from v14_coord.design import DECISION_PERIOD_S, CONTROL_PERIOD_S, reference_rates
from v15_support.release_tasks import HAND_CLEARANCE_N, HAND_DWELL_S
from .transfer_observer import TransferObserver
from .hierarchy import solve_geometry_release


CLOCK_TOL = 1e-9
RELEASE_DEADLINE_S = 3.5


class ReleaseSupervisor:
    def __init__(self, entry, *, open18, observer: TransferObserver):
        if type(entry) is not ReleaseEntry or type(observer) is not TransferObserver:
            raise TypeError('Actual typed release entry and public transfer observer required')
        if abs(entry.started_s-observer.request) > CLOCK_TOL:
            raise ValueError('Release and observer origin mismatch')
        self.entry = entry
        self.open18 = finite_array(open18, (18,), 'public open reference18').copy()
        self.observer = observer
        self.phase = 'APPROACH_AND_UNLOAD'; self.version = 0
        self.contact_lift_reference = None
        self.retreat_start_s = None; self.retreat_start_lift = None
        self.last_time = None; self.last_acquisition = None; self.receiver_streak = 0
        self.low_since = None; self.was_clear = False; self.low_current = False
        self.last_transfer = None; self.public_state = None
        self.last_decision = -np.inf; self.plan_start = None; self.plan_current = None; self.plan_delta = None
        self.plan_version = None; self.debug = {}; self.decision_evidence = []
        self.plan_generation = None

    def _phase(self, phase):
        if phase != self.phase:
            self.phase = phase; self.version += 1

    def observe(self, *, frame, transfer_inputs, current21):
        now = float(frame.time)
        # Typed observer enforces grid, payload identity, freshness and gaps.
        evidence = self.observer.update(**transfer_inputs)
        if self.last_time is not None and abs(now-self.last_time) <= CLOCK_TOL:
            return self.public_state
        if self.last_time is not None and now < self.last_time:
            raise ValueError('Release public clock rewind')
        current = finite_array(current21, (21,), 'current authoritative reference')
        receiver = transfer_inputs['receiver']; hand = transfer_inputs['hand']
        eligible = bool(receiver.valid and hand.valid and
            receiver.acquisition_time_s is not None and hand.acquisition_time_s is not None and
            abs(receiver.time_s-now) <= CLOCK_TOL and abs(hand.read_time_s-now) <= CLOCK_TOL and
            abs(receiver.acquisition_time_s-hand.acquisition_time_s) <= CLOCK_TOL and
            abs(now-receiver.acquisition_time_s-.010) <= CLOCK_TOL and
            receiver.acquisition_time_s >= self.entry.started_s-CLOCK_TOL)
        acq = receiver.acquisition_time_s if eligible else None
        adjacent = bool(eligible and self.last_acquisition is not None and
            abs(acq-self.last_acquisition-CONTROL_PERIOD_S) <= CLOCK_TOL and
            self.last_time is not None and abs(now-self.last_time-CONTROL_PERIOD_S) <= CLOCK_TOL)
        compressed = bool(eligible and receiver.normal_load_N > receiver.uncertainty6[2])
        self.receiver_streak = (self.receiver_streak+1 if adjacent else 1) if compressed else 0
        if self.contact_lift_reference is None and self.receiver_streak >= 2:
            self.contact_lift_reference = float(current[18])
            self._phase('TRANSFER_AND_CLEAR')
        n = np.asarray(frame.normal, float); valid = np.asarray(frame.valid, bool)
        pressure_available = bool(eligible and n.shape == (5,) and valid.shape == (5,) and valid.all() and
            np.isfinite(n).all() and frame.auxiliary_valid and np.isfinite(frame.auxiliary_pressure) and
            abs(now-float(frame.auxiliary_age)-acq) <= CLOCK_TOL)
        pressure = float(np.maximum(n, 0).sum()+max(0., float(frame.auxiliary_pressure))) if pressure_available else None
        self.low_current = bool(pressure_available and pressure <= HAND_CLEARANCE_N)
        if self.low_current:
            if not adjacent or self.low_since is None:
                self.low_since = max(acq, self.entry.started_s)
        else:
            self.low_since = None
        low_confirmed = bool(self.low_since is not None and acq-self.low_since >= HAND_DWELL_S-CLOCK_TOL)
        self.was_clear = self.was_clear or low_confirmed
        payload = evidence.continuous_support_evidence
        coordinated = evidence.stage_permission == 'EXPERIMENTAL_COORDINATION_EVIDENCE'
        if (self.retreat_start_s is None and low_confirmed and coordinated and
                self.contact_lift_reference is not None):
            self.retreat_start_s = now; self.retreat_start_lift = float(current[18])
            self._phase('RETREAT_AND_MONITOR')
        if self.retreat_start_s is not None and now-self.retreat_start_s >= LIFT_DESCENT_TIME_S-CLOCK_TOL:
            self._phase('MONITOR')
        deadline = now-self.entry.started_s >= RELEASE_DEADLINE_S-CLOCK_TOL
        self.last_transfer = evidence.to_public()
        self.public_state = dict(schema='v16_experimental_release_process_v1',time_s=now,
            phase=self.phase,version=self.version,started_s=self.entry.started_s,
            holding_preempted=True,holding_force_tasks_active=False,
            first_receiver_event_latched=self.contact_lift_reference is not None,
            receiver_current_compression_resolved=compressed,
            receiver_fresh_streak=self.receiver_streak,
            low_hand_pressure_N=pressure,low_hand_pressure_confirmed=low_confirmed,
            sampled_hand_low_was_observed=self.was_clear,
            geometric_full_hand_clearance_certified=False,
            transfer=evidence.to_public(),
            retrospective_cop_motion_proxy=(payload.get('cop_trend') or {}).get('motion_resolved_against_engineering_proxy'),
            coordinated_retreat_started=self.retreat_start_s is not None,
            deadline_reached=deadline,deadline_without_coordinated_retreat=deadline and self.retreat_start_s is None,
            delivery_confirmed=False,placement_certified=False,
            no_regrip_after_release=True,release_request_delayed=False)
        self.last_time = now; self.last_acquisition = acq
        return deepcopy(self.public_state)

    def lift_interval(self, current_reference):
        current = current_reference.values
        if self.contact_lift_reference is not None and self.retreat_start_s is None:
            return float(current[18]), float(current[18])
        # A positive retreat is never requested. Receiver loss does not resume
        # downward approach/regrasp; it stays an unconfirmed monitored process.
        return 0., float(current[18])

    def nominal(self, query_time_s, lower21, upper21):
        now = float(query_time_s); elapsed = now-self.entry.started_s
        if elapsed < -CLOCK_TOL:
            raise ValueError('Release query before request')
        start = self.entry.reference21.values
        value = start.copy()
        fraction = float(np.clip(elapsed/NONTHUMB_CLEAR_TIME_S, 0., 1.))
        # One direct OPEN task for all digits replaces the special thumb tuck.
        value[:18] = start[:18]+fraction*(self.open18-start[:18])
        if self.retreat_start_s is not None:
            value[18] = self.retreat_start_lift*(1.-np.clip((now-self.retreat_start_s)/LIFT_DESCENT_TIME_S, 0., 1.))
        elif self.contact_lift_reference is not None:
            value[18] = self.contact_lift_reference
        else:
            value[18] = start[18]*(1.-np.clip(elapsed/LIFT_DESCENT_TIME_S, 0., 1.))
        lo = finite_array(lower21, (21,), 'actual lower'); hi = finite_array(upper21, (21,), 'actual upper')
        if np.any(lo > hi):
            raise ValueError('Empty release proposal permissions')
        return Reference21(np.clip(value, lo, hi), now)

    def action(self, *, frame, adapter, current21, lower21, upper21, generation):
        now = float(frame.time); current = finite_array(current21, (21,), 'current reference')
        lower = finite_array(lower21, (21,), 'actual lower').copy(); upper = finite_array(upper21, (21,), 'actual upper').copy()
        lift_lo, lift_hi = self.lift_interval(Reference21(current,now))
        lower[18] = max(lower[18],lift_lo); upper[18] = min(upper[18],lift_hi)
        if np.any(lower > upper):
            raise ValueError('Release process and actual authority do not intersect')
        rates = reference_rates('release')
        if self.plan_version != self.version or self.plan_generation != generation:
            self.last_decision = -np.inf
        if now >= self.last_decision+DECISION_PERIOD_S-CLOCK_TOL:
            began = time.perf_counter_ns()
            nominal = self.nominal(now+DECISION_PERIOD_S, lower, upper).values
            step_lo = np.maximum(lower-current, -rates*DECISION_PERIOD_S)
            step_hi = np.minimum(upper-current, rates*DECISION_PERIOD_S)
            if np.any(step_lo > step_hi):
                raise ValueError('Empty actual20ms release domain')
            # Declared OPEN task does not overshoot its endpoint or reverse any
            # finger toward the old grasp. This is narrower than real authority.
            open_delta = self.open18-current[:18]
            step_lo[:18] = np.maximum(step_lo[:18], np.minimum(0., open_delta))
            step_hi[:18] = np.minimum(step_hi[:18], np.maximum(0., open_delta))
            wanted = np.clip(nominal-current, step_lo, step_hi)
            tasks = adapter.current_geometry_tasks(frame,current,wanted+current,step_lo,step_hi)
            solved = None; error = None
            try:
                solved = solve_geometry_release(tasks, reference_step_scale21=rates*DECISION_PERIOD_S)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                error = type(exc).__name__+': '+str(exc)
            accepted = solved is not None and solved.available
            delta = solved.x*(rates*DECISION_PERIOD_S) if accepted else wanted
            self.plan_start = now; self.plan_current = current.copy(); self.plan_delta = delta.copy()
            self.plan_version = self.version; self.last_decision = now
            self.plan_generation = generation
            evidence = dict(decision_id=len(self.decision_evidence),time_s=now,phase=self.phase,
                mode_generation=int(generation),process_version=self.version,
                current_reference21=current.tolist(),nominal_reference21=nominal.tolist(),
                actual_step_lower21=step_lo.tolist(),actual_step_upper21=step_hi.tolist(),
                scale21=(rates*DECISION_PERIOD_S).tolist(),
                geometry_status=tasks.status,geometry=deepcopy(tasks.diagnostics),
                normal_rows= None if tasks.normal_matrix_m_per_reference is None else tasks.normal_matrix_m_per_reference.tolist(),
                tangent_rows=None if tasks.tangent_matrix_m_per_reference is None else tasks.tangent_matrix_m_per_reference.tolist(),
                hierarchy_status=None if solved is None else solved.status,
                hierarchy_levels=[] if solved is None else list(solved.levels),
                hierarchy_diagnostics=None if solved is None else solved.diagnostics,
                proposed_reference_increment21=delta.tolist(),
                geometry_proposal_accepted=accepted, numerical_error=error,
                fallback=None if accepted else 'legal_direct_OPEN_nominal_without_geometry_certificate',
                priority_order=['actual_bounds','old_plane_noninward_proposal','OPEN_scalar_progress','low_tangent','closest_nominal'],
                public_process=deepcopy(self.public_state),
                solver_ms=(time.perf_counter_ns()-began)/1e6,
                future_pressure_prediction_available=False,actual_servo_motion_predicted=False,
                physical_safety_certified=False,placement_certified=False)
            self.decision_evidence.append(evidence)
            self.debug = dict(architecture='public_transfer_and_progress_priority_geometry_release',
                mode='release',decision_id=evidence['decision_id'],decision_time_s=now,
                geometry_status=tasks.status,geometry_proposal_accepted=accepted,
                hierarchy_status=evidence['hierarchy_status'],fallback=evidence['fallback'],
                numerical_error=error,solver_timing_ms=dict(whole_decision=evidence['solver_ms']))
        fraction = np.clip((now-self.plan_start+CONTROL_PERIOD_S)/DECISION_PERIOD_S, 0., 1.)
        proposal = self.plan_current+fraction*self.plan_delta
        lo = np.maximum(lower, current-rates*CONTROL_PERIOD_S)
        hi = np.minimum(upper, current+rates*CONTROL_PERIOD_S)
        # The task contract applies to every issued5ms command, including when
        # the real actor advanced farther than this20ms snapshot anticipated.
        open_delta = self.open18-current[:18]
        lo[:18] = np.maximum(lo[:18],current[:18]+np.minimum(0.,open_delta))
        hi[:18] = np.minimum(hi[:18],current[:18]+np.maximum(0.,open_delta))
        if np.any(lo > hi):
            raise ValueError('Empty actual5ms release domain')
        value = np.clip(proposal, lo, hi)
        self.debug.update(process=deepcopy(self.public_state),mode_generation=int(generation),
            plan_snapshot_scope='original20ms virtual geometry; no fresh intermediate optimality/physics certificate',
            holding_force_tasks_active=False,pressure_prediction_available=False)
        return Reference21(value,now), deepcopy(self.debug)

    def reconcile(self, receipt):
        """A changed actual command invalidates the unused snapshot trajectory."""
        if receipt.execution_changed or receipt.stale_plan_replaced:
            self.last_decision = -np.inf
        self.debug.update(actual_reference21=receipt.executed.values.tolist(),
            execution_changed_plan=receipt.execution_changed,
            stale_plan_replaced=receipt.stale_plan_replaced)

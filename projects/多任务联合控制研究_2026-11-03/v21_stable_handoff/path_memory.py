"""Executed exit-pose memory and a clocked, explicitly experimental lift plan.

No external pose, palm point, object identity or future collision is inferred.
Old planes can CREATE a plan only inside the original geometry age domain.
The later stored endpoint is a robot reference plan, never a current plane.
"""
from copy import deepcopy
import numpy as np
from v13_a1.codec import finite_array
from v16_release_geometry.public_adapter import _mask, _real, _tick
from v15_support.release_tasks import HAND_CLEARANCE_N, HAND_MAX_AGE_S
from .design import MAX_GEOMETRY_AGE_S


class ExitPathMemory:
    def __init__(self):
        self.last_read_s = None
        self.last_acquisition_s = None
        self.last_known_positive = False
        self.last_surface = None
        self.capture = None
        self.episode = 0
        self.last_receipt = None

    def update(self, frame, *, tasks, current21, lower21, upper21):
        _, now = _tick(frame.time)
        current = finite_array(current21, (21,), 'authoritative current reference')
        lower = finite_array(lower21, (21,), 'original lower reference')
        upper = finite_array(upper21, (21,), 'original upper reference')
        if self.last_read_s is not None and now < self.last_read_s-1e-9:
            raise ValueError('exit path public clock rewind')
        if (float(frame.robot_control_time_s) != now or
            finite_array(frame.robot_reference21, (21,), 'public reference').tobytes() != current.tobytes()):
            raise ValueError('path capture requires authoritative current reference')
        known = bool(tasks is not None and tasks.get('available'))
        positive = bool(known and len(tasks['pressure_P_N']))
        reason = None
        acq = None
        fresh = False
        known_empty = False
        try:
            acq = _real(frame.sample_time, 'hand acquisition', nonnegative=True)
            age = _real(frame.age, 'hand age', nonnegative=True)
            if not 0 <= now-acq <= MAX_GEOMETRY_AGE_S+1e-9 or abs(now-acq-age) > 1e-9:
                raise ValueError('hand acquisition/age mismatch')
            if self.last_acquisition_s is not None and acq < self.last_acquisition_s-1e-9:
                raise ValueError('hand acquisition rewind')
            fresh = self.last_acquisition_s is None or acq > self.last_acquisition_s+1e-9
            valid = _mask(frame.valid, 'hand valid') & _mask(frame.distributed_valid, 'geometry valid')
            present = _mask(frame.contact_present, 'public contact presence')
            known_empty = bool(known and not positive and valid.all() and not present.any()
                and not tasks['diagnostics'].get('partial_geometry', True))
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            reason = str(exc)
        created = False
        if known and positive and reason is None and fresh:
            patches = tasks['diagnostics']['patches']
            stamps = [_real(p['acquisition_time_s'], 'old plane acquisition', nonnegative=True) for p in patches]
            if len(stamps) != len(tasks['pressure_P_N']) or not stamps:
                raise ValueError('positive rows require same-order acquired planes')
            self.last_surface = dict(read_time_s=now, oldest_acquisition_s=min(stamps),
                Npx21=np.asarray(tasks['Npx21'], float).copy(),
                pressure_P_N=np.asarray(tasks['pressure_P_N'], float).copy())
            self.last_known_positive = True
        elif known_empty and reason is None and fresh:
            if self.last_known_positive or self.capture is None:
                self.episode += 1
                direction = None
                scope = 'no_valid_recent_finger_plane_direction'
                age_at_creation = None
                surface = self.last_surface
                if surface is not None:
                    age_at_creation = now-surface['oldest_acquisition_s']
                    if 0 <= age_at_creation <= MAX_GEOMETRY_AGE_S+1e-9:
                        column = surface['Npx21'][:, 18]
                        if np.all(column >= 0) and np.any(column > 0):
                            direction = -1
                        elif np.all(column <= 0) and np.any(column < 0):
                            direction = 1
                        scope = ('all_known_finger_planes_noninward_lift_consensus'
                            if direction is not None else 'conflicting_or_zero_finger_plane_lift_projection')
                endpoint = None if direction is None else float(lower[18] if direction < 0 else upper[18])
                self.capture = dict(episode=self.episode, read_time_s=now, empty_acquisition_s=acq,
                    actual_committed_reference21=current.copy(), lift_direction=direction,
                    lift_endpoint_m=endpoint, direction_scope=scope,
                    source_surface_read_s=None if surface is None else surface['read_time_s'],
                    source_surface_acquisition_s=None if surface is None else surface['oldest_acquisition_s'],
                    source_plane_age_at_creation_s=age_at_creation,
                    endpoint_role='declared_robot_reference_plan_not_collision_clearance_certificate')
                created = True
            self.last_known_positive = False
        if reason is None and fresh:
            self.last_acquisition_s = acq
        aux_known = False
        aux_low = False
        try:
            aux = _real(frame.auxiliary_pressure, 'auxiliary pressure', nonnegative=True)
            aux_age = _real(frame.auxiliary_age, 'auxiliary age', nonnegative=True)
            if not isinstance(frame.auxiliary_valid, (bool, np.bool_)) or not frame.auxiliary_valid:
                raise ValueError('auxiliary validity unavailable')
            aux_known = bool(aux_age <= HAND_MAX_AGE_S+1e-9 and acq is not None
                and abs(now-aux_age-acq) <= 1e-9)
            aux_low = bool(aux_known and aux <= HAND_CLEARANCE_N)
        except (ValueError, TypeError, AttributeError):
            pass
        target = None if self.capture is None else self.capture['actual_committed_reference21'].copy()
        direction = None if self.capture is None else self.capture['lift_direction']
        advance = bool(known_empty and reason is None and aux_low and direction is not None)
        if advance:
            endpoint = self.capture['lift_endpoint_m']
            advance = bool(direction*(endpoint-current[18]) > 0)
        self.last_read_s = now
        capture = None if self.capture is None else deepcopy(self.capture)
        if capture is not None:
            origin = capture['source_surface_acquisition_s']
            capture['source_plane_age_now_s'] = None if origin is None else now-origin
            capture['source_plane_current_geometry_valid'] = bool(origin is not None
                and 0 <= now-origin <= MAX_GEOMETRY_AGE_S+1e-9)
        return dict(schema='v21_exit_path_memory_v1', time_s=now, acquisition_s=acq,
            fresh_acquisition=fresh, positive_pressure_rows=positive,
            known_empty_finger_geometry=known_empty, unknown_reason=reason,
            capture_created=created, capture=capture,
            path_reference21=target if known_empty else None, lift_plan_advance=advance,
            requested_lift_direction=direction if advance else 0,
            auxiliary_pressure_known=aux_known, auxiliary_pressure_low=aux_low,
            auxiliary_local_geometry_available=False,
            captured_reference_role='authoritative_already_committed_reference_not_unexecuted_proposal',
            continued_endpoint_is_a_robot_plan_not_updated_external_geometry=True,
            reference_hold_is_not_actual_stopping=True, full_hand_clearance_certified=False,
            placement_certified=False)

    def reconcile(self, receipt):
        self.last_receipt = dict(time_s=float(receipt.executed.time_s),
            reference21=receipt.executed.values.copy(),
            mode_generation=int(receipt.mode_generation),
            execution_changed=bool(receipt.execution_changed),
            stale_plan_replaced=bool(receipt.stale_plan_replaced))

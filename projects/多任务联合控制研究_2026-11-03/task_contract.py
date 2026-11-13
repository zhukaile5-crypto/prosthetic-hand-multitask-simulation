"""Public task intent, kept separate from tactile belief and evaluation truth.

The contract contains a user request and robot command history. It contains no
object state, material parameters, target-contact labels or disturbance schedule.
``deadline_s`` is the absolute deadline of the CURRENT requested stage; callers
must explicitly pass the release completion deadline after a release request.
This module neither plans motion nor certifies contact/force feasibility.
"""
from dataclasses import asdict, dataclass
import math
from numbers import Real
from collections.abc import Mapping


SCHEMA = 'public-hand-task-request-v1'
MOTION_AXES = ('lift_m', 'WRJ2_rad', 'WRJ1_rad')
PUBLIC_RELEASE_AT_S = 12.0
PUBLIC_RELEASE_DEADLINE_S = 3.5
MODES = frozenset(('preparation', 'hold', 'transport', 'recovery', 'release', 'monitor'))
CONTACT_PROXIES = frozenset(('unknown', 'compatible', 'incipient', 'sliding', 'contact_loss'))
FORCE_SCOPE = 'external_all_instrumented_fingers'
OBSERVATION_SCOPE = 'online_contact_proxy_separate_from_offline_object_truth'


def _scalar(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value,Real):
        raise ValueError(f'{name} must be a finite number, not a boolean')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result) or (nonnegative and result < 0.):
        raise ValueError(f'{name} must be finite' + (' and nonnegative' if nonnegative else ''))
    return result


def _vector(value, size, name, *, nonnegative=False, allow_none=False):
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f'{name} must be a vector of length {size}')
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a vector of length {size}') from exc
    if len(items) != size:
        raise ValueError(f'{name} must be a vector of length {size}')
    return tuple(None if x is None and allow_none else _scalar(x, name, nonnegative=nonnegative)
                 for x in items)


def _mode(value):
    if not isinstance(value, str) or value not in MODES:
        raise ValueError(f'mode must be one of {sorted(MODES)}')
    return value


@dataclass(frozen=True)
class TaskRequest:
    """Immutable intent snapshot. ``as_dict`` returns a fresh JSON-safe copy."""
    schema: str
    time_s: float
    mode: str
    motion_axes: tuple
    motion_target: tuple
    executed_motion_reference: tuple
    motion_error: tuple
    tracking_tolerances: tuple
    limits: tuple
    budget: float
    deadline_s: float
    remaining_time_s: float
    deadline_exceeded: bool
    release_at_s: float
    release_deadline_s: float
    force_scope: str
    observation_scope: str
    model_domain_transfer: str
    object_state_available: bool
    target_contact_identity_available: bool
    task_kind: str

    def as_dict(self):
        value = asdict(self)
        for key in ('motion_axes', 'motion_target', 'executed_motion_reference',
                    'motion_error', 'tracking_tolerances', 'limits'):
            value[key] = list(value[key])
        return value


def build_task_request(time_s, mode, motion_target, executed_motion_reference,
                       deadline_s, limits, budget, tracking_tolerances=None, *,
                       motion_preview=None, motion_reference_bounds=None, sensor_characteristics=None):
    """Build a plain dictionary accepted by a controller's set_task_request.

    Motion values are [lift metres, WRJ2 radians, WRJ1 radians]. User requests
    are not clipped to a model domain: the controller must diagnose an infeasible
    request rather than redefine success by silently changing its target.

    Tracking tolerances are user-task tolerances, not contact or object-stability
    thresholds. The default .01 m lift tolerance is already in the research task;
    unregistered wrist tolerances remain None until explicitly supplied. The
    fixed release schedule is public task information, never a future disturbance.
    """
    now = _scalar(time_s, 'time_s', nonnegative=True)
    deadline = _scalar(deadline_s, 'deadline_s', nonnegative=True)
    target = _vector(motion_target, 3, 'motion_target')
    executed = _vector(executed_motion_reference, 3, 'executed_motion_reference')
    tolerances = _vector((.01, None, None) if tracking_tolerances is None else tracking_tolerances,
                         3, 'tracking_tolerances', nonnegative=True, allow_none=True)
    request = TaskRequest(SCHEMA, now, _mode(mode), MOTION_AXES, target, executed,
        tuple(t-e for t, e in zip(target, executed)), tolerances,
        _vector(limits, 5, 'limits', nonnegative=True),
        _scalar(budget, 'budget', nonnegative=True), deadline,
        max(0., deadline-now), now > deadline, PUBLIC_RELEASE_AT_S,
        PUBLIC_RELEASE_DEADLINE_S, FORCE_SCOPE, OBSERVATION_SCOPE,
        'experimental_local_model_not_admitted', False, False,
        'user_robot_motion_and_lifecycle_request')
    result=request.as_dict()
    if motion_preview is not None:
        from task_trajectory import validate_preview
        result['motion_preview']=validate_preview(motion_preview,now,target)
    if motion_reference_bounds is not None:
        if not isinstance(motion_reference_bounds,Mapping) or set(motion_reference_bounds)!={'lower','upper'}:
            raise ValueError('Motion bounds require exactly lower/upper')
        lower=_vector(motion_reference_bounds['lower'],3,'motion_reference_bounds.lower')
        upper=_vector(motion_reference_bounds['upper'],3,'motion_reference_bounds.upper')
        if any(a>b for a,b in zip(lower,upper)):
            raise ValueError('Empty actuator reference interval')
        result['motion_reference_bounds']=dict(lower=list(lower),upper=list(upper))
    if sensor_characteristics is not None:
        names={'force_sigma_N','tangential_sigma_N','position_sigma_m','slip_sigma_m_s','effort_sigma','tactile_delay_s'}
        optional={'auxiliary_force_sigma_N','auxiliary_moment_sigma_Nm'}
        if not isinstance(sensor_characteristics,Mapping) or not names<=set(sensor_characteristics) or set(sensor_characteristics)-names-optional:
            raise ValueError('Only public sensor characteristics are permitted')
        result['sensor_characteristics']={k:_scalar(sensor_characteristics[k],k,nonnegative=True) for k in sorted(sensor_characteristics)}
    if len(result)>len(TaskRequest.__dataclass_fields__):result['schema']='public-hand-task-request-v2'
    return result


def validate_task_request(value):
    """Rebuild a strict public contract, rejecting hidden fields or stale math.

    This checks the contract's shape and internal consistency, not whether a
    caller obtained a permitted number from the correct physical sensor. That
    information boundary also requires a runtime/source audit.
    """
    if not isinstance(value, Mapping):
        raise ValueError('Task request must be a mapping')
    keys = set(TaskRequest.__dataclass_fields__)
    extensions={'motion_preview','motion_reference_bounds','sensor_characteristics'}
    if not keys<=set(value) or set(value)-keys-extensions:
        raise ValueError('Task request fields do not match the public schema')
    checked = build_task_request(value['time_s'], value['mode'], value['motion_target'],
        value['executed_motion_reference'], value['deadline_s'], value['limits'],
        value['budget'], value['tracking_tolerances'],
        **{k:value[k] for k in extensions if k in value})
    for key in checked:
        actual = list(value[key]) if isinstance(checked[key], list) else value[key]
        if key=='motion_error':actual=list(_vector(actual,3,'motion_error'))
        if isinstance(checked[key],float):actual=_scalar(actual,key)
        if actual != checked[key]:
            raise ValueError(f'Inconsistent or nonpublic task field: {key}')
        if isinstance(checked[key], bool) and type(actual) is not bool:
            raise ValueError(f'{key} must be a boolean')
    return checked


def select_task_mode(command, preparation_complete, motion_target,
                     executed_motion_reference, *, online_contact_proxy='unknown',
                     release_active=False, clear_confirmed=False):
    """Select a mode from current intent and causal robot/contact status only.

    ``clear_confirmed`` means past sustained skin-force clearance, NOT geometric
    free space or future collision freedom. A recovery mode retains the same
    motion target; it is not authorization to discard the transport deadline.
    ``motion_error`` uses a numerical equality test, never a hidden task tolerance.
    No scenario name, physical object state, or future event is accepted.
    """
    if command not in ('hold', 'release'):
        raise ValueError('command must be hold or release')
    for name, value in (('preparation_complete', preparation_complete),
                        ('release_active', release_active), ('clear_confirmed', clear_confirmed)):
        if type(value) is not bool:
            raise ValueError(f'{name} must be a boolean')
    if not isinstance(online_contact_proxy, str) or online_contact_proxy not in CONTACT_PROXIES:
        raise ValueError('Unknown online contact proxy')
    target = _vector(motion_target, 3, 'motion_target')
    executed = _vector(executed_motion_reference, 3, 'executed_motion_reference')
    if command == 'release' or release_active:
        return 'monitor' if clear_confirmed else 'release'
    if not preparation_complete:
        return 'preparation'
    if online_contact_proxy in ('incipient', 'sliding', 'contact_loss'):
        return 'recovery'
    if any(abs(t-e) > 1e-9 for t, e in zip(target, executed)):
        return 'transport'
    return 'hold'

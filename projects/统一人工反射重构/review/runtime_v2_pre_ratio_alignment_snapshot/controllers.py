"""Unified local tactile predictive coordination and common-action baselines.

Only SensorFrame and task limits enter this module. No object truth, mass,
friction parameters, disturbance schedule or independent evaluation state enters.
The calibrated benchmark has symmetric five-pad force references; archived
Shadow development is kept separately and is not performance evidence.
Each upper layer selects five closure increments every 20 ms; the common 5 ms
execution interface subdivides them. The local force model is a calibration
approximation, not a physical safety certificate.
"""
from dataclasses import dataclass, asdict, fields
from collections import deque
import time
import numpy as np

METHODS = ('fixed', 'ratio', 'independent', 'unified',
           'unified_no_continuity', 'unified_no_geometry', 'unified_fixed_stage')
METHOD_TITLES = {'fixed': '固定握力', 'ratio': '文献摩擦比迁移基线',
                 'independent': '独立叠加反射', 'unified': '统一预测接触协调',
                 'unified_no_continuity': '统一协调去除连续性目标',
                 'unified_no_geometry': '统一协调去除接触几何耦合',
                 'unified_fixed_stage': '统一协调固定阶段权重'}
WEIGHTS = np.ones(5)


@dataclass(frozen=True)
class Gains:
    decision_period: float = .020
    horizon: int = 3
    force_gain: tuple = (14.25, 14.25, 14.25, 14.25, 14.25)
    force_tau: float = .005
    close_rate: float = .45
    open_rate: float = .60
    release_rate: float = .60
    alpha_max: float = 1.2
    force_margin: float = .12
    baseline_tracking_gain: float = .20
    base: float = 1.2
    acquisition_force: tuple = (2., 2., 2., 2., 2.)
    support_contact_floor: float = .30
    load_uncertainty_weight: float = 1.
    ratio_kp: float = 1.
    ratio_ki: float = .3
    ratio_floor: float = 1.8
    slip_boost_rate: float = 4.
    nominal_mu: float = .45
    friction_margin: float = .82
    force_cost: float = .06
    support_weight: float = 9.
    balance_weight: float = .6
    moment_weight: float = .35
    characteristic_length: float = .050
    contact_weight: float = .9
    force_continuity: float = .9
    action_continuity: float = .35
    slip_load_gain: float = 110.
    support_buffer: float = .20
    support_reserve_fraction: float = .50
    reference_continuity: float = 3.
    support_slack_cost: float = 120.
    observer_bias_tau: float = .10
    qp_passes: int = 120
    qp_tolerance: float = 2e-5
    sensor_tau: float = .030


@dataclass
class ControlAction:
    delta_closure: np.ndarray
    predicted_normal: np.ndarray
    status: str
    pause_transport: bool = False
    debug: dict = None

    def __array__(self, dtype=None, copy=None):
        return np.array(self.delta_closure, dtype=dtype, copy=True if copy else None)


def gains_dict(gains):
    return asdict(gains)


def make_gains(overrides=None):
    if isinstance(overrides, Gains):
        return overrides
    values = dict(overrides or {})
    allowed = {f.name for f in fields(Gains)}
    unknown = set(values)-allowed
    if unknown:
        raise ValueError(f'Unknown gains: {sorted(unknown)}')
    for name in ('force_gain', 'acquisition_force'):
        if name in values:
            values[name] = tuple(values[name])
    return Gains(**values)


def project_force(target, upper, budget):
    """Euclidean five-force box and total-budget projection for all baselines."""
    y = np.asarray(target, float)
    cap = np.maximum(0., np.asarray(upper, float))
    result = np.clip(y, 0., cap)
    if result.sum() <= budget:
        return result
    lo, hi = 0., max(0., float(np.max(y)))+float(np.max(cap))
    for _ in range(28):
        mid = .5*(lo+hi)
        if np.clip(y-mid, 0., cap).sum() > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(y-.5*(lo+hi), 0., cap)


class TactileState:
    """One causal, sensor-derived estimate shared by every method."""
    def __init__(self, gains):
        self.g = gains
        self.normal = np.zeros(5)
        self.tangential = np.zeros((5, 2))
        self.slip = np.zeros((5, 2))
        self.pos = np.zeros((5, 3))
        self.direction = np.zeros((5, 3))
        self.basis = np.zeros((5, 2, 3))
        self.rate = np.zeros(5)
        self.mu = np.full(5, gains.nominal_mu)
        self.valid = np.zeros(5, bool)
        self.contact = np.zeros(5, bool)
        self.confidence = np.zeros(5)
        self.alpha = np.zeros(5)
        self.load = 0.
        self.tactile_load = 0.
        self.motor_load = 0.
        self.load_uncertainty = .80
        self.load_confidence = 0.
        self.motor_effort = None
        self.load_sources = {}
        self.up = np.array([0., 0., 1.])
        self.slip_speed = 0.
        self.time = 0.
        self.force_gain = np.asarray(gains.force_gain, float).copy()
        self.force_tau = float(gains.force_tau)
        self.sliding_since = np.zeros(5)
        self.mu_observable = np.zeros(5, bool)
        self.closure_permission = np.ones(5, bool)
        self.closure_max = np.full(5, gains.alpha_max)
        self.closure_min = np.zeros(5)
        self.nominal_weights = np.ones(5)
        self.observation_age = np.zeros(5)
        self.delayed_normal = np.zeros(5)
        self.force_bias = np.zeros(5)
        self.latent_alpha = np.zeros(5)
        self.contact_model_active = np.zeros(5, bool)
        self.contact_loss_duration = np.zeros(5)
        self.force_uncertainty = np.full(5, .3)
        self.action_history = deque(maxlen=140)

    def update(self, frame, dt):
        raw_n = np.asarray(frame.normal, float)
        raw_t = np.asarray(frame.tangential, float)
        raw_s = np.asarray(frame.slip_local, float)
        age = np.broadcast_to(np.asarray(frame.age, float), (5,))
        self.observation_age = age.copy()
        valid = (np.asarray(frame.valid, bool) & np.isfinite(raw_n)
                 & np.isfinite(raw_t).all(axis=1) & np.isfinite(raw_s).all(axis=1)
                 & (age <= .12))
        old_n = self.normal.copy()
        a = 1.-np.exp(-dt/max(.001, self.g.sensor_tau))
        self.delayed_normal[valid] += a*(np.maximum(0., raw_n[valid])-self.delayed_normal[valid])
        self.tangential[valid] += a*(raw_t[valid]-self.tangential[valid])
        self.slip[valid] += a*(raw_s[valid]-self.slip[valid])
        self.valid = valid
        self.confidence = valid.astype(float)*np.clip(1.-age/.15, 0., 1.)
        self.alpha = np.asarray(getattr(frame, 'alpha', frame.joint), float).copy()
        # Public developer calibration parameters are identical for all
        # methods; these are not per-trial mass/friction/object-state truth.
        self.force_gain = np.asarray(getattr(frame, 'force_gain', self.g.force_gain), float).copy()
        self.force_tau = float(getattr(frame, 'force_tau', self.g.force_tau))
        # Fixed-lag offset observer. A delayed tactile measurement updates the
        # force/closure relation at its own acquisition time; known encoder
        # reference history then propagates that relation to the current time.
        # Before an actual skin contact is observed, closure motion is a search
        # through free space and is never converted into inferred contact force.
        response = 1.-np.exp(-dt/max(.001, self.force_tau))
        if not self.action_history:
            self.latent_alpha = self.alpha.copy()
        else:
            self.latent_alpha += response*(self.alpha-self.latent_alpha)
        self.action_history.append((float(frame.time), self.latent_alpha.copy()))
        history_time = np.array([item[0] for item in self.action_history])
        history_alpha = np.array([item[1] for item in self.action_history])
        delayed_alpha = np.array([np.interp(float(frame.time)-age[i], history_time,
                                           history_alpha[:, i]) for i in range(5)])
        geometric_contact = np.linalg.norm(np.asarray(frame.contact_normal, float), axis=1) > .5
        observed_contact = valid & geometric_contact & (raw_n > .08) & (self.delayed_normal > .08)
        new_contact = observed_contact & ~self.contact_model_active
        self.force_bias[new_contact] = raw_n[new_contact]-self.force_gain[new_contact]*delayed_alpha[new_contact]
        self.contact_model_active |= new_contact
        lost = valid & (~geometric_contact | (self.delayed_normal < .04))
        self.contact_loss_duration = np.where(lost, self.contact_loss_duration+dt, 0.)
        self.contact_model_active[self.contact_loss_duration >= .030] = False
        innovation = raw_n-(self.force_gain*delayed_alpha+self.force_bias)
        identifiable = valid & self.contact_model_active
        bias_gain = 1.-np.exp(-dt/max(.01, self.g.observer_bias_tau))
        self.force_bias[identifiable] += bias_gain*innovation[identifiable]
        estimate = np.maximum(0., self.force_gain*self.latent_alpha+self.force_bias)
        self.normal = np.where(self.contact_model_active, estimate, 0.)
        self.force_uncertainty[identifiable] = .97*self.force_uncertainty[identifiable]+.03*np.abs(innovation[identifiable])
        self.force_uncertainty[~identifiable] = np.minimum(.8, self.force_uncertainty[~identifiable]+dt*.2)
        self.rate = .75*self.rate+.25*np.clip((self.normal-old_n)/max(1e-6, dt), -50., 50.)
        self.contact = valid & self.contact_model_active & observed_contact & (self.normal > .15)
        self.closure_permission = np.asarray(
            getattr(frame, 'closure_permission', np.ones(5, bool)), bool).copy()
        self.closure_max = np.broadcast_to(np.asarray(
            getattr(frame, 'closure_max', self.g.alpha_max), float), (5,)).copy()
        self.closure_min = np.broadcast_to(np.asarray(
            getattr(frame, 'closure_min', 0.), float), (5,)).copy()
        profile = np.asarray(getattr(frame, 'nominal_force_weights', WEIGHTS), float).copy()
        if profile.shape != (5,) or not np.isfinite(profile).all() or np.any(profile <= 0.):
            raise ValueError('Public calibrated nominal_force_weights must be five positive finite values')
        self.nominal_weights = profile/float(np.mean(profile))
        self.pos[valid] = np.asarray(frame.contact_pos, float)[valid]
        self.direction[valid] = np.asarray(frame.contact_normal, float)[valid]
        self.basis[valid] = np.asarray(frame.tangent_basis, float)[valid]
        self.up = np.asarray(getattr(frame, 'support_axis', [0., 0., 1.]), float).copy()
        self.up /= max(1e-9, float(np.linalg.norm(self.up)))
        speed = np.linalg.norm(self.slip, axis=1)
        self.slip_speed = float(np.max(np.where(self.contact, speed, 0.)))
        ratio = np.linalg.norm(self.tangential, axis=1)/np.maximum(.15, self.delayed_normal)
        holding = str(getattr(frame, 'phase', 'HOLDING')).upper() in ('HOLD', 'HOLDING')
        supported = np.count_nonzero(self.contact) >= 3 and self.contact[0]
        still_joints = np.abs(np.asarray(frame.joint_velocity, float)) < .08
        self.mu_observable = self.contact & still_joints & holding & supported
        incipient = self.mu_observable & (speed > .004) & (ratio > .06)
        self.sliding_since = np.where(incipient, self.sliding_since+dt, 0.)
        sustained = incipient & (self.sliding_since >= .080)
        # Sliding observations give a conservative friction estimate. No true
        # scene friction is read; no nonsliding ratio is declared equal to mu.
        measured_mu = np.maximum(.10, .92*ratio[sustained])
        self.mu[sustained] += (1.-np.exp(-dt/.30))*(
            np.minimum(self.mu[sustained], measured_mu)-self.mu[sustained])
        vector_t = np.einsum('ij,ijk->ik', self.tangential, self.basis)
        vectors = self.delayed_normal[:, None]*self.direction+vector_t
        supported_load = max(0., float(np.sum(vectors[self.contact]@self.up)))
        if np.count_nonzero(self.contact) >= 2:
            # Remember observed support across transient loss without treating
            # missing force samples as a newly measured zero object load.
            blend = 1.-np.exp(-dt/.16)
            self.tactile_load += blend*(supported_load-self.tactile_load)
        tactile_conf = float(np.mean(self.confidence[self.contact])) if np.any(self.contact) else 0.
        tactile_conf *= min(1., np.count_nonzero(self.contact)/3.)
        tactile_unc = .55+.20*max(0, 3-np.count_nonzero(self.contact))
        motor_estimate = float(getattr(frame, 'load_estimate', np.nan))
        motor_unc = float(getattr(frame, 'load_uncertainty', .80))
        motor_valid = bool(getattr(frame, 'load_valid', np.isfinite(motor_estimate)))
        motor_valid &= np.isfinite(motor_estimate) and np.isfinite(motor_unc) and motor_unc > 0.
        motor_conf = float(np.clip(getattr(frame, 'load_confidence', 1.), 0., 1.)) if motor_valid else 0.
        self.motor_effort = float(frame.motor_effort) if hasattr(frame, 'motor_effort') else None
        if motor_valid:
            self.motor_load = max(0., motor_estimate)
        channel_values = np.array([self.tactile_load, self.motor_load])
        channel_unc = np.array([tactile_unc, max(.15, motor_unc)])
        channel_conf = np.array([tactile_conf, motor_conf])
        weights = channel_conf/channel_unc**2
        if weights.sum() > 1e-8:
            fused = float(weights@channel_values/weights.sum())
            # Engineering uncertainty, not a claimed probabilistic CI. The
            # two channels share the same mechanism and need not be independent.
            width = float(weights@channel_unc/weights.sum())
            if np.all(channel_conf > .05):
                width = max(width, .5*abs(channel_values[0]-channel_values[1]))
            self.load = max(0., fused)
            self.load_uncertainty = max(.20, width)
            self.load_confidence = float(np.max(channel_conf))
        else:
            self.load_uncertainty = max(.80, self.load_uncertainty)
            self.load_confidence = 0.
        self.load_sources = dict(tactile_N=float(self.tactile_load),
                                 tactile_confidence=tactile_conf,
                                 motor_N=float(self.motor_load) if motor_valid else None,
                                 motor_confidence=motor_conf,
                                 fused_weights=weights.tolist())
        self.time = float(frame.time)
        return self


@dataclass
class QPResult:
    x: np.ndarray
    residual: float
    iterations: int
    status: str
    converged: bool
    diagnostics: dict

    def __iter__(self):
        yield self.x
        yield self.residual
        yield self.iterations


def solve_qp(Q, q, A, b, passes=120, tolerance=2e-5, *, feasible_initial=None):
    """Feasible primal active-set QP with explicit optimization status.

    Diagonal variable preconditioning and unit constraint rows separate force
    and motion scales. Given a checked feasible initial point, every accepted
    optimization iterate stays feasible. An iteration limit is an optimization
    failure, never evidence that physical support is impossible. The three-value
    unpacking interface remains compatible with the independent audit.
    """
    Q, q, A, b = map(lambda v: np.asarray(v, float), (Q, q, A, b))
    scale = 1./np.sqrt(np.maximum(1e-12, np.diag(Q)))
    Qs = Q*scale[:, None]*scale[None, :]
    qs = q*scale
    As = A*scale[None, :]
    row_scale = np.maximum(1e-12, np.linalg.norm(As, axis=1))
    As = As/row_scale[:, None]
    bs = b/row_scale
    inv = np.linalg.inv(Qs)
    unconstrained = -(inv@qs)
    if np.max(A@(scale*unconstrained)-b) <= min(tolerance, 1e-9):
        return QPResult(scale*unconstrained, max(0., float(np.max(A@(scale*unconstrained)-b))),
                        0, 'optimal', True, {'stationarity': 0., 'feasible_start': True})
    initial = np.zeros(len(q)) if feasible_initial is None else np.asarray(feasible_initial, float)
    feasible_start = bool(np.max(A@initial-b) <= min(tolerance, 1e-9))
    y = initial/scale
    feasibility_iterations = 0
    if not feasible_start:
        # Euclidean Dykstra phase I is only a search. If it does not produce a
        # checked point, status remains unknown numerical feasibility failure.
        dual = np.zeros(len(b))
        for j in range(max(200, 3*passes)):
            active = np.flatnonzero((As@y-bs > 1e-9) | (dual > 1e-10))
            for row in active:
                nxt = max(0., dual[row]+As[row]@y-bs[row])
                y -= (nxt-dual[row])*As[row]
                dual[row] = nxt
            feasibility_iterations = j+1
            if np.max(A@(scale*y)-b) <= 1e-9:
                feasible_start = True
                break
        if not feasible_start:
            x = scale*y
            return QPResult(x, max(0., float(np.max(A@x-b))), 0,
                            'numerical_feasibility_not_found', False,
                            {'feasibility_iterations': feasibility_iterations, 'feasible_start': False})
    working = []
    converged = False
    stationarity = None
    used = 0
    for iteration in range(max(1, passes)):
        gradient = Qs@y+qs
        if working:
            W = As[working]
            gram = W@inv@W.T
            multipliers = np.linalg.lstsq(gram, -(W@inv@gradient), rcond=1e-10)[0]
            p = -inv@(gradient+W.T@multipliers)
        else:
            multipliers = np.zeros(0)
            p = -inv@gradient
        stationarity = float(np.max(np.abs(p)))
        used = iteration+1
        if stationarity <= max(1e-8, tolerance*.1):
            if not working or np.min(multipliers) >= -max(1e-8, tolerance*.1):
                converged = True
                break
            working.pop(int(np.argmin(multipliers)))
            continue
        direction = As@p
        slack = bs-As@y
        candidates = np.flatnonzero(direction > 1e-10)
        if working:
            candidates = candidates[~np.isin(candidates, working)]
        if len(candidates):
            fractions = np.maximum(0., slack[candidates])/direction[candidates]
            index = int(np.argmin(fractions))
            step = min(1., float(fractions[index]))
            blocker = int(candidates[index])
        else:
            step, blocker = 1., None
        trial = y+step*p
        # Arithmetic drift is handled by backtracking along an already feasible
        # segment, not by reporting a clamped residual as though it were zero.
        for _ in range(12):
            if np.max(A@(scale*trial)-b) <= min(tolerance, 1e-8):
                break
            step *= .5
            trial = y+step*p
        if np.max(A@(scale*trial)-b) > min(tolerance, 1e-8):
            break
        y = trial
        if blocker is not None and step < 1.-1e-9:
            # Keep independent equality rows, including degenerate active faces.
            proposed = working+[blocker]
            if np.linalg.matrix_rank(As[proposed], tol=1e-8) > len(working):
                working.append(blocker)
            elif step < 1e-12:
                # Dependent tight faces may oppose the direction due roundoff.
                # The current iterate remains checked feasible and is returned
                # with explicit nonconvergence rather than false infeasibility.
                break
    x = scale*y
    residual = max(0., float(np.max(A@x-b)))
    return QPResult(x, residual, used, 'optimal' if converged else 'numerical_iteration_limit_feasible',
                    converged, {'stationarity': stationarity, 'feasible_start': feasible_start,
                                'feasibility_iterations': feasibility_iterations,
                                'active_constraints': len(working)})


class CommonController:
    def __init__(self, name, limits, budget, gains=None):
        if name not in METHODS:
            raise ValueError(name)
        self.name = name
        self.g = make_gains(gains)
        self.limits = np.asarray(limits, float)
        self.budget = float(budget)
        self.state = TactileState(self.g)
        self.desired = np.zeros(5)
        self.debug = {}
        self.label = 'contact_establishment'
        self.last_decision = -np.inf
        self.previous_time = None
        self.previous_action = np.zeros(5)
        self.interval_action = np.zeros(5)
        self.predicted = np.zeros(5)
        self.pause_transport = False
        self.switches = 0
        self.control_ticks = 0
        self.infeasible_ticks = 0
        self.numerical_failures = 0
        self.previous_plan = np.zeros(5*self.g.horizon)

    def _action_constraints(self, P, constant, support_floor=None):
        h = self.g.horizon
        size = 5*h
        d = self.g.decision_period
        eye = np.eye(size)
        cumulative = np.kron(np.tril(np.ones((h, h))), np.eye(5))
        caps = np.maximum(0., self.limits-self.g.force_margin)
        force_upper = np.tile(caps, h)
        budget_rows = np.kron(np.eye(h), np.ones((1, 5)))@P
        rows = [eye, -eye, cumulative, -cumulative, P, -P, budget_rows]
        values = [np.full(size, self.g.close_rate*d),
                  np.full(size, self.g.open_rate*d),
                  np.tile(self.state.closure_max-self.state.alpha, h),
                  np.tile(self.state.alpha-self.state.closure_min, h),
                  force_upper-constant,
                  constant-(np.zeros(size) if support_floor is None else support_floor),
                  np.full(h, max(0., self.budget-2.*self.g.force_margin))
                  -np.kron(np.eye(h), np.ones((1, 5)))@constant]
        invalid = ~self.state.valid | ~self.state.closure_permission
        if np.any(invalid):
            invalid_rows = eye[np.tile(invalid, h)]
            rows.append(invalid_rows)
            values.append(np.zeros(len(invalid_rows)))
        return np.vstack(rows), np.concatenate(values)

    def _predictor(self):
        h = self.g.horizon
        period = self.g.decision_period
        gain = self.state.force_gain*self.state.contact_model_active
        a = np.exp(-period/max(.005, self.state.force_tau))
        P = np.zeros((5*h, 5*h))
        for j in range(h):
            for k in range(j+1):
                P[5*j:5*j+5, 5*k:5*k+5] = np.diag(gain*(1.-a**(j-k+1)))
        # Bounded recent local force evolution; this is causal extrapolation,
        # not access to future disturbance. Never assume stale force is exact.
        constant = np.tile(self.state.normal, h)
        return P, constant

    def _clip_first(self, action):
        period = self.g.decision_period
        result = np.clip(action, -self.g.open_rate*period,
                         self.g.close_rate*period)
        result = np.clip(result, self.state.closure_min-self.state.alpha,
                         self.state.closure_max-self.state.alpha)
        blocked = ~self.state.valid | ~self.state.closure_permission
        result[blocked] = np.minimum(0., result[blocked])
        return result

    def _optimize(self, Q, q, P, constant, A, b, support_floor=None):
        """Known feasibility is kept separate from optimizer convergence."""
        h = self.g.horizon
        dimension = 5*h
        candidates = [np.zeros(dimension), self.previous_plan.copy(),
                      np.r_[self.previous_plan[5:], np.zeros(5)]]
        minimum = np.zeros(dimension)
        cumulative = np.zeros(5)
        blocked = ~self.state.valid | ~self.state.closure_permission
        positive_model = bool(np.all(self.state.force_gain > 0.) and self.state.force_tau > 0.)
        force_lower = np.zeros(dimension) if support_floor is None else np.asarray(support_floor)
        for j in range(h):
            base = constant[j*5:j*5+5]+P[j*5:j*5+5]@minimum
            response = np.diag(P[j*5:j*5+5, j*5:j*5+5])
            lower = np.maximum(-self.g.open_rate*self.g.decision_period,
                               self.state.closure_min-self.state.alpha-cumulative)
            upper = np.minimum(self.g.close_rate*self.g.decision_period,
                               self.state.closure_max-self.state.alpha-cumulative)
            upper[blocked] = np.minimum(0., upper[blocked])
            if positive_model:
                lower = np.maximum(lower, (force_lower[j*5:j*5+5]-base)/np.maximum(1e-12, response))
            delta = np.minimum(upper, lower)
            minimum[j*5:j*5+5] = delta
            cumulative += delta
        candidates.append(minimum)
        feasible = [x for x in candidates if np.isfinite(x).all() and np.max(A@x-b) <= 1e-9]
        initial = min(feasible, key=lambda x: float(.5*x@Q@x+q@x)) if feasible else None
        # Independent, structural first-step lower-force certificate. With
        # positive diagonal response, no allowed first action can produce less
        # force than maximal opening. Never infer model infeasibility merely
        # from an iterative solver's failure to converge.
        first_minimum = self._clip_first(np.full(5, -self.g.open_rate*self.g.decision_period))
        force_minimum = np.maximum(0., constant[:5]+P[:5, :5]@first_minimum)
        cap = np.maximum(0., self.limits-self.g.force_margin)
        certificate = bool(positive_model and (
            np.any(force_minimum > cap+1e-9) or
            force_minimum.sum() > max(0., self.budget-2.*self.g.force_margin)+1e-9))
        contact_certificate = bool(support_floor is not None and (
            np.any(force_lower.reshape(h, 5) > cap+1e-9) or
            np.any(force_lower.reshape(h, 5).sum(axis=1)
                   > max(0., self.budget-2.*self.g.force_margin)+1e-9)))
        certificate |= contact_certificate
        if certificate:
            emergency = minimum.copy()
            emergency[:5] = first_minimum
            result = QPResult(emergency, max(0., float(np.max(A@emergency-b))), 0,
                              'model_force_infeasible', False,
                              {'certified_first_step_minimum': force_minimum.tolist(),
                               'contact_floor_capacity_conflict': contact_certificate,
                               'feasible_initial_found': False})
            self.infeasible_ticks += 1
            self.pause_transport = True
        else:
            result = solve_qp(Q, q, A, b, self.g.qp_passes, self.g.qp_tolerance,
                              feasible_initial=initial)
            if not result.converged:
                self.numerical_failures += 1
            if result.residual > min(self.g.qp_tolerance, 1e-8):
                if initial is not None:
                    result = QPResult(initial, max(0., float(np.max(A@initial-b))), result.iterations,
                                      'numerical_failure_feasible_fallback', False,
                                      {'optimizer_status': result.status, 'feasible_initial_found': True})
                else:
                    # The support capability is unknown here, not proven
                    # insufficient. Bounded opening is an explicit degraded
                    # control response to numerical feasibility failure.
                    self.pause_transport = True
                    result.diagnostics['model_infeasibility_certified'] = False
        if result.residual <= min(self.g.qp_tolerance, 1e-8):
            self.previous_plan = result.x.copy()
        self.debug.update(qp_status=result.status, qp_optimal=result.converged,
                          numerical_solver_failures=self.numerical_failures,
                          model_force_infeasible=certificate,
                          model_contact_floor_infeasible=contact_certificate,
                          known_feasible_candidate=initial is not None,
                          qp_diagnostics=result.diagnostics)
        return result

    def _baseline_action(self, target):
        upper = np.maximum(0., self.limits-self.g.force_margin)
        target = project_force(target, upper, max(0., self.budget-.24))
        self.desired = target.copy()
        P, constant = self._predictor()
        # Baselines share action space, local force model, limits and actuator
        # permissions; this conversion is not another task coordination layer.
        desired_action = self.g.baseline_tracking_gain*(target-self.state.normal)/np.maximum(.01, self.state.force_gain)
        horizon_target = np.zeros(5*self.g.horizon)
        horizon_target[:5] = desired_action
        A, b = self._action_constraints(P, constant)
        Q = np.eye(len(horizon_target))
        candidate, residual, iterations = self._optimize(Q, -horizon_target, P, constant, A, b)
        result = self._clip_first(candidate[:5])
        self.predicted = np.maximum(0., constant[:5]+P[:5, :5]@result)
        self.debug.update(qp_primal_residual=residual, qp_iterations=iterations,
                          force_target=target.tolist(),
                          baseline_tracking_gain=self.g.baseline_tracking_gain)
        return result

    def _decide(self, frame, phase):
        self.label = 'fixed_force'
        acquisition = phase in ('CLOSING', 'ACQUIRE', 'CONTACT_ESTABLISHMENT', 'LIFT', 'LIFTING')
        reference = ((np.asarray(self.g.acquisition_force) if acquisition else np.full(5, self.g.base))
                     *self.state.nominal_weights)
        return self._baseline_action(reference)

    def step(self, frame, phase=None, command='hold'):
        dt = float(getattr(frame, 'dt', .005))
        if self.previous_time is not None:
            dt = max(.0001, min(.05, float(frame.time)-self.previous_time))
        self.previous_time = float(frame.time)
        self.state.update(frame, dt)
        self.control_ticks += 1
        phase = str(phase or getattr(frame, 'phase', 'HOLDING')).upper()
        if command == 'release' or phase in ('RELEASE', 'RELEASING'):
            old_label = self.label
            self.label = 'user_release'
            if old_label != self.label:
                self.switches += 1
            self.interval_action[:] = 0.
            self.previous_action[:] = 0.
            self.desired[:] = 0.
            self.pause_transport = False
            delta = -np.minimum(np.maximum(0., self.state.alpha-self.state.closure_min),
                                self.g.release_rate*dt)
            interval_release = -np.minimum(
                np.maximum(0., self.state.alpha-self.state.closure_min),
                self.g.release_rate*self.g.decision_period)
            response = self.state.force_gain*(1.-np.exp(
                -self.g.decision_period/max(.005, self.state.force_tau)))
            self.predicted = np.maximum(0., self.state.normal+response*interval_release)
            self.debug = {'release_authority': True, 'sensor_valid': self.state.valid.tolist()}
            return ControlAction(delta, self.predicted.copy(), self.label, False, self.debug.copy())
        if frame.time-self.last_decision >= self.g.decision_period-1e-8:
            old_label = self.label
            self.pause_transport = np.count_nonzero(self.state.valid) < 3
            self.debug = {}
            if np.count_nonzero(self.state.valid) == 0:
                self.interval_action[:] = 0.
                self.label = 'unobservable_pause'
            else:
                self.interval_action = self._decide(frame, phase)
            if old_label != self.label:
                self.switches += 1
            self.previous_action = self.interval_action.copy()
            self.last_decision = float(frame.time)
            self.debug.update(estimated_mu=self.state.mu.tolist(),
                              estimated_load=self.state.load,
                              load_uncertainty=self.state.load_uncertainty,
                              load_confidence=self.state.load_confidence,
                              load_sources=self.state.load_sources,
                              friction_observable=self.state.mu_observable.tolist(),
                              local_slip_speed=self.state.slip_speed,
                              estimated_normal=self.state.normal.tolist(),
                              valid=self.state.valid.tolist(),
                              decision_period=self.g.decision_period,
                              calibration_force_gain=self.state.force_gain.tolist(),
                              calibration_force_tau=self.state.force_tau,
                              closure_max=self.state.closure_max.tolist(),
                              closure_min=self.state.closure_min.tolist(),
                              nominal_force_weights=self.state.nominal_weights.tolist(),
                              observation_age=self.state.observation_age.tolist(),
                              sensor_filter_tau=self.g.sensor_tau,
                              fixed_lag_observer=True,
                              contact_model_active=self.state.contact_model_active.tolist(),
                              force_observer_uncertainty=self.state.force_uncertainty.tolist(),
                              infeasible_decisions=self.infeasible_ticks)
        delta = self._clip_first(self.interval_action)*min(1., dt/self.g.decision_period)
        # Convert interval bounds to the current execution subinterval. This
        # second clamp only enforces common execution coordinate permissions.
        delta = np.clip(delta, self.state.closure_min-self.state.alpha,
                        self.state.closure_max-self.state.alpha)
        return ControlAction(delta, self.predicted.copy(), self.label,
                             self.pause_transport, self.debug.copy())


class RatioController(CommonController):
    """Matos section 2.1 friction-ratio concept transfer, not paper replication.

    A local tactile-slip signal replaces vibration bands. One nonsliding second
    initializes a five-observation shear/normal target; low-gain scalar PI then
    regulates this ratio. Fixed posture/actuation and all constraints are shared.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar = self.g.base
        self.boost = 0.
        self.target_ratio = None
        self.samples = deque(maxlen=5)
        self.stable_since = None
        self.integral = 0.

    def _ratio_target(self, phase):
        supported = self.state.contact
        slipping = self.state.slip_speed > .004 and np.count_nonzero(supported) >= 2
        if np.count_nonzero(supported) >= 2:
            shear = np.linalg.norm(self.state.tangential[supported], axis=1).sum()
            ratio = float(shear/max(.25, self.state.normal[supported].sum()))
            self.samples.append(ratio)
        ratio = float(np.mean(self.samples)) if self.samples else 0.
        if phase in ('CLOSING', 'ACQUIRE', 'CONTACT_ESTABLISHMENT', 'LIFT', 'LIFTING'):
            self.label = 'shared_acquisition_reference'
            return np.asarray(self.g.acquisition_force)*self.state.nominal_weights
        if self.target_ratio is None:
            self.label = 'ratio_initialization'
            if slipping or np.count_nonzero(supported) < 3:
                if slipping:
                    self.boost += self.g.slip_boost_rate*self.g.decision_period
                self.stable_since = None
            else:
                self.boost *= np.exp(-self.g.decision_period/1.5)
                if self.stable_since is None:
                    self.stable_since = self.state.time
                if self.state.time-self.stable_since >= 1.:
                    self.target_ratio = max(.025, ratio)
                    self.scalar = self.g.base+self.boost
            target = self.g.base+self.boost
        else:
            self.label = 'ratio_feedback'
            error = ratio-self.target_ratio
            integral = np.clip(self.integral+error*self.g.decision_period, -.35, .75)
            raw = self.scalar+self.g.ratio_kp*error+self.g.ratio_ki*integral
            if self.g.ratio_floor <= raw <= 5. or raw > 5. and error < 0. or raw < self.g.ratio_floor and error > 0.:
                self.integral = integral
            target = np.clip(raw, self.g.ratio_floor, 5.)
        self.debug.update(target_ratio=self.target_ratio, ratio=ratio)
        return target*self.state.nominal_weights

    def _decide(self, frame, phase):
        return self._baseline_action(self._ratio_target(phase))


class IndependentReflexController(CommonController):
    """Explicit independently superimposed anti-slip and pressure reflexes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slip_memory = np.zeros(5)

    def _decide(self, frame, phase):
        if phase in ('CLOSING', 'ACQUIRE', 'CONTACT_ESTABLISHMENT', 'LIFT', 'LIFTING'):
            self.label = 'shared_acquisition_reference'
            return self._baseline_action(np.asarray(self.g.acquisition_force)*self.state.nominal_weights)
        speed = np.linalg.norm(self.state.slip, axis=1)
        active = self.state.contact & (speed > .004)
        self.slip_memory += np.where(active, self.g.slip_boost_rate,
                                     -.8*self.slip_memory)*self.g.decision_period
        self.slip_memory = np.clip(self.slip_memory, 0., 3.)
        anti_slip = self.g.base*self.state.nominal_weights+self.slip_memory
        protect = 1.5*np.maximum(0., self.state.normal-(self.limits-.3))
        self.label = 'independent_reflexes'
        self.debug.update(slip_memory=self.slip_memory.tolist(),
                          pressure_reflex=protect.tolist())
        return self._baseline_action(np.maximum(0., anti_slip-protect))


class UnifiedCoordinator(CommonController):
    """Continuous requested-force trajectory with a common observed executor.

    References r, predicted actual forces n and closure increments u are three
    distinct variables. Capacity inequalities never impersonate realized shear.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_set = ()
        self.force_reference = np.asarray(self.g.acquisition_force, float).copy()
        self.reference_plan = np.tile(self.force_reference, self.g.horizon)

    def _geometry(self):
        s = self.state
        normal = s.direction.copy()
        magnitude = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(1e-9, magnitude[:, None])
        active = s.contact & (magnitude > .5)
        center = np.mean(s.pos[active], axis=0) if np.any(active) else np.zeros(3)
        lever = s.pos-center
        wrench = np.vstack((normal.T, np.cross(lever, normal).T/self.g.characteristic_length))
        wrench[:, ~active] = 0.
        shear_vector = np.einsum('ij,ijk->ik', s.tangential, s.basis)
        actual_shear_wrench = np.r_[shear_vector[active].sum(axis=0),
            np.cross(lever[active], shear_vector[active]).sum(axis=0)/self.g.characteristic_length]
        safe_mu = s.mu*self.g.friction_margin*np.clip(s.confidence, .35, 1.)
        tangent_up = s.up-normal*(normal@s.up)[:, None]
        capacity = np.maximum(0., normal@s.up)+safe_mu*np.linalg.norm(tangent_up, axis=1)
        capacity[~active] = 0.
        if self.name == 'unified_no_geometry':
            capacity = np.where(active, safe_mu, 0.)
        return active, wrench, actual_shear_wrench, capacity, safe_mu

    def _support_selection(self, active, capacity):
        choices = np.flatnonzero(active & (np.arange(5) != 0))
        if active[0] and len(choices) >= 2:
            candidates = []
            for a in range(len(choices)):
                for b in range(a+1, len(choices)):
                    selected = (0, int(choices[a]), int(choices[b]))
                    confidence = float(np.mean(self.state.confidence[list(selected)]))
                    if self.name == 'unified_no_geometry':
                        score = confidence+.10*float(np.sum(self.state.normal[list(selected)]))/max(1., self.limits.sum())
                    else:
                        span = np.linalg.norm(self.state.pos[selected[1]]-self.state.pos[selected[2]])
                        score = confidence+.35*span/self.g.characteristic_length+.25*capacity[list(selected)].sum()
                    candidates.append((float(score), selected))
            score, best = max(candidates)
            old = next((v for v, selected in candidates if selected == self.support_set), -np.inf)
            if len(self.support_set) == 3 and old >= .95*score:
                best = self.support_set
            self.support_set = best
        return self.support_set

    def _trajectory_model(self):
        h, period = self.g.horizon, self.g.decision_period
        beta = self.g.baseline_tracking_gain
        response = 1.-np.exp(-period/max(.001, self.state.force_tau))
        gamma = beta*response*self.state.contact_model_active.astype(float)
        P = np.zeros((5*h, 5*h))
        c = np.concatenate([(1.-gamma)**(j+1)*self.state.normal for j in range(h)])
        for j in range(h):
            for k in range(j+1):
                P[5*j:5*j+5, 5*k:5*k+5] = np.diag(gamma*(1.-gamma)**(j-k))
        previous_P = np.vstack((np.zeros((5, 5*h)), P[:-5]))
        previous_c = np.r_[self.state.normal, c[:-5]]
        scale = np.tile(beta/np.maximum(.01, self.state.force_gain), h)
        U = scale[:, None]*(np.eye(5*h)-previous_P)
        uc = -scale*previous_c
        return P, c, U, uc

    def _decide(self, frame, phase):
        started = time.perf_counter()
        h = self.g.horizon
        nr, dimension = 5*h, 6*h
        P, c, U, uc = self._trajectory_model()
        PN = np.column_stack((P, np.zeros((nr, h))))
        UA = np.column_stack((U, np.zeros((nr, h))))
        R = np.column_stack((np.eye(nr), np.zeros((nr, h))))
        slack_rows = np.column_stack((np.zeros((h, nr)), np.eye(h)))
        active, wrench, shear_wrench, capacity, safe_mu = self._geometry()
        selected = self._support_selection(active, capacity)
        acquisition = phase in ('CLOSING', 'ACQUIRE', 'CONTACT_ESTABLISHMENT')
        lifting = phase in ('LIFT', 'LIFTING')
        rebuilding = (not acquisition and (len(selected) < 3 or
            not all(active[i] and self.state.normal[i] >= self.g.support_contact_floor-.025 for i in selected)))
        profile_reference = ((np.asarray(self.g.acquisition_force) if acquisition or lifting or rebuilding
                              else np.full(5, self.g.base))*self.state.nominal_weights)
        Q = np.eye(dimension)*1e-6
        q = np.zeros(dimension)

        def objective(matrix, offset, weight):
            nonlocal Q, q
            if weight <= 0.:
                return
            Q += 2.*weight*(matrix.T@matrix)
            q += 2.*weight*(matrix.T@offset)

        if self.name == 'unified_fixed_stage':
            contact_weight, continuity_scale = 1., 1.
        elif acquisition or lifting or rebuilding:
            contact_weight, continuity_scale = 4., .45
        elif self.state.slip_speed > .004:
            contact_weight, continuity_scale = .0, .45
        else:
            contact_weight, continuity_scale = .0, 1.
        objective(R, np.zeros(nr), self.g.force_cost)
        objective(R, -np.tile(profile_reference, h), contact_weight)
        if self.name != 'unified_no_geometry':
            actual_normal_wrench = wrench@self.state.normal
            GW = np.kron(np.eye(h), wrench)
            weights = np.r_[np.full(3, self.g.balance_weight), np.full(3, self.g.moment_weight)]
            weights = np.tile(np.sqrt(weights), h)
            objective(weights[:, None]*(GW@PN), weights*(GW@c-np.tile(actual_normal_wrench, h)), 1.)
        D = np.kron(np.eye(h)-np.eye(h, k=-1), np.eye(5))
        if self.name != 'unified_no_continuity':
            previous_r = np.zeros(nr)
            previous_r[:5] = self.force_reference
            objective(D@R, -previous_r, self.g.reference_continuity*continuity_scale)
            previous_n = np.zeros(nr)
            previous_n[:5] = self.state.normal
            objective(D@PN, D@c-previous_n, self.g.force_continuity*continuity_scale)
            previous_u = np.zeros(nr)
            previous_u[:5] = self.previous_action
            action_scale = np.tile(self.state.force_gain, h)
            objective(action_scale[:, None]*(D@UA), action_scale*(D@uc-previous_u),
                      self.g.action_continuity*continuity_scale)
        objective(slack_rows, np.zeros(h), self.g.support_slack_cost)
        identity = np.eye(nr)
        cumulative = np.kron(np.tril(np.ones((h, h))), np.eye(5))
        sums = np.kron(np.eye(h), np.ones((1, 5)))
        caps = np.maximum(0., self.limits-self.g.force_margin)
        budget = max(0., self.budget-2.*self.g.force_margin)
        rows = [R, -R, sums@R, UA, -UA, cumulative@UA, -cumulative@UA,
                PN, -PN, sums@PN, -slack_rows]
        bounds = [np.tile(caps, h), np.zeros(nr), np.full(h, budget),
                  np.full(nr, self.g.close_rate*self.g.decision_period)-uc,
                  np.full(nr, self.g.open_rate*self.g.decision_period)+uc,
                  np.tile(self.state.closure_max-self.state.alpha, h)-cumulative@uc,
                  np.tile(self.state.alpha-self.state.closure_min, h)+cumulative@uc,
                  np.tile(caps, h)-c, c, np.full(h, budget)-sums@c, np.zeros(h)]
        blocked = ~self.state.valid | ~self.state.closure_permission
        if np.any(blocked):
            rows.append(UA[np.tile(blocked, h)])
            bounds.append(-uc[np.tile(blocked, h)])
        force_floor = np.zeros((h, 5))
        reference_floor = np.zeros((h, 5))
        if not acquisition:
            attainable = (P@np.tile(caps, h)+c).reshape(h, 5)
            for i in selected:
                maximum_reference = self.state.normal[i]+self.state.force_gain[i]*max(
                    0., min(self.g.close_rate*self.g.decision_period,
                            self.state.closure_max[i]-self.state.alpha[i]))/max(.01, self.g.baseline_tracking_gain)
                reference_floor[:, i] = min(self.g.support_contact_floor, maximum_reference)
                if active[i]:
                    max_action_force = self.state.normal[i]+self.state.force_gain[i]*self.g.close_rate*self.g.decision_period*np.arange(1,h+1)
                    force_floor[:, i] = np.minimum(self.g.support_contact_floor,
                                                  np.minimum(attainable[:, i], max_action_force))
            rows += [-R, -PN]
            bounds += [-reference_floor.ravel(), c-force_floor.ravel()]
        load_demand = (self.state.load*(1.+self.g.support_reserve_fraction)
                       +self.g.load_uncertainty_weight*self.state.load_uncertainty+self.g.support_buffer
                       +self.g.slip_load_gain*min(.04, self.state.slip_speed))
        tangent_demand = ((1.+self.g.support_reserve_fraction)*np.linalg.norm(self.state.tangential, axis=1)
                          +self.g.slip_load_gain*np.minimum(.015, np.linalg.norm(self.state.slip,axis=1))/5.)
        # Measured shear demand and potential capacity are inequalities with
        # explicit reserve slack. They are not fictitious realized force vectors.
        if np.count_nonzero(active) >= 3 and active[0]:
            for j in range(h):
                for i in np.flatnonzero(active):
                    row = -safe_mu[i]*PN[5*j+i].copy()
                    row[nr+j] = -1.
                    rows.append(row[None, :])
                    bounds.append(np.array([safe_mu[i]*c[5*j+i]-tangent_demand[i]]))
                row = -(capacity@PN[5*j:5*j+5])
                row[nr+j] = -1.
                rows.append(row[None, :])
                bounds.append(np.array([capacity@c[5*j:5*j+5]-load_demand]))
        A, b = np.vstack(rows), np.concatenate(bounds)
        feasible = []
        for reference in (np.tile(self.state.normal,h), self.reference_plan,
                          np.tile(self.force_reference,h), np.tile(project_force(profile_reference,caps,budget),h)):
            trial = np.r_[reference, np.zeros(h)]
            # Slack only relaxes the declared support reserve, never the
            # pressure, action-domain, cumulative-coordinate or force-floor rows.
            for j in range(h):
                includes = A[:,nr+j] < -.5
                if np.any(includes):
                    trial[nr+j] = max(0., float(np.max(A[includes]@trial-b[includes])))+1e-8
            if np.max(A@trial-b) <= 1e-9:
                feasible.append(trial)
        seed = min(feasible,key=lambda x:float(.5*x@Q@x+q@x)) if feasible else None
        result = solve_qp(Q,q,A,b,self.g.qp_passes,self.g.qp_tolerance,feasible_initial=seed)
        if not result.converged:
            self.numerical_failures += 1
        if result.residual > 1e-8:
            if seed is not None:
                result = QPResult(seed,max(0.,float(np.max(A@seed-b))),result.iterations,
                                  'numerical_failure_feasible_fallback',False,{'optimizer_status':result.status})
            else:
                reference = project_force(profile_reference,caps,budget)
                self.label = 'contact_reestablishment' if rebuilding else 'trajectory_feasibility_unknown'
                action = self._baseline_action(reference)
                self.pause_transport = True
                self.debug.update(trajectory_qp_status=result.status,trajectory_qp_residual=result.residual,
                                  support_reserve_slack=None,contact_reestablishment=bool(rebuilding),
                                  selected_support_contacts=list(selected),local_force_reference=profile_reference.tolist())
                return action
        reference = result.x[:5]
        self.reference_plan = result.x[:nr].copy()
        self.force_reference = reference.copy()
        reserve_slack = result.x[nr:].copy()
        planned_prediction = c[:5]+P[:5]@result.x[:nr]
        action = self._baseline_action(reference)
        # Common execution reports the same requested reference semantics as
        # every baseline, with a separately timestamped actual-force forecast.
        self.desired = reference.copy()
        self.pause_transport |= rebuilding or float(np.max(reserve_slack)) > .20
        self.label = ('contact_reestablishment' if rebuilding else 'support_reserve_limited'
                      if np.max(reserve_slack) > .20 else 'contact_establishment'
                      if acquisition else 'disturbance_recovery' if self.state.slip_speed > .004 else 'stable_hold')
        self.debug.update(trajectory_qp_status=result.status,trajectory_qp_optimal=result.converged,
                          trajectory_qp_residual=result.residual,trajectory_qp_iterations=result.iterations,
                          qp_seconds=time.perf_counter()-started,
                          local_force_reference=profile_reference.tolist(),force_reference=reference.tolist(),
                          planned_actual_force=planned_prediction.tolist(),support_demand=float(load_demand),
                          support_reserve_fraction=self.g.support_reserve_fraction,
                          support_reserve_slack=reserve_slack.tolist(),
                          support_slack_is_capability_certificate=False,
                          predicted_support=float(capacity@self.predicted),geometry_contacts=active.tolist(),
                          selected_support_contacts=list(selected),predicted_support_floor=force_floor.tolist(),
                          support_contact_floor=self.g.support_contact_floor,contact_reestablishment=bool(rebuilding),
                          estimated_actual_normal_wrench=(wrench@self.state.normal).tolist(),
                          measured_shear_wrench=shear_wrench.tolist(),
                          characteristic_length=self.g.characteristic_length,
                          stage_weights=[contact_weight,continuity_scale],
                          force_prediction=self.predicted.tolist(),
                          local_model='delay_aligned_force_offset_observer_and_shared_force_trajectory_executor')
        return action


def make_controller(name, limits, budget, gains=None):
    cls = (RatioController if name == 'ratio' else IndependentReflexController
           if name == 'independent' else UnifiedCoordinator if name.startswith('unified')
           else CommonController)
    return cls(name, limits, budget, gains)

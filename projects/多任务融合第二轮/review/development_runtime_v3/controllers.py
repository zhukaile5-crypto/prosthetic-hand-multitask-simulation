"""Literature-transfer baselines and a constrained multi-task tactile coordinator.

All classes receive SensorFrame and task limits only, never the MuJoCo model,
true friction/mass, object pose, applied disturbance, or the future schedule.
"""
from dataclasses import dataclass, asdict, replace
from collections import deque
import numpy as np
from physics import DT, WEIGHTS


@dataclass(frozen=True)
class Gains:
    base: float = 1.8
    step_rate: float = 9.
    ratio_kp: float = 7.
    ratio_ki: float = 6.
    ratio_floor: float = .65
    adaptive_floor: float = .85
    adaptive_rate: float = 14.
    fusion_slip_rate: float = 32.
    fusion_floor: float = 1.05
    predict_horizon: float = .040
    decay_seconds: float = 1.5
    coord_boost_rate: float = 12.
    coord_boost_max: float = 2.
    coord_guard_margin: float = .25
    coord_guard_gain: float = 2.


def box_budget_projection(desired, upper, budget, weight=None, preserve_total=False):
    """Unique weighted least-square projection; optional total-force redistribution.

    argmin .5 sum_i w_i (n_i-d_i)^2, 0<=n_i<=upper_i,
    sum n<=budget (baseline) or sum n=min(sum d,budget,sum upper) (fusion).
    KKT solution n_i=clip(d_i-lambda/w_i,0,upper_i), solved by bisection.
    """
    d = np.asarray(desired,dtype=float)
    upper = np.maximum(0.,np.asarray(upper,dtype=float))
    w = np.ones(5) if weight is None else np.maximum(.05,weight)
    clipped = np.clip(d,0.,upper)
    total = min(budget,float(np.sum(upper)),max(0.,float(np.sum(d))))
    if not preserve_total and np.sum(clipped) <= budget+1e-9:
        return clipped
    if abs(np.sum(clipped)-total) < 1e-8:
        return clipped
    low = float(np.min((d-upper)*w))-1.
    high = float(np.max(d*w))+1.
    for _ in range(28):
        middle = (low+high)/2
        trial = np.clip(d-middle/w,0.,upper)
        if np.sum(trial) > total:
            low = middle
        else:
            high = middle
    return np.clip(d-(low+high)/2/w,0.,upper)


class Common:
    def __init__(self, name, limits, budget, gains=None):
        self.name = name
        self.limits = np.array(limits,dtype=float)
        self.budget = float(budget)
        self.g = gains or Gains()
        self.normal = np.zeros(5)
        self.shear = np.zeros(5)
        self.previous = np.zeros(5)
        self.slip_filtered = 0.
        self.desired = box_budget_projection(self.g.base*WEIGHTS,self.limits-.18,self.budget)
        self.label = 'acquire'
        self.switches = 0
        self.last_label = self.label
        self.slip_events = 0
        self.last_slip = -10.
        self.debug = {}
        self.control_ticks = 0

    def update_sensors(self, frame):
        valid = frame.valid & np.isfinite(frame.normal) & np.isfinite(frame.shear)
        self.previous = self.normal.copy()
        self.normal[valid] = .7*self.normal[valid]+.3*frame.normal[valid]
        self.shear[valid] = .7*self.shear[valid]+.3*frame.shear[valid]
        self.slip_filtered = .6*self.slip_filtered+.4*frame.slip
        return valid

    def step(self, frame, phase, command):
        valid = self.update_sensors(frame)
        self.control_ticks += 1
        # Release authority, acquisition and the physical servo are shared by all baselines.
        if command == 'release' or phase == 'RELEASING':
            self.label = 'release'
            self.desired[:] = 0.
        elif phase in ('CLOSING','LIFTING') or frame.time < 8.:
            self.label = 'acquire'
            self.desired = self.acquire(frame,phase,valid)
        else:
            self.desired = self.hold(frame,valid)
        if self.label != self.last_label:
            self.switches += 1
            self.last_label = self.label
        return self.desired.copy()

    def hold(self, frame, valid):
        self.label = 'hold'
        return box_budget_projection(self.g.base*WEIGHTS,self.limits-.18,self.budget)

    def acquire(self, frame, phase, valid):
        return box_budget_projection(self.g.base*WEIGHTS,self.limits-.18,self.budget)

    def is_slipping(self):
        return self.slip_filtered > .004 and np.count_nonzero(self.normal > .25) >= 2


class EventBaseline(Common):
    """Gautham eq.(8) concept transfer: threshold crossing -> incremental setpoint.

    Protection and release are added for all methods from the supplied references;
    this is a strong three-rule engineering baseline, not the original sensor experiment.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scalar = self.g.base

    def hold(self, frame, valid):
        overload = np.any(self.normal > self.limits)
        slip = self.is_slipping()
        if overload:
            self.label = 'protect'
            self.scalar = max(self.g.base,self.scalar-12.*DT)
        elif slip:
            self.label = 'slip'
            self.scalar = min(5.,self.scalar+self.g.step_rate*DT)
            if frame.time-self.last_slip > .10:
                self.slip_events += 1
                self.last_slip = frame.time
        else:
            self.label = 'hold'
        return box_budget_projection(self.scalar*WEIGHTS,self.limits-.18,self.budget)


class RatioBaseline(Common):
    """Matos sec.2.1 concept transfer to the common normal-force outer loop.

    A tactile-slip signal substitutes for bandpass vibrations; after one stable
    second, initialize a five-sample shear/normal target. PI regulates that ratio.
    Positive ratio error increases closure here, reversing aperture sign in eq.(1).
    No exact soft-hand, filter-bank, EMG or paper-performance replication is claimed.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scalar = self.g.base
        self.mode = 'slip_initialization'
        self.stable_since = None
        self.ratio_samples = deque(maxlen=5)
        self.target_ratio = None
        self.integral = 0.
        self.slip_boost = 0.

    def hold(self, frame, valid):
        supported = valid & (self.normal > .25)
        contact_count = np.count_nonzero(supported)
        if contact_count >= 2:
            observation = float(np.sum(self.shear[supported])/max(.25,np.sum(self.normal[supported])))
            self.ratio_samples.append(observation)
        ratio = float(np.mean(self.ratio_samples)) if self.ratio_samples else 0.
        slip = self.is_slipping() and contact_count >= 2
        if self.mode == 'slip_initialization':
            self.label = 'ratio_initialize'
            if slip or contact_count < 3:
                if slip:
                    self.slip_boost += self.g.step_rate*DT
                self.stable_since = None
            else:
                self.slip_boost *= np.exp(-DT/self.g.decay_seconds)
                if self.stable_since is None:
                    self.stable_since = frame.time
                if frame.time-self.stable_since >= 1.:
                    self.target_ratio = max(.025,float(np.mean(self.ratio_samples)))
                    self.mode = 'friction_ratio'
                    self.scalar = self.g.base+self.slip_boost
            target = self.g.base+self.slip_boost
        else:
            self.label = 'ratio_feedback'
            if contact_count < 2:
                return self.desired.copy()
            error = ratio-self.target_ratio
            # Anti-windup is identical across trials, avoiding an artificially weak PI.
            new_integral = np.clip(self.integral+error*DT,-.35,.75)
            raw = self.scalar+self.g.ratio_kp*error+self.g.ratio_ki*new_integral
            if self.g.ratio_floor <= raw <= 5. or (raw>5. and error<0) or (raw<self.g.ratio_floor and error>0):
                self.integral = new_integral
            target = np.clip(raw,self.g.ratio_floor,5.)
        if slip and frame.time-self.last_slip > .10:
            self.slip_events += 1
            self.last_slip = frame.time
        self.debug = dict(target_ratio=self.target_ratio,ratio=ratio)
        return box_budget_projection(target*WEIGHTS,self.limits-.18,self.budget)


class AdaptiveConeBaseline(Common):
    """Zhang et al., TNSRE 31:738-748, doi:10.1109/TNSRE.2022.3231972.

    Transfer the per-finger shear/normal safety-margin law (sec.III.C) to the
    common force servo; margin .8 -> .6 in .1 steps on detected slip. The actual
    virtual-stiffness/admittance, EMG classifier and wavelet sensor are not replicated.
    This control ensures that distributed tactile adaptation itself is not treated
    as a new contribution of the proposed fusion layer.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.margin = .8
        self.last_margin_update = -10.

    def hold(self, frame, valid):
        slip = self.is_slipping()
        if slip and frame.time-self.last_margin_update >= .1:
            self.margin = max(.6,self.margin-.1)
            self.last_margin_update = frame.time
            self.slip_events += 1
        self.label = 'adaptive_cone'
        desired = np.maximum(self.g.adaptive_floor*WEIGHTS,self.shear/(.4*self.margin))
        desired = np.clip(desired,self.desired-self.g.adaptive_rate*DT,
                          self.desired+self.g.adaptive_rate*DT)
        self.debug = dict(margin=self.margin,nominal_friction=.4)
        return box_budget_projection(desired,self.limits-.18,self.budget)


class Fusion(Common):
    """Task-gated tactile fusion with anticipatory force limits and redistribution."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.target_ratio = None
        self.initial_stable_since = None
        self.reference_anchor = self.desired.copy()
        self.ratio_integral = np.zeros(5)
        self.force_rate = np.zeros(5)
        self.innovation = np.full(5,.04)
        self.slip_debt = 0.
        self.slip_latched = False
        self.stable_since = 8.
        self.last_valid = np.zeros(5)
        self.raw_window = deque(maxlen=5)
        self.last_excess = False
        self.infeasible_ticks = 0
        self.protect_until = 0.

    def update_sensors(self, frame):
        valid = frame.valid & np.isfinite(frame.normal) & np.isfinite(frame.shear)
        # An invalid channel can contain a finite stale value. Exclude it from
        # history as well as the current update, rather than relying on NaNs.
        self.raw_window.append(np.where(valid,frame.normal,np.nan))
        raw = frame.normal.copy()
        if self.name != 'fusion_no_filter' and len(self.raw_window) >= 3:
            stack = np.array(self.raw_window)
            for i in range(5):
                values = stack[:,i]
                values = values[np.isfinite(values)]
                if len(values) >= 3:
                    raw[i] = np.median(values)
        innovation = np.abs(np.nan_to_num(raw-self.normal,nan=0.))
        self.innovation[valid] = .97*self.innovation[valid]+.03*np.minimum(.5,innovation[valid])
        if self.name == 'fusion_no_filter':
            # Keep the common low-pass and slip observer; only the fusion
            # temporal median is removed. Prediction/confidence stay active.
            valid = Common.update_sensors(self,frame)
        else:
            self.previous = self.normal.copy()
            self.normal[valid] = .7*self.normal[valid]+.3*raw[valid]
            self.shear[valid] = .7*self.shear[valid]+.3*frame.shear[valid]
            self.slip_filtered = .6*self.slip_filtered+.4*frame.slip
        derivative = np.clip((self.normal-self.previous)/DT,-60.,60.)
        self.force_rate = .8*self.force_rate+.2*derivative
        self.last_valid[valid] = frame.time
        return valid

    def acquire(self, frame, phase, valid):
        # No anti-slip response while the intended grasp is still being established.
        horizon = 0. if self.name == 'fusion_no_prediction' else self.g.predict_horizon+frame.age
        predicted = self.normal+horizon*np.maximum(0.,self.force_rate)
        margin = .35+np.minimum(.25,1.5*self.innovation)
        upper = np.maximum(.25,self.limits-margin-horizon*np.maximum(0.,self.force_rate))
        desired = self.g.base*WEIGHTS-2.*np.maximum(0.,predicted-self.limits+.35)
        budget = max(2.,self.budget-.65-horizon*max(0.,float(np.sum(self.force_rate))))
        budget -= min(budget-2.,2.*max(0.,float(np.sum(predicted))-budget))
        target = box_budget_projection(desired,upper,budget)
        return target

    def hold(self, frame, valid):
        ratio = self.shear/np.maximum(.25,self.normal)
        contacts = self.normal > .25
        supported = contacts & valid
        if self.target_ratio is None:
            stable_contact = np.count_nonzero(supported) >= 3 and self.slip_filtered < .0025
            if stable_contact:
                if self.initial_stable_since is None:
                    self.initial_stable_since = frame.time
            else:
                self.initial_stable_since = None
            if (frame.time >= 8.45 and self.initial_stable_since is not None and
                    frame.time-self.initial_stable_since >= .20-1e-8):
                self.target_ratio = np.clip(ratio*.95,.025,.75)
                self.reference_anchor = self.desired.copy()
            else:
                self.label = 'fused_initialize'
                # Keep reacting to measured force risk while waiting for a
                # valid stable ratio, rather than freezing the last command.
                return self.acquire(frame,'HOLDING',valid)
        corroborated = np.any(ratio[supported] > 1.10*self.target_ratio[supported]) if np.any(supported) else False
        if self.slip_filtered > .006 and corroborated and np.count_nonzero(supported) >= 2:
            self.slip_latched = True
            self.stable_since = frame.time
        elif self.slip_filtered < .0025 or np.count_nonzero(supported) < 2:
            self.slip_latched = False
        confidence = np.clip(1.-(frame.time-self.last_valid)/.35,.15,1.)
        confidence *= np.clip(1.-frame.age/.20,.35,1.)
        confidence *= 1./(1.+2.*self.innovation)
        horizon = 0. if self.name == 'fusion_no_prediction' else self.g.predict_horizon+frame.age
        predicted = self.normal+horizon*np.maximum(0.,self.force_rate)
        pressure_risk = np.any(predicted > self.limits-.35) or np.sum(predicted) > self.budget-.65
        if pressure_risk:
            self.protect_until = frame.time+.15
        headroom_gain = float(np.clip((self.budget-np.sum(predicted)-.45)/1.2,0.,1.))
        if frame.time < self.protect_until:
            headroom_gain = 0.
        if self.slip_latched and headroom_gain > 0.:
            self.label = 'fused_slip'
            self.slip_debt = min(18.,self.slip_debt+self.g.fusion_slip_rate*DT*headroom_gain)
            relevant = contacts & valid & (ratio > 1.15*self.target_ratio)
            self.target_ratio[relevant] = np.maximum(.025,self.target_ratio[relevant]*.9995)
            if frame.time-self.last_slip > .10:
                self.slip_events += 1
                self.last_slip = frame.time
        else:
            self.label = 'fused_hold'
            if pressure_risk:
                self.label = 'fused_protect'
            if frame.time-self.stable_since > .15:
                self.slip_debt *= np.exp(-DT/self.g.decay_seconds)
        # Lost or invalid contact is not a new friction-ratio observation.
        # Retain its memory, but do not integrate a stale shear/normal ratio.
        error = np.where(supported,ratio-self.target_ratio,0.)
        candidate_integral = np.clip(self.ratio_integral+error*DT,-.35,.65)
        local = self.reference_anchor+self.g.ratio_kp*error+self.g.ratio_ki*candidate_integral
        antiwindup = supported & (local >= .5) & (local <= self.limits-.18)
        self.ratio_integral[antiwindup] = candidate_integral[antiwindup]
        local = np.maximum(self.g.fusion_floor*WEIGHTS,local)
        urgency = np.clip(ratio/np.maximum(.025,self.target_ratio),.2,1.5)
        priority = confidence*urgency*WEIGHTS
        local_gate = np.clip((self.limits-predicted-.30)/.60,0.,1.)
        priority *= local_gate
        proposal = local+self.slip_debt*priority/max(.1,np.sum(priority))
        protection_request = 2.*np.maximum(0.,predicted-self.limits+.35)
        proposal = np.maximum(.4,proposal-protection_request)
        margin = .35+np.minimum(.25,1.5*self.innovation)
        upper = np.maximum(.25,self.limits-margin-horizon*np.maximum(0.,self.force_rate))
        upper[~valid] = np.minimum(upper[~valid],self.desired[~valid])
        budget_margin = .65+np.sum(self.innovation)*.5+horizon*max(0.,float(np.sum(self.force_rate)))
        effective_budget = max(2.,self.budget-budget_margin)
        budget_feedback = 2.*max(0.,float(np.sum(predicted))-effective_budget)
        effective_budget = max(2.,effective_budget-budget_feedback)
        projected = box_budget_projection(proposal,upper,effective_budget,
                                          weight=.3+priority,preserve_total=True)
        tracked_proposal = proposal
        if self.name == 'fusion_no_distribution':
            common = np.sum(proposal)/np.sum(WEIGHTS)
            tracked_proposal = common*WEIGHTS
            projected = box_budget_projection(tracked_proposal,upper,effective_budget)
        if np.sum(proposal) > min(effective_budget,np.sum(upper))+.05:
            self.infeasible_ticks += 1
            self.label = 'fused_capacity_limit'
        # Bounded command rate; rapid unloading can still respect a new safety ceiling.
        result = np.clip(projected,self.desired-40.*DT,self.desired+14.*DT)
        result = np.minimum(result,upper)
        result = box_budget_projection(result,upper,effective_budget)
        # Coupled anti-windup: an active force budget must also constrain the
        # independent tactile PI memories, rather than merely clipping their outputs.
        # Back-calculation compares like-for-like force requests. Comparing the
        # executed force with PI alone would integrate the separate slip boost
        # even at zero ratio error. Only unavailable requested force is tracked.
        tracking = np.minimum(0.,result-tracked_proposal)/max(1.,self.g.ratio_ki)
        tracking[~supported] = 0.
        self.ratio_integral = np.clip(self.ratio_integral+.8*tracking*DT,-.35,.65)
        self.debug = dict(target_ratio=self.target_ratio.tolist(),slip_debt=self.slip_debt,
                          predicted_upper=upper.tolist(),confidence=confidence.tolist(),
                          effective_budget=effective_budget,headroom_gain=headroom_gain,
                          infeasible_ticks=self.infeasible_ticks)
        return result


class CoordinatedRatio(RatioBaseline):
    """Event-triggered tactile supervision of the low-gain ratio baseline.

    Outside a force-limit or sensor-validity conflict, this returns the
    baseline command exactly, including during slip. Slip can occur when shear/normal
    falls after friction loss, so slip confirmation never requires its rise.
    Confirmed slip permits compensation of force already removed by a measured
    pressure guard; total requested force never exceeds the baseline request.
    Compensation is bounded and assigned only to valid current contacts.
    Local pressure recovery uses finite-duration caps, rather than continuously
    integrating five independent friction-ratio errors. Force removed from a
    limited finger during a hold is redistributed only while slip remains confirmed.
    The pressure observer uses three valid fresh raw samples independently of
    the baseline PI low-pass, so one force impulse cannot leave a pressure tail.
    The same guard covers acquisition and lifting. Closing has no compensation;
    lifting can replace pressure-removed force across at least three valid
    contacts without waiting for slip. Neither acquisition phase runs ratio PI.

    The four coord_* gains are force-request parameters, not contact-model
    stiffness or hidden information about object mass/friction/pose.
    """
    def __init__(self, name, limits, budget, gains=None):
        # Explicit gains are also the manifest's actual runtime gains. The
        # caller shares its selected ratio core with all coordinator candidates.
        g = gains if gains is not None else replace(Gains(), base=1.2,
                                                    ratio_kp=1., ratio_ki=.3,
                                                    ratio_floor=1.8)
        super().__init__(name, limits, budget, g)
        self.baseline_desired = self.desired.copy()
        self.coordination_boost = 0.
        self.slip_confirm_seconds = 0.
        self.slip_clear_seconds = 0.
        self.slip_confirmed = False
        self.guard_caps = np.full(5, np.inf)
        self.guard_until = np.zeros(5)
        self.total_cap = np.inf
        self.total_guard_until = 0.
        self.coordination_ticks = 0
        self.redistribution_ticks = 0
        self.pressure_window = deque(maxlen=3)
        self.pressure_observer = np.zeros(5)

    def update_sensors(self, frame):
        # The ratio baseline receives exactly its original low-pass processing.
        valid = RatioBaseline.update_sensors(self, frame)
        fresh = valid & (frame.age <= .10)
        self.pressure_window.append(np.where(fresh,frame.normal,np.nan))
        stack = np.asarray(self.pressure_window)
        for i in range(5):
            values = stack[:,i]
            values = values[np.isfinite(values)]
            self.pressure_observer[i] = float(np.median(values)) if len(values) else 0.
        return valid

    def acquire(self, frame, phase, valid):
        previous_output = self.desired.copy()
        baseline = RatioBaseline.acquire(self, frame, phase, valid)
        self.baseline_desired = baseline.copy()
        fresh, observed, margin, guarded = self._update_force_guard(frame, valid, baseline)
        supported = fresh & (self.normal > .25) & (frame.normal > .15)
        upper = np.minimum(self.limits-.18, self.guard_caps)
        budget = min(self.budget, self.total_cap)
        pressure_limited = (baseline.copy() if not np.any(guarded) and not np.isfinite(self.total_cap)
                            else box_budget_projection(baseline,upper,budget))
        pressure_removed = max(0.,float(np.sum(baseline)-np.sum(pressure_limited)))
        upper[~supported] = np.minimum(upper[~supported],baseline[~supported])
        upper[~fresh] = np.minimum(upper[~fresh], previous_output[~fresh])
        protected_baseline = (pressure_limited.copy() if np.all(fresh) else
                              box_budget_projection(baseline,upper,budget))
        # Closing establishes contact without extra requests. During lifting,
        # pressure conflict alone can authorize bounded compensation, because
        # waiting for slip can lose the grasp before a hold is established.
        result,redistributed = self._compensate_pressure(
            protected_baseline,upper,budget,observed,margin,supported,
            pressure_removed,phase == 'LIFTING' and np.count_nonzero(supported) >= 3)
        self.slip_confirm_seconds = 0.
        self.slip_clear_seconds = 0.
        self.slip_confirmed = False
        pressure_guard = np.any(guarded) or np.isfinite(self.total_cap)
        if pressure_guard or np.any(~fresh):
            self.coordination_ticks += 1
            self.label = 'coordinated_acquire_protect' if pressure_guard else 'coordinated_acquire_sensor_hold'
        self.debug = dict(target_ratio=self.target_ratio,baseline_request=baseline.tolist(),
                          acquisition_phase=phase,slip_confirmed=False,
                          coordination_boost=float(self.coordination_boost),
                          pressure_observer_N=observed.tolist(),supported=supported.tolist(),
                          guard_caps=np.minimum(self.limits-.18,self.guard_caps).tolist(),
                          effective_budget=float(budget),redistributed_force=redistributed,
                          pressure_removed_request=pressure_removed,
                          coordination_ticks=self.coordination_ticks,
                          redistribution_ticks=self.redistribution_ticks)
        return result

    def _update_force_guard(self, frame, valid, baseline):
        """Shared measured-force supervision across acquisition and holding."""
        fresh = valid & (frame.age <= .10)
        # Invalid/stale samples are absent from the independent median history;
        # current invalidity also excludes the channel from measured pressure.
        observed = np.where(fresh,self.pressure_observer,0.)
        margin = max(0., self.g.coord_guard_margin)
        local_threshold = np.maximum(0., self.limits-margin)
        local_active = fresh & (observed > local_threshold+1e-9)
        total_observed = float(np.sum(observed))
        total_threshold = max(0., self.budget-2.*margin)
        total_active = total_observed > total_threshold+1e-9
        if self.name != 'coordinated_no_guard':
            # Proportional request reduction avoids cancelling an otherwise
            # sufficient grasp for a modest overshoot. The exponent is the
            # declared response gain, not a hidden model-dependent scale.
            exponent = max(.01, self.g.coord_guard_gain)
            scaling = np.clip(local_threshold/np.maximum(1e-9, observed),0.,1.)**exponent
            requested_caps = baseline*scaling
            if np.any(local_active):
                # Track current pressure instead of indefinitely retaining the
                # minimum from an old impulse. Unloading is immediate; recovery
                # while pressure eases is bounded at 2 N/s per channel.
                self.guard_caps[local_active] = np.minimum(requested_caps[local_active],
                                                           self.guard_caps[local_active]+2.*DT)
                self.guard_until[local_active] = frame.time+.15
            recovering = np.isfinite(self.guard_caps) & ~local_active & (frame.time >= self.guard_until)
            self.guard_caps[recovering] += 2.*DT
            recovered = recovering & (self.guard_caps >= baseline)
            self.guard_caps[recovered] = np.inf
            if total_active:
                total_scale = min(1., total_threshold/max(1e-9,total_observed))**exponent
                requested_total = float(np.sum(baseline))*total_scale
                self.total_cap = min(requested_total,self.total_cap+4.*DT)
                self.total_guard_until = frame.time+.15
            elif np.isfinite(self.total_cap) and frame.time >= self.total_guard_until:
                self.total_cap += 4.*DT
                if self.total_cap >= float(np.sum(baseline)):
                    self.total_cap = np.inf
        return fresh, observed, margin, np.isfinite(self.guard_caps)

    def _compensate_pressure(self, protected_baseline, upper, budget, observed,
                             margin, supported, pressure_removed, allowed):
        guarded = np.isfinite(self.guard_caps)
        capacity = np.maximum(0.,self.limits-margin-observed)
        eligible = supported & ~guarded & (capacity > .15)
        enabled = (self.name != 'coordinated_no_redistribution' and allowed and
                   pressure_removed > 1e-9 and np.count_nonzero(eligible) >= 2)
        if enabled:
            self.coordination_boost = min(self.g.coord_boost_max,pressure_removed,
                                          self.coordination_boost+self.g.coord_boost_rate*DT)
        else:
            self.coordination_boost = 0.
        result = protected_baseline.copy()
        if enabled and self.coordination_boost > 0.:
            priority = WEIGHTS*np.where(eligible,np.minimum(capacity,1.),0.)
            receiving_upper = result.copy()
            receiving_upper[eligible] = np.minimum(upper[eligible],result[eligible]+capacity[eligible])
            request = result+self.coordination_boost*priority/max(1e-12,float(np.sum(priority)))
            result = box_budget_projection(request,receiving_upper,budget,preserve_total=True)
        redistributed = max(0.,float(np.sum(result)-np.sum(protected_baseline)))
        self.coordination_boost = min(self.coordination_boost,redistributed)
        if redistributed > 1e-8:
            self.redistribution_ticks += 1
        return result,redistributed

    def hold(self, frame, valid):
        previous_output = self.desired.copy()
        # The baseline's missing-contact fallback refers to its own previous
        # command, not a previously supervised or boosted command.
        self.desired = self.baseline_desired.copy()
        previous_integral = self.integral
        previous_slip_boost = self.slip_boost
        baseline = RatioBaseline.hold(self, frame, valid)
        self.baseline_desired = baseline.copy()
        baseline_label = self.label
        self.desired = previous_output

        fresh, observed, margin, guarded = self._update_force_guard(frame, valid, baseline)
        supported = fresh & (self.normal > .25) & (frame.normal > .15)
        enough_contacts = np.count_nonzero(supported) >= 2
        if enough_contacts and self.slip_filtered > .008:
            self.slip_confirm_seconds += DT
            self.slip_clear_seconds = 0.
            if self.slip_confirm_seconds >= .025-1e-10:
                self.slip_confirmed = True
        else:
            self.slip_confirm_seconds = 0.
            if not enough_contacts or self.slip_filtered < .003:
                self.slip_clear_seconds += DT
                if self.slip_clear_seconds >= .08-1e-10:
                    self.slip_confirmed = False

        upper = np.minimum(self.limits-.18, self.guard_caps)
        budget = min(self.budget, self.total_cap)
        if not np.any(guarded) and not np.isfinite(self.total_cap):
            pressure_limited = baseline.copy()
        else:
            pressure_limited = box_budget_projection(baseline, upper, budget)
        pressure_removed = max(0.,float(np.sum(baseline)-np.sum(pressure_limited)))
        # Invalidity alone does not authorize positive compensation.
        upper[~supported] = np.minimum(upper[~supported], baseline[~supported])
        upper[~fresh] = np.minimum(upper[~fresh], previous_output[~fresh])
        protected_baseline = (pressure_limited.copy() if np.all(fresh) else
                              box_budget_projection(baseline, upper, budget))
        result,redistributed = self._compensate_pressure(
            protected_baseline,upper,budget,observed,margin,supported,
            pressure_removed,self.slip_confirmed)
        # Coupled saturation feedback: no positive PI or boost memory is stored
        # for a request that the measured safety caps could not execute.
        reduced_baseline = np.any(result < baseline-1e-8)
        if reduced_baseline and self.integral > previous_integral:
            self.integral = previous_integral
        if reduced_baseline and self.slip_boost > previous_slip_boost:
            self.slip_boost = previous_slip_boost
        intervention = (np.any(guarded) or
                        np.isfinite(self.total_cap) or np.any(~fresh))
        if intervention:
            self.coordination_ticks += 1
            self.label = 'coordinated_protect' if np.any(guarded) or np.isfinite(self.total_cap) else 'coordinated_slip' if self.slip_confirmed else 'coordinated_sensor_hold'
        else:
            self.label = baseline_label
        self.debug = dict(target_ratio=self.target_ratio,
                          baseline_request=baseline.tolist(),
                          slip_confirmed=self.slip_confirmed,
                          coordination_boost=self.coordination_boost,
                          supported=supported.tolist(),
                          pressure_observer_N=observed.tolist(),
                          guard_caps=np.minimum(self.limits-.18,self.guard_caps).tolist(),
                          effective_budget=float(budget), redistributed_force=redistributed,
                          pressure_removed_request=pressure_removed,
                          coordination_ticks=self.coordination_ticks,
                          redistribution_ticks=self.redistribution_ticks)
        return result


METHODS = ('constant','event','ratio','adaptive','fusion','legacy_fusion','fusion_no_prediction','fusion_no_distribution','fusion_no_filter','coordinated','coordinated_no_guard','coordinated_no_redistribution')
METHOD_TITLES = {'constant':'固定握力', 'event':'事件规则基线', 'ratio':'摩擦比强基线',
                 'adaptive':'逐指自适应摩擦锥基线',
                 'fusion':'第一轮融合', 'legacy_fusion':'第一轮融合（兼容名）', 'fusion_no_prediction':'旧融合去除力预测',
                 'fusion_no_distribution':'旧融合去除分配', 'fusion_no_filter':'旧融合去除时间中值滤波',
                 'coordinated':'事件触发多任务协调', 'coordinated_no_guard':'协调去除力保护监督',
                 'coordinated_no_redistribution':'协调去除有效接触重分配'}


def make_controller(name, limits, budget, gains=None):
    cls = EventBaseline if name == 'event' else RatioBaseline if name == 'ratio' else AdaptiveConeBaseline if name == 'adaptive' else CoordinatedRatio if name.startswith('coordinated') else Fusion if name.startswith('fusion') or name == 'legacy_fusion' else Common
    if name not in METHODS:
        raise ValueError(name)
    return cls(name,limits,budget,gains)


def gains_dict(gains):
    return asdict(gains)

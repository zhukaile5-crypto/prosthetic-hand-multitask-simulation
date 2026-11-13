"""Causal articulated-hand virtual sensors, separate from evaluation truth.

Each tick consumes exactly 48 Gaussian and six uniform random variates.
Touch, geometry and local slip share one acquisition time. Current encoders
and motor effort remain current. No object pose/mass/friction/schedule enters
the returned SensorFrame. Full contact patches remain in diagnostics only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from copy import deepcopy
import numpy as np

from contracts import (SensorFrame, SensorNoise, SENSOR_PERIOD, WRENCH_LENGTH,
                       public_calibration, tangent_basis, vector)


def _patch_vector(patch, key, alias=None, default=None):
    value = patch.get(key, patch.get(alias, default) if alias else default)
    return vector(value, (3,), f'patch.{key}')


def aggregate_patches(patches, tolerance=.10, length=WRENCH_LENGTH, support_axis=None):
    """Represent all external skin patches, retaining their exact wrench.

    Equivalent normal is the sum of measured patch normal pressures. The
    corresponding direction is the normalized normal-weighted direction, and
    position/slip are normal-weighted averages. This does not silently force
    exactness: the actually lost force and moment are explicitly measured.
    The production distributed representation separately retains normal-wrench
    columns plus measured non-normal wrench, avoiding this point compression.
    Exact current-geometry reconstruction is not a prediction guarantee when
    contact locations or pressure distribution subsequently change.
    """
    if not 0. < length:
        raise ValueError('wrench normalization length must be positive')
    normal = np.zeros(5)
    force = np.zeros((5, 3)); moment = np.zeros((5, 3))
    weighted_pos = np.zeros((5, 3)); weighted_direction = np.zeros((5, 3))
    weighted_slip = np.zeros((5, 3)); count = np.zeros(5, dtype=int)
    malformed = np.zeros(5, dtype=bool)
    target_normal = np.zeros(5); palm_normal = other_normal = 0.
    auxiliary_force=np.zeros(3);auxiliary_moment=np.zeros(3)
    palm_wrench=np.zeros(6);other_wrench=np.zeros(6);auxiliary_count=0
    normal_wrench=np.zeros((6,5));non_normal=np.zeros((5,6))
    shear_magnitude=np.zeros(5);slip_magnitude=np.zeros(5)
    normal_support=np.zeros(5);tangent_support=np.zeros(5)
    up=np.array([0.,0.,1.]) if support_axis is None else vector(support_axis,(3,),'support_axis')
    up/=max(1e-12,float(np.linalg.norm(up)))
    preserved = []
    for original in patches:
        p = dict(original)
        digit = int(p.get('digit', -1))
        n = float(p.get('normal', 0.))
        if not np.isfinite(n) or n < -1e-8:
            raise ValueError('patch normal must be finite and nonnegative')
        n = max(0., n)
        # A recorded zero-contact patch is inert; it must not invent geometry.
        if n <= 1e-10:
            continue
        f = _patch_vector(p, 'force')
        x = _patch_vector(p, 'position')
        d = _patch_vector(p, 'normal_dir', 'normal_direction')
        slip = _patch_vector(p, 'slip_velocity', 'relative_velocity', np.zeros(3))
        torque = _patch_vector(p, 'torque', default=np.zeros(3))
        d_length = float(np.linalg.norm(d))
        p.update(normal=n, force=f.tolist(), position=x.tolist(),
                 normal_dir=d.tolist(), slip_velocity=slip.tolist(), torque=torque.tolist())
        preserved.append(p)
        if digit < 0 or digit >= 5:
            wrench=np.r_[f,np.cross(x,f)+torque]
            auxiliary_force+=f;auxiliary_moment+=wrench[3:];auxiliary_count+=1
            if p.get('part') == 'palm' or p.get('is_palm', False):
                palm_normal += n;palm_wrench+=wrench
            else:
                other_normal += n;other_wrench+=wrench
            continue
        count[digit] += 1
        normal[digit] += n
        force[digit] += f
        moment[digit] += np.cross(x, f) + torque
        weighted_pos[digit] += n * x
        weighted_slip[digit] += n * slip
        if d_length > 1e-8:
            unit=d/d_length
            weighted_direction[digit] += n * unit
            fn=n*unit;ft=f-fn
            normal_wrench[:,digit]+=np.r_[fn,np.cross(x,fn)]
            non_normal[digit]+=np.r_[ft,np.cross(x,ft)+torque]
            shear_magnitude[digit]+=float(np.linalg.norm(ft))
            slip_magnitude[digit]=max(slip_magnitude[digit],float(np.linalg.norm(slip-unit*(unit@slip))))
            normal_support[digit]+=n*float(unit@up)
            tangent_support[digit]+=n*float(np.linalg.norm(up-unit*(unit@up)))
        else:
            malformed[digit] = True
        if p.get('is_target', p.get('is_object', False)):
            target_normal[digit] += n
    pos = np.zeros((5, 3)); direction = np.zeros((5, 3))
    basis = np.zeros((5, 2, 3)); tangent = np.zeros((5, 2)); slip_local = np.zeros((5, 2))
    reconstructed_force = np.zeros((5, 3)); reconstructed_moment = np.zeros((5, 3))
    errors = np.zeros(5); eligible = np.ones(5, dtype=bool)
    reasons = [[] for _ in range(5)]
    for i in range(5):
        if normal[i] > 1e-10:
            pos[i] = weighted_pos[i] / normal[i]
            nlen = np.linalg.norm(weighted_direction[i])
            if nlen > 1e-8:
                direction[i] = weighted_direction[i] / nlen
            else:
                eligible[i] = False; reasons[i].append('opposing_or_undefined_normals')
            if malformed[i]:
                eligible[i] = False; reasons[i].append('invalid_patch_direction')
            basis[i] = tangent_basis(direction[i])
            tangent[i] = basis[i] @ force[i]
            slip_local[i] = basis[i] @ (weighted_slip[i] / normal[i])
            reconstructed_force[i] = normal[i] * direction[i] + basis[i].T @ tangent[i]
            reconstructed_moment[i] = np.cross(pos[i], reconstructed_force[i])
            delta = np.r_[reconstructed_force[i] - force[i],
                          (reconstructed_moment[i] - moment[i]) / length]
            actual = np.r_[force[i], moment[i] / length]
            errors[i] = float(np.linalg.norm(delta) / max(.1, np.linalg.norm(actual)))
            if errors[i] > tolerance:
                eligible[i] = False; reasons[i].append('equivalent_wrench_error_exceeds_tolerance')
        else:
            basis[i] = tangent_basis(np.zeros(3))
    distributed_error=np.zeros(5);distributed_valid=~malformed.copy()
    for i in range(5):
        if normal[i]>1e-10:
            normal_wrench[:,i]/=normal[i]
            normal_support[i]/=normal[i];tangent_support[i]/=normal[i]
        reconstructed=normal[i]*normal_wrench[:,i]+non_normal[i]
        actual=np.r_[force[i],moment[i]]
        delta=(reconstructed-actual)*np.r_[np.ones(3),np.full(3,1./length)]
        denominator=max(.1,np.linalg.norm(actual*np.r_[np.ones(3),np.full(3,1./length)]))
        distributed_error[i]=float(np.linalg.norm(delta)/denominator)
        distributed_valid[i]&=bool(distributed_error[i]<=1e-9)
    measurement = dict(normal=normal, tangential=tangent, contact_pos=pos,
                       contact_normal=direction, tangent_basis=basis, slip_local=slip_local)
    diagnostic = dict(patches=preserved, patch_count=count.tolist(),
                      aggregation_valid=eligible.tolist(), aggregation_error=errors.tolist(),
                      aggregation_reasons=reasons, aggregation_tolerance=float(tolerance),
                      wrench_length_m=float(length),
                      wrench_error_definition='norm([F_equiv-F_raw,(M_equiv-M_raw)/L])/max(0.1,norm([F_raw,M_raw/L]))',
                      raw_force=force.tolist(), raw_moment=moment.tolist(),
                      equivalent_force=reconstructed_force.tolist(),
                      equivalent_moment=reconstructed_moment.tolist(),
                      unresolved_moment=(moment-reconstructed_moment).tolist(),
                      target_normal_for_evaluation=target_normal.tolist(),
                      palm_normal_for_evaluation=float(palm_normal),
                      other_normal_for_evaluation=float(other_normal),
                      raw_auxiliary_wrench=np.r_[auxiliary_force,auxiliary_moment].tolist(),
                      auxiliary_pressure=float(palm_normal+other_normal),
                      auxiliary_patch_count=auxiliary_count,
                      palm_wrench_for_evaluation=palm_wrench.tolist(),
                      other_wrench_for_evaluation=other_wrench.tolist())
    diagnostic.update(normal_wrench_map=normal_wrench.tolist(),
        finger_non_normal_wrench=np.sum(non_normal,axis=0).tolist(),
        finger_non_normal_by_finger=non_normal.tolist(),
        shear_magnitude=shear_magnitude.tolist(),slip_magnitude=slip_magnitude.tolist(),
        normal_support_coeff=normal_support.tolist(),tangent_support_coeff=tangent_support.tolist(),
        distributed_valid=distributed_valid.tolist(),distributed_error=distributed_error.tolist(),
        contact_present=((normal>1e-10)&distributed_valid).tolist(),
        distributed_scope='Measured multi-patch normal-pressure wrench columns plus measured non-normal wrench; no unitization of averaged normals; current geometry only.',
        distributed_tolerance=1e-9)
    return measurement, diagnostic


class HandSensors:
    def __init__(self, calibration, noise=None, seed=0):
        self.calibration = public_calibration(calibration)
        self.noise = noise if isinstance(noise, SensorNoise) else SensorNoise(**(noise or {}))
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed + 910_000)
        self.queue = deque(maxlen=102)
        self.last_time = None; self.last_frame = None
        self.last_diagnostics = {}; self.previous_lift_velocity = None
        self.encoder_acceleration = 0.; self.load_estimate = 0.
        self.samples = 0

    def sample(self, raw, time=None, phase=None):
        t = float(raw['time'] if time is None else time)
        if not np.isfinite(t): raise ValueError('sensor time must be finite')
        if self.last_time is not None:
            if abs(t-self.last_time) < 1e-10:
                # No new acquisition, RNG draws, delay advancement or filter update.
                return deepcopy(self.last_frame)
            if abs(t-self.last_time-SENSOR_PERIOD) > 1e-7:
                raise ValueError('sample exactly once per 5 ms sensor tick; skipped/backward tick')
        measurement, diagnostic = aggregate_patches(raw.get('patches', raw.get('contacts', [])),
                                                   self.noise.aggregation_tolerance,
                                                   support_axis=raw['support_axis'])
        c, noise = self.calibration, self.noise
        n_noise = self.rng.normal(0., noise.force_sigma, 5)
        t_noise = self.rng.normal(0., noise.tangential_sigma, (5, 2))
        p_noise = self.rng.normal(0., noise.position_sigma, (5, 3))
        v_noise = self.rng.normal(0., noise.slip_sigma, (5, 2))
        effort_noise = float(self.rng.normal(0., noise.effort_sigma))
        auxiliary_force_sigma=noise.force_sigma if noise.auxiliary_force_sigma is None else noise.auxiliary_force_sigma
        auxiliary_moment_sigma=auxiliary_force_sigma*WRENCH_LENGTH if noise.auxiliary_moment_sigma is None else noise.auxiliary_moment_sigma
        auxiliary_noise=np.r_[self.rng.normal(0.,auxiliary_force_sigma,3),
                               self.rng.normal(0.,auxiliary_moment_sigma,3)]
        auxiliary_pressure_noise=float(self.rng.normal(0.,auxiliary_force_sigma))
        uniform = self.rng.random(6)
        contact = measurement['normal'] > .02
        packet = {k: np.array(v, copy=True) for k, v in measurement.items()}
        packet['normal'] = np.maximum(0., packet['normal'] + n_noise)
        packet['tangential'] += t_noise
        packet['contact_pos'][contact] += p_noise[contact]
        packet['slip_local'] += v_noise
        packet['slip_local'][~contact] = 0.
        valid = uniform[:5] >= noise.dropout_probability
        for start, end, fingers in noise.dropout_windows:
            if start <= t < end: valid[np.asarray(fingers, dtype=int)] = False
        aggregation_valid = np.asarray(diagnostic['aggregation_valid'], bool)
        distributed_valid=np.asarray(diagnostic['distributed_valid'],bool)
        if noise.strict_aggregation:
            valid &= distributed_valid if noise.geometry_mode=='distributed_wrench' else aggregation_valid
        for value in packet.values(): value[~valid] = np.nan
        normal_map=np.asarray(diagnostic['normal_wrench_map'],float).copy()
        non_normal=np.asarray(diagnostic['finger_non_normal_by_finger'],float).copy()
        for i in range(5):
            shift=p_noise[i] if contact[i] else np.zeros(3)
            normal_map[3:,i]+=np.cross(shift,normal_map[:3,i])
            extra_tangent=measurement['tangent_basis'][i].T@t_noise[i]
            non_normal[i,3:]+=np.cross(shift,non_normal[i,:3])+np.cross(measurement['contact_pos'][i]+shift,extra_tangent)
            non_normal[i,:3]+=extra_tangent
        observed_shear=np.maximum(0.,np.asarray(diagnostic['shear_magnitude'])+t_noise[:,0])
        observed_slip=np.maximum(0.,np.asarray(diagnostic['slip_magnitude'])+v_noise[:,0])
        observed_slip[~contact]=0.
        normal_map[:,~valid]=np.nan;non_normal[~valid]=np.nan
        observed_shear[~valid]=np.nan;observed_slip[~valid]=np.nan
        auxiliary_probability=noise.dropout_probability if noise.auxiliary_dropout_probability is None else noise.auxiliary_dropout_probability
        auxiliary_valid=bool(noise.instrument_auxiliary and uniform[5]>=auxiliary_probability)
        for start,end in noise.auxiliary_dropout_windows:
            if start<=t<end:auxiliary_valid=False
        auxiliary_wrench=np.asarray(diagnostic['raw_auxiliary_wrench'])+auxiliary_noise
        auxiliary_pressure=max(0.,float(diagnostic['auxiliary_pressure'])+auxiliary_pressure_noise)
        if not auxiliary_valid:
            auxiliary_wrench[:]=np.nan;auxiliary_pressure=float('nan')
        packet.update(time=t, valid=valid.copy(),
                      aggregation_valid=aggregation_valid.copy(),
                      aggregation_error=np.asarray(diagnostic['aggregation_error']),
                      aggregation_reasons=deepcopy(diagnostic['aggregation_reasons']),
                      auxiliary_wrench=auxiliary_wrench,auxiliary_pressure=auxiliary_pressure,
                      auxiliary_valid=auxiliary_valid,
                      normal_wrench_map=normal_map,finger_non_normal_by_finger=non_normal,
                      finger_non_normal_wrench=np.nansum(non_normal,axis=0),
                      shear_magnitude=observed_shear,slip_magnitude=observed_slip,
                      normal_support_coeff=np.asarray(diagnostic['normal_support_coeff']),
                      tangent_support_coeff=np.asarray(diagnostic['tangent_support_coeff']),
                      distributed_valid=distributed_valid.copy(),
                      distributed_error=np.asarray(diagnostic['distributed_error']),
                      contact_present=np.asarray(diagnostic['contact_present'],bool))
        self.queue.append(packet)
        delay_ticks = int(round(noise.delay/SENSOR_PERIOD))
        acquired = self.queue[max(0, len(self.queue)-1-delay_ticks)]

        joint = vector(raw['joint'], (5,), 'joint')
        joint_velocity = vector(raw['joint_velocity'], (5,), 'joint_velocity')
        alpha = vector(raw['alpha'], (5,), 'alpha')
        lower = vector(raw['closure_min'], (5,), 'closure_min')
        upper = vector(raw['closure_max'], (5,), 'closure_max')
        if np.any(lower > upper): raise ValueError('closure bounds must be ordered')
        permission = np.asarray(raw.get('closure_permission', np.ones(5, bool)), bool)
        if permission.shape != (5,): raise ValueError('closure_permission must have length five')
        up = vector(raw['support_axis'], (3,), 'support_axis')
        if np.linalg.norm(up) < 1e-8: raise ValueError('support axis must have a direction')
        up /= np.linalg.norm(up)
        lift_v = float(raw.get('lift_velocity', 0.))
        accel = 0. if self.previous_lift_velocity is None else (lift_v-self.previous_lift_velocity)/SENSOR_PERIOD
        self.previous_lift_velocity = lift_v
        self.encoder_acceleration += (1.-np.exp(-SENSOR_PERIOD/.025))*(accel-self.encoder_acceleration)
        effort = float(raw['motor_effort']) + effort_noise
        hand_weight = float(raw.get('bare_hand_weight', raw.get('handload_prior', 0.)))
        if not np.isfinite(hand_weight): raise ValueError('known bare hand weight must be finite')
        inertia = float(raw.get('known_hand_inertia', c.get('known_hand_inertia', 0.)))
        damping = float(raw.get('lift_damping', c.get('lift_damping', 0.)))
        prior = hand_weight + float(c.get('effort_bias_N', 0.)) + inertia*self.encoder_acceleration + damping*lift_v
        self.load_estimate += (1.-np.exp(-SENSOR_PERIOD/.045))*(effort-prior-self.load_estimate)
        declared_load_bound = c.get('load_error_bound_N')
        load_calibrated = declared_load_bound is not None and np.isfinite(declared_load_bound) and float(declared_load_bound) > 0.
        speed = float(np.linalg.norm(joint_velocity))
        bound = (float(declared_load_bound) if load_calibrated else 2.) + .1*abs(self.encoder_acceleration) + .25*speed
        current_phase = str(phase or raw.get('phase', 'READY')).upper()
        load_valid = bool(load_calibrated and float(raw['lift']) > .01 and
                          current_phase in ('LIFT', 'LIFTING', 'HOLD', 'HOLDING') and
                          abs(self.encoder_acceleration) < 2.)
        quiet = abs(lift_v) < .005 and abs(self.encoder_acceleration) < .05 and speed < .02
        frame = SensorFrame(time=t,
            **{k: acquired[k].copy() for k in measurement},
            joint=joint, joint_velocity=joint_velocity, valid=acquired['valid'].copy(),
            age=float(t-acquired['time']), alpha=alpha, phase=current_phase,
            lift=float(raw['lift']), lift_ref=float(raw['lift_ref']), support_axis=up,
            raw_joint=np.asarray(raw.get('raw_joint', joint), float).copy(),
            raw_joint_velocity=np.asarray(raw.get('raw_joint_velocity', joint_velocity), float).copy(),
            force_gain=c['force_gain'].copy(), closure_permission=permission.copy(),
            closure_max=upper, closure_min=lower, nominal_force_weights=c['nominal_force_weights'].copy(),
            motor_effort=effort, handload_prior=float(prior), load_estimate=max(0., float(self.load_estimate)),
            load_uncertainty=max(.1, bound), load_valid=load_valid,
            load_confidence=(.85 if quiet else .5) if load_valid else 0.,
            force_tau=c['force_tau'], force_matrix=c['force_matrix'].copy(),
            model_reference_alpha=c['model_reference_alpha'].copy(),
            response_error_bound=c['response_error_bound'].copy(),
            aggregation_valid=acquired['aggregation_valid'].copy(),
            aggregation_error=acquired['aggregation_error'].copy(),
            sample_time=float(acquired['time']),
            auxiliary_wrench=acquired['auxiliary_wrench'].copy(),
            auxiliary_pressure=float(acquired['auxiliary_pressure']),
            auxiliary_valid=bool(acquired['auxiliary_valid']),auxiliary_age=float(t-acquired['time']),
            auxiliary_uncertainty=float(.10+3.*np.sqrt(3.)*auxiliary_force_sigma),
            **{k:acquired[k].copy() for k in ('normal_wrench_map','finger_non_normal_wrench',
                'finger_non_normal_by_finger','shear_magnitude','slip_magnitude',
                'normal_support_coeff','tangent_support_coeff','distributed_valid',
                'distributed_error','contact_present')})
        diagnostic.update(current_time=t, sample_time=frame.sample_time, age=frame.age,
            samples=self.samples+1, gaussian_draws_per_tick=48, uniform_draws_per_tick=6,
            sensor_configuration=asdict(noise), valid=frame.valid.tolist(),
            observed_aggregation_valid=frame.aggregation_valid.tolist(),
            observed_aggregation_error=frame.aggregation_error.tolist(),
            observed_aggregation_reasons=deepcopy(acquired['aggregation_reasons']),
            geometry_mode=noise.geometry_mode,
            observed_distributed_valid=frame.distributed_valid.tolist(),
            observed_distributed_error=frame.distributed_error.tolist(),
            observed_auxiliary_wrench=frame.auxiliary_wrench.tolist(),
            observed_auxiliary_pressure=frame.auxiliary_pressure,
            observed_auxiliary_valid=frame.auxiliary_valid,
            auxiliary_age=frame.auxiliary_age,
            auxiliary_uncertainty_N=frame.auxiliary_uncertainty,
            auxiliary_uncertainty_kind='Engineering noise/resolution proxy, not a hard bound or statistical guarantee.',
            auxiliary_scope='Idealized virtual tactile coverage of ALL external palm/non-finger contacts; no target-identity selection; moment about palm origin.',
            patch_diagnostic_time=t,
            load_calibrated=load_calibrated, load_bound_kind='public_bare_hand_calibration' if load_calibrated else 'unavailable_motor_channel',
            scope='Raw patches/wrenches describe current diagnostic time; observed_* and valid describe delayed sample_time. Target identity is never a SensorFrame field.')
        self.last_diagnostics = diagnostic
        self.last_time = t; self.last_frame = deepcopy(frame); self.samples += 1
        return frame

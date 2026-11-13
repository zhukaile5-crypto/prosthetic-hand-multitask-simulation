"""Shared signed/coupled fixed-lag tactile observer for Shadow migration.

The lifecycle, confidence, friction-observability and load-fusion rules follow
the frozen TactileState. Only its local force/closure model is generalized:
each delayed output row uses all five coordinates at that row's acquisition
time, and the present prediction uses the full signed public response matrix.
The frozen vendor implementation is never modified.
"""
from __future__ import annotations

import numpy as np

from vendor.frozen_controllers import TactileState
from force_model import ForceResponse
from contracts import vector


class CoupledTactileState(TactileState):
    def __init__(self, gains):
        super().__init__(gains)
        self.force_matrix = None
        self.model_reference_alpha = np.zeros(5)
        self.response_error_bound = np.zeros(5)
        self.last_delayed_coordinates = np.zeros((5, 5))
        self.force_gain_role = 'positive_normalization_scale_not_signed_response'
        self._calibration_signature = None
        self.auxiliary_wrench = np.zeros(6)
        self.auxiliary_valid = False
        self.auxiliary_support = 0.
        self.auxiliary_support_signed = 0.
        self.auxiliary_support_uncertainty = float('inf')
        self.auxiliary_observation_age = float('inf')
        self.auxiliary_pressure = 0.
        self.finger_support_signed = 0.
        self.normal_wrench_map=np.zeros((6,5))
        self.finger_non_normal_by_finger=np.zeros((5,6))
        self.finger_non_normal_wrench=np.zeros(6)
        self.shear_magnitude=np.zeros(5)
        self.slip_magnitude=np.zeros(5)
        self.normal_support_coeff=np.zeros(5)
        self.tangent_support_coeff=np.zeros(5)
        self.distributed_valid=np.zeros(5,bool)

    def update(self, frame, dt):
        if not np.isfinite(dt) or dt <= 0.:
            raise ValueError('observer dt must be finite and positive')
        raw_n = np.asarray(frame.normal, float)
        raw_t = np.asarray(frame.tangential, float)
        raw_s = np.asarray(frame.slip_local, float)
        raw_shear=(np.linalg.norm(raw_t,axis=1) if getattr(frame,'shear_magnitude',None) is None
                   else np.asarray(frame.shear_magnitude,float))
        raw_speed=(np.linalg.norm(raw_s,axis=1) if getattr(frame,'slip_magnitude',None) is None
                   else np.asarray(frame.slip_magnitude,float))
        distributed=getattr(frame,'normal_wrench_map',None) is not None
        age = np.broadcast_to(np.asarray(frame.age, float), (5,))
        self.observation_age = age.copy()
        valid = (np.asarray(frame.valid, bool) & np.isfinite(raw_n)
                 & np.isfinite(raw_t).all(axis=1) & np.isfinite(raw_s).all(axis=1)
                 & np.isfinite(raw_shear) & np.isfinite(raw_speed)
                 & np.isfinite(age) & (age >= 0.) & (age <= .12))
        if distributed:
            distribution_valid=np.asarray(frame.distributed_valid,bool)
            distribution_map=np.asarray(frame.normal_wrench_map,float)
            non_normal=np.asarray(frame.finger_non_normal_by_finger,float)
            if distribution_map.shape!=(6,5) or non_normal.shape!=(5,6):
                raise ValueError('distributed wrench arrays must have shapes (6,5) and (5,6)')
            valid &= distribution_valid & np.isfinite(distribution_map).all(axis=0) & np.isfinite(non_normal).all(axis=1)
        old_n = self.normal.copy()
        a = 1.-np.exp(-dt/max(.001, self.g.sensor_tau))
        self.delayed_normal[valid] += a*(np.maximum(0., raw_n[valid])-self.delayed_normal[valid])
        self.tangential[valid] += a*(raw_t[valid]-self.tangential[valid])
        self.slip[valid] += a*(raw_s[valid]-self.slip[valid])
        self.shear_magnitude[valid]+=a*(np.maximum(0.,raw_shear[valid])-self.shear_magnitude[valid])
        self.slip_magnitude[valid]+=a*(np.maximum(0.,raw_speed[valid])-self.slip_magnitude[valid])
        self.valid = valid
        self.confidence = np.where(valid, np.clip(1.-age/.15, 0., 1.), 0.)
        self.alpha = vector(getattr(frame, 'alpha', frame.joint), (5,), 'frame.alpha')
        # Positive scales normalize costs only; they never become force response.
        self.force_gain = vector(frame.force_gain, (5,), 'frame.force_gain scale')
        if np.any(self.force_gain <= 0.):
            raise ValueError('force_gain contains positive scales, not signed response')
        B = vector(frame.force_matrix, (5, 5), 'frame.force_matrix')
        if self.force_matrix is not None and not np.allclose(B, self.force_matrix, rtol=0., atol=1e-12):
            raise ValueError('Changing public force calibration requires a fresh trial/observer')
        self.force_matrix = B
        self.force_tau = float(frame.force_tau)
        self.model_reference_alpha = vector(frame.model_reference_alpha, (5,), 'model_reference_alpha')
        self.response_error_bound = vector(frame.response_error_bound, (5,), 'response_error_bound')
        if np.any(self.response_error_bound < 0.):
            raise ValueError('response_error_bound must be nonnegative')
        # ForceResponse validates the signed matrix and positive time constant.
        self.force_model = ForceResponse(B, self.force_tau,
            residual_bound=float(np.max(self.response_error_bound)), source='public full-hand calibration')
        signature=(B.copy(),self.model_reference_alpha.copy(),self.force_tau)
        if self._calibration_signature is not None:
            old_B,old_reference,old_tau=self._calibration_signature
            if (not np.allclose(B,old_B,rtol=0.,atol=1e-12) or
                not np.allclose(self.model_reference_alpha,old_reference,rtol=0.,atol=1e-12) or
                abs(self.force_tau-old_tau)>1e-12):
                raise ValueError('Response matrix/reference/time-constant changes require a fresh trial/observer')
        self._calibration_signature=signature
        response = 1.-np.exp(-dt/self.force_tau)
        if not self.action_history:
            self.latent_alpha = self.alpha.copy()
        else:
            self.latent_alpha += response*(self.alpha-self.latent_alpha)
        self.action_history.append((float(frame.time), self.latent_alpha.copy()))
        history_time = np.array([item[0] for item in self.action_history])
        history_alpha = np.array([item[1] for item in self.action_history])
        # Row i of B must see ALL five delayed coordinates at time t-age_i.
        delayed_coordinates = np.array([
            [np.interp(float(frame.time)-age[i], history_time, history_alpha[:, j])
             for j in range(5)] for i in range(5)])
        self.last_delayed_coordinates = delayed_coordinates.copy()
        delayed_model = np.einsum('ij,ij->i', B,
                                 delayed_coordinates-self.model_reference_alpha[None, :])
        current_model = B@(self.latent_alpha-self.model_reference_alpha)
        present=getattr(frame,'contact_present',None)
        geometric_contact=(np.linalg.norm(np.asarray(frame.contact_normal,float),axis=1)>.5
                           if present is None else np.asarray(present,bool))
        observed_contact = valid & geometric_contact & (raw_n > .08) & (self.delayed_normal > .08)
        new_contact = observed_contact & ~self.contact_model_active
        self.force_bias[new_contact] = raw_n[new_contact]-delayed_model[new_contact]
        self.contact_model_active |= new_contact
        lost = valid & (~geometric_contact | (self.delayed_normal < .04))
        self.contact_loss_duration = np.where(lost, self.contact_loss_duration+dt, 0.)
        self.contact_model_active[self.contact_loss_duration >= .030] = False
        innovation = raw_n-(delayed_model+self.force_bias)
        identifiable = valid & self.contact_model_active
        bias_gain = 1.-np.exp(-dt/max(.01, self.g.observer_bias_tau))
        self.force_bias[identifiable] += bias_gain*innovation[identifiable]
        estimate = np.maximum(0., current_model+self.force_bias)
        self.normal = np.where(self.contact_model_active, estimate, 0.)
        self.force_uncertainty[identifiable] = np.maximum(
            self.response_error_bound[identifiable],
            .97*self.force_uncertainty[identifiable]+.03*np.abs(innovation[identifiable]))
        self.force_uncertainty[~identifiable] = np.maximum(
            self.response_error_bound[~identifiable],
            np.minimum(.8, self.force_uncertainty[~identifiable]+dt*.2))
        self.rate = .75*self.rate+.25*np.clip((self.normal-old_n)/max(1e-6, dt), -50., 50.)
        self.contact = valid & self.contact_model_active & observed_contact & (self.normal > .15)
        self.closure_permission = np.asarray(
            getattr(frame, 'closure_permission', np.ones(5, bool)), bool).copy()
        self.closure_max = np.broadcast_to(np.asarray(
            getattr(frame, 'closure_max', self.g.alpha_max), float), (5,)).copy()
        self.closure_min = np.broadcast_to(np.asarray(
            getattr(frame, 'closure_min', 0.), float), (5,)).copy()
        profile = np.asarray(getattr(frame, 'nominal_force_weights', np.ones(5)), float).copy()
        if profile.shape != (5,) or not np.isfinite(profile).all() or np.any(profile <= 0.):
            raise ValueError('Public nominal_force_weights must be five positive finite values')
        self.nominal_weights = profile/float(np.mean(profile))
        self.pos[valid] = np.asarray(frame.contact_pos, float)[valid]
        self.direction[valid] = np.asarray(frame.contact_normal, float)[valid]
        self.basis[valid] = np.asarray(frame.tangent_basis, float)[valid]
        self.up = np.asarray(getattr(frame, 'support_axis', [0., 0., 1.]), float).copy()
        self.up /= max(1e-9, float(np.linalg.norm(self.up)))
        if distributed:
            self.normal_wrench_map[:,valid]=distribution_map[:,valid]
            self.finger_non_normal_by_finger[valid]+=a*(non_normal[valid]-self.finger_non_normal_by_finger[valid])
            self.normal_support_coeff[valid]=np.asarray(frame.normal_support_coeff,float)[valid]
            self.tangent_support_coeff[valid]=np.asarray(frame.tangent_support_coeff,float)[valid]
            self.distributed_valid=np.asarray(frame.distributed_valid,bool).copy()
        else:
            # Explicit single-point compatibility only for mathematical limits.
            self.normal_wrench_map[:3]=self.direction.T
            self.normal_wrench_map[3:]=np.cross(self.pos,self.direction).T
            legacy_t=np.einsum('ij,ijk->ik',self.tangential,self.basis)
            self.finger_non_normal_by_finger=np.c_[legacy_t,np.cross(self.pos,legacy_t)]
            self.normal_support_coeff=self.direction@self.up
            self.tangent_support_coeff=np.linalg.norm(self.up[None,:]-self.normal_support_coeff[:,None]*self.direction,axis=1)
            self.distributed_valid=valid.copy()
        self.finger_non_normal_wrench=np.sum(self.finger_non_normal_by_finger[self.contact],axis=0)
        speed = self.slip_magnitude
        self.slip_speed = float(np.max(np.where(self.contact, speed, 0.)))
        ratio = self.shear_magnitude/np.maximum(.15, self.delayed_normal)
        holding = str(getattr(frame, 'phase', 'HOLDING')).upper() in ('HOLD', 'HOLDING')
        supported = np.count_nonzero(self.contact) >= 3 and self.contact[0]
        still_joints = np.abs(np.asarray(frame.joint_velocity, float)) < .08
        self.mu_observable = self.contact & still_joints & holding & supported
        incipient = self.mu_observable & (speed > .004) & (ratio > .06)
        self.sliding_since = np.where(incipient, self.sliding_since+dt, 0.)
        sustained = incipient & (self.sliding_since >= .080)
        measured_mu = np.maximum(.10, .92*ratio[sustained])
        self.mu[sustained] += (1.-np.exp(-dt/.30))*(
            np.minimum(self.mu[sustained], measured_mu)-self.mu[sustained])
        vectors=(self.normal_wrench_map[:3]*self.delayed_normal[None,:]).T+self.finger_non_normal_by_finger[:,:3]
        # Whole-hand load is estimated BEFORE fusion with total motor load.
        # Auxiliary skin includes all external palm/non-finger contacts. It is
        # not labeled object support and never becomes another controlled finger.
        auxiliary=getattr(frame,'auxiliary_wrench',None)
        auxiliary=np.full(6,np.nan) if auxiliary is None else np.asarray(auxiliary,float)
        auxiliary_age=float(getattr(frame,'auxiliary_age',float('inf')))
        auxiliary_pressure=float(getattr(frame,'auxiliary_pressure',float('nan')))
        auxiliary_uncertainty=float(getattr(frame,'auxiliary_uncertainty',float('inf')))
        self.auxiliary_valid=bool(getattr(frame,'auxiliary_valid',False) and
            auxiliary.shape==(6,) and np.isfinite(auxiliary).all() and
            np.isfinite(auxiliary_age) and 0.<=auxiliary_age<=.12 and
            np.isfinite(auxiliary_pressure) and auxiliary_pressure>=0. and
            np.isfinite(auxiliary_uncertainty) and auxiliary_uncertainty>=0.)
        self.auxiliary_observation_age=auxiliary_age
        self.auxiliary_pressure=auxiliary_pressure
        if self.auxiliary_valid:
            self.auxiliary_wrench+=a*(auxiliary-self.auxiliary_wrench)
        auxiliary_touch=self.auxiliary_valid and auxiliary_pressure>.02
        self.auxiliary_support_signed=float(self.auxiliary_wrench[:3]@self.up) if auxiliary_touch else 0.
        self.auxiliary_support=max(0.,self.auxiliary_support_signed)
        self.auxiliary_support_uncertainty=(auxiliary_uncertainty+
            abs(float((auxiliary[:3]-self.auxiliary_wrench[:3])@self.up))+.5*auxiliary_age
            if self.auxiliary_valid else float('inf'))
        self.finger_support_signed=float(np.sum(vectors[self.contact]@self.up))
        supported_load=max(0.,self.finger_support_signed+self.auxiliary_support_signed)
        if self.auxiliary_valid and (np.count_nonzero(self.contact)>=2 or auxiliary_touch):
            blend = 1.-np.exp(-dt/.16)
            self.tactile_load += blend*(supported_load-self.tactile_load)
        tactile_conf = float(np.mean(self.confidence[self.contact])) if np.any(self.contact) else 0.
        tactile_conf *= min(1., np.count_nonzero(self.contact)/3.)
        auxiliary_conf=(float(np.clip(1.-auxiliary_age/.15,0.,1.)) if self.auxiliary_valid else 0.)
        tactile_conf=min(tactile_conf,auxiliary_conf)
        tactile_unc = (.55+.20*max(0, 3-np.count_nonzero(self.contact))+
                       self.auxiliary_support_uncertainty if self.auxiliary_valid else 10.)
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
                                 tactile_scope='total external full-hand support projection, not target-identified load',
                                 finger_signed_N=self.finger_support_signed,
                                 auxiliary_signed_N=self.auxiliary_support_signed,
                                 auxiliary_positive_N=self.auxiliary_support,
                                 auxiliary_valid=self.auxiliary_valid,
                                 auxiliary_age=auxiliary_age,
                                 motor_N=float(self.motor_load) if motor_valid else None,
                                 motor_confidence=motor_conf,
                                 fused_weights=weights.tolist())
        self.time = float(frame.time)
        return self

"""One causal public force posterior and one executed-reference memory.

No CoupledTactileState and no legacy force estimator are run here. Only the
immutable measured kernel/error loader and finite-ramp history primitives are
reused. Covariances describe propagation of known device measurement noise,
conditioned on the actual regression weights and public geometry. Calibration
error and unexplained dynamics remain separate; none is an admission theorem.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
from coordinator_model import ContactConditionedModel

SENSOR_PERIOD=.005
RAMP_PERIOD=.020
DIM=16


def _copy(value):
    a=np.array(value,copy=True);a.setflags(write=False);return a


def _skew(value):
    x,y,z=value
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def _vector(value,n,name):
    out=np.asarray(value,float)
    if out.shape!=(n,) or not np.isfinite(out).all():raise ValueError(f'{name} requires {n} finite values')
    return out


def _effective_specs(value):
    if value is None:return None
    if not isinstance(value,dict):raise ValueError('Public sensor specifications must be a dictionary')
    required=('force_sigma_N','tangential_sigma_N','position_sigma_m')
    if any(k not in value for k in required):raise ValueError('Public pressure/tangent/position noise specifications are required')
    result={k:float(value[k]) for k in required}
    aux=value.get('auxiliary_force_sigma_N');moment=value.get('auxiliary_moment_sigma_Nm')
    result['auxiliary_force_sigma_N']=result['force_sigma_N'] if aux is None else float(aux)
    result['auxiliary_moment_sigma_Nm']=.05*result['auxiliary_force_sigma_N'] if moment is None else float(moment)
    if not np.isfinite(list(result.values())).all() or min(result.values())<0.:
        raise ValueError('Device standard deviations must be finite and nonnegative')
    return result


class MeasuredResponseHistory:
    """Measured 20ms total increments, executed as four 5ms finite commands.

    No pressure estimate/beta/history from the old estimator is retained. A
    prefix view ensures later executed inputs cannot change a past residual,
    including acquisition times between public input samples.
    """
    def __init__(self,calibration,model_document=None):
        loaded=ContactConditionedModel(calibration,document=model_document)
        for name in ('K','horizons','error_times','error_by_horizon','reference','public_error',
                     'input_lower','input_upper','input_step_limits','identified_inputs'):
            setattr(self,name,_copy(getattr(loaded,name)))
        self.available_directions=tuple(dict(x) for x in loaded.available_directions)
        self.source_path=loaded.source_path;self.source_sha256=loaded.source_sha256
        self.shape=loaded.shape;self.period=RAMP_PERIOD
        self.input_history=[];self.input_scope=[];self.origin=None

    kernel_at=ContactConditionedModel.kernel_at
    _input_at=ContactConditionedModel._input_at
    def actual_increment(self,start,end):
        if not self.input_history or start<self.origin-1e-9 or end>self.input_history[-1][0]+1e-9:return None
        a=np.asarray(self.input_history)
        def at(t):return a[max(0,np.searchsorted(a[:,0],t+1e-10,side='right')-1),1:]
        return at(end)-at(start)

    def record_input(self,time,value,known_input_mask=None):
        time=float(time);value=_vector(value,8,'actual executed reference')
        mask=np.r_[np.ones(5,bool),np.zeros(3,bool)] if known_input_mask is None else np.asarray(known_input_mask,bool)
        if mask.shape!=(8,):raise ValueError('Eight explicit known-response flags required')
        if self.input_history:
            prior=self.input_history[-1]
            if time<prior[0]-1e-9:raise ValueError('Actual executed history cannot move backwards')
            if abs(time-prior[0])<1e-9:
                if not np.array_equal(prior[1:],value):raise ValueError('An observed actual input cannot be rewritten at the same time')
                if not np.array_equal(self.input_scope[-1],mask):raise ValueError('Recorded actual response scope cannot be rewritten retrospectively')
                return
        ContactConditionedModel.record_input(self,time,value);self.input_scope.append(mask.copy())

    def unknown_increment_details(self,start,end):
        """Unidentified executed increments; net cancellation is not evidence.

        Each row's scope describes the actual increment arriving at that row.
        The finite-ramp effect additionally withholds every mixed 20ms cell.
        """
        if not self.input_history or start<self.origin-1e-9 or end>self.input_history[-1][0]+1e-9:return None
        events=[]
        for i in range(1,len(self.input_history)):
            before=self.input_history[i-1];after=self.input_history[i]
            if after[0]<=start+1e-10 or before[0]>=end-1e-10:continue
            delta=after[1:]-before[1:];unknown=(~self.input_scope[i])&(abs(delta)>1e-12)
            if unknown.any():events.append(dict(start_s=float(before[0]),end_s=float(after[0]),
                axes=np.flatnonzero(unknown).tolist(),actual_delta=delta.tolist()))
        return events

    def input_effect(self,query_time,available_through=None,input_columns=None):
        if not self.input_history:return np.zeros(DIM)
        cutoff=min(float(query_time) if available_through is None else float(available_through),self.input_history[-1][0])
        count_prefix=sum(x[0]<=cutoff+1e-10 for x in self.input_history)
        if not count_prefix or cutoff<=self.origin+1e-10:return np.zeros(DIM)
        history=np.asarray(self.input_history[:count_prefix]);scope=np.asarray(self.input_scope[:count_prefix])
        count=int(np.ceil((cutoff-self.origin)/self.period-1e-9));starts=self.origin+np.arange(count)*self.period
        ends=np.minimum(starts+self.period,cutoff);fraction=np.clip((ends-starts)/self.period,1e-10,1.)
        def at(times):return np.stack([np.interp(times,history[:,0],history[:,i+1]) for i in range(8)],axis=-1)
        increments=(at(ends)-at(starts))/fraction[:,None]
        left=np.searchsorted(history[:,0],starts-1e-10,side='left');right=np.searchsorted(history[:,0],ends+1e-10,side='right')
        unknown=np.vstack((np.zeros((1,8),int),np.cumsum(~scope,axis=0)))
        known=(unknown[right]-unknown[left])==0
        selected=np.ones(8,bool) if input_columns is None else np.isin(np.arange(8),input_columns)
        increments*=known&selected
        ages=float(query_time)-starts;old=ages>=self.horizons[-1]
        effect=self.K[-1]@increments[old].sum(axis=0) if old.any() else np.zeros(DIM)
        if (~old).any():effect+=np.einsum('koi,ki->o',self.kernel_at(ages[~old]),increments[~old])
        return effect

    def calibration_at(self,age):
        age=np.broadcast_to(np.asarray(age,float),(DIM,))
        return np.array([np.interp(age[i],self.error_times,self.error_by_horizon[:,i]) for i in range(DIM)])


@dataclass(frozen=True)
class BeliefSnapshot:
    mean16: np.ndarray
    covariance: np.ndarray
    valid16: np.ndarray
    age16: np.ndarray
    observation_time: float
    current_time: float
    mean_time: float
    observed_pressure5: np.ndarray
    observed_pressure_valid5: np.ndarray
    observed_shear5: np.ndarray
    active_contact5: np.ndarray
    contact_observation_valid5: np.ndarray
    observed_tangent_2d: np.ndarray
    observed_nonnormal_by_finger: np.ndarray
    observed_auxiliary_wrench: np.ndarray
    drift_rate16: np.ndarray
    fit_error16: np.ndarray
    observation_available: bool
    diagnostics: dict

    @property
    def normal(self):return self.mean16[:5]
    @property
    def covariance16(self):return self.covariance


@dataclass(frozen=True)
class BeliefForecast:
    P: np.ndarray
    c: np.ndarray
    calibration_error: np.ndarray
    covariance: np.ndarray
    posterior_std: np.ndarray
    fit_error: np.ndarray
    valid: np.ndarray
    model_usable: bool
    observation_available: bool
    reasons: tuple
    diagnostics: dict
    known_input8: np.ndarray
    joint_covariance: np.ndarray

    @property
    def usable_input_mask(self):return self.known_input8

    @property
    def posterior_covariance(self):return self.covariance


class UnifiedBelief:
    """Causal fading-window local-linear posterior of kernel residuals.

    Sensor covariance is propagated through the actual linear WLS weights,
    including cross-output and cross-horizon correlations. The window/clock,
    not a seed or scenario, fixes these weights. Local-linear drift is reported
    with its statistical covariance, not treated as exact disturbance demand.
    """
    def __init__(self,calibration,sensor_specs=None,model_document=None):
        self.response=MeasuredResponseHistory(calibration,model_document)
        self.alpha_radius=calibration.get('selected_radius',calibration.get('trust_radius'))
        self._requested_input_scope=np.r_[np.ones(5,bool),np.zeros(3,bool)]
        self._current_input_scope=np.zeros(8,bool);self._scope_reason='motion_response_unknown_by_default'
        self.sensor_specs=_effective_specs(sensor_specs)
        self.window_s=.25;self.forgetting_s=.10;self.minimum_linear_span_s=.055
        self.maximum_age_s=.12
        self.packets=deque();self._segment_start=np.full(DIM,-np.inf)
        self._active=np.zeros(5,bool);self._active_known=np.zeros(5,bool)
        self._last_valid_time=np.full(DIM,-np.inf);self._last_time=None;self._last_sample=-np.inf
        self._last_frame_signature=None
        self._weights=None;self._slope_weights=None;self._values=None;self._R=None
        self._residual_now=np.full(DIM,np.nan);self._slope=np.zeros(DIM);self._fit=np.full(DIM,np.nan)
        self._counts=np.zeros(DIM,int);self._spans=np.zeros(DIM)
        self.snapshot=None;self._last_measurement=None;self._last_measurement_valid=np.zeros(DIM,bool)
        self._last_noise_diagnostics={};self._last_discontinuity={};self._external_mode_reasons=()
        self._posterior_time=None;self._unidentified_recent_input=False

    @property
    def input_history(self):return self.response.input_history
    @property
    def current_time(self):return None if self.snapshot is None else self.snapshot.current_time
    def actual_increment(self,start,end):return self.response.actual_increment(start,end)

    def set_mode_evidence(self,compatible,reasons=()):
        """Optional public contact/domain evidence, never object truth.

        This only withholds local model usability. Passing True does not grant
        global admission or bypass the posterior/clock/model checks.
        """
        self._external_mode_reasons=() if compatible else tuple(str(x) for x in reasons or ('public_contact_mode_unknown',))

    def set_response_scope(self,known_input8,reason):
        """Public calibrated local-context evidence; never retroactive.

        Unknown inputs stay in the total-output nuisance residual. They are
        not subtracted as an 'explained' point response. New scope applies to
        subsequent fully covered 20ms intervals, never to old unknown inputs.
        """
        mask=np.asarray(known_input8,bool)
        if mask.shape!=(8,) or not isinstance(reason,str) or not reason:
            raise ValueError('Explicit eight-column local response evidence and reason required')
        self._requested_input_scope=mask.copy();self._scope_reason=reason

    def _device_covariance(self,frame,present):
        R=np.zeros((DIM,DIM));spec=self.sensor_specs
        if spec is None:return np.full((DIM,DIM),np.nan),dict(known=False,reason='public_device_specification_unavailable')
        sn=spec['force_sigma_N'];st=spec['tangential_sigma_N'];sx=spec['position_sigma_m']
        sa=spec['auxiliary_force_sigma_N'];sm=spec['auxiliary_moment_sigma_Nm']
        R[np.arange(5),np.arange(5)]=sn*sn;R[np.arange(5,10),np.arange(5,10)]=st*st
        R[10:,10:]=np.diag(np.r_[np.full(3,sa*sa),np.full(3,sm*sm)])
        required=('normal_wrench_map','tangent_basis','contact_pos','finger_non_normal_by_finger')
        if any(getattr(frame,k,None) is None for k in required):
            return R,dict(known=True,wrench_geometry_available=False)
        M=np.asarray(frame.normal_wrench_map,float);B=np.asarray(frame.tangent_basis,float)
        X=np.asarray(frame.contact_pos,float);T=np.asarray(frame.finger_non_normal_by_finger,float);p=np.asarray(frame.normal,float)
        if M.shape!=(6,5) or B.shape!=(5,2,3) or X.shape!=(5,3) or T.shape!=(5,6):
            raise ValueError('Public wrench geometry has invalid dimensions')
        finite=np.isfinite(M).all(axis=0)&np.isfinite(B).all(axis=(1,2))&np.isfinite(X).all(axis=1)&np.isfinite(T).all(axis=1)&np.isfinite(p)
        # Invalid fingers do not fabricate a complete whole-hand wrench. Their
        # pressure/shear marginal R is still defined for a later valid packet.
        for i in np.flatnonzero(finite):
            jn=np.zeros(DIM);jn[i]=1.;jn[10:]=M[:,i]
            jt=np.zeros((DIM,2));jt[i+5,0]=1.
            jt[10:]=np.vstack((B[i].T,_skew(X[i])@B[i].T))
            jx=np.zeros((DIM,3));force=M[:3,i]*p[i]+T[i,:3]
            jx[13:]=-_skew(force)
            # Replace the two marginal terms already initialized above.
            R[i,i]-=sn*sn;R[i+5,i+5]-=st*st
            R+=sn*sn*np.outer(jn,jn)+st*st*(jt@jt.T)
            if present[i]:
                R+=sx*sx*(jx@jx.T)
                V=st*st*(B[i].T@B[i])+sn*sn*np.outer(M[:3,i],M[:3,i])
                R[13:,13:]+=sx*sx*(np.trace(V)*np.eye(3)-V)
        return (R+R.T)*.5,dict(known=True,wrench_geometry_available=bool(finite.all()),
            scope='device-draw first-order public-geometry plug-in plus positive second-order moment terms',
            statistical_only=True,hard_error_bound=False,signed_response_kernel_available=False)

    def _measure(self,frame):
        p=np.asarray(frame.normal,float);tau=np.asarray(frame.shear_magnitude,float)
        if p.shape!=(5,) or tau.shape!=(5,):raise ValueError('Five pressure and distributed-shear observations required')
        valid=np.asarray(frame.valid,bool)&np.isfinite(p)&np.isfinite(tau)
        if valid.shape!=(5,):raise ValueError('Five public validity flags required')
        present=np.asarray(getattr(frame,'contact_present',p>.02),bool)
        if present.shape!=(5,):raise ValueError('Five public contact flags required')
        y=np.r_[p,tau,np.full(6,np.nan)];good=np.r_[valid,valid,np.zeros(6,bool)]
        M=getattr(frame,'normal_wrench_map',None);T=getattr(frame,'finger_non_normal_by_finger',None)
        A=getattr(frame,'auxiliary_wrench',None)
        if valid.all() and bool(getattr(frame,'auxiliary_valid',False)) and M is not None and T is not None and A is not None:
            M=np.asarray(M,float);T=np.asarray(T,float);A=np.asarray(A,float)
            if M.shape==(6,5) and T.shape==(5,6) and A.shape==(6,) and np.isfinite(np.r_[M.ravel(),T.ravel(),A]).all():
                y[10:]=M@p+T.sum(axis=0)+A;good[10:]=True
        R,noise=self._device_covariance(frame,present)
        return y,good,valid,present,R,noise

    def _fit_at(self,query):
        packets=list(self.packets);n=len(packets)
        W=np.zeros((DIM,n));D=np.zeros((DIM,n));fit=np.full(DIM,np.nan);counts=np.zeros(DIM,int);spans=np.zeros(DIM)
        values=np.full((n,DIM),np.nan);Rs=np.zeros((n,DIM,DIM))
        for k,p in enumerate(packets):values[k]=p['residual'];Rs[k]=np.nan_to_num(p['R'],nan=0.)
        for i in range(DIM):
            eligible=[k for k,p in enumerate(packets) if p['valid'][i] and p['covered'] and np.isfinite(p['R'][i,i]) and p['time']>=self._segment_start[i]-1e-10]
            if not eligible:continue
            ts=np.array([packets[k]['time'] for k in eligible]);z=values[eligible,i]
            counts[i]=len(ts);spans[i]=float(ts[-1]-ts[0])
            variance=np.maximum(1e-18,np.array([packets[k]['R'][i,i] for k in eligible]))
            fade=np.exp(-(ts[-1]-ts)/self.forgetting_s)/variance
            # Centering preserves rank/units and avoids cancellation at t=16s.
            center=float(fade@ts/fade.sum());x=ts-center
            if len(ts)>=5 and spans[i]>=self.minimum_linear_span_s-1e-10:
                design=np.column_stack((np.ones(len(ts)),x));normal=design.T@(fade[:,None]*design)
                operator=np.linalg.solve(normal,design.T*fade)
                W[i,eligible]=np.array([1.,query-center])@operator;D[i,eligible]=operator[1]
                fitted=design@(operator@z)
            else:
                W[i,eligible]=fade/fade.sum();fitted=np.full(len(ts),float(W[i,eligible]@z))
            fit[i]=float(np.sqrt(np.sum(fade*(z-fitted)**2)/fade.sum()))
        clean=np.nan_to_num(values,nan=0.);mean=np.einsum('ik,ki->i',W,clean)
        slope=np.einsum('ik,ki->i',D,clean)
        mean[counts==0]=np.nan;slope[counts==0]=0.
        covariance=np.einsum('ik,kij,jk->ij',W,Rs,W)
        invalid=counts==0;covariance[invalid,:]=np.nan;covariance[:,invalid]=np.nan
        return mean,slope,covariance,W,D,values,Rs,fit,counts,spans

    def update(self,frame,executed_motion3,known_history_input8=None):
        now=float(frame.time);sample=float(frame.sample_time)
        if not np.isfinite([now,sample]).all() or sample>now+1e-9:raise ValueError('Acquisition must be finite and no later than current time')
        motion=_vector(executed_motion3,3,'executed motion reference');alpha=_vector(frame.alpha,5,'actual alpha')
        if self._last_time is not None:
            if now<self._last_time-1e-9:raise ValueError('Belief time cannot move backwards')
            if now>self._last_time+1e-9 and abs(now-self._last_time-SENSOR_PERIOD)>1e-7:raise ValueError('Belief acquisition follows the actual 5ms public clock')
            if sample<self._last_sample-1e-9:raise ValueError('Acquisition time cannot move backwards')
        scope=self._requested_input_scope.copy()
        if self.alpha_radius is None or np.any(abs(alpha-self.response.reference)>float(self.alpha_radius)+1e-9):scope[:5]=False
        if known_history_input8 is None:history_scope=scope.copy()
        else:
            actual_scope=np.asarray(known_history_input8)
            if actual_scope.shape!=(8,) or actual_scope.dtype.kind!='b':raise ValueError('known_history_input8 requires eight boolean flags')
            history_scope=scope&actual_scope
        self.response.record_input(now,np.r_[alpha,motion],history_scope);self._current_input_scope=scope
        incoming=getattr(frame,'sensor_characteristics',None)
        if incoming is not None:
            effective=_effective_specs(incoming)
            if self.sensor_specs is not None and any(abs(effective[k]-self.sensor_specs[k])>1e-12 for k in effective):
                raise ValueError('Constructor/frame device specifications disagree; start a fresh belief for a new device')
            self.sensor_specs=effective
        y,good,contact_valid,present,R,noise=self._measure(frame)
        signature=(sample,y.copy(),good.copy(),present.copy())
        if self._last_time is not None and abs(now-self._last_time)<1e-9:
            previous=self._last_frame_signature
            if sample!=previous[0] or any(not np.array_equal(a,b,equal_nan=True) for a,b in zip(signature[1:],previous[1:])):
                raise ValueError('A public observation cannot be rewritten at the same current time')
            return self.snapshot
        self._last_time=now;self._last_frame_signature=signature
        changed=contact_valid&self._active_known&(present!=self._active)
        if changed.any():
            rows=np.r_[np.flatnonzero(changed),np.flatnonzero(changed)+5,np.arange(10,16)]
            self._segment_start[rows]=sample
        self._active[contact_valid]=present[contact_valid];self._active_known[contact_valid]=True
        covered=bool(self.response.origin is not None and sample>=self.response.origin-1e-9)
        effect=self.response.input_effect(sample,sample)
        discontinuity=[]
        if covered and self.packets:
            prior=self._fit_at(sample)
            expected=prior[0]+effect;prior_var=np.diag(prior[2]);cal=self.response.calibration_at(np.full(DIM,max(RAMP_PERIOD,sample-self._last_sample)))
            scale=cal+6.*np.sqrt(np.maximum(0.,np.nan_to_num(prior_var,nan=0.)+np.diag(np.nan_to_num(R,nan=0.))))
            test=good&np.isfinite(expected)&(prior[8]>=5)&(prior[9]>=self.minimum_linear_span_s)
            jump=test&(abs(y-expected)>scale)
            if jump.any():
                rows=np.flatnonzero(jump)
                if np.any(rows<10):rows=np.unique(np.r_[rows,np.arange(10,16)])
                self._segment_start[rows]=sample;discontinuity=rows.tolist()
        # Exact public absence conditions a physical inactive channel to zero.
        # No active affine pressure/shear estimate or forecast is clipped.
        inactive=contact_valid&~present
        for i in np.flatnonzero(inactive):
            for row in (i,i+5):
                y[row]=0.;R[row,:]=0.;R[:,row]=0.
        if sample>self._last_sample+1e-9:
            self.packets.append(dict(time=sample,residual=y-effect,valid=good.copy(),R=R.copy(),covered=covered))
            self._last_sample=sample;self._last_valid_time[good]=sample
        while self.packets and self.packets[0]['time']<sample-self.window_s-1e-10:self.packets.popleft()
        current_fit=self._fit_at(now)
        increment=self.response.actual_increment(sample,now)
        recent_start=max(self.response.origin,min(sample,now-RAMP_PERIOD))
        unknown_events=self.response.unknown_increment_details(recent_start,now)
        unidentified_recent=increment is None or bool(np.any(abs(increment[~scope])>1e-12)) or bool(unknown_events)
        cold=bool(np.any((current_fit[8]<5)|(current_fit[9]<self.minimum_linear_span_s-1e-10)))
        # Unknown changes are not propagated with the old kernel. The same
        # total-output posterior is exposed at its explicit acquisition time;
        # feedback sees that timestamp, not a fictitious current force state.
        estimate_time=sample if unidentified_recent or changed.any() or discontinuity or self._external_mode_reasons or cold else now
        result=current_fit if estimate_time==now else self._fit_at(estimate_time)
        residual,slope,cov,W,D,values,Rs,fit,counts,spans=result
        # Before actual history covers the first delayed samples, expose the
        # single observed posterior without pretending a model propagation.
        if np.any(counts==0):
            for i in np.flatnonzero((counts==0)&good):
                residual[i]=y[i]-self.response.input_effect(estimate_time,estimate_time)[i];cov[i,i]=R[i,i]
            fallback=(counts==0)&good
            if fallback.any():cov[np.ix_(fallback,fallback)]=R[np.ix_(fallback,fallback)]
        mean=residual+self.response.input_effect(estimate_time,estimate_time)
        for i in np.flatnonzero(self._active_known&~self._active):
            mean[[i,i+5]]=0.;slope[[i,i+5]]=0.
            cov[[i,i+5],:]=0.;cov[:,[i,i+5]]=0.
        age=now-self._last_valid_time
        valid=(age<=self.maximum_age_s+1e-9)&np.isfinite(mean)&np.isfinite(np.diag(cov))
        self._weights=W;self._slope_weights=D;self._values=values;self._R=Rs
        self._residual_now=residual;self._slope=slope;self._fit=fit;self._counts=counts;self._spans=spans
        self._posterior_time=estimate_time;self._unidentified_recent_input=unidentified_recent
        self._last_measurement=y.copy();self._last_measurement_valid=good.copy();self._last_noise_diagnostics=noise
        self._last_discontinuity=dict(contact_change_digits=np.flatnonzero(changed).tolist(),innovation_change_outputs=discontinuity)
        diag=dict(single_force_posterior=True,legacy_estimator_called=False,measurement_covariance_only=True,
            covariance_interpretation='conditioned on realized clock/validity/segment/linear weights and public geometry; not a total-error CI',
            response_history_covered=covered,fit_samples=counts.tolist(),fit_span_s=spans.tolist(),
            noise=noise,device_specs=self.sensor_specs,mode_event=self._last_discontinuity,
            no_global_admission=True,model_error_is_not_measurement_variance=True,
            observed_signed_tangent_response_identified=False,known_response_inputs=scope.tolist(),
            known_history_input8=history_scope.tolist(),unknown_executed_increments=unknown_events,
            response_scope_reason=self._scope_reason,unknown_inputs='remain joint total-output nuisance; not point-subtracted')
        diag.update(mean_time=estimate_time,mean_at_current_time=abs(estimate_time-now)<1e-10,
            unidentified_input_changed_since_acquisition=unidentified_recent)
        self.snapshot=BeliefSnapshot(_copy(mean),_copy((cov+cov.T)*.5),_copy(valid),_copy(age),sample,now,estimate_time,
            _copy(np.asarray(frame.normal,float)),_copy(contact_valid),_copy(np.asarray(frame.shear_magnitude,float)),
            _copy(self._active),_copy(contact_valid),_copy(np.asarray(getattr(frame,'tangential',np.full((5,2),np.nan)),float)),
            _copy(np.asarray(getattr(frame,'finger_non_normal_by_finger',np.full((5,6),np.nan)),float)),
            _copy(np.asarray(getattr(frame,'auxiliary_wrench',np.full(6,np.nan)),float)),
            _copy(slope),_copy(fit),bool(good.any()),diag)
        return self.snapshot

    def forecast(self,horizon=3,period=RAMP_PERIOD):
        if self.snapshot is None:raise RuntimeError('Acquire public data before forecasting')
        if type(horizon) is not int or horizon<1 or abs(period-RAMP_PERIOD)>1e-10:
            raise ValueError('Forecast only the calibrated 20ms total-increment ramp definition')
        s=self.snapshot;now=s.current_time;P=np.zeros((DIM*horizon,8*horizon))
        c=np.empty(DIM*horizon);cal=np.empty_like(c);fits=np.tile(np.nan_to_num(self._fit,nan=np.inf),horizon)
        weights=[];reasons=[];valid=np.tile(s.valid16,horizon)
        if not s.observation_available:reasons.append('observation_unavailable')
        if self.sensor_specs is None:reasons.append('public_device_specification_unavailable')
        if not self._current_input_scope.any():reasons.append('no_known_local_input_response')
        if not self._last_noise_diagnostics.get('wrench_geometry_available',False):reasons.append('wrench_noise_geometry_unavailable')
        if self._unidentified_recent_input:reasons.append('unidentified_input_changed_since_acquisition')
        if not s.valid16.all():reasons.append('posterior_output_missing_or_stale')
        inactive=self._active_known&~self._active
        required=np.ones(DIM,bool);required[np.r_[np.flatnonzero(inactive),np.flatnonzero(inactive)+5]]=False
        if np.any(required&((self._counts<5)|(self._spans<self.minimum_linear_span_s-1e-10))):reasons.append('mode_history_not_yet_identified')
        noise_sigma=np.sqrt(np.maximum(0.,np.diag(self._R[-1]))) if self._R is not None and len(self._R) else np.full(DIM,np.inf)
        empirical_reference=np.maximum(self.response.error_by_horizon.max(axis=0),6.*noise_sigma)
        if np.any(required&np.isfinite(self._fit)&(3.*self._fit>empirical_reference)):
            reasons.append('unexplained_local_dynamics')
        for j in range(horizon):
            h=(j+1)*period;rows=slice(DIM*j,DIM*(j+1))
            extrapolation=now+h-self._posterior_time
            c[rows]=self._residual_now+extrapolation*self._slope+self.response.input_effect(now+h,now)
            for k in range(j+1):P[rows,8*k:8*(k+1)]=self.response.kernel_at((j-k+1)*period)
            cal[rows]=self.response.calibration_at(np.where(np.isfinite(s.age16),s.age16+h,h))
            W=self._weights+extrapolation*self._slope_weights;W=W.copy()
            for i in np.flatnonzero(inactive):
                for row in (i,i+5):P[DIM*j+row]=0.;c[DIM*j+row]=0.;W[row]=0.
            weights.append(W)
        cov=np.zeros((DIM*horizon,DIM*horizon))
        for j in range(horizon):
            for k in range(horizon):
                cov[DIM*j:DIM*(j+1),DIM*k:DIM*(k+1)]=np.einsum('ik,kij,jk->ij',weights[j],self._R,weights[k])
        joint=np.zeros((DIM*(horizon+1),DIM*(horizon+1)))
        all_weights=[self._weights]+weights
        for j in range(horizon+1):
            for k in range(horizon+1):
                joint[DIM*j:DIM*(j+1),DIM*k:DIM*(k+1)]=np.einsum('ik,kij,jk->ij',all_weights[j],self._R,all_weights[k])
        absent=np.tile(self._counts==0,horizon);cov[absent,:]=np.nan;cov[:,absent]=np.nan
        absent_joint=np.tile(self._counts==0,horizon+1);joint[absent_joint,:]=np.nan;joint[:,absent_joint]=np.nan
        raw=c.reshape(horizon,DIM)
        if not np.isfinite(c).all():reasons.append('nonfinite_forecast')
        else:
            if raw[:,:5].min()<-1e-8:reasons.append('negative_active_pressure_forecast')
            if raw[:,5:10].min()<-1e-8:reasons.append('unsigned_shear_mode_unknown')
        if horizon*period>self.response.error_times[-1]+1e-9:reasons.append('forecast_horizon_not_validated')
        if np.any(required&(s.age16+horizon*period>self.response.error_times[-1]+1e-9)):
            reasons.append('observation_age_plus_horizon_not_validated')
        reasons.extend(self._external_mode_reasons)
        diag=dict(single_posterior=True,mean_source='the same residual posterior as snapshot.mean16',
            covariance_scope='measurement-conditioned realized regression weights including cross-horizon correlation',
            calibration_error_separate=True,fit_error_separate=True,signed_tangent_kernel_available=False,
            actual_input_history_only=True,global_admitted=False,cumulative_domain_certified=False,
            known_input8=self._current_input_scope.tolist(),
            P_scope='Measured P columns may be used only with known_input8; unknown is not a zero physical gain',
            unknown_history_scope='absorbed as nuisance in the same total-output posterior, never retroactively claimed',
            response_source=self.response.source_path,response_sha256=self.response.source_sha256,
            ramp_period_s=RAMP_PERIOD,arbitrary_5ms_impulse_model_certified=False,
            fit_reference=empirical_reference.tolist(),mode_event=self._last_discontinuity)
        diag.update(mean_time=self._posterior_time,current_time=now,
            joint_covariance_times=[self._posterior_time]+[now+(j+1)*period for j in range(horizon)],
            joint_covariance_scope='same total-output posterior and future estimates; device measurement contribution only')
        return BeliefForecast(_copy(P),_copy(c),_copy(cal),_copy((cov+cov.T)*.5),
            _copy(np.sqrt(np.maximum(0.,np.diag(cov)))),_copy(fits),_copy(valid),not reasons,
            s.observation_available,tuple(dict.fromkeys(reasons)),diag,_copy(self._current_input_scope),_copy((joint+joint.T)*.5))

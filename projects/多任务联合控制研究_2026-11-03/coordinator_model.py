"""One causal 16-output/8-input contact-response memory for research V5.

Outputs: pressure5, distributed shear magnitude5, full net force3/moment3.
Inputs: alpha5 and executed lift/WRJ2/WRJ1 references. The complete measured
kernel includes passive reactions. Neither shear demand nor palm wrench is
frozen while planning pressure changes. This remains a local experimental
model, not a global contact/friction or motion-admission certificate.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np

PUBLIC_MODEL=Path(__file__).resolve().parent/'calibration/public_contact_model_phase2_v1.json'


@dataclass
class OutputForecast:
    output_now: np.ndarray
    drift_rate: np.ndarray
    drift_error_rate: np.ndarray
    fit_error: np.ndarray
    fit_samples: np.ndarray
    slope_confidence: np.ndarray
    output_valid: np.ndarray
    last_measured: np.ndarray
    last_measurement_time: np.ndarray
    motion_identified: bool=True

    @property
    def normal_now(self):return self.output_now[:5]


class ContactConditionedModel:
    def __init__(self,calibration,window=.25,minimum_span=.055,document=None):
        self.reference=np.asarray(calibration.get('reference_alpha',calibration.get('model_reference_alpha')),float)
        self.public_error=np.asarray(calibration['response_error_bound'],float)
        if self.reference.shape!=(5,) or self.public_error.shape!=(5,) or np.any(self.public_error<0):
            raise ValueError('Explicit pressure calibration is required')
        shape=calibration.get('config',{}).get('shape',calibration.get('shape'))
        if document is None:
            payload=PUBLIC_MODEL.read_bytes();document=json.loads(payload);self.source_sha256=hashlib.sha256(payload).hexdigest()
        else:self.source_sha256='synthetic_fixture_not_runtime'
        derived=(document.get('schema')=='public_contact_response_kernel_16x8_phase2_v1'
                 and document.get('derived_offline_no_new_physics') is True
                 and document.get('calibration_not_performance') is True)
        if not document.get('execution_complete') and not derived:
            raise ValueError('Incomplete or unidentified contact-response experiment')
        entries=[x for x in document['shapes'] if x['shape']==shape]
        if len(entries)!=1:raise ValueError('Exactly one declared shape required')
        if not entries[0].get('execution_complete'):
            e=entries[0]
            if not derived or int(e.get('executed_branches',-1))+len(e.get('unavailable_branches',[]))!=int(e.get('planned_branches',-2)):
                raise ValueError('Completed and unavailable calibration branches must account for the full plan')
        entry=entries[0];kernel=entry['coordinates']['current_palm']
        self.source_path=str(PUBLIC_MODEL);self.shape=shape;self.period=.02
        self.horizons=np.asarray(kernel['response_kernel_times_s'],float)
        self.K=np.asarray(kernel['response_to_20ms_interval_increment'],float)
        if self.K.shape!=(len(self.horizons),16,8) or not np.isfinite(self.K).all():
            raise ValueError('Measured 16-output/8-input dense response is required')
        if self.horizons[0]!=0 or not np.allclose(np.diff(self.horizons),.005,atol=1e-9):
            raise ValueError('Actual 5ms dense kernel timestamps required')
        self.available_directions=[dict(x) for x in entry['input_availability']]
        self.input_lower=np.asarray(entry['measured_input_delta_min'],float)
        self.input_upper=np.asarray(entry['measured_input_delta_max'],float)
        self.input_step_limits=np.maximum(np.abs(self.input_lower),np.abs(self.input_upper))
        self.identified_inputs=np.asarray(kernel['identified_inputs'],bool)
        if self.identified_inputs.shape!=(8,):raise ValueError('Explicit input availability required')
        errors=kernel['holdout_complete_causal_by_horizon'];self.error_times=np.array(sorted(float(x) for x in errors))
        differential=np.asarray(kernel['holdout_differential_response']['max_abs_per_output'],float)
        self.error_by_horizon=np.asarray([np.maximum(differential,np.asarray(errors[str(float(t))]['max_abs_per_output'],float))
            for t in self.error_times])
        if self.error_by_horizon.shape!=(len(self.error_times),16) or not np.isfinite(self.error_by_horizon).all():
            raise ValueError('Complete causal and differential errors must be explicit')
        self.error_by_horizon[:,:5]=np.maximum(self.error_by_horizon[:,:5],self.public_error)
        self.window=float(window);self.minimum_span=float(minimum_span)
        self.noise_floor=np.r_[np.full(10,.003),np.full(3,.002),np.full(3,.0001)]
        self.history=[deque(maxlen=100) for _ in range(16)];self.last_sample=np.full(16,-np.inf)
        self.input_history=[];self.origin=None;self.last_update=-np.inf
        self.drift=np.zeros(16);self.beta=np.full(16,np.nan)
        self.last_measured=np.full(16,np.nan);self.last_measured_time=np.full(16,-np.inf)
        self.last_contact=None;self.last_position=None;self.snapshot=None
        self.active_mask=np.ones(5,bool)
        self.last_trajectory_components=None
        self.sensor_characteristics=None
        self.noise_diagnostics=dict(role='legacy_unspecified_device_noise_floor',propagation_available=False)

    def actual_increment(self,start,end):
        """Known reference history only; absence is not a zero increment."""
        if not self.input_history or start<self.input_history[0][0]-1e-9 or end>self.input_history[-1][0]+1e-9:
            return None
        return self._input_at(float(end))-self._input_at(float(start))

    def kernel_at(self,ages):
        ages=np.asarray(ages,float);flat=self.K.reshape(len(self.horizons),-1)
        out=np.stack([np.interp(ages,self.horizons,flat[:,i],left=0.,right=flat[-1,i]) for i in range(128)],axis=-1)
        return out.reshape(ages.shape+(16,8))

    def _input_at(self,times):
        history=np.asarray(self.input_history,float)
        return np.stack([np.interp(times,history[:,0],history[:,i+1]) for i in range(8)],axis=-1)

    def record_input(self,time,value):
        time=float(time);value=np.asarray(value,float)
        if value.shape!=(8,) or not np.isfinite(value).all():raise ValueError('Finite actual input8 required')
        if self.origin is None:self.origin=time
        if self.input_history and time<self.input_history[-1][0]-1e-9:raise ValueError('Nonchronological input')
        row=np.r_[time,value]
        if self.input_history and abs(time-self.input_history[-1][0])<1e-9:self.input_history[-1]=row
        else:self.input_history.append(row)

    def input_effect(self,query_time,available_through=None,input_columns=None):
        if not self.input_history:return np.zeros(16)
        cutoff=min(float(query_time) if available_through is None else float(available_through),float(self.input_history[-1][0]))
        if cutoff<=self.origin+1e-10:return np.zeros(16)
        count=int(np.ceil((cutoff-self.origin)/self.period-1e-9))
        first=min(count,max(0,int(np.floor((float(query_time)-self.horizons[-1]-self.origin)/self.period))))
        selection=np.ones(8) if input_columns is None else np.isin(np.arange(8),input_columns).astype(float)
        effect=self.K[-1]@((self._input_at(self.origin+first*self.period)-self._input_at(self.origin))*selection)
        starts=self.origin+np.arange(first,count)*self.period
        if not len(starts):return effect
        ends=np.minimum(starts+self.period,cutoff);fractions=np.clip((ends-starts)/self.period,1e-10,1.)
        increments=(self._input_at(ends)-self._input_at(starts))/fractions[:,None]*selection
        return effect+np.einsum('koi,ki->o',self.kernel_at(float(query_time)-starts),increments)

    def set_sensor_characteristics(self,value):
        """A shared public noise specification, never an environment schedule."""
        if not value:return
        force=float(value.get('force_sigma_N',.01));shear=float(value.get('tangential_sigma_N',.005))
        auxiliary=float(value.get('auxiliary_force_sigma_N',force))
        moment=float(value.get('auxiliary_moment_sigma_Nm',.05*auxiliary))
        position=float(value.get('position_sigma_m',0.))
        if not np.isfinite([force,shear,auxiliary,moment,position]).all() or min(force,shear,auxiliary,moment,position)<0.:
            raise ValueError('Public sensor noise sigmas must be finite and nonnegative')
        self.noise_floor[:5]=max(1e-12,force)
        # The public distributed shear magnitude receives t_noise[:,0].
        self.noise_floor[5:10]=max(1e-12,shear)
        self.sensor_characteristics=dict(value)
        self.noise_floor[10:13]=max(1e-12,auxiliary)
        self.noise_floor[13:16]=max(1e-12,moment)

    @staticmethod
    def _skew(value):
        x,y,z=value
        return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

    def _wrench_noise(self,frame):
        if self.sensor_characteristics is None:return
        M=getattr(frame,'normal_wrench_map',None);B=getattr(frame,'tangent_basis',None)
        X=getattr(frame,'contact_pos',None);T=getattr(frame,'finger_non_normal_by_finger',None)
        if any(x is None for x in (M,B,X,T)):
            self.noise_diagnostics=dict(role='public_device_noise',propagation_available=False,reason='missing_public_wrench_geometry');return
        M=np.asarray(M,float);B=np.asarray(B,float);X=np.asarray(X,float);T=np.asarray(T,float);p=np.asarray(frame.normal,float)
        if M.shape!=(6,5) or B.shape!=(5,2,3) or X.shape!=(5,3) or T.shape!=(5,6) or not np.isfinite(np.r_[M.ravel(),B.ravel(),X.ravel(),T.ravel(),p]).all():
            self.noise_diagnostics=dict(role='public_device_noise',propagation_available=False,reason='invalid_public_wrench_geometry');return
        spec=self.sensor_characteristics;force=float(spec.get('force_sigma_N',.01))
        tangent=float(spec.get('tangential_sigma_N',.005));position=float(spec.get('position_sigma_m',0.))
        auxiliary=float(spec.get('auxiliary_force_sigma_N',force))
        moment=float(spec.get('auxiliary_moment_sigma_Nm',.05*auxiliary))
        covariance=np.diag(np.r_[np.full(3,auxiliary**2),np.full(3,moment**2)])
        present=np.asarray(getattr(frame,'contact_present',p>.02),bool)
        for i in range(5):
            tangent_basis=B[i].T
            Jt=np.vstack((tangent_basis,self._skew(X[i])@tangent_basis))
            estimated_force=M[:3,i]*p[i]+T[i,:3]
            Jx=np.vstack((np.zeros((3,3)),-self._skew(estimated_force)))
            position_variance=position**2 if present[i] else 0.
            covariance+=force**2*np.outer(M[:,i],M[:,i])+tangent**2*(Jt@Jt.T)+position_variance*(Jx@Jx.T)
            # Positive second-order position x tangent/normal-noise terms.
            local_force_noise=tangent**2*(tangent_basis@tangent_basis.T)+force**2*np.outer(M[:3,i],M[:3,i])
            covariance[3:,3:]+=position_variance*(np.trace(local_force_noise)*np.eye(3)-local_force_noise)
        self.noise_floor[10:]=np.maximum(1e-12,np.sqrt(np.maximum(0.,np.diag(covariance))))
        self.noise_diagnostics=dict(role='statistical_model_mode_gate_not_hard_error_bound',propagation_available=True,
            marginal_sigma=self.noise_floor[10:].tolist(),covariance=covariance.tolist(),
            assumptions='independent known device draws; first-order public noisy geometry/force plug-in plus positive second-order cross-noise terms; not exact distribution or deterministic coverage',
            normal_sigma_N=force,tangent_sigma_N=tangent,position_sigma_m=position,
            auxiliary_force_sigma_N=auxiliary,auxiliary_moment_sigma_Nm=moment,
            residual_fit_sigma_multiplier=3.,mode_gate_noise_sigma_multiplier=6.,
            calibration_error_modified=False)

    def assess(self,raw_trajectory):
        """A bad local mode is unknown, not an invented need to change force.

        In particular an unsigned shear extrapolation below zero does not
        create a physical command constraint. Reject that forecast as a whole;
        keep the observed feedback task and the original unmodified error.
        """
        if self.snapshot is None:return dict(usable=False,reasons=['no_causal_snapshot'])
        s=self.snapshot;raw=np.asarray(raw_trajectory,float).reshape(-1,16);reasons=[]
        if not np.all(s.output_valid):reasons.append('incomplete_fresh_contact_wrench_observation')
        if self.sensor_characteristics is not None and not self.noise_diagnostics.get('propagation_available'):
            reasons.append('public_wrench_noise_geometry_unavailable')
        if not np.isfinite(raw).all():reasons.append('nonfinite_forecast')
        else:
            if np.min(raw[:,:5])<-1e-8:reasons.append('negative_active_pressure_forecast')
            if np.min(raw[:,5:10])<-1e-8:reasons.append('unsigned_shear_forecast_outside_mode')
        expected=np.maximum(np.max(self.error_by_horizon,axis=0),6.*self.noise_floor)
        unexplained=s.fit_error>expected
        if np.any(unexplained):reasons.append('contact_dynamics_residual_exceeds_measured_model_error')
        return dict(usable=not reasons,reasons=reasons,
            fit_error=s.fit_error.tolist(),local_residual_reference=expected.tolist(),
            unexplained_outputs=np.flatnonzero(unexplained).tolist(),
            calibration_error_preserved=True,noise_gate=self.noise_diagnostics,global_domain_admitted=False)

    @staticmethod
    def measure(frame,state):
        p=np.asarray(frame.normal,float);tau=np.asarray(frame.shear_magnitude,float)
        good=np.asarray(state.valid,bool)&np.isfinite(p)&np.isfinite(tau)
        y=np.r_[p,tau,np.full(6,np.nan)];valid=np.r_[good,good,np.zeros(6,bool)]
        M=getattr(frame,'normal_wrench_map',None);T=getattr(frame,'finger_non_normal_by_finger',None);A=getattr(frame,'auxiliary_wrench',None)
        if np.all(good) and getattr(frame,'auxiliary_valid',False) and M is not None and T is not None and A is not None:
            M=np.asarray(M,float);T=np.asarray(T,float);A=np.asarray(A,float)
            if M.shape==(6,5) and T.shape==(5,6) and A.shape==(6,) and np.isfinite(np.r_[M.ravel(),T.ravel(),A]).all():
                y[10:]=M@p+T.sum(axis=0)+A;valid[10:]=True
        return y,valid

    def update(self,frame,state,executed_motion=None):
        now=float(frame.time);sample=float(frame.sample_time)
        motion=np.r_[float(frame.lift_ref),0.,0.] if executed_motion is None else np.asarray(executed_motion,float)
        self.record_input(now,np.r_[np.asarray(frame.alpha,float),motion])
        if abs(now-self.last_update)<1e-10:return self.snapshot
        self.last_update=now;self._wrench_noise(frame);y,valid=self.measure(frame,state)
        public_contact=getattr(frame,'contact_present',None)
        seen=np.asarray(state.valid,bool)
        observed=(np.asarray(state.contact_model_active,bool) if public_contact is None else np.asarray(public_contact,bool))
        # A dropped observation is unknown, not an observed absence of contact.
        self.active_mask[seen]=observed[seen]
        contact=self.active_mask.copy();positions=np.asarray(frame.contact_pos,float)
        if self.last_contact is not None:
            jump=contact!=self.last_contact
            if self.last_position is not None:jump|=np.linalg.norm(positions-self.last_position,axis=1)>.012
            for i in np.flatnonzero(jump):
                for row in (i,i+5):self.history[row].clear();self.drift[row]=0.
            if np.any(jump):
                for row in range(10,16):self.history[row].clear();self.drift[row]=0.
        self.last_contact=contact.copy();self.last_position=positions.copy()
        acquired=self.input_effect(sample,sample);current=self.input_effect(now,now)
        fit_error=3.*self.noise_floor;slope_error=np.r_[np.full(13,.5),np.full(3,.025)]
        counts=np.zeros(16,int);confidence=np.zeros(16)
        for i in range(16):
            if valid[i] and sample>self.last_sample[i]+1e-9:
                self.history[i].append((sample,float(y[i]-acquired[i])));self.last_sample[i]=sample
                self.last_measured[i]=y[i];self.last_measured_time[i]=sample
            while self.history[i] and self.history[i][0][0]<sample-self.window:self.history[i].popleft()
            values=np.asarray(self.history[i],float);counts[i]=len(values)
            if len(values)>=5 and values[-1,0]-values[0,0]>=self.minimum_span:
                ts=values[:,0];values_y=values[:,1];center=float(ts.mean());x=ts-center;xx=float(x@x)
                raw=float(x@(values_y-values_y.mean())/max(1e-12,xx));residual=values_y-(values_y.mean()+raw*x)
                sigma=max(float(self.noise_floor[i]),float(np.sqrt(np.mean(residual**2))));se=sigma/np.sqrt(max(1e-12,xx))
                confidence[i]=raw*raw/(raw*raw+(2.*se)**2+1e-20)
                self.drift[i]=np.clip(raw*confidence[i],-5. if i<13 else -.25,5. if i<13 else .25)
                self.beta[i]=values_y.mean()+self.drift[i]*(now-center)
                fit_error[i]=3.*sigma;slope_error[i]=max(3.*se,abs(raw-self.drift[i]))
            elif len(values):self.beta[i]=values[-1,1];self.drift[i]=0.
            else:self.beta[i]=np.nan;self.drift[i]=0.
        output=self.beta+current;fresh=(now-self.last_measured_time<=.12)&np.isfinite(output)
        # A measured inactive skin/contact mode has conditional pressure and
        # shear zero until recontact. This does not clip an active affine model
        # or use target identity; inactive pressure noise is not physical load.
        for finger in np.flatnonzero(~self.active_mask):
            for row in (finger,finger+5):output[row]=0.;self.drift[row]=0.
        self.snapshot=OutputForecast(output,self.drift.copy(),slope_error,fit_error,counts,confidence,fresh,
            self.last_measured.copy(),self.last_measured_time.copy(),bool(np.all(self.identified_inputs[5:])))
        return self.snapshot

    def trajectory(self,state,horizon,period,*unused):
        if self.snapshot is None:raise RuntimeError('Acquire model before prediction')
        if abs(period-self.period)>1e-10:raise ValueError('20ms finite ramps required')
        s=self.snapshot;now=float(self.input_history[-1][0]);P=np.zeros((16*horizon,8*horizon))
        c=np.zeros(16*horizon);uncertainty=np.zeros(16*horizon)
        calibration_error=np.zeros_like(c);fit_error=np.zeros_like(c);slope_error=np.zeros_like(c)
        memory_change=np.zeros_like(c);drift_change=np.zeros_like(c)
        present_memory=self.input_effect(now,now)
        for j in range(horizon):
            age=(j+1)*period;rows=slice(16*j,16*(j+1))
            c[rows]=self.beta+s.drift_rate*age+self.input_effect(now+age,now)
            for k in range(j+1):P[rows,8*k:8*(k+1)]=self.kernel_at((j-k+1)*period)
            errors=np.asarray([np.interp(age,self.error_times,self.error_by_horizon[:,i]) for i in range(16)])
            uncertainty[rows]=errors+s.fit_error+age*s.drift_error_rate
            calibration_error[rows]=errors;fit_error[rows]=s.fit_error;slope_error[rows]=age*s.drift_error_rate
            memory_change[rows]=self.input_effect(now+age,now)-present_memory
            drift_change[rows]=age*s.drift_rate
            for finger in np.flatnonzero(~self.active_mask):
                for row in (finger,finger+5):P[16*j+row,:]=0.;c[16*j+row]=0.
        self.last_trajectory_components=dict(calibration_error=calibration_error,fit_error=fit_error,
            slope_error=slope_error,past_input_memory_change=memory_change,residual_drift_change=drift_change)
        return P,c,uncertainty


def synthetic_document(K=None):
    times=np.arange(101)*.005
    if K is None:
        G=np.zeros((16,8));G[:5,:5]=np.eye(5)*10.;G[5:10,:5]=np.eye(5)*6.
        G[10:13,:5]=.001;G[10:13,5:]=np.eye(3)*2.
        K=np.asarray([G*(1.-np.exp(-t/.02)) for t in times])
    error=dict(max_abs_per_output=np.zeros(16).tolist())
    coordinates=dict(response_kernel_times_s=times.tolist(),response_to_20ms_interval_increment=np.asarray(K).tolist(),
        identified_inputs=[True]*8,holdout_differential_response=error,
        holdout_complete_causal_by_horizon={str(t):error for t in (.02,.04,.06,.1)})
    entries=[dict(shape=shape,execution_complete=True,input_availability=[dict(axis=i,plus=not(i==5 or shape=='sphere' and i==1),minus=True) for i in range(8)],
        measured_input_delta_min=[-.006]*5+[-.0008,-.003,-.003],
        measured_input_delta_max=[.006]*5+[0.,.003,.003],coordinates={'current_palm':coordinates}) for shape in ('capsule','box','sphere')]
    return dict(execution_complete=True,model_admitted=False,shapes=entries)


def verify_model_algebra():
    from types import SimpleNamespace
    cal=dict(reference_alpha=np.zeros(5),response_error_bound=np.full(5,.508),config={'shape':'capsule'})
    model=ContactConditionedModel(cal,document=synthetic_document());delta=np.array([.003,-.002,.001,0.,-.001,-.0005,.002,0.])
    for k in range(61):model.record_input(k*.005,delta*min(1.,k/4.))
    checks={}
    checks['same_actual_ramp_kernel_pressure_shear_wrench']=all(np.allclose(model.input_effect(t),model.kernel_at(t)@delta) for t in (.02,.04,.06,.1,.2))
    checks['delayed_effect_uses_only_acquired_input']=bool(np.allclose(model.input_effect(.01,.01),model.kernel_at(.01)@delta))
    old=model.input_effect(.1,.1).copy();model.record_input(.31,delta+1.)
    checks['future_actual_input_does_not_change_past_residual']=bool(np.allclose(old,model.input_effect(.1,.1)))
    checks['full_public_pressure_error_not_truncated']=bool(np.min(model.error_by_horizon[:,:5])>=.508)
    checks['shear_changes_with_same_input_as_pressure']=bool(np.allclose(model.K[4,5:10,:5],.6*model.K[4,:5,:5]))
    fresh=ContactConditionedModel(cal,document=synthetic_document());state=SimpleNamespace(valid=np.ones(5,bool),contact_model_active=np.ones(5,bool))
    slope=np.r_[.2,np.zeros(15)];base=np.r_[np.full(5,2.),np.full(5,1.2),[0.,0.,1.2,0.,0.,0.]]
    for k in range(71):
        t=k*.005;inputs=delta*min(1.,k/4.);fresh.record_input(t,inputs);y=base+slope*t+fresh.input_effect(t)
        f=SimpleNamespace(time=t,sample_time=t,normal=y[:5],shear_magnitude=y[5:10],alpha=inputs[:5],lift_ref=inputs[5],
            contact_pos=np.zeros((5,3)),normal_wrench_map=np.zeros((6,5)),finger_non_normal_by_finger=np.zeros((5,6)),auxiliary_wrench=y[10:],auxiliary_valid=True)
        fresh.update(f,state,inputs[5:])
    checks['same_kernel_residual_preserves_true_slow_drift']=bool(abs(fresh.snapshot.drift_rate[0]-.2)<.01)
    checks['static_residual_slopes_shrink']=bool(np.max(np.abs(fresh.snapshot.drift_rate[1:]))<1e-10)
    f.auxiliary_valid=False;y,valid=fresh.measure(f,state)
    checks['missing_palm_cannot_become_zero_wrench']=bool(np.isnan(y[10:]).all() and not valid[10:].any())
    warm=ContactConditionedModel(cal,document=synthetic_document())
    f.time=7.3;f.sample_time=7.29;f.alpha=np.zeros(5);f.lift_ref=.1;f.auxiliary_valid=True
    warm.update(f,state)
    f.time=7.305;f.sample_time=7.295;warm.update(f,state,np.array([.1,0.,0.]))
    checks['warmup_actual_lift_reference_not_a_false_point_one_step']=bool(np.allclose(warm.input_effect(7.32),0.))
    f.time=7.31;f.sample_time=7.3;f.contact_present=np.array([True,False,True,True,True]);f.normal=np.full(5,.01)
    warm.update(f,state,np.array([.1,0.,0.]));PP,cc,ee=warm.trajectory(state,3,.02)
    inactive_rows=np.array([1,6,17,22,33,38])
    checks['public_inactive_mode_is_conditional_zero_not_negative_pressure_noise']=bool(
        np.all(PP[inactive_rows]==0.) and np.all(cc[inactive_rows]==0.) and np.all(ee[[1,17,33]]>=.508))
    components=warm.last_trajectory_components
    checks['full_uncertainty_has_auditable_calibration_fit_and_slope_parts']=bool(np.allclose(
        ee,components['calibration_error']+components['fit_error']+components['slope_error']))
    f.time=7.315;f.sample_time=7.305;f.contact_present=np.ones(5,bool);f.normal=np.full(5,2.)
    warm.update(f,state,np.array([.1,0.,0.]));previous_active=warm.active_mask.copy()
    f.time=7.32;f.sample_time=7.31;f.contact_present=np.zeros(5,bool);state.valid[:]=False
    warm.update(f,state,np.array([.1,0.,0.]))
    checks['observation_dropout_is_not_measured_contact_absence']=bool(np.array_equal(warm.active_mask,previous_active))
    checks['missing_actual_input_interval_is_unknown_not_zero']=warm.actual_increment(7.,7.32) is None
    if not all(checks.values()):raise AssertionError(checks)
    return checks

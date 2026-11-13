"""V5: minimum necessary intervention on an explicit eight-input task.

Nominal action holds finger coordinates and follows bounded user motion.
It passes unchanged when the implemented execution/force/contact checks pass.
Otherwise: force protection -> contact compatibility -> task progress -> least
correction and correction continuity. No historical pressure point is tracked.
The measured local model remains experimental; passing these checks is not a
global model admission or a physical counterfactual safety certificate.
"""
from dataclasses import dataclass,asdict
import numpy as np
import research_imports
from vendor.frozen_controllers import make_gains,solve_qp,ControlAction
from state_estimator import CoupledTactileState
from coordinator_model import ContactConditionedModel,OutputForecast,synthetic_document,verify_model_algebra
from coordinator_contacts import ContactCorridorObserver,verify_contact_membership


@dataclass
class CoordinatorSettings:
    motion_rates: tuple=(.035,.15,.10)
    forecast_margin_N: float=.05
    priority_tolerance_N: float=.0005
    progress_guard_tolerance: float=1e-6
    qp_passes: int=240


class _ObservedState(CoupledTactileState):
    def __init__(self,gains,owner):super().__init__(gains);self.owner=owner
    def update(self,frame,dt):
        super().update(frame,dt)
        o=self.owner;prediction=o.model.update(frame,self,o.executed_motion if o.motion_initialized else None)
        o.contacts.update(frame,self,prediction)
        limits,budget=o._force_limits()
        o.contacts.adopt_stable_reference(prediction,limits,budget,float(frame.time),o.last_intervention)


class JointCoordinator:
    def __init__(self,calibration,limits,budget,gains=None,model_document=None):
        B=np.asarray(calibration['force_matrix'],float);g=dict(gains or {})
        g.update(force_gain=tuple(np.maximum(.05,np.linalg.norm(B,axis=0))),force_tau=float(calibration['force_tau']),release_rate=.8)
        self.g=make_gains(g);self.settings=CoordinatorSettings()
        self.model=ContactConditionedModel(calibration,document=model_document)
        self.contacts=ContactCorridorObserver(self.g.nominal_mu)
        self.motion_initialized=False;self.state=_ObservedState(self.g,self)
        self.limits=self._vector(limits,5,'limits');self.budget=float(budget)
        if np.any(self.limits<0.) or not np.isfinite(self.budget) or self.budget<0.:raise ValueError('Nonnegative physical force limits required')
        # Public robot actuator reference bounds, inherited unchanged from the
        # foundation Shadow wrist defaults and configured reachable lift target.
        lift=float(calibration.get('config',{}).get('lift_target',.1))
        self.motion_lower=np.array([0.,-.523599,-.698132]);self.motion_upper=np.array([lift,.174533,.488692])
        self.name='joint';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan)
        self.last_decision=-np.inf;self.previous_time=None;self.last_intervention=-np.inf
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3)
        self.motion_request=np.zeros(3);self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.execution_anchor=None;self.motion_anchor=None;self.previous_plan=None
        self.previous_nominal=None;self.previous_correction=np.zeros(8)
        self.release_latched=False
        self.task_request={};self.debug={};self.label='joint_v5_initializing'
        self.numerical_failures=0;self.infeasible_ticks=0;self.control_ticks=0;self.force_reference=np.zeros(5)

    @staticmethod
    def _vector(value,size,name):
        result=np.asarray(value,float)
        if result.shape!=(size,) or not np.isfinite(result).all():raise ValueError(f'{name} requires {size} finite values')
        return result.copy()

    def set_motion_request(self,target,executed_reference):
        self.motion_request=self._vector(target,3,'motion_target')
        self.executed_motion=self._vector(executed_reference,3,'executed_motion_reference')
        self.motion_reference=self.executed_motion.copy();self.motion_initialized=True

    def set_task_request(self,value):
        """Consume an ordinary public task dict; no schedule or object oracle."""
        if not isinstance(value,dict):raise ValueError('Public task request must be an ordinary dict')
        target=value.get('motion_target',self.motion_request)
        executed=value.get('executed_motion_reference',self.executed_motion)
        self.set_motion_request(target,executed)
        request={}
        for key in ('schema','mode','force_scope','model_domain_transfer','priorities'):
            if key in value:request[key]=value[key].copy() if isinstance(value[key],dict) else value[key]
        for key in ('time_s','task_time_s','deadline_s','remaining_time_s'):
            if value.get(key) is not None:
                x=float(value[key])
                if not np.isfinite(x):raise ValueError(f'{key} must be finite or absent')
                request[key]=x
        if 'limits' in value:
            x=self._vector(value['limits'],5,'task limits')
            if np.any(x<0.):raise ValueError('Task force limits cannot be negative')
            request['limits']=x
        if 'budget' in value:
            x=float(value['budget'])
            if not np.isfinite(x) or x<0.:raise ValueError('Task budget must be finite and nonnegative')
            request['budget']=x
        if 'tracking_tolerances' in value:
            x=list(value['tracking_tolerances'])
            if len(x)!=3 or any(y is not None and (not np.isfinite(float(y)) or float(y)<0.) for y in x):
                raise ValueError('Each tracking tolerance is nonnegative or explicitly unknown')
            request['tracking_tolerances']=[None if y is None else float(y) for y in x]
        self.task_request=request
        if str(request.get('mode','')).lower() in ('release','monitor'):self.release_latched=True

    def _force_limits(self):
        # Task intent may tighten material limits, never silently relax them.
        return np.minimum(self.limits,self.task_request.get('limits',self.limits)),min(self.budget,self.task_request.get('budget',self.budget))

    @staticmethod
    def _cost(Q,q,A,b,weight=1.):
        Q+=2.*weight*(A.T@A);q+=2.*weight*(A.T@b)

    @staticmethod
    def support_band(wrench,uncertainty,up,load,load_uncertainty):
        support=float(np.asarray(up)@wrench[:3]);error=max(.06,float(np.abs(up)@uncertainty[:3]))
        measured=np.array([support-error,support+error]);external=np.full(2,np.nan)
        if np.isfinite(load) and np.isfinite(load_uncertainty) and load_uncertainty>=0.:
            external=np.array([max(0.,load-load_uncertainty),load+load_uncertainty])
            overlap=np.array([max(measured[0],external[0]),min(measured[1],external[1])])
            if overlap[0]<=overlap[1]:return overlap,False,measured,external
        return measured,True,measured,external

    def _unavailable(self,reason,raw=None,residual=0.):
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.predicted=np.full(5,np.nan)
        self.desired=np.zeros(5) if self.model.snapshot is None else self.model.snapshot.normal_now.copy()
        self.force_reference=self.desired.copy();self.label='joint_v5_model_unavailable';self.numerical_failures+=1
        self.debug=dict(controller_architecture='nominal_task_minimum_intervention_16x8_v5',prediction_available=False,
            prediction_invalid_reason=reason,rejected_raw_contact_prediction=[] if raw is None else np.asarray(raw).tolist(),
            qp_primal_residual=float(residual),model_admitted=False,support_infeasibility_certified=False,
            fallback='zero executable correction; local model unavailable, no clipping or physical impossibility claim')

    def _step_bounds(self):
        period=self.g.decision_period
        positive=np.minimum(np.r_[np.full(5,self.g.close_rate*period),np.asarray(self.settings.motion_rates)*period],np.maximum(0.,self.model.input_upper))
        negative=np.minimum(np.r_[np.full(5,self.g.open_rate*period),np.asarray(self.settings.motion_rates)*period],np.maximum(0.,-self.model.input_lower))
        # A missing direction is not measured zero gain. Return to a visited
        # anchor may use the measured opposite linearization experimentally.
        for i,available in enumerate(self.model.available_directions):
            if not available['plus']:positive[i]=min((self.g.close_rate if i<5 else self.settings.motion_rates[i-5])*period,self.model.input_step_limits[i])
            if not available['minus']:negative[i]=min((self.g.open_rate if i<5 else self.settings.motion_rates[i-5])*period,self.model.input_step_limits[i])
        positive[~self.model.identified_inputs]=0.;negative[~self.model.identified_inputs]=0.
        return positive,negative

    def _decision(self):
        s=self.state;f=self.model.snapshot;cs=self.contacts.snapshot;h=self.g.horizon;period=self.g.decision_period;nr=8*h
        if f is None or cs is None or not np.all(f.output_valid[:10]):
            self._unavailable('pressure_or_shear_model_unavailable');return
        if self.execution_anchor is None:
            self.contacts.freeze_handoff(f);self.execution_anchor=s.alpha.copy();self.motion_anchor=self.executed_motion.copy()
        limits,budget=self._force_limits();P,c,error=self.model.trajectory(s,h,period)
        pi=np.concatenate([np.arange(16*j,16*j+5) for j in range(h)])
        ti=np.concatenate([np.arange(16*j+5,16*j+10) for j in range(h)])
        wi=np.concatenate([np.arange(16*j+10,16*j+16) for j in range(h)])
        cp=c[pi];ct=c[ti];ep=error[pi];et=error[ti]
        if not np.isfinite(np.r_[cp,ct,ep,et,P[pi].ravel(),P[ti].ravel()]).all():
            self._unavailable('nonfinite_pressure_or_shear_forecast');return
        positive,negative=self._step_bounds()
        lower=s.closure_min.copy();upper=s.closure_max.copy()
        motion_target=np.clip(self.motion_request,self.motion_lower,self.motion_upper)
        mlower=np.minimum(self.executed_motion,motion_target);mupper=np.maximum(self.executed_motion,motion_target)
        for i,available in enumerate(self.model.available_directions):
            if i<5:
                if not available['plus']:upper[i]=min(upper[i],self.execution_anchor[i])
                if not available['minus']:lower[i]=max(lower[i],self.execution_anchor[i])
            else:
                j=i-5
                if not available['plus']:mupper[j]=min(mupper[j],self.motion_anchor[j])
                if not available['minus']:mlower[j]=max(mlower[j],self.motion_anchor[j])
        mlower=np.maximum(mlower,self.motion_lower);mupper=np.minimum(mupper,self.motion_upper)
        if np.any(lower>s.alpha+1e-8) or np.any(upper<s.alpha-1e-8) or np.any(mlower>self.executed_motion+1e-8) or np.any(mupper<self.executed_motion-1e-8):
            self._unavailable('current_executed_reference_outside_known_authority');return
        operating_target=np.clip(motion_target,mlower,mupper)
        preview=np.asarray([np.clip(operating_target-self.executed_motion,-negative[5:]*(j+1),positive[5:]*(j+1)) for j in range(h)])
        nominal=np.zeros((h,8));nominal[:,5:]=np.diff(np.vstack((np.zeros(3),preview)),axis=0);un=nominal.ravel()
        raw_nominal=(c+P@un).reshape(h,16)
        active=self.model.active_mask&(f.normal_now>.15)&f.output_valid[:5]&f.output_valid[5:10]
        contact_known=bool(active[0] and np.count_nonzero(active)>=3 and np.count_nonzero((cs.mode=='STICK_CONFIRMED')&active)>=3
            and np.all(cs.speed_lower[active]<.002))
        complete=bool(np.all(f.output_valid[10:]) and np.isfinite(c[wi]).all())
        ratio=np.divide(f.output_now[5:10],f.normal_now,out=np.zeros(5),where=f.normal_now>.15)
        # The descriptive stable reference is used only for uncertain contact
        # rows; it is never a historical force target or a pressure corridor.
        use_reference=(cs.mode!='STICK_CONFIRMED')&cs.anchor_valid
        ratio[use_reference]=cs.anchor_ratio[use_reference]
        floor=np.where(active,np.minimum(.15,f.normal_now),0.)
        tube=et+np.tile(ratio,h)*ep+.02
        robust_c=cp+ep+self.settings.forecast_margin_N
        # u[8h], pressure[5]/budget[1], contact-retention[5], shear[5], wrench[7].
        dim=nr+23;ie=slice(nr,nr+5);ib=nr+5;ir=slice(nr+6,nr+11);it=slice(nr+11,nr+16);iw=slice(nr+16,dim)
        I=np.eye(dim);N=np.zeros((5*h,dim));N[:,:nr]=P[pi];T=np.zeros_like(N);T[:,:nr]=P[ti]
        rows=[];rhs=[]
        def constraint(A,b):rows.append(np.atleast_2d(A));rhs.append(np.atleast_1d(b))
        constraint(I[:nr],np.tile(positive,h));constraint(-I[:nr],np.tile(negative,h))
        C=np.kron(np.tril(np.ones((h,h))),np.eye(8));CU=np.zeros((nr,dim));CU[:,:nr]=C
        apos=np.concatenate([np.arange(8*j,8*j+5) for j in range(h)]);mpos=np.concatenate([np.arange(8*j+5,8*j+8) for j in range(h)])
        constraint(CU[apos],np.tile(upper-s.alpha,h));constraint(-CU[apos],np.tile(s.alpha-lower,h))
        constraint(CU[mpos],np.tile(mupper-self.executed_motion,h));constraint(-CU[mpos],np.tile(self.executed_motion-mlower,h))
        blocked=~s.valid|~s.closure_permission
        if np.any(blocked):constraint(I[apos][np.tile(blocked,h)],np.zeros(np.count_nonzero(blocked)*h))
        missing=~self.model.identified_inputs
        if np.any(missing):
            Z=I[:nr][np.tile(missing,h)];constraint(Z,np.zeros(len(Z)));constraint(-Z,np.zeros(len(Z)))
        constraint(-N,cp);constraint(-T,ct)
        hard_A=np.vstack(rows);hard_b=np.concatenate(rhs)
        for sl in (ie,ir,it,iw):constraint(-I[sl],np.zeros(sl.stop-sl.start))
        constraint(-I[ib],0.)
        Eu=np.tile(I[ie],(h,1));Er=np.tile(I[ir],(h,1));Et=np.tile(I[it],(h,1));Eb=np.tile(I[ib],(h,1))
        constraint(N-Eu,np.tile(limits,h)-robust_c)
        sums=np.kron(np.eye(h),np.ones((1,5)));constraint(sums@N-Eb,np.full(h,budget)-sums@robust_c)
        constraint(-N-Er,cp-np.tile(floor,h))
        constraint(T-np.tile(ratio,h)[:,None]*N-Et,tube-(ct-np.tile(ratio,h)*cp))
        WN=None;wc=None;wl=None;wh=None;support_info={}
        if complete:
            projection=np.vstack((np.eye(6),np.r_[s.up,np.zeros(3)]));PH=np.kron(np.eye(h),projection)
            WN=np.zeros((6*h,dim));WN[:,:nr]=P[wi];WN=PH@WN;wc=PH@c[wi]
            me=np.maximum(f.fit_error[10:],np.r_[np.full(3,.06),np.full(3,.003)])
            band,inconsistent,unused,load_band=self.support_band(f.output_now[10:],me,s.up,float(s.load),float(s.load_uncertainty))
            we=(np.abs(PH)@error[wi]).reshape(h,7)
            wl=np.r_[f.output_now[10:]-me,band[0]]-we;wh=np.r_[f.output_now[10:]+me,band[1]]+we
            EW=np.tile(I[iw],(h,1));constraint(WN-EW,wh.ravel()-wc);constraint(-WN-EW,wc-wl.ravel())
            support_info=dict(support_band_N=band.tolist(),load_interval_N=load_band.tolist(),load_wrench_interval_inconsistent=bool(inconsistent))
        A=np.vstack(rows);b=np.concatenate(rhs)
        def slack_completion(u):
            x=np.zeros(dim);x[:nr]=u;p=(N@x+cp).reshape(h,5);tau=(T@x+ct).reshape(h,5)
            x[ie]=np.maximum(0.,p+ep.reshape(h,5)+self.settings.forecast_margin_N-limits).max(axis=0)
            x[ib]=max(0.,float((p+ep.reshape(h,5)+self.settings.forecast_margin_N).sum(axis=1).max()-budget))
            x[ir]=np.maximum(0.,floor-p).max(axis=0)
            x[it]=np.maximum(0.,tau-ratio*p-tube.reshape(h,5)).max(axis=0)
            if complete:
                w=(WN@x+wc).reshape(h,7);x[iw]=np.maximum(0.,np.maximum(w-wh,wl-w)).max(axis=0)
            return x
        nominal_x=slack_completion(un)
        observed_guard=bool(np.all(f.last_measured[:5]<=limits) and f.last_measured[:5].sum()<=budget)
        check=dict(execution_authority=bool(np.max(hard_A@nominal_x-hard_b)<=1e-8),
            measured_force_guard=observed_guard,predicted_force_guard=bool(np.max(nominal_x[ie])<=1e-8 and nominal_x[ib]<=1e-8),
            contact_state_available=contact_known,contact_retention=bool(np.max(nominal_x[ir])<=1e-8),
            shear_compatibility=bool(np.max(nominal_x[it])<=1e-8),complete_wrench_available=complete,
            complete_wrench_compatibility=bool(complete and np.max(nominal_x[iw])<=1e-8))
        previous_actual=self.model.actual_increment(float(self.model.input_history[-1][0])-period,float(self.model.input_history[-1][0]))
        self.previous_correction=np.zeros(8) if previous_actual is None or self.previous_nominal is None else previous_actual-self.previous_nominal
        statuses=[];residuals=[];checked_residuals=[];fallbacks=[];nominal_pass=all(check.values())
        final_A=A;final_b=b
        if nominal_pass:
            x=nominal_x;self.label='joint_v5_nominal_task_unmodified'
        else:
            # Missing contact/task evidence is not a fictitious zero wrench or
            # a material capacity certificate. Hold motion while protecting
            # observable pressure, and expose the incomplete task model.
            if not contact_known or not complete:
                A=np.vstack((A,I[mpos],-I[mpos]));b=np.r_[b,np.zeros(6*h)]
            seed=slack_completion(np.zeros(nr))
            seed[ie]+=.001;seed[ib]+=.001;seed[ir]+=.001;seed[it]+=.001;seed[iw]+=.001
            if np.max(A@seed-b)>1e-8:
                repair=solve_qp(np.eye(dim),np.zeros(dim),A,b,self.settings.qp_passes,2e-5)
                repair_actual=float(np.max(A@repair.x-b)) if np.isfinite(repair.x).all() else float('inf')
                if repair.residual>1e-8 or repair_actual>1e-8:
                    self._unavailable('no_checked_nonnegative_contact_trajectory',np.r_[cp,ct],max(repair.residual,repair_actual));return
                seed=repair.x.copy()
            def optimize(Q,q,AA,bb,initial):
                result=solve_qp(Q,q,AA,bb,self.settings.qp_passes,2e-5,feasible_initial=initial)
                statuses.append(result.status);residuals.append(float(result.residual))
                actual=float(np.max(AA@result.x-bb)) if np.isfinite(result.x).all() else float('inf')
                if result.residual<=2e-5 and actual<=2e-5:
                    candidate=result.x.copy();fallbacks.append(None)
                else:
                    candidate=initial.copy();fallbacks.append('checked_previous_layer')
                checked=float(np.max(AA@candidate-bb));checked_residuals.append(checked)
                if not np.isfinite(checked) or checked>2e-5:
                    raise RuntimeError('Neither solver output nor previous priority layer is checked feasible')
                if not result.converged:self.numerical_failures+=1
                return candidate
            norm=np.maximum(.5,limits);Q=np.eye(dim)*1e-8;q=np.zeros(dim)
            self._cost(Q,q,I[ie]/norm[:,None],np.zeros(5));self._cost(Q,q,I[ib:ib+1]/max(1.,budget),np.zeros(1))
            x_force=optimize(Q,q,A,b,seed)
            GF=I[np.r_[np.arange(ie.start,ie.stop),ib]]
            AF=np.vstack((A,GF));bf=np.r_[b,GF@x_force+self.settings.priority_tolerance_N]
            Q=np.eye(dim)*1e-8;q=np.zeros(dim)
            self._cost(Q,q,I[ir]/norm[:,None],np.zeros(5));self._cost(Q,q,I[it]/norm[:,None],np.zeros(5))
            ws=np.r_[np.ones(3),np.full(3,1./self.g.characteristic_length),1.]
            if complete:self._cost(Q,q,ws[:,None]*I[iw],np.zeros(7))
            x_contact=optimize(Q,q,AF,bf,x_force)
            GC=I[np.r_[np.arange(ir.start,ir.stop),np.arange(it.start,it.stop),np.arange(iw.start,iw.stop)]]
            tolerances=np.r_[np.full(10,self.settings.priority_tolerance_N),self.settings.priority_tolerance_N/ws]
            AC=np.vstack((AF,GC));bc=np.r_[bf,GC@x_contact+tolerances]
            ms=np.tile(1./np.maximum(1e-6,np.maximum(positive[5:],negative[5:])),h)
            MP=ms[:,None]*CU[mpos];mt=ms*preview.ravel()
            Q=np.eye(dim)*1e-8;q=np.zeros(dim);self._cost(Q,q,MP,-mt)
            x_progress=optimize(Q,q,AC,bc,x_contact)
            deviation=np.abs(MP@x_progress-mt)+self.settings.progress_guard_tolerance
            AP=np.vstack((AC,MP,-MP));bp=np.r_[bc,mt+deviation,-mt+deviation]
            Q=np.eye(dim)*1e-8;q=np.zeros(dim)
            scales=np.tile(1./np.maximum(1e-6,np.maximum(positive,negative)),h)
            self._cost(Q,q,scales[:,None]*I[:nr],-scales*un)
            D8=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(8))
            old=np.r_[self.previous_correction,np.zeros(nr-8)]
            self._cost(Q,q,scales[:,None]*(D8@I[:nr]),-scales*(D8@un+old))
            DP=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(5));force_scale=np.tile(1./norm,h)
            self._cost(Q,q,force_scale[:,None]*(DP@N),-force_scale*(DP@N@nominal_x))
            x=optimize(Q,q,AP,bp,x_progress);final_A=AP;final_b=bp
            self.label='joint_v5_minimum_task_correction'
        raw_p=N@x+cp;raw_tau=T@x+ct
        if np.min(np.r_[raw_p,raw_tau])<-1e-8:
            self._unavailable('negative_raw_contact_magnitude_after_solver',np.r_[raw_p,raw_tau],float(-np.min(np.r_[raw_p,raw_tau])));return
        # The final executable plan, not the solver status alone, is checked.
        final_residual=float(np.max(A@x-b))
        hierarchy_residual=float(np.max(final_A@x-final_b));authority_residual=float(np.max(hard_A@x-hard_b))
        if final_residual>2e-5 or hierarchy_residual>2e-5 or authority_residual>1e-8:
            self._unavailable('unchecked_executable_or_priority_plan',np.r_[raw_p,raw_tau],max(final_residual,hierarchy_residual,authority_residual));return
        correction=x[:nr]-un;self.interval_action=x[:5].copy();self.interval_motion=x[5:8].copy()
        self.previous_plan=x.copy();self.previous_nominal=nominal[0].copy()
        if np.max(np.abs(correction[:8]))>1e-8:self.last_intervention=float(self.model.input_history[-1][0])
        self.predicted=raw_p[:5].copy();self.desired=raw_nominal[0,:5].copy();self.force_reference=self.desired.copy()
        missing_directions=[]
        for i,d in enumerate(self.model.available_directions):
            if (not d['plus'] and np.any(nominal[:,i]>1e-10)) or (not d['minus'] and np.any(nominal[:,i]<-1e-10)):missing_directions.append(i)
        now=float(self.model.input_history[-1][0]);deadline=self.task_request.get('deadline_s')
        remaining=None if deadline is None else max(0.,deadline-now)
        reach_time=float(np.max(np.abs(operating_target-self.executed_motion)/np.maximum(1e-6,np.asarray(self.settings.motion_rates))))
        components=self.model.last_trajectory_components
        self.debug=dict(controller_architecture='nominal_task_minimum_intervention_16x8_v5',prediction_available=True,
            nominal_task_preserved=bool(nominal_pass),nominal_operating_checks=check,
            nominal_failed_checks=[key for key,value in check.items() if not value],
            nominal_action_8=nominal[0].tolist(),nominal_plan_8=nominal.tolist(),correction_8=correction[:8].tolist(),
            actual_previous_interval_correction_8=self.previous_correction.tolist(),actual_previous_increment_available=previous_actual is not None,
            qp_status='nominal_candidate_preserved' if not statuses else statuses[-1],priority_status=statuses,
            qp_optimal=None if not statuses else bool(all(status=='optimal' for status in statuses)),
            qp_primal_residual=max([max(0.,final_residual)]+residuals),
            qp_checked_priority_residuals=checked_residuals,qp_priority_fallbacks=fallbacks,
            final_priority_residual=hierarchy_residual,final_authority_residual=authority_residual,
            predicted_pressure_violation_N=x[ie].tolist(),predicted_budget_violation_N=float(x[ib]),
            contact_retention_slack_N=x[ir].tolist(),contact_consistency_slack_N=x[it].tolist(),wrench_band_violation=x[iw].tolist(),
            pressure_now=f.normal_now.tolist(),shear_now=f.output_now[5:10].tolist(),force_prediction=self.predicted.tolist(),
            force_target=self.desired.tolist(),nominal_force_prediction=raw_nominal[:,:5].tolist(),
            predicted_extra_pressure_variation_N=(np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(5))@(P[pi]@correction)).reshape(h,5).tolist(),
            predicted_shear=raw_tau[:5].tolist(),predicted_pressure_min_N=float(raw_p.min()),predicted_shear_min_N=float(raw_tau.min()),
            estimated_normal=s.normal.tolist(),observed_complete_hand_wrench=f.output_now[10:].tolist(),
            complete_wrench_prediction_available=complete,
            predicted_complete_hand_wrench=((WN@x+wc).reshape(h,7)[:,:6].tolist() if complete else np.full((h,6),np.nan).tolist()),
            motion_request=self.motion_request.tolist(),operating_motion_target=operating_target.tolist(),interval_motion=self.interval_motion.tolist(),
            task_mode=self.task_request.get('mode','legacy_motion_request'),task_deadline_s=deadline,remaining_time_s=remaining,
            deadline_infeasible_under_reference_rate=None if remaining is None else reach_time>remaining+1e-9,
            tracking_tolerances=self.task_request.get('tracking_tolerances',[None,None,None]),
            force_scope='external_all_instrumented_fingers',contact_modes=cs.mode.tolist(),
            friction_lower_bound=cs.mu_lower.tolist(),friction_upper_bound_available=False,absolute_friction_reserve_identified=False,
            contact_reference_updated_at=self.contacts.reference_updated_at,contact_reference_updates=self.contacts.reference_updates,
            contact_reference_scope='causally observed stable contact; no pressure target or model-domain expansion',
            pressure_calibration_error_N=components['calibration_error'][pi].reshape(h,5).tolist(),
            pressure_residual_fit_error_N=components['fit_error'][pi].reshape(h,5).tolist(),
            pressure_slope_error_N=components['slope_error'][pi].reshape(h,5).tolist(),
            pressure_past_input_memory_change_N=components['past_input_memory_change'][pi].reshape(h,5).tolist(),
            pressure_residual_drift_change_N=components['residual_drift_change'][pi].reshape(h,5).tolist(),
            forecast_margin_N=self.settings.forecast_margin_N,model_admitted=False,model_domain_transfer='experimental_local_model_not_admitted',
            cumulative_model_domain_certified=False,missing_measured_direction_axes=missing_directions,
            opposite_direction_use='experimental return to visited anchor only',model_source=self.model.source_path,
            model_source_sha256=self.model.source_sha256,model_frame='current_palm',future_slip_model_available=False,
            support_infeasibility_certified=False,actual_input_history_only=True,pressure_shear_wrench_same_kernel_history=True,
            release_scope='external release outside local holding model',settings=asdict(self.settings),**support_info)

    def step(self,frame,phase=None,command='hold'):
        dt=.005 if self.previous_time is None else max(.0001,min(.05,float(frame.time)-self.previous_time))
        self.previous_time=float(frame.time);self.state.update(frame,dt);self.control_ticks+=1
        if command=='release' or str(phase or frame.phase).upper() in ('RELEASE','RELEASING') or str(self.task_request.get('mode','')).lower() in ('release','monitor'):
            self.release_latched=True
        if self.release_latched:
            self.label='joint_common_release';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.motion_reference=self.executed_motion.copy()
            delta=-np.minimum(np.maximum(0.,self.state.alpha-self.state.closure_min),self.g.release_rate*dt)
            return ControlAction(delta,self.predicted.copy(),self.label,False,dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='external_mechanical_release_outside_held_contact_model'))
        if np.count_nonzero(self.state.valid)==0:
            self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.predicted=np.full(5,np.nan)
            self.motion_reference=self.executed_motion.copy();self.previous_nominal=np.zeros(8)
            self.label='joint_observation_unavailable';self.debug=dict(prediction_available=False,prediction_unavailable_reason='all_tactile_channels_invalid')
            return ControlAction(np.zeros(5),self.predicted.copy(),self.label,False,self.debug.copy())
        if frame.time-self.last_decision>=self.g.decision_period-1e-8:
            self._decision();self.last_decision=float(frame.time)
        fraction=min(1.,dt/self.g.decision_period)
        delta=np.clip(self.interval_action*fraction,self.state.closure_min-self.state.alpha,self.state.closure_max-self.state.alpha)
        blocked=~self.state.valid|~self.state.closure_permission;delta[blocked]=np.minimum(0.,delta[blocked])
        self.motion_reference=self.executed_motion+self.interval_motion*fraction
        self.debug['executed_motion_reference']=self.motion_reference.tolist()
        return ControlAction(delta,self.predicted.copy(),self.label,False,self.debug.copy())


def create_joint(calibration,limits,budget,gains=None):return JointCoordinator(calibration,limits,budget,gains)


def self_test():
    from types import SimpleNamespace
    checks={**verify_model_algebra(),**verify_contact_membership()}
    cal=dict(force_matrix=np.eye(5)*10.,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02),config={'shape':'capsule','lift_target':.1})
    def fixture(K=None,pressure=None,drift=None,limits=None):
        ctl=JointCoordinator(cal,np.full(5,5.) if limits is None else limits,18.,model_document=synthetic_document(K))
        p=np.full(5,2.) if pressure is None else np.asarray(pressure,float);tau=.6*p
        state=SimpleNamespace(normal=p.copy(),alpha=np.zeros(5),closure_min=np.full(5,-.04),closure_max=np.full(5,.04),
            valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),contact=np.ones(5,bool),
            confidence=np.ones(5),slip_magnitude=np.zeros(5),auxiliary_wrench=np.zeros(6),auxiliary_valid=True,
            load=1.2,load_uncertainty=.5,up=np.array([0.,0.,1.]),force_gain=np.full(5,10.),slip_speed=0.)
        ctl.state=state;y=np.r_[p,tau,[0.,0.,1.2,0.,0.,0.]]
        ctl.model.record_input(7.98,np.r_[np.zeros(5),.1,0.,0.]);ctl.model.record_input(8.,np.r_[np.zeros(5),.1,0.,0.]);ctl.model.beta=y.copy()
        ctl.model.snapshot=OutputForecast(y,np.zeros(16) if drift is None else drift,np.full(16,.001),np.full(16,.01),
            np.full(16,50),np.ones(16),np.ones(16,bool),y.copy(),np.full(16,8.))
        for k in range(50):
            frame=SimpleNamespace(time=7.7+k*.005,normal=p,shear_magnitude=tau,contact_pos=np.zeros((5,3)))
            ctl.contacts.update(frame,state,ctl.model.snapshot)
        ctl.set_motion_request([.1,0.,0.],[.1,0.,0.]);return ctl
    still=fixture();still._decision()
    checks['stable_conflicting_friction_prior_is_not_a_grip_task']=bool(np.all(still.interval_action==0.) and still.debug['nominal_task_preserved'])
    moving=fixture();moving.set_motion_request([.099,.002,0.],[.1,0.,0.]);moving._decision()
    checks['feasible_user_motion_passes_exactly_without_alpha_changes']=bool(moving.debug['nominal_task_preserved'] and np.all(moving.interval_action==0.) and np.allclose(moving.interval_motion,moving.debug['nominal_action_8'][5:]))
    checks['nominal_motion_has_no_artificial_extra_force_variation']=bool(np.allclose(moving.debug['predicted_extra_pressure_variation_N'],0.))
    checks['formal_model_unknown_not_mislabeled_admitted']=bool(moving.debug['model_admitted'] is False and moving.debug['cumulative_model_domain_certified'] is False)
    four=fixture();four.model.active_mask[1]=False;four.model.snapshot.output_now[[1,6]]=0.;four.contacts.snapshot.mode[1]='UNKNOWN';four.contacts.anchor_valid[1]=False
    four._decision();checks['inactive_index_does_not_disable_nominal_task']=bool(four.debug['nominal_task_preserved'])
    fragile=fixture(limits=np.full(5,1.95));fragile._decision()
    checks['real_force_conflict_triggers_bounded_unloading']=bool(not fragile.debug.get('nominal_task_preserved',True) and np.any(fragile.interval_action<0.))
    checks['all_raw_pressure_and_shear_nonnegative']=bool(fragile.debug['predicted_pressure_min_N']>=-1e-8 and fragile.debug['predicted_shear_min_N']>=-1e-8)
    checks['four_layers_recorded_when_intervention_needed']=bool(len(fragile.debug['priority_status'])==4)
    checks['final_executable_and_priority_matrices_both_checked']=bool(fragile.debug['final_priority_residual']<=2e-5 and fragile.debug['final_authority_residual']<=1e-8)
    # Lower settled pressure must not be restored to an earlier force point.
    lower=fixture();lower._decision();lower.model.beta[:10]*=.8;lower.model.snapshot.output_now[:10]*=.8;lower.model.snapshot.last_measured[:10]*=.8
    lower._decision();checks['new_safe_lower_grip_is_not_pulled_back_to_old_anchor']=bool(lower.debug['nominal_task_preserved'] and np.all(lower.interval_action==0.))
    missing=fixture();missing.model.snapshot.output_valid[10:]=False;missing.set_motion_request([.099,.002,0.],[.1,0.,0.]);missing._decision()
    checks['missing_full_wrench_is_not_nominal_pass']=bool(not missing.debug.get('nominal_task_preserved',True) and not missing.debug.get('complete_wrench_prediction_available',True) and np.max(np.abs(missing.interval_motion))<1e-8)
    dead=fixture(K=np.zeros((101,16,8)),pressure=np.full(5,.01),drift=np.r_[np.full(5,-5.),np.zeros(11)]);dead._decision()
    checks['unreachable_negative_active_forecast_unavailable_not_clipped']=bool(np.isnan(dead.predicted).all() and dead.debug['prediction_available'] is False)
    task=fixture();task.set_task_request(dict(mode='transport',motion_target=[.099,.002,0.],executed_motion_reference=[.1,0.,0.],
        time_s=8.,deadline_s=12.,limits=[8.]*5,budget=30.,tracking_tolerances=[.01,None,None]));task._decision()
    lim,bu=task._force_limits();checks['task_limits_cannot_relax_material_limits']=bool(np.all(lim==5.) and bu==18.)
    checks['public_dict_task_and_unknown_wrist_tolerances_supported']=bool(task.debug['task_mode']=='transport' and task.debug['tracking_tolerances'][1:] ==[None,None])
    checks['task_deadline_is_public_absolute_time']=bool(task.debug['task_deadline_s']==12. and task.debug['remaining_time_s']==4.)
    unavailable=fixture();unavailable.state.valid[:]=False;unavailable.state.update=lambda f,dt:None;unavailable.predicted=np.ones(5);unavailable.interval_action[:]=.001
    bad=unavailable.step(SimpleNamespace(time=8.,phase='HOLDING'));checks['observation_loss_immediately_clears_prediction_and_action']=bool(np.isnan(bad.predicted_normal).all() and np.all(bad.delta_closure==0.))
    unavailable.predicted=np.ones(5);between=unavailable.step(SimpleNamespace(time=8.005,phase='HOLDING'))
    checks['observation_loss_between_decisions_cannot_retimestamp_old_forecast']=bool(np.isnan(between.predicted_normal).all())
    release=unavailable.step(SimpleNamespace(time=8.01,phase='RELEASING'),command='release')
    checks['external_release_has_no_holding_prediction']=bool(np.isnan(release.predicted_normal).all() and release.debug.get('prediction_unavailable_reason'))
    monitor=fixture();monitor.set_task_request(dict(mode='monitor',motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.]))
    monitor.state.update=lambda f,dt:None
    observed=monitor.step(SimpleNamespace(time=8.,phase='HOLDING'))
    monitor.set_task_request(dict(mode='hold',motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.]))
    resumed=monitor.step(SimpleNamespace(time=8.005,phase='HOLDING'))
    checks['post_release_monitor_and_later_hold_cannot_regrip_same_episode']=bool(np.isnan(observed.predicted_normal).all() and np.isnan(resumed.predicted_normal).all() and np.all(resumed.delta_closure<=0.))
    # A reported residual is not trusted when the actual solver matrix says
    # otherwise. No physics or implementation-shaped mock assertion is used.
    import copy
    global solve_qp
    real_solve=solve_qp;call_count=[0]
    def bad_last_report(*args,**kwargs):
        result=real_solve(*args,**kwargs);call_count[0]+=1
        if call_count[0]==4:
            result=copy.deepcopy(result);result.x=result.x.copy();result.x[0]+=100.;result.residual=0.
        return result
    try:
        solve_qp=bad_last_report;adversarial=fixture(limits=np.full(5,1.95));adversarial._decision()
    finally:solve_qp=real_solve
    checks['false_solver_residual_cannot_bypass_priority_or_actuator_checks']=bool(
        adversarial.debug['qp_priority_fallbacks'][-1]=='checked_previous_layer' and adversarial.debug['final_priority_residual']<=2e-5)
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks,scope='pure algebra/synthetic public interface; no physical stepping or performance claim')


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),ensure_ascii=False))

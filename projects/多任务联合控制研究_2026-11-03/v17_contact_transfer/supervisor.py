"""One bounded experimental support-transfer / contact-unloading lifecycle.

Only public delayed devices/current robot command memory enter. This component
does not sample sensors, run physics, write ports or certify placement/forces.
"""
from copy import deepcopy
from numbers import Real
import time
import numpy as np
from v13_a1.codec import Reference21,finite_array,finite_scalar
from v13_joint.lifecycle import ReleaseEntry
from v14_coord.design import DECISION_PERIOD_S,CONTROL_PERIOD_S,reference_rates
from v15_support.release_tasks import HAND_CLEARANCE_N,HAND_DWELL_S,HAND_MAX_AGE_S
from v15_support.receiver_sensor import ReceiverMeasurement,DELAY_S
from v16_release_geometry.transfer_observer import TransferObserver


CLOCK_TOL=1e-9
RELEASE_DEADLINE_S=3.5
_WIDE=float(np.finfo(float).max)


def _clock(value):
    t=finite_scalar(value,'public time');k=int(round(t/CONTROL_PERIOD_S))
    if t<0 or abs(t-k*CONTROL_PERIOD_S)>CLOCK_TOL:raise ValueError('registered nonnegative200Hz clock required')
    return k,t


def _bool5(value):
    a=np.asarray(value,dtype=object)
    if a.shape!=(5,) or any(not isinstance(x,(bool,np.bool_)) for x in a):raise ValueError('five explicit validity flags required')
    return np.asarray(value,bool)


def _contact_solve(tasks,*,reference_step_scale21,motion_task):
    # Lazy import permits independent interface tests without substituting a
    # plant or silently importing another optimizer into the lifecycle.
    from .task_solver import solve_contact_unloading
    return solve_contact_unloading(tasks,reference_step_scale21=reference_step_scale21,motion_task=motion_task)


class ReleaseSupervisor:
    def __init__(self,entry,*,open18,observer:TransferObserver):
        if type(entry) is not ReleaseEntry or type(observer) is not TransferObserver:
            raise TypeError('actual typed release entry and public TransferObserver required')
        if abs(entry.started_s-observer.request)>CLOCK_TOL:raise ValueError('release/observer origins differ')
        self.entry=entry;self.open18=finite_array(open18,(18,),'correct release clearance18').copy()
        self.open18.setflags(write=False);self.observer=observer
        self.phase='APPROACH_SUPPORT';self.version=0
        self.last_time=None;self.last_tick=None;self.last_acquisition=None;self.receiver_streak=0
        self.receiver_event_seen=False;self.receiver_event_at=None;self.receiver_lift_reference_at_event=None
        self.low_since=None;self.last_hand_acquisition=None;self.was_clear=False;self.low_current=False
        self.recontact_episode_count=0;self._recontact_active=False;self._current21=entry.reference21.values.copy()
        self.last_transfer=None;self.public_state=None
        self.last_decision=-np.inf;self.plan_start=None;self.plan_current=None;self.plan_delta=None
        self.plan_version=None;self.plan_generation=None
        self.debug={};self.decision_evidence=[]

    def _phase(self,value):
        if self.phase!=value:self.phase=value;self.version+=1

    def _hand_low(self,frame,now,gap):
        """Sampled low pressure; independent of receiver/motion observability."""
        reason=None;pressure=None;acq=None
        try:
            valid=_bool5(frame.valid)
            normal=finite_array(frame.normal,(5,),'public pressure5')
            aux=finite_scalar(frame.auxiliary_pressure,'public auxiliary pressure')
            if np.any(normal<0) or aux<0:raise ValueError('negative pressure is not a valid low-pressure observation')
            if not isinstance(frame.auxiliary_valid,(bool,np.bool_)):raise ValueError('explicit auxiliary validity required')
            age=finite_scalar(frame.age,'hand age');aux_age=finite_scalar(frame.auxiliary_age,'aux age')
            acq=finite_scalar(frame.sample_time,'hand acquisition')
            if (not valid.all() or not frame.auxiliary_valid or min(age,aux_age)<0 or
                max(age,aux_age)>HAND_MAX_AGE_S+CLOCK_TOL or acq>now+CLOCK_TOL or
                abs(now-acq-age)>CLOCK_TOL or abs(now-aux_age-acq)>CLOCK_TOL):
                raise ValueError('invalid/stale/noncoacquired hand pressure')
            if acq<self.entry.started_s-CLOCK_TOL:raise ValueError('hand pressure is not postrequest')
            pressure=float(np.maximum(normal,0.).sum()+max(0.,aux))
        except (ValueError,TypeError,AttributeError):
            reason='hand_pressure_missing_stale_or_ineligible';self.low_since=None;self.last_hand_acquisition=None
            self.low_current=False
            return dict(available=False,pressure_N=None,acquisition_s=acq,confirmed=False,reason=reason)
        previous=self.last_hand_acquisition
        fresh=previous is None or acq>previous+CLOCK_TOL
        adjacent=previous is not None and abs(acq-previous-CONTROL_PERIOD_S)<=CLOCK_TOL
        if previous is not None and acq<previous-CLOCK_TOL:
            self.low_since=None;self.last_hand_acquisition=None;self.low_current=False
            return dict(available=False,pressure_N=None,acquisition_s=acq,confirmed=False,reason='hand_acquisition_rewind')
        self.low_current=pressure<=HAND_CLEARANCE_N
        if gap or not self.low_current:self.low_since=None
        if self.low_current and fresh and (self.low_since is None or not adjacent):self.low_since=max(acq,self.entry.started_s)
        if fresh:self.last_hand_acquisition=acq
        confirmed=bool(self.low_since is not None and acq-self.low_since>=HAND_DWELL_S-CLOCK_TOL)
        if self.was_clear and not self.low_current:
            if not self._recontact_active:self.recontact_episode_count+=1
            self._recontact_active=True
        elif self.low_current:self._recontact_active=False
        self.was_clear=self.was_clear or confirmed
        return dict(available=True,pressure_N=pressure,acquisition_s=acq,fresh=fresh,
            low=self.low_current,confirmed=confirmed,dwell_s=None if self.low_since is None else acq-self.low_since,reason=None)

    def observe(self,*,frame,transfer_inputs,current21):
        tick,now=_clock(frame.time)
        if now<self.entry.started_s-CLOCK_TOL:raise ValueError('release observation before request')
        evidence=self.observer.update(**transfer_inputs)
        if self.last_tick is not None and tick==self.last_tick:return deepcopy(self.public_state)
        if self.last_tick is not None and tick<self.last_tick:raise ValueError('public clock rewind')
        current=finite_array(current21,(21,),'current authoritative reference21')
        gap=self.last_tick is not None and tick!=self.last_tick+1
        receiver=transfer_inputs['receiver']
        if type(receiver) is not ReceiverMeasurement:raise TypeError('typed already-sampled receiver required')
        eligible=bool(receiver.valid and receiver.acquisition_time_s is not None and
            abs(receiver.time_s-now)<=CLOCK_TOL and receiver.age_s is not None and
            abs(receiver.age_s-DELAY_S)<=CLOCK_TOL and
            receiver.acquisition_time_s>=self.entry.started_s-CLOCK_TOL)
        acq=receiver.acquisition_time_s if eligible else None
        fresh=eligible and (self.last_acquisition is None or acq>self.last_acquisition+CLOCK_TOL)
        adjacent=eligible and self.last_acquisition is not None and abs(acq-self.last_acquisition-CONTROL_PERIOD_S)<=CLOCK_TOL and not gap
        compressed=bool(eligible and receiver.normal_load_N>receiver.uncertainty6[2])
        if not compressed or gap:self.receiver_streak=0
        if compressed and fresh:self.receiver_streak=self.receiver_streak+1 if adjacent else 1
        if eligible and fresh:self.last_acquisition=acq
        if not self.receiver_event_seen and self.receiver_streak>=2:
            self.receiver_event_seen=True;self.receiver_event_at=now;self.receiver_lift_reference_at_event=float(current[18])
            if self.phase=='APPROACH_SUPPORT' and not self.was_clear:self._phase('CONTACT_UNLOADING')
        hand=self._hand_low(frame,now,gap)
        if hand['confirmed']:
            self._phase('FREE_CLEAR_AND_RETREAT')
        elif self.phase=='APPROACH_SUPPORT':
            arithmetic=64*np.finfo(float).eps*max(1.,abs(float(current[18])))
            if abs(float(current[18]))<=arithmetic and not self.receiver_event_seen:
                self._phase('UNSUPPORTED_FREE_CLEAR')
        deadline=now-self.entry.started_s>=RELEASE_DEADLINE_S-CLOCK_TOL
        self._current21=current.copy();self.last_transfer=evidence.to_public()
        self.public_state=dict(schema='v17_contact_transfer_process_v1',time_s=now,phase=self.phase,version=self.version,
            started_s=self.entry.started_s,holding_preempted=True,holding_force_tasks_active=False,
            first_receiver_event_latched=self.receiver_event_seen,receiver_event_at=self.receiver_event_at,
            receiver_current_compression_resolved=compressed,receiver_fresh_streak=self.receiver_streak,
            receiver_event_is_any_actor_candidate_not_placement=True,
            low_hand_pressure_N=hand['pressure_N'],low_hand_pressure_available=hand['available'],
            low_hand_pressure_confirmed=hand['confirmed'],hand_low_evidence=hand,
            sampled_hand_low_was_observed=self.was_clear,recontact_episode_count=self.recontact_episode_count,
            transfer=evidence.to_public(),transfer_permission_does_not_gate_all_release_motion=True,
            unsupported_clearance=not self.receiver_event_seen and self.phase in ('UNSUPPORTED_FREE_CLEAR','FREE_CLEAR_AND_RETREAT'),
            unconfirmed_receiver_support=not self.receiver_event_seen,
            coordinated_retreat_started=self.phase=='FREE_CLEAR_AND_RETREAT',
            deadline_reached=deadline,deadline_without_sampled_hand_clearance=deadline and not self.was_clear,
            delivery_confirmed=False,placed=False,placement_certified=False,geometric_full_hand_clearance_certified=False,
            no_regrip_after_release=True,release_request_delayed=False)
        self.last_time=now;self.last_tick=tick
        return deepcopy(self.public_state)

    def lift_interval(self,current_reference):
        if not isinstance(current_reference,Reference21):raise TypeError('typed current reference21 required')
        # Neutral TASK interval. Plant supplies finite actual0..target and rate;
        # CONTACT does not freeze lift or silently add a downward-only policy.
        return 0.,_WIDE

    def nominal(self,query_time_s,lower21,upper21):
        query=finite_scalar(query_time_s,'known nominal query')
        if query<self.entry.started_s-CLOCK_TOL:raise ValueError('nominal before release request')
        lower=finite_array(lower21,(21,),'actual reference lower21');upper=finite_array(upper21,(21,),'actual reference upper21')
        if np.any(lower>upper):raise ValueError('empty actual reference domain')
        value=self._current21.copy()
        if self.phase=='APPROACH_SUPPORT':
            value[:18]=self.entry.reference21.values[:18];value[19:]=self.entry.reference21.values[19:]
        else:value[:18]=self.open18
        value[18]=0.
        return Reference21(np.minimum(upper,np.maximum(lower,value)),query)

    def action(self,*,frame,adapter,current21,lower21,upper21,generation):
        _,now=_clock(frame.time)
        if self.last_time is None or abs(now-self.last_time)>CLOCK_TOL:raise ValueError('observe this public tick before action')
        if finite_scalar(frame.dt,'control dt')!=CONTROL_PERIOD_S:raise ValueError('one5ms public control tick required')
        if isinstance(generation,(bool,np.bool_)) or not isinstance(generation,(int,np.integer)) or generation<0:
            raise ValueError('explicit nonnegative mechanical generation required')
        current=finite_array(current21,(21,),'actual current reference21')
        self._current21=current.copy()
        lower=finite_array(lower21,(21,),'actual lower21');upper=finite_array(upper21,(21,),'actual upper21')
        if np.any(lower>upper):raise ValueError('empty actual authority')
        rates=reference_rates('release');scale=rates*DECISION_PERIOD_S
        if self.plan_version!=self.version or self.plan_generation!=generation:self.last_decision=-np.inf
        if now>=self.last_decision+DECISION_PERIOD_S-CLOCK_TOL:
            began=time.perf_counter_ns();nominal=self.nominal(now+DECISION_PERIOD_S,lower,upper).values
            lo=np.maximum(lower-current,-scale);hi=np.minimum(upper-current,scale)
            if np.any(lo>hi):raise ValueError('empty genuine20ms step domain')
            wanted=np.minimum(hi,np.maximum(lo,nominal-current))
            tasks=None;solved=None;error=None;motion=None;motion_reason='not_requested_in_direct_stage'
            if self.phase=='CONTACT_UNLOADING':
                tasks=adapter.current_geometry_tasks(frame,current,current+wanted,lo,hi)
                motion=getattr(adapter,'motion_task',None)
                motion_reason='public_record' if motion is not None else 'unknown_not_a_zero_motion_measurement'
                try:solved=_contact_solve(tasks,reference_step_scale21=scale,motion_task=motion)
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:error=type(exc).__name__+': '+str(exc)
            accepted=solved is not None and solved.available
            if accepted:
                try:delta=finite_array(solved.x,(21,),'normalized kernel proposal21')*scale
                except (ValueError,TypeError):
                    accepted=False;error='invalid_normalized_kernel_proposal';delta=wanted
            else:delta=wanted
            if not np.isfinite(delta).all() or np.any(delta<lo-1e-9) or np.any(delta>hi+1e-9):
                accepted=False;error='solver_output_outside_original_real_domain';delta=wanted
            self.plan_start=now;self.plan_current=current.copy();self.plan_delta=delta.copy()
            self.plan_version=self.version;self.plan_generation=int(generation);self.last_decision=now
            stage_direct=self.phase!='CONTACT_UNLOADING'
            fallback=None if accepted or stage_direct else 'legal_canonical_nominal_no_geometry_or_motion_certificate'
            ev=dict(decision_id=len(self.decision_evidence),time_s=now,phase=self.phase,
                mode_generation=int(generation),process_version=self.version,current_reference21=current.tolist(),
                nominal_reference21=nominal.tolist(),actual_step_lower21=lo.tolist(),actual_step_upper21=hi.tolist(),
                scale21=scale.tolist(),proposed_reference_increment21=delta.tolist(),
                geometry_status='NOT_REQUESTED_DIRECT_STAGE' if tasks is None else tasks.status,
                geometry=None if tasks is None else deepcopy(tasks.diagnostics),
                normal_rows=None if tasks is None or tasks.normal_matrix_m_per_reference is None else tasks.normal_matrix_m_per_reference.tolist(),
                tangent_rows=None if tasks is None or tasks.tangent_matrix_m_per_reference is None else tasks.tangent_matrix_m_per_reference.tolist(),
                hierarchy_status=None if solved is None else solved.status,hierarchy_levels=[] if solved is None else list(solved.levels),
                hierarchy_diagnostics=None if solved is None else solved.diagnostics,
                geometry_proposal_accepted=accepted,numerical_error=error,fallback=fallback,
                motion_task_available=motion is not None,motion_task_role=motion_reason,
                low_tangent_zero_is_declared_expectation_not_measured_stationarity=True,
                no_all18_OPEN_direction_task_cone=True,public_process=deepcopy(self.public_state),
                solver_ms=(time.perf_counter_ns()-began)/1e6,
                future_pressure_prediction_available=False,actual_servo_motion_predicted=False,
                physical_safety_certified=False,placement_certified=False)
            self.decision_evidence.append(ev)
            self.debug=dict(architecture='support_mode_contact_unloading_then_free_clearance',mode='release',
                decision_id=ev['decision_id'],decision_time_s=now,geometry_status=ev['geometry_status'],
                geometry_proposal_accepted=accepted,hierarchy_status=ev['hierarchy_status'],fallback=fallback,
                numerical_error=error,solver_timing_ms=dict(whole_decision=ev['solver_ms']))
        fraction=np.clip((now-self.plan_start+CONTROL_PERIOD_S)/DECISION_PERIOD_S,0.,1.)
        proposal=self.plan_current+fraction*self.plan_delta
        lo=np.maximum(lower,current-rates*CONTROL_PERIOD_S);hi=np.minimum(upper,current+rates*CONTROL_PERIOD_S)
        if np.any(lo>hi):raise ValueError('empty genuine5ms authority')
        # Real actor/mode bounds are authoritative. There is deliberately no
        # additional all18-to-OPEN cone: CONTACT may use available thumb motion.
        value=np.minimum(hi,np.maximum(lo,proposal))
        self.debug.update(process=deepcopy(self.public_state),mode_generation=int(generation),
            holding_force_tasks_active=False,pressure_prediction_available=False,
            plan_snapshot_scope='original20ms proposal; intermediate samples do not acquire a fresh model certificate')
        return Reference21(value,now),deepcopy(self.debug)

    def reconcile(self,receipt):
        if receipt.execution_changed or receipt.stale_plan_replaced or receipt.mode_generation!=self.plan_generation:
            self.last_decision=-np.inf
        self.debug.update(actual_reference21=receipt.executed.values.tolist(),
            execution_changed_plan=receipt.execution_changed,stale_plan_replaced=receipt.stale_plan_replaced)

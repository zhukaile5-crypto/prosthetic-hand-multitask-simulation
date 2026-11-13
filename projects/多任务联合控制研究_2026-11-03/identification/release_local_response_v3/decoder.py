"""Slow-only offsets on the REAL native5ms-held carrier; no physics at import."""
from numbers import Real
import numpy as np
from identification.release_basis_v2.decoder import ReleasePath2D,COMMAND_AXES

VARIANTS={'zero':(0.,0.),'phi_slow':(1.,0.),'lift_slow':(0.,1.),
 'mixed_50_50':(.5,.5),'mixed_50_100':(.5,1.),'mixed_100_50':(1.,.5),'mixed_100_100':(1.,1.)}
TRAIN=('zero','phi_slow','lift_slow')
ORIGINS={'phi_150ms':30,'phi_350ms':70}
RAMP_TICKS=4
DT=.005

def coordinates(decoder,native_tau,relative_control_tick,variant,*,release_control_tick,release_time_s,probe_origin_tick):
 if variant not in VARIANTS:raise ValueError('Registered one-sided direction required')
 if isinstance(probe_origin_tick,(bool,np.bool_)) or not isinstance(probe_origin_tick,(int,np.integer)) or probe_origin_tick not in ORIGINS.values():raise ValueError('Registered30/70 tick native stage required')
 if isinstance(relative_control_tick,(bool,np.bool_)) or not isinstance(relative_control_tick,(int,np.integer)) or relative_control_tick<0:raise ValueError('Actual nonnegative control tick required')
 if isinstance(native_tau,(bool,np.bool_)) or not isinstance(native_tau,Real) or not np.isfinite(native_tau):raise ValueError('Actual native control clock required')
 if isinstance(release_control_tick,(bool,np.bool_)) or not isinstance(release_control_tick,(int,np.integer)):raise ValueError('Actual release latch tick required')
 expected=(release_control_tick+relative_control_tick)*DT-release_time_s
 if abs(native_tau-expected)>1e-12:raise ValueError('Carrier must be actual held control clock, not physical substep time')
 a,b=VARIANTS[variant];weight=float(np.clip((relative_control_tick-probe_origin_tick)/RAMP_TICKS,0.,1.))
 requested_phi=-a*(RAMP_TICKS*DT)*weight
 requested_lift=b*(decoder.entry_lift/.6)*(RAMP_TICKS*DT)*weight
 # Tick arithmetic preserves exact held coordinates on full-slow quarters,
 # avoiding cancellation-induced1e-15s progress reversals at absolute12s.
 phi=float(native_tau) if a==0. else (release_control_tick+relative_control_tick-a*RAMP_TICKS*weight)*DT-release_time_s
 lift_clock=float(native_tau) if b==0. else (release_control_tick+relative_control_tick-b*RAMP_TICKS*weight)*DT-release_time_s
 # Clip AFTER offsetting the unbounded lowering clock. Adding a constant to
 # an already-bottom-clipped lift would incorrectly keep a3.33mm reference forever.
 lift=decoder.native_lift(lift_clock)
 raw_lift=decoder.entry_lift*(1.-lift_clock/.6)
 zero_lift=decoder.native_lift(native_tau)
 if phi< -1e-12 or lift_clock< -1e-12:raise ValueError('Slowdown attempted negative absolute progress')
 return dict(phi_s=phi,lift_reference_m=float(lift),native_phi_s=float(native_tau),native_lift_reference_m=float(zero_lift),requested_phi_offset_s=float(requested_phi),requested_lift_offset_m=float(requested_lift),actual_phi_offset_s=float(phi-native_tau),actual_lift_offset_m=float(lift-zero_lift),lift_clock_s=float(lift_clock),unclipped_lift_reference_m=float(raw_lift),lift_clipped=bool(raw_lift<0. or raw_lift>decoder.entry_lift),ramp_weight=weight)

def native_envelope(decoder,release_tick,control_dt=.005):
 start=release_tick*control_dt
 tau=[(release_tick+i)*control_dt-start for i in range(round(4./control_dt)+1)]
 command=np.array([decoder.at(t,'native2d').command21 for t in tau])
 return dict(maximum_held_tick_increment21=np.max(abs(np.diff(command,axis=0)),axis=0),total_variation21=np.sum(abs(np.diff(command,axis=0)),axis=0),minimum21=command.min(axis=0),maximum21=command.max(axis=0))

def make_plant_class(probe_origin_tick):
 if isinstance(probe_origin_tick,(bool,np.bool_)) or not isinstance(probe_origin_tick,(int,np.integer)) or probe_origin_tick not in ORIGINS.values():raise ValueError('Registered probe origin required')
 from hand_plant import HandPlant
 class SlowOffsetPlant(HandPlant):
  def __init__(self,config,variant):
   if variant not in VARIANTS:raise ValueError('Registered pilot variant required')
   self.probe_variant=variant;self.probe_origin_tick=int(probe_origin_tick);self.probe_decoder=None;self.probe_release_tick=None
   self.probe_rows=[];self.probe_robot_rows=[];self.probe_last_physical_time=None
   self.probe_last_tick=None;self.probe_last_command=None;self.probe_last_phi=None;self.probe_last_lift=None
   self.probe_total_variation=np.zeros(21);self.probe_mismatches=[];self.probe_envelope=None
   super().__init__(config)
  def begin_release(self):
   fresh=self.release_started is None;super().begin_release()
   if fresh:
    self.probe_release_tick=self.control_steps;self.probe_decoder=ReleasePath2D.from_plant(self)
    self.probe_envelope=native_envelope(self.probe_decoder,self.probe_release_tick,self.config.control_dt)
  def _contacts(self,patches=True):
   result=HandPlant._contacts(self,patches)
   if hasattr(self,'raw_qadr') and hasattr(self,'lift_qadr'):
    time=float(self.data.time)
    if self.probe_last_physical_time is None or time>self.probe_last_physical_time+1e-12:
     q=np.r_[self.data.qpos[self.raw_qadr],self.data.qpos[self.lift_qadr]]
     velocity=np.r_[self.data.qvel[self.raw_dadr],self.data.qvel[self.lift_dadr]]
     self.probe_robot_rows.append(np.r_[time,q,velocity,result['normal'],result['skin_normal'],result['palm'],result['hand_other'],result['all_hand_object'],result['support']])
     self.probe_last_physical_time=time
   return result
  def _apply_ctrl(self):
   HandPlant._apply_ctrl(self)
   if self.release_started is None:return
   if self.probe_decoder is None:raise RuntimeError('Actual native release entry missing')
   native=self.data.ctrl.copy();native_lift=float(self.lift_ref);native_alpha=self.alpha.copy()
   tick=self.control_steps-self.probe_release_tick;tau=self.time-self.release_started
   info=coordinates(self.probe_decoder,tau,tick,self.probe_variant,release_control_tick=self.probe_release_tick,release_time_s=self.release_started,probe_origin_tick=self.probe_origin_tick)
   decoded=self.probe_decoder.decode(info['phi_s'],info['lift_reference_m'])
   aids=np.concatenate(self.actuator_groups);wrist=np.array([self.aid('rh_A_WRJ2'),self.aid('rh_A_WRJ1')],int)
   candidate=native.copy();candidate[aids]=decoded.finger_reference;candidate[wrist]=decoded.wrist_reference
   candidate[self.lift_actuator]=decoded.lift_reference+self.model.body_subtreemass[self.hand]*9.81/1500.
   native21=np.r_[native[aids],native_lift,native[wrist]];command=decoded.command21
   errors=[]
   if self.probe_variant=='zero' and (not np.array_equal(candidate,native) or not np.array_equal(decoded.alpha,native_alpha)):errors.append('zero_not_native_full_ctrl_alpha_exact')
   if np.any(command[:18]<self.probe_decoder.lower-1e-10) or np.any(command[:18]>self.probe_decoder.upper+1e-10):errors.append('finger_actuator_position_violation')
   if not -1e-12<=decoded.lift_reference<=self.probe_decoder.entry_lift+1e-12:errors.append('lift_reference_position_violation')
   if not np.array_equal(candidate[wrist],native[wrist]):errors.append('wrist_changed')
   if self.probe_last_tick is not None:
    delta=abs(command-self.probe_last_command)
    if self.control_steps==self.probe_last_tick:
     if not np.array_equal(command,self.probe_last_command):errors.append('reference_changed_inside_held_control_tick')
    else:
     clock=(self.control_steps-self.probe_last_tick)*self.config.control_dt
     if decoded.path_time_s<self.probe_last_phi-1e-12 or decoded.path_time_s-self.probe_last_phi>clock+1e-12:errors.append('phi_monotonic_rate_violation')
     if decoded.lift_reference>self.probe_last_lift+1e-12 or self.probe_last_lift-decoded.lift_reference>self.probe_decoder.entry_lift/.6*clock+1e-12:errors.append('lift_monotonic_native_rate_violation')
     if np.any(delta>self.probe_envelope['maximum_held_tick_increment21']+1e-10):errors.append('actual21_held_tick_increment_exceeds_complete_native_envelope')
    self.probe_total_variation+=delta
    if np.any(self.probe_total_variation>self.probe_envelope['total_variation21']+1e-9):errors.append('actual21_cumulative_variation_exceeds_complete_native_envelope')
   if abs(info['actual_phi_offset_s'])>.02+1e-12 or info['actual_lift_offset_m']< -1e-12 or info['actual_lift_offset_m']>self.probe_decoder.entry_lift/.6*.02+1e-12:errors.append('actual_cumulative_coordinate_offset_violation')
   if errors:
    self.probe_mismatches.append(dict(control_time=self.time,physical_time=float(self.data.time),errors=errors))
    raise AssertionError('Release probe authority/exactness failed before integration: '+','.join(errors))
   self.data.ctrl[:]=candidate;self.alpha=decoded.alpha.copy();self.lift_ref=decoded.lift_reference;self.wrist_ref=decoded.wrist_reference.copy()
   self.probe_last_tick=self.control_steps;self.probe_last_command=command.copy();self.probe_last_phi=decoded.path_time_s;self.probe_last_lift=decoded.lift_reference
   scalars=[self.time,float(self.data.time),tick,tau,info['phi_s'],info['lift_clock_s'],info['ramp_weight'],info['requested_phi_offset_s'],info['requested_lift_offset_m'],info['actual_phi_offset_s'],info['actual_lift_offset_m'],info['unclipped_lift_reference_m'],float(info['lift_clipped'])]
   self.probe_rows.append(np.r_[scalars,command,native21,candidate,native])
 return SlowOffsetPlant

"""ROOT ONLY: complete fixed-input causal diagnostic, never a new controller."""
from pathlib import Path
from dataclasses import asdict
import gzip,json,time,traceback
import numpy as np
from .inputs import canonical,exact,sha,reference_program,unchanged
from .program import imports,config,set_release_event,apply

TRUTH_FIELDS=('normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position','relative','rotation',
    'relative_velocity','angular_velocity','lift','qpos','qvel','penetration','world_penetration','warning_count')
SENSOR_FIELDS=('legacy_common_sensor_json','observation_json','observed_normal','validity','observed_age','observed_auxiliary_wrench','observed_auxiliary_valid','raw_patches_json')
COMMAND_FIELDS=('time','ctrl_before','ctrl','alpha_before','alpha','phase','wrist_position','robot_actuator_target21','robot_control_time_s',
    'requested_reference21','executed_reference21','reference_increment21','mode_generation','executed_motion_reference','environment_json','command')
def recorded_plant():
    _,_,Plant=imports()
    class RecordedPlant(Plant):
        def __init__(self,c,stream):
            self.replay_inside_advance=False;self.replay_last_end=None;self.replay_states=[];self.replay_stream=stream
            super().__init__(c)
        def _contacts(self,patches=True):
            value=super()._contacts(patches);end=float(self.data.time)
            if self.replay_inside_advance and end!=self.replay_last_end:
                self.replay_last_end=end
                self.replay_states.append((end,self.data.qpos.copy(),self.data.qvel.copy(),self.data.ctrl.copy(),int(sum(int(x.number) for x in self.data.warning))))
                if 12.-1e-9<=end<=16.+1e-9:
                    from session import jsonable
                    full=super()._contacts(True)
                    self.replay_stream.write(json.dumps(jsonable(dict(start_s=end-self.model.opt.timestep,end_s=end,phase=self.phase,patches=full['patches'])),ensure_ascii=False)+'\n')
            return value
        def advance(self,step=None):
            self.replay_inside_advance=True
            try:return super().advance(step)
            finally:self.replay_inside_advance=False
    return RecordedPlant
def arrays(rows,p):
    result={k:np.asarray([row[k] for row in rows]) for k in rows[0]} if rows else {}
    raw=np.asarray(p.substep_records,float)
    if len(raw):
        for key,col in dict(time=0,duration=1,normal=slice(2,7),skin_normal=slice(7,12),palm=12,hand_other=13,all_hand_object=14,support=15,penetration=16,world_penetration=17).items():result['substep_'+key]=raw[:,col]
        result.update(substep_state_end_time=np.array([x[0] for x in p.replay_states]),substep_qpos=np.array([x[1] for x in p.replay_states]),
            substep_qvel=np.array([x[2] for x in p.replay_states]),substep_actual_ctrl=np.array([x[3] for x in p.replay_states]),substep_warning_count=np.array([x[4] for x in p.replay_states]))
    return result
def compare(values,record):
    a=record['arrays'];fields={}
    required=set(TRUTH_FIELDS+SENSOR_FIELDS+COMMAND_FIELDS)|{k for k in a if k.startswith('substep_')}
    for key in sorted(required):
        ok=key in values and key in a and exact(values[key],a[key])
        fields[key]=dict(exact=ok,source_shape=None if key not in a else list(a[key].shape),actual_shape=None if key not in values else list(values[key].shape))
        if not ok and key in values and key in a and values[key].shape==a[key].shape and values[key].dtype.kind in 'fiu':fields[key]['max_abs_diff']=float(np.max(abs(values[key]-a[key])))
    excluded={k:'source_* provenance only; original policy/observer/timing not executed or claimed exact' for k in a if k not in required}
    return dict(passed=all(x['exact'] for x in fields.values()),required_fields=sorted(required),fields=fields,explicit_unrecomputed_source_fields=excluded,
        scope='Every independent physics/public sensor/current-reference column required; no policy field copied then counted as an exact physical output.')
def diagnostics(v,meta):
    st=v['substep_time'];dt=v['substep_duration'];f=v['substep_skin_normal'];limits=np.asarray(meta['limits']);budget=meta['budget'];out={}
    for name,lo,hi in (('preparation',0.,7.5),('holding',7.5,12.),('release',12.,16.)):
        ids=np.flatnonzero((st>=lo-1e-9)&(st<hi-1e-9));i,j=np.unravel_index(np.argmax(f[ids]),f[ids].shape);k=ids[i]
        out[name]=dict(duration_s=float(dt[ids].sum()),peak_N=float(f[k,j]),digit=int(j),peak_start_s=float(st[k]),peak_end_s=float(st[k]+dt[k]),
            target_same_digit_N=float(v['substep_normal'][k,j]),non_target_same_digit_N=float(max(0.,f[k,j]-v['substep_normal'][k,j])),
            local_excess_Ns=float(np.sum(np.maximum(f[ids]-limits,0)*dt[ids,None])),total_excess_Ns=float(np.maximum(f[ids].sum(1)-budget,0)@dt[ids]),
            above_local_limit_duration_s=float(dt[ids][(f[ids]>limits+.02).any(1)].sum()),palm_at_peak_N=float(v['substep_palm'][k]),support_at_peak_N=float(v['substep_support'][k]),
            max_penetration_m=float(v['substep_world_penetration'][ids].max()),qpos_at_peak_evaluation_only=v['substep_qpos'][k].tolist(),actualctrl_at_peak=v['substep_actual_ctrl'][k].tolist())
    from evaluation import first_dwell
    out['low_force_clearance']=dict(threshold_N=.2,dwell_s=.2,deadline_s=15.5,
        frame200Hz_first_by_deadline=first_dwell(v['time'],v['all_hand_object']<=.2,12.,15.5,.2),
        raw_integration_end_first_by_deadline=first_dwell(st+dt,v['substep_all_hand_object']<=.2,12.,15.5,.2),
        meaning='Same low-force contact criterion, not geometric gap or permanent continuous-time separation certificate; not an algorithm task-success metric')
    return out
def run_job(record,variant,target):
    """INTEGRATES; only the root-dispatched -m launcher may call this."""
    from .run import write
    target=Path(target);target.mkdir(parents=True,exist_ok=False);p=None;stream=None;rows=[];began=time.perf_counter();result=None
    try:
        mj,_,_=imports()
        from contracts import SensorNoise
        from sensors import HandSensors
        from skin_geometry import DelayedSkinGeometry
        from session import jsonable
        from v13_joint.runtime import JointSession,LEGACY_MODEL_FIELDS,V12Native_REFERENCE_RATE_LIMITS
        a=record['arrays'];meta=record['metadata'];program=reference_program(record,variant)
        stream=gzip.open(target/'cached_contact_packets.jsonl.gz','xt',encoding='utf-8');p=recorded_plant()(config(record),stream)
        sensors=HandSensors(meta['calibration'],SensorNoise(**meta['noise']),seed=meta['seed']);skin=DelayedSkinGeometry(p)
        shim=JointSession.__new__(JointSession);shim.plant=p
        if not all((exact(p.data.qpos,a['qpos'][0]),exact(p.data.qvel,a['qvel'][0]),exact(p.data.ctrl,a['ctrl_before'][0]))):raise ValueError('Initial physical reset mismatch')
        spec=mj.mjtState.mjSTATE_INTEGRATION;initial=np.empty(mj.mj_stateSize(p.model,spec));mj.mj_getState(p.model,p.data,initial,spec)
        np.savez_compressed(target/'reset_integration_state.npz',state=initial,state_spec=int(spec))
        obstacle=int(p.model.geom('integration_obstacle_geom').id);ids=p.actor21.indices21
        for k,t in enumerate(a['time']):
            if p.time!=float(t):raise ValueError('Public clock mismatch')
            truth=p.truth();raw=p.raw_sensors();before=canonical(sensors.rng.bit_generator.state)
            obs=sensors.sample(raw);skin.acquire(raw,obs);common=json.dumps(jsonable(vars(obs)),ensure_ascii=False)
            after=canonical(sensors.rng.bit_generator.state)
            for key in LEGACY_MODEL_FIELDS:vars(obs).pop(key,None)
            obs.robot_actuator_target21=p.data.ctrl[ids].copy();obs.robot_control_time_s=float(p.time);obs.robot_actuator_axes21=list(meta['robot_actuator_telemetry']['axes21'])
            obs.robot_position25=np.r_[obs.raw_joint,obs.lift];obs.robot_velocity25=np.r_[obs.raw_joint_velocity,float(raw['lift_velocity'])]
            obs.executed_motion_reference=np.r_[p.lift_ref,p.wrist_ref];obs.sensor_characteristics=meta['task_contract']['sensor_characteristics']
            obs.motion_reference_rate_limits=np.asarray(V12Native_REFERENCE_RATE_LIMITS,float);obs.lift_velocity_m_s=float(raw['lift_velocity'])
            set_release_event(p,record,k);JointSession._attach_execution_state(shim,obs)
            row=dict(time=float(t),**{key:truth[key] for key in TRUTH_FIELDS},ctrl_before=truth['ctrl'],alpha_before=truth['alpha'],
                wrist_position=truth['qpos'][[p.model.joint(n).qposadr[0] for n in ('rh_WRJ2','rh_WRJ1')]],legacy_common_sensor_json=common,
                robot_actuator_target21=obs.robot_actuator_target21.copy(),robot_control_time_s=obs.robot_control_time_s,
                observation_json=json.dumps(jsonable(vars(obs)),ensure_ascii=False),observed_normal=obs.normal,validity=obs.valid,observed_age=obs.age,
                observed_auxiliary_wrench=obs.auxiliary_wrench,observed_auxiliary_valid=obs.auxiliary_valid,raw_patches_json=json.dumps(jsonable(raw['patches']),ensure_ascii=False),
                rng_before=before,rng_after=after,collision=float(sum(x['normal'] for x in raw['patches'] if int(x.get('external_geom',-1))==obstacle)))
            apply(p,record,k,program[k]);row.update(ctrl=p.data.ctrl.copy(),alpha=p.alpha.copy(),phase=p.phase,
                requested_reference21=program[k].copy(),executed_reference21=p.reference_snapshot().values.copy(),reference_increment21=p.reference_snapshot().values-obs.robot_reference21,
                mode_generation=p.mode_generation,executed_motion_reference=np.r_[p.lift_ref,p.wrist_ref],environment_json=str(a['environment_json'][k]),command=str(a['command'][k]))
            rows.append({key:value.copy() if isinstance(value,np.ndarray) else value for key,value in row.items()})
            if k<3200:p.set_environment(record['environments'][k]);p.advance(.005)
        unchanged(record);values=arrays(rows,p);bridge=compare(values,record) if variant=='both' else None
        # Explicit source aliases preserve historical labels, never replace a
        # freshly measured field or masquerade as independent reproduction.
        excluded=set(a)-set(values)
        for key in excluded:values['source_'+key]=a[key]
        replay_meta=dict(schema='v13_release_factorial_fixed_input_v1',role='causal_diagnostic_not_policy_performance',variant=variant,source_trace=str(record['path']),
            source_trace_sha256=record['trace_sha256'],config=asdict(p.config),model_sha256=p.model_sha256,limits=meta['limits'],budget=meta['budget'],source_handoff_historical=meta['handoff_passed'],
            new_feedback_decisions=0,new_strategy_success_evaluated=False,source_entry_reference21=record['entry_reference21'].tolist(),source_entry_time_s=12.,fixed_input_program='source_post_command21 with two factor masks at12; current target not q state',
            raw_clock='Cached integration force START plus duration gives END.200Hz force includes separate boundary forward and is not simple decimation.',source_excluded_columns=sorted(excluded))
        values['metadata']=json.dumps(replay_meta,ensure_ascii=False);np.savez_compressed(target/'trace.npz',**values)
        write(target/'metadata.json',replay_meta);write(target/'diagnostics.json',diagnostics(values,meta))
        if bridge is not None:write(target/'both_bridge_comparison.json',bridge)
        np.savez_compressed(target/'fixed_reference_program.npz',time=a['time'],reference21=program,source_entry21=record['entry_reference21'])
        result=dict(variant=variant,complete=True,rows=len(rows),physics_intervals=len(p.substep_records),bridge_all_recomputed_exact=None if bridge is None else bridge['passed'],
            source_trace_sha256=record['trace_sha256'],trace_sha256=sha(target/'trace.npz'),noise_rng_final=canonical(sensors.rng.bit_generator.state),
            cached_contact_packet_sha256=None,wall_seconds=time.perf_counter()-began,diagnostics=read_diagnostic(target/'diagnostics.json'))
    except Exception as exc:
        result=dict(variant=variant,complete=False,error=repr(exc),traceback=traceback.format_exc(),saved_rows=len(rows),wall_seconds=time.perf_counter()-began)
        if p is not None:
            try:np.savez_compressed(target/'partial_trace.npz',**arrays(rows,p))
            except Exception as e:result['partial_save_error']=repr(e)
    finally:
        if stream is not None:stream.close()
        if result is not None:
            if (target/'cached_contact_packets.jsonl.gz').exists():result['cached_contact_packet_sha256']=sha(target/'cached_contact_packets.jsonl.gz')
            write(target/'result.json',result)
    return result
def read_diagnostic(path):return json.loads(Path(path).read_text(encoding='utf-8'))

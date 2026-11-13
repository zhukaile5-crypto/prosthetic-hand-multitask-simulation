"""Mocked-contact sensor checks: no model construction, physics, FK or solver."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v15_support.receiver_sensor import ReceiverMount,ReceiverNoise,ReceiverMeasurement,ReceiverSensor,ROLE,PHASES

class Model:
    def __init__(self):
        self.names={'support':0,'floor':1,'support_post':2,'any_object':3,'robot':4}
        self.geom_type=np.array([6,0,6,2,6]);self.geom_bodyid=np.array([0,0,0,1,2])
        self.geom_size=np.array([[.07,.16,.005],[1,1,.1],[.025,.045,.166],[.02,.02,.02],[.01,.01,.01]])
        self.geom_pos=np.zeros((5,3));self.geom_pos[0,2]=-.005
        self.geom_quat=np.tile([1.,0,0,0],(5,1))
    def geom(self,name):return SimpleNamespace(id=self.names[name])
class Data:
    def __init__(self,time_s,contacts=()):self.time=time_s;self.contact=list(contacts);self.ncon=len(self.contact)
    def __getattribute__(self,name):
        if name in ('qpos','qvel','xpos','object_mass','object_pose','true_support','is_target','target_id','future_schedule'):raise AssertionError('Truth access '+name)
        return super().__getattribute__(name)
class Contact:
    def __init__(self,g0=0,g1=3,point=(.04,-.02,0.),frame=None,wrench=(2.,0,0,0,0,0),dist=0.):
        self.geom=np.array([g0,g1]);self.pos=np.array(point);self.frame=np.array([[0,0,1.],[1.,0,0],[0,1.,0]]) if frame is None else np.array(frame)
        self.local_wrench=np.asarray(wrench,float);self.dist=dist
    def __getattribute__(self,name):
        if name in ('is_target','object_id','external_body','target_id','true_pose'):raise AssertionError('Contact identity accessed')
        return super().__getattribute__(name)
def reader(model,data,index,out):out[:]=data.contact[index].local_wrench
def mount(**changes):
    values=dict(geom_name='support',excluded_geom_names=('floor','support_post'),origin_in_geom_m=[0,0,.005],top_plane_tolerance_m=1e-6,normal_alignment_min_cos=.999)
    values.update(changes);return ReceiverMount(**values)
def device(model=None,noise=None,mounted=None,seed=617901,**kwargs):
    return ReceiverSensor(Model() if model is None else model,mount() if mounted is None else mounted,ReceiverNoise(0.,0.,cop_min_load_N=.1) if noise is None else noise,seed,
        acquisition_phase='boundary_after_forward',contact_force_reader=reader,**kwargs)
def acquire(sensor,contacts=(),ticks=3,start=0.,available=True):
    result=None
    for i in range(ticks):
        t=start+i*.005;result=sensor.sample(Data(t,contacts),time_s=t,sampling_phase='boundary_after_forward',available=available)
    return result
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    checks=[];forbidden=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+' '+str(detail))
    def reject(name,call):
        try:call()
        except (ValueError,TypeError,AttributeError,KeyError):check(name,True)
        else:check(name,False)
    # Importing a library is allowed; every model/data constructor and all
    # dynamics/FK/contact solves are guarded. Only the force-reader mock runs.
    import mujoco
    for name in ('MjModel','MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_makeConstraint','mj_projectConstraint'):
        if hasattr(mujoco,name):
            def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('No engine construction/pipeline '+_name)
            setattr(mujoco,name,deny)
    s=device();m=acquire(s,[Contact()]);expected=np.array([0.,0.,-2.,.04,.08,0.])
    check('geom0_external_on_receiver_sign_force_and_moment',m.valid and np.allclose(m.wrench6,expected,atol=1e-15,rtol=0),m.to_public())
    check('synthetic_Newton_receiver_down_actor_up_balances2N',m.wrench6[2]==-2. and -m.wrench6[2]-2.==0.)
    check('normal_load_same_measured_Fz',m.normal_load_N==2.)
    check('equivalent_CoP_not_object_pose',np.allclose(m.cop_xy_m,[.04,-.02],atol=1e-15,rtol=0))
    other=Contact(3,0,frame=[[0,0,-1.],[1,0,0],[0,-1.,0]])
    swapped=acquire(device(),[other]);check('geom_order_swap_same_receiver_result',np.array_equal(swapped.wrench6,m.wrench6))
    Q=np.array([[.6,0,.8],[0,1.,0],[-.8,0,.6]]);w=np.array([5.,1,-.5,.2,-.1,.05])
    oblique=Contact(point=(.03,-.02,.04),frame=Q,wrench=w,dist=.1)
    o=acquire(device(mounted=mount(normal_alignment_min_cos=.5)),[oblique])
    check('oblique_friction_contact_torque_plus_lever_moment',o.valid and np.allclose(o.wrench6,[-3.4,-1.,-3.7,.034,.075,-.288],atol=1e-14,rtol=0),o.to_public())
    D=np.diag([-1.,1.,-1.]);ow=np.r_[-D@w[:3],-D@w[3:]]
    os=acquire(device(mounted=mount(normal_alignment_min_cos=.5)),[Contact(3,0,point=(.03,-.02,.04),frame=D@Q,wrench=ow,dist=.1)])
    check('oblique_order_swap_preserves_force_and_torque',os.valid and np.array_equal(o.wrench6,os.wrench6))
    # Same worldbody0 has floor+post. ID-specific top installation must exclude
    # unrelated contacts, not aggregate all contacts on body0.
    z=acquire(device(),[Contact(1,3,wrench=(300,0,0,0,0,0)),Contact(2,3,wrench=(400,0,0,0,0,0))])
    check('floor_and_support_post_not_receiver',z.valid and np.array_equal(z.wrench6,np.zeros(6)) and z.normal_load_N==0.)
    robot=acquire(device(),[Contact(g1=4)]);check('robot_and_other_actors_not_target_filtered',np.array_equal(robot.wrench6,m.wrench6))
    two=acquire(device(),[Contact(),Contact(g1=4,wrench=(3,0,0,0,0,0))]);check('all_actors_resultant_adds',two.normal_load_N==5.)
    side=Contact(point=(.07,0,-.002),frame=np.eye(3),wrench=(99,0,0,0,0,0))
    bottom=Contact(point=(0,0,-.01),frame=[[0,0,-1.],[1,0,0],[0,-1,0]],wrench=(99,0,0,0,0,0))
    outside=Contact(point=(.071,0,0),wrench=(99,0,0,0,0,0))
    z=acquire(device(),[side,bottom,outside]);check('side_bottom_outside_top_coverage_excluded',z.valid and np.array_equal(z.wrench6,np.zeros(6)))
    penetrated=Contact(point=(.04,-.02,-.001),dist=-.002)
    p=acquire(device(),[penetrated]);check('receiver_surface_from_midpoint_and_signed_distance',p.valid and np.array_equal(p.wrench6,expected))
    rotated_model=Model();rotated_model.geom_quat[0]=[np.sqrt(.5),0,0,np.sqrt(.5)]
    R=np.array([[0,-1.,0],[1.,0,0],[0,0,1.]]);rc=Contact(point=R@np.array([.04,-.02,0]),frame=np.array([[0,0,1.],[1,0,0],[0,1.,0]])@R.T)
    rm=acquire(device(model=rotated_model),[rc]);check('known_mount_frame_rotation',rm.valid and np.allclose(rm.wrench6,expected,atol=1e-14,rtol=0))
    s=device();data=Data(0.,[Contact()]);a=s.sample(data,time_s=0.,sampling_phase=PHASES[0]);data.time=.005
    b=s.sample(data,time_s=.005,sampling_phase=PHASES[0]);data.time=.010;c=s.sample(data,time_s=.010,sampling_phase=PHASES[0])
    check('10ms_head_unknown_no_fake_history',not a.valid and not b.valid and a.wrench6 is None and a.acquisition_time_s is None and c.valid and c.acquisition_time_s==0. and c.age_s==.010)
    state=deepcopy(s._rng.bit_generator.state);before=s.instrument_metadata();data.contact=[Contact(wrench=(999,0,0,0,0,0))]
    duplicate=s.sample(data,time_s=.010,sampling_phase=PHASES[0]);check('same_tick_no_reacquisition_or_noise',duplicate is c and s._rng.bit_generator.state==state and before==s.instrument_metadata())
    reject('public_array_cannot_be_reenabled_writable',lambda:c.wrench6.setflags(write=True))
    reject('frozen_public_no_truth_extra_assignment',lambda:setattr(c,'true_support',2.))
    fields=c.to_public();check('public_output_no_external_ID_pose_mass_or_true_support',not(set(fields)&{'object_id','target_id','is_target','object_pose','object_mass','true_support','future_schedule'}))
    reject('public_constructor_rejects_extra_truth_kw',lambda:ReceiverMeasurement(c.time_s,c.acquisition_time_s,c.age_s,c.valid,c.wrench6,c.normal_load_N,c.sigma6,c.uncertainty6,c.cop_xy_m,c.sampling_phase,None,true_support=2.))
    s=device();s.sample(Data(0,[Contact()]),time_s=0.,sampling_phase=PHASES[0]);s.sample(None,time_s=.005,sampling_phase=PHASES[0],available=False)
    s.sample(Data(.010,[Contact()]),time_s=.010,sampling_phase=PHASES[0]);missing=s.sample(Data(.015,[Contact()]),time_s=.015,sampling_phase=PHASES[0])
    check('missing_packet_delayed_unknown_not_zero_or_last_good',not missing.valid and missing.wrench6 is None and missing.normal_load_N is None and missing.acquisition_time_s==.005 and missing.invalid_reason=='acquisition_unavailable')
    s=device(noise=ReceiverNoise(0,0,dropout_windows=((.005,.010),)))
    values=[s.sample(Data(t,[Contact()]),time_s=t,sampling_phase=PHASES[0]) for t in (0.,.005,.010,.015)]
    check('dropout_window_uses_acquisition_clock_then_delay',values[2].valid and not values[3].valid and values[3].invalid_reason=='scheduled_dropout')
    s=device(noise=ReceiverNoise(0,0,dropout_probability=1.));q=acquire(s,[Contact()]);check('random_dropout_unknown_and_fixed_draws',not q.valid and q.invalid_reason=='random_dropout' and s.instrument_metadata()['gaussian_draws']==18)
    s=device();s.sample(Data(0),time_s=0.,sampling_phase=PHASES[0]);state=deepcopy(s._rng.bit_generator.state)
    reject('skipped_tick_refused',lambda:s.sample(Data(.010),time_s=.010,sampling_phase=PHASES[0]))
    check('skipped_tick_does_not_consume_rng',s._rng.bit_generator.state==state)
    s.sample(Data(.005),time_s=.005,sampling_phase=PHASES[0]);reject('clock_rewind_refused',lambda:s.sample(Data(0),time_s=0.,sampling_phase=PHASES[0]))
    reject('phase_cannot_be_relabelled',lambda:s.sample(Data(.010),time_s=.010,sampling_phase=PHASES[1]))
    for name,value in (('string','0.010'),('bool',True),('numpybool',np.bool_(True)),('negative',-.01),('nan',np.nan)):
        reject('strict_public_clock_'+name,lambda value=value:device().sample(Data(0),time_s=value,sampling_phase=PHASES[0]))
    for name,value in (('string','0.01'),('bool',True),('numpybool',np.bool_(True)),('negative',-.01),('inf',np.inf)):
        reject('strict_sigma_'+name,lambda value=value:ReceiverNoise(value,0.))
    reject('unused_fallback_sigma_still_strict',lambda:ReceiverNoise.from_auxiliary(force_sigma_N=True,auxiliary_force_sigma_N=.01))
    reject('seed_bool_refused',lambda:device(seed=True));reject('wrong_sensor_period',lambda:ReceiverNoise(0,0,period_s=.01))
    reject('wrong_delay',lambda:ReceiverNoise(0,0,delay_s=0.));reject('unknown_mount',lambda:device(mounted=mount(geom_name='missing')))
    reject('floor_mount_refused',lambda:device(mounted=mount(geom_name='floor',excluded_geom_names=('support','support_post'),origin_in_geom_m=[0,0,.1])))
    reject('wrong_top_origin_refused',lambda:device(mounted=mount(origin_in_geom_m=[0,0,0])))
    moving=Model();moving.geom_bodyid[0]=2;reject('moving_body_mount_unsupported',lambda:device(model=moving))
    s=device();acquire(s,[Contact()]);s._model.geom_pos[0,0]=.01
    changed=s.sample(Data(.015,[Contact()]),time_s=.015,sampling_phase=PHASES[0]);check('changed_fixed_mount_immediately_latches_unknown',not changed.valid and changed.invalid_reason=='mount_definition_changed')
    s._model.geom_pos[0,0]=0.;again=s.sample(Data(.020,[Contact()]),time_s=.020,sampling_phase=PHASES[0]);check('restoring_geometry_does_not_revalidate_old_install',not again.valid)
    low=acquire(device(),[Contact(wrench=(.05,0,0,0,0,0))]);check('low_load_CoP_unknown',low.valid and low.cop_xy_m is None)
    disabled=acquire(device(noise=ReceiverNoise(0,0)),[Contact()]);check('CoP_disabled_unless_explicit_gate',disabled.valid and disabled.cop_xy_m is None)
    built=ReceiverNoise.from_auxiliary(force_sigma_N=.01);check('existing_auxiliary_fallback_units',built.force_sigma_N==.01 and built.moment_sigma_Nm==.0005)
    explicit=ReceiverNoise.from_auxiliary(force_sigma_N=.01,auxiliary_force_sigma_N=.02,auxiliary_moment_sigma_Nm=.003)
    check('explicit_aux_scales_reused',explicit.force_sigma_N==.02 and explicit.moment_sigma_Nm==.003)
    # Independent stream: unaltered original-tactile draws and global RNG state.
    tactile=np.random.default_rng(617901+910000);baseline=np.random.default_rng(617901+910000);s1=device(noise=built);s2=device(noise=built)
    global_before=np.random.get_state();original=[];reference=[]
    for i in range(10):
        t=i*.005;original.append(tactile.standard_normal(48));reference.append(baseline.standard_normal(48))
        s1.sample(Data(t,[Contact()] if i%2 else []),time_s=t,sampling_phase=PHASES[0],available=i%3!=1)
        s2.sample(Data(t,[Contact(wrench=(9,0,0,0,0,0))]),time_s=t,sampling_phase=PHASES[0])
    check('original_tactile_rng_sequence_unchanged',np.array_equal(original,reference))
    check('receiver_rng_contact_and_missing_independent',s1._rng.bit_generator.state==s2._rng.bit_generator.state)
    global_after=np.random.get_state();check('global_random_state_untouched',all(np.array_equal(a,b) for a,b in zip(global_before,global_after)))
    check('fixed_six_gaussian_one_uniform_per_tick',s1.instrument_metadata()['gaussian_draws']==60 and s1.instrument_metadata()['uniform_draws']==10)
    # Exercise the actual default API route with a mocked mj_contactForce only.
    calls=[]
    def force_api(m,d,i,out):calls.append(i);reader(m,d,i,out)
    mujoco.mj_contactForce=force_api
    default=ReceiverSensor(Model(),mount(),ReceiverNoise(0,0),1,acquisition_phase=PHASES[0]);q=acquire(default,[Contact()])
    check('default_adapter_only_calls_mj_contactForce_reader',q.valid and len(calls)==3 and np.array_equal(q.wrench6,expected))
    check('no_model_physics_FK_or_solver_calls',not forbidden,forbidden)
    return dict(schema='v15_receiver_sensor_mocked_contact_author_v1',passed=True,checks=checks,physics_steps=0,model_constructions=0,forbidden_calls=forbidden,
        scope='Synthetic/mocked-contact arithmetic, timing, type and RNG contract only. No physical bridge, receiver calibration, hardware measurement or lifecycle/controller integration.',
        official_sources=['https://mujoco.readthedocs.io/en/3.15.0/APIreference/APIfunctions.html#mj-contactforce','https://github.com/google-deepmind/mujoco/blob/3.15.0/include/mujoco/mjdata.h#L34-L38','https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_smooth.c#L2349-L2357'])
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    module=ROOT/'v15_support/receiver_sensor.py';before=sha(module)
    try:report=run()
    except Exception as exc:report=dict(schema='v15_receiver_sensor_mocked_contact_author_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),physics_steps=0)
    report.update(source_sha256=before,source_unchanged=sha(module)==before,checker_sha256=sha(__file__))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(report.get('checks',[])),path=str(out),sha256=sha(out))))
    raise SystemExit(0 if report['passed'] else 1)

"""Independent small synthetic receiver review; no MuJoCo import/construction."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from copy import deepcopy
import hashlib, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v15_support.receiver_sensor import ReceiverSensor, ReceiverMount, ReceiverNoise, ReceiverMeasurement

EXPECTED='90ba997ca39d6879e6b43c453c1021630e3bb6cecb4613ed3a0412838626c6b8'
PHASE='cached_after_step_end_label'
ENGINE_CALLS=[]
class DenyEngine(ModuleType):
    def __getattr__(self,name):
        ENGINE_CALLS.append(name)
        raise AssertionError('Engine API forbidden: '+name)
sys.modules['mujoco']=DenyEngine('mujoco')

class Model:
    def __init__(self):
        self.geom_type=np.array([6,0,6,2,6])
        self.geom_bodyid=np.array([0,0,0,1,2])
        self.geom_size=np.array([[.3,.4,.2],[1,1,.1],[.1,.1,.1],[.01,.01,.01],[.01,.01,.01]])
        self.geom_pos=np.zeros((5,3));self.geom_quat=np.tile([1.,0,0,0],(5,1))
    def geom(self,name):return SimpleNamespace(id={'receiver':0,'floor':1,'post':2,'actor_a':3,'actor_b':4}[name])
    def __getattr__(self,name):raise AssertionError('Undeclared model access '+name)

class Contact:
    def __init__(self,pair=(0,3),point=(.03,-.02,.198),Q=None,w=(4,1,-2,.2,-.1,.3),dist=-.004):
        self.geom=np.array(pair);self.pos=np.array(point);self.frame=np.array([[0,0,1],[1,0,0],[0,1,0]],float) if Q is None else np.array(Q)
        self.dist=dist;self._w=np.array(w,float)
    def __getattr__(self,name):raise AssertionError('Contact truth/unknown field '+name)
class Data:
    def __init__(self,t,contacts=()):self.time=t;self.contact=list(contacts);self.ncon=len(contacts)
    def __getattr__(self,name):raise AssertionError('Plant truth/unknown field '+name)

READS=[]
def reader(model,data,i,out):READS.append(i);out[:]=data.contact[i]._w
def make(*,model=None,noise=None,seed=917):
    model=Model() if model is None else model
    mount=ReceiverMount('receiver',('floor','post'),[0.,0.,.2],1e-7,.999)
    return ReceiverSensor(model,mount,ReceiverNoise(0.,0.,cop_min_load_N=.01) if noise is None else noise,seed,
                          acquisition_phase=PHASE,contact_force_reader=reader)
def at(s,t,contacts=(),available=True):return s.sample(Data(t,contacts) if available else None,time_s=t,sampling_phase=PHASE,available=available)
def deliver(s,contacts):
    for t in (0.,.005,.010):m=at(s,t,contacts)
    return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    checks=[];facts={}
    def check(name,ok,details=None):
        checks.append(dict(name=name,passed=bool(ok),details=details))
        if not ok:raise AssertionError(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,TypeError,AttributeError):check(name,True)
        else:check(name,False)
    # Nonzero tangent/torque, receiver-side penetration point and cached lever arm.
    c=Contact();s=make();m=deliver(s,[c]);want=np.array([-1.,2.,-4.,.184,-.178,-.16])
    check('independent_nonzero_tangent_torque_cached_lever_oracle',m.valid and np.allclose(m.wrench6,want,atol=2e-15,rtol=0),m.to_public())
    D=np.diag([-1.,1.,-1.]);cs=Contact((3,0),Q=D@c.frame,w=np.r_[-D@c._w[:3],-D@c._w[3:]])
    swapped=deliver(make(),[cs]);check('Newton_geom_order_swap_including_tangential_and_torsional_components',np.array_equal(m.wrench6,swapped.wrench6))
    # Unrelated worldbody0 contacts and a real receiver side never enter top channel.
    side=Contact(point=(.3,0,0),Q=np.eye(3),dist=0.)
    floor=Contact((1,3));post=Contact((2,4));other_actor=Contact((0,4))
    mixed=deliver(make(),[c,side,floor,post,other_actor])
    check('explicit_top_only_and_any_two_actors_add_without_identity_filter',np.allclose(mixed.wrench6,2*want,atol=3e-15,rtol=0))
    check('only_top_contacts_call_trusted_force_reader',READS[-6:]==[0,4,0,4,0,4])
    # Delivery at t must retain acquisition t-.010; no backfill or zero imputation.
    s=make();v=[at(s,0,[c]),at(s,.005,available=False),at(s,.010),at(s,.015),at(s,.020)]
    check('startup_two_unknown_then_actual_old_acquisition',not v[0].valid and not v[1].valid and v[0].wrench6 is None and v[2].acquisition_time_s==0 and np.array_equal(v[2].wrench6,m.wrench6))
    check('missing_is_delayed_null_with_own_clock_not_zero_or_last_good',not v[3].valid and v[3].wrench6 is None and v[3].normal_load_N is None and v[3].acquisition_time_s==.005 and v[3].invalid_reason=='acquisition_unavailable')
    check('complete_no_contact_is_valid_zero_distinct_from_missing',v[4].valid and np.array_equal(v[4].wrench6,np.zeros(6)) and v[4].normal_load_N==0 and v[4].cop_xy_m is None)
    state=deepcopy(s._rng.bit_generator.state);before=s.instrument_metadata();nr=len(READS)
    check('duplicate_same_tick_is_identical_cached_object',at(s,.020,[c]) is v[4])
    check('duplicate_does_not_read_or_advance_rng',len(READS)==nr and before==s.instrument_metadata() and state==s._rng.bit_generator.state)
    rejects('skip_tick_rejected_before_advancing_stream',lambda:at(s,.030,[c]))
    check('skip_preserves_rng_and_acquisition_counts',before==s.instrument_metadata() and state==s._rng.bit_generator.state)
    rejects('same_episode_sampling_phase_cannot_be_relabelled',lambda:s.sample(Data(.025),time_s=.025,sampling_phase='boundary_after_forward'))
    # A dropout at acquisition .005 must appear at delivery .015, not .005.
    sd=make(noise=ReceiverNoise(0,0,dropout_windows=((.005,.010),)))
    dv=[at(sd,t,[c]) for t in (0,.005,.010,.015)]
    check('dropout_phase_is_acquisition_then_fixed_delay',dv[2].valid and not dv[3].valid and dv[3].acquisition_time_s==.005 and dv[3].invalid_reason=='scheduled_dropout')
    # Direct independent random schedule reproduces device noise without global state.
    global_state=np.random.get_state();spec=ReceiverNoise(.03,.002);sr=make(noise=spec,seed=824)
    oracle=np.random.default_rng(np.random.SeedSequence([824,0x56313552]));expected=[]
    for i in range(5):
        e=oracle.standard_normal(6)*np.array([.03]*3+[.002]*3);oracle.random();expected.append(e)
        out=at(sr,i*.005,[c] if i%2==0 else (),available=i!=1)
        if i>=2 and i-2!=1:check('independent_rng_packet_'+str(i),np.array_equal(out.wrench6,(m.wrench6 if (i-2)%2==0 else 0)+expected[i-2]))
    after=np.random.get_state();check('numpy_global_state_untouched',all(np.array_equal(a,b) for a,b in zip(global_state,after)))
    check('fixed_draw_schedule_even_missing_packet',sr.instrument_metadata()['gaussian_draws']==30 and sr.instrument_metadata()['uniform_draws']==5 and sr._rng.bit_generator.state==oracle.bit_generator.state)
    # Invalid mount immediately suppresses even queued old valid samples and latches.
    model=Model();sm=make(model=model);deliver(sm,[c]);model.geom_size[0,0]+=.001
    invalid=at(sm,.015,[c]);model.geom_size[0,0]-=.001;restored=at(sm,.020,[c])
    check('mount_mutation_invalidates_and_restoring_bytes_does_not_unlock',not invalid.valid and not restored.valid and invalid.invalid_reason=='mount_definition_changed' and restored.wrench6 is None)
    # Positive noisy loading is only a resultant; no target identity or placement certificate.
    pub=m.to_public();forbidden={'object_id','is_target','target_id','object_pose','object_mass','true_support','place_success','future_schedule'}
    check('public_schema_identity_free_and_cached_phase_explicit',not(set(pub)&forbidden) and pub['sampling_phase']==PHASE and pub['normal_load_N']==4.)
    rejects('wrench_bytes_cannot_be_made_writable',lambda:m.wrench6.setflags(write=True))
    rejects('public_type_rejects_unknown_truth_field',lambda:ReceiverMeasurement(**{**{k:getattr(m,k) for k in ReceiverMeasurement.__dataclass_fields__},'true_support':True}))
    low=deliver(make(),[Contact(w=(.001,0,0,0,0,0))]);check('low_load_CoP_remains_unknown',low.valid and low.normal_load_N==.001 and low.cop_xy_m is None)
    noisy_zero=deliver(make(noise=ReceiverNoise(1.,.01,cop_min_load_N=.01),seed=42),[])
    check('noise_alone_cannot_override_three_sigma_CoP_gate_in_this_fixture',noisy_zero.cop_xy_m is None,dict(load_N=noisy_zero.normal_load_N))
    check('zero_engine_API_model_data_or_pipeline_calls',not ENGINE_CALLS,ENGINE_CALLS)
    facts.update(independent_tangent_torque_expected=want.tolist(),phase=PHASE,model_constructor='strict_python_fake_only',actual_engine_imported=False,
                 fixtures='One composite contact/order/coverage fixture; one delay/missing/dropout stream; one independent RNG stream; one mount mutation; typed/CoP guards.')
    return dict(passed=True,checks=checks,facts=facts)

if __name__=='__main__':
    out=ROOT/'review/v15_receiver_sensor_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    source=ROOT/'v15_support/receiver_sensor.py';contract=ROOT/'v15_support/receiver_sensor_contract.md'
    initial=sha(source);assert initial==EXPECTED
    try:r=run()
    except Exception as e:r=dict(passed=False,error=repr(e),traceback=traceback.format_exc())
    r.update(schema='v15_receiver_sensor_independent_v1',role='independent_peer_synthetic_oracles',source_path=str(source),source_sha256=initial,
             source_unchanged=sha(source)==initial,contract_sha256=sha(contract),checker_sha256=sha(__file__),
             counts=dict(physics_steps=0,engine_model_constructions=0,engine_data_constructions=0,forward_FK_contact_solver_calls=0),
             reviewer_revision=dict(prior_attempt='review/v15_receiver_sensor_independent_attempt1.json',reason='RNG byte oracle originally added noise to decimal-rounded analytic moment instead of the separately verified exact noiseless moment. Corrected test baseline; production unchanged.'),
             scope_limits=['No physical bridge, hardware calibration, or closed-loop integration.','The runtime must bind model/data identity and sampling phase; a string cannot prove source provenance.',
                 'Contact-only top-face resultant includes any actor; it does not identify target support, placement, future force or safety.',
                 'Gaussian three-sigma is an engineering proxy, not a hard bound. Equivalent CoP is not target position.','Extreme finite-number overflow robustness was not certified.'])
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r.get('checks',[])),sha256=sha(out),path=str(out)),ensure_ascii=False))
    raise SystemExit(0 if r['passed'] else 1)

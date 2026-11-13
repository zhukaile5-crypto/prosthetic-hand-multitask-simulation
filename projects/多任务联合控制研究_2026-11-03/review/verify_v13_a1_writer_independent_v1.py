"""Small actual-writer fault gate. No HandPlant/model/data construction/physics."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,hashlib,importlib,inspect,json,sys,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PINS={'v13_a1/codec.py':'ece85d993da2d751f6d7cd329776aed9392f1dc96e2365fee1f172423322b196',
      'v13_a1/proposals.py':'7a6d0b48e47e8e669f57e9f25e5b34963813ba774914de0340570e7e3e9d8236',
      'v13_a1/plant.py':'3e925baadafba4a5491ca53ff9767412adc694c0a59d5872b1670706e0c26972'}
OUT=ROOT/'review/v13_a1_writer_independent_v1.json'

def main():
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in PINS.items())
    sys.path.insert(0,str(ROOT));import mujoco
    forbidden={};calls={}
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_tendon','mj_transmission','mj_contactForce'):
        if hasattr(mujoco,name):
            forbidden[name]=getattr(mujoco,name);calls[name]=0
            def reject(*args,_name=name,**kwargs):calls[_name]+=1;raise AssertionError('forbidden '+_name)
            setattr(mujoco,name,reject)
    c=importlib.import_module('v13_a1.codec');pr=importlib.import_module('v13_a1.proposals');pm=importlib.import_module('v13_a1.plant')
    old=pm.HandPlant._apply_ctrl;legacy_calls=[]
    def old_forbidden(self):legacy_calls.append(True);raise AssertionError('legacy writer bypass')
    pm.HandPlant._apply_ctrl=old_forbidden
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    class Buffer:
        def __init__(self):self.values=np.full(22,.123);self.writes=[];self.rejected=0
        @property
        def shape(self):return self.values.shape
        def __getitem__(self,key):return self.values[key]
        def __setitem__(self,key,value):
            caller=inspect.currentframe().f_back.f_code
            robot=caller is c.ReferenceActor21.apply.__code__ and np.array_equal(np.asarray(key),np.r_[np.arange(3,21),2,0,1])
            obstacle=caller is pm.A1Plant._apply_ctrl.__code__ and np.isscalar(key) and int(key)==21
            if not (robot or obstacle):self.rejected+=1;raise AssertionError('unauthorized direct ctrl write')
            self.writes.append('robot21' if robot else 'obstacle1');self.values[key]=value
    names=list(c.ACTUATORS21);ids={name:i for name,i in zip(names,list(range(3,21))+[2,0,1])}
    model=NS(nu=22,actuator=lambda name:NS(id=ids[name]),body=lambda name:NS(id=0),
        actuator_ctrlnum=np.ones(22,int),actuator_ctrladr=np.arange(22),actuator_ctrllimited=np.ones(22,bool),
        actuator_ctrlrange=np.tile([-2.,2.],(22,1)),actuator_gainprm=np.ones((22,3)),body_subtreemass=np.array([3.794]))
    model.actuator_gainprm[2,0]=1500.;model.actuator_ctrlrange[3:8]=[-1.,1.]
    def holder():
        p=pm.A1Plant.__new__(pm.A1Plant);p.model=model;p.data=NS(ctrl=Buffer(),time=987.)
        p.control_steps=1500;p.config=NS(control_dt=.005,release_open_rate=.8,lift_target=.1)
        p.actuator_groups=[np.arange(a,b) for a,b in ((3,8),(8,11),(11,14),(14,17),(17,21))]
        p.open_pose=[np.linspace(.05,.1,len(g)) for g in p.actuator_groups];p.closed_pose=[x+1. for x in p.open_pose]
        p.alpha=np.full(5,.3);p.lift_ref=.1;p.wrist_ref=np.array([.01,-.02]);p.max_hidden_clip=0.
        p.release_started=None;p.release_thumb=None;p.release_alpha=None;p.release_lift=0.;p.phase='HOLDING'
        p.lift_actuator=2;p.obstacle_motor=21;p.hand=0;p.aid=lambda name:ids[name];p.env=dict(obstacle_x=.5)
        p.closure_min=np.zeros(5);p.closure_max=np.full(5,1.55);return p
    try:
        p=holder();p._apply_ctrl()
        check('actual_override_sole_robot_and_environment_write',p.data.ctrl.writes==['robot21','obstacle1'] and not legacy_calls,p.data.ctrl.writes)
        check('nonrobot_separate_clip',p.data.ctrl.values[21]==.1)
        check('nonzero_lift_bias_exact_once',p.data.ctrl.values[2]==p.lift_ref+float(model.body_subtreemass[0]*9.81/1500.))
        before=p.data.ctrl.values.copy()
        try:p.data.ctrl[3]=0.;bypass_rejected=False
        except AssertionError:bypass_rejected=True
        check('injected_direct_writer_bypass_rejected',bypass_rejected and np.array_equal(before,p.data.ctrl.values))
        # Actor-only apply must leave the independent environment channel alone.
        p.data.ctrl.writes.clear();obstacle=p.data.ctrl.values[21];p.a1_actor.apply(p.data,c.Reference21(np.r_[np.zeros(18),.05,.02,-.01],7.5))
        check('actor21_apply_once_no_environment_write',p.data.ctrl.writes==['robot21'] and p.data.ctrl.values[21]==obstacle)
        # Release boundaries and fake substep clock change: actual public clock wins.
        boundary=[]
        for tick in (2400,2520,2560,2680,2840):
            p=holder();p.control_steps=tick;p.release_started=12.;p.release_thumb=np.array([.4,.3,.2,.1,0.]);p.release_alpha=np.full(5,10.);p.release_lift=.1
            elapsed=p.time-12.;tuck=np.array([-1.03,1.21,.13,-.02,.985]);end=np.array([-1.,.1,0.,0.,0.])
            expected=p.release_thumb+np.clip(elapsed/.8,0,1)*(tuck-p.release_thumb) if elapsed<.8 else tuck if elapsed<1.4 else tuck+np.clip((elapsed-1.4)/.8,0,1)*(end-tuck)
            p._apply_ctrl();a=p.a1_last_engine_port21.values.copy();stamp=p.a1_last_engine_port21.time_s
            p.data.time+=.0005;p._apply_ctrl();b=p.a1_last_engine_port21.values.copy()
            boundary.append(bool(np.array_equal(a,b) and stamp==p.time and np.array_equal(a[:5],np.clip(expected,-1.,1.))))
        check('release_D18_boundaries_and_held_public_clock',all(boundary),dict(public_ticks=[2400,2520,2560,2680,2840],all_exact=boundary))
        check('release_raw_outside_range_disclosed_not_silently_reclipped',any(p.a1_actor.last_diagnostics['raw_command_outside_native_ctrlrange21']) and np.max(p.a1_last_engine_port21.values[5:18])>2.)
        p=holder();p.closure_max[:]=.301;requested=np.ones(5);actual=p.execute_delta(requested)
        check('late_rate_and_closure_clip_returns_actual_not_request',np.allclose(actual,.001) and np.array_equal(p.raw_requested_delta,requested) and not np.array_equal(actual,requested),dict(requested=requested.tolist(),actual=actual.tolist()))
        # Model XML structural oracle only: no mj_tendon/FK invocation needed.
        xml=ROOT.parent/'五指策略接入_2026-11-03/assets/legacy_model.xml';tree=ET.fromstring(xml.read_bytes())
        qnames=tuple(n.attrib['name'] for n in tree.findall('.//worldbody//joint') if n.attrib.get('name','').startswith('rh_'))+('transport_lift',)
        tendons={n.attrib['name']:n for n in tree.findall('./tendon/fixed')};A=np.zeros((21,25));tests=[]
        for digit in ('FF','MF','RF','LF'):
            tendon=tendons['rh_'+digit+'J0'];row=np.zeros(25)
            for j in tendon.findall('joint'):row[qnames.index(j.attrib['joint'])]=float(j.attrib['coef'])
            idsq=[qnames.index('rh_'+digit+'J2'),qnames.index('rh_'+digit+'J1')]
            diff=np.zeros(25);diff[idsq]=[.02,-.02]
            tests.append(bool(np.array_equal(row[idsq],[1.,1.]) and row@diff==0.))
            A[names.index('rh_A_'+digit+'J0')]=row
        check('four_actual_XML_J0_rows_11_and_difference_null',len(qnames)==25 and all(tests),dict(each=tests,q_axes25=list(qnames)))
        check('robot_q_axes_not_control_axes',qnames[:21]!=tuple(c.ACTUATORS21) and qnames[-1]=='transport_lift')
    finally:
        pm.HandPlant._apply_ctrl=old
        for name,fn in forbidden.items():setattr(mujoco,name,fn)
    check('legacy_writer_never_touched',not legacy_calls)
    check('forbidden_dynamics_FK_contact_calls_zero',not any(calls.values()),calls)
    root_evidence=ROOT/'review/v13_a1_runtime_zero_root_v1.json';binding=json.loads(root_evidence.read_text(encoding='utf-8'))
    check('root_source87_binding',binding['source_count']==87 and all(binding['source_sha256'][name]==pin for name,pin in PINS.items()))
    check('final3_unchanged',all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in PINS.items()))
    value=dict(schema='v13_a1_writer_independent_v1',role='independent_solewriter_boundary_peer',passed=all(c['passed'] for c in checks),
        checks=checks,checks_count=len(checks),source_sha256=PINS,source87_declared_binding=binding['source_sha256'],
        root_zero_evidence_sha256=hashlib.sha256(root_evidence.read_bytes()).hexdigest(),real_HandPlant_constructors=0,real_model_constructors=0,
        real_data_constructors=0,physics_steps=0,forbidden_calls=calls,source_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        limitations=['Actual final overrides/actor/proposal executed against strict tracked fake22buffer and model metadata; not a real-model/dynamics test.',
            'A1 intentionally preserves original raw release ports and MuJoCo effective clipping, with no new rate clamp; raw out-of-range is disclosed, not certified independent21 authority.',
            'XML J0 check is a structural tendon oracle, not a new native transmission/FK evaluation. Parent owns typed/full96stream and root real zero-step model checks.',
            'Solewriter is observed on tested paths with an instrumentation guard; this is not language-level memory isolation, arbitrary buffer transactionality, or physical safety.'])
    assert not OUT.exists();OUT.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),passed=value['passed'],checks=len(checks)),ensure_ascii=False))
    if not value['passed']:raise SystemExit(1)

if __name__=='__main__':main()

"""Finite public motion/receipt author checks. No integration or device draws."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import FrozenInstanceError
import argparse, hashlib, json, sys, traceback
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reject(fn):
    try:fn()
    except (ValueError,TypeError,AttributeError,FrozenInstanceError):return True
    return False
def bits(a,b):return np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).tobytes()==np.asarray(b).tobytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=ROOT/args.out
    if out.exists():raise FileExistsError('Fresh evidence required')
    checks=[];forbidden=[];truth=[];archive={};error=None;parent={};before={};after={}
    def ck(name,ok,details=None):checks.append(dict(name=name,passed=bool(ok),details=details))
    def deny(*a,**k):forbidden.append('forbidden_pipeline');raise AssertionError('no pipeline')
    class Frame(SimpleNamespace):
        def __getattribute__(self,key):
            if key in {'external_geom','is_target','object_position','object_velocity','object_pose','mass','friction','target_id','raw_qpos','true_support'}:
                truth.append(key);raise AssertionError('truth access '+key)
            return super().__getattribute__(key)
    try:
        from v25_response_identification.observer import (build,PublicMotionObserver,RobotAcquisitionSnapshot25,SLIP_DOMAIN_ROLE)
        from v13_a1.codec import Reference21,EnginePort21,REFERENCE_AXES21
        from v13_joint.plant import ExecutionReceipt21
        from v24_canonical_task_fusion.runtime import source_identity
        parent=source_identity(['ratio_effort']);before=dict(parent)
        production={'v25_response_identification/observer.py':sha(ROOT/'v25_response_identification/observer.py')}
        axes=tuple('rh_fixture_'+str(i) for i in range(24))+('transport_lift',)
        q=np.zeros(25);v=np.zeros(25);v[:3]=[.03,-.04,.05]
        snap=RobotAcquisitionSnapshot25(.01,q,v,np.eye(3),axes)
        J=np.zeros((3,25));J[:3,:3]=np.eye(3)
        calls=[]
        def fk(sample,patch):
            calls.append((sample.time_s,patch.robot_body))
            if patch.robot_body not in range(5):raise ValueError('outside known fixture robot')
            return SimpleNamespace(time_s=sample.time_s,axes25=axes,point_world=patch.link_point,
                point_velocity_world=J@sample.velocity25,jacobian3x25=J)
        basis=np.tile(np.array([[1.,0,0],[0,1.,0]]),(5,1,1))
        f=Frame(time=.02,sample_time=.01,age=.01,normal=np.array([2.,0,0,0,0]),
            valid=np.ones(5,bool),distributed_valid=np.ones(5,bool),contact_present=np.array([True,False,False,False,False]),
            slip_valid5=np.array([True,False,False,False,False]),tangent_basis=basis,slip_local=np.tile([.1,.2],(5,1)),
            slip_domain_role=SLIP_DOMAIN_ROLE,slip_domain_available=True,slip_domain_acquisition_time_s=.01,slip_domain_time_s=.02,
            robot_position25=q,robot_velocity25=v,
            sensor_characteristics=dict(slip_sigma_m_s=.003,position_sigma_m=.0002,force_sigma_N=.02),
            skin_patches=[dict(digit=0,robot_body=0,link_point=[.1,.2,0.],link_normal=[0.,0.,1.],
                reference_position=[.1,.2,0.],reference_normal=[0.,0.,1.],sample_time=.01,age=.01,normal=2.)])
        def run(frame=f,snapshot=snap):return build(frame,robot_snapshot=snapshot,point_kinematics=fk,stream_id='s',epoch_id='e')
        p=run();d=p.digits[0]
        ck('one digit creates exactly one pair and same-acquisition velocity is added once',p.row_keys==((0,0),(0,1)) and np.allclose(d.external_projection2_m_s,[.13,.16],atol=1e-16,rtol=0))
        xi=np.array([.2,-.1,.3,.4,-.2,.1]);expected=np.r_[np.eye(3),np.zeros((0,3))] if False else np.array([.2,-.1,.3])+np.cross(xi[3:],np.array([.1,.2,0.]))
        ck('conditional C has independent rigid velocity-field sign oracle',np.allclose(d.conditional_C2x6@xi,expected[:2],atol=1e-16,rtol=0))
        ck('no identity or unobserved full twist from C',p.diagnostics['actor_association']=='UNASSIGNED' and not p.diagnostics['same_actor_verified'] and not p.diagnostics['full_twist_available'])
        ck('known rows and nested diagnostics are immutable including write flags',reject(lambda:d.external_projection2_m_s.setflags(write=True)) and reject(lambda:d.diagnostics.__setitem__('x',1)) and reject(lambda:setattr(p,'status','x')))
        ck('source arrays are copied rather than aliased',not np.shares_memory(d.raw_relative_slip2_m_s,f.slip_local))
        ck('packet is finite JSON with unknowns represented as null',bool(json.dumps(p.to_dict(),allow_nan=False)) and p.digits[1].raw_relative_slip2_m_s is None)
        split=deepcopy(f);split.skin_patches=[dict(f.skin_patches[0],normal=1.),dict(f.skin_patches[0],normal=1.)]
        sp=run(split)
        ck('same-location half-pressure splitting preserves e C and row count',bits(sp.external_projection_m_s,p.external_projection_m_s) and bits(sp.conditional_C_Mx6,p.conditional_C_Mx6) and len(sp.row_keys)==2)
        two=deepcopy(split);two.skin_patches[1]['link_point']=[.2,.3,0.];a=run(two);two.skin_patches.reverse();b=run(two)
        ck('patch permutation preserves aggregate geometry within float roundoff',np.allclose(a.conditional_C_Mx6,b.conditional_C_Mx6,atol=2e-16,rtol=0) and bits(a.external_projection_m_s,b.external_projection_m_s))
        scaled=deepcopy(split);scaled.normal[0]*=3
        for patch in scaled.skin_patches:patch['normal']*=3
        ck('uniform normal rescaling does not create more motion information',bits(run(scaled).conditional_C_Mx6,p.conditional_C_Mx6) and run(scaled).digits[0].noise_group==d.noise_group)
        bad=deepcopy(split);bad.skin_patches.pop()
        qbad=run(bad)
        ck('missing pressure support is unknown not renormalized to complete instrument',not qbad.available and qbad.digits[0].raw_relative_slip2_m_s is not None and 'public_patch_pressure_coverage_incomplete' in qbad.digits[0].reasons)
        bad=deepcopy(f);bad.skin_patches=[]
        ck('positive contact without geometry is unknown not known P0',not run(bad).available and not run(bad).known_empty)
        empty=deepcopy(f);empty.skin_patches=[];empty.normal[:]=0.;empty.contact_present[:]=False;empty.slip_valid5[:]=False
        ep=run(empty)
        ck('explicit valid no contact is a known empty row set not measured stationary motion',ep.known_empty and not ep.available and ep.external_projection_m_s.shape==(0,) and ep.conditional_C_Mx6.shape==(0,6))
        bad=deepcopy(empty);del bad.contact_present
        ck('missing contact-present mask cannot claim known no contact',not run(bad).known_empty)
        bad=deepcopy(f);bad.valid[0]=False;bad.slip_valid5[0]=False;bad.normal[0]=np.nan
        ck('dropout preserves unknown pair and no zero row',not run(bad).available and run(bad).digits[0].raw_relative_slip2_m_s is None)
        bad=deepcopy(f);bad.slip_domain_available=False
        ck('instrument warmstart-domain missing remains unknown',not run(bad).available)
        ck('missing same-age robot snapshot preserves raw slip but no external projection',not run(snapshot=None).available and run(snapshot=None).digits[0].raw_relative_slip2_m_s is not None)
        shifted=RobotAcquisitionSnapshot25(.005,q,v,np.eye(3),axes)
        ck('current or wrong-age robot history is not substituted',not run(snapshot=shifted).available)
        bad=deepcopy(f);bad.age=.005
        ck('age clock mismatch invalidates derived rows',not run(bad).available)
        bad=deepcopy(f);bad.sample_time=.025
        ck('future acquisition and non-grid/bool/string clock rejected',reject(lambda:run(bad)) and reject(lambda:run(Frame(**dict(vars(f),time=.0201)))) and reject(lambda:run(Frame(**dict(vars(f),time=True)))) and reject(lambda:run(Frame(**dict(vars(f),time='.02')))))
        bad=deepcopy(f);bad.skin_patches[0]['external_geom']=77
        ck('public patch external identity extras rejected',reject(lambda:run(bad)))
        Q=np.array([[0.,1.],[-1.,0.]])
        rotated=deepcopy(f);rotated.tangent_basis[0]=Q@basis[0];rotated.slip_local[0]=Q@f.slip_local[0]
        rp=run(rotated)
        ck('basis change is retained as coordinates rather than unmarked fixed-channel difference',np.allclose(rp.external_projection_m_s,Q@p.external_projection_m_s,atol=1e-16,rtol=0) and not bits(rp.digits[0].projection_world2x3,d.projection_world2x3) and rp.diagnostics['basis_changes_require_explicit_common_subspace_not_zero_filled_rotation'])
        ck('position/slip noise are shared per digit and full covariance is not manufactured',d.noise_group==sp.digits[0].noise_group and not d.diagnostics['complete_error_covariance_available'] and d.diagnostics['shared_digit_position_and_slip_noise'])
        class Geometry:
            raw_joint_names=axes[:24]
            def point_kinematics(self,sample,patch):
                ss=RobotAcquisitionSnapshot25(sample.time_s,sample.q25,sample.velocity25,np.eye(3),axes)
                return fk(ss,patch)
        ad=SimpleNamespace(_last_time=.02,_motion_samples={2:dict(time_s=.01,q=q,v=v,R=np.eye(3),reason=None)},geometry=Geometry(),
            observe=deny,sample=deny,current_contact_tasks=deny,servo=SimpleNamespace(evaluate=deny))
        obs=PublicMotionObserver(stream_id='s',epoch_id='e');op=obs.observe(f,ad)
        ck('observer consumes existing cache without sample/observe/native admission calls',op.available and not forbidden)
        ck('identical repeated read returns same immutable packet',obs.observe(f,ad) is op and len(obs.packets())==1)
        refreshed=deepcopy(f);refreshed.robot_reference21=np.ones(21);refreshed.robot_actuator_target21=np.ones(21)
        ck('command-memory refresh on same tick is not a new measurement',obs.observe(refreshed,ad) is op)
        bad=deepcopy(f);bad.slip_local[0,0]+=.01
        ck('changed consumed sensor payload on duplicate tick rejects',reject(lambda:obs.observe(bad,ad)))
        bad=deepcopy(f);bad.time=.015;bad.age=.005;bad.slip_domain_time_s=.015
        ck('read rewind requires new epoch',reject(lambda:obs.observe(bad,ad)))
        cold=PublicMotionObserver(stream_id='s',epoch_id='cold');ad2=deepcopy(ad);ad2._motion_samples={}
        ck('missing adapter history does not cause sampling or future fill',not cold.observe(f,ad2).available and not forbidden)
        bias=.001234;pre=np.zeros(21);pre[18]=.02;requested=pre.copy();requested[0]=.002;executed=pre.copy();executed[0]=.001
        def receipt(t=.02):
            eng=executed.copy();eng[18]+=bias
            return ExecutionReceipt21(Reference21(requested,t),Reference21(executed,t),EnginePort21(eng,t,bias),'release',2,True,False)
        pe=pre.copy();pe[18]+=bias
        snapshot=dict(time_s=.02,reference21=Reference21(pre,.02),engine21=EnginePort21(pe,.02,bias),robot_position25=q,robot_velocity25=v)
        event=obs.on_receipt(receipt(),snapshot)
        ck('actual receipt increment is executed not requested and bias is not counted twice',event.actual_reference_increment21[0]==.001 and event.actual_reference_increment21[18]==0 and bits(event.executed_engine21,receipt().engine.values))
        ck('receipt immutable/idempotent and does not backfill old acquisition input',obs.on_receipt(receipt(),snapshot) is event and event.diagnostics['old_acquisition_input_not_backfilled'] and len(obs.packets())==1 and reject(lambda:event.executed_reference21.setflags(write=True)))
        bad=dict(snapshot,object_pose=[0]*7)
        ck('precommit truth and clock/authority mismatch reject',reject(lambda:obs.on_receipt(receipt(),bad)) and reject(lambda:obs.on_receipt(receipt(.025),snapshot)) and reject(lambda:obs.on_receipt(receipt(),dict(snapshot,engine21=np.zeros(21)))))
        s2=dict(snapshot,time_s=.025,reference21=Reference21(pre,.025),engine21=EnginePort21(pe,.025,bias))
        obs.on_receipt(receipt(.025),s2)
        ck('causal query cannot see future receipt and adjacent intervals do not prove integration hold',len(obs.receipts(available_at_s=.02))==1 and len(obs.input_intervals())==1 and not obs.input_intervals()[0]['substep_hold_verified'])
        snap3=dict(snapshot,time_s=.035,reference21=Reference21(pre,.035),engine21=EnginePort21(pe,.035,bias))
        obs.on_receipt(receipt(.035),snap3)
        ck('missing receipt interval is explicitly nonconsecutive not filled',not obs.input_intervals()[-1]['consecutive_5ms'])
        ck('constructor rejects fake numeric strings/bools in known robot geometry',reject(lambda:RobotAcquisitionSnapshot25(.01,q.astype(str),v,np.eye(3),axes)) and reject(lambda:RobotAcquisitionSnapshot25(True,q,v,np.eye(3),axes)))

        # A bounded archived PUBLIC stream through real private robot FK only.
        import research_imports,mujoco,hand_plant
        from foundation_config import FoundationConfig,OPEN
        from v20_contact_exit.adapter import ContactExitAdapter
        from unittest.mock import patch
        pipeline=('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce')
        path=ROOT/'output/v20_small_nominal_transport_v1/runs/nominal__ratio_effort__619741/trace.npz'
        archive['trace_sha256']=sha(path)
        with np.load(path,allow_pickle=False) as z:times=z['time'];encoded=z['observation_json']
        frames=[Frame(**json.loads(str(encoded[i]))) for i in np.flatnonzero((times>=11.8-1e-9)&(times<=12.1+1e-9))]
        with patch.multiple(mujoco,**{name:deny for name in pipeline if hasattr(mujoco,name)}):
            model,_=hand_plant._model(FoundationConfig());adapter=ContactExitAdapter(model,open18=np.concatenate(OPEN))
            actual=PublicMotionObserver(stream_id='archive_public',epoch_id='0');packets=[]
            for frame in frames:
                adapter.observe(frame)
                # Measurement creation must not ask the proposal/native gates.
                adapter.current_contact_tasks=deny
                packets.append(actual.observe(frame,adapter))
            available=[x for x in packets if x.available]
        archive.update(frames=len(frames),available=len(available),rows_by_packet=[len(x.row_keys) for x in packets],
            source='existing archived observation_json only; private static robot FK; no new physical trajectories')
        ck('real archived public stream builds aligned pairs without integration or native/proposal gate',len(frames)==61 and len(available)>0 and not forbidden,archive)
        ck('real archived packets are finite JSON and never duplicate a digit pair',all(len(x.row_keys)==2*len({k[0] for k in x.row_keys}) and bool(json.dumps(x.to_dict(),allow_nan=False)) for x in packets))
        after=source_identity(['ratio_effort'])
        ck('all parent192 sources unchanged',len(before)==192 and before==after)
        ck('no physics/device truth access observed',not forbidden and not truth)
    except Exception:
        error=traceback.format_exc()
    result=dict(schema='V25_PUBLIC_OBSERVER_AUTHOR_V1',passed=error is None and all(c['passed'] for c in checks),
        physics_steps=0,new_sensor_draws=0,checks=checks,error=error,forbidden_calls=forbidden,truth_access=truth,
        production_source_sha256=locals().get('production',{}),parent192_source_sha256=before,parent192_unchanged=before==after,
        checker_sha256=sha(__file__),archive=archive,created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Pure closed-form fixtures plus bounded archived public packets/private robot FK; no controller, actor write, integration, forward, contact force query, 6D estimator, model fitting or physical performance claim.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],error=error,out=str(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

"""Pure/zero-integration A1 author checks. No physical rollout is executed."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import mujoco
import research_imports
from foundation_config import FoundationConfig
from hand_plant import HandPlant
from .codec import Reference21, EnginePort21, ReferenceActor21, REFERENCE_AXES21
from .proposals import NativeProposalContext, NativeControlState, build_native_proposal
from .plant import A1Plant

ROOT = Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    target=Path(args.out)
    if target.exists():raise FileExistsError('Fresh author evidence required')
    checks=[];guarded={};calls={}
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,TypeError):check(name,True);return
        raise AssertionError(name+' did not reject')
    for name in ('mj_step','mj_step1','mj_step2'):
        guarded[name]=getattr(mujoco,name);calls[name]=0
        def forbid(*a,_name=name,**kw):calls[_name]+=1;raise AssertionError('Forbidden integration '+_name)
        setattr(mujoco,name,forbid)
    try:
        for name,value in [('wrong shape',np.zeros(20)),('bool',np.ones(21,bool)),('nonfinite',np.r_[np.nan,np.zeros(20)]),('string', ['0']*21)]:
            rejects('Reference21 '+name,lambda v=value:Reference21(v,0.))
        rejects('axis reorder',lambda:Reference21(np.zeros(21),0.,tuple(reversed(REFERENCE_AXES21))))
        rejects('invalid clock',lambda:Reference21(np.zeros(21),True))
        rejects('negative reference clock',lambda:Reference21(np.zeros(21),-.005))
        rejects('negative engine clock',lambda:EnginePort21(np.zeros(21),-.005,.02))
        rejects('negative state clock',lambda:NativeControlState(-.005,np.zeros(5),0.,np.zeros(2),0.))
        rejects('negative hidden clip',lambda:NativeControlState(0.,np.zeros(5),0.,np.zeros(2),-.001))
        rejects('negative release clock',lambda:NativeControlState(0.,np.zeros(5),0.,np.zeros(2),0.,-.005,np.zeros(5),np.zeros(5),0.))
        rejects('future release clock',lambda:NativeControlState(0.,np.zeros(5),0.,np.zeros(2),0.,.005,np.zeros(5),np.zeros(5),0.))
        check('legal negative alpha/lift reference stays allowed',NativeControlState(0.,-np.ones(5),-.01,np.zeros(2),0.).lift_ref==-.01)
        ref=Reference21(np.zeros(21),0.)
        check('typed input owns readonly values',not ref.values.flags.writeable)
        original=HandPlant._apply_ctrl
        old_writer_calls=[0]
        def forbidden_old_writer(self):old_writer_calls[0]+=1;raise AssertionError('A1 called old engine writer')
        # Only an author-test monkeypatch, restored before constructing old oracle.
        # Production files contain no monkeypatch/global factory replacement.
        HandPlant._apply_ctrl=forbidden_old_writer
        try:
            isolated=A1Plant(FoundationConfig())
            isolated.set_closure(np.ones(5)*.8);isolated.set_motion(.1,[.01,-.02]);isolated.begin_release()
            isolated.control_steps=123;isolated._apply_ctrl();isolated.reset()
        finally:HandPlant._apply_ctrl=original
        check('all A1 paths avoid old writer',old_writer_calls[0]==0)
        check('first reset already routes actor',isolated.actor21.write_count>=5)
        rejects('EnginePort21 cannot masquerade as Reference21',lambda:isolated.actor21.encode(EnginePort21(np.zeros(21),0.,isolated.actor21.lift_bias_m)))
        rejects('raw array cannot masquerade as Reference21',lambda:isolated.actor21.apply(isolated.data,np.zeros(21)))
        ctx=isolated.a1_context
        state=NativeControlState(0.,np.ones(5)*.7,.1,np.zeros(2),0.)
        alpha_input=state.alpha5.copy();pure=build_native_proposal(ctx,state)
        check('proposal did not mutate input',same(state.alpha5,alpha_input))
        check('proposal outputs readonly alpha',not pure.alpha_after5.flags.writeable)
        fake=SimpleNamespace(ctrl=np.full(22,17.))
        before=fake.ctrl.copy();count=isolated.actor21.write_count
        port=isolated.actor21.apply(fake,pure.reference21)
        nonrobot=np.array([i for i in range(22) if i not in set(isolated.actor21.indices21)],int)
        check('sole actor leaves environment control untouched',same(fake.ctrl[nonrobot],before[nonrobot]))
        check('one actor call is one counted write',isolated.actor21.write_count==count+1)
        check('lift bias applied once',port.values[18]==pure.reference21.values[18]+isolated.actor21.lift_bias_m)
        check('typed codec roundtrip legal old-like lift',same(isolated.actor21.decode(port).values,pure.reference21.values))
        comparisons=[]
        for shape in ('capsule','box','sphere'):
            old=HandPlant(FoundationConfig(shape=shape));new=A1Plant(FoundationConfig(shape=shape))
            check(shape+' reset qpos exact',same(old.data.qpos,new.data.qpos))
            check(shape+' reset qvel exact',same(old.data.qvel,new.data.qvel))
            check(shape+' reset all ctrl exact',same(old.data.ctrl,new.data.ctrl))
            check(shape+' no reset actual-data warmstart drift',same(old.data.qacc_warmstart,new.data.qacc_warmstart))
            model_before={key:np.asarray(getattr(new.model,key)).copy() for key in ('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','actuator_dyntype')}
            # Both legal and deliberately preclip-outside reference states;
            # these are algebra fixtures, not physical actions or new permissions.
            for index,alpha in enumerate((np.zeros(5),np.ones(5)*.8,np.array([-3.,2.,3.,-2.,4.]))):
                for p in (old,new):
                    p.alpha=np.asarray(alpha,float).copy();p.lift_ref=.08;p.wrist_ref=np.array([.01,-.02]);p.env['obstacle_x']=-.3;p._apply_ctrl()
                check(shape+' holding ctrl '+str(index),same(old.data.ctrl,new.data.ctrl))
                check(shape+' holding hidden clip '+str(index),same([old.max_hidden_clip],[new.max_hidden_clip]))
            for p in (old,new):p.set_closure(np.ones(5)*.9);p.set_motion(.1,[.02,-.03]);p.begin_release()
            for tick in (0,1,119,120,159,160,279,280,439,440,800):
                for p in (old,new):p.control_steps=tick;p.data.time=tick*.005+.0025;p._apply_ctrl()
                check(shape+' release ctrl tick'+str(tick),same(old.data.ctrl,new.data.ctrl))
                check(shape+' release alpha tick'+str(tick),same(old.alpha,new.alpha))
                check(shape+' release lift tick'+str(tick),same([old.lift_ref],[new.lift_ref]))
                check(shape+' release hidden clip tick'+str(tick),same([old.max_hidden_clip],[new.max_hidden_clip]))
            for key,value in model_before.items():check(shape+' native model unchanged '+key,same(getattr(new.model,key),value))
            comparisons.append(dict(shape=shape,release_ticks=11,all_exact=True))
        # Static single-writer/side-effect boundary: proposal module has no ctrl writes.
        tree=ast.parse((ROOT/'v13_a1/proposals.py').read_text(encoding='utf-8'))
        check('pure proposals have no data.ctrl store',not any(isinstance(n,ast.Attribute) and n.attr=='ctrl' and isinstance(n.ctx,ast.Store) for n in ast.walk(tree)))
        check('zero integration calls',all(value==0 for value in calls.values()))
        sources={name:sha(ROOT/'v13_a1'/name) for name in ('codec.py','proposals.py','plant.py','author_verify.py')}
        result=dict(schema='v13_a1_author_pure_zero_integration_v1',passed=True,checks=checks,physics_steps=0,
                    integration_calls=calls,old_writer_calls_during_a1_test=old_writer_calls[0],source_sha256=sources,
                    original_model_oracles=comparisons,
                    scope='Pure/zero-integration mechanical and type checks only; independent full96 command-stream and physical9 bridge remain separate gates')
        with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps(dict(passed=True,checks=len(checks),physics_steps=0,out=str(target),sha256=sha(target)),ensure_ascii=False))
    finally:
        for name,function in guarded.items():setattr(mujoco,name,function)

if __name__=='__main__':main()

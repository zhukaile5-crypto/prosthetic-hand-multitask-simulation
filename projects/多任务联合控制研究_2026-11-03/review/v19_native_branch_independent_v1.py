"""Ten selected native branch cross-component scopes, not author91 rerun.

Reviewer authored task_qp, did not implement native_branch. Static compile and
private F067 reset/fixed-tendon/transmission allowed; no plant, integration,
forward, FK, contact, inverse or actor write. Fixtures use another tendon axis.
"""
from pathlib import Path
from copy import copy
from dataclasses import replace
from contextlib import ExitStack
from unittest.mock import patch
from collections import Counter
import sys,json,hashlib,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SOURCE=ROOT/'v19_contact_reference/native_branch.py';PIN='10ed80e816d0d5c10d6bba4b814db26bcb8c3942aece1e64bd39b59974528cb0'
STEM=ROOT/'review/v19_native_branch_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def main():
    if any(STEM.with_suffix(s).exists() for s in ('.json','.md')):raise FileExistsError(STEM)
    if sha(SOURCE)!=PIN:raise ValueError('Candidate source differs')
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v19_contact_reference.native_branch import NativeBranchBuilder21
    from v13_a1.codec import Reference21,ReferenceActor21
    from v11.robot_servo_effort import RobotServoEffort
    scopes=[];forbidden=[];allowed=Counter();context={};time_s=12.345;axis=7
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden '+name)
        return f
    def scope(name,fn):
        try:scopes.append(dict(name=name,passed=True,detail=fn()))
        except Exception as exc:scopes.append(dict(name=name,passed=False,error=repr(exc),traceback=traceback.format_exc()))
    with ExitStack() as guards:
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_fwdAcceleration','mj_fwdConstraint','mj_Euler','mj_RungeKutta','mj_kinematics','mj_comPos','mj_collision','mj_contactForce','mj_inverse'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant')))
        guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor.apply')))
        for name in ('mj_resetData','mj_tendon','mj_transmission'):
            original=getattr(mujoco,name)
            def wrapped(*a,_name=name,_original=original,**kw):allowed[_name]+=1;return _original(*a,**kw)
            guards.enter_context(patch.object(mujoco,name,wrapped))
        static,xmlsha=hand_plant._model(FoundationConfig())
        def setup(gain=4.,bias=.0,modified=True):
            model=copy(static)
            if modified:
                aid=int(model.actuator('rh_A_FFJ0').id);ci=int(model.actuator_ctrladr[aid]);oi=int(model.actuator_outadr[aid])
                model.actuator_gainprm[aid,:]=0.;model.actuator_gainprm[aid,0]=gain;model.actuator_gaintype[aid]=mujoco.mjtGain.mjGAIN_FIXED
                model.actuator_biasprm[aid,:]=0.;model.actuator_biasprm[aid,0]=bias;model.actuator_biastype[aid]=mujoco.mjtBias.mjBIAS_AFFINE
                model.actuator_ctrllimited[ci]=True;model.actuator_ctrlrange[ci]=[-.75,.75]
                model.actuator_forcelimited[oi]=True;model.actuator_forcerange[oi]=[-.5,.5]
            b=NativeBranchBuilder21(model);q=model.qpos0[b.servo.qadr].copy();q[24]=.06;v=np.zeros(25)
            return b,q,v
        def build(b,q,v,current,preferred,lo=-.5,hi=.5,**changes):
            u=np.zeros(21);u[18]=.06;u[axis]=current;p=u.copy();p[axis]=preferred;lower=u.copy();upper=u.copy();lower[axis]=lo;upper[axis]=hi
            ref=Reference21(u,time_s);kw=dict(q25=q,v25=v,robot_axes25=b.robot_axes25,time_s=time_s,current_reference=ref,
                actual_engine=b.actor.encode(ref),preferred_reference=Reference21(p,time_s),reference_lower21=lower,reference_upper21=upper)
            kw.update(changes);return b.build(**kw)
        def samples(b,q,v,branch,points):
            assert branch.available,branch.reason
            oracle=RobotServoEffort(b.servo.model);records=[]
            for point in points:
                u=branch.anchor_reference21.copy();u[axis]=point;ref=Reference21(u,time_s)
                got=b.evaluate_reference(branch,ref);actual=oracle.evaluate(q,v,b.actor.encode(ref).values,time_s=time_s)
                assert got['available'],got
                assert got['native_effort21'].tobytes()==actual.predicted_effort21.tobytes()
                assert got['actual_delta_effort21'].tobytes()==(actual.predicted_effort21-branch.p0_effort21).tobytes()
                expected=branch.offset21+branch.gain21*(u-branch.current_reference21)
                assert got['affine_delta_effort21'].tobytes()==expected.tobytes()
                assert got['affine_error21'].tobytes()==(got['actual_delta_effort21']-expected).tobytes()
                assert not got['writer_called'] and not got['inverse_called'] and not got['physical_safety_certified']
                records.append(dict(reference=float(point),native_effort=got['native_effort21'][axis],actual_delta=got['actual_delta_effort21'][axis],affine_error=got['affine_error21'][axis]))
            return records
        b,q,v=setup();context.update(builder=b,q=q,v=v)
        def ordinary():
            branch=build(b,q,v,0.,.3);assert branch.gain21[axis]==4 and branch.offset21[axis]==0 and branch.reference_upper21[axis]==.125
            context['ordinary']=branch
            return dict(kind=branch.branch_kind21[axis],samples=samples(b,q,v,branch,[-.1,0.,.1]))
        scope('affine FFJ0 fixed tendon full-native points',ordinary)
        def plateau():
            branch=build(b,q,v,.25,.4);assert branch.gain21[axis]==0 and branch.reference_upper21[axis]>branch.reference_lower21[axis]
            got=b.evaluate_reference(branch,Reference21(np.where(np.arange(21)==axis,.4,branch.anchor_reference21),time_s))
            assert got['available'] and got['actual_delta_effort21'][axis]==0 and got['reference21'][axis]==.4
            return dict(reference_can_change_without_effort_delta=True,kind=branch.branch_kind21[axis],samples=samples(b,q,v,branch,[.25,.4]))
        scope('known force platform retains reference degree',plateau)
        def kink():
            right=build(b,q,v,.125,.3);left=build(b,q,v,.125,0.)
            assert right.gain21[axis]==0 and right.reference_lower21[axis]==.125
            assert left.gain21[axis]==4 and left.reference_upper21[axis]==.125
            return dict(right_kind=right.branch_kind21[axis],left_kind=left.branch_kind21[axis],samples=samples(b,q,v,left,[0.,.125]))
        scope('same exact kink selects distinct declared sides',kink)
        def negative():
            nb,nq,nv=setup(-4.);branch=build(nb,nq,nv,0.,.3);assert branch.gain21[axis]==-4
            return dict(samples=samples(nb,nq,nv,branch,[0.,.1]))
        scope('negative native slope retains sign',negative)
        def zero():
            zb,zq,zv=setup(0.,.125);branch=build(zb,zq,zv,0.,.3)
            assert branch.available and branch.branch_kind21[axis]=='KNOWN_ZERO_NATIVE_GAIN' and branch.gain21[axis]==0
            assert branch.reference_lower21[axis]==-.5 and branch.reference_upper21[axis]==.5
            return dict(samples=samples(zb,zq,zv,branch,[-.4,.4]))
        scope('known zero gain not missing model',zero)
        def debt():
            branch=build(b,q,v,.25,0.,-.1,.1)
            assert branch.available and branch.p0_effort21[axis]==.5 and branch.anchor_effort21[axis]==0 and abs(branch.offset21[axis]-.5)<1e-15
            assert not branch.diagnostics['current_inside_actual_domain21'][axis] and branch.delta_upper21[axis]<0
            result=samples(b,q,v,branch,[-.1,0.,.1]);fixed=build(b,q,v,.25,0.,0.,0.)
            assert fixed.available and fixed.gain21[axis]==0 and fixed.offset21[axis]==-.5 and fixed.delta_lower21[axis]==fixed.delta_upper21[axis]==-.25
            return dict(offset=branch.offset21[axis],p0=branch.p0_effort21[axis],anchor=branch.anchor_effort21[axis],samples=result)
        scope('current outside TASK preserves originalp0 and nonzerooffset',debt)
        def clocks():
            branch=context['ordinary'];wrong=build(b,q,v,0.,.3,time_s=time_s+.005)
            assert not wrong.available
            assert not b.evaluate_reference(branch,Reference21(branch.anchor_reference21,time_s+.005))['available']
            axes=list(b.robot_axes25);axes[0],axes[1]=axes[1],axes[0]
            assert not build(b,q,v,0.,.3,robot_axes25=axes).available
            other,_,_=setup();assert not other.evaluate_reference(branch,Reference21(branch.anchor_reference21,time_s))['available']
            return dict(build_clock_rejected=True,final_clock_rejected=True,foreign_owner_rejected=True,axes_order_rejected=True)
        scope('typed clock axes and owner boundary',clocks)
        def domain():
            branch=context['ordinary'];u=branch.anchor_reference21.copy();u[axis]=np.nextafter(branch.reference_upper21[axis],np.inf)
            assert not b.evaluate_reference(branch,Reference21(u,time_s))['available']
            assert not build(b,q,v,0.,.6).available
            altered=replace(branch,offset21=branch.offset21+np.eye(21)[axis]);assert not b.evaluate_reference(altered,Reference21(branch.anchor_reference21,time_s))['available']
            return dict(oneULP_outside_rejected=True,preferred_not_clipped=True,descriptor_mutation_rejected=True)
        scope('actual selected-cell endpoint and descriptor guards',domain)
        def standard():
            sb,sq,sv=setup(modified=False);branch=build(sb,sq,sv,.02,.021,.015,.025)
            values=samples(sb,sq,sv,branch,[branch.reference_lower21[axis],branch.reference_upper21[axis]])
            axis_names=sb.servo.axes25;row=branch.transmission21x25[axis]
            nonzero=[(axis_names[i],float(row[i])) for i in np.flatnonzero(row)]
            assert len(nonzero)==2 and all(c==1 for _,c in nonzero)
            return dict(unmodified_known_robot=True,tendon_coefficients=nonzero,samples=values)
        scope('unmodified model and actual tendon two-coefficient transmission',standard)
        def mutation():
            branch=context['ordinary'];aid=b.servo.model.actuator('rh_A_FFJ0').id;b.servo.model.actuator_group[aid]=3
            assert not b.evaluate_reference(branch,Reference21(branch.anchor_reference21,time_s))['available']
            return dict(bound_robot_mode_mutation_rejected=True)
        scope('source mode mutation invalidates old branch',mutation)
    old=json.loads((ROOT/'review/v18_small_plan_v1.json').read_text(encoding='utf-8'))['source_sha256']
    preserved=all(sha((ROOT.parent/'五指策略接入_2026-11-03'/k[11:]) if k.startswith('foundation/') else ROOT/k)==v for k,v in old.items())
    report=dict(schema='v19_native_branch_independent_v1',passed=len(scopes)==10 and all(s['passed'] for s in scopes) and not forbidden and sha(SOURCE)==PIN and preserved,
        role='cross-component independent branch review; reviewer authored task_qp but not native_branch',scopes=scopes,source_sha256=PIN,checker_sha256=sha(__file__),
        static_generated_xml_sha256=xmlsha,axis='FFJ0',forbidden_calls=forbidden,allowed_private_robot_calls=dict(allowed),old144_preserved=preserved,
        physics_steps=0,forward_calls=0,world_FK_calls=0,contact_queries=0,actor_writes=0,
        limitations=['Modified known-robot native laws are finite algebra fixtures, not new plant experiments.','Independent F067 value instance, not an independent implementation of F067.','Reference freedom on gain0 does not certify actual future robot geometry, contact force or placement.'])
    STEM.with_suffix('.json').write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    STEM.with_suffix('.md').write_text(f"# V19原生支域独立有限门\n\n{'PASS' if report['passed'] else 'FAIL'}；{sum(s['passed'] for s in scopes)}/10 scope。绑定源码 `{PIN}`。审者未参与native_branch实现，是task_qp作者的跨组件审查。\n\n采用另一FFJ0腱轴的解析native律和少量独立F067点值核对：平台参考自由、零/负gain、kink单侧、offdomain当前p0与offset、单点偿债、时钟/owner/source/1ULP越界和原模型tendon两系数。无Model以外的物理构造；仅私有F067 Data/reset/tendon/transmission，0积分/forward/FK/contact/actor写入。\n\n不是作者91批量复跑，也不是实际几何/压力/放置认证。详情见[JSON]({STEM.name}.json)。\n",encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],scopes=len(scopes),failed=[s for s in scopes if not s['passed']],sha256=sha(STEM.with_suffix('.json')),md_sha256=sha(STEM.with_suffix('.md'))),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

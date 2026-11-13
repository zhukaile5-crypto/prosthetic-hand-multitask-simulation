"""Actual JointPlant construction/reset only; zero integration steps."""
from pathlib import Path
import hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(F));sys.dont_write_bytecode=True
EXPECTED={'design.py':'c995635d5f05c3f757a6480d95db8ce159c3aab08c4a7ec4c4e1ca8ff25d6c54',
 'lifecycle.py':'cc36f4a38fc45f87a187723e20d297ebd724a54d1849cb869330eaec593bbb25',
 'plant.py':'357efc8cb24757a8512320a55f66e963e0c869cc21c694cd867d7728b256721e'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    checks=[];forbidden=[];forward_count=[0]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+' '+str(detail))
    def reject(name,call):
        try:call()
        except (ValueError,TypeError,RuntimeError):check(name,True)
        else:check(name,False,'invalid/stale input accepted')
    check('bound_final_three_sources',all(sha(ROOT/'v13_joint'/n)==s for n,s in EXPECTED.items()))
    import research_imports,mujoco,hand_plant
    for name in ('mj_step','mj_step1','mj_step2'):
        def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('No integration '+_name)
        setattr(mujoco,name,deny)
    native_forward=mujoco.mj_forward
    def forward(*a,**k):forward_count[0]+=1;return native_forward(*a,**k)
    mujoco.mj_forward=forward
    def old_writer(*a,**k):forbidden.append('old_HandPlant_writer');raise AssertionError('Old writer forbidden')
    hand_plant.HandPlant._apply_ctrl=old_writer
    from foundation_config import FoundationConfig
    from v13_joint.plant import JointPlant
    from v13_joint.lifecycle import release_nominal
    from v13_joint.design import reference_rates
    from v13_a1.codec import Reference21
    p=JointPlant(FoundationConfig());initial=p.reference_snapshot();gen=p.mode_generation
    check('first_generation_zero_and_zero_integration',gen==0 and p.control_steps==0 and p.data.time==0.)
    check('initial_reference_engine_identity',p.actor21.encode(initial).values.tobytes()==p.engine_snapshot().values.tobytes())
    lo,hi=p.reference_authority21();axis=5;request=initial.values.copy();request[axis]=min(hi[axis],initial.values[axis]+.1)
    first=p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=gen)
    second=p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=gen)
    allowed=reference_rates('holding')*p.config.control_dt
    check('same_tick_repeated_commit_not_accumulated',first.executed.values.tobytes()==second.executed.values.tobytes() and
        np.all(abs(second.executed.values-initial.values)<=allowed+1e-15))
    check('receipt_is_committed_actual_port',second.engine.values.tobytes()==p.engine_snapshot().values.tobytes() and
        second.executed.values.tobytes()==p.reference_snapshot().values.tobytes() and second.execution_changed)
    held=p.engine_snapshot().values.copy();p.alpha[:]=10.
    for t in (.0005,.0025,.0045):p.data.time=t;p._apply_ctrl()
    check('substep_callback_never_reconstructs_alpha',p.engine_snapshot().values.tobytes()==held.tobytes() and p.reference_snapshot().time_s==0.)
    reject('legacy_closure_setter_rejected',lambda:p.set_closure(np.zeros(5)))
    reject('legacy_motion_setter_rejected',lambda:p.set_motion(.1,[0.,0.]))
    reject('legacy_execute_delta_rejected',lambda:p.execute_delta(np.zeros(5)))
    reject('clock_mismatch_rejected',lambda:p.commit_reference21(Reference21(request,.005),mode='holding',expected_generation=gen))
    reject('bool_generation_rejected',lambda:p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=True))
    reject('release_commit_requires_begin',lambda:p.commit_reference21(Reference21(request,p.time),mode='release',expected_generation=gen))
    actual_entry=p.reference_snapshot();p.begin_release()
    check('release_actual21_entry_not_alpha',p.joint_release_entry.reference21.values.tobytes()==actual_entry.values.tobytes() and
        p.mode_generation==gen+1 and p.release_started==0.)
    stale=p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=gen)
    check('stale_holding_preempted_by_release',stale.mode=='release' and stale.stale_plan_replaced and
        p._joint_mode=='release' and stale.requested.values.tobytes()==request.tobytes())
    check('same_tick_modechange_respects_release_anchor_envelope',np.all(abs(stale.executed.values-initial.values)<=reference_rates('release')*.005+1e-15))
    middle=p.release_proposal(.7)
    check('independent_non_synergy_axis_released_from_actual_entry',abs(middle.values[axis]-.5*actual_entry.values[axis])<1e-15)
    reject('release_before_entry_rejected',lambda:p.release_proposal(-.001))
    open18=np.concatenate(p.open_pose)
    reject('string_open_rejected',lambda:release_nominal(p.joint_release_entry,.1,['0']*18,lo,hi))
    reject('bool_authority_rejected',lambda:release_nominal(p.joint_release_entry,.1,open18,np.zeros(21,dtype=bool),hi))
    oldgen=p.mode_generation;p.reset()
    check('reset_clears_committed_release_anchor_and_increments_generation',p.mode_generation==oldgen+1 and p._joint_committed is None and
        p.release_started is None and p.joint_release_entry is None and p._joint_anchor_time is None)
    resetref=p.reference_snapshot();old=p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=gen)
    check('reset_does_not_accept_ABA_old_generation',old.stale_plan_replaced and old.executed.values.tobytes()==resetref.values.tobytes())
    p.control_steps=3;p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=p.mode_generation)
    p.control_steps=2
    reject('public_command_clock_rewind_rejected',lambda:p.commit_reference21(Reference21(request,p.time),mode='holding',expected_generation=p.mode_generation))
    p.reset();idx=int(p.actor21.indices21[5]);p.data.ctrl[idx]+=.001;before=p.mode_generation
    reject('tampered_engine_cannot_be_claimed_actual_release_entry',p.begin_release)
    check('entry_mismatch_does_not_latch_release',p.release_started is None and p.mode_generation==before)
    check('integration_and_old_writer_calls_zero',not forbidden,forbidden)
    check('three_sources_unchanged',all(sha(ROOT/'v13_joint'/n)==s for n,s in EXPECTED.items()))
    return dict(schema='v13_joint_lifecycle_independent_zero_step_v1',passed=True,checks=checks,
        source_sha256=EXPECTED,forbidden_calls=forbidden,forward_calls_for_constructor_and_resets=forward_count[0],
        integration_steps=0,physics_trajectory_executed=False,
        scope=['Actual constructor/reset static forward is allowed and counted; no advance or mj_step.',
            'Same-tick envelope bounds net deviation from its anchor, not cumulative command TV over multiple same-time replacements.',
            'Rates are declared controller design limits, not hardware velocity or future tracking certificates.',
            'This source is not connected to a policy runtime; no force-quality or closed-loop success claim.'])

if __name__=='__main__':
    out=ROOT/'review/v13_joint_lifecycle_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    try:value=run()
    except Exception as exc:value=dict(schema='v13_joint_lifecycle_independent_zero_step_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),integration_steps=0)
    value['checker_sha256']=sha(Path(__file__))
    with out.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=value['passed'],checks=len(value.get('checks',[])),output=str(out),sha256=sha(out))))
    raise SystemExit(0 if value['passed'] else 1)

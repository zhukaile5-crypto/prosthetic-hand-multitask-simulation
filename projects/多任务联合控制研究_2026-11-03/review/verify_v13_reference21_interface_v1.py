"""Robot-only zero-integration inventory and old48 command codec bridge.

This is interface evidence, not a V13 actor/controller or physical experiment.
Old object coordinates are excluded before the known robot actuation API call.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('development_v12_pilot_v1', 'development_v12_hold_seed617901_v1',
           'development_v12_hold_seed617902_v1', 'development_v12_transport_v1')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--out', required=True)
    args = args.parse_args(); target = Path(args.out)
    if target.exists(): raise FileExistsError('Fresh interface evidence required')
    sys.path.insert(0, str(ROOT))
    import research_imports
    import mujoco
    from hand_plant import _model
    from foundation_config import FoundationConfig
    from v11.robot_servo_effort import RobotServoEffort, ACTUATORS21
    from v10.robot_contact_space import MechanicalReferenceBasis
    guarded = {}; calls = {}
    for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_forwardSkip', 'mj_inverse', 'mj_collision', 'mj_kinematics', 'mj_comPos'):
        if hasattr(mujoco, name):
            guarded[name] = getattr(mujoco, name); calls[name] = 0
            def forbidden(*a, _name=name, **k):
                calls[_name] += 1
                raise AssertionError('Forbidden pipeline call in zero-step interface inventory: '+_name)
            setattr(mujoco, name, forbidden)
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition: raise AssertionError(name)
    f067 = ROOT/'v11/robot_servo_effort.py'; initial_f067 = sha(f067)
    check('f067 is frozen original', initial_f067 == 'f0671211d2e15cc96eeb8c87207a43bc1839d55512ab18bdc2cebff4ed0a052c')
    try:
        # Compile the declared nominal platform. No HandPlant/MjData from an
        # actual simulator is accepted; all actuation calculations use private data.
        model, model_sha = _model(FoundationConfig(shape='capsule'))
        servo = RobotServoEffort(model)
        check('known static affine robot model available', not servo.unavailable)
        bias = float(model.body_subtreemass[model.body('rh_forearm').id])*9.81/float(model.actuator_gainprm[model.actuator('lift_motor').id,0])
        basis = MechanicalReferenceBasis(model, lift_lower_m=0., lift_upper_m=.1)
        check('engine port order matches exact21 basis', same(servo.ctrladr, basis.actuator_indices21))
        check('known lift bias equals existing mechanical basis', bias == basis.lift_motor_offset)
        q = np.asarray(model.qpos0[servo.qadr],float); v = np.zeros(25)
        sample = servo.evaluate(q, v, np.mean(servo.ctrl_range, axis=1))
        check('known actuation snapshot produced', sample.status == 'KNOWN_ROBOT_ACTUATION_ALGEBRA')
        A = np.asarray(sample.transmission21x25)
        check('full actuator transmission rank21', np.linalg.matrix_rank(A) == 21)
        check('original holding reference basis rank8', np.linalg.matrix_rank(np.asarray(basis.held_to_command21)) == 8)
        rows = []
        for i, name in enumerate(ACTUATORS21):
            aid = int(servo.aids[i]); supported = np.flatnonzero(A[i])
            trn = int(model.actuator_trntype[aid]); target_id = int(model.actuator_trnid[aid,0])
            if trn == int(mujoco.mjtTrn.mjTRN_TENDON):
                target_name = model.tendon(target_id).name
            else:
                target_name = model.joint(target_id).name
            rows.append(dict(index21=i, actuator_name=name, engine_ctrl_address=int(servo.ctrladr[i]),
                reference_unit='m' if i==18 else 'rad (fixed angular tendon sum for J0)',
                effort_unit='N' if i==18 else 'Nm in transmission coordinate',
                transmission_type='fixed_joint_tendon' if trn==int(mujoco.mjtTrn.mjTRN_TENDON) else 'single_robot_joint',
                transmission_target=target_name, generalized_columns=[dict(axis=servo.axes25[int(j)],coefficient=float(A[i,j])) for j in supported],
                raw_engine_ctrl_range=servo.ctrl_range[i].tolist(), control_limited=bool(servo.ctrl_limited[i]),
                native_force_range=servo.force_range[i].tolist(), force_limited=bool(servo.force_limited[i]),
                gain3=model.actuator_gainprm[aid,:3].tolist(), bias3=model.actuator_biasprm[aid,:3].tolist(),
                physical_reference_rate_status='UNKNOWN', algorithm_rate_policy_may_be_declared=True))
        j0 = [i for i,n in enumerate(ACTUATORS21) if n.endswith('J0')]
        check('four J0 tendons', len(j0)==4)
        nulls = []
        for i in j0:
            support = np.flatnonzero(A[i]); check(ACTUATORS21[i]+' exactly two coefficients1',len(support)==2 and np.array_equal(A[i,support],np.ones(2)))
            vector = np.zeros(25);vector[support[0]]=1.;vector[support[1]]=-1.
            check(ACTUATORS21[i]+' difference coordinate unactuated', np.array_equal(A@vector,np.zeros(21)))
            nulls.append(dict(tendon=ACTUATORS21[i],difference_axes=[servo.axes25[int(j)] for j in support]))
        jobs=[]
        for cohort in COHORTS:
            base=ROOT/'output'/cohort;done=read(base/'execution_complete.json')
            check(cohort+' complete old cohort',done['execution_complete'] and not done['runtime_failures'])
            jobs.extend(read(base/'manifest.json')['jobs'])
        check('all old48 included',len(jobs)==48)
        native_oracles=[];records=[]
        maxima={phase:np.zeros(21) for phase in ('preparation','holding','release')}
        for job in jobs:
            path=Path(job['destination'])/'trace.npz';fingerprint=sha(path);meta=read(path.with_name('metadata.json'))
            check(job['method']+' '+str(job['seed'])+' source complete and immutable',meta['completed'] and fingerprint==meta['trace_sha256'])
            with np.load(path,allow_pickle=False) as z:
                t=z['time'];before=z['ctrl_before'];after=z['ctrl'];actual21=z['robot_actuator_target21']
                # Whitelist before any model call; no object free pose or velocity.
                q25=np.asarray(z['qpos'])[:,servo.qadr].copy();v25=np.asarray(z['qvel'])[:,servo.dadr].copy()
            check('actual pre21 equals selected ctrl_before '+str(len(records)),same(actual21,before[:,servo.ctrladr]))
            codecs={}
            for label,source in (('before',before[:,servo.ctrladr]),('after',after[:,servo.ctrladr])):
                reference=source.copy();reference[:,18]-=bias
                engine=reference.copy();engine[:,18]+=bias
                check('lift bias codec bytes '+label+' '+str(len(records)),same(engine,source))
                codecs[label]=dict(frames=len(t),all21_byte_exact=True,engine_outside_ctrlrange_count21=np.sum((source<servo.ctrl_range[:,0])|(source>servo.ctrl_range[:,1]),axis=0).tolist())
            actual_ref=after[:,servo.ctrladr].copy();actual_ref[:,18]-=bias
            rates=np.abs(np.diff(actual_ref,axis=0))/.005
            for phase,lo,hi in (('preparation',0.,7.5),('holding',7.5,12.),('release',12.,16.0001)):
                selected=(t[1:]>=lo-1e-9)&(t[1:]<hi-1e-9)
                maxima[phase]=np.maximum(maxima[phase],rates[selected].max(axis=0))
            # No physics: compare actual21 vs native22 API layouts at fixed
            # predeclared robot encoder states, including release.
            oracle=[]
            for k in (0,1500,2460,3200):
                direct=servo.evaluate(q25[k],v25[k],before[k,servo.ctrladr],time_s=float(t[k]))
                selected=servo.evaluate(q25[k],v25[k],before[k],control_layout='model22',time_s=float(t[k]))
                okay=(direct.status==selected.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and same(direct.generalized_actuator_effort25,selected.generalized_actuator_effort25) and same(direct.predicted_effort21,selected.predicted_effort21))
                check('native21/22 effort exact '+str(len(records))+' tick'+str(k),okay)
                oracle.append(dict(tick=k,time_s=float(t[k]),exact=True))
            check('original trace SHA remains unchanged '+str(len(records)),sha(path)==fingerprint)
            records.append(dict(method=job['method'],scenario=job['scenario'],seed=job['seed'],motion_mode=job['motion_mode'],trace=str(path),trace_sha256=fingerprint,codecs=codecs,native_oracles=oracle))
        check('all forbidden integration/FK/force pipeline calls zero',all(v==0 for v in calls.values()))
        check('f067 source unchanged',sha(f067)==initial_f067)
        output=dict(schema='v13_full21_reference_interface_zero_dynamics_v1',created_utc=datetime.now(timezone.utc).isoformat(),
            passed=True,physics_steps=0,checks=checks,forbidden_calls=calls,author_script_sha256=sha(__file__),f067_sha256=initial_f067,
            nominal_platform_model_sha256=model_sha,robot_axes25=list(servo.axes25),actuator_ports21=rows,
            lift_engine_bias_m=bias,full_transmission_rank=21,holding_basis_rank=8,j0_unactuated_difference_coordinates=nulls,
            complete_old48_records=records,observed_reference_rate_maxima_by_phase={k:v.tolist() for k,v in maxima.items()},
            rate_note='Observed rates are retrospective descriptions, not hardware limits or A2 policy choices. A2 may predeclare separate software rates/regularization and test them in simulation.',
            command_scope='A1 codec bridge only: no independent21-axis action, no dynamic equivalence or contact-force prediction claim',
            model_only_effort=True,physical_safety_certified=False,global_model_admitted=False)
        with target.open('x',encoding='utf-8') as stream:json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps(dict(passed=True,checks=len(checks),old_records=len(records),before_after_frames=2*48*3201,native_oracles=48*4,physics_steps=0,sha256=sha(target),out=str(target)),ensure_ascii=False))
    finally:
        for name,function in guarded.items():setattr(mujoco,name,function)

if __name__=='__main__':main()

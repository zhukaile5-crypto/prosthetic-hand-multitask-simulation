"""Software-only gate for the fixed five-fork protocol."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import gzip
import json
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from identification.v26_small_reference_response import run_identification as run
from v13_a1.codec import Reference21, EnginePort21
from v13_joint.plant import ExecutionReceipt21


def main():
    out = ROOT/'review/v26_small_reference_software_v1.json'
    if out.exists():
        raise FileExistsError(out)
    identity = run.source_identity()
    checks, forbidden = [], []
    error = None

    def ck(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))

    def deny(*args, **kwargs):
        forbidden.append('forbidden physical/model operation')
        raise AssertionError(forbidden[-1])

    try:
        import mujoco
        names = ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_forwardSkip', 'mj_contactForce', 'mj_collision', 'MjModel', 'MjData')
        with patch.multiple(mujoco, **{name:deny for name in names if hasattr(mujoco,name)}):
            # Validate the real archived bridge chain and exact 199+42 identities.
            old = run.parent.read(ROOT/'review/v25_first_five_registration_v2.json')
            numeric, environment = run.parent.numerical_identity(), run.parent.environment()
            reg = dict(old, assignment=run.ASSIGNMENT, source_sha256=identity, identification_contract=run.protocol())
            digest = run.validate_registration(reg, identity, numeric, environment)
            ck('fixed assignment, current parent198 and archived exact bridge fingerprint agree',
               len(identity)==199 and len(digest)==64 and tuple(run.COMPONENTS)==run.ARMS)
            wrong = dict(reg, assignment=dict(run.ASSIGNMENT, seed=619742))
            try:
                run.validate_registration(wrong, identity, numeric, environment)
                rejected = False
            except ValueError:
                rejected = True
            ck('different context rejected before a branch can start', rejected)
            readout = run.parent.read(ROOT/'review/v25_first_five_response_v1.json')
            archived = readout['branches']['zero']['program'][0]
            carrier = np.array(archived['carrier_at_origin21'])
            lower = np.array(archived['actual_TASK_lower21'])
            upper = np.array(archived['actual_TASK_upper21'])

            class Nominal:
                def __init__(self, low=lower):
                    self.low = low
                    self.decision_evidence = []
                    self.public_state = {'phase':'TRANSFER'}
                    self.reconciled = []

                def action(self, frame, **kw):
                    proposal = carrier.copy()
                    if frame.time>run.ORIGIN_S+run.EPS:
                        proposal[run.LIFT_AXIS] += .0002  # Adaptive path differs from carrier.
                    self.decision_evidence.append(dict(actual_reference_lower21=self.low.tolist(),
                        actual_reference_upper21=upper.tolist(), geometry_proposal_accepted=True,
                        emitted_hierarchy_score={'available':True}))
                    return Reference21(proposal, frame.time), {}

                def reconcile(self, receipt):
                    self.reconciled.append(receipt)

            def selection(arm, nominal=None, *, ticks=1, generation=3):
                probe = run.SmallProbeCoordinator(nominal or Nominal(), run.Program(arm))
                for i in range(ticks):
                    frame = SimpleNamespace(time=run.ORIGIN_S+i*.005)
                    ref, debug = probe.action(frame=frame,generation=generation)
                return probe, ref, debug

            zero, zr, zd = selection('zero', ticks=40)
            repeated, rr, rd = selection('zero_repeat', ticks=40)
            ck('two named zero controls emit the same payload and fixed 21-axis target',
               zr.values.tobytes()==rr.values.tobytes() and zd==rd and
               all(r['carrier_hold_active'] for r in zero.records+repeated.records))
            lift, lr, _ = selection('lift_plus')
            finger, fr, _ = selection('finger_single')
            mixed, mr, _ = selection('mixed_holdout')
            lift_delta, finger_delta, mixed_delta = lr.values-carrier, fr.values-carrier, mr.values-carrier
            ck('archived actual-domain amplitudes deliver the two preregistered signed inputs',
               np.flatnonzero(lift_delta).tolist()==[18] and abs(lift_delta[18]-.00045)<1e-14 and
               np.flatnonzero(finger_delta).tolist()==[7] and abs(finger_delta[7]+.001556460157595807)<1e-14 and
               np.array_equal(mixed_delta,lift_delta+finger_delta))
            blocked = lower.copy()
            blocked[run.FINGER_AXIS] = carrier[run.FINGER_AXIS]
            unavailable, ur, _ = selection('finger_single', Nominal(blocked), ticks=40)
            partial, pr, _ = selection('mixed_holdout', Nominal(blocked), ticks=40)
            ck('unsupported finger holds zero carrier across adaptive nominal proposals',
               unavailable.origin_exposure_status=='UNEXPOSED' and np.array_equal(ur.values,zr.values) and
               all(r['carrier_hold_active'] and not r['active'] for r in unavailable.records))
            ck('partial mixed keeps only its declared supported lift component',
               partial.origin_exposure_status=='PARTIAL_UNSUPPORTED' and np.array_equal(pr.values,lr.values))
            nominal = Nominal()
            plant = SimpleNamespace(v16_supervisor=nominal,joint_release_entry=object(),bind_supervisor=deny)
            fake = SimpleNamespace(time=run.ORIGIN_S,sensors=SimpleNamespace(_last_time=13.935),
                v16_supervisor=nominal,release_driver=nominal,plant=plant,probe_coordinator=None)
            run.arm_program(fake,run.Program('lift_plus'))
            ck('selector installation preserves all three owner identities without bind',
               fake.v16_supervisor is fake.release_driver is plant.v16_supervisor is nominal and
               fake.probe_coordinator.nominal_supervisor is nominal)
            port = EnginePort21(lr.values,run.ORIGIN_S,0.)
            receipt = ExecutionReceipt21(lr,lr,port,'release',3,False,False)
            lift.reconcile(receipt)
            ck('frozen reconcile forwards the one typed receipt and retains separate actual evidence',
               lift.nominal_supervisor.reconciled==[receipt] and
               lift.records[0]['actual_receipt']['actual_reference21']==lr.values.tolist() and
               lift.records[0]['nominal_task_certificate_applies_to_actual_reference'] is False)
            fake.last_obs = SimpleNamespace(time=run.ORIGIN_S,receiver_surface=dict(time_s=run.ORIGIN_S,
                acquisition_time_s=run.ORIGIN_S-.010,age_s=.010,valid=False,normal_load_N=None,
                uncertainty6=[1.]*6,invalid_reason='dropout',sampling_phase='poststep'))
            r = run.receiver_row(fake)
            ck('receiver sidecar retains real acquisition clock, delivery and null invalid value',
               r['acquisition_time_s']==run.ORIGIN_S-.010 and r['available_at_s']==run.ORIGIN_S and
               r['valid'] is False and r['normal_load_N'] is None)
            with TemporaryDirectory(dir=ROOT/'review',prefix='.v26_software_') as d:
                path = Path(d).resolve()
                assert path.parent==(ROOT/'review').resolve()
                with patch.object(run,'fork_session',side_effect=RuntimeError('injected independent fork failure')) as fork:
                    failures = []
                    rows = run.execute_branches(object(),path,digest,failures)
                ck('five failures retain all five assignments and continue in order',
                   fork.call_count==5 and [r['key'] for r in rows]==list(run.ARMS) and
                   all(r['status']=='runtime_failed' for r in rows) and len(failures)==5)
                for arm in ('zero','zero_repeat'):
                    folder = path/'branches'/arm
                    np.savez(folder/'trace.npz',time=np.array([0.,.005]),position=np.array([1.,2.]))
                    for name in ('response_observations.jsonl.gz','response_receipts.jsonl.gz',
                                 'response_input_intervals.jsonl.gz',run.RECEIVER_STREAM,'probe_decisions.jsonl.gz'):
                        row = dict(sample=1.)
                        if name=='probe_decisions.jsonl.gz':
                            row['arm'] = arm
                        with gzip.open(folder/name,'wt',encoding='utf-8') as stream:
                            stream.write(json.dumps(row)+'\n')
                ck('zero gate compares short trace and exact public payloads',run.compare_zeros(path)['passed'])
                with gzip.open(path/'branches/zero_repeat'/run.RECEIVER_STREAM,'wt',encoding='utf-8') as stream:
                    stream.write('{"sample": 2}\n')
                ck('a receiver mismatch invalidates zero reproduction',not run.compare_zeros(path)['passed'])
    except Exception:
        error = traceback.format_exc()
    ck('all production sources unchanged and no physical calls',run.source_identity()==identity and not forbidden)
    result = dict(passed=error is None and all(c['passed'] for c in checks),checks=checks,error=error,
                  physics_steps=0,model_constructions=0,forbidden_calls=forbidden,
                  production_source_sha256=identity,checker_sha256=run.parent.sha(__file__))
    run.parent.write(out,result)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),error=error,output=str(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

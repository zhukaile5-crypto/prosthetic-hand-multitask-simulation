"""Root-only explicit physical dispatch of fixed primitive development.

Import, --describe and pure checks never step a plant. Root alone runs the
real-model --preflight, pins sources, runs --regression, then --execute.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from identification.release_basis_v2 import run_pilot as codec
from identification.release_local_response_v2.run_pilot import save_diagnostics, PHYSICAL_FIELDS
from identification.release_local_response_v2.decoder import make_plant_class

from identification.release_primitive_validation_v2 import spec
from identification.release_primitive_validation_v2.provenance import (
    PACKAGE, identity, require_lock, require_seed_inventory, snapshot, sha, write)

EXTRA_PREFIX_FIELDS = ('observation_json', 'task_request_json', 'raw_patches_json',
    'requested_action', 'estimated_normal', 'observer_uncertainty',
    'contact_model_active', 'observed_normal', 'validity', 'observed_age',
    'estimated_load')
NATIVE_REFERENCE = ROOT/'output/development_v10_hold_v1/runs/nominal__ratio_preview__610212/trace.npz'


def prefix_checks(zero,branch,origin_s):
    differences=[]
    with np.load(zero,allow_pickle=False) as a,np.load(branch,allow_pickle=False) as b:
        for field in PHYSICAL_FIELDS+('observation_json',):
            clock=a['substep_time'] if field.startswith('substep_') else a['time']
            mask=clock<=origin_s+1e-9
            x,y=a[field][mask],b[field][mask]
            equal=np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'biufc' else np.array_equal(x,y)
            if not equal:differences.append(field)
    return dict(through_inclusive_s=origin_s,fields=len(PHYSICAL_FIELDS)+1,
                mismatches=differences,exact=not differences)


def compare_full_native(actual,reference):
    fields={}
    with np.load(actual,allow_pickle=False) as a,np.load(reference,allow_pickle=False) as b:
        for field in PHYSICAL_FIELDS:
            x,y=a[field],b[field]
            fields[field]=bool(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes())
        observed=bool(np.array_equal(a['observation_json'],b['observation_json']))
    return dict(reference=str(reference),reference_sha256=sha(reference),
                actual40fields_bitexact=fields,public_observations_exact=observed,
                passed=all(fields.values()) and observed)


def make_session(job, sources):
    from v10.runtime import V10Session, public_motion_reference_bounds
    from hand_plant import CommonPreparation
    from skin_geometry import DelayedSkinGeometry
    regression = bool(job.get('bridge_regression', False))
    if regression:
        if (job['scenario'], job['motion_mode'], job['seed'], job['variant']) != (
                'nominal', 'hold', 610212, 'zero'):
            raise ValueError('Bridge regression must be the single frozen reference context')
    else:
        spec.validate_context(job['scenario'], job['motion_mode'], job['seed'], job['variant'])

    class PrimitiveSession(V10Session):
        def step_one_sensor_tick(self):
            if self.plant.release_started is None:
                self.plant.prospective_entry_eligible=bool(
                    self.handoff_passed and self.tick==spec.RELEASE_TICK)
            return super().step_one_sensor_tick()

        def metadata(self):
            m = super().metadata()
            m.update(release_primitive_validation_revision='v2',
                release_primitive_variant=job['variant'],
                release_primitive_role='bridge_regression' if regression else 'prospective_development',
                release_local_response_source_identity=sources,
                fixed_release_origin_s=spec.ORIGIN_S,
                fixed_release_full_input_control_s=spec.FULL_INPUT_S,
                fixed_release_first_full_input_physical_end_s=spec.FIRST_FULL_END_S,
                native_full16_reference_availability='FROZEN_610212_REFERENCE' if regression else 'UNAVAILABLE_NEW_SEED',
                release_response_model_available=False,
                release_force_safety_certified=False,
                primitive_entry_eligible=bool(self.plant.prospective_entry_eligible_at_release),
                primitive_offset_enabled=bool(self.plant.prospective_offset_enabled),
                primitive_actual_variant=self.plant.probe_variant,
                actual_mechanical_release_s=self.plant.release_started,
                release_scope='One fixed delayed-lowering primitive; not complete fusion architecture or force model')
            return m

    s = PrimitiveSession(method='ratio_preview', scenario=job['scenario'],
        shape='capsule', seed=job['seed'], split='robustness',
        motion_mode=job['motion_mode'])
    if s.time != 0. or s.tick != 0 or s.trace or s.release_driver is not None:
        raise RuntimeError('A pristine V10 native-release session is required')
    if s.config.timestep != .0005 or s.config.control_dt != .005 or s.observer_warmup_s != .2:
        raise ValueError('Retained V10 clock/warmup contract changed')
    original_model_sha = s.plant.model_sha256
    old_bounds = s.motion_reference_bounds
    BasePlant=make_plant_class()
    class EligiblePlant(BasePlant):
        def __init__(self,config,variant):
            self.probe_origin_tick=spec.ORIGIN_TICK
            self.prospective_requested_variant=variant
            self.prospective_entry_eligible=False
            self.prospective_entry_eligible_at_release=False
            self.prospective_offset_enabled=False
            super().__init__(config,variant)

        def begin_release(self):
            if self.release_started is None:
                eligible=bool(self.prospective_entry_eligible and self.control_steps==spec.RELEASE_TICK)
                self.prospective_entry_eligible_at_release=eligible
                self.prospective_offset_enabled=bool(eligible and self.prospective_requested_variant=='lift_slow')
                # A failed public preparation/handoff never receives the new
                # primitive. Its original early native safe release is intact.
                self.probe_variant=self.prospective_requested_variant if eligible else 'zero'
            return super().begin_release()

    s.plant = EligiblePlant(s.config, job['variant'])
    if s.plant.model_sha256 != original_model_sha:
        raise RuntimeError('Replacing recorder plant changed compiled physical model')
    s.obstacle_geom = int(s.plant.model.geom('integration_obstacle_geom').id)
    s.preparation = CommonPreparation(s.plant)
    s.last_truth = s.plant.truth()
    s.skin_geometry = DelayedSkinGeometry(s.plant)
    s.motion_reference_bounds = public_motion_reference_bounds(s.plant)
    if s.motion_reference_bounds != old_bounds:
        raise RuntimeError('Robot command authority changed during recorder replacement')
    # Keep the SAME untouched sensor/RNG, controller, follower, user trajectory.
    s.research_sources = dict(sources)
    return s


def entry_snapshot(s, target):
    result = codec.entry_snapshot(s, target)
    p = s.plant
    # The old codec covers force observer/RNG/queues/FK and task; V10 follower
    # is explicit here because it did not exist in that original session.
    follower=dict(exact_type=type(s.preview_follower).__module__+'.'+type(s.preview_follower).__qualname__,
        implementation_path='v10/runtime.py',implementation_sha256=sha(ROOT/'v10/runtime.py'),
        exact_mutable_state=vars(s.preview_follower))
    causal = dict(preview_follower=follower,
        motion_mode=s.motion_mode, observer_warmup_s=s.observer_warmup_s,
        user_trajectory=s.user_trajectory, task_request=s.task_request)
    frozen = codec.freeze(causal)
    write(target/'v10_preview_causal_state.json', frozen)
    result['v10_preview_causal_state_sha256'] = codec.digest(codec.without_wallclock(frozen))
    local = dict(probe_origin_tick=p.probe_origin_tick,
        release_tick=p.probe_release_tick, last_applied_tick=p.probe_last_tick,
        last_phi_s=p.probe_last_phi, last_lift_reference_m=p.probe_last_lift,
        last_command21=p.probe_last_command,
        total_variation21=p.probe_total_variation,
        native_reference_envelope=p.probe_envelope,
        release_decoder_state=None if p.probe_decoder is None else vars(p.probe_decoder))
    encoded = codec.freeze(local)
    write(target/'local_control_state.json', encoded)
    result['local_control_state_sha256'] = codec.digest(encoded)
    return result


def expanded_prefix(zero, branch):
    pref = prefix_checks(zero, branch, spec.ORIGIN_S)
    extra, absent = [], []
    with np.load(zero, allow_pickle=False) as a, np.load(branch, allow_pickle=False) as b:
        mask = a['time'] <= spec.ORIGIN_S+1e-9
        for field in EXTRA_PREFIX_FIELDS:
            if field not in a.files or field not in b.files:
                absent.append(field)
                continue
            x, y = a[field][mask], b[field][mask]
            equal = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in 'biufc' else np.array_equal(x, y)
            if not equal:
                extra.append(field)
    return dict(physical_and_observation=pref, extra_mismatches=extra,
                missing_required_fields=absent, exact=pref['exact'] and not extra and not absent)


def worker(job, sources):
    target = Path(job['directory'])
    target.mkdir(parents=True, exist_ok=False)
    write(target/'job.json', job)
    s, entries = None, {}
    began = time.perf_counter()
    try:
        if identity() != sources:
            raise RuntimeError('Source identity changed before worker construction')
        s = make_session(job, sources)
        while s.running:
            for tick, name in ((1500, 'handoff_boundary'),
                               (spec.RELEASE_TICK, 'scheduled_release_boundary'),
                               (spec.RELEASE_TICK+spec.ORIGIN_TICK, 'scheduled_probe_boundary')):
                if s.plant.control_steps == tick and name not in entries:
                    folder = target/name
                    folder.mkdir()
                    entries[name] = entry_snapshot(s, folder)
            s.step_one_sensor_tick()
        if s.time != 16. or len(entries) != 3:
            raise RuntimeError('Complete16s and all three pre-action boundaries required')
        s.save(target/'record')
        from evaluate_runs import evaluate_trace
        from evaluation import evaluate_research
        result = evaluate_trace(target/'record/trace.npz')
        result['research_metrics'] = evaluate_research(target/'record/trace.npz')
        result.update(component_variant=job['variant'],
            motion_mode=job['motion_mode'], prospective_development=True,
            bridge_regression=bool(job.get('bridge_regression', False)))
        write(target/'result.json', result)
        diagnostics = save_diagnostics(s, target)
        expected_applications=1+11*(3200-s.plant.probe_release_tick)
        if diagnostics['applications'] != expected_applications or diagnostics['robot_physical_samples'] != 32001:
            raise RuntimeError('Release application/raw physical coverage is incomplete')
        if diagnostics['authority_mismatches']:
            raise RuntimeError('Actual command authority check failed')
        from identification.release_primitive_validation_v2.clearance_diagnostics import evaluate
        diagnostics_clear=evaluate(target/'record/trace.npz',target/'robot_physical_stream.npz',
            json.loads((ROOT/'protocol.json').read_text(encoding='utf-8')),
            original_v3_release=result['research_metrics']['release'])
        write(target/'clearance_diagnostics.json',diagnostics_clear)
        write(target/'pilot_metadata.json', dict(source_identity=sources,
            config=vars(s.config), entry=entries,
            actual_mechanical_release_s=s.plant.release_started,
            actual_release_entry_snapshot_name=('scheduled_release_boundary' if s.plant.release_started==12. else
                'handoff_boundary' if s.plant.release_started==7.5 else None),
            primitive_entry_eligible=s.plant.prospective_entry_eligible_at_release,
            primitive_offset_enabled=s.plant.prospective_offset_enabled,
            requested_variant=job['variant'],actual_variant=s.plant.probe_variant,
            clearance_diagnostics_sha256=sha(target/'clearance_diagnostics.json'),
            release_clock_contract='native5ms-held; full input first physical end12.0405',
            command_record_sha256=sha(target/'decoder_commands.npz'),
            robot_stream_sha256=sha(target/'robot_physical_stream.npz'),
            record_trace_sha256=sha(target/'record/trace.npz'),
            no_hydration_used=True, physical_execution_owned_by='root',
            native_reference_availability='FROZEN_610212_REFERENCE' if job.get('bridge_regression') else 'UNAVAILABLE_NEW_SEED'))
        if identity() != sources:
            raise RuntimeError('Source changed during physical worker')
        return dict(job_id=job['job_id'], scenario=job['scenario'],
            motion_mode=job['motion_mode'], seed=job['seed'], variant=job['variant'],
            status='complete', entries=entries, decoder=diagnostics,
            actual_mechanical_release_s=s.plant.release_started,
            primitive_entry_eligible=s.plant.prospective_entry_eligible_at_release,
            primitive_offset_enabled=s.plant.prospective_offset_enabled,
            trace_sha256=sha(target/'record/trace.npz'),
            task_success=result['task_success'],
            full_motion_mission_success=result['research_metrics']['full_motion_mission_success'],
            guard_pass=result['research_metrics']['safety']['guard_pass'],
            release=result['research_metrics']['release'], elapsed_s=time.perf_counter()-began)
    except Exception as exc:
        failure = dict(job_id=job['job_id'], scenario=job['scenario'],
            motion_mode=job['motion_mode'], seed=job['seed'], variant=job['variant'],
            status='failed', error=repr(exc), traceback=traceback.format_exc(),
            entries=entries, elapsed_s=time.perf_counter()-began)
        if s is not None:
            failure['stopped_at_s'] = s.time
            try:
                s.save(target/'partial_record')
                failure['partial_record'] = str(target/'partial_record')
            except Exception as sub:
                failure['partial_save_error'] = repr(sub)
            try:
                if not (target/'decoder_commands.npz').exists():
                    save_diagnostics(s, target)
            except Exception as sub:
                failure['diagnostic_save_error'] = repr(sub)
        write(target/'failure.json', failure)
        return failure


def preflight():
    """Actual-model construction/command algebra only; ROOT executes this."""
    import mujoco
    guards = {name:getattr(mujoco, name) for name in ('mj_step','mj_step1','mj_step2')}
    calls = {name:0 for name in guards}
    def block(name):
        def reject(*args, **kwargs):
            calls[name] += 1
            raise RuntimeError('Dynamics forbidden during preflight')
        return reject
    for name in guards:
        setattr(mujoco, name, block(name))
    try:
        sources = identity()
        reports = []
        for job in spec.jobs(Path('not_executed')):
            s = make_session(job, sources)
            p = s.plant
            p.set_closure(np.full(5, .75))
            p.set_motion(.1, [0., 0.])
            p.control_steps = spec.RELEASE_TICK
            p.prospective_entry_eligible=True
            p.begin_release()
            for tick in range(spec.RELEASE_TICK, 3201):
                p.control_steps = tick
                p._apply_ctrl()
            # Exercise the actual snapshot codec including V10 follower without
            # sampling a sensor, updating a controller or integrating a plant.
            encoded = codec.freeze(dict(base=codec.side_state(s),
                preview_follower=dict(exact_type=type(s.preview_follower).__module__+'.'+type(s.preview_follower).__qualname__,
                    implementation_sha256=sha(ROOT/'v10/runtime.py'),
                    exact_mutable_state=vars(s.preview_follower))))
            if p.data.time != 0. or p.probe_mismatches or p.lift_ref != 0.:
                raise AssertionError('No-dynamics actual command bridge failed')
            reports.append(dict(job_id=job['job_id'], applications=len(p.probe_rows),
                physical_time=float(p.data.time), warmup_s=s.observer_warmup_s,
                snapshot_codec_sha256=codec.digest(codec.without_wallclock(encoded)),
                zero_fullctrl_exact=all(np.array_equal(row[-p.model.nu:], row[-2*p.model.nu:-p.model.nu])
                     for row in p.probe_rows) if job['variant']=='zero' else None))
        if sources != identity():
            raise RuntimeError('Source changed during no-dynamics preflight')
        return dict(passed=True, physics_steps=0, dynamics_guard_calls=calls,
                    sources=sources, reports=reports,
                    scope='32 actual V10 constructor/recorder/command/codec contexts; no sensor/controller ticks, no dynamics')
    finally:
        for name, original in guards.items():
            setattr(mujoco, name, original)


def fresh_output(value):
    out = (ROOT/value).resolve()
    allowed = (ROOT/'output').resolve()
    if not out.is_relative_to(allowed) or out == allowed or out.exists():
        raise ValueError('Fresh strict descendant of research/output required')
    return out


def bridge_job(out):
    return dict(job_id='bridge__nominal__hold__610212__zero',
        scenario='nominal', motion_mode='hold', seed=610212, variant='zero',
        method='ratio_preview', shape='capsule', split='robustness',
        bridge_regression=True, directory=str(out/'runs/bridge__nominal__hold__610212__zero'))


def bridge_passed(evidence, lock):
    path = Path(evidence).resolve()
    allowed = (ROOT/'output').resolve()
    if not path.is_relative_to(allowed):
        raise ValueError('Regression evidence must be in research/output')
    record = json.loads(path.read_text(encoding='utf-8'))
    if (record.get('passed') is not True or record.get('source_identity') != lock['sources']
            or record.get('dependency_lock_sha256') != sha(PACKAGE/'dependency_lock.json')):
        raise RuntimeError('Native full16s bridge regression failed or used different source lock')
    # Verify actual evidence and result bytes rather than trusting its passed flag.
    folder = path.parent
    for rel, digest in record['input_sha256'].items():
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT/p
        if not p.is_file() or sha(p) != digest:
            raise RuntimeError('Regression input identity changed')
    return dict(path=str(path), sha256=sha(path))


def summarize(out, results, sources):
    pairs = []
    bykey = {(r['scenario'],r['motion_mode'],r['seed'],r['variant']):r for r in results}
    for seed in spec.SEEDS:
        for scenario in spec.SCENARIOS:
            for motion in spec.MOTION_MODES:
                z = bykey.get((scenario,motion,seed,'zero'))
                s = bykey.get((scenario,motion,seed,'lift_slow'))
                key = dict(scenario=scenario,motion_mode=motion,seed=seed)
                if z is None or s is None or z['status']!='complete' or s['status']!='complete':
                    pairs.append(dict(**key,paired=False,reason='incomplete assigned pair retained'))
                    continue
                zp = out/'runs'/z['job_id']
                sp = out/'runs'/s['job_id']
                pref = expanded_prefix(zp/'record/trace.npz',sp/'record/trace.npz')
                with np.load(zp/'decoder_commands.npz',allow_pickle=False) as a, np.load(sp/'decoder_commands.npz',allow_pickle=False) as b:
                    clocks = all(np.array_equal(a[k],b[k]) for k in ('control_time','physical_time'))
                    actual_offsets = (np.array_equal(b['actual_phi_s']-a['actual_phi_s'],b['actual_phi_offset_s'])
                        and np.array_equal(b['actual_command21'][:,18]-a['actual_command21'][:,18],b['actual_lift_offset_m']))
                    native = np.array_equal(b['native_carrier_command21'],a['actual_command21'])
                    only_lift = np.array_equal(a['actual_command21'][:,:18],b['actual_command21'][:,:18]) and np.array_equal(a['actual_command21'][:,19:],b['actual_command21'][:,19:])
                exact_entries = z['entries'] == s['entries']
                pairs.append(dict(**key,paired=bool(pref['exact'] and exact_entries and clocks and actual_offsets and native and only_lift),
                    prefix=pref,release_and_origin_entries_exact=exact_entries,
                    decoder_clocks_exact=clocks,actual_offsets_against_executed_zero=actual_offsets,
                    native_carrier_matches_executed_zero=native,only_lift_reference_changed=only_lift))
    unchanged = identity() == sources
    failures = [r for r in results if r['status']!='complete']
    return dict(execution_complete=len(results)==32,planned=32,
        completed=sum(r['status']=='complete' for r in results),results=results,failures=failures,
        pairs=pairs,sources_unchanged=unchanged,
        passed=bool(len(results)==32 and not failures and unchanged and len(pairs)==16 and all(p['paired'] for p in pairs)),
        passed_scope='execution/paired identities only; material force failures are retained outcomes',
        performance_claim_allowed=False,model_admitted=False,material_safety_certified=False)


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ('describe','preflight','pin-sources','regression','execute'):
        mode.add_argument('--'+flag,action='store_true')
    p.add_argument('--out')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--regression-evidence')
    args = p.parse_args()
    if args.describe:
        print((PACKAGE/'plan.json').read_text(encoding='utf-8'))
        return 0
    if args.preflight:
        report = preflight()
        path = PACKAGE/f'preflight_{time.time_ns()}.json'
        write(path,report)
        print(json.dumps(dict(output=str(path),passed=True,physics_steps=0)))
        return 0
    if args.pin_sources:
        inventory = require_seed_inventory()
        lock = dict(schema='prospective_release_primitive_source_lock_v2',sources=identity(),
            seed_inventory_sha256=sha(PACKAGE/'seed_inventory.json'),
            inherited_v2_lock_sha256=sha(ROOT/'identification/release_local_response_v2/dependency_lock.json'),
            environment=codec.environment(),physics_started=False)
        write(PACKAGE/'dependency_lock.json',lock)
        print(json.dumps(dict(sources=len(lock['sources']),lock_sha256=sha(PACKAGE/'dependency_lock.json'))))
        return 0
    if not args.out:
        p.error('Fresh --out required for root physical execution')
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must be1..8')
    lock = require_lock()
    out = fresh_output(args.out)
    out.mkdir()
    snapshot(out,lock)
    jobs = [bridge_job(out)] if args.regression else spec.jobs(out)
    write(out/'plan.json',json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8')))
    write(out/'seed_inventory.json',require_seed_inventory())
    write(out/'environment.json',codec.environment())
    write(out/'manifest.json',dict(schema='release_primitive_validation_manifest_v2',
        jobs=jobs,scientific_assignments=0 if args.regression else 32,
        bridge_regression_assignments=1 if args.regression else 0,sources=lock['sources'],
        plan_sha256=sha(out/'plan.json'),dependency_lock_sha256=sha(PACKAGE/'dependency_lock.json'),
        source_snapshot_manifest_sha256=sha(out/'source_snapshot_manifest.json'),
        performance_claim_allowed=False))
    ledger = [dict(job=j,status='assigned') for j in jobs]
    write(out/'assignment_ledger.json',ledger)
    write(out/'results_partial.json',[])
    write(out/'runtime_failures.json',[])
    if args.regression:
        r = worker(jobs[0],lock['sources'])
        comparison = None
        if r['status']=='complete':
            comparison = compare_full_native(Path(jobs[0]['directory'])/'record/trace.npz',NATIVE_REFERENCE)
        passed = bool(r['status']=='complete' and comparison and comparison['passed'] and identity()==lock['sources'])
        record = dict(schema='release_primitive_native_bridge_full16s_v2',passed=passed,
            source_identity=lock['sources'],dependency_lock_sha256=sha(PACKAGE/'dependency_lock.json'),
            result=r,comparison=comparison,scientific_samples=0,
            input_sha256={str(NATIVE_REFERENCE.relative_to(ROOT)):sha(NATIVE_REFERENCE)})
        if r['status']=='complete':
            path=Path(jobs[0]['directory'])/'record/trace.npz'
            record['input_sha256'][str(path.relative_to(ROOT))]=sha(path)
        ledger[0]['status']=r['status']
        (out/'assignment_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
        write(out/'results.json',[r])
        write(out/'regression_complete.json',record)
        print(json.dumps(dict(output=str(out),passed=passed,scientific_samples=0)))
        return 0 if passed else 2
    # A failed/missing bridge keeps the entire prospective assignment as
    # unexecuted. No unregistered native repeats are hidden in scientific32.
    try:
        if not args.regression_evidence:
            raise RuntimeError('--regression-evidence required before prospective32')
        bridge = bridge_passed(args.regression_evidence,lock)
    except Exception as exc:
        failure=dict(stage='native_bridge_gate',error=repr(exc))
        for row in ledger:
            row['status']='not_executed_due_to_native_bridge_failure'
        for name,value in [('assignment_ledger.json',ledger),('runtime_failures.json',[failure])]:
            (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        write(out/'results.json',[])
        write(out/'execution_complete.json',dict(execution_complete=False,planned=32,completed=0,
            failures=[failure],sources_unchanged=identity()==lock['sources'],physics_started=False))
        return 2
    results=[]
    began=time.perf_counter()
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,j,lock['sources']):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                result=future.result()
            except Exception as exc:
                result=dict(job_id=job['job_id'],scenario=job['scenario'],motion_mode=job['motion_mode'],
                    seed=job['seed'],variant=job['variant'],status='worker_exception',error=repr(exc))
            results.append(result)
            next(row for row in ledger if row['job']==job)['status']=result['status']
            for name,value in [('assignment_ledger.json',ledger),('results_partial.json',results),
                ('runtime_failures.json',[r for r in results if r['status']!='complete'])]:
                (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            print(json.dumps(dict(completed=len(results),planned=32,job_id=job['job_id'],status=result['status'])),flush=True)
    final=summarize(out,results,lock['sources'])
    final.update(wall_seconds=time.perf_counter()-began,bridge_regression=bridge)
    write(out/'results.json',results)
    write(out/'summary.json',final)
    write(out/'execution_complete.json',dict(execution_complete=final['execution_complete'],
        passed=final['passed'],planned=32,completed=final['completed'],runtime_failures=final['failures'],
        sources_unchanged=final['sources_unchanged'],results_sha256=sha(out/'results.json'),
        summary_sha256=sha(out/'summary.json'),bridge_regression=bridge,performance_claim_allowed=False))
    print(json.dumps(dict(output=str(out),passed=final['passed'],completed=final['completed'])),flush=True)
    return 0 if final['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())

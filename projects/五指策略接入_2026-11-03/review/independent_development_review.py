"""Independent read-only audit of saved development data; no simulator imports.

Writes only its own timestamped review certificate. Does not execute a policy,
import an experiment evaluator, or modify an admission/source/experimental file.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'output/development_integration_v1'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def close(a, b, tol=1e-9):
    return bool(np.allclose(a, b, atol=tol, rtol=0., equal_nan=True))


def dwell(t, flag, lo, hi, duration):
    # Count continuous grid intervals rather than counting N samples as N*dt.
    indexes = np.flatnonzero((t >= lo-1e-9) & (t <= hi+1e-9))
    start = None
    for i in indexes:
        if flag[i]:
            if start is None:
                start = i
            if t[i] - t[start] >= duration-1e-9:
                return float(t[i])
        else:
            start = None
    return None


def aggregate(patches):
    pressure = np.zeros(5)
    force = np.zeros((5, 3))
    position = np.zeros((5, 3))
    direction = np.zeros((5, 3))
    slip = np.zeros((5, 3))
    shear = np.zeros(5)
    max_slip = np.zeros(5)
    aux = np.zeros(6)
    aux_pressure = 0.
    for p in patches:
        n = max(0., float(p['normal']))
        if n <= 1e-10:
            continue
        f = np.asarray(p['force'])
        x = np.asarray(p['position'])
        torque = np.asarray(p.get('torque', [0., 0., 0.]))
        i = int(p['digit'])
        if not 0 <= i < 5:
            aux += np.r_[f, np.cross(x, f)+torque]
            aux_pressure += n
            continue
        nd = np.asarray(p['normal_dir'])
        nd = nd/np.linalg.norm(nd)
        v = np.asarray(p['slip_velocity'])
        pressure[i] += n
        force[i] += f
        position[i] += n*x
        direction[i] += n*nd
        slip[i] += n*v
        shear[i] += np.linalg.norm(f-n*nd)
        max_slip[i] = max(max_slip[i], float(np.linalg.norm(v-nd*(nd@v))))
    basis = np.zeros((5, 2, 3))
    for i in range(5):
        if pressure[i] > 1e-10:
            position[i] /= pressure[i]
            slip[i] /= pressure[i]
        norm = np.linalg.norm(direction[i])
        if norm < 1e-12:
            basis[i] = [[0., 0., 1.], [0., 1., 0.]]
        else:
            n = direction[i]/norm
            anchor = np.array([0., 0., 1.]) if abs(n[2]) <= .9 else np.array([0., 1., 0.])
            tangent = anchor - n*(anchor@n)
            tangent /= np.linalg.norm(tangent)
            basis[i] = [tangent, np.cross(n, tangent)]
    return pressure, force, position, slip, basis, shear, max_slip, aux, aux_pressure


def audit_rng(z, m, observations):
    """Recreate fixed draw stream and observation packets from saved raw skin."""
    noise = m['noise']
    rng = np.random.default_rng(int(m['seed'])+910000)
    packets = []
    max_errors = {k: 0. for k in ('normal', 'tangential', 'contact_pos', 'slip_local',
                                'shear_magnitude', 'slip_magnitude', 'auxiliary_wrench',
                                'auxiliary_pressure', 'sample_time', 'age')}
    valid_matches = True
    draws_hash = hashlib.sha256()
    for tick, t in enumerate(z['time']):
        n, f, x, v, basis, shear, slips, aux, auxp = aggregate(json.loads(z['raw_patches_json'][tick]))
        # Preserve exact distribution-call shapes/order, including draws for
        # absent contacts and channels not used in a particular task phase.
        nn = rng.normal(0., noise['force_sigma'], 5)
        tn = rng.normal(0., noise['tangential_sigma'], (5, 2))
        pn = rng.normal(0., noise['position_sigma'], (5, 3))
        vn = rng.normal(0., noise['slip_sigma'], (5, 2))
        en = rng.normal(0., noise['effort_sigma'])
        afs = noise['force_sigma'] if noise['auxiliary_force_sigma'] is None else noise['auxiliary_force_sigma']
        ams = afs*.05 if noise['auxiliary_moment_sigma'] is None else noise['auxiliary_moment_sigma']
        an = np.r_[rng.normal(0., afs, 3), rng.normal(0., ams, 3)]
        apn = rng.normal(0., afs)
        u = rng.random(6)
        draws_hash.update(np.r_[nn, tn.ravel(), pn.ravel(), vn.ravel(), en, an, apn, u].tobytes())
        contact = n > .02
        xn = x.copy()
        xn[contact] += pn[contact]
        vl = np.einsum('ijk,ik->ij', basis, v)+vn
        vl[~contact] = 0.
        sm = np.maximum(0., slips+vn[:, 0])
        sm[~contact] = 0.
        valid = u[:5] >= noise['dropout_probability']
        for lo, hi, digits in noise['dropout_windows']:
            if lo <= t < hi:
                valid[digits] = False
        ap = noise['dropout_probability'] if noise['auxiliary_dropout_probability'] is None else noise['auxiliary_dropout_probability']
        av = bool(noise['instrument_auxiliary'] and u[5] >= ap)
        for lo, hi in noise['auxiliary_dropout_windows']:
            if lo <= t < hi:
                av = False
        packet = dict(normal=np.maximum(0., n+nn),
            tangential=np.einsum('ijk,ik->ij', basis, f)+tn,
            contact_pos=xn, slip_local=vl,
            shear_magnitude=np.maximum(0., shear+tn[:, 0]), slip_magnitude=sm,
            auxiliary_wrench=aux+an, auxiliary_pressure=max(0., auxp+apn),
            valid=valid, auxiliary_valid=av, sample_time=float(t))
        # The distributed representation is valid in all six development runs;
        # assert recorded validity separately instead of silently assuming it.
        for k in ('normal', 'tangential', 'contact_pos', 'slip_local', 'shear_magnitude', 'slip_magnitude'):
            packet[k][~valid] = np.nan
        if not av:
            packet['auxiliary_wrench'][:] = np.nan
            packet['auxiliary_pressure'] = np.nan
        packets.append(packet)
        acquired = packets[max(0, tick-int(round(noise['delay']/.005)))]
        obs = observations[tick]
        for k in max_errors:
            expected = float(t-acquired['sample_time']) if k == 'age' else acquired[k]
            actual = np.asarray(obs[k], dtype=float)
            expected = np.asarray(expected, dtype=float)
            if not np.array_equal(np.isnan(actual), np.isnan(expected)):
                max_errors[k] = float('inf')
            elif np.any(np.isfinite(expected)):
                max_errors[k] = max(max_errors[k], float(np.nanmax(np.abs(actual-expected))))
        valid_matches &= np.array_equal(obs['valid'], acquired['valid']) and obs['auxiliary_valid'] == acquired['auxiliary_valid']
    return dict(passed=bool(valid_matches and max(max_errors.values()) < 1e-9),
        max_absolute_errors=max_errors, valid_matches=bool(valid_matches), ticks=len(packets),
        gaussian_draws_per_tick=48, uniform_draws_per_tick=6,
        reconstructed_draw_stream_sha256=draws_hash.hexdigest(),
        scope='Recomputed normal/tangent/position/slip/shear/aux packets throughout all raw timestamps; effort consumes its prescribed draw, but raw effort itself was not saved for independent de-noising.')


def metrics(z, m):
    t = z['time']
    normal, skin = z['normal'], z['skin_normal']
    support = (normal[:, 0] > .2) & ((normal > .2).sum(axis=1) >= 3)
    raised = z['position'][:, 2] >= np.asarray(m['initial_position'])[2]+.025
    motion = (abs(z['lift']-z['lift_ref']) <= .01) & (z['support'] <= .05)
    relative_error = np.linalg.norm(z['relative']-m['reference_relative'], axis=1)
    safe = (skin <= np.asarray(m['limits'])+.02).all(axis=1) & (skin.sum(axis=1) <= m['budget']+.02)
    stable = support & raised & motion & safe & (relative_error <= .005) & (np.linalg.norm(z['relative_velocity'], axis=1) <= .005) & (np.linalg.norm(z['angular_velocity'], axis=1) <= .15)
    post = dwell(t, stable, 7.5, 8.9, .2)
    release_end = dwell(t, z['all_hand_object'] <= .2, 12., 15.5, .2)
    duration, st = z['substep_duration'], z['substep_time']
    dt = float(np.median(duration))
    half = int(round(.05/(2*dt)))
    kt = np.arange(-half, half+1)*dt
    kernel = 2*50*dt*np.sinc(2*50*kt)*np.hanning(2*half+1)
    kernel /= kernel.sum()
    signal = np.vstack([z['substep_normal'], normal[-1]])
    filtered = np.column_stack([np.convolve(np.pad(signal[:, i], (half, half), mode='reflect'), kernel, mode='valid') for i in range(5)])
    aligned = np.column_stack([np.interp(t, np.r_[st, st[-1]+duration[-1]], filtered[:, i]) for i in range(5)])
    mask = (t >= 8.2-1e-9) & (t <= 8.9+1e-9)
    limits = np.asarray(m['limits'])
    excess = np.maximum(z['substep_skin_normal']-limits, 0.)
    total_excess = np.maximum(z['substep_skin_normal'].sum(axis=1)-m['budget'], 0.)
    stages = {}
    for name, subset in dict(preparation=st < 7.5-1e-9,
        policy_holding=(st >= 7.5-1e-9) & (st < 12.-1e-9), mechanical_release_and_after=st >= 12.-1e-9).items():
        stages[name] = dict(duration_s=float(duration[subset].sum()),
            finger_peak_N=z['substep_skin_normal'][subset].max(axis=0).tolist(),
            total_finger_peak_N=float(z['substep_skin_normal'][subset].sum(axis=1).max()),
            local_excess_integral_Ns=float(np.sum(excess[subset]*duration[subset, None])),
            total_excess_integral_Ns=float(total_excess[subset]@duration[subset]))
    return dict(A_S=1.4 if post is None else post-7.5,
        first_stable_after_handoff=post, whole_task_first_stable=dwell(t, stable, 0., 16., .2),
        mean_force=float(normal[mask].sum(axis=1).mean()),
        finger_tv_rate=float(np.abs(np.diff(aligned[mask], axis=0)).sum()/.7),
        raw_finger_tv_rate=float(np.abs(np.diff(normal[mask], axis=0)).sum()/.7),
        finger_sd_mean=float(np.std(normal[mask], axis=0).mean()), total_sd=float(np.std(normal[mask].sum(axis=1))),
        release_latency=None if release_end is None else release_end-12.,
        release_success=bool(release_end is not None and support[np.argmin(abs(t-12.))] and m['release_supported_at_command']),
        local_excess_integral=float(np.sum(excess*duration[:, None])), total_excess_integral=float(total_excess@duration),
        force_protection_pass=bool(np.all(z['substep_skin_normal'] <= limits+.02) and np.all(z['substep_skin_normal'].sum(axis=1) <= m['budget']+.02) and np.all(skin[-1] <= limits+.02) and skin[-1].sum() <= m['budget']+.02),
        finger_peak=np.maximum(z['substep_skin_normal'].max(axis=0), skin[-1]).tolist(), stages=stages,
        support_continuous_during_hold=bool(support[(t >= 7.5) & (t < 12.)].all()),
        no_warning=bool(z['warning_count'].max() == 0),
        record_complete=bool(len(t) == 3201 and len(st) == 32000 and close(t, np.arange(3201)*.005, 1e-8) and close(st, np.arange(32000)*.0005, 1e-7) and close(duration, .0005) and abs(duration.sum()-16.) < 1e-8),
        stable_hold_fraction=float(stable[(t >= 7.5) & (t < 12.)].mean()))


def main():
    checks, runs, pairs = {}, [], []
    trials = read(FOLDER/'trials.json')
    batch = read(FOLDER/'manifest.json')
    complete = read(FOLDER/'execution_complete.json')
    live = ROOT/'review/admission_manifest.json'
    admission = read(live)
    plan = read(ROOT/'experiment_plan.json')
    checks['trial_file_hash'] = digest(FOLDER/'trials.json') == complete['trials_sha256'] == 'a00db66016df0fa384553b188ad140903efc4585e67fd7a4e1f7fa0c537daeed'
    checks['batch_file_hash'] = digest(FOLDER/'manifest.json') == complete['manifest_sha256']
    checks['exact_six_planned_jobs'] = len(trials) == 6 and complete['execution_complete'] and complete['runtime_failures'] == 0 and batch['complete_phase_requested']
    identity = digest(live)
    checks['single_real_admission_identity'] = batch['admission_sha256'] == identity and all(r['admission_sha256'] == identity for r in trials)
    archive = ROOT/f'review/admission_history/admission_{identity}.json'
    checks['admission_archive_exact_bytes'] = archive.is_file() and archive.read_bytes() == live.read_bytes()
    saved_admissions = sorted((ROOT/'review').glob('independent_admission_check_*.json'))
    checks['one_actual_admission_signing'] = len(saved_admissions) == 1 and digest(saved_admissions[0]) == identity
    checks['same_admission_source_map'] = batch['source_sha256'] == admission['source_sha256']
    checks['frozen_sources_still_current'] = all(digest(ROOT/k) == v for k, v in admission['source_sha256'].items())
    checks['frozen_plan_current'] = digest(ROOT/'experiment_plan.json') == batch['experiment_plan_sha256'] == admission['evidence']['experiment_plan']['sha256'] and plan['status'] == 'frozen'
    loaded = {}
    for row in trials:
        d = Path(row['trace_path']).parent
        m, result = read(d/'metadata.json'), read(d/'result.json')
        with np.load(d/'trace.npz', allow_pickle=False) as zz:
            z = {k: zz[k] for k in zz.files}
        embedded = json.loads(str(z['metadata']))
        obs = [json.loads(x) for x in z['observation_json']]
        dbg = [json.loads(x) for x in z['debug_json']]
        key = f"{m['shape']}/{m['method']}"
        ck = {}
        ck['trace_hash_and_result_identity'] = digest(d/'trace.npz') == m['trace_sha256'] == row['trace_sha256'] and result == row
        ck['embedded_metadata_matches'] = {k:v for k,v in m.items() if k != 'trace_sha256'} == embedded
        ck['admission_source_identity'] = m['admission_sha256'] == identity and m['source_sha256'] == batch['source_sha256']
        active = z['controller_active']; hold = (z['time'] >= 7.5-1e-9) & (z['time'] < 12.-1e-9)
        release = z['time'] >= 12.-1e-9
        ck['real_handoff_epoch'] = not active[:1500].any() and active[1500:].all() and np.array_equal(z['local_domain_active'], hold)
        ck['nonzero_real_actions'] = np.max(np.abs(z['requested_action'][hold])) > 1e-6 and np.max(np.abs(z['executed_action'][hold])) > 1e-6
        ck['actual_alpha_delta_matches'] = close(z['executed_action'], z['alpha']-z['alpha_before'])
        ck['request_equals_actual_holding'] = close(z['requested_action'][hold], z['executed_action'][hold])
        ck['public_rate_bounds'] = np.max(abs(z['executed_action'][hold])) <= .8*.005+1e-9
        ck['local_domain_obeyed'] = np.all(z['alpha'][hold] >= z['local_domain_lower'][hold]-1e-9) and np.all(z['alpha'][hold] <= z['local_domain_upper'][hold]+1e-9)
        ck['held_predictions_finite'] = np.isfinite(z['predicted_normal'][hold]).all() and np.isfinite(z['prediction_at'][hold]).all()
        ck['release_predictions_unavailable'] = np.isnan(z['predicted_normal'][release]).all() and np.isnan(z['prediction_at'][release]).all() and all(v.get('prediction_available') is False and v.get('release_domain_override') is True for v in np.asarray(dbg, dtype=object)[release])
        ck['release_actions_only_unload'] = np.max(z['requested_action'][release]) <= 1e-9
        ck['no_false_infeasibility_certificate'] = all(not v.get('model_infeasibility_certified', False) for v in dbg)
        ck['no_hidden_actuator_clip'] = m['max_hidden_ctrl_clip'] == 0.
        recomputed = metrics(z, m)
        ck['raw_clocks_and_coverage'] = recomputed['record_complete']
        fields = ('A_S','first_stable_after_handoff','whole_task_first_stable','mean_force','finger_tv_rate','raw_finger_tv_rate','finger_sd_mean','total_sd','release_latency','stable_hold_fraction')
        ck['independent_endpoint_recalculation'] = all(close(recomputed[k], row[k], 1e-8) for k in fields)
        sf = ('local_excess_integral','total_excess_integral','finger_peak','force_protection_pass')
        ck['independent_raw_safety_recalculation'] = all(close(recomputed[k], row['safety'][k], 1e-8) for k in sf) and recomputed['stages'] == row['safety']['stages']
        ck['release_recalculation'] = recomputed['release_success'] == row['release_success']
        # All six nominal records satisfy stronger uninterrupted support; this
        # independently suffices for their advertised task_success, not a
        # replacement rule for future perturbation failures.
        ck['six_nominal_successes_supported_by_truth'] = row['task_success'] and recomputed['support_continuous_during_hold'] and recomputed['no_warning'] and recomputed['record_complete'] and recomputed['release_success'] and m['handoff_passed']
        rng_review = audit_rng(z, m, obs)
        ck['actual_saved_random_packets_reconstructed'] = rng_review['passed']
        boundary = np.any((z['alpha']-z['local_domain_lower'] <= 1e-6) | (z['local_domain_upper']-z['alpha'] <= 1e-6), axis=1)
        ck['boundary_occupancy_recalculated'] = close(float(boundary[hold].mean()), row['local_domain_boundary_fraction'])
        checks.update({key+'/'+k: bool(v) for k, v in ck.items()})
        runs.append(dict(key=key, checks=ck, random_packets=rng_review, recomputed=recomputed,
            peak_requested_action=float(abs(z['requested_action'][hold]).max()),
            peak_actual_action=float(abs(z['executed_action'][hold]).max()),
            local_domain_boundary_fraction=float(boundary[hold].mean()),
            trace_sha256=digest(d/'trace.npz'), metadata_sha256=digest(d/'metadata.json')))
        loaded[(m['shape'], m['method'])] = (z, m, obs, rng_review)
        print(key, all(ck.values()), flush=True)
    for shape in ('capsule','box','sphere'):
        a, am, ao, ar = loaded[(shape, 'ratio')]
        b, bm, bo, br = loaded[(shape, 'unified')]
        before = a['time'] <= 7.5+1e-9
        fields = ('qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','raw_patches_json','environment_json','observed_normal','observed_auxiliary_wrench')
        equal_fields = {k: bool(np.array_equal(a[k][before], b[k][before], equal_nan=True)) if a[k].dtype.kind not in 'US' else bool(np.array_equal(a[k][before], b[k][before])) for k in fields}
        ck = dict(preparation_every_saved_truth_equal=all(equal_fields.values()),
            preparation_every_observation_field_equal=ao[:1501] == bo[:1501],
            full_random_sequence_identical=ar['reconstructed_draw_stream_sha256'] == br['reconstructed_draw_stream_sha256'],
            physics_parameters_identical=am['config'] == bm['config'] and am['model_sha256'] == bm['model_sha256'],
            noise_parameters_identical=am['noise'] == bm['noise'],
            public_calibration_identical=am['calibration'] == bm['calibration'],
            actual_policy_trajectories_diverge=bool(np.max(abs(a['qpos'][~before]-b['qpos'][~before])) > 1e-7))
        checks.update({shape+'/paired/'+k:bool(v) for k,v in ck.items()})
        pairs.append(dict(shape=shape, checks=ck, exact_preparation_fields=equal_fields,
            compared_prepare_samples=1501, sensor_draw_stream_sha256=ar['reconstructed_draw_stream_sha256']))
    confirmation = plan['phases']['confirmation']
    ck = dict(frozen_before_any_confirmation=plan['status'] == 'frozen',
        full_16_by_2_by_8_plan=len(confirmation['scenarios']) == 16 and len(confirmation['seeds']) == 8 and confirmation['expected_trials'] == 256,
        no_development_seed_reuse=520101 not in confirmation['seeds'],
        no_endpoint_or_source_change=all(digest(ROOT/k) == v for k,v in plan['source_sha256'].items()),
        same_current_admission=identity == batch['admission_sha256'])
    checks.update({'confirmation_readiness/'+k:bool(v) for k,v in ck.items()})
    result = dict(schema='independent-saved-development-review-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        passed=all(checks.values()), checks=checks, failed=[k for k,v in checks.items() if not v],
        trials_sha256=digest(FOLDER/'trials.json'), batch_manifest_sha256=digest(FOLDER/'manifest.json'),
        admission_sha256=identity, experiment_plan_sha256=digest(ROOT/'experiment_plan.json'),
        source_sha256=admission['source_sha256'], reviewer_script_sha256=digest(__file__),
        runs=runs, paired= pairs, confirmation_readiness=ck,
        limits=['Development nominal smoke is correctness evidence, not superiority or all-scenario success.',
            'Local-domain boundary occupancy and complete release-stage safety peaks remain separate reported outcomes.',
            'Fixed calibrated local model is not experimentally proven for every later size, mass, friction or external-contact variant.',
            'No strategy, simulator, original project module, or production evaluator was imported or run in this review.',
            'No admission was re-signed; only one actual first signing exists, all saved runs and batch share identical bytes/hash.',
            'Complete original 16GB preservation rehash is deferred to final delivery.'])
    out = ROOT / ('review/independent_development_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), failed=result['failed'], output=str(out), sha256=digest(out)),ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

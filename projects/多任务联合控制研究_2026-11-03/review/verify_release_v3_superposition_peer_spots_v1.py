"""Independent prespecified two-branch/four-window arithmetic review; no plant.

This deliberately does not import the author's analyzer or any simulation module.
Future matched zero is an OFFLINE differential oracle, not online free evolution.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/release_local_response_capsule_v3'
AUTHOR = ROOT / 'review/release_local_response_v3_superposition_v1.json'
AUDIT = ROOT / 'review/release_local_response_v3_terminal_audit_v1.json'
OUT = ROOT / 'review/release_v3_superposition_peer_spots_v1.json'
SELECTED = [('phi_150ms', 'mixed_50_50', 12.15),
            ('phi_350ms', 'mixed_100_100', 12.35)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = read(RUN / 'summary.json')
    assert summary['execution_complete'] and summary['completed'] == 14
    assert read(AUDIT)['passed'] is True
    author = read(AUTHOR)
    checks, issues, results, sources = [], [], [], {}

    def check(name, ok, detail=None):
        row = dict(name=name, passed=bool(ok), detail=detail)
        checks.append(row)
        if not ok:
            issues.append(row)

    def close(name, actual, expected, atol=1e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        ok = a.shape == b.shape and np.allclose(a, b, rtol=0, atol=atol,
                                               equal_nan=False)
        detail = None if ok else dict(actual=actual, expected=expected)
        check(name, ok, detail)

    # Independent arithmetic sanity checks, including absence rather than zero.
    synthetic = np.array([[.25, -.5], [.25, -.5]])
    close('known_signed_interaction_RMSE', np.sqrt((synthetic**2).mean(0)),
          [.25, .5])
    check('no_valid_data_returns_null_not_zero',
          (None if not np.array([], bool).any() else 0) is None)

    def load(stage, variant):
        folder = RUN / 'runs' / ('capsule__' + stage + '__' + variant)
        raw, cmd, trace = (folder / 'robot_physical_stream.npz',
                           folder / 'decoder_commands.npz',
                           folder / 'record/trace.npz')
        for p in (raw, cmd, trace):
            sources[str(p.relative_to(ROOT))] = sha(p)
        with np.load(raw, allow_pickle=False) as z:
            pt = z['physical_time_s']
            groups = {
                'robot_position25': z['position25'],
                'robot_velocity25': z['velocity25'],
                'physical_external_finger_pressure5_N':
                    z['external_finger_pressure5_N'],
                'offline_target_palm_other_pressure2_N':
                    np.stack((z['palm_pressure_N'], z['other_hand_pressure_N']),
                             axis=1),
            }
        with np.load(cmd, allow_pickle=False) as z:
            t = z['control_time']
            inputs = np.stack((z['actual_phi_s'],
                               z['actual_command21'][:, 18]), axis=1)
        with np.load(trace, allow_pickle=False) as z:
            public_t = z['time']
            frames = [json.loads(str(v)) for v in z['observation_json']]
        observed = np.asarray([f['normal'] for f in frames], float)
        valid = np.asarray([f['valid'] for f in frames], bool)
        aux = np.asarray([[np.nan if f.get('auxiliary_pressure') is None
                           else f['auxiliary_pressure']] for f in frames])
        aux_valid = np.asarray([[f.get('auxiliary_valid', False) and
                                 f.get('auxiliary_pressure') is not None]
                                for f in frames], bool)
        return dict(physical_clock=pt, physical=groups,
                    control_clock=t, inputs=inputs, public_clock=public_t,
                    public=dict(public_observed_pressure5_N=observed,
                                public_auxiliary_pressure1_N=aux),
                    public_valid=dict(public_observed_pressure5_N=valid,
                                      public_auxiliary_pressure1_N=aux_valid))

    for stage, variant, origin in SELECTED:
        data = {v: load(stage, v) for v in
                ('zero', 'phi_slow', 'lift_slow', variant)}
        base, phi, lift, mixed = (data[v] for v in
                                 ('zero', 'phi_slow', 'lift_slow', variant))
        astage = next(s for s in author['stages'] if s['stage'] == stage)
        reported = next(h for h in astage['holdout_results']
                        if h['variant'] == variant)
        for v, d in data.items():
            for k in ('control_clock', 'physical_clock', 'public_clock'):
                check(stage + '/' + v + '/' + k,
                      np.array_equal(d[k], base[k]))
        k = int(np.argmin(abs(base['control_clock'] - (origin + .02))))
        check(stage + '/formation_clock',
              abs(base['control_clock'][k] - (origin + .02)) < 1e-10)
        train_phi = phi['inputs'][:, 0] - base['inputs'][:, 0]
        train_lift = lift['inputs'][:, 1] - base['inputs'][:, 1]
        mixed_phi = mixed['inputs'][:, 0] - base['inputs'][:, 0]
        mixed_lift = mixed['inputs'][:, 1] - base['inputs'][:, 1]
        a, b = float(mixed_phi[k] / train_phi[k]), float(mixed_lift[k] /
                                                         train_lift[k])
        close(stage + '/actual_input_train_denominators',
              [train_phi[k], train_lift[k]],
              [astage['axis_actual_full_formation_denominators']['phi_s'],
               astage['axis_actual_full_formation_denominators']
                     ['lift_reference_m']])
        close(stage + '/actual_input_mixed_ratios', [a, b],
              [reported['actual_input_amplitude_ratios']['phi'],
               reported['actual_input_amplitude_ratios']['lift']])
        close(stage + '/mixed_actual_amplitudes',
              [mixed_phi[k], mixed_lift[k]],
              [reported['actual_formation_amplitudes']['phi_s'],
               reported['actual_formation_amplitudes']['lift_m']])
        local = (base['control_clock'] >= origin-1e-9) & (
                 base['control_clock'] <= origin+.2+1e-9)
        profile_residual = np.stack((mixed_phi - a*train_phi,
                                     mixed_lift - b*train_lift), axis=1)
        check(stage + '/local_input_proportionality_before_floor',
              np.max(abs(profile_residual[local])) < 1e-12)
        outcome = dict(stage=stage, variant=variant,
                       actual_input_ratios=[a, b], windows={})
        for ms in (20, 40, 100, 200):
            name = 'cumulative_0_' + str(ms) + 'ms'
            compact = {}
            for kind in ('physical', 'public'):
                clock = base[kind + '_clock']
                mask = (clock >= origin-1e-9) & (
                        clock <= origin + ms/1000 + 1e-9)
                for group, observed in mixed[kind].items():
                    # Different evaluation order to avoid copying analyzer.
                    # f_mix -(a f_phi + b f_lift +(1-a-b) f_zero).
                    predicted_absolute = (a*phi[kind][group] +
                                          b*lift[kind][group] +
                                          (1-a-b)*base[kind][group])
                    residual = observed - predicted_absolute
                    active = np.broadcast_to(mask[:, None], observed.shape).copy()
                    active &= np.isfinite(observed) & np.isfinite(predicted_absolute)
                    if kind == 'public':
                        for d in data.values():
                            active &= d['public_valid'][group]
                    stats = dict(samples_per_axis=[], rmse_per_axis=[],
                                 max_abs_per_axis=[], signed_mean_per_axis=[])
                    for j in range(observed.shape[1]):
                        e = residual[active[:, j], j]
                        stats['samples_per_axis'].append(int(len(e)))
                        stats['rmse_per_axis'].append(
                            float(np.linalg.norm(e)/np.sqrt(len(e))) if len(e) else None)
                        stats['max_abs_per_axis'].append(float(np.max(abs(e))) if len(e) else None)
                        stats['signed_mean_per_axis'].append(float(sum(e)/len(e)) if len(e) else None)
                    expected = reported['windows'][name]['outputs'][group]
                    check(stage+'/'+name+'/'+group+'/sample_count',
                          stats['samples_per_axis'] == expected['samples_per_axis']
                          and int(mask.sum()) == expected['expected_clock_samples'])
                    for statistic in ('rmse_per_axis','max_abs_per_axis',
                                      'signed_mean_per_axis'):
                        for j, (av, ev) in enumerate(zip(stats[statistic],
                                                        expected[statistic])):
                            if av is None or ev is None:
                                check(stage+'/'+name+'/'+group+'/'+statistic+'/'+str(j),
                                      av is None and ev is None)
                            else:
                                check(stage+'/'+name+'/'+group+'/'+statistic+'/'+str(j),
                                      abs(av-ev) <= 1e-12)
                    compact[group] = dict(samples=int(mask.sum()),
                                         maximum_abs_error=max(v for v in stats['max_abs_per_axis'] if v is not None),
                                         maximum_axis_RMSE=max(v for v in stats['rmse_per_axis'] if v is not None))
            outcome['windows'][name] = compact
        # A zero residual with no pressure excitation is not pressure validation.
        mask20 = (base['physical_clock'] >= origin-1e-9) & (
                  base['physical_clock'] <= origin+.02+1e-9)
        outcome['pressure_all_zero_all4_first20ms'] = all(
            np.count_nonzero(d['physical']['physical_external_finger_pressure5_N'][mask20]) == 0
            for d in data.values())
        results.append(outcome)

    report = dict(schema='release_v3_prespecified_superposition_peer_spots_v1',
                  passed=not issues, checks=checks, issues=issues,
                  selected_before_result_review=SELECTED,
                  input_sha256=sources,
                  author_analysis_sha256=sha(AUTHOR),
                  full14_independent_raw_audit_sha256=sha(AUDIT),
                  checker_sha256=sha(Path(__file__)), results=results,
                  physics_steps=0, parameters_fitted=False,
                  limitations=['Two mixed branches and four cumulative windows only; the all14 raw audit is a separate artifact.',
                               'All physical25q/25v/5pressure/2offline palm-other and public5pressure/1aux statistics were independently recomputed.',
                               'Matched future zero is offline oracle, never online free evolution.',
                               'No numerical model-admission threshold was registered; no safety or model-admission conclusion.'])
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),passed=report['passed'],
                          checks=len(checks),issues=len(issues))))


if __name__ == '__main__':
    main()

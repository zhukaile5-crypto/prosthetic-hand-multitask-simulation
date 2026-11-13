"""Saved nine-arm execution review. Never optimize or advance physics.

All-nine terminal barrier precedes raw/result access. The three v20 arms must
reproduce their complete original V20 traces (declared wall-clock leaves only
excluded). New arms retain pre12 complete observations and37 physical columns.
Certificates are evaluated at saved points and actual emitted commands; failed
lower trials are not certificates. Author of new task math, not an independent
proof of that math; independent component/integration receipts remain separate.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import argparse, gzip, hashlib, importlib.util, json, sys, traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; FOUNDATION = ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(FOUNDATION)); sys.dont_write_bytecode = True
PLAN = ROOT/'review/v21_small_plan_v1.json'
ARCHITECTURES = dict(v20='v20_contact_conditioned_whole_hand_exit', prefix='v21_prefix_only_v20_contact_exit', v21='v21_stable_handoff_exit')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def source_path(k): return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
def matrix(value, rows, n):
    """Only an explicit zero-row JSON list restores its declared second axis."""
    if rows == 0 and type(value) is list and len(value) == 0: return np.empty((0, n))
    a = np.asarray(value, float)
    if a.shape != (rows, n) or not np.isfinite(a).all(): raise ValueError('declared finite matrix shape mismatch/unknown')
    return a
def near(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.all(abs(a-b) <= 256*np.finfo(float).eps*np.maximum(1., np.maximum(abs(a), abs(b))))
def stable_rows(G, h):
    """No norm overflow may silently turn a finite hard row into zero."""
    mag = np.max(abs(G), axis=1, initial=0.); nonzero = mag > 0
    if np.any(h[~nonzero] < 0): raise ValueError('impossible constant hard inequality')
    scaled = G[nonzero]/mag[nonzero, None]; length = np.linalg.norm(scaled, axis=1)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        return scaled/length[:, None], (h[nonzero]/mag[nonzero])/length


def expected_tasks(q, variant, rates):
    """Rebuild original task expressions, without calling either optimizer."""
    t = q['contact_tasks']; d = q['hierarchy']['diagnostics']; p = np.asarray(t['pressure_P_N'], float); P = len(p)
    if P == 0:
        if not (t['available'] and t['diagnostics']['normal_task_available'] is False and
                all(type(t[k]) is list and len(t[k]) == 0 for k in ('normal_jacobian_Px25', 'Npx21', 'y_p')) and t['diagnostics']['patches'] == []):
            raise ValueError('knownP0 requires all original explicit empty geometry declarations')
    J = matrix(t['normal_jacobian_Px25'], P, 25); B = np.asarray(t['proposal_map25x21'], float)
    N = J@B; u = np.asarray(q['current_reference21']); units = rates*.005
    lo = np.asarray(q['actual_reference_lower21'])-u; hi = np.asarray(q['actual_reference_upper21'])-u
    vn = J@np.asarray(t['robot_velocity25']); debt = u-np.asarray(t['encoder_reference21']); an = N@debt
    s = np.sum(abs(N)*units[None, :], axis=1); s_eff = np.where(s == 0, 1., s)
    w = p/p.max() if P else np.empty(0)
    if P: w = w/w.sum()
    M = e = ms = None
    if t.get('motion_matrix_Mx21') is not None:
        M = np.asarray(t['motion_matrix_Mx21'], float); e = np.asarray(t['motion_error_M_m_s'], float)
        ms = np.sum(abs(M)*units[None, :], axis=1); ms = np.where(ms == 0, 1., ms)
    result = {}; extra = dict(P=P, N=N, J=J, B=B, units=units, lower=lo/units, upper=hi/units, motion=M, e=e, motion_scale=ms)
    if variant == 'prefix':
        b = -.005*(np.maximum(0., -np.sum(np.where(N >= 0, N*lo[None, :], N*hi[None, :]), axis=1))/.005+np.maximum(vn, 0.)+np.maximum(an/.02, 0.))
        if P: result['pressure_share_current_contact_exit'] = (np.sqrt(w)[:, None]*(N*units[None, :])/s_eff[:, None], np.sqrt(w)*b/s_eff)
        if M is not None: result['public_external_projection_low_motion'] = (M*units[None, :]/ms[:, None], -.005*e/ms)
        result['canonical_reference_intent_and_minimal_wrist_action'] = (np.eye(21), np.asarray(q['reference_target_delta21'])/units)
        extra.update(n=21, normal_target=b, actual_normal_matrix=N)
    else:
        n = 21+P; b = -.005*(np.maximum(vn, 0.)+np.maximum(an/.02, 0.))
        if P:
            A = np.zeros((P, n)); A[:, 21:] = np.diag(np.sqrt(w)); result['normal_inward_correction_slack'] = (A, np.zeros(P))
        work = -w@(N*units[None, :]) if P else np.zeros(21)
        boxmax = float(np.sum(np.where(work >= 0, work*extra['upper'], work*extra['lower'])))
        scale = max(float(np.max(abs(work), initial=0.)), abs(boxmax)); scale = 1. if scale == 0 else scale
        A = np.zeros((1, n)); A[0, :21] = work/scale; result['scalar_pressure_share_normal_withdrawal'] = (A, np.array([boxmax/scale]))
        if M is not None:
            A = np.zeros((len(M), n)); A[:, :21] = M*units[None, :]/ms[:, None]
            result['public_external_projection_low_motion'] = (A, -.005*e/ms)
        target = q.get('path_reference_target_delta21')
        target = np.asarray(q['reference_target_delta21'] if target is None else target)
        A = np.zeros((21, n)); A[:, :21] = np.eye(21); result['canonical_or_explicit_path_reference'] = (A, target/units)
        extra.update(n=n, normal_target=b, slack_units=s_eff, debt=debt, vn=vn, an=an)
    return result, extra


def audit_prefix(q, variant, frame, actual, rates):
    errors = []; counts = Counter()
    def ck(name, ok):
        if not bool(ok): errors.append(name)
    phase = q['phase']; exit_phase = phase in ('TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT')
    counts['phase:'+phase] += 1
    if not exit_phase:
        counts['direct_search_capture'] += 1
        ck('direct_not_solver_failure', q['hierarchy'] is None and q['fallback'] is None and not q['geometry_proposal_accepted'])
        return errors, counts
    tasks = q.get('contact_tasks'); h = q.get('hierarchy'); accepted = q['geometry_proposal_accepted']
    P = len(tasks['pressure_P_N']) if tasks is not None and tasks.get('available') else None
    role = 'positive_pressure' if P else 'known_P0' if P == 0 else 'geometry_unknown'; counts[role+':attempted'] += 1
    completed = 0 if h is None else h['completed_prefix']; fully = False if h is None else h['fully_converged']
    counts[role+(':accepted_full' if fully else ':accepted_prefix' if completed else ':zero_prefix_fallback')] += 1
    ck('accepted_requires_nonempty_prefix', not accepted or completed > 0)
    if h is None:
        ck('unknown_geometry_no_certificate', not accepted); return errors, counts
    certs = h['certified_prefix']; ck('prefix_count', completed == len(certs))
    if h['failed_layer'] is not None: counts['failed_layer:'+h['failed_layer']['name']+':'+h['failed_layer']['status']] += 1
    if not completed:
        ck('zero_prefix_not_accepted', not accepted); return errors, counts
    expected, extra = expected_tasks(q, variant, rates); n = extra['n']; d = h['diagnostics']
    raw = np.asarray(d.get('extended_normalized_x', h['x']), float)
    ck('correct_external_internal_coordinates', len(raw) == n and len(h['x']) == 21 and near(raw[:21], h['x']))
    current = np.asarray(q['current_reference21']); proposed = np.asarray(q['proposed_reference21']); units = extra['units']
    emitted = raw.copy(); emitted[:21] = (proposed-current)/units
    actual_x = raw.copy(); actual_x[:21] = (actual-current)/units
    roundoff = np.zeros(n); roundoff[:21] = 128*np.finfo(float).eps*np.maximum(1., np.maximum(abs(current), abs(proposed)))/units
    score = q.get('emitted_hierarchy_score'); ck('accepted_has_score', not accepted or score is not None and score['available'])
    if variant == 'v21':
        ck('raw_normal_correction_no_capacity', near(extra['normal_target'], d['normal_correction_target_P_m']))
        xA = np.asarray(d['certified_L1_extended_anchor']); eta = np.asarray(d['certified_slack_eta_P'])
        ck('slack_frozen_not_normal_displacement', near(raw[21:], eta) and near(d['certified_slack_P_m'], eta*extra['slack_units']))
        if P:
            ck('L1_witness_is_certified_actual_point', near(xA, certs[0]['x']) and certs[0]['name'] == 'normal_inward_correction_slack')
        else: ck('P0_uses_genuine_box_anchor', near(xA[:21], np.minimum(extra['upper'], np.maximum(extra['lower'], 0))))
        M = extra['motion']; e = extra['e']; ms = extra['motion_scale']
        normal_A = np.column_stack((extra['N']*units[None, :]/extra['slack_units'][:, None], -np.eye(P)))
        G = normal_A; upper = extra['normal_target']/extra['slack_units']
        if M is not None:
            mat = M*units[None, :]/ms[:, None]; offset = .005*e/ms; rA = mat@xA[:21]+offset
            budgetA = np.zeros((2*len(M), n)); budgetA[:len(M), :21] = mat; budgetA[len(M):, :21] = -mat
            budgeth = np.r_[abs(rA)-offset, abs(rA)+offset]
            ck('witnessed_motion_budget_not_zero_or_final_anchor', near(rA, d['motion_task']['witness_normalized_residual_M']) and near(abs(rA), d['motion_task']['normalized_absolute_budget_M']) and near(budgetA, d['motion_task']['budget_inequality_matrix']) and near(budgeth, d['motion_task']['budget_inequality_upper']))
            G = np.vstack((G, budgetA)); upper = np.r_[upper, budgeth]
        ck('entire_original_second_hard_domain', near(G, matrix(d['second_stage_hard_matrix'], len(upper), n)) and near(upper, d['second_stage_hard_upper']))
        GN, hn = stable_rows(G, upper); max_tol = max(c['KKT']['coordinate_tolerance'] for c in certs)
        ck('new_budget_feasible_at_actual_L1_witness', np.all(GN@xA-hn <= max_tol))
        ck('emitted_slack_normal_and_motion_hard_budget', np.all(GN@emitted-hn <= max_tol+abs(GN)@roundoff))
        ck('final_actual_slack_normal_and_motion_hard_budget', np.all(GN@actual_x-hn <= max_tol+abs(GN)@roundoff))
    else:
        ck('prefix_only_original_V20_capacity_target', near(extra['normal_target'], d['normal_target_b_P_m']))
    for index, cert in enumerate(certs):
        name = cert['name']; ck('cert_global_index', cert['index'] == index)
        if name not in expected: errors.append('unknown_certified_task_name'); continue
        A, b = expected[name]; C = np.asarray(cert['matrix']); target = np.asarray(cert['target']); point = np.asarray(cert['x']); kkt = cert['KKT']; tol = kkt['coordinate_tolerance']
        ck('original_task_coefficients_'+name, near(A, C) and near(b, target))
        attained = C@point; residual = attained-target; gradient = 2*C.T@residual
        ck('certificate_original_attainment_'+name, near(attained, cert['attained']) and near(residual, cert['residual']) and near(residual@residual, cert['squared_loss']))
        ck('certificate_original_gradient_'+name, near(gradient, kkt['gradient']) and kkt['certified'] and all(kkt['checks'].values()))
        working = matrix(kkt['original_loop_working_matrix'], len(kkt['original_loop_multipliers']), n); dual = np.asarray(kkt['original_loop_multipliers'])
        ck('saved_original_KKT_stationarity_'+name, np.max(abs(gradient+(working.T@dual if len(dual) else 0.)), initial=0.) <= kkt['gradient_tolerance'])
        domain = cert['hard_domain']; hg = np.asarray(domain['normalized_inequality_upper']); hf = np.asarray(domain['fixed_equality_target'])
        G = matrix(domain['normalized_inequality_matrix'], len(hg), n); E = matrix(domain['fixed_equality_matrix'], len(hf), n)
        ck('original_certified_hard_primal_'+name, np.all(G@point-hg <= tol) and np.all(abs(E@point-hf) <= tol))
        for previous in cert['previous_frozen']:
            F = np.asarray(previous['matrix']); ck('previous_frozen_at_saved_point_'+name, np.all(abs(F@point-np.asarray(previous['attained'])) <= tol))
        original = np.asarray(cert['attained']); returned = C@raw; now = C@emitted
        allowance = abs(C)@roundoff+128*np.finfo(float).eps*np.maximum(1., abs(C)@abs(emitted))
        ck('only_certified_prefix_at_returned_and_emitted_'+name, np.all(abs(returned-original) <= tol) and np.all(abs(now-returned) <= allowance))
        ck('emitted_original_hard_domain_'+name, np.all(G@emitted-hg <= tol+abs(G)@roundoff) and np.all(abs(E@emitted-hf) <= tol+abs(E)@roundoff))
        ck('final_actual_original_hard_domain_'+name, np.all(G@actual_x-hg <= tol+abs(G)@roundoff) and np.all(abs(E@actual_x-hf) <= tol+abs(E)@roundoff))
        if accepted:
            es = score['levels'][index]; er = now-target
            ck('recorded_emitted_score_'+name, near(now, es['attained']) and near(er, es['residual']) and near(er@er, es['squared_loss']) and es['certified_attainment_and_emission_roundoff_respected'])
    if accepted:
        ck('native_full_value_available', q['native_current'] is not None and q['native_proposed'] is not None and q['native_current']['status'] == q['native_proposed']['status'] == 'KNOWN_ROBOT_ACTUATION_ALGEBRA')
        ck('native_current_public_port', near(q['native_current']['commanded_target21'], frame['robot_actuator_target21']))
        for name in ('actuator_length21', 'actuator_velocity21', 'transmission21x25'): ck('native_same_fixed_snapshot_'+name, near(q['native_current'][name], q['native_proposed'][name]))
        counts['accepted_full' if fully else 'accepted_prefix'] += 1
    else:
        counts['proposal_not_committed'] += 1
        ck('fallback_explicit_legal_anchor', q['fallback'] == 'legal_current_q_recovery_anchor_without_exit_certificate')
    return errors, counts


def execute(plan, out, md):
    aux = module('v21_aux_barrier', ROOT/'review/read_v21_small_auxiliary_v1.py')
    markers = {s['name']: read(ROOT/s['output']/'execution_complete.json') if (ROOT/s['output']/'execution_complete.json').exists() else None for s in plan['stages']}
    barrier = aux.terminal_barrier(plan, markers)
    if not barrier['ready_to_read_metrics']: print(json.dumps(dict(status='PENDING', barrier=barrier))); return 2
    if out.exists() or md.exists(): raise FileExistsError('Fresh execution audit paths required')
    helper = module('v21_old_exec_helpers', ROOT/'review/read_v20_small_execution_v1.py')
    old = module('v21_complete_bridge_helper', ROOT/'review/read_v19_small_execution_v1.py')
    prefix = module('v21_prefix37_helper', ROOT/'review/v16_small_prefix_storage_independent_v1.py')
    timing = module('v21_declared_timing_helper', ROOT/'review/v18_passive_trace_comparison_v3.py')
    import research_imports, mujoco, hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    checks, issues, rows, inputs, forbidden = [], [], [], {}, []
    def ck(name, ok, detail=None):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))
        if not ok: issues.append(dict(name=name, detail=detail))
    def h(p):
        p = str(Path(p).resolve())
        if p not in inputs: inputs[p] = sha(p)
        return inputs[p]
    source = plan['source_sha256']; parent = read(ROOT/'review/v20_small_plan_v1.json')['source_sha256']; no = read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256']
    ck('frozen170 parent160 NoMotion156', len(source) == 170 and len(parent) == 160 and all(source.get(k) == v for k, v in parent.items()) and len(no) == 156 and all(h(source_path(k)) == v for k, v in no.items()) and all(h(source_path(k)) == v for k, v in source.items()))
    ck('all42 pinned environment binaries', len(plan['environment']['native_binaries_sha256']) == 42 and all(h(k) == v for k, v in plan['environment']['native_binaries_sha256'].items()))
    def deny(name):
        def stop(*a, **kw): forbidden.append(name); raise AssertionError('Forbidden '+name)
        return stop
    original_cache = {}
    with ExitStack() as guards:
        for name in ('MjData', 'mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_forwardSkip', 'mj_kinematics', 'mj_comPos', 'mj_collision', 'mj_contactForce'):
            if hasattr(mujoco, name): guards.enter_context(patch.object(mujoco, name, deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant, '__init__', deny('HandPlant'))); guards.enter_context(patch.object(ReferenceActor21, 'apply', deny('actor.apply')))
        for stage in plan['stages']:
            tag = stage['name']; variant = stage['variant']; ass = stage['assignment']; folder = ROOT/stage['output']; row = dict(stage=tag, variant=variant, assignment=ass, status='unassessed'); rows.append(row)
            try:
                mark = markers[tag]; reg = read(ROOT/stage['registration']); manifest = read(folder/'manifest.json'); snap = read(folder/'source_snapshot_manifest.json'); trials = read(folder/'trials.json'); ledger = read(folder/'assignment_ledger.json')
                ck(tag+'/assignment source and completion hashes', reg['assignment'] == ass and reg['source_sha256'] == source and reg['environment'] == plan['environment'] and manifest['research_source_sha256'] == source and len(trials) == len(ledger) == 1 and ledger[0]['status'] == 'evaluated' and all(mark[k+'_sha256'] == h(folder/(k+'.json')) for k in ('trials', 'assignment_ledger', 'plan', 'manifest')))
                ck(tag+'/complete170 snapshot and actualenv', set(snap['files']) == set(source) and all(h(folder/snap['files'][k]['snapshot_path']) == v for k, v in source.items()) and manifest['source_snapshot_manifest_sha256'] == h(folder/'source_snapshot_manifest.json') and read(folder/'environment.json')['pinned_environment'] == plan['environment'])
                context = aux.context(ass); run = folder/'runs'/f'{context[0]}__ratio_effort__{context[2]}'
                ck(tag+'/assigned destination', Path(ledger[0]['job']['destination']).resolve() == run.resolve())
                meta = read(run/'metadata.json'); result = read(run/'result.json'); aux.metadata_context(meta, ass)
                ck(tag+'/result and actualtrace identity', result == trials[0] and meta['research_source_sha256'] == source and result['trace_sha256'] == meta['trace_sha256'] == h(run/'trace.npz'))
                with np.load(run/'trace.npz', allow_pickle=False) as z: a = {k: z[k] for k in z.files}
                time = a['time']; active = a['controller_active'].astype(bool)
                ck(tag+'/complete16 raw clocks', meta['completed'] and np.array_equal(time, np.arange(3201)*.005) and len(a['substep_time']) == 32000 and np.isfinite(a['qpos']).all() and np.isfinite(a['ctrl']).all())
                original = ROOT/stage['original_v20_run']; ck(tag+'/predeclared originalV20 trace', h(original/'trace.npz') == stage['original_v20_trace_sha256'])
                if str(original) not in original_cache:
                    with np.load(original/'trace.npz', allow_pickle=False) as z: original_cache[str(original)] = {k: z[k] for k in z.files}
                b = original_cache[str(original)]; comparisons = prefix.compare(a, b, time, b['time'], variant != 'v20')
                ck(tag+'/complete37 payload original input', all(c['exact_payload_equal'] for c in comparisons), [c for c in comparisons if not c['exact_payload_equal']])
                mask = time < 12-1e-9 if variant != 'v20' else np.ones(len(time), bool)
                ck(tag+'/full public observation payload no stripping', all(str(x).encode('utf-8') == str(y).encode('utf-8') for x, y in zip(a['observation_json'][mask], b['observation_json'][mask])))
                bridge = old.passive_complete_bridge(a, b, timing) if variant == 'v20' else None
                if bridge is not None: ck(tag+'/all full16 columns bridge', bridge['passed'], [c for c in bridge['columns'] if not c['passed']])
                model, xml = hand_plant._model(FoundationConfig(**meta['config'])); actor = ReferenceActor21(model)
                encoded = a['executed_reference21'].copy(); encoded[:, 18] += actor.lift_bias_m
                ck(tag+'/actual21 actor once and incoming public port', xml == meta['model_sha256'] and helper.bits(encoded, a['ctrl'][:, actor.indices21]) and helper.bits(a['robot_actuator_target21'], a['ctrl_before'][:, actor.indices21]))
                obs = [json.loads(str(x)) for x in a['observation_json']]; debug = [json.loads(str(x)) for x in a['debug_json']]; errors = []
                for i in np.flatnonzero(active):
                    f = obs[i]; ex = a['executed_reference21'][i]; current = np.asarray(f['robot_reference21']); gen = int(a['mode_generation'][i])
                    if np.any(ex < np.asarray(f['reference_lower21'])-1e-12) or np.any(ex > np.asarray(f['reference_upper21'])+1e-12) or np.max(abs(ex-current)-reference_rates(debug[i]['mode'])*.005) > 1e-12: errors.append((int(i), 'actual_mechanical_mode_rate'))
                    if gen != f['mode_generation'] or gen != debug[i]['mode_generation'] or abs(f['robot_control_time_s']-time[i]) > 1e-9 or not helper.fbits(ex-current, a['reference_increment21'][i]): errors.append((int(i), 'clock_generation_delta'))
                ck(tag+'/all actual5ms21 permissions', not errors, errors[:12])
                dm = read(run/'release_decision_manifest.json'); ck(tag+'/decision hash', h(run/dm['file']) == dm['sha256'])
                expected_indices = np.flatnonzero([active[i] and debug[i].get('architecture') == ARCHITECTURES[variant] for i in range(len(time))])
                counts = Counter(); errors = []; indices = []
                with gzip.open(run/dm['file'], 'rt', encoding='utf-8') as f:
                    for k, line in enumerate(f):
                        q = json.loads(line); i = round(q['time_s']/.005); indices.append(i); frame = obs[i]; ex = a['executed_reference21'][i]; current = np.asarray(q['current_reference21']); g = q['governor']; receipt = q.get('execution_receipt'); rates = reference_rates('release')
                        if q['decision_id'] != k or q['time_s'] != time[i] or q['mode_generation'] != int(a['mode_generation'][i]) or not helper.fbits(current, frame['robot_reference21']): errors.append((k, 'record_current_snapshot'))
                        if q['public_process']['holding_force_tasks_active'] or q['placement_certified'] or q['future_motion_or_pressure_prediction_available']: errors.append((k, 'holding_or_falseclaim'))
                        L = np.maximum(np.asarray(frame['reference_lower21']), current-rates*.005); U = np.minimum(np.asarray(frame['reference_upper21']), current+rates*.005)
                        if not helper.fbits(L, q['original_mechanical_rate_lower21']) or not helper.fbits(U, q['original_mechanical_rate_upper21']): errors.append((k, 'original_reference_authority'))
                        qlift, vlift = float(frame['robot_position25'][24]), float(frame['robot_velocity25'][24]); interval, reachable = helper.independent_lift_interval(g, qlift, vlift, L[18], U[18], q['phase'] in helper.EXIT_PHASES)
                        if not helper.fbits(interval, q['governor_task_domain']['task_reference_interval_m']) or q['governor_task_domain']['budget_band_reachable_this_tick'] != reachable or not interval[0] <= ex[18] <= interval[1]: errors.append((k, 'TASK_lift_domain'))
                        if receipt is None: errors.append((k, 'missing_actual_receipt')); continue
                        debt = qlift-float(ex[18]); budget = -g['downward_recovery_reference_budget_m'] <= debt <= g['recovery_reference_budget_m']; changed = not helper.fbits(q['proposed_reference21'], ex)
                        if not helper.fbits(receipt['actual_reference21'], ex) or receipt['same_snapshot_q_lift_m'] != qlift or receipt['same_snapshot_reference_debt_m'] != debt or receipt['current_q_budget_respected'] != budget or receipt['actual_task_interval_respected'] != (interval[0] <= ex[18] <= interval[1]) or receipt['execution_changed'] != changed: errors.append((k, 'receipt_actual_flags'))
                        counts['execution_changed'] += changed; counts['stale_plan_replaced'] += receipt['stale_plan_replaced']; counts['exact_budget_false'] += not budget
                        es, cs = helper.independent_tasks(q, frame, ex, rates) if variant == 'v20' else audit_prefix(q, variant, frame, ex, rates)
                        errors.extend((k, e) for e in es); counts.update(cs)
                ck(tag+'/all actual decisions retained', len(indices) == dm['records'] and np.array_equal(indices, expected_indices))
                ck(tag+'/original tasks certified prefix eta budget and emission', not errors, errors[:20])
                ck(tag+'/prediction remains unknown', not np.any(a['prediction_available']) and np.isnan(a['predicted_normal']).all())
                row.update(status='audited', run=str(run.relative_to(ROOT)), input_sha256={n: h(run/n) for n in ('metadata.json', 'result.json', 'trace.npz')}, decisions=len(indices), counts=dict(counts),
                    original_task_success=result['task_success'], original_full_motion=result['research_metrics']['full_motion_mission_success'], original_guard=result['research_metrics']['safety']['guard_pass'],
                    prefix37=comparisons, complete_originalV20_bridge=bridge, native_scope='Saved fullF067 original same-q/v records; model/optimizer not replayed')
            except Exception as exc:
                row.update(status='reader_error', error=repr(exc)); issues.append(dict(name=tag+'/reader_error', error=repr(exc), traceback=traceback.format_exc()))
    ck('all9 assigned assessed no failed selection', len(rows) == 9 and all(r['status'] == 'audited' for r in rows))
    ck('source170 unchanged after read', all(sha(source_path(k)) == v for k, v in source.items())); ck('zero forbidden calls', not forbidden, forbidden)
    result = dict(schema='v21_small_execution_v1', passed=not issues, checks=checks, issues=issues, rows=rows, planned=9, source_sha256=source, environment=plan['environment'], plan_sha256=sha(PLAN), input_sha256=inputs,
        reader_sha256=sha(__file__), helper_sha256={n: sha(ROOT/n) for n in ('review/read_v20_small_execution_v1.py', 'review/read_v21_small_auxiliary_v1.py', 'review/v16_small_prefix_storage_independent_v1.py', 'review/v18_passive_trace_comparison_v3.py')},
        scope=dict(physics_steps=0, optimizer_replays=0, plant_constructions=0, data_constructions=0, saved_certificate_gradient_and_original_domain_checked=True),
        limitations=['Task math author execution review, not independent proof of authored math.', 'Actual motion/force/placement not guaranteed by reference geometry/certified TASK budgets.', 'Partial prefix is not full hierarchy convergence; P0 is not positive-contact optimization.', 'Original failures/NULL unchanged; three contexts share two physical seeds.'])
    out.parent.mkdir(parents=True, exist_ok=True); md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(helper.plain(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    md.write_text('# V21九条三臂执行审查\n\n'+('PASS' if result['passed'] else 'FAIL')+f'；{sum(c["passed"] for c in checks)}/{len(checks)}项，问题{len(issues)}。\n\n'
        '全部九条终态后检查170来源/42环境、3个V20完整16s复现桥、6条新臂pre12完整37列与公共JSON、实际21权限/回执。前缀证书、内部η、原硬域/运动预算与实际发出参考分开核算；只检查已认证层，不把失败低层当全部最优。阶段/正压/P0/完整与部分前缀及回退分母保留。\n\n没有运行优化器、物理积分、forward或接触查询；任务失败和NULL不被执行PASS改写。数学组件作者交叉执行审查，不替代独立数学门。详见同名JSON。\n', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), issues=issues[:8], out=str(out), sha256=sha(out)), ensure_ascii=False)); return 0 if result['passed'] else 1


def self_test():
    checks = []
    G, h = stable_rows(np.array([[1e200]]), np.array([0.])); checks.append(dict(category='finite_large_hardrow_not_lost', passed=bool((G@np.array([.5])-h)[0] > 0)))
    for name, value, rows, n in (('unknown_matrix_rejected', None, 0, 21), ('wrong_shape_not_padded', [0.], 0, 21)):
        try: matrix(value, rows, n); ok = False
        except (ValueError, TypeError): ok = True
        checks.append(dict(category=name, passed=ok))
    checks.append(dict(category='declared_zero_row_shape', passed=matrix([], 0, 21).shape == (0, 21)))
    a = module('v21_exec_aux_software_gate', ROOT/'review/read_v21_small_auxiliary_v1.py'); auxgate = a.self_test()
    for c in auxgate['checks'][:4]: checks.append(dict(category='barrier/'+c['category'], passed=c['passed']))
    return dict(schema='v21_execution_reader_protection_v1', passed=all(c['passed'] for c in checks), checks=checks, source_sha256={'review/read_v21_small_execution_v1.py': sha(__file__)}, new_outcomes_read=False, physics_steps=0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); p.add_argument('--prepare', action='store_true'); p.add_argument('--self-test', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'review/v21_small_execution_v1.json'); p.add_argument('--md', type=Path, default=ROOT/'review/V21九条三臂执行独审_v1.md'); a = p.parse_args()
    if a.self_test:
        if a.out.exists(): raise FileExistsError(a.out)
        d = self_test(); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(dict(passed=d['passed'], checks=len(d['checks']), out=str(a.out), sha256=sha(a.out), new_outcomes_read=False))); return 0 if d['passed'] else 1
    if not PLAN.exists(): print(json.dumps(dict(status='PENDING_PLAN', branch_results_read=False))); return 2
    plan = read(PLAN)
    if not a.execute: print(json.dumps(dict(prepared=True, planned=plan['planned'], sources=len(plan['source_sha256']), branch_results_read=False, reader_sha256=sha(__file__)))); return 0
    return execute(plan, a.out, a.md)


if __name__ == '__main__': raise SystemExit(main())

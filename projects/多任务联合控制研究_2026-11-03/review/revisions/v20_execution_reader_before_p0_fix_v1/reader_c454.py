"""Terminal-gated V20 four-run saved-execution review. No simulation/solver.

Reviewer authored geometry/adapter, not the runtime or task solver. Source,
complete physical prefixes, emitted commands and recorded arithmetic are
audited here; software PASS is not a policy/placement performance claim.
Every NPZ column is loaded once. Only fixed representative fullF067 robot
algebra is re-evaluated, using public q/v/commands; no FK/forward/step/contact.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import argparse, gzip, hashlib, importlib.util, json, sys, traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT.parent / '五指策略接入_2026-11-03'
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(FOUNDATION))
sys.dont_write_bytecode = True
PLAN = ROOT / 'review/v20_small_plan_v1.json'
EXIT_PHASES = ('TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()
def source_path(k): return FOUNDATION / k[11:] if k.startswith('foundation/') else ROOT / k
def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x
def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
def fbits(a, b): return bits(np.asarray(a, float), np.asarray(b, float))
def near(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): return False
    return bool(np.all(abs(a-b) <= 256*np.finfo(float).eps*np.maximum(1., np.maximum(abs(a), abs(b)))))
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def terminal_gate(plan, reader=read):
    states = []
    for stage in plan['stages']:
        p = ROOT / stage['output'] / 'execution_complete.json'
        try: done = reader(p)
        except FileNotFoundError: done = None
        states.append(dict(stage=stage['name'], execution_complete=done is not None and done.get('execution_complete') is True))
    return dict(ready=len(states) == 4 and all(x['execution_complete'] for x in states), stages=states,
                no_partial_performance_read=True)


def independent_lift_interval(g, q, v, L, U, exit_phase):
    up = g['recovery_reference_budget_m']; down = g['downward_recovery_reference_budget_m']
    a, b = max(L, q-up), min(U, q+down); reachable = a <= b
    if not reachable: a = b = min(U, max(L, q))
    elif exit_phase:
        tau = g['servo_velocity_scale_s']
        eps = 32*np.finfo(float).eps*max(1., abs(v), up/tau, down/tau)
        brake = min(b, max(a, q))
        if v < -up/tau-eps: a = brake
        elif v > down/tau+eps: b = brake
    elif g['braking_intent_active']:
        brake = min(b, max(a, q))
        if v < 0: a = brake
        elif v > 0: b = brake
        else: a = b = brake
    return np.array([a, b]), reachable


def independent_tasks(record, frame, actual, rates):
    """All cheap recorded formula checks; never solve the optimization."""
    errors = []; count = Counter()
    def ck(name, condition):
        if not bool(condition): errors.append(name)
    current = np.asarray(record['current_reference21'], float)
    lower = np.asarray(record['actual_reference_lower21'], float)
    upper = np.asarray(record['actual_reference_upper21'], float)
    preferred = np.asarray(record['preferred_reference21'], float)
    proposed = np.asarray(record['proposed_reference21'], float)
    count['phase:'+record['phase']] += 1
    ck('original_current_public_reference', fbits(current, frame['robot_reference21']))
    ck('actual_proposal_domain', np.isfinite(proposed).all() and np.all(proposed >= lower) and np.all(proposed <= upper))
    anchor = np.minimum(upper, np.maximum(lower, current))
    ck('actual_feasible_anchor', fbits(anchor, record['feasible_recovery_anchor21']))
    ck('raw_target_not_retargeted', fbits(preferred-current, record['reference_target_delta21']))
    ck('no_force_or_placed_claim', record['future_motion_or_pressure_prediction_available'] is False and record['placement_certified'] is False)
    phase = record['phase']; exit_phase = phase in EXIT_PHASES
    accepted = record['geometry_proposal_accepted']
    count['geometry_accepted'] += accepted
    count['search_capture_direct'] += not exit_phase
    count['exit_fallback'] += exit_phase and not accepted
    count['phase:'+phase+':accepted'] += accepted
    if not exit_phase:
        ck('search_capture_no_contact_solver', record['contact_tasks'] is None and record['hierarchy'] is None and record['fallback'] is None)
        ck('search_capture_canonical_legal_action', fbits(proposed, np.minimum(upper, np.maximum(lower, preferred))))
        return errors, count
    tasks = record['contact_tasks']; hierarchy = record['hierarchy']
    if tasks is None or not tasks['available']:
        count['geometry_unknown'] += 1
        ck('unknown_geometry_no_solver_or_native_invention', hierarchy is None and not accepted and record['native_proposed'] is None)
    if hierarchy is not None:
        count['hierarchy:'+hierarchy['status']] += 1
        d = hierarchy['diagnostics']
        if d.get('actual_increment_available'):
            count['hierarchy_linear_available'] += 1
            p = np.asarray(tasks['pressure_P_N'], float); P = len(p)
            J = np.asarray(tasks['normal_jacobian_Px25'], float).reshape(P, 25)
            B = np.asarray(tasks['proposal_map25x21'], float)
            N = J @ B; units = rates*.005
            lo, hi = lower-current, upper-current
            debt = current-np.asarray(tasks['encoder_reference21'], float)
            velocity = np.asarray(frame['robot_velocity25'], float)
            vn = J @ velocity; an = N @ debt
            minimum = np.sum(np.where(N >= 0, N*lo[None, :], N*hi[None, :]), axis=1)
            capacity = np.maximum(0., -minimum)/.005
            target = -.005*(capacity+np.maximum(vn, 0.)+np.maximum(an/.02, 0.))
            ru = np.sum(abs(N)*units[None, :], axis=1); effective = np.where(ru == 0, 1., ru)
            if P:
                shares = p/p.max(); shares /= shares.sum()
            else: shares = np.empty(0)
            ck('normal_raw_coefficients_no_truth', near(N, d['normal_matrix_N_Px21']) and near(debt, d['reference_debt21']) and near(vn, d['normal_current_velocity_P_m_s']) and near(an, d['normal_reference_debt_P_m']))
            ck('directional_box_capacity_not_symmetric_rate_claim', near(minimum, d['normal_box_min_increment_P_m']) and near(capacity, d['normal_outward_box_capacity_P_m_s']))
            ck('one_sided_current_velocity_debt_target', near(target, d['normal_target_b_P_m']))
            ck('normal_shares_units_and_no_rho_gate', near(shares, d['pressure_share_P']) and near(ru, d['normal_reference_row_units_P_m']) and d['no_reaction_share_gate'] and d['no_hard_contact_noninward_constraint'])
            matrices = []
            if P: matrices.append(('pressure_share_current_contact_exit', np.sqrt(shares)[:, None]*(N*units[None, :])/effective[:, None], np.sqrt(shares)*target/effective))
            motion = d['motion_task']
            if motion['available']:
                M = np.asarray(tasks['motion_matrix_Mx21'], float); e = np.asarray(tasks['motion_error_M_m_s'], float)
                mu = np.sum(abs(M)*units[None, :], axis=1); meff = np.where(mu == 0, 1., mu)
                matrices.append(('public_external_projection_low_motion', M*units[None, :]/meff[:, None], -.005*e/meff))
                ck('motion_same_current_construction_and_no_dt_double', tasks['motion_time_s'] == record['time_s'] and near(M, motion['matrix_Mx21']) and near(e, motion['error_M_m_s']))
                count['motion_task_available'] += 1
            else:
                ck('unknown_motion_three_fields_none', all(tasks.get(k) is None for k in ('motion_matrix_Mx21', 'motion_error_M_m_s', 'motion_time_s')))
            matrices.append(('canonical_reference_intent_and_minimal_wrist_action', np.eye(21), (preferred-current)/units))
            saved = d['task_matrices_and_targets']; x = np.asarray(hierarchy['x'], float)
            ck('all_original_layers_present', len(matrices) == len(saved))
            for (name, matrix, b), s in zip(matrices, saved):
                matrix = np.asarray(matrix); b = np.asarray(b)
                attained = matrix @ x; residual = attained-b
                ck('layer_coefficients_'+name, s['name'] == name and near(matrix, s['matrix']) and near(b, s['target']))
                ck('layer_attainment_'+name, near(attained, s['attained']) and near(residual, s['residual']) and near(residual@residual, s['squared_loss']))
            ck('actual_box_original_current_not_reanchored', near(d['actual_step_lower21'], lo) and near(d['actual_step_upper21'], hi) and near(d['normalized_anchor21'], np.minimum(hi/units, np.maximum(lo/units, 0))))
            if accepted:
                emitted = record['emitted_hierarchy_score']; xe = (proposed-current)/units
                ck('emitted_score_available', emitted is not None and emitted['available'] and near(emitted['normalized_emitted_delta21'], xe))
                for s, es in zip(saved, emitted['levels']):
                    m = np.asarray(s['matrix']); b = np.asarray(s['target']); attained = m @ xe; residual = attained-b
                    roundoff = 128*np.finfo(float).eps*np.maximum(1., np.maximum(abs(current), abs(proposed)))/units
                    tolerance = abs(m)@roundoff+128*np.finfo(float).eps*np.maximum(1., abs(m)@abs(xe))
                    ck('emitted_actual_hierarchy_'+s['name'], near(attained, es['attained']) and near(residual, es['residual']) and near(residual@residual, es['squared_loss']) and np.all(abs(attained-np.asarray(s['attained'])) <= tolerance) and es['original_attainment_preserved_within_reconstruction_roundoff'])
    if accepted:
        ck('accepted_has_full_native_value', record['native_current'] is not None and record['native_proposed'] is not None and record['fallback'] is None)
        for key in ('actuator_length21', 'actuator_velocity21', 'transmission21x25'):
            ck('native_same_fixed_robot_state_'+key, fbits(record['native_current'][key], record['native_proposed'][key]))
        ck('native_value_success_no_contact_claim', record['native_current']['status'] == record['native_proposed']['status'] == 'KNOWN_ROBOT_ACTUATION_ALGEBRA' and not record['native_proposed']['contact_force_prediction_available'])
    else:
        ck('unaccepted_legal_anchor_and_explicit_fallback', record['fallback'] == 'legal_current_q_recovery_anchor_without_exit_certificate' and fbits(proposed, anchor))
    return errors, count


def execute(plan, out, md):
    terminal = terminal_gate(plan)
    if not terminal['ready']:
        print(json.dumps(dict(status='PENDING', terminal_gate=terminal))); return 2
    if out.exists() or md.exists(): raise FileExistsError('Fresh review outputs required')
    old = load_module('v19_execution_review_helpers', ROOT/'review/read_v19_small_execution_v1.py')
    prefix = load_module('v20_prefix37_reader', ROOT/'review/v16_small_prefix_storage_independent_v1.py')
    timing = load_module('v20_timing_leaf_contract', ROOT/'review/v18_passive_trace_comparison_v3.py')
    if timing.TIMING_COLUMNS != ('control_ms', 'actor_callback_ms', 'full_control_loop_ms', 'observation_ms', 'physics_advance_ms') or timing.DEBUG_TIMING_KEYS != ('qp', 'native_inverse', 'whole_decision'):
        raise ValueError('Timing whitelist changed')
    import research_imports, mujoco, hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    from v11.robot_servo_effort import RobotServoEffort
    checks, issues, rows, inputs, forbidden = [], [], [], {}, []
    def ck(name, ok, detail=None):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))
        if not ok: issues.append(dict(name=name, detail=detail))
    def h(p):
        p = str(Path(p).resolve())
        if p not in inputs: inputs[p] = sha(p)
        return inputs[p]
    parent = read(ROOT/'review/v19_small_plan_v1.json')['source_sha256']
    ablation = read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256']
    ck('parent152 preserved and exactly eight new sources', len(parent) == 152 and all(plan['source_sha256'].get(k) == v for k, v in parent.items()) and len(plan['source_sha256']) == 160 and len(set(plan['source_sha256'])-set(parent)) == 8)
    ck('NoMotion156 unchanged', len(ablation) == 156 and all(h(source_path(k)) == v for k, v in ablation.items()))
    ck('actual160 and42 environment binaries pinned', all(h(source_path(k)) == v for k, v in plan['source_sha256'].items()) and len(plan['environment']['native_binaries_sha256']) == 42 and all(h(k) == v for k, v in plan['environment']['native_binaries_sha256'].items()))
    def deny(name):
        def blocked(*a, **kw): forbidden.append(name); raise AssertionError('Forbidden '+name)
        return blocked
    with ExitStack() as guards:
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_forwardSkip', 'mj_kinematics', 'mj_comPos', 'mj_collision', 'mj_contactForce'):
            if hasattr(mujoco, name): guards.enter_context(patch.object(mujoco, name, deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant, '__init__', deny('HandPlant')))
        guards.enter_context(patch.object(ReferenceActor21, 'apply', deny('actor.apply')))
        for stage in plan['stages']:
            tag = stage['name']; folder = ROOT/stage['output']; assignment = stage['assignment']
            row = dict(stage=tag, assignment=assignment, status='unassessed'); rows.append(row)
            try:
                done = read(folder/'execution_complete.json'); reg = read(ROOT/stage['registration']); manifest = read(folder/'manifest.json'); snap = read(folder/'source_snapshot_manifest.json')
                trials = read(folder/'trials.json'); jobs = read(folder/'assignment_ledger.json'); failures = read(folder/'runtime_failures.json')
                ck(tag+'/all assignment retained and source unchanged', done['execution_complete'] and done['sources_unchanged'] and done['planned'] == done['trials'] == stage['planned'] == len(trials) == len(jobs) == 1 and not failures and reg['assignment'] == assignment and reg['source_sha256'] == plan['source_sha256'] and reg['environment'] == plan['environment'])
                ck(tag+'/actual full snapshot and environment', manifest['research_source_sha256'] == plan['source_sha256'] and set(snap['files']) == set(plan['source_sha256']) and all(h(folder/snap['files'][k]['snapshot_path']) == v for k, v in plan['source_sha256'].items()) and read(folder/'environment.json')['pinned_environment'] == plan['environment'])
                ck(tag+'/terminal manifest hashes', all(done[k+'_sha256'] == h(folder/(k+'.json')) for k in ('trials', 'assignment_ledger', 'plan', 'manifest')) and manifest['source_snapshot_manifest_sha256'] == h(folder/'source_snapshot_manifest.json'))
                scene, method, seed = assignment['scenarios'][0], assignment['methods'][0], assignment['seeds'][0]
                run = folder/'runs'/f'{scene}__{method}__{seed}'; meta = read(run/'metadata.json'); result = read(run/'result.json')
                ck(tag+'/original result and source identity', result == trials[0] and result['scenario'] == scene and result['method'] == method and result['seed'] == seed and meta['research_source_sha256'] == plan['source_sha256'] and result['trace_sha256'] == meta['trace_sha256'] == h(run/'trace.npz'))
                with np.load(run/'trace.npz', allow_pickle=False) as z: a = {k: z[k] for k in z.files}
                t = a['time']; active = a['controller_active'].astype(bool)
                ck(tag+'/complete3201 frames32000 intervals', meta['completed'] and np.array_equal(t, np.arange(3201)*.005) and len(a['substep_time']) == 32000 and np.isfinite(a['qpos']).all() and np.isfinite(a['ctrl']).all())
                comparator = ROOT/stage['original_same_condition']
                ck(tag+'/predeclared original input retained', h(comparator/'trace.npz') == stage['original_trace_sha256'])
                blind = assignment['receiver_policy'] == 'blind'
                with np.load(comparator/'trace.npz', allow_pickle=False) as z: b = {k: z[k] for k in (z.files if blind else prefix.COLS)}
                comparison = prefix.compare(a, b, t, b['time'], not blind)
                ck(tag+'/strict37 physical and full public sensor prefix', all(v['exact_payload_equal'] for v in comparison), [v for v in comparison if not v['exact_payload_equal']])
                bridge = old.passive_complete_bridge(a, b, timing) if blind else None
                if blind: ck(tag+'/complete16 passive all columns', bridge['passed'], [v for v in bridge['columns'] if not v['passed']])
                model, xmlsha = hand_plant._model(FoundationConfig(**meta['config'])); actor = ReferenceActor21(model)
                encoded = a['executed_reference21'].copy(); encoded[:, 18] += actor.lift_bias_m
                ck(tag+'/actual21 encoder and public incoming21', xmlsha == meta['model_sha256'] and bits(encoded, a['ctrl'][:, actor.indices21]) and bits(a['robot_actuator_target21'], a['ctrl_before'][:, actor.indices21]))
                observations = [json.loads(str(s)) for s in a['observation_json']]; debug = [json.loads(str(s)) for s in a['debug_json']]
                actual_errors = []
                for i in np.flatnonzero(active):
                    f = observations[i]; before = np.asarray(f['robot_reference21']); ex = a['executed_reference21'][i]; mode = debug[i]['mode']; gen = int(a['mode_generation'][i])
                    if np.any(ex < np.asarray(f['reference_lower21'])-1e-12) or np.any(ex > np.asarray(f['reference_upper21'])+1e-12) or np.max(abs(ex-before)-reference_rates(mode)*.005) > 1e-12: actual_errors.append((int(i), 'mechanical_mode_rate'))
                    if gen != f['mode_generation'] or gen != debug[i]['mode_generation'] or abs(f['robot_control_time_s']-t[i]) > 1e-9 or not fbits(ex-before, a['reference_increment21'][i]): actual_errors.append((int(i), 'generation_clock_delta'))
                ck(tag+'/all active5ms21 permissions clocks rates', not actual_errors, actual_errors[:12])
                dm = read(run/'release_decision_manifest.json'); ck(tag+'/decision gzip identity', h(run/dm['file']) == dm['sha256'])
                with gzip.open(run/dm['file'], 'rt', encoding='utf-8') as f: records = [json.loads(s) for s in f]
                indices = np.asarray([round(q['time_s']/.005) for q in records], int)
                expected = np.flatnonzero([active[i] and debug[i].get('architecture') == 'v20_contact_conditioned_whole_hand_exit' for i in range(len(t))])
                ck(tag+'/all coordinated5ms decisions retained', len(records) == dm['records'] and np.array_equal(indices, expected) and (blind or len(records) == 801))
                errors = []; counts = Counter(); debts = []
                for k, record in enumerate(records):
                    i = indices[k]; f = observations[i]; ex = a['executed_reference21'][i]; current = np.asarray(record['current_reference21']); g = record['governor']; receipt = record.get('execution_receipt')
                    if record['decision_id'] != k or record['time_s'] != t[i] or record['mode_generation'] != int(a['mode_generation'][i]) or record['process_version'] != record['public_process']['version'] or not fbits(current, f['robot_reference21']): errors.append((k, 'record_public_snapshot'))
                    if record['public_process']['holding_force_tasks_active']: errors.append((k, 'holding_force_tasks_in_release'))
                    rates = reference_rates('release'); L = np.maximum(np.asarray(f['reference_lower21']), current-rates*.005); U = np.minimum(np.asarray(f['reference_upper21']), current+rates*.005)
                    if not fbits(L, record['original_mechanical_rate_lower21']) or not fbits(U, record['original_mechanical_rate_upper21']): errors.append((k, 'original_actual_authority'))
                    qlift, vlift = float(f['robot_position25'][24]), float(f['robot_velocity25'][24])
                    task, reachable = independent_lift_interval(g, qlift, vlift, L[18], U[18], record['phase'] in EXIT_PHASES)
                    if not fbits(task, record['governor_task_domain']['task_reference_interval_m']) or record['governor_task_domain']['budget_band_reachable_this_tick'] != reachable or not task[0] <= ex[18] <= task[1]: errors.append((k, 'contact_exit_lift_task_domain'))
                    if receipt is None: errors.append((k, 'missing_final_receipt')); continue
                    debt = qlift-float(ex[18]); debts.append(debt); budget = -g['downward_recovery_reference_budget_m'] <= debt <= g['recovery_reference_budget_m']
                    changed = not fbits(record['proposed_reference21'], ex)
                    if not fbits(receipt['actual_reference21'], ex) or receipt['same_snapshot_q_lift_m'] != qlift or receipt['same_snapshot_reference_debt_m'] != debt or receipt['current_q_budget_respected'] != budget or receipt['actual_task_interval_respected'] != (task[0] <= ex[18] <= task[1]) or receipt['execution_changed'] != changed: errors.append((k, 'receipt_actual_debt_flags'))
                    counts['task_budget_outside'] += not budget; counts['execution_changed'] += changed; counts['stale_plan_replaced'] += receipt['stale_plan_replaced']
                    local_errors, local_counts = independent_tasks(record, f, ex, rates)
                    errors.extend((k, s) for s in local_errors); counts.update(local_counts)
                ck(tag+'/all current tasks layers anchors and actual receipts', not errors, errors[:20])
                representatives = []
                accepted = [k for k, r in enumerate(records) if r['geometry_proposal_accepted']]
                if accepted:
                    representatives = sorted({accepted[0], accepted[-1]} | set([next((k for k in accepted if any(records[k]['native_current']['diagnostics']['force_clipped21'])), accepted[0])]))
                oracle_rows = []
                if representatives:
                    servo = RobotServoEffort(model)
                    for k in representatives:
                        rec = records[k]; f = observations[indices[k]]
                        for name, ref in (('native_current', rec['current_reference21']), ('native_proposed', rec['proposed_reference21'])):
                            value = servo.evaluate(np.asarray(f['robot_position25']), np.asarray(f['robot_velocity25']), actor.encode(__import__('v13_a1.codec', fromlist=['Reference21']).Reference21(ref, rec['time_s'])).values, time_s=rec['time_s'])
                            logged = rec[name]
                            ok = value.status == logged['status'] and all(fbits(getattr(value, key), logged[key]) for key in ('actuator_length21', 'actuator_velocity21', 'predicted_effort21', 'unbounded_effort21'))
                            oracle_rows.append(dict(decision_index=k, role=name, exact_native_values=ok))
                ck(tag+'/predeclared representative canonicalF067 values', all(r['exact_native_values'] for r in oracle_rows), oracle_rows)
                ck(tag+'/prediction unknown NaN retained', not np.any(a['prediction_available']) and np.isnan(a['predicted_normal']).all())
                row.update(status='audited', trace_sha256=h(run/'trace.npz'), metadata_sha256=h(run/'metadata.json'), result_sha256=h(run/'result.json'),
                    decisions=len(records), counts=dict(counts), actual_active5ms=int(active.sum()), prefix37=comparison, complete_passive_bridge=bridge,
                    original_task_success=result['task_success'], original_full_motion=result['research_metrics']['full_motion_mission_success'],
                    original_force_guard=result['research_metrics']['safety']['guard_pass'], final_debt_abs_max_m=None if not debts else float(np.max(abs(debts))),
                    full_native_recheck_scope='first,last,and_first_forceclipped accepted decision per candidate; current and proposed, public q/v only', native_representatives=oracle_rows)
            except Exception as exc:
                row.update(status='reader_error', error=repr(exc)); issues.append(dict(name=tag+'/reader_exception', error=repr(exc), traceback=traceback.format_exc()))
    ck('all4 assigned assessed including physical failures', len(rows) == 4 and all(r['status'] == 'audited' for r in rows))
    ck('source160 unchanged after read', all(sha(source_path(k)) == v for k, v in plan['source_sha256'].items()))
    ck('no forbidden physical calls', not forbidden, forbidden)
    report = dict(schema='v20_small_execution_independent_v1', passed=not issues, checks=checks, issues=issues, rows=rows,
        all_assigned_checked=len(rows) == 4 and all(r['status'] == 'audited' for r in rows), assigned_count=4,
        terminal_gate=terminal, plan_sha256=sha(PLAN), source_sha256=plan['source_sha256'], input_sha256=inputs, reader_sha256=sha(__file__),
        review_helpers_sha256={p: sha(ROOT/p) for p in ('review/read_v19_small_execution_v1.py', 'review/v16_small_prefix_storage_independent_v1.py', 'review/v18_passive_trace_comparison_v3.py')},
        scope=dict(physics_steps=0, solver_replays=0, all_coordinated5ms_records_checked=True, model_compiles=4, plant_constructions=0,
                   native_representatives='first,last,first-forceclipped accepted; original native transmission algebra only'),
        reviewer_role='geometry/adapter author; independent saved runtime/task-expression/actual-execution review, not independent proof of authored geometry',
        limitations=['Software PASS is not clearance, placement, force safety or superiority.', 'No outcome/NULL/failed denominator is changed.',
                    'Mixed-time old-plane reference tasks do not predict future robot/object velocity or force.', 'Control timing is not end-to-end integration/real-time latency.'])
    out.parent.mkdir(parents=True, exist_ok=True); md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    md.write_text('# V20四条执行证据审查\n\n'+('PASS' if report['passed'] else 'FAIL')+f'；{sum(x["passed"] for x in checks)}/{len(checks)}项，问题{len(issues)}。\n\n'
        '全分配4条、冻结来源160与环境42、Full152/NoMotion156保持、原完整被动桥及候选pre12完整37列和公共JSON、逐5ms实际21权限/代际/回执/恢复预算、全部接触任务原矩阵目标与实际发出增量重算，未知与回退按阶段保留。每候选首/末/首force-clipped accepted决策独立核原F067完整值；没有运行优化器、物理积分、forward、世界FK或接触查询。\n\n'
        '审者为geometry/adapter作者，因此不是该组件独立数学证明。软件验收不改变原任务失败/NULL，也不构成清离、放置或整体性能胜利。细节与限制见同名JSON。\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=len(checks), issues=issues[:8], out=str(out), sha256=sha(out)), ensure_ascii=False))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute', action='store_true'); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'review/v20_small_execution_v1.json')
    parser.add_argument('--md', type=Path, default=ROOT/'review/V20四条执行独立读数_v1.md'); args = parser.parse_args()
    if not PLAN.exists(): print(json.dumps(dict(status='PENDING_PLAN', branch_results_read=False))); return 2
    plan = read(PLAN)
    if not args.execute:
        print(json.dumps(dict(prepared=True, plan_sha256=sha(PLAN), stages=len(plan['stages']), source_count=len(plan['source_sha256']),
            branch_results_read=False, physics_steps=0, reader_sha256=sha(__file__)), ensure_ascii=False)); return 0
    return execute(plan, args.out, args.md)


if __name__ == '__main__': raise SystemExit(main())

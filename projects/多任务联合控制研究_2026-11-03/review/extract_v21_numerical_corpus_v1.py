"""Fixed-selection saved numerical corpus. No optimizer/model/physics call.

Classify every original positive-pressure decision. Source task construction is
captured with saved-result stubs, not solved; independent coefficient formulas
and the original saved certificates provide numerical cross-checks. A missing
failed-trial iterate is never filled from the last returned certified point.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from collections import Counter
from unittest.mock import patch
import argparse, ast, gzip, hashlib, importlib.util, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
POLICY = ROOT/'review/v21_numerical_corpus_selection_v1.json'
PLAN = ROOT/'review/v21_complete16_plan_v1.json'
AUDIT = ROOT/'review/v21_complete16_execution_v1.json'
LAYERS = ('normal_inward_correction_slack', 'scalar_pressure_share_normal_withdrawal',
          'public_external_projection_low_motion', 'canonical_or_explicit_path_reference')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b): return hashlib.sha256(b).hexdigest()
def plain(x):
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return plain(x.item())
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x
def array(value, n):
    a = np.asarray(value, float)
    if n == 0 and type(value) is list and not value: return np.empty((0, 0))
    return a
def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()
def check(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    maximum = None if a.shape != b.shape else float(np.max(abs(a-b), initial=0.))
    near = bool(ok and np.all(abs(a-b) <= 256*np.finfo(float).eps*np.maximum(1., np.maximum(abs(a), abs(b)))))
    return dict(passed=near, bit_equal=bool(ok and same(a, b)), max_abs_difference=maximum)
def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def captures_from_source(q, producer, kernel, rates):
    """Run only task algebra, with both real optimizer entry points disabled."""
    h = q['hierarchy']; d = h['diagnostics']; t = q['contact_tasks']; captures = []; forbidden = []
    first_cert = [c for c in h['certified_prefix'] if c.get('task_stage') == 'normal_slack']
    second_cert = [c for c in h['certified_prefix'] if c.get('task_stage') == 'motion_budget_exit']
    def optimizer_forbidden(*a, **kw): forbidden.append('real_optimizer'); raise RuntimeError('Optimizer execution forbidden')
    def saved_result_stub(**kwargs):
        stage_index = len(captures)
        capture = {k: deepcopy(v) for k, v in kwargs.items() if k != 'levels'}
        capture['levels'] = [dict(name=l.name, matrix=np.asarray(l.matrix).copy(), target=np.asarray(l.target).copy()) for l in kwargs['levels']]
        captures.append(capture)
        if stage_index == 0:
            diag = d['first_stage_diagnostics']; failed = h['failed_layer'] if not first_cert else None
            return SimpleNamespace(available=bool(first_cert), x=np.asarray(first_cert[0]['x'] if first_cert else diag['original_feasible_anchor']),
                certified_prefix=tuple(deepcopy(first_cert)), fully_converged=bool(first_cert), failed_layer=deepcopy(failed),
                levels=tuple(deepcopy(h['levels'][:1])), diagnostics=deepcopy(diag), status='CAPTURED_ORIGINAL_FIRST_STAGE_RESULT')
        if stage_index != 1: raise RuntimeError('Unexpected task-builder optimizer call count')
        certs = []
        for c in second_cert:
            c = deepcopy(c); c['index'] = c['solver_local_index']; certs.append(c)
        failed = deepcopy(h['failed_layer'])
        if failed is not None: failed['index'] -= len(first_cert)
        return SimpleNamespace(available=bool(certs), x=np.asarray(d['extended_normalized_x']), certified_prefix=tuple(certs),
            fully_converged=h['fully_converged'], failed_layer=failed, levels=tuple(deepcopy(h['levels'][1:])),
            diagnostics=deepcopy(d['second_stage_diagnostics']), status='CAPTURED_ORIGINAL_SECOND_STAGE_RESULT')
    records = h['levels']; atol = records[0]['numerical_atol']; rtol = records[0]['numerical_rtol']
    kw = dict(time_s=q['time_s'], dt_s=.005, current_reference21=np.asarray(q['current_reference21']),
        encoder_reference21=np.asarray(t['encoder_reference21']),
        step_lower21=np.asarray(q['actual_reference_lower21'])-np.asarray(q['current_reference21']),
        step_upper21=np.asarray(q['actual_reference_upper21'])-np.asarray(q['current_reference21']),
        reference_rates21=rates, normal_jacobian_Px25=np.asarray(t['normal_jacobian_Px25']),
        proposal_map25x21=np.asarray(t['proposal_map25x21']), robot_velocity25=np.asarray(t['robot_velocity25']),
        pressure_P_N=np.asarray(t['pressure_P_N']), reference_target_delta21=np.asarray(q['reference_target_delta21']),
        motion_matrix_Mx21=None if t['motion_matrix_Mx21'] is None else np.asarray(t['motion_matrix_Mx21']),
        motion_error_M_m_s=None if t['motion_error_M_m_s'] is None else np.asarray(t['motion_error_M_m_s']),
        motion_time_s=t['motion_time_s'], path_reference_target_delta21=None if q['path_reference_target_delta21'] is None else np.asarray(q['path_reference_target_delta21']),
        max_iterations=d.get('numerical_budget', {}).get('max_iterations', 400), atol=atol, rtol=rtol)
    with patch.object(kernel, 'solve_hierarchy', optimizer_forbidden), patch.object(producer, 'solve_hierarchy', saved_result_stub):
        result = producer.solve_contact_exit(**kw)
    expected_calls = 2 if first_cert else 1
    if len(captures) != expected_calls or forbidden: raise ValueError('Task construction did not follow saved stages without optimization')
    return captures, dict(real_optimizer_calls=0, saved_result_stub_calls=len(captures), constructed_status=result.status,
                          original_atol=atol, original_rtol=rtol, failed_trial_reconstructed=False)


def build_sample(selected, helper, producer, kernel, rates):
    q = selected['original_record']; h = q['hierarchy']; d = h['diagnostics']; failed = h['failed_layer']
    captured, construction = captures_from_source(q, producer, kernel, rates)
    independent, extra = helper.expected_tasks(q, 'v21', rates)
    cert_by_name = {c['name']: c for c in h['certified_prefix']}; consistency = []; problems = []
    for stage_index, capture in enumerate(captured):
        stage_name = 'normal_slack' if stage_index == 0 else 'motion_budget_exit'
        sd = d['first_stage_diagnostics' if stage_index == 0 else 'second_stage_diagnostics']; hd = sd['hard_domain']
        lower = np.asarray(hd['lower']); upper = np.asarray(hd['upper']); n = len(lower)
        Gin = np.asarray(capture['inequality_matrix']); hin = np.asarray(capture['inequality_upper'])
        free = np.flatnonzero(lower != upper); fixed = np.flatnonzero(lower == upper)
        expanded_G = np.vstack((Gin, np.eye(n)[free], -np.eye(n)[free])); expanded_h = np.r_[hin, upper[free], -lower[free]]
        for name, a, b in (
            ('lower', capture['lower'], lower), ('upper', capture['upper'], upper), ('original_stage_initial', capture['initial'], sd['original_feasible_anchor']),
            ('expanded_original_G', expanded_G, hd['original_inequality_matrix_with_box']), ('expanded_original_h', expanded_h, hd['original_inequality_upper_with_box']),
            ('fixed_E', np.eye(n)[fixed], np.asarray(hd['fixed_equality_matrix']).reshape(len(fixed), n)), ('fixed_f', lower[fixed], hd['fixed_equality_target'])):
            consistency.append(dict(stage=stage_name, component=name, **check(a, b)))
        previous = []
        for local_index, level in enumerate(capture['levels']):
            name = level['name']; A = level['matrix']; b = level['target']; ia, ib = independent[name]
            consistency.append(dict(stage=stage_name, component=name+'/independent_A', **check(A, ia)))
            consistency.append(dict(stage=stage_name, component=name+'/independent_b', **check(b, ib)))
            cert = cert_by_name.get(name)
            if cert is not None:
                consistency.append(dict(stage=stage_name, component=name+'/saved_A', **check(A, cert['matrix'])))
                consistency.append(dict(stage=stage_name, component=name+'/saved_b', **check(b, cert['target'])))
                # Saved certified matrices retain original bits where available.
                A = np.asarray(cert['matrix']); b = np.asarray(cert['target']); reconstructed = False
            else: reconstructed = True
            fixedE = np.asarray(hd['fixed_equality_matrix']).reshape(len(hd['fixed_equality_target']), n)
            E = np.vstack([fixedE]+[np.asarray(c['matrix']) for c in previous])
            f = np.concatenate([np.asarray(hd['fixed_equality_target'])]+[np.asarray(c['attained']) for c in previous])
            initial = np.asarray(previous[-1]['x'] if previous else sd['original_feasible_anchor'])
            is_failed = failed is not None and failed['name'] == name
            problem = dict(layer_name=name, stage_name=stage_name, solver_local_index=local_index,
                original_global_index=local_index if stage_index == 0 else 1+local_index,
                is_original_failed_layer=is_failed, A=A, b=b,
                G=hd['original_inequality_matrix_with_box'], h=hd['original_inequality_upper_with_box'], E=E, f=f,
                initial_x=initial, stage_initial_x=sd['original_feasible_anchor'], lower=lower, upper=upper,
                input_inequality_matrix=Gin, input_inequality_upper=hin,
                previous_frozen=[dict(name=c['name'], matrix=c['matrix'], attained=c['attained']) for c in previous],
                atol=construction['original_atol'], rtol=construction['original_rtol'], max_iterations=capture['max_iterations'],
                variable_count=n, variable_order='[du21/(rates*.005),etaP=rP/sP]', objective='sum((A @ x - b)**2)',
                matrix_target_reconstructed=reconstructed,
                matrix_target_source='captured_frozen_source_task_construction_no_solver' if reconstructed else 'original_saved_certified_matrix',
                failed_trial_x=None, failed_trial_x_status='NOT_SAVED_IN_ORIGINAL_LOG' if is_failed else 'NOT_APPLICABLE',
                original_failed_record=failed if is_failed else None,
                original_certificate=cert, raw_hard_domain=hd,
                initial_raw_inequality_violation=float(max(0., np.max(np.asarray(hd['original_inequality_matrix_with_box'])@initial-np.asarray(hd['original_inequality_upper_with_box']), initial=0.))),
                initial_equality_violation=float(np.max(abs(E@initial-f), initial=0.)))
            if name == 'scalar_pressure_share_normal_withdrawal':
                work = np.asarray(d['scalar_normal_work_row']); cap = d['scalar_normal_work_box_upper_m']; scale = max(float(np.max(abs(work), initial=0.)), abs(cap)); scale = 1. if scale == 0 else scale
                problem.update(work_row=work, work_box_upper=cap, work_common_scale=scale,
                    scalar_LP_equivalence='maximize work_row @ x[:21]; original box upper bounds every feasible scalar, so distance-squared to that upper bound is monotone nonincreasing up to it; eta objective columns are zero')
            problems.append(problem)
            if cert is not None: previous.append(cert)
            if is_failed: break
    selected_name = failed['name'] if failed is not None else h['certified_prefix'][-1]['name']
    chosen = next(p for p in problems if p['layer_name'] == selected_name)
    if not all(c['passed'] for c in consistency): raise ValueError('Source reconstruction inconsistent with saved problem: '+str([c for c in consistency if not c['passed']][:4]))
    return dict(**selected, failed_stage_problem=chosen, stage_problems=problems,
        original_certified_prefix=h['certified_prefix'], original_failed_layer=failed,
        original_returned_normalized_reference21=h['x'],
        diagnostic_problem_role='original_failed_layer' if failed is not None else 'full_success_control_final_certified_layer',
        source_construction_check=construction, numerical_consistency_checks=consistency,
        all_numerical_consistency_passed=True)


def main(out, schema_out):
    if out.exists() or schema_out.exists(): raise FileExistsError('Fresh numerical corpus output required')
    policy = read(POLICY); plan = read(PLAN); audit = read(AUDIT)
    if not audit['passed'] or audit['issues'] or audit['source_sha256'] != plan['source_sha256']: raise ValueError('Passed original all16 execution receipt required')
    source_path = ROOT/'v21_stable_handoff/task_solver.py'; kernel_path = ROOT/'v21_stable_handoff/prefix_hierarchy.py'; helper_path = ROOT/'review/read_v21_small_execution_v1.py'
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in (source_path, kernel_path, ROOT/'v21_stable_handoff/design.py', helper_path)}
    for p in (source_path, kernel_path, ROOT/'v21_stable_handoff/design.py'):
        if pins[p.relative_to(ROOT).as_posix()] != plan['source_sha256'][p.relative_to(ROOT).as_posix()]: raise ValueError('Frozen source changed')
    tree = ast.parse(source_path.read_text(encoding='utf-8')); function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_contact_exit')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'solve_hierarchy']
    if len(calls) != 2: raise ValueError('Expected exactly two explicitly replaced optimizer call sites')
    ast_pin = digest(ast.dump(function, include_attributes=False).encode('utf-8'))
    helper = load('v21_frozen_independent_formulas', helper_path)
    from v21_stable_handoff import task_solver as producer
    from v21_stable_handoff import prefix_hierarchy as kernel
    from v14_coord.design import reference_rates
    rates = reference_rates('release'); classification = []; first_failure = {}; first_success = {}; logs = []; counts = Counter()
    for entry in sorted(plan['roster'], key=lambda r: tuple(r['context_key'])):
        context = tuple(entry['context_key']); run = ROOT/entry['run']; dm_path = run/'release_decision_manifest.json'; dm = read(dm_path); logfile = run/dm['file']; loghash = sha(logfile)
        if loghash != dm['sha256']: raise ValueError('Original log hash changed')
        logs.append(dict(context_key=list(context), path=logfile.relative_to(ROOT).as_posix(), sha256=loghash, manifest_sha256=sha(dm_path)))
        with gzip.open(logfile, 'rt', encoding='utf-8', newline='') as f:
            for line_no, line in enumerate(f, 1):
                q = json.loads(line); t = q.get('contact_tasks'); h = q.get('hierarchy')
                if not (t and t.get('available') and len(t['pressure_P_N']) > 0): continue
                failed = None if h is None else h.get('failed_layer'); fully = h is not None and h.get('fully_converged') is True and q['geometry_proposal_accepted'] is True
                name = None if failed is None else failed['name']; category = 'full_success' if fully else name if name is not None else 'other_unaccepted'
                counts[category] += 1
                origin = dict(path=logfile.relative_to(ROOT).as_posix(), compressed_sha256=loghash, jsonl_line_1based=line_no,
                    raw_utf8_line_sha256=digest(line.encode('utf-8')), decision_id=q['decision_id'], time_s=q['time_s'])
                classification.append(dict(context_key=list(context), source=origin, phase=q['phase'], pressure_rows=len(t['pressure_P_N']),
                    variable_count=21+len(t['pressure_P_N']), classification=category,
                    completed_prefix=0 if h is None else h['completed_prefix'], fully_converged=fully,
                    proposal_committed=q['geometry_proposal_accepted'], original_failed_layer=None if failed is None else dict(index=failed['index'], name=name, status=failed['status'])))
                key = (q['time_s'], line_no, q['decision_id'])
                candidate = dict(context_key=list(context), source=origin, original_record=q,
                    canonical_parsed_record_sha256=digest(json.dumps(q, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')))
                if name is not None:
                    selected_key = (*context, name)
                    if selected_key not in first_failure or key < first_failure[selected_key][0]: first_failure[selected_key] = (key, dict(candidate, selection_kind='first_failed_layer'))
                if fully and (context not in first_success or key < first_success[context][0]): first_success[context] = (key, dict(candidate, selection_kind='first_full_success_control'))
    if len(classification) != 1377 or counts['full_success'] != 686 or [counts[x] for x in LAYERS] != [5, 111, 285, 290] or counts['other_unaccepted']:
        raise ValueError('Original all1377 classified denominator differs: '+str(counts))
    chosen = []
    for context in sorted(tuple(r['context_key']) for r in plan['roster']):
        for name in LAYERS:
            if (*context, name) in first_failure: chosen.append(first_failure[(*context, name)][1])
        if context in first_success: chosen.append(first_success[context][1])
    samples = []
    for index, selected in enumerate(chosen):
        sample = build_sample(selected, helper, producer, kernel, rates)
        sample['sample_id'] = 'v21_numeric_'+str(index).zfill(3); samples.append(sample)
    if any(sha(ROOT/p) != v for p, v in pins.items()): raise ValueError('Bound source/helper changed during extraction')
    context_set = {tuple(r['context_key']) for r in plan['roster']}
    report = dict(schema='v21_numerical_corpus_v1', complete=True, selection_policy_path=POLICY.relative_to(ROOT).as_posix(), selection_policy_sha256=sha(POLICY),
        plan_sha256=sha(PLAN), execution_audit_sha256=sha(AUDIT), source_sha256=plan['source_sha256'], construction_source_sha256=pins,
        solve_contact_exit_AST_sha256=ast_pin, optimizer_call_sites_replaced=2, classification_count=len(classification), classification_counts=dict(counts),
        classifications=classification, sample_count=len(samples), samples=samples,
        selected_failure_count=len(first_failure), selected_full_success_controls=len(first_success), missing_full_success_controls=[list(c) for c in sorted(context_set-set(first_success))],
        original_log_manifests=logs, all_reconstruction_checks_passed=True, reader_sha256=sha(__file__),
        scope=dict(real_optimizer_calls=0, task_construction_only_with_saved_result_stubs=True, model_constructions=0, physics_steps=0, original170_or_results_changed=False),
        limitations=['Failure iterates were not saved; no failed trial x is inferred from returned certified points.',
                    'G/h retain raw original non-fixed box and normal/motion rows; E/f retain fixed coordinates and prior attained rows, including redundancy.',
                    'failed_stage_problem for full-success controls is the final certified-layer problem with is_original_failed_layer=false.',
                    'This is a numerical development corpus, not new performance evidence or a solver proof.'])
    out.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    schema_out.write_text('# V21同问题离线数值诊断语料\n\n'
        f'全1377条有压决策分类；全层成功686，L1/L2/L3/L4失败5/111/285/290。固定首失败选择{len(first_failure)}条，首全层成功对照{len(first_success)}条，总samples={len(samples)}。两物理seed/16已见条件，不是新独立样本。\n\n'
        '`classifications`为全部1377索引。`samples`含每(context,failed-layer)首条失败及同context首全成功；source给压缩文件/hash、原1-based JSONL行、UTF8原行hash、decision/time。`original_record`完整保留；`original_certified_prefix`、`original_failed_layer`和所有stage_problems同存。\n\n'
        '`sample.failed_stage_problem`可直接供oracle：A,b为原LS；G,h为原始非固定box+normal/motion行；E,f为固定坐标+同stage前认证attained行；initial_x为该层原初点，stage_initial_x为原solve调用初点；lower/upper、previous_frozen、atol/rtol与scalar work_row/work_box_upper也存。对全成功control，该键明确is_original_failed_layer=false，指向其末层。可单独使用G/h及E/f；若另行应用lower/upper，将重复box/fixed约束但可行集相同。保留冗余，不静默做rank约简。\n\n'
        '认证层A/b直接取原记录；未存的失败层A/b由冻结原task函数构造、两处优化调用均替换为保存结果stub，不重解。同时对照独立公开公式、原认证矩阵与原stage硬域/initial，并绑定AST和source SHA。失败trial完整x原日志未存，字段保持None，不能用last-certified点伪造。\n\n'
        f'完整JSON：`{out}`，SHA `{sha(out)}`。0真实优化器、0模型、0物理，未改变170源/原结果。\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, classification_count=len(classification), counts=dict(counts), samples=len(samples), failed_samples=len(first_failure), success_controls=len(first_success), out=str(out), sha256=sha(out)), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=ROOT/'review/v21_numerical_corpus_v1.json'); p.add_argument('--md', type=Path, default=ROOT/'review/V21同问题数值语料schema_v1.md'); args = p.parse_args(); main(args.out, args.md)

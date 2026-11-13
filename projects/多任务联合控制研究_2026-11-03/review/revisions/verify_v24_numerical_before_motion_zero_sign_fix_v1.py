"""Independent V24 canonical-witness assembly gate; no physical engine use.

Run only with the final delivered task SHA. Importing this file does not solve
or construct a model.
"""
from pathlib import Path
from dataclasses import fields, is_dataclass, replace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import hashlib
import json
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'review'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def bits(left, right):
    return (np.shape(left) == np.shape(right)
            and np.asarray(left, float).tobytes()
            == np.asarray(right, float).tobytes())


def original_problem_from_certificate(cert, *, atol=1e-9, rtol=1e-8):
    """Reconstruct all original rows, including every genuinely frozen layer."""
    hard = cert['hard_domain']
    n = len(cert['x'])
    E = np.asarray(hard['fixed_equality_matrix'], float).reshape((-1, n))
    f = np.asarray(hard['fixed_equality_target'], float)
    frozen = cert['previous_frozen']
    return dict(kind='least_squares', A=cert['matrix'], b=cert['target'],
                G=hard['original_inequality_matrix_with_box'],
                h=hard['original_inequality_upper_with_box'],
                E=np.vstack([E] + [np.asarray(z['matrix']) for z in frozen]),
                f=np.r_[f, *[np.asarray(z['attained']) for z in frozen]],
                previous_frozen=frozen, atol=atol, rtol=rtol)


def verify_certificate(cert):
    from v21_original_matrix_kkt_verifier_v1 import verify_stage
    problem = original_problem_from_certificate(cert)
    k = cert['KKT']
    return verify_stage(problem, cert['x'], duals=dict(
        inequality_unit=k['full_normalized_inequality_dual'],
        equality_raw=k['equality_raw_dual'],
        provenance='independent original-matrix V24 certificate check'))


def normal_definition(public):
    """Independent current-geometry formulation from saved public inputs.

    These are proposal rows, not a prediction of future robot/contact force.
    No previous solver point or hidden object property enters this assembly.
    """
    dt = float(public['dt_s'])
    units = np.asarray(public['reference_rates21']) * dt
    J = np.asarray(public['normal_jacobian_Px25'])
    B = np.asarray(public['proposal_map25x21'])
    y = np.asarray(public['pressure_P_N'])
    P = len(y)
    N = J @ B
    Nunit = N * units[None, :]
    debt = np.asarray(public['current_reference21']) - np.asarray(public['encoder_reference21'])
    from v21_stable_handoff.design import COMMAND_DEBT_INTENT_PERIOD_S
    b = -dt * (np.maximum(J @ public['robot_velocity25'], 0.)
               + np.maximum((N @ debt) / COMMAND_DEBT_INTENT_PERIOD_S, 0.))
    rowunits = np.sum(abs(Nunit), axis=1)
    effective = np.where(rowunits == 0., 1., rowunits)
    rel = y / y.max() if P else np.empty(0)
    shares = rel / rel.sum() if P else np.empty(0)
    A = np.zeros((P, 21 + P))
    A[:, 21:] = np.diag(np.sqrt(shares))
    canonical_A = np.column_stack((np.eye(21), np.zeros((21, P))))
    slo, shi = np.asarray(public['step_lower21']), np.asarray(public['step_upper21'])
    maxnormal = np.sum(np.where(N >= 0., N*shi[None, :], N*slo[None, :]), axis=1)
    eta_upper = np.maximum(0., maxnormal-b)/effective
    return dict(A=A, b=np.zeros(P), canonical_A=canonical_A,
                canonical_b=np.zeros(21), normal_G=np.column_stack(
                    (Nunit / effective[:, None], -np.eye(P))),
                normal_h=b / effective, units=units, eta_weights=np.sqrt(shares),
                lower=np.r_[slo/units, np.zeros(P)],
                upper=np.r_[shi/units, eta_upper], shares=shares, Nunit=Nunit)


def motion_definition(public, canonical_x):
    M = public['motion_matrix_Mx21']
    e = public['motion_error_M_m_s']
    if M is None or e is None:
        return None
    M, e = np.asarray(M), np.asarray(e)
    units = np.asarray(public['reference_rates21']) * public['dt_s']
    rowunits = np.sum(abs(M) * units[None, :], axis=1)
    scale = np.where(rowunits == 0., 1., rowunits)
    m = M * units[None, :] / scale[:, None]
    offset = public['dt_s'] * e / scale
    budget = abs(m @ np.asarray(canonical_x)[:21] + offset)
    return dict(matrix=m, offset=offset, budget=budget,
                h=np.r_[budget - offset, budget + offset])


def emitted_score(result, public):
    from v21_stable_handoff.prefix_supervisor import score_emitted_reference
    current = np.asarray(public['current_reference21'])
    scale = np.asarray(public['reference_rates21']) * public['dt_s']
    lo = current + public['step_lower21']
    hi = current + public['step_upper21']
    raw = current + result.x * scale
    emitted = np.minimum(hi, np.maximum(lo, raw))
    roundoff = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(abs(raw), abs(emitted)))
    score = score_emitted_reference(result, current21=current,
                                   proposed21=emitted, scale21=scale)
    return dict(roundoff_only=bool(np.all(abs(raw-emitted) <= roundoff)),
                projection_inf=float(np.max(abs(raw-emitted), initial=0.)),
                emitted21=emitted, samegoal=score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--expected-task-sha', required=True)
    args = parser.parse_args()
    destination = ROOT/args.out
    assert not destination.exists()
    assert sha(ROOT/'v24_canonical_task_fusion/task_solver.py') == args.expected_task_sha
    from v22_joint_task_fusion.numerics import ensure_numeric_environment, numerical_identity
    ensure_numeric_environment()
    import mujoco
    forbidden = []
    def deny(*a, **kw):
        forbidden.append('engine_model_step_forward_contact_writer')
        raise AssertionError(forbidden[-1])
    class NoModel:
        def __new__(cls, *a, **kw): return deny()
        from_xml_path = staticmethod(deny)
        from_xml_string = staticmethod(deny)
        from_binary_path = staticmethod(deny)
    mujoco.MjModel, mujoco.MjData = NoModel, deny
    for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_forwardSkip',
                 'mj_kinematics', 'mj_fwdPosition', 'mj_fwdVelocity', 'mj_collision',
                 'mj_contactForce'):
        if hasattr(mujoco, name): setattr(mujoco, name, deny)
    from v13_a1.codec import ReferenceActor21
    ReferenceActor21.apply = deny
    from v24_canonical_task_fusion import task_solver as task
    from v24_canonical_task_fusion.runtime import source_identity
    from v23_task_execution import finite_solver as finite
    from v22_joint_task_fusion import finite_solver as oldfinite
    checks, issues, rows, special = [], [], [], {}
    source, identity = {}, {}
    def ck(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=plain(detail)))
        if not passed: issues.append(name)
    def inspect(name, public, result, formulation):
        d, P = result.diagnostics, len(public['pressure_P_N'])
        expected = normal_definition(public)
        pair = d.get('normal_witness_certified_prefix', ())
        record = dict(name=name, formulation=formulation, status=result.status,
                      completed_prefix=result.completed_prefix,
                      fully_converged=result.fully_converged, available=result.available)
        ck(name+'/known_input_actual_prefix_available', result.available)
        if P:
            ck(name+'/positive_eta_weights_uniqueness_precondition',
               np.all(expected['eta_weights'] > 0.))
            ck(name+'/two_certified_witness_layers', len(pair) == 2
               and d.get('canonical_witness_available') is True
               and d.get('canonical_witness_certified') is True)
            if len(pair) != 2: return record
            c0, c1 = pair
            domain = c0['hard_domain']
            lo, hi = expected['lower'], expected['upper']
            free, fixed = np.flatnonzero(lo != hi), np.flatnonzero(lo == hi)
            G = np.vstack((expected['normal_G'], np.eye(len(lo))[free], -np.eye(len(lo))[free]))
            h = np.r_[expected['normal_h'], hi[free], -lo[free]]
            ck(name+'/normal_original_objective_from_public_inputs',
               bits(c0['matrix'], expected['A']) and bits(c0['target'], expected['b']))
            ck(name+'/normal_entire_original_domain_from_public_inputs',
               bits(domain['lower'], lo) and bits(domain['upper'], hi)
               and bits(domain['original_inequality_matrix_with_box'], G)
               and bits(domain['original_inequality_upper_with_box'], h)
               and bits(domain['fixed_equality_matrix'], np.eye(len(lo))[fixed])
               and bits(domain['fixed_equality_target'], lo[fixed]))
            ck(name+'/canonical_only_identity_x21_no_eta_penalty',
               c1['name'] == task.CANONICAL_WITNESS_LEVEL
               and bits(c1['matrix'], expected['canonical_A'])
               and bits(c1['target'], expected['canonical_b']))
            frozen = c1['previous_frozen']
            ck(name+'/only_normal_attained_frozen_for_selector', len(frozen) == 1
               and bits(frozen[0]['matrix'], c0['matrix'])
               and bits(frozen[0]['attained'], c0['attained']))
            verifications = [verify_certificate(c) for c in pair]
            ck(name+'/both_witness_original_KKT', all(v['kkt_passed'] for v in verifications))
            xA = np.asarray(c1['x'])
            ck(name+'/selector_literal_box', np.all(xA >= lo) and np.all(xA <= hi))
            ck(name+'/final_canonical_witness_is_budget_anchor',
               bits(xA, d['certified_L1_extended_anchor']))
            ck(name+'/original_normal_attained_retained',
               np.all(abs(expected['A']@xA-np.asarray(c0['attained']))
                      <= c1['KKT']['coordinate_tolerance']))
            motion = motion_definition(public, xA)
            if motion is not None:
                saved = d['motion_task']
                ck(name+'/motion_budget_rebuilt_from_canonical_not_first_point',
                   bits(saved['witness_normalized_x21'], xA[:21])
                   and bits(saved['normalized_absolute_budget_M'], motion['budget'])
                   and bits(saved['budget_inequality_upper'], motion['h']))
            if formulation == 'backend':
                ck(name+'/backend_only_original_normal_certificate_submitted',
                   result.certified_prefix[0]['name'] == c0['name']
                   and bits(result.certified_prefix[0]['matrix'], c0['matrix'])
                   and d['normal_first_certificate_is_submitted_prefix'])
                ck(name+'/backend_eta_frozen_only_after_canonical',
                   bits(d['second_stage_lower'][21:], xA[21:])
                   and bits(d['second_stage_upper'][21:], xA[21:]))
            else:
                ck(name+'/joint_eta_original_bounds_not_normal_optimum_fixed',
                   bits(d['second_stage_lower'][21:], lo[21:])
                   and bits(d['second_stage_upper'][21:], hi[21:])
                   and not d['normal_first_certificate_is_submitted_prefix'])
                ck(name+'/joint_only_true_actual_objective_first',
                   result.certified_prefix[0]['name'] in
                   ('joint_normal_and_observed_projection', 'normal_correction_motion_unknown'))
            record.update(witness_original_matrix_verifications=verifications,
                          canonical_xA=xA, canonical_eta=xA[21:],
                          canonical_objective=c1['squared_loss'],
                          normal_attained=c0['attained'],
                          motion_budget=d.get('motion_task', {}).get('normalized_absolute_budget_M'))
        else:
            ck(name+'/knownP0_witness_not_invented', not pair
               and not d['canonical_witness_required']
               and d['normal_witness_selector'] == 'NOT_REQUIRED_KNOWN_P0')
        ck(name+'/minnorm_certificate_never_actual_or_frozen',
           all(c['name'] != task.CANONICAL_WITNESS_LEVEL for c in result.certified_prefix)
           and all(z['name'] != task.CANONICAL_WITNESS_LEVEL for z in result.levels)
           and all(not bits(p['matrix'], expected['canonical_A'])
                   for c in result.certified_prefix for p in c['previous_frozen']))
        # Rebuild the ACTUAL goals too, not merely their returned duals.
        dim = 21 + P
        work = -expected['shares'] @ expected['Nunit'] if P else np.zeros(21)
        work_upper = float(np.sum(np.where(work >= 0., work*expected['upper'][:21],
                                           work*expected['lower'][:21])))
        work_scale = max(float(np.max(abs(work), initial=0.)), abs(work_upper)) or 1.
        work_row = np.zeros((1, dim)); work_row[0, :21] = work/work_scale
        chosen = public.get('path_reference_target_delta21')
        if chosen is None: chosen = public['reference_target_delta21']
        goals = {'normal_inward_correction_slack': (expected['A'], expected['b']),
                 'normal_correction_motion_unknown': (expected['A'], expected['b']),
                 'scalar_pressure_share_normal_withdrawal': (work_row, np.array([work_upper/work_scale])),
                 'canonical_or_explicit_path_reference': (expected['canonical_A'], chosen/expected['units'])}
        md = motion_definition(public, d['certified_L1_extended_anchor'])
        if md is not None:
            projection = np.column_stack((md['matrix'], np.zeros((len(md['matrix']), P))))
            factor = 1/float(np.sqrt(len(projection)))
            goals['public_external_projection_low_motion'] = (projection, -md['offset'])
            goals['joint_normal_and_observed_projection'] = (
                np.vstack((expected['A'], factor*projection)),
                np.r_[expected['b'], factor*(-md['offset'])])
            goals['observed_projection_without_positive_normal_rows'] = (factor*projection, factor*(-md['offset']))
        second_lo, second_hi = expected['lower'].copy(), expected['upper'].copy()
        if formulation == 'backend' and P:
            second_lo[21:] = second_hi[21:] = np.asarray(d['certified_L1_extended_anchor'])[21:]
        hard_A, hard_b = expected['normal_G'], expected['normal_h']
        if md is not None:
            projection = np.column_stack((md['matrix'], np.zeros((len(md['matrix']), P))))
            hard_A = np.vstack((hard_A, projection, -projection))
            hard_b = np.r_[hard_b, md['h']]
        free2, fixed2 = np.flatnonzero(second_lo != second_hi), np.flatnonzero(second_lo == second_hi)
        originalG2 = np.vstack((hard_A, np.eye(dim)[free2], -np.eye(dim)[free2]))
        originalh2 = np.r_[hard_b, second_hi[free2], -second_lo[free2]]
        actual_later = [c for c in result.certified_prefix
                        if c['name'] != 'normal_inward_correction_slack']
        ck(name+'/actual_entire_hard_domain_rebuilt_no_hidden_identity_freeze',
           all(bits(c['hard_domain']['lower'], second_lo)
               and bits(c['hard_domain']['upper'], second_hi)
               and bits(c['hard_domain']['original_inequality_matrix_with_box'], originalG2)
               and bits(c['hard_domain']['original_inequality_upper_with_box'], originalh2)
               and bits(c['hard_domain']['fixed_equality_matrix'], np.eye(dim)[fixed2])
               and bits(c['hard_domain']['fixed_equality_target'], second_lo[fixed2])
               for c in actual_later))
        ck(name+'/every_actual_objective_rebuilt_from_public_input',
           all(c['name'] in goals and bits(c['matrix'], goals[c['name']][0])
               and bits(c['target'], goals[c['name']][1]) for c in result.certified_prefix))
        actual = [verify_certificate(c) for c in result.certified_prefix]
        ck(name+'/all_submitted_original_matrix_KKT', all(v['kkt_passed'] for v in actual))
        ext = np.asarray(d['extended_normalized_x'])
        last_domain = result.certified_prefix[-1]['hard_domain']
        ck(name+'/actual_literal_box_and_readonly',
           np.all(ext >= last_domain['lower']) and np.all(ext <= last_domain['upper'])
           and not result.x.flags.writeable)
        score = emitted_score(result, public)
        ck(name+'/actual_reference_roundoff_and_samegoal',
           score['roundoff_only'] and score['samegoal']['available'])
        record.update(actual_original_matrix_verifications=actual,
                      actual_certificate_names=[c['name'] for c in result.certified_prefix],
                      emission=score)
        return record
    try:
        source, identity = source_identity(['ratio_effort']), numerical_identity()
        parent = json.loads((ROOT/'review/v23_small_plan_v1.json').read_text('utf8'))['source_sha256']
        ck('source192_parent186_exact', len(source) == 192 and len(parent) == 186
           and all(source.get(k) == v for k, v in parent.items()))
        ck('frozen_finite4b_numerics2a_legacy42',
           source['v23_task_execution/finite_solver.py'] == '4b2ac6a4d00896729aded914bece9efa21421643c0cc9b2e598f7be46c099dbf'
           and source['v22_joint_task_fusion/numerics.py'] == '2a4241deafbab2269ca72e670bb99c76dd73249e9d52a78e6bb45243c501875a'
           and identity['original_legacy42_unchanged'])
        artifact_path = ROOT/'review/v23_joint_friction_first_divergence_v1.json'
        artifact = json.loads(artifact_path.read_text('utf8'))
        public_inputs = []
        for row in artifact['local_same_public_input_counterfactuals']:
            public = deepcopy(row['all_kwargs'])
            for key, value in list(public.items()):
                if isinstance(value, list): public[key] = np.asarray(value, float)
            public_inputs.append(public)
        ck('fixed_three_saved_public_inputs_no_outcome_selection', len(public_inputs) == 3)
        real_results = {}
        for index, public in enumerate(public_inputs):
            for formulation in ('backend', 'joint'):
                name = f'public{index}_{public["time_s"]}_{formulation}'
                start = time.perf_counter()
                result = task.solve_contact_exit(**dict(public, formulation=formulation))
                row = inspect(name, public, result, formulation)
                row['elapsed_s'] = time.perf_counter()-start
                rows.append(row)
                real_results[index, formulation] = result
            # Alternative numerical route: same task, only finite candidate order differs.
            def alternative(**kw):
                adapted = dict(kw)
                adapted['levels'] = tuple(oldfinite.Level(z.name, z.matrix, z.target) for z in kw['levels'])
                return oldfinite.solve_hierarchy(**adapted)
            with patch.object(task, 'solve_hierarchy', side_effect=alternative):
                alt = task.solve_contact_exit(**dict(public, formulation='joint'))
            row = inspect(f'public{index}_alternate_f304_route_joint', public, alt, 'joint')
            reference = real_results[index, 'joint'].diagnostics
            row['numerical_route_comparison'] = dict(
                canonical_x_difference_inf=float(np.max(abs(
                    np.asarray(alt.diagnostics['certified_L1_extended_anchor'])
                    - np.asarray(reference['certified_L1_extended_anchor'])), initial=0.)),
                budget_difference_inf=float(np.max(abs(
                    np.asarray(alt.diagnostics['motion_task']['normalized_absolute_budget_M'])
                    - np.asarray(reference['motion_task']['normalized_absolute_budget_M'])), initial=0.)),
                finite_attained_equality_is_not_required_to_be_bit_identical=True)
            rows.append(row)
        rates = np.ones(21)
        units = rates*.005
        B = np.zeros((25, 21)); B[:21] = np.eye(21)
        J = np.zeros((1, 25)); J[0, 0] = 1.
        M = np.zeros((1, 21)); M[0, 0] = 1.
        fixture = dict(time_s=12.5, dt_s=.005, current_reference21=np.zeros(21),
            encoder_reference21=np.zeros(21), step_lower21=-units, step_upper21=units,
            reference_rates21=rates, normal_jacobian_Px25=J, proposal_map25x21=B,
            robot_velocity25=np.zeros(25), pressure_P_N=np.array([1.]),
            reference_target_delta21=np.zeros(21), motion_matrix_Mx21=M,
            motion_error_M_m_s=np.array([1.]), motion_time_s=12.5)
        for form in ('backend', 'joint'):
            result = task.solve_contact_exit(**fixture, formulation=form)
            rows.append(inspect('exit_freedom_'+form, fixture, result, form))
            ck('closed_form_exit_freedom_not_minnorm_freeze/'+form,
               abs(result.x[0]+1.) <= 1.1e-8
               and abs(result.diagnostics['certified_L1_extended_anchor'][0]) <= 1.1e-8)
        unknown = dict(fixture, motion_matrix_Mx21=None, motion_error_M_m_s=None, motion_time_s=None)
        u = task.solve_contact_exit(**unknown, formulation='joint')
        rows.append(inspect('motion_unknown_not_zero', unknown, u, 'joint'))
        ck('motion_unknown_explicitly_omitted', u.diagnostics['motion_unknown_not_zero']
           and u.certified_prefix[0]['name'] == 'normal_correction_motion_unknown')
        p0 = dict(unknown, normal_jacobian_Px25=np.empty((0,25)), pressure_P_N=np.empty(0),
                  reference_target_delta21=np.r_[.002, np.zeros(20)],
                  path_reference_target_delta21=np.r_[.001, np.zeros(20)])
        z = task.solve_contact_exit(**p0, formulation='joint')
        rows.append(inspect('knownP0_explicit_path', p0, z, 'joint'))
        ck('knownP0_path_not_OPEN_reset', abs(z.x[0]-.2) <= 1.1e-8)
        for name, changed in (
                ('unknown_normal', {'normal_jacobian_Px25': None}),
                ('partial_motion', {'motion_error_M_m_s': None}),
                ('old_motion_clock', {'motion_time_s': 12.495}),
                ('boolean_pressure', {'pressure_P_N': [True]})):
            r = task.solve_contact_exit(**dict(fixture, **changed), formulation='joint')
            ck('invalid_unknown_no_actual_prefix/'+name,
               not r.available and not r.certified_prefix and r.completed_prefix == 0)
            special[name] = dict(status=r.status, reason=r.diagnostics.get('reason'))
        actual_solve = finite.solve_hierarchy
        def incomplete_selector(**kw):
            if len(kw['levels']) == 2 and kw['levels'][1].name == task.CANONICAL_WITNESS_LEVEL:
                limited = dict(kw, levels=(kw['levels'][0],))
                first = actual_solve(**limited)
                return replace(first, fully_converged=False,
                               failed_layer=dict(index=1, name=task.CANONICAL_WITNESS_LEVEL,
                                                 status='INJECTED_SELECTOR_NO_CERTIFICATE'))
            return actual_solve(**kw)
        for form in ('backend', 'joint'):
            with patch.object(task, 'solve_hierarchy', side_effect=incomplete_selector):
                r = task.solve_contact_exit(**fixture, formulation=form)
            ck('canonical_failure_never_builds_budget_or_actual_prefix/'+form,
               not r.available and not r.certified_prefix and not r.levels
               and r.completed_prefix == 0 and r.failed_layer is None
               and r.diagnostics['witness_completed_prefix'] == 1
               and r.diagnostics['witness_failed_layer']['index'] == 1
               and not r.diagnostics['normal_witness_budget_defined'])
            special['canonical_failure_'+form] = plain(r)
        def no_actual_certificate(**kw):
            if len(kw['levels']) == 2 and kw['levels'][1].name == task.CANONICAL_WITNESS_LEVEL:
                return actual_solve(**kw)
            anchor = actual_solve(**dict(kw, levels=()))
            return replace(anchor, fully_converged=False,
                           failed_layer=dict(index=0, name=kw['levels'][0].name,
                                             status='INJECTED_ACTUAL_FIRST_LAYER_FAILURE'))
        for form in ('backend', 'joint'):
            with patch.object(task, 'solve_hierarchy', side_effect=no_actual_certificate):
                r = task.solve_contact_exit(**fixture, formulation=form)
            expected = 1 if form == 'backend' else 0
            ck('later_failure_preserves_only_genuine_actual_prefix/'+form,
               r.completed_prefix == expected and len(r.certified_prefix) == expected
               and all(c['name'] != task.CANONICAL_WITNESS_LEVEL for c in r.certified_prefix)
               and not r.fully_converged and r.failed_layer['index'] == expected)
            special['later_failure_'+form] = plain(r)
        ck('source192_and_numerical_identity_unchanged',
           source == source_identity(['ratio_effort']) and identity == numerical_identity())
        ck('zero_model_step_forward_contact_writer', not forbidden)
        status = 'complete_fixed_public_assembly_and_finite_counterexamples'
    except Exception as exc:
        issues.append('UNEXPECTED:'+repr(exc))
        special['unexpected'] = dict(error=repr(exc), traceback=traceback.format_exc())
        status = 'independent_attempt_error'
    subset = {path: source[path] for path in (
        'v24_canonical_task_fusion/task_solver.py', 'v24_canonical_task_fusion/supervisor.py',
        'v23_task_execution/finite_solver.py', 'v22_joint_task_fusion/task_bundle.py',
        'v22_joint_task_fusion/numerics.py') if path in source}
    payload = dict(schema='v24_numerical_independent_v1', passed=not issues,
        status=status, checks=checks, issues=issues, rows=rows, special=special,
        production_source_sha256=subset, source_sha256=source,
        numerical_identity=identity, checker_sha256=sha(__file__),
        fixed_public_input_sha256=sha(ROOT/'review/v23_joint_friction_first_divergence_v1.json'),
        independent_verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'),
        forbidden_calls=forbidden, physics_steps=0, model_constructions=0,
        scope='Three preselected saved public mathematical inputs; new canonical assembly rebuilt and checked, not old81 reused. Both witness layers separate from actual-task prefix; original full-row KKT, strictbox and actual emitted samegoal checked. Alternative f304 route only a review mathematical probe. No new physical performance or realtime claim.')
    with destination.open('x', encoding='utf8') as stream:
        json.dump(plain(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=payload['passed'], checks=len(checks), issues=issues,
                         inspected_results=len(rows), out=str(destination),
                         sha256=sha(destination)), ensure_ascii=False), flush=True)
    return 0 if payload['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

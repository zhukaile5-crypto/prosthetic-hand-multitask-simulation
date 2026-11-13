"""V24 integration AUTHOR gate; no rollout or task-mathematics certification."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib, json, sys, traceback
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def function(path, name):
    return next(n for n in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8')))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(n):
    return ast.dump(n, include_attributes=False)


class NormalizeNamespace(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('v24_canonical_task_fusion', 'v23_task_execution')
        return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero-step', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    assert not out.exists(), 'fresh gate output required'
    checks = []

    def ck(name, good, **extra):
        checks.append(dict(name=name, passed=bool(good), **extra))

    old = ROOT / 'v23_task_execution/supervisor.py'
    new = ROOT / 'v24_canonical_task_fusion/supervisor.py'
    for name in ('action', 'observe', 'nominal', 'reconcile', '__init__'):
        ck(name + '_AST_unchanged_except_namespace',
           dump(function(old, name)) == dump(NormalizeNamespace().visit(copy.deepcopy(function(new, name)))))
    ck('new_task_import_only_and_original_emitted_score',
       'from .task_solver import solve_contact_exit' in new.read_text(encoding='utf-8')
       and 'from v21_stable_handoff.prefix_supervisor import contact_exit_lift_interval, score_emitted_reference'
       in new.read_text(encoding='utf-8'))

    mod = importlib.import_module('v24_canonical_task_fusion.runtime')
    runner = importlib.import_module('v24_canonical_task_fusion.run_development')
    design = importlib.import_module('v24_canonical_task_fusion.design')
    from v23_task_execution.supervisor import ReleaseSupervisor as RealFrozenV23
    identity = mod.source_identity(['ratio_effort'])
    inherited = mod.frozen_v23_files(['ratio_effort'])
    oldplan = json.loads((ROOT / 'review/v23_small_plan_v1.json').read_text(encoding='utf-8'))
    v22plan = json.loads((ROOT / 'review/v22_small_plan_v1.json').read_text(encoding='utf-8'))
    ck('complete192_source_set_and_frozen186_byteidentity',
       len(inherited) == 186 and len(identity) == 192
       and {k: identity[k] for k in inherited} == oldplan['source_sha256']
       and set(identity) - set(inherited) == {'v24_canonical_task_fusion/' + n for n in design.SOURCE_NAMES})
    ck('frozen179_byteidentity_and_no_new_finite_bundle_or_numerics',
       {k: identity[k] for k in v22plan['source_sha256']} == v22plan['source_sha256']
       and set(design.SOURCE_NAMES) == {'__init__.py', 'design.py', 'task_solver.py', 'supervisor.py', 'runtime.py', 'run_development.py'})
    ck('frozen_role_uses_exact_original_V23_class', mod.FrozenV23Supervisor is RealFrozenV23)
    numeric = mod.numerical_identity()
    ck('all_roles_retain_original_numerical_identity', numeric == oldplan['numerical_identity'])

    for role in design.RELEASE_VARIANTS:
        calls, bound, base = [], [], []

        def frozen_factory(*a, **kw):
            calls.append(('v23_joint', kw))
            return SimpleNamespace(kind='v23_joint', formulation=kw['formulation'])

        def new_factory(*a, **kw):
            calls.append(('canonical', kw))
            return SimpleNamespace(kind='canonical', formulation=kw['formulation'])

        entry = SimpleNamespace(started_s=12.)
        obj = mod.ReleaseSession.__new__(mod.ReleaseSession)
        obj.release_variant = role
        obj.v16_release_policy = 'coordinated'
        obj.v16_supervisor = None
        obj.plant = SimpleNamespace(joint_release_entry=entry, model=object(),
            open_pose=(np.zeros(5), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(4)),
            reference_authority21=lambda **k: (np.zeros(21), np.ones(21)),
            bind_supervisor=lambda value: bound.append(value))
        obj.release_adapter = SimpleNamespace(begin_release=lambda t: dict(prior=True),
            receiver_half_size_xy_m=np.array([.1, .1]), receiver_up_axis3=np.array([0., 0., 1.]))
        obj.controller = SimpleNamespace(limits=np.ones(5), budget=5., g=SimpleNamespace(force_margin=.1))

        def parent_step(self, obs):
            base.append(obs)
            return 'same_parent_actor'

        with patch.object(mod, 'FrozenV23Supervisor', frozen_factory), \
             patch.object(mod, 'ReleaseSupervisor', new_factory), \
             patch.object(mod, 'TransferObserver', lambda **kw: SimpleNamespace(**kw)), \
             patch.object(mod, 'release_clearance_goal', lambda *a, **kw: SimpleNamespace(reference18=np.zeros(18))), \
             patch.object(mod.FrozenSession, '_lifecycle_step', parent_step):
            a = obj._lifecycle_step(object())
            b = obj._lifecycle_step(object())
        ck('lifecycle_' + role + '_first_binding_and_parent_actor',
           a == b == 'same_parent_actor' and len(calls) == len(bound) == 1 and len(base) == 2
           and obj.release_driver is bound[0]
           and calls[0][0] == ('v23_joint' if role == 'v23_joint' else 'canonical')
           and calls[0][1]['formulation'] == design.ROLE_FORMULATIONS[role])

    for value in ('output', 'review/invalid'):
        try:
            runner.output_path(value)
            rejected = False
        except ValueError:
            rejected = True
        ck('runner_rejects_nonfresh_output_scope_' + value, rejected)
    main_node = function(ROOT / 'v24_canonical_task_fusion/run_development.py', 'main')
    argument = next(n for n in ast.walk(main_node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
        and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--release-variant')
    ck('CLI_three_explicit_required_roles',
       any(k.arg == 'required' and ast.literal_eval(k.value) is True for k in argument.keywords)
       and any(k.arg == 'choices' and ast.literal_eval(k.value) == design.RELEASE_VARIANTS for k in argument.keywords))
    constructed, written = [], []
    with patch.object(runner, 'SupportSession', lambda **k: constructed.append(k)), \
         patch.object(runner, 'write', lambda p, v: written.append((str(p), v))), \
         patch.object(Path, 'mkdir', lambda *a, **k: None):
        try:
            runner.run_one(dict(destination=str(ROOT / 'review/virtual_bad_v24_numeric'), method='ratio_effort'),
                           identity, dict(wrong='numeric-pin'))
            rejected = False
        except RuntimeError as exc:
            rejected = 'numerical dependency identity mismatch' in str(exc)
    ck('worker_rejects_wrong_numeric_before_construction_and_keeps_error',
       rejected and not constructed and len(written) == 1
       and 'numerical dependency identity mismatch' in written[0][1]['error'])

    forwards, steps, time_records = 0, 0, []
    if args.zero_step:
        import mujoco, hand_plant
        original_forward = mujoco.mj_forward

        def forward(*a, **kw):
            nonlocal forwards
            forwards += 1
            return original_forward(*a, **kw)

        def forbidden(*a, **kw):
            nonlocal steps
            steps += 1
            raise RuntimeError('integration forbidden in author gate')

        with patch.object(mujoco, 'mj_forward', forward), patch.object(mujoco, 'mj_step', forbidden), \
             patch.object(mujoco, 'mj_step1', forbidden), patch.object(mujoco, 'mj_step2', forbidden), \
             patch.object(hand_plant.HandPlant, 'advance', forbidden):
            for role in design.RELEASE_VARIANTS:
                s = mod.ReleaseSession(method='ratio_effort', scenario='nominal', seed=619741,
                    split='robustness', motion_mode='transport', task_policy='integrated',
                    receiver_policy='coordinated', release_variant=role)
                m = s.metadata()
                time_records.append(dict(role=role, time=s.time, tick=s.tick, samples=s.sensors.samples))
                ck('real_zero_step_' + role + '_coherent_metadata_and_identity',
                   s.time == 0. and s.tick == 0 and s.sensors.samples == 0 and not s.trace
                   and m['release_role'] == m['release_variant'] == role
                   and m['research_source_sha256'] == identity
                   and m['frozen_v23_source_sha256'] == oldplan['source_sha256']
                   and m['numerical_identity'] == s.numerical_sources == numeric
                   and m['v24_release_design']['declared_task_formulation'] == design.ROLE_FORMULATIONS[role]
                   and m['v24_release_design']['canonical_task_contract'] == (role != 'v23_joint'))

    ck('executed_sources_and_numeric_identity_unchanged',
       identity == mod.source_identity(['ratio_effort']) and numeric == mod.numerical_identity())
    result = dict(schema='v24_integration_author_v1', passed=all(c['passed'] for c in checks),
        checks=checks, production_source_sha256=identity, source_sha256=identity,
        numerical_identity=numeric, source_count=len(identity), full192=len(identity) == 192,
        parent186_source_sha256={k: identity[k] for k in inherited},
        zero_step_requested=args.zero_step, initialization_forward_calls=forwards,
        integration_entry_calls=steps, time_records=time_records, physics_steps=0,
        checker_sha256=sha(__file__),
        scope='Integration author zero-step and route checks, not independent task mathematics or physical performance.')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(path=str(out), sha256=sha(out), passed=result['passed'], checks=len(checks),
                         source_count=len(identity), initialization_forward_calls=forwards, integration_entry_calls=steps), ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = ROOT / sys.argv[sys.argv.index('--out') + 1]
        result = dict(schema='v24_integration_author_v1', passed=False, error=repr(exc),
                      traceback=traceback.format_exc(), physics_steps=0, source_gate_complete=False)
        if not out.exists():
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(result['traceback'])
        raise SystemExit(1)

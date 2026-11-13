"""No simulator imports: inventory/clocks/equality semantics and launch gates."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
from .source_record import PILOT, METHODS, load, read, canonical, exact, require_unchanged

def main():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    check('no simulator imported before tests', 'mujoco' not in sys.modules)
    check('canonical config JSON roundtrip preserves exact values', canonical({'x': (1., 2.)}) == canonical(json.loads(canonical({'x': (1., 2.)}))))
    check('byte exact distinguishes signed zero', not exact(np.array([0.]), np.array([-0.])))
    check('byte exact does not silently cast float precision', not exact(np.array([1.]), np.array([1.], dtype=np.float32)))
    plan = read(PILOT/'plan.json')
    check('three named source methods fixed', tuple(plan['methods']) == METHODS)
    check('two finer steps fixed', plan['refinement_timesteps_s'] == [.00025, .000125])
    check('bridge step fixed', plan['source_timestep_s'] == .0005)
    for path in PILOT.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    check('all prototype Python sources parse', True)
    for method in METHODS:
        record = load(method); a = record['arrays']
        check(method+' actual randomized mass, not standard substituted', record['metadata']['config']['mass'] == .12806493902364582)
        check(method+' actual randomized friction retained', record['metadata']['config']['friction'] == .5531784863439101)
        check(method+' all22ctrl finite and complete', a['ctrl'].shape == (3201,22) and np.isfinite(a['ctrl']).all())
        check(method+' full16s raw coverage', abs(a['substep_duration'].sum()-16.) < 1e-10 and len(a['substep_time']) == 32000)
        check(method+' release starts at12s', a['time'][record['release_index']] == 12.)
        check(method+' current env obstacle onset recovered', record['environments'][1860]['obstacle_x'] == -.30 and record['environments'][1859]['obstacle_x'] == 0.)
        require_unchanged(record)
        check(method+' immutable trace SHA verified', True)
        del record, a
    check('no simulator imported by actual input inventory', 'mujoco' not in sys.modules)
    print(json.dumps(dict(passed=True, checks=len(checks), physics_steps=0, simulator_imported=False), ensure_ascii=False))

if __name__ == '__main__':
    main()

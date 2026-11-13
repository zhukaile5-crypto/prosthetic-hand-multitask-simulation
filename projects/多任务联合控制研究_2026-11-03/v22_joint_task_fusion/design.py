"""V22 integration declarations; physics, devices and permissions remain V21."""
from v21_stable_handoff.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v21_stable_handoff.design import COMMAND_DEBT_INTENT_PERIOD_S, MAX_GEOMETRY_AGE_S

SCHEMA = 'v22_joint_task_fusion_lifecycle_v1'
RELEASE_VARIANTS = ('v21', 'backend', 'joint')
FORMULATIONS = ('backend', 'joint')
SOURCE_NAMES = ('__init__.py', 'design.py', 'task_bundle.py', 'numerics.py', 'finite_solver.py',
                'task_solver.py', 'supervisor.py', 'runtime.py', 'run_development.py')

"""V23 finite-backend integration; the V22 task and execution rules persist."""
from v22_joint_task_fusion.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v22_joint_task_fusion.design import COMMAND_DEBT_INTENT_PERIOD_S, MAX_GEOMETRY_AGE_S
SCHEMA = 'v23_task_execution_lifecycle_v1'
RELEASE_VARIANTS = ('v21', 'backend', 'joint')
FORMULATIONS = ('backend', 'joint')
SOURCE_NAMES = ('__init__.py', 'design.py', 'finite_solver.py', 'task_solver.py',
                'supervisor.py', 'runtime.py', 'run_development.py')

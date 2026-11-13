"""Explicit V24 task roles on unchanged physical and execution authority."""
from v23_task_execution.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v23_task_execution.design import COMMAND_DEBT_INTENT_PERIOD_S, MAX_GEOMETRY_AGE_S

SCHEMA = 'v24_canonical_task_fusion_lifecycle_v1'
RELEASE_VARIANTS = ('v23_joint', 'canonical_backend', 'canonical_joint')
FORMULATIONS = ('backend', 'joint')
ROLE_FORMULATIONS = {'v23_joint': 'joint', 'canonical_backend': 'backend', 'canonical_joint': 'joint'}
SOURCE_NAMES = ('__init__.py', 'design.py', 'task_solver.py',
                'supervisor.py', 'runtime.py', 'run_development.py')

"""Engineering task choices fixed before V21 physical development."""
from v20_contact_exit.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v20_contact_exit.design import COMMAND_DEBT_INTENT_PERIOD_S
from v18_effort_transfer.design import MAX_GEOMETRY_AGE_S

SCHEMA = 'v21_motion_budget_and_exit_path_lifecycle_v1'
RELEASE_VARIANTS = ('v20', 'prefix', 'v21')
SOURCE_NAMES = ('__init__.py', 'design.py', 'prefix_hierarchy.py',
                'task_solver.py', 'prefix_task.py', 'path_memory.py',
                'supervisor.py', 'prefix_supervisor.py', 'runtime.py',
                'run_development.py')

"""V20 engineering task choices, frozen before its first physical trial."""
from v18_effort_transfer.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v14_coord.design import DECISION_PERIOD_S

SCHEMA = 'v20_contact_conditioned_whole_hand_exit_v1'
# This is the existing software recovery budget interval, not an identified
# actuator response time or a physical stopping guarantee.
COMMAND_DEBT_INTENT_PERIOD_S = DECISION_PERIOD_S
SOURCE_NAMES = ('__init__.py', 'design.py', 'geometry.py', 'adapter.py',
                'task_solver.py', 'supervisor.py', 'runtime.py',
                'run_development.py')

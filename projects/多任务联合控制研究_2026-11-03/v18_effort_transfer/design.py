"""V18 structural choices, declared before its first physical trial."""
from v14_coord.design import CONTROL_PERIOD_S, DECISION_PERIOD_S

SCHEMA = 'v18_current_effort_contact_transfer_v1'
PHASES = ('SEARCH', 'CAPTURE', 'TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT')
RELEASE_DEADLINE_S = 3.5
# A newly declared engineering effort-intent scale, not a calibrated contact
# damping or a passivity/safety bound. It is fixed before the development run.
DEVICE_TANGENT_DAMPING_N_S_M = 10.0
MAX_GEOMETRY_AGE_S = .12
SOURCE_NAMES = ('__init__.py', 'design.py', 'adapter.py', 'governor.py',
                'effort_task.py', 'supervisor.py', 'runtime.py',
                'run_development.py')

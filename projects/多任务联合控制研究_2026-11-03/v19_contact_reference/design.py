"""One declared V19 task, fixed before physical development trials."""
from v18_effort_transfer.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v18_effort_transfer.design import DEVICE_TANGENT_DAMPING_N_S_M

SCHEMA='v19_explicit_reference_and_contact_motion_v1'
# Reuse the previous engineering intent scale, without fitting these two new
# task meanings to a favorable trajectory. Neither is a calibrated damping.
EXTERNAL_PROJECTION_DAMPING_N_S_M=DEVICE_TANGENT_DAMPING_N_S_M
ROBOT_PROJECTION_DAMPING_N_S_M=DEVICE_TANGENT_DAMPING_N_S_M
PROGRESS_SCALE=1.
SOURCE_NAMES=('__init__.py','design.py','adapter.py','native_branch.py',
              'task_qp.py','supervisor.py','runtime.py','run_development.py')

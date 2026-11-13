"""Evidence-only child of the frozen V17 decision owner.

The parent makes every decision and returns the same command/debug objects.
Only an already-created MotionTask is copied into each new decision record.
There is no sensing, optimization, mutation of task inputs, or extra decision.
"""
from v17_contact_transfer.supervisor import ReleaseSupervisor as FrozenSupervisor


class ReleaseSupervisor(FrozenSupervisor):
    def action(self, **kwargs):
        first = len(self.decision_evidence)
        result = super().action(**kwargs)
        for record in self.decision_evidence[first:]:
            task = kwargs['adapter'].motion_task if record['motion_task_available'] else None
            record['motion_task_public'] = None if task is None else task.to_public()
            record['motion_evidence_role'] = 'already_created_public_task_copy_after_command_decision'
        return result

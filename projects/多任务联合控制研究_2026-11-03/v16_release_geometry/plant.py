"""Independent lifecycle authority around the frozen21 sole execution writer."""
import numpy as np
from v15_support.plant import SupportPlant


class ReleasePlant(SupportPlant):
    def reset(self):
        self.v16_supervisor = None
        self.v16_last_version = None
        return super().reset()

    def bind_supervisor(self, supervisor):
        if self.joint_release_entry is None:
            raise ValueError('Actual release must preempt before supervisor binding')
        if self.v16_supervisor is not None and self.v16_supervisor is not supervisor:
            raise ValueError('One release supervisor owns this lifecycle')
        self.v16_supervisor = supervisor
        self.v16_last_version = supervisor.version

    def sync_supervisor_generation(self):
        graph = self.v16_supervisor
        if graph is not None and graph.version != self.v16_last_version:
            self.mode_generation += 1
            self.v16_last_version = graph.version

    def reference_authority21(self, *, include_mode=True):
        lo, hi = super().reference_authority21(include_mode=include_mode)
        if include_mode and self.v16_supervisor is not None:
            lower, upper = self.v16_supervisor.lift_interval(self.reference_snapshot())
            lo[18] = max(lo[18], lower); hi[18] = min(hi[18], upper)
            if np.any(lo > hi):
                raise ValueError('Actual authority and release lift task do not intersect')
        return lo, hi

    def release_proposal(self, query_time_s):
        graph = self.v16_supervisor
        if graph is None:
            return super().release_proposal(query_time_s)
        lo, hi = self.reference_authority21(include_mode=False)
        # The stale-plan replacement must use the new graph as well.
        return graph.nominal(query_time_s, lo, hi)

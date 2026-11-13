"""Independent V15 plant adapter: same physics/sole writer, public release tasks.

Delivery evidence is privileged offline data. It is captured from the existing
cached integration solve, never returned as a public sensor measurement.
"""
import numpy as np
from v14_coord.plant import CoordinatedPlant
from .release_tasks import ReceiverReleaseTasks

class SupportPlant(CoordinatedPlant):
    def reset(self):
        self.receiver_release_enabled=False
        self.receiver_release_tasks=None
        self.receiver_release_version=None
        self.delivery_ledger=None
        self._delivery_inside_advance=False
        self._delivery_last_end=None
        return super().reset()

    def begin_release(self):
        first=self.release_started is None
        super().begin_release()
        if first and self.receiver_release_enabled:
            self.receiver_release_tasks=ReceiverReleaseTasks(self.joint_release_entry,np.concatenate(self.open_pose))
            self.receiver_release_version=self.receiver_release_tasks.version

    def update_public_release(self,obs,receiver):
        graph=self.receiver_release_tasks
        if graph is None:return None
        state=graph.update(time_s=float(obs.time),current_ref21=self.reference_snapshot(),
            robot_lift_m=float(obs.lift),normal5=obs.normal,valid5=obs.valid,age5=obs.age,
            aux_pressure=obs.auxiliary_pressure,aux_valid=obs.auxiliary_valid,
            aux_age=obs.auxiliary_age,receiver=receiver)
        if graph.version!=self.receiver_release_version:
            self.mode_generation+=1
            self.receiver_release_version=graph.version
        return state

    def reference_authority21(self,*,include_mode=True):
        lo,hi=super().reference_authority21(include_mode=include_mode)
        graph=self.receiver_release_tasks
        if include_mode and graph is not None:
            lower,upper=graph.lift_interval(self.reference_snapshot())
            lo[18]=max(lo[18],lower);hi[18]=min(hi[18],upper)
            if np.any(lo>hi):raise ValueError('Receiver task and mechanical authority have empty intersection')
        return lo,hi

    def release_proposal(self,query_time_s):
        graph=self.receiver_release_tasks
        if graph is None:return super().release_proposal(query_time_s)
        lo,hi=self.reference_authority21(include_mode=False)
        return graph.proposal(query_time_s,lo,hi)

    def _contacts(self,patches=True):
        value=super()._contacts(patches)
        ledger=self.delivery_ledger
        end=float(self.data.time)
        if ledger is not None and self._delivery_inside_advance and end!=self._delivery_last_end:
            self._delivery_last_end=end
            ledger.capture_cached_interval(self.data,end-float(self.model.opt.timestep),end)
        return value

    def advance(self,step=None):
        self._delivery_inside_advance=True
        try:return super().advance(step)
        finally:self._delivery_inside_advance=False

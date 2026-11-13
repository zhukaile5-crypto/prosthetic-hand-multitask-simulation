"""Core carrier; reuse physical platform, shared observation and holding fusion."""
import hashlib
from pathlib import Path
import numpy as np
from v21_stable_handoff.runtime import ReleaseSession as FrozenSession, source_files as frozen_sources
from v16_release_geometry.transfer_observer import TransferObserver
from v16b_release_goal.goal import release_clearance_goal
from .supervisor import CoreSupervisor
from .approach import receiver_top

ROOT=Path(__file__).resolve().parents[1]
ARMS=("core","without_anticipation","without_handoff")
def source_files():
    result=dict(frozen_sources(["ratio_effort"]))
    result["identification/v21_public_support_preview_v1.py"]=ROOT/"identification/v21_public_support_preview_v1.py"
    result.update({p.relative_to(ROOT).as_posix():p for p in (ROOT/"midterm_core_v1").glob("*.py")})
    return dict(sorted(result.items()))
def identity():
    return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source_files().items()}

class CoreSession(FrozenSession):
    def __init__(self, *, arm="core", **kwargs):
        if arm not in ARMS:
            raise ValueError("Unregistered core arm")
        self.core_arm=arm
        super().__init__(method="ratio_effort",task_policy="integrated",
                         receiver_policy="coordinated",**kwargs)
        if self.shape!="capsule":
            raise ValueError("Core pilot scope is the shared nominal capsule")
        self.core_mount=receiver_top(self.plant.model,self.receiver_mount)
        self.research_sources=identity()
    def _lifecycle_step(self,obs):
        if self.v16_release_policy=="coordinated" and self.v16_supervisor is None:
            entry=self.plant.joint_release_entry
            if entry is None:
                raise ValueError("Release must preempt holding before coordination")
            lower,upper=self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal=release_clearance_goal(entry,
                preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower,mechanical_upper21=upper)
            prior=self.release_adapter.begin_release(entry.started_s)
            observer=TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3,prior=prior)
            self.v16_supervisor=CoreSupervisor(entry,approach_mount=self.core_mount,
                anticipation=self.core_arm!="without_anticipation",
                coordinated=self.core_arm!="without_handoff",observer=observer,
                open18=self.release_clearance_goal.reference18,robot_model=self.plant.model,
                limits5=self.controller.limits,budget=self.controller.budget,
                force_margin=self.controller.g.force_margin)
            self.release_driver=self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)
    def metadata(self):
        meta=super().metadata()
        meta.update(schema="midterm_macro_core_v1",benchmark_id="midterm_core_macro_ablation",
            experiment_arm=self.core_arm,decision_algorithm="shared_holding_plus_distance_approach_plus_contact_handoff",
            controller_scope="unchanged integrated holding; simplified release policy",
            core_policy=dict(anticipation=self.core_arm!="without_anticipation",
                coordinated_handoff=self.core_arm!="without_handoff",
                exit_path_memory=False,optional_withdrawal_budget=False,
                wrist_tracking_lead=False,press_load_prediction=False,
                native_actuation_and_safety_checks_preserved=True))
        return meta

"""Small behavior and architecture checks; no simulation integration."""
import inspect
import numpy as np
from .approach import approach_request
from .runtime import CoreSession, identity
from .supervisor import CoreSupervisor
from v15_support.controller import UnifiedTaskController

def preflight():
    checks={}
    def packet(gap):
        return dict(available=True,read_s=12.,acquisition_s=11.99,
                    engineering_gap_m=[gap-.002,gap+.002],conditional_plane_gap_m=[gap,gap])
    gap=np.linspace(-.003,.020,101)
    speed=np.array([-approach_request(-.04,.04,12.,packet(x))[0] for x in gap])
    checks["monotone_distance_speed_and_bounded_creep"]=bool(np.all(np.diff(speed)>=-1e-12) and
                                                           speed.min()==.01 and speed.max()==.04)
    checks["no_anticipation_is_identity"]=approach_request(-.04,.04,12.,packet(0.),False)[0]==-.04
    checks["stale_and_missing_preview_do_not_invent_geometry"]=all(
        approach_request(-.04,.04,12.,x)[0]==-.04 for x in
        (None,{},dict(packet(0.),read_s=11.9),dict(packet(0.),acquisition_s=11.9)))
    code=inspect.getsource(CoreSupervisor)
    checks["fine_modules_not_called"]=not any(x in code for x in
        ("ExitPathMemory","solve_reserved_transfer","MotionReservedSupervisor","AlignmentController","PressLift"))
    session=CoreSession(arm="core",scenario="nominal",motion_mode="transport",split="robustness",seed=619741)
    checks["same_integrated_holding_controller"]=type(session.controller) is UnifiedTaskController
    checks["construction_has_no_physics_or_sensor_sample"]=session.tick==0 and session.time==0. and session.sensors.samples==0
    checks["source_manifest_attached"]=session.research_sources==identity()
    if not all(checks.values()):
        raise ValueError(checks)
    return dict(passed=True,checks=checks,physics_steps=0)
if __name__=="__main__":
    print(preflight())

from pathlib import Path
import sys,json,time,hashlib
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v13_joint.runtime import JointSession,source_identity,METHODS
from session import jsonable
from task_contract import build_task_request
from task_trajectory import unknown_future_point
rows=[]
with patch('mujoco.mj_step',side_effect=AssertionError('physics forbidden in zero gate')) as forbidden:
    for method in METHODS:
        s=JointSession(method=method,scenario='nominal',seed=617901,split='robustness',motion_mode='hold')
        assert s.time==0 and s.tick==0 and not s.trace and not s.plant.substep_records
        truth,raw,obs,common,observation_ms=s.observe_public()
        actual=s.plant.reference_snapshot().values.copy()
        s.task_request=build_task_request(0.,'hold',actual[18:],actual[18:],12.,s.limits,s.budget,tracking_tolerances=[.01,.015,.015],motion_preview=unknown_future_point(0.,actual[18:]),motion_reference_bounds=s.motion_reference_bounds,sensor_characteristics=s.sensor_characteristics)
        desired,pred,debug=s._execute_joint(obs,'hold')
        assert debug['mode']=='holding' and not debug['pressure_prediction_available']
        assert s.controller.native.__class__.__name__=='PreparedValueEffort21'
        first=s.controller.decision_evidence[-1]
        assert first['native']['diagnostics']['prepared_value_final_full_recheck']['passed']
        s.plant.begin_release()
        actual=s.plant.reference_snapshot().values.copy()
        s.task_request=s._public_task(0.,'release',actual[18:],actual[18:],3.5,True)
        desired,pred,debug=s._execute_joint(obs,'release')
        last=s.controller.decision_evidence[-1]
        assert last['native']['diagnostics']['prepared_value_final_full_recheck']['passed']
        assert last['qp']['status']=='converged'
        assert debug['holding_load_task_inactive'] and np.array_equal(desired,np.zeros(5))
        assert s.plant.actor21.encode(s.plant.reference_snapshot()).values.tobytes()==s.plant.engine_snapshot().values.tobytes()
        assert s.time==0 and s.tick==0 and not s.trace and not s.plant.substep_records
        rows.append(dict(method=method,holding_native=first['native']['status'],release_native=last['native']['status'],release_qp=last['qp']['status'],generation=s.plant.mode_generation,decision_ms=debug['solver_timing_ms']['whole_decision'],prepared_full_recheck=True))
    assert forbidden.call_count==0
value=dict(passed=True,rows=rows,source_sha256=source_identity(METHODS),physics_steps=0,scope='Actual constructors/public sampling/holding-without-contact and release decisions/native inverse/sole21 commit, zero integration; no contact trajectory outcome')
out=ROOT/'review/v13_joint_runtime_root_zero_v5.json'
with out.open('x',encoding='utf-8') as f:json.dump(jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(passed=True,rows=rows,source_count=len(value['source_sha256']),physics_steps=0),ensure_ascii=False))

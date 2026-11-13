"""Pure algebra/QP counterexample; never constructs a hand or simulation."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import sys,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from coordinator import JointCoordinator
from coordinator_model import PressureForecast
from force_model import ForceResponse

B=np.eye(5)*100.
cal=dict(force_matrix=B,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02))
joint=JointCoordinator(cal,np.full(5,5.),2.)
normal=np.array([.21,5.5,.21,.21,.21])
state=SimpleNamespace(normal=normal.copy(),alpha=np.zeros(5),latent_alpha=np.zeros(5),closure_min=np.full(5,-.02),closure_max=np.full(5,.02),
    valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),
    contact=np.ones(5,bool),mu=np.full(5,.45),confidence=np.ones(5),shear_magnitude=np.zeros(5),
    normal_wrench_map=np.r_[np.tile([[0.],[0.],[1.]],(1,5)),np.zeros((3,5))],
    finger_non_normal_by_finger=np.zeros((5,6)),auxiliary_wrench=np.zeros(6),auxiliary_valid=True,
    auxiliary_support_uncertainty=.2,load=1.2,up=np.array([0.,0.,1.]),force_gain=np.full(5,100.),
    slip_speed=0.,force_model=ForceResponse(B,.02))
joint.state=state
joint.model.snapshot=PressureForecast(normal,np.zeros(5),np.zeros(5),np.zeros(5),np.full(5,50),np.zeros((5,3)),np.zeros((5,3)),False)
joint.set_motion_request([.1,0.,0.],[.1,0.,0.])
joint._solve()
h=joint.g.horizon;P,c=state.force_model.action_prediction(normal,h,joint.g.decision_period,
    state.contact_model_active,alpha=state.alpha,latent=state.latent_alpha,execution_dt=.005)
raw=(P@joint.previous_plan[:5*h]+c)[:5]
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),raw_affine_pressure=raw.tolist(),
    displayed_clipped_pressure=joint.predicted.tolist(),raw_sum=float(raw.sum()),
    clipped_sum=float(joint.predicted.sum()),budget=joint.budget,
    reported_budget_violation=joint.debug['predicted_budget_violation_N'],
    reported_pressure_violation=joint.debug['predicted_pressure_violation_N'],
    qp_status=joint.debug['qp_status'],qp_residual=joint.debug['qp_primal_residual'],
    counterexample_found=bool(raw.min() < -1e-3 and joint.predicted.sum()>joint.budget+.1 and joint.debug['predicted_budget_violation_N']<.01),
    sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('coordinator.py','coordinator_model.py')},
    scope='Synthetic permitted-coordinate QP only; no hand, dynamics, task performance or physical feasibility claim. Demonstrates affine negative-pressure credit versus clipped output semantics.')
out=ROOT/'review/independent_joint_budget_counterexample.json'
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

"""Pure no-physics candidate and information-boundary counterexamples."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import json
import numpy as np

spec=importlib.util.spec_from_file_location('short_predictor',Path(__file__).with_name('predictor.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def verify():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def unavailable(name,call):
        try:call()
        except m.PredictionUnavailable:check(name,True)
        else:raise AssertionError(name)
    K=np.zeros((101,25,8));t=np.arange(101)*.005
    for axis in range(8):K[:,:,axis]=np.minimum(t[:,None]/.02,1.)*(axis+1)*np.arange(1,26)[None,:]
    ft=np.arange(6)*.02;u=np.zeros((6,8));u[1:,0]=.004;q=np.arange(25,dtype=float);v=np.arange(25)*.1
    A=m.predict_velocity(K,q,v,ft,u,[.02]);check('A_current_actual_velocity_plus_finite_ramp',np.allclose(A['prediction25'][0],q+.02*v+.004*np.arange(1,26)))
    zeros=np.zeros((6,8));A=m.predict_velocity(K,q,v,ft,zeros,[.02]);check('constant_future_reference_keeps_nonzero_past_memory_nuisance_velocity',np.allclose(A['prediction25'][0],q+.02*v))
    check('no_past_reference_parameter','uses_past_reference' in A and A['uses_past_reference'] is False)
    unavailable('missing_complete_velocity_unavailable',lambda:m.predict_velocity(K,q,None,ft,u,[.02]))
    bad=v.copy();bad[-1]=np.nan;unavailable('missing_lift_velocity_not_zero',lambda:m.predict_velocity(K,q,bad,ft,u,[.02]))
    sample=SimpleNamespace(raw_joint_names=tuple('rh_J'+str(i) for i in range(24)),q25=q,velocity25=None)
    unavailable('RobotJointSample_missing_lift_velocity_unavailable',lambda:m.predict_robot_sample(sample,K,ft,u,[.02]))
    qt=np.arange(41)*.005;history=q[None,:]+qt[:,None]*v[None,:]
    B=m.predict_position_ols(K,qt,history,ft,zeros,[.02]);check('B_linear_motion_exact',np.allclose(B['prediction25'][0],q+(.2+.02)*v))
    check('B_fixed_120ms_window',abs(B['past_times_used_s'][0]-.08)<1e-12 and len(B['past_times_used_s'])==25)
    check('B_OLS_weights_affine_sum',np.allclose(B['past_observation_weights'].sum(axis=1),1.))
    unavailable('B_one_observation_rank_unavailable',lambda:m.predict_position_ols(K,[0.],history[:1],ft,zeros,[.02]))
    unavailable('B_unknown_sampling_gap_unavailable',lambda:m.predict_position_ols(K,[0.,.01],history[:2],ft,zeros,[.02]))
    future_q=history.copy();prefix=future_q[:21].copy();before=m.predict_position_ols(K,qt[:21],prefix,ft,u,[.02])['prediction25']
    future_q[21:]+=1e6
    after=m.predict_position_ols(K,qt[:21],future_q[:21],ft,u,[.02])['prediction25'];check('future_q_pollution_does_not_change_prefix_prediction',np.array_equal(before,after))
    future_v=np.tile(v,(41,1));before=m.predict_velocity(K,q,future_v[20],ft,u,[.02])['prediction25'];future_v[21:]+=1e6
    after=m.predict_velocity(K,q,future_v[20],ft,u,[.02])['prediction25'];check('future_velocity_pollution_does_not_change_current_prediction',np.array_equal(before,after))
    u_late=u.copy();u_late[2:]+=100
    check('future_command_beyond_horizon_not_backdated',np.array_equal(m.future_response(K,ft,u,[.02]),m.future_response(K,ft,u_late,[.02])))
    unavailable('arbitrary5ms_future_not_declared20ms_ramp',lambda:m.future_response(K,np.arange(21)*.005,np.zeros((21,8)),[.02]))
    unavailable('unscored_horizon_rejected',lambda:m.future_response(K,ft,u,[.025]))
    unavailable('future_schedule_short_rejected',lambda:m.future_response(K,ft[:2],u[:2],[.1]))
    return dict(passed=True,new_physics=False,checks=checks)


if __name__=='__main__':print(json.dumps(verify(),indent=2))

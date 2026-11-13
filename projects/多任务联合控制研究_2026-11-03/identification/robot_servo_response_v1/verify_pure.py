"""Independent-of-physics algebra/clock fixtures for offline identification."""
from pathlib import Path
import importlib.util
import numpy as np

spec=importlib.util.spec_from_file_location('servo_identify',Path(__file__).with_name('identify.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def verify():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    grid=np.arange(101)*.005
    K=np.empty((101,25,8))
    for o in range(25):
        for i in range(8):K[:,o,i]=(o+1)*(i+1)*np.minimum(grid/.02,1.)
    t=np.array([0.,.02,.04]);u=np.zeros((3,8));u[1:,0]=.004
    check('zero_at_origin',np.array_equal(m.effect(K,t,u,0.),np.zeros(25)))
    check('quarter_finite_ramp',np.allclose(m.effect(K,t,u,.005),.001*np.arange(1,26)))
    check('one20ms_response_no_second_fast_filter',np.allclose(m.effect(K,t,u,.02),.004*np.arange(1,26)))
    u_future=u.copy();u_future[2]=.8
    check('future_input_cannot_enter_past',np.array_equal(m.effect(K,t,u,.01),m.effect(K,t,u_future,.01)))
    check('multiple_intervals_sum',np.allclose(m.effect(K,t,np.vstack([np.zeros(8),np.eye(8)[0]*.004,np.eye(8)[0]*.006]),.04),.006*np.arange(1,26)))
    residual=3.+grid[:,None]*np.arange(1,26)[None,:]
    forecast,w=m.residual_forecast(grid[:25],residual[:25],.18)
    check('causal_linear_residual_exact',np.allclose(forecast,3.+.18*np.arange(1,26)))
    check('weights_affine_unbiased',abs(w.sum()-1)<1e-12)
    forecast,w=m.residual_forecast(grid[:1],residual[:1],.1)
    check('single_initial_state_no_future_zero',np.array_equal(forecast,residual[0]))
    zero=dict(q=np.tile(np.arange(25),(101,1)))
    branches={}
    for i in range(8):
        for sign,label in ((1,'plus'),(-1,'minus')):
            d=sign*(.004 if sign>0 else .003);ref=np.zeros((101,8));ref[1:,i]=d*np.minimum(grid[1:]/.02,1)
            branches[f'micro_{i}_{label}']=dict(available=True,reference=ref,q=zero['q']+K[:,:,i]*d)
    fitted,plus,minus,training,side,missing=m.fit_micro(branches,zero)
    check('actual_signed_unequal_denominators',np.allclose(fitted,K))
    branches['micro_2_plus']=dict(available=False)
    fitted,plus,minus,*_=m.fit_micro(branches,zero)
    check('one_sided_not_zero_gain',not plus[2] and minus[2] and np.allclose(fitted[:,:,2],K[:,:,2]))
    branches['micro_2_minus']=dict(available=False)
    fitted,plus,minus,training,side,missing=m.fit_micro(branches,zero)
    check('entire_missing_column_nan_not_fake_zero',missing==[2] and np.isnan(fitted[:,:,2]).all())
    errors=np.zeros((3,25));errors[:,24]=.001;errors[:,0]=.01
    summary=m.score(errors)
    check('angular_and_metric_errors_separate',summary['angular_rad']['max_axis_rmse']==.01 and summary['lift_m']['rmse']==.001)
    check('no_pressure_gate_reuse','pressure' not in str(summary))
    return checks


if __name__=='__main__':
    import json
    print(json.dumps(dict(passed=True,checks=verify()),indent=2))

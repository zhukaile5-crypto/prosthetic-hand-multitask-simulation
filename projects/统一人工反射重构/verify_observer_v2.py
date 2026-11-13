"""Causal fixed-lag observation and no-contact/recontact mechanism checks.

Synthetic local mechanics only: these checks do not demonstrate comparative
MuJoCo task performance. All truth signals stay in this verifier.
"""
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
from controllers import Gains, TactileState, make_controller, METHODS

ROOT = Path(__file__).resolve().parent


def observation(t, alpha, measured, age, geometry=True):
    angle = np.arange(5)*2.*np.pi/5.
    direction = -np.column_stack((np.cos(angle), np.sin(angle), np.zeros(5)))
    if not geometry:
        direction[:] = 0.
    return SimpleNamespace(time=t, dt=.005, normal=np.broadcast_to(measured, (5,)).copy(),
        tangential=np.tile([.2, 0.], (5, 1)), slip_local=np.zeros((5, 2)),
        contact_pos=-.03*direction, contact_normal=direction,
        tangent_basis=np.stack((np.tile([0.,0.,1.],(5,1)),np.cross(direction,[0.,0.,1.])),axis=1),
        joint=np.full(5, alpha), joint_velocity=np.zeros(5), alpha=np.full(5,alpha),
        valid=np.ones(5,bool), age=age, phase='HOLDING', support_axis=np.array([0.,0.,1.]),
        force_gain=np.full(5,14.25), force_tau=.005, closure_min=np.zeros(5),
        closure_max=np.full(5,1.55), nominal_force_weights=np.ones(5),load_valid=False)


def frequency_case(frequency, delay):
    times = np.arange(0., 6., .005)
    alpha = .50+.008*np.sin(2.*np.pi*frequency*times)
    force = np.zeros(len(times))
    force[0] = 14.25*(alpha[0]-.333)
    a = np.exp(-.005/.005)
    for k in range(1,len(times)):
        force[k] = a*force[k-1]+(1.-a)*14.25*(alpha[k]-.333)
    rng = np.random.default_rng(411917)
    state = TactileState(Gains())
    observed, estimated, old_ema = [], [], []
    ema = 0.
    for k,t in enumerate(times):
        index = max(0,k-round(delay/.005))
        measurement = force[index]+rng.normal(0,.025)
        f = observation(t,alpha[k],measurement,min(delay,t))
        state.update(f,.005)
        ema += (1.-np.exp(-.005/.03))*(measurement-ema)
        observed.append(measurement);estimated.append(state.normal.mean());old_ema.append(ema)
    select = times >= 1.
    estimate, ema = np.array(estimated),np.array(old_ema)
    return dict(frequency_hz=frequency,delay=delay,
        aligned_observer_rmse=float(np.sqrt(np.mean((estimate[select]-force[select])**2))),
        delayed_ema_rmse=float(np.sqrt(np.mean((ema[select]-force[select])**2))))


def contact_lifecycle():
    rows = []
    for name in METHODS:
        controller = make_controller(name,[5.]*5,22.)
        phantom = False;search = False
        for k in range(100):
            f = observation(k*.005,.005*k,0.,.04,False);f.phase='CLOSING'
            action = controller.step(f,'CLOSING')
            phantom |= np.any(controller.state.normal > 1e-10) or np.any(controller.state.contact)
            search |= np.any(action.delta_closure > 0.)
        for k in range(100):
            f = observation(.5+k*.005,.50,1.6,.04,True)
            controller.step(f,'HOLDING')
        contacted = bool(np.count_nonzero(controller.state.contact)==5)
        for k in range(100):
            f = observation(1.+k*.005,.50,0.,.04,False)
            controller.step(f,'HOLDING')
        lost_reset = bool(not np.any(controller.state.contact_model_active)
                          and np.all(controller.state.normal==0.))
        for k in range(100):
            f = observation(1.5+k*.005,.65,1.2,.04,True)
            controller.step(f,'HOLDING')
        reacquired = bool(np.count_nonzero(controller.state.contact)==5)
        release = controller.step(observation(2.,.65,1.2,.04,True),'RELEASING','release')
        released = bool(np.all(release.delta_closure<0.) and np.all(controller.desired==0.))
        rows.append(dict(method=name,no_phantom_contact=not phantom,closing_search=bool(search),
                         observed_contact=contacted,loss_resets_model=lost_reset,
                         observed_reacquisition=reacquired,user_release=released))
    return rows


if __name__ == '__main__':
    frequencies = [frequency_case(f,d) for f in (1.,4.,8.) for d in (.01,.06)]
    lifecycle = contact_lifecycle()
    checks = dict(no_contact_contact_loss_reacquisition_release=all(
        all(v for k,v in row.items() if k!='method') for row in lifecycle),
        action_alignment_reduces_known_model_phase_error=all(
            row['aligned_observer_rmse'] < row['delayed_ema_rmse'] for row in frequencies))
    report = dict(scope='Causal calibrated-local synthetic verification, no task-performance claim',
                  checks=checks,passed=all(checks.values()),frequencies=frequencies,lifecycle=lifecycle)
    (ROOT/'review'/'observer_v2_verification.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)

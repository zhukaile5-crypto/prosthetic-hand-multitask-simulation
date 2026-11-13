"""Pure sign/clock/reference counterexamples, no runtime or simulator import."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh evidence required')
    checks=[]
    def check(name,x):
        if not x:raise AssertionError(name)
        checks.append(name)
    source=(ROOT/'v10/robot_contact_space.py').read_text(encoding='utf-8')
    check('actual projection row is inward normal@J','row=normal@k.jacobian3x25' in source)
    check('actual separation proxy has opposite sign','separation=-float(normal@(k.point_world-reference))' in source)
    J=np.zeros((3,25));J[0,0]=1.;n=np.array([1.,0,0]);error=-2.;tau=J.T@n*error
    check('unload produces negative inward generalized effort',tau[0]==-2.)
    outward=-n
    check('SDF outward uses negative transpose sign',np.array_equal(-J.T@outward*error,tau))
    dq=np.linspace(-.01,.01,25)
    check('virtual work identity',abs(float(tau@dq)-error*float(n@(J@dq)))<1e-15)
    weights=np.array([.7,.3]);allocated=sum(w*J.T@n*error for w in weights)
    check('shared finger force task allocated once',np.array_equal(allocated,tau))
    duplicate=sum(J.T@n*error for _ in weights)
    check('duplicated full finger error is a real factor2 error',np.array_equal(duplicate,2*tau))
    rate=.8;dt=.005;whole=.02*rate;one=float(np.clip(whole,-rate*dt,rate*dt));four=4*float(np.clip(whole/4,-rate*dt,rate*dt))
    check('20ms total cannot be issued as one5ms native delta',one<whole and four==whole)
    q=.1;v=.03;kp=40.;kv=1.5;u0=.11;correction=.02
    E=lambda u:kp*(u-q)-kv*v
    u1=u0+correction/kp;u2=u1+correction/kp
    check('same error repeatedly adds actual-reference feedback effort',abs(E(u2)-E(u0)-2*correction)<1e-14)
    check('nativeD cancels once in fixedqv incremental reference match',abs(E(u1)-E(u0)-correction)<1e-14)
    observed=5.;desired=4.;baseline=observed
    check('pure force-error baseline not absolute force added twice',baseline+(desired-observed)==desired and baseline+desired!=desired)
    motor_bias=.02481276;lift_kp=1500.
    check('lift bias twice is37N-scale extra internal effort',abs(lift_kp*motor_bias-37.21914)<1e-10)
    native=(FOUNDATION/'hand_plant.py').read_text(encoding='utf-8')
    check('native alpha cap1p55 remains actual implementation authority','self.closure_max=np.minimum(np.array(upper),1.55)' in native)
    check('native execution rate is peractual dt','bounded=np.clip(delta,-.8*dt,.8*dt)' in native)
    check('native lift command is clamped to task lift target','self.lift_ref=float(np.clip(lift,0.,self.config.lift_target))' in native)
    result=dict(schema='v12_integration_conventions_pure_v1',passed=True,checks=checks,new_physics=False,
        source_sha256={str(x):sha(x) for x in [ROOT/'v12/effort_tracker.py',ROOT/'v11/robot_servo_effort.py',ROOT/'v10/robot_contact_space.py',FOUNDATION/'hand_plant.py',Path(__file__)]},
        algebra_counterexamples=dict(one_tick_realization_of20ms_delta=one,correct4tick_sum=four,
            requested20ms_delta=whole,twice_repeated_fixed_error_effort_change=E(u2)-E(u0),double_lift_bias_extra_N=lift_kp*motor_bias),
        scope='No pressure response, motion, stability or safety is proved. Fixedqv equations only expose required integration semantics.')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,checks=len(checks),out=str(out),sha256=sha(out)),ensure_ascii=False))

if __name__=='__main__':main()

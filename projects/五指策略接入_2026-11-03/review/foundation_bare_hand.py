"""No-object effort/encoder calibration of the common articulated mechanism."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hand_plant import HandPlant
from foundation_config import FoundationConfig

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='review/foundation_bare_hand_v2');args=parser.parse_args()
    destination=ROOT/args.out
    destination.mkdir(exist_ok=False)
    config=FoundationConfig(object_enabled=False)
    p=HandPlant(config);rows=[];previous=None;filtered_acc=0.
    for tick in range(3201):
        t=p.time
        if t<1.:alpha=0.
        elif t<4.:alpha=1.1*(t-1.)/3.
        elif t<13.:alpha=1.1
        else:alpha=1.1*max(0.,1-(t-13.)/3.)
        if t<5.:lift=0.
        elif t<8.:lift=.1*(t-5.)/3.
        elif t<11.:lift=.1
        else:lift=.1*max(0.,1-(t-11.)/3.)
        p.set_closure(np.full(5,alpha));p.set_motion(lift)
        raw=p.raw_sensors();v=raw['lift_velocity']
        acc=0. if previous is None else (v-previous)/config.control_dt
        previous=v;filtered_acc+=(1-np.exp(-config.control_dt/.025))*(acc-filtered_acc)
        prior=raw['bare_hand_weight']+raw['known_hand_inertia']*filtered_acc+raw['lift_damping']*v
        normspeed=float(np.linalg.norm(raw['joint_velocity']))
        pressure=sum(x['normal'] for x in raw['patches'] if x['digit']>=0)
        rows.append([t,raw['motor_effort'],prior,filtered_acc,v,normspeed,raw['lift'],pressure])
        if tick<3200:p.advance()
    a=np.asarray(rows);residual=a[:,1]-a[:,2]
    quiet=(np.abs(a[:,3])<.05)&(np.abs(a[:,4])<.005)&(a[:,5]<.02)&(a[:,6]>.03)
    bias=float(np.median(residual[quiet]));valid=(np.abs(a[:,3])<2.)&(a[:,6]>.01)
    error=residual-bias
    result=dict(config=config.to_dict(),model_sha256=p.model_sha256,effort_bias_N=bias,
        load_error_bound_N=float(max(.1,np.max(np.abs(error[valid]))+.05)),
        valid_error_p95_N=float(np.quantile(np.abs(error[valid]),.95)),valid_samples=int(valid.sum()),
        quiet_samples=int(quiet.sum()),zero_object_skin_force_max=float(a[:,7].max()),
        warning_count=int(sum(int(x.number) for x in p.data.warning)),
        calibration_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('hand_plant.py','foundation_config.py')},
        scope='No object mass used: own CAD hand weight, lift inertia/damping and motor effort only; bound is empirical development coverage.')
    (destination/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(destination/'raw.npz',time=a[:,0],motor_effort=a[:,1],handload_prior=a[:,2],
        acceleration=a[:,3],lift_velocity=a[:,4],joint_speed=a[:,5],lift=a[:,6],skin_force=a[:,7],valid=valid,quiet=quiet,error=error)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()

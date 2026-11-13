"""Actual archived public-stream software equivalence, no plant/physics."""
from pathlib import Path
from dataclasses import fields
import json,sys,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
from contracts import SensorFrame
from controller_adapter import create_controller
from v9.controller import create_joint

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frame(value):
    core={x.name for x in fields(SensorFrame)}
    payload={k:np.asarray(v) if isinstance(v,list) else v for k,v in value.items() if k in core}
    f=SensorFrame(**payload)
    for k,v in value.items():
        if k not in core:setattr(f,k,np.asarray(v) if k in ('executed_motion_reference','motion_reference_rate_limits') else v)
    return f

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();dest=ROOT/a.out
    if dest.exists():raise ValueError('Fresh replay evidence required')
    base=ROOT/'output/development_v8_counterexamples_v1/runs/nominal__ratio__610212'
    m=json.loads((base/'metadata.json').read_text(encoding='utf-8'))
    r=create_controller('ratio',m['calibration'],m['limits'],m['budget'],m['gains'])
    c=create_joint(m['calibration'],m['limits'],m['budget'],m['gains'])
    with np.load(base/'trace.npz',allow_pickle=False) as archive:
        time=archive['time'];obs=archive['observation_json'];tasks=archive['task_request_json']
    differences=[];interventions=[];clocks=[];state_difference=0.;maxdelta=0.
    for i in np.flatnonzero((time>=7.3-1e-9)&(time<12.-1e-9)):
        f=frame(json.loads(str(obs[i])));f2=frame(json.loads(str(obs[i])))
        if time[i]<7.5-1e-9:r.state.update(f,.005);c.state.update(f2,.005);continue
        task=json.loads(str(tasks[i]));c.set_task_request(task)
        x=r.step(f);y=c.step(f2)
        state_difference=max(state_difference,float(np.max(abs(r.state.normal-c.state.normal))))
        diff=float(np.max(abs(x.delta_closure-y.delta_closure)));maxdelta=max(maxdelta,diff)
        if diff>0.:differences.append(dict(time=float(time[i]),maxdelta=diff,debug=c.debug))
        if not c.use_nominal_subinterval:interventions.append(float(time[i]))
        clocks.append(float(time[i]))
    result=dict(schema='v9-archived-healthy-public-software-replay-v1',physics_steps=0,plant_constructions=0,
        scope='Same actual archived nominal baseline observations/robot history. Software wiring/proposal equivalence only, not a counterfactual V9 physical rollout.',
        source={n:sha(ROOT/n) for n in ('v9/controller.py','v9/feedback.py','v9/runtime.py')},
        trace_sha256=sha(base/'trace.npz'),policy_ticks=len(clocks),state_difference_max=state_difference,
        action_difference_max=maxdelta,nonexact_action_ticks=len(differences),projected_ticks=len(interventions),
        exact_unique_baseline_state=state_difference==0.,all_actions_exact=maxdelta==0.,
        first_difference=differences[0] if differences else None,intervention_times=interventions)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source','first_difference','intervention_times')},ensure_ascii=False))

if __name__=='__main__':main()

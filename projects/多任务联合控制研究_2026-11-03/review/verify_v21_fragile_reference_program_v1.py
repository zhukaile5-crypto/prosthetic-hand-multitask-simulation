"""Check the holding programme with archived bounds and synthetic typed receipts only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from identification import v21_fragile_reference_program_v1 as program
from v13_a1.codec import EnginePort21, Reference21, REFERENCE_AXES21
from v13_joint.plant import ExecutionReceipt21
from v14_coord.design import reference_rates


ORIGIN=9.280
DT=program.CONTROL_PERIOD_S
ARCHIVE=ROOT/'review/v21_stage_attribution_v1.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def synthetic_receipt(request,applied=None,*,generation=0,mode='holding',corrupt_engine=False):
    values=request.values if applied is None else applied
    actual=Reference21(values,request.time_s)
    bias=.004  # An arbitrary stand-in for the frozen writer's encode operation.
    engine=actual.values.copy()
    engine[18]+=bias
    if corrupt_engine:
        engine[18]+=bias
    return ExecutionReceipt21(request,actual,EnginePort21(engine,request.time_s,bias),
                              mode,generation,not np.array_equal(request.values,actual.values),False)


def faults(pair, lower, upper):
    t=pair.origin_tick*DT
    initial=Reference21(pair.carrier21,t)
    gate=lambda: program.HoldingProgramGate(pair,'combined')
    step=lambda g: g.select(initial,lower,upper,mode='holding',generation=pair.generation)
    test={}
    g=gate()
    desired=step(g)
    try:
        step(g)
    except RuntimeError:
        test['requires_receipt_before_second_selection']=True
    else:
        test['requires_receipt_before_second_selection']=False
    g.accept(synthetic_receipt(desired,initial.values,generation=pair.generation))
    test['changed_actual_cancels']=g.status=='cancelled_receipt_mismatch'
    g=gate()
    desired=step(g)
    g.accept(synthetic_receipt(desired,generation=pair.generation,corrupt_engine=True))
    test['double_bias_cancels']=g.status=='cancelled_receipt_mismatch'
    g=gate()
    test['mode_change_disables_selection']=g.select(initial,lower,upper,mode='release',
        generation=pair.generation+1) is None and g.status=='cancelled_mode_or_generation'
    g=gate()
    invalid=lower.copy()
    invalid[0]=initial.values[0]
    test['reduced_authority_cancels_without_clipping']=g.select(initial,invalid,upper,mode='holding',
        generation=pair.generation) is None and g.status=='cancelled_fresh_authority'
    g=gate()
    test['late_tick_cancels']=g.select(Reference21(initial.values,t+DT),lower,upper,mode='holding',
        generation=pair.generation) is None and g.status=='cancelled_missing_tick_or_changed_current'
    restricted=initial.values.copy()
    restricted[0]=.2
    try:
        program.build_pair(Reference21(restricted,t),lower,upper,generation=pair.generation)
    except ValueError as exc:
        test['missing_open_authority_unexposed']='unexposed' in str(exc)
    else:
        test['missing_open_authority_unexposed']=False
    assert all(test.values())
    return test


def replay(record):
    trace=ROOT/record['metrics']['full_V21']['trace']
    assert sha(trace)==record['metrics']['full_V21']['trace_sha256']
    start=program.tick(ORIGIN)
    sl=slice(start,start+len(program.WAVE))
    with np.load(trace,allow_pickle=False) as a:
        frames=[json.loads(str(s)) for s in a['observation_json'][sl]]
        debug=[json.loads(str(s)) for s in a['debug_json'][sl]]
    f0=frames[0]
    assert tuple(f0['robot_actuator_axes21'][:18])==REFERENCE_AXES21[:18]
    initial=Reference21(f0['robot_reference21'],f0['time'])
    lower,upper=np.array(f0['reference_lower21']),np.array(f0['reference_upper21'])
    pair=program.build_pair(initial,lower,upper,generation=f0['mode_generation'])
    outputs={}
    for arm in program.ARMS:
        g=program.HoldingProgramGate(pair,arm)
        previous=initial.values
        rows=[]
        for i,(frame,d) in enumerate(zip(frames,debug)):
            t=frame['time']
            assert abs(t-(ORIGIN+i*DT))<1e-9 and abs(frame['sample_time']-(t-.010))<1e-9
            assert d['mode']=='holding'
            low,high=np.array(frame['reference_lower21']),np.array(frame['reference_upper21'])
            chosen=g.select(Reference21(previous,t),low,high,mode=d['mode'],generation=frame['mode_generation'])
            assert chosen is not None, (record['context'],arm,i,g.status)
            step=reference_rates('holding')*DT
            # Independently repeat the original holding plant's numerical clip.
            actual=np.clip(chosen.values,np.maximum(low,previous-step),np.minimum(high,previous+step))
            assert np.array_equal(chosen.values,actual)
            g.accept(synthetic_receipt(chosen,actual,generation=frame['mode_generation']))
            previous=actual
            rows.append(dict(issue_s=t,acquired_s=frame['sample_time'],desired=chosen.values.tolist(),
                             simulated_clip_result=actual.tolist()))
        assert g.status=='complete' and np.array_equal(previous,initial.values)
        outputs[arm]=rows
    x,y=(np.array([q['simulated_clip_result'] for q in outputs[a]]) for a in program.ARMS)
    off=[i for i in range(21) if i not in (0,9,10)]
    assert np.array_equal(x[:,off],y[:,off]) and np.array_equal(x[-1],y[-1])
    assert np.allclose(y-x,np.array(program.WAVE)[:,None]*pair.peak_delta21,atol=1e-12,rtol=0)
    return dict(context=record['context'],trace_sha256=sha(trace),generation=pair.generation,
                carrier21=list(pair.carrier21),peak_delta21=list(pair.peak_delta21),alpha_reduction=list(pair.alpha_reduction),
                assigned_axis_names=[REFERENCE_AXES21[i] for i in (0,9,10)],
                checks=faults(pair,lower,upper),synthetic_arm_outputs=outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out
    a=read(ARCHIVE)
    r=[replay(q) for q in a['rows'] if q['context'][0]=='fragile_conflict']
    assert len(r)==4
    artifact=dict(schema='v21_fragile_reference_program_offline_gate_v1',software_gate_passed=True,
                  physics_steps=0,real_execution_receipts=0,runtime_integrated=False,origin_s=ORIGIN,
                  ramp_ticks=program.RAMP_TICKS,hold_ticks=program.HOLD_TICKS,
                  arms=list(program.ARMS),rate_utilization=program.RATE_UTILIZATION,rows=r,
                  source_sha256={x.name:sha(x) for x in (ARCHIVE,Path(program.__file__),Path(__file__))})
    payload=json.dumps(artifact,indent=2,ensure_ascii=False,allow_nan=False)
    with out.open('x',encoding='utf-8') as f:
        f.write(payload+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),contexts=len(r)),ensure_ascii=False))


if __name__=='__main__':
    main()

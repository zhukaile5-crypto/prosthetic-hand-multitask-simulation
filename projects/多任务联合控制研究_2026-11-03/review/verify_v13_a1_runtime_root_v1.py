"""Zero-integration A1 factory/runtime and exact initial-public-stream checks."""
from pathlib import Path
import argparse,json,hashlib,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
from hand_plant import HandPlant
from v12_native.runtime import V12NativeSession,source_identity as parent_identity
from v13_a1.runtime import V13A1Session,source_identity,METHODS
from v13_a1.plant import A1Plant
from session import jsonable

def exact(x,y):
    a,b=np.asarray(x),np.asarray(y)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=ROOT/args.out;assert not out.exists();before=source_identity(METHODS);checks=[];calls={'steps':0,'old_writer':0}
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    def no_step(*a,**kw):calls['steps']+=1;raise AssertionError('No integration in zero-step gate')
    def no_old(*a,**kw):calls['old_writer']+=1;raise AssertionError('A1 called original robot writer')
    step_saved={key:getattr(mujoco,key) for key in ('mj_step','mj_step1','mj_step2')}
    old_write=HandPlant._apply_ctrl
    for key in step_saved:setattr(mujoco,key,no_step)
    try:
        for context in (dict(scenario='nominal',seed=617901,motion_mode='hold'),dict(scenario='collision',seed=617902,motion_mode='hold'),dict(scenario='fragile_conflict',seed=617902,motion_mode='transport')):
            for method in METHODS:
                label=str((context,method));kwargs=dict(context,method=method,split='robustness')
                original=V12NativeSession(**kwargs)
                HandPlant._apply_ctrl=no_old
                try:candidate=V13A1Session(**kwargs)
                finally:HandPlant._apply_ctrl=old_write
                check(label+' A1 exact factory',type(candidate.plant) is A1Plant and candidate.preparation.plant is candidate.plant)
                check(label+' no integration or trace',candidate.time==0 and candidate.tick==0 and not candidate.trace and not candidate.plant.substep_records)
                check(label+' first actual21 write at reset',candidate.plant.actor21.write_count==1)
                check(label+' same native controller',type(original.controller) is type(candidate.controller))
                for key in ('qpos','qvel','ctrl','alpha'):
                    value=(original.plant.data.ctrl if key=='ctrl' else original.plant.alpha if key=='alpha' else getattr(original.plant.data,key))
                    actual=(candidate.plant.data.ctrl if key=='ctrl' else candidate.plant.alpha if key=='alpha' else getattr(candidate.plant.data,key))
                    check(label+' initial '+key,exact(actual,value))
                check(label+' true model hash',candidate.plant.model_sha256==original.plant.model_sha256)
                check(label+' config',candidate.config==original.config)
                for key in ('normal','skin_normal','palm','hand_other','all_hand_object','support','relative','rotation','lift','penetration','world_penetration'):
                    check(label+' initial physics '+key,exact(candidate.plant.truth()[key],original.plant.truth()[key]))
                a=candidate.sensors.sample(candidate.plant.raw_sensors());b=original.sensors.sample(original.plant.raw_sensors())
                check(label+' first complete common sensor',json.dumps(jsonable(vars(a)),sort_keys=True)==json.dumps(jsonable(vars(b)),sort_keys=True))
                check(label+' sensor rng',json.dumps(candidate.sensors.rng.bit_generator.state,sort_keys=True)==json.dumps(original.sensors.rng.bit_generator.state,sort_keys=True))
                check(label+' task/sensor contracts',candidate.motion_reference_bounds==original.motion_reference_bounds and candidate.sensor_characteristics==original.sensor_characteristics and candidate.actual_ratio_task_gains==original.actual_ratio_task_gains)
        check('never old writer in candidate',calls['old_writer']==0);check('zero integration',calls['steps']==0)
        check('candidate sources unchanged',source_identity(METHODS)==before)
        parent=parent_identity(METHODS);check('parent80 contained and unchanged',len(parent)==80 and all(before.get(k)==v for k,v in parent.items()))
    finally:
        for key,value in step_saved.items():setattr(mujoco,key,value)
        HandPlant._apply_ctrl=old_write
    result=dict(schema='v13_a1_root_runtime_zero_step_v1',passed=True,checks=checks,forbidden_calls=calls,physics_steps=0,source_sha256=before,source_count=len(before),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Nine actual constructors and first public sample, not full physical equivalence or new strategy benefit')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=True,checks=len(checks),source_count=len(before),forbidden_calls=calls)))
if __name__=='__main__':main()

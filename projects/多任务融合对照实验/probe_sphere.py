"""Development-only common grasp geometry search; no method-dependent placements."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import mujoco
import run_experiments as run
import physics

original_init = physics.Plant.__init__
current_x = .075


def placed_init(self,scenario,p,timestep=.001):
    original_init(self,scenario,p,timestep)
    if scenario.shape == 'sphere':
        j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'object_free')
        self.data.qpos[self.model.jnt_qposadr[j]]=current_x
        mujoco.mj_forward(self.model,self.data)
        self.start_position=self.data.xpos[self.object].copy()


if __name__ == '__main__':
    physics.Plant.__init__ = placed_init
    base = next(s for s in run.SCENARIOS if s.name == 'sphere')
    results = []
    for size in (.70,.85,1.):
        for x in (.05,.065,.08,.095):
            current_x = x
            run.SCENARIOS[run.SCENARIOS.index(next(s for s in run.SCENARIOS if s.name=='sphere'))] = replace(base,size=size,duration=8.5)
            r=run.run_episode(dict(scenario='sphere',method='constant',seed=1001))
            results.append(dict(size=size,x=x,setup=r['setup_ok'],safe=r['safe_completed'],
                                initial_peak=r['initial_force_peak_N'],position=r['final_position'],
                                force=r['final_force'],grasp_time=r['grasp_time_s']))
            print(json.dumps(results[-1]),flush=True)
    out=physics.ROOT/'output/sphere_geometry_probe.json'
    out.write_text(json.dumps(results,indent=2),encoding='utf-8')

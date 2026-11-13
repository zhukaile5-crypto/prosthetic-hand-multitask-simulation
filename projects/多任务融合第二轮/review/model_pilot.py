"""Independent public-model repair pilots; never used for formal conclusions."""
from pathlib import Path
import sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import mujoco
OLD = Path(__file__).resolve().parents[2]/'多任务融合对照实验'
sys.path.insert(0,str(OLD))
from physics import Plant, SensorPipeline, DT
import physics as phys
from scenarios import SCENARIOS, parameters, schedule
from controllers import make_controller, Gains

def run(job):
    variant, scen, seed, method, timestep = job
    selected=json.loads((OLD/'selected_gains.json').read_text(encoding='utf-8'))
    scenario=next(s for s in SCENARIOS if s.name==scen)
    p=parameters(scenario,seed)
    phys.OPEN[0]=np.array([.2,.55,.1,.15,.35] if variant.startswith('compliant') else [.2,1.15,.1,.2,1.15])
    plant=Plant(scenario,p,timestep)
    high = variant in ('hard','hard_slow','hard_gain','discrete_hard')
    if variant in ('noslip0','discrete','discrete_hard'):
        plant.model.opt.noslip_iterations=0
        plant.model.opt.tolerance=1e-10
    if variant in ('discrete','discrete_hard'):
        plant.model.opt.integrator=mujoco.mjtIntegrator.mjINT_DISCRETE
    for gid in range(plant.model.ngeom):
        if plant.model.geom_contype[gid]:
            if not variant.endswith('_original'):
                plant.model.geom_solref[gid]=[max(.002,p['solref']*(.1 if high else .25)),1.]
                plant.model.geom_solimp[gid]=[.999,.9999,.0005,.5,2.] if high else [.99,.999,.001,.5,2.]
    if 'effort' in variant:
        for ids in plant.actuators:
            plant.model.actuator_forcerange[ids]*=(.1 if '10' in variant else .3)
    if variant.startswith('primitive'):
        for gid in plant.geom_finger:
            if plant.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                plant.model.geom_type[gid]=mujoco.mjtGeom.mjGEOM_BOX
                plant.model.geom_dataid[gid]=-1
    joint=mujoco.mj_name2id(plant.model,mujoco.mjtObj.mjOBJ_JOINT,'object_free')
    adr=plant.model.jnt_qposadr[joint]
    if scenario.shape!='sphere': plant.data.qpos[adr]=.027+.028*p['size']
    mujoco.mj_forward(plant.model,plant.data)
    plant.start_position=plant.data.xpos[plant.object].copy()
    pipeline=SensorPipeline(scenario,p,seed)
    controller=make_controller(method,scenario.limit,scenario.budget,Gains(**selected[method]))
    md=0.;peak=0.;ex=0.;setup=False;dropped=False;since=None;unsafe=False;anchor=None;maxshift=0.;maxct=None
    for tick in range(round(scenario.duration/DT)):
        t=plant.time; tr=plant.truth(); obs=pipeline.sample(tr,t); env=schedule(scenario,p,t)
        command='release' if scenario.release_at is not None and t>=scenario.release_at-1e-8 else 'hold'
        if command=='release':plant.request_release()
        desired=controller.step(obs,plant.phase,command)
        before=plant.data.ctrl.copy(); before_alpha=plant.alpha.copy()
        plant.execute(desired,obs,np.asarray(scenario.limit),env)
        if variant=='hard_slow' and plant.phase in ('CLOSING','LIFTING'):
            for indices in plant.actuators:
                plant.data.ctrl[indices]=np.clip(plant.data.ctrl[indices],before[indices]-.5*DT,before[indices]+.5*DT)
        if variant=='hard_gain' and plant.phase not in ('RELEASING','FAILED','DROPPED'):
            plant.alpha=before_alpha+.5*(plant.alpha-before_alpha)
            plant.set_actuators(env['wrist'])
        for c in plant.data.contact[:plant.data.ncon]:
            if plant.object_geom in (c.geom1,c.geom2):
                gid=int(c.geom2 if c.geom1==plant.object_geom else c.geom1)
                if gid in plant.geom_finger or gid in plant.palm_geoms:
                    if -float(c.dist)>md:
                        md=-float(c.dist);maxct=[t,int(gid),mujoco.mj_id2name(plant.model,mujoco.mjtObj.mjOBJ_BODY,int(plant.model.geom_bodyid[gid]))]
        fn=float(np.max(tr['normal']));peak=max(peak,fn)
        ex+=float(np.maximum(0,tr['normal']-np.asarray(scenario.limit)).sum())*DT
        over=bool((tr['normal']>np.asarray(scenario.limit)+.25).any() or tr['normal'].sum()>scenario.budget+.5)
        if over:
            if since is None: since=t
            if t-since>=.03-1e-8:unsafe=True
        else: since=None
        if t>=8:
            if anchor is None:anchor=tr['relative'].copy();setup=plant.phase=='HOLDING' and tr['position'][2]-plant.start_position[2]>.04 and np.count_nonzero(tr['normal']>.2)>=2
            shift=np.linalg.norm(tr['relative']-anchor);maxshift=max(maxshift,shift)
            if setup and command=='hold' and (shift>.08 or tr['position'][2]<plant.start_position[2]+.020):dropped=True
        plant.advance(env)
        mujoco.mj_forward(plant.model,plant.data)
    return dict(variant=variant,scenario=scen,seed=seed,method=method,timestep=timestep,penetration_mm=md*1000,peak_force=peak,excess_Ns=ex,setup=bool(setup),dropped=bool(dropped),unsafe=bool(unsafe),shift_mm=maxshift*1000,phase=plant.phase,max_contact=maxct)

if __name__=='__main__':
    variants = sys.argv[1].split(',') if len(sys.argv)>1 else ('hard','hard_slow','hard_gain')
    jobs=[(variant,scen,seed,method,.001) for variant in variants for scen,seed in [('nominal',3001),('friction_drop',3007),('fragile_conflict',3001)] for method in ('ratio','fusion')]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as executor:
        for f in as_completed([executor.submit(run,j) for j in jobs]):
            row=f.result();results.append(row);print(json.dumps(row),flush=True)
    (Path(__file__).parent/('model_variants_'+','.join(variants)+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')

"""Development-only foundation identification and timestep checks.

Runs no confirmation comparison and does not select an upper coordinator.
"""
import json
from pathlib import Path
import time
import argparse
import numpy as np
from physics import Plant, SensorPipeline, DT, SUBSTEP_COLUMNS
from scenarios import SCENARIOS, parameters, schedule

ROOT = Path(__file__).resolve().parent


def identify_local_model(seed=410601):
    """Central differences around an actually held development cylinder.

    The two signs start from identical copied qpos/qvel/controller states. This
    measures the complete passive coupling, rather than labelling a guess fit.
    """
    import copy
    s=next(s for s in SCENARIOS if s.name=='nominal')
    p=parameters(s,seed,False)
    plant,sensor=Plant(s,p),SensorPipeline(s,p,seed)
    for _ in range(round(6./DT)):
        truth=plant.truth();obs=sensor.sample(truth,plant.time)
        delta=np.clip((np.array([2.3,1.6,1.6,1.6,1.6])-obs.normal)*.0015,-.003,.003)
        env=schedule(s,p,plant.time);plant.execute(delta,obs,env);plant.advance(env)
    anchor=plant.truth();results=[];wave=[]
    for digit in range(5):
        for sign in (-1,1):
            sim=copy.deepcopy(plant)
            pipe=SensorPipeline(s,{**p,'force_noise':0.,'delay':0.},seed)
            samples=[]
            for tick in range(61):
                tr=sim.truth();obs=pipe.sample(tr,sim.time)
                u=np.zeros(5)
                if tick<4:u[digit]=sign*.003
                samples.append([tick*DT,*tr['normal']])
                env=schedule(s,p,sim.time);sim.execute(u,obs,env);sim.advance(env)
            a=np.asarray(samples);wave.append(a)
            results.append(dict(digit=digit,sign=sign,delta_closure=sign*.012,
                baseline_normal=anchor['normal'].tolist(),
                settled_normal=a[-20:,1:].mean(axis=0).tolist(),
                force_change=(a[-20:,1:].mean(axis=0)-anchor['normal']).tolist(),
                initial_alpha=plant.alpha.tolist(),final_alpha=sim.alpha.tolist()))
    jac=np.zeros((5,5))
    for i in range(5):
        minus,plus=results[2*i:2*i+2]
        realized=float(plus['final_alpha'][i]-minus['final_alpha'][i])
        jac[:,i]=(np.array(plus['settled_normal'])-np.array(minus['settled_normal']))/realized if realized>1e-8 else np.nan
    tau=[];rmse=[]
    for i in range(5):
        difference=wave[2*i+1][:,i+1]-wave[2*i][:,i+1]
        times=wave[2*i][:,0]
        amplitude=float(np.mean(difference[-20:]))
        candidates=np.geomspace(.005,.3,150)
        errors=[float(np.mean((difference-amplitude*(1-np.exp(-np.maximum(times-.01,0)/v)))**2)) for v in candidates]
        k=int(np.argmin(errors));tau.append(float(candidates[k]));rmse.append(float(np.sqrt(errors[k])))
    return dict(seed=seed,scenario='nominal',linearization_normal=anchor['normal'].tolist(),
        full_jacobian_N_per_closure=jac.tolist(),diagonal_gain=np.diag(jac).tolist(),
        probes=results,time_constant_s=tau,paired_step_model_rmse_N=rmse,
        scope='held development cylinder; short actuator-coordinate steps with other commands fixed; not universal force identification'),np.asarray(wave)


def bare_hand_calibration(timestep=.0005):
    """Known fixture/encoder motor-load residual without any grasped object."""
    s=next(s for s in SCENARIOS if s.name=='nominal');p=parameters(s,410401,False)
    plant=Plant(s,p,timestep)
    # This is an explicit no-object foundation apparatus, never a benchmark trial.
    plant.model.geom_contype[plant.object_geom]=0
    plant.model.geom_conaffinity[plant.object_geom]=0
    import mujoco
    mujoco.mj_forward(plant.model,plant.data)
    pipe=SensorPipeline(s,{**p,'force_noise':0.,'delay':0.},410401)
    samples=[]
    for _ in range(round(17./DT)):
        t=plant.time
        if t>=1. and plant.phase=='CLOSING':
            plant.phase='LIFTING';plant.lift_started=t
        truth=plant.truth();obs=pipe.sample(truth,t)
        target=np.array([.70,.95,.95,.95,.95]) if t<12. else np.zeros(5)
        delta=np.clip(target-plant.alpha,-.002,.002)
        env=schedule(s,p,t)
        if 9.<=t<12.:
            env['lift_delta']=.015*np.sin(2*np.pi*.7*(t-9.))
            env['wrist']=.12*np.sin(2*np.pi*.45*(t-9.))
        samples.append([t,truth['lift'],truth['lift_velocity'],obs.motor_effort,
            obs.handload_prior,pipe.encoder_acceleration,pipe.load_estimate,obs.load_uncertainty,
            float(obs.load_valid),float(obs.load_confidence)])
        plant.execute(delta,obs,env,'release' if t>=12. else 'hold');plant.advance(env)
    a=np.asarray(samples)
    valid=a[:,8]>.5;static=(a[:,0]>=6)&(a[:,0]<8.)
    residual=a[:,3]-a[:,4]
    filtered=a[:,6]
    result=dict(timestep=timestep,bare_hand_weight_N=float(truth['bare_hand_weight']),
        known_inertia_kg=float(truth['known_hand_inertia']),damping_Ns_per_m=float(truth['lift_damping']),
        measured_static_effort_N=float(np.median(a[static,3])),
        static_offset_N=float(np.median(residual[static])),
        static_residual_max_N=float(np.abs(residual[static]).max()),
        valid_filtered_residual_p95_N=float(np.percentile(np.abs(filtered[valid]),95)),
        valid_filtered_residual_max_N=float(np.abs(filtered[valid]).max()),
        uncertainty_coverage=float(np.mean(np.abs(filtered[valid])<=a[valid,7])),
        interpretation='Zero object load; own-hand CAD priors checked against no-object close/lift/transport/release motor response.')
    return result,a


def probe(name='nominal', seed=410001, timestep=.0005, duration=10., release=False,
          target=(3.5,2.,2.,2.,2.), thumb_delay=0.,p_overrides=None):
    s = next(s for s in SCENARIOS if s.name == name)
    p = parameters(s, seed, False)
    p.update(p_overrides or {})
    plant, sensor = Plant(s,p,timestep), SensorPipeline(s,p,seed)
    trace = [];geometry=[]
    started = time.perf_counter()
    for _ in range(round(duration/DT)):
        truth = plant.truth()
        obs = sensor.sample(truth, plant.time)
        measured = np.nan_to_num(obs.normal, nan=0.)
        # Fixed common force response probe, not a proposed benchmark algorithm.
        delta = np.clip((np.asarray(target)-measured)*.0015, -.003, .003)
        if plant.time < thumb_delay:
            delta[0] = 0.
        if release and plant.time >= 8.:
            delta[:] = -.003
        env = schedule(s,p,plant.time)
        plant.execute(delta, obs, env, 'release' if release and plant.time >= 8. else 'hold')
        trace.append((plant.time, *truth['normal'], *truth['relative'], *truth['relvel'],
            *truth['joint'], *plant.alpha, truth['lift'], *truth['position']))
        geometry.append(np.r_[truth['rotation'].ravel(),truth['contact_pos'].ravel(),
            truth['contact_normal'].ravel(),truth['tangential'].ravel(),truth['slip_local'].ravel()])
        plant.advance(env)
    a, sub = np.array(trace), np.array(plant.substep_records)
    post = a[a[:,0] >= max(4.,duration-2.)]
    result = dict(scenario=name,seed=seed,timestep=timestep,duration=duration,
        physical_parameters=p,
        wall_seconds=time.perf_counter()-started,phase=plant.phase,
        initial_penetration_m=plant.initial_penetration,
        peak_force_N=float(sub[:,1:6].max()), peak_total_N=float(sub[:,1:6].sum(axis=1).max()),
        peak_penetration_m=float(sub[:,SUBSTEP_COLUMNS.index('penetration_m')].max()),
        peak_world_penetration_m=float(sub[:,SUBSTEP_COLUMNS.index('world_penetration_m')].max()),
        peak_skin_force_N=float(sub[:,[SUBSTEP_COLUMNS.index(f'skin_normal_{i}_N') for i in range(5)]].max()),
        final_force=a[-1,1:6].tolist(), mean_post_force=post[:,1:6].mean(axis=0).tolist(),
        force_std=post[:,1:6].std(axis=0).tolist(), final_relative=a[-1,6:9].tolist(),
        reference_relative=plant.reference_relative.tolist(),
        relative_drift_m=float(np.linalg.norm(a[-1,6:9]-plant.reference_relative)),
        final_lift=float(a[-1,22]), final_position=a[-1,23:26].tolist(),
        support_final_N=float(sub[-1,12]), alpha_final=plant.alpha.tolist(),
        finite=bool(np.isfinite(a).all() and np.isfinite(sub).all()),
        substep_count=int(len(sub)))
    result['final_rotation']=np.array(geometry)[-1,:9].reshape(3,3).tolist()
    result['final_contacts']=int(np.count_nonzero(a[-1,1:6]>.20))
    result['final_relvel_norm']=float(np.linalg.norm(a[-1,9:12]))
    result['commanded_lift']=plant.task_lift_height
    return result,a,sub,np.asarray(geometry)


def foundation_suite(folder):
    """Pre-declared model checks across two contact families and parameter range."""
    import hashlib
    source=hashlib.sha256((ROOT/'physics.py').read_bytes()).hexdigest()
    cases=[]
    for name in ('nominal','box_heavy','sphere'):
        for family,solref in (('nominal',.020),('compliant',.030)):
            for step in (.0005,.00025):
                key=f'{name}_{family}_{step:g}'
                r,a,sub,g=probe(name,seed=410201,timestep=step,duration=17.,
                    p_overrides={'solref':solref})
                r.update(case=key,family=family,physics_sha256=source)
                cases.append(r)
                np.savez_compressed(folder/(key+'.npz'),trace=a,substeps=sub,geometry=g,
                    substep_columns=np.array(SUBSTEP_COLUMNS))
                (folder/'suite.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(r,ensure_ascii=False),flush=True)
    for name in ('nominal','box_heavy','sphere'):
        s=next(s for s in SCENARIOS if s.name==name)
        for variant, massscale, frictionscale, sizescale in (
                ('light_easy',.80,1.15,.97),('heavy_slippery',1.20,.80,1.03)):
            p=parameters(s,410211,False)
            override=dict(mass=p['mass']*massscale,friction=p['friction']*frictionscale,
                size=round(p['size']*sizescale,4))
            key=f'{name}_{variant}'
            r,a,sub,g=probe(name,seed=410211,timestep=.0005,duration=17.,p_overrides=override)
            r.update(case=key,family='range',physics_sha256=source)
            cases.append(r)
            np.savez_compressed(folder/(key+'.npz'),trace=a,substeps=sub,geometry=g,
                substep_columns=np.array(SUBSTEP_COLUMNS))
            (folder/'suite.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(r,ensure_ascii=False),flush=True)


def grid_probe(job):
    name,x,y=job
    s=next(s for s in SCENARIOS if s.name==name);p=parameters(s,410301,False)
    p['pregrasp_offset']=[x,y,0.]
    plant=Plant(s,p)
    key=f'{name}_{x:+.3f}_{y:+.3f}'
    if plant.initial_penetration>.0005:
        return dict(case=key,scenario=name,offset=[x,y,0.],skipped='initial_penetration',
            initial_penetration_m=plant.initial_penetration),None,None,None
    r,a,sub,g=probe(name,seed=410301,duration=17.,p_overrides={'pregrasp_offset':[x,y,0.]})
    r.update(case=key,offset=[x,y,0.])
    tail=a[:,0]>=12.
    active=(np.count_nonzero(a[:,1:6]>.2,axis=1)>=3)&(a[:,1]>.2)
    r['last5_contact_fraction']=float(np.mean(active[tail]))
    r['last5_max_speed']=float(np.linalg.norm(a[tail,9:12],axis=1).max())
    r['last5_displacement']=float(np.linalg.norm(a[-1,6:9]-a[np.flatnonzero(tail)[0],6:9]))
    r['potential_long_hold']=bool(r['last5_contact_fraction']==1. and r['last5_max_speed']<.01)
    return r,a,sub,g


def pregrasp_grid(folder):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    jobs=[(name,x,y) for name in ('box_heavy','sphere') for x in (-.006,0.,.006) for y in (-.012,0.,.012)]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(grid_probe,j) for j in jobs]
        for f in as_completed(futures):
            r,a,s,g=f.result();results.append(r)
            if a is not None:
                np.savez_compressed(folder/(r['case']+'.npz'),trace=a,substeps=s,geometry=g,
                    substep_columns=np.array(SUBSTEP_COLUMNS))
            (folder/'grid.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:r[k] for k in ('case','initial_penetration_m','potential_long_hold','last5_contact_fraction','last5_max_speed') if k in r}),flush=True)


def reference_probe(job):
    name,seed=job
    return probe(name,seed=seed,duration=17.)


def calibrate_reference(folder):
    """One shared reference from the predeclared 8.2--8.9 s development window."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import hashlib
    source=hashlib.sha256((ROOT/'physics.py').read_bytes()).hexdigest()
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(reference_probe,(name,seed)) for name in
            ('nominal','box_heavy','sphere') for seed in (410001,410002,410003)]
        for f in as_completed(futures):
            r,a,s,g=f.result();key=f'{r["scenario"]}__{r["seed"]}'
            window=(a[:,0]>=8.2-1e-9)&(a[:,0]<=8.9+1e-9)
            contact=(np.count_nonzero(a[:,1:6]>.2,axis=1)>=3)&(a[:,1]>.2)
            r['reference_window']=[8.2,8.9]
            r['window_support_fraction']=float(np.mean(contact[window]))
            r['window_valid']=bool(contact[window].all())
            r['window_relative_median']=np.median(a[window,6:9],axis=0).tolist()
            r['window_relative_offset']=(np.median(a[window,6:9],axis=0)-np.array(r['reference_relative'])).tolist()
            r['physics_sha256']=source
            results.append(r)
            np.savez_compressed(folder/(key+'.npz'),trace=a,substeps=s,geometry=g,
                substep_columns=np.array(SUBSTEP_COLUMNS))
            (folder/'reference_trials.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:r[k] for k in ('scenario','seed','window_valid','window_support_fraction','window_relative_offset','final_contacts','peak_force_N','peak_penetration_m')},ensure_ascii=False),flush=True)
    references={}
    for name,shape in (('nominal','cylinder'),('box_heavy','box'),('sphere','sphere')):
        cohort=[r for r in results if r['scenario']==name and r['window_valid']]
        references[shape]=dict(relative_offset=np.median([r['window_relative_offset'] for r in cohort],axis=0).tolist() if cohort else None,
            valid_seed_count=len(cohort),attempted_seed_count=3,window=[8.2,8.9],
            seeds=[r['seed'] for r in cohort],physics_sha256=source)
    (folder/'references.json').write_text(json.dumps(references,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(references,ensure_ascii=False),flush=True)


def box_candidate_probe(job):
    label,overrides=job
    s=next(s for s in SCENARIOS if s.name=='box_heavy');p=parameters(s,410501,False);p.update(overrides)
    plant=Plant(s,p)
    if plant.initial_penetration>.0005:
        return dict(case=label,skipped='initial_penetration',initial_penetration_m=plant.initial_penetration,
            overrides=overrides),None,None,None
    r,a,sub,g=probe('box_heavy',seed=410501,duration=17.,p_overrides=overrides)
    tail=a[:,0]>=12.;contact=(np.count_nonzero(a[:,1:6]>.2,axis=1)>=3)&(a[:,1]>.2)
    r.update(case=label,overrides=overrides,last5_contact_fraction=float(np.mean(contact[tail])),
        last5_max_speed=float(np.linalg.norm(a[tail,9:12],axis=1).max()))
    r['potential_long_hold']=bool(r['last5_contact_fraction']==1 and r['last5_max_speed']<.01)
    return r,a,sub,g


def box_candidates(folder):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    candidates=[]
    for j4 in (.9,1.15):
        for y in (-.012,.012):
            candidates.append((f'cube_j4{j4}_y{y}',dict(box_half=[.028,.028,.028],
                thumb_j4=j4,thumb_j5_open=-.7,pregrasp_offset=[0,y,0])))
    candidates += [('original_y12',dict(thumb_j4=1.15,thumb_j5_open=0.,pregrasp_offset=[0,.012,0])),
        ('small_hand_box',dict(box_half=[.023,.040,.025],thumb_j4=.9,thumb_j5_open=-.7,pregrasp_offset=[.002,.012,0]))]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(box_candidate_probe,j) for j in candidates]
        for f in as_completed(futures):
            r,a,s,g=f.result();results.append(r)
            if a is not None:np.savez_compressed(folder/(r['case']+'.npz'),trace=a,substeps=s,geometry=g,
                substep_columns=np.array(SUBSTEP_COLUMNS))
            (folder/'candidates.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:r[k] for k in ('case','initial_penetration_m','potential_long_hold','last5_contact_fraction','last5_max_speed') if k in r}),flush=True)


def box_orientation_candidates(folder):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    candidates=[(f'angle{angle}_thumb{j4}',dict(box_angle=np.deg2rad(angle),
        thumb_j4=j4,thumb_j5_open=-.7)) for angle in (45,90) for j4 in (.6,.9,1.15)]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(box_candidate_probe,j) for j in candidates]
        for f in as_completed(futures):
            r,a,s,g=f.result();results.append(r)
            if a is not None:np.savez_compressed(folder/(r['case']+'.npz'),trace=a,substeps=s,geometry=g,
                substep_columns=np.array(SUBSTEP_COLUMNS))
            (folder/'candidates.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:r[k] for k in ('case','initial_penetration_m','potential_long_hold','last5_contact_fraction','last5_max_speed') if k in r}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--names',nargs='+',default=['nominal','box_heavy','sphere'])
    ap.add_argument('--duration',type=float,default=10.)
    ap.add_argument('--timestep',type=float,default=.0005)
    ap.add_argument('--release',action='store_true')
    ap.add_argument('--tag',default='foundation_dev')
    ap.add_argument('--target',type=float,nargs=5,default=[3.5,2.,2.,2.,2.])
    ap.add_argument('--thumb-delay',type=float,default=0.)
    ap.add_argument('--identify',action='store_true')
    ap.add_argument('--suite',action='store_true')
    ap.add_argument('--grid',action='store_true')
    ap.add_argument('--bare-hand',action='store_true')
    ap.add_argument('--reference',action='store_true')
    ap.add_argument('--box-candidates',action='store_true')
    ap.add_argument('--box-orientations',action='store_true')
    args=ap.parse_args()
    folder=ROOT/'review'/args.tag
    folder.mkdir(parents=True,exist_ok=True)
    if args.box_orientations:
        box_orientation_candidates(folder)
        return
    if args.box_candidates:
        box_candidates(folder)
        return
    if args.reference:
        calibrate_reference(folder)
        return
    if args.bare_hand:
        r,a=bare_hand_calibration(args.timestep)
        (folder/'bare_hand.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(folder/'bare_hand.npz',trace=a,
            columns=np.array(['time','lift','lift_velocity','motor_effort','handload_prior',
                'encoder_acceleration','signed_filtered_load_residual','uncertainty','valid','confidence']))
        print(json.dumps(r,ensure_ascii=False),flush=True)
        return
    if args.grid:
        pregrasp_grid(folder)
        return
    if args.suite:
        foundation_suite(folder)
        return
    if args.identify:
        result,wave=identify_local_model()
        (folder/'identification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(folder/'identification.npz',wave=wave)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    results=[]
    for name in args.names:
        result,trace,sub,geometry=probe(name,timestep=args.timestep,duration=args.duration,release=args.release,
            target=args.target,thumb_delay=args.thumb_delay)
        results.append(result)
        np.savez_compressed(folder/(name+'.npz'), trace=trace, substeps=sub,geometry=geometry,
            substep_columns=np.array(SUBSTEP_COLUMNS))
        (folder/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

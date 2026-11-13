"""Development-only five-pad foundation, reference and local-model validation."""
import argparse,copy,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import mujoco
from physics import Plant,SensorPipeline,DT,SUBSTEP_COLUMNS
from scenarios import SCENARIOS,parameters,schedule

ROOT=Path(__file__).resolve().parent


def probe(name='nominal',seed=410001,timestep=.0005,duration=17.,release=False,
          target=(2.,2.,2.,2.,2.),p_overrides=None):
    s=next(s for s in SCENARIOS if s.name==name);p=parameters(s,seed,False);p.update(p_overrides or {})
    plant,pipe=Plant(s,p,timestep),SensorPipeline(s,p,seed);rows=[];geometry=[];started=time.perf_counter()
    for _ in range(round(duration/DT)):
        tr=plant.truth();obs=pipe.sample(tr,plant.time);measured=np.nan_to_num(obs.normal,nan=0.)
        delta=np.clip((np.asarray(target)-measured)*.0015,-.003,.003)
        if release and plant.time>=8.:delta[:]=-.003
        env=schedule(s,p,plant.time);plant.execute(delta,obs,env,'release' if release and plant.time>=8. else 'hold')
        rows.append([plant.time,*tr['normal'],*tr['relative'],*tr['relvel'],*tr['joint'],*plant.alpha,tr['lift'],*tr['position']])
        geometry.append(np.r_[tr['rotation'].ravel(),tr['contact_pos'].ravel(),tr['contact_normal'].ravel(),tr['tangential'].ravel(),tr['slip_local'].ravel()])
        plant.advance(env)
    a,sub,g=np.asarray(rows),np.asarray(plant.substep_records),np.asarray(geometry)
    tail=a[:,0]>=duration-2.;hold=sub[:,0]>=6.
    normal=sub[:,1:6];width=SUBSTEP_COLUMNS.index('penetration_m');dur=sub[:,SUBSTEP_COLUMNS.index('duration_s')]
    extra=np.maximum(0.,normal-np.asarray(s.limit)).sum(axis=1)
    result=dict(scenario=name,seed=seed,timestep=timestep,duration=duration,physical_parameters=p,
        physics_sha256=hashlib.sha256((ROOT/'physics.py').read_bytes()).hexdigest(),model_metadata=plant.model_metadata,
        wall_seconds=time.perf_counter()-started,phase=plant.phase,initial_penetration_m=plant.initial_penetration,
        peak_force_N=float(normal.max()),peak_total_N=float(normal.sum(axis=1).max()),
        peak_penetration_m=float(sub[:,width].max()),hold_peak_penetration_m=float(sub[hold,width].max()),
        force_exposure_Ns=float(extra@dur),post_force_mean=a[tail,1:6].mean(axis=0).tolist(),
        post_force_sd=a[tail,1:6].std(axis=0).tolist(),
        post_finger_tv_N_per_s=float(np.abs(np.diff(a[tail,1:6],axis=0)).sum()/(a[tail,0][-1]-a[tail,0][0])),
        final_force=a[-1,1:6].tolist(),final_relative=a[-1,6:9].tolist(),reference_relative=plant.reference_relative.tolist(),
        final_rotation=g[-1,:9].reshape(3,3).tolist(),final_lift=float(a[-1,22]),
        final_position=a[-1,23:26].tolist(),final_contacts=int(np.count_nonzero(a[-1,1:6]>.2)),
        final_speed=float(np.linalg.norm(a[-1,9:12])),final_support_N=float(sub[-1,SUBSTEP_COLUMNS.index('support_N')]),
        alpha_final=plant.alpha.tolist(),closure_min=plant.closure_min.tolist(),closure_max=plant.closure_max.tolist(),
        finite=bool(np.isfinite(a).all() and np.isfinite(sub).all()),warnings=sum(int(w.number) for w in plant.data.warning),
        substep_count=len(sub))
    return result,a,sub,g


def save_probe(folder,key,result,a,sub,g):
    np.savez_compressed(folder/(key+'.npz'),trace=a,substeps=sub,geometry=g,substep_columns=np.array(SUBSTEP_COLUMNS))
    (folder/(key+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


def run_job(job):
    return probe(**job)


def reference_suite(folder):
    jobs=[dict(name=n,seed=s,duration=17.) for n in ('nominal','box_heavy','sphere') for s in (410001,410002,410003)]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_job,j) for j in jobs]
        for f in as_completed(futures):
            r,a,sub,g=f.result();window=(a[:,0]>=8.2-1e-9)&(a[:,0]<=8.9+1e-9)
            contact=(np.count_nonzero(a[:,1:6]>.2,axis=1)>=3)&(a[:,1]>.2)
            r.update(window_valid=bool(contact[window].all()),window_support_fraction=float(np.mean(contact[window])),
                reference_window=[8.2,8.9],window_relative_offset=(np.median(a[window,6:9],axis=0)-np.array(r['reference_relative'])).tolist())
            results.append(r);save_probe(folder,f'{r["scenario"]}__{r["seed"]}',r,a,sub,g)
            print(json.dumps({k:r[k] for k in ('scenario','seed','window_valid','window_relative_offset','peak_force_N','peak_penetration_m')},ensure_ascii=False),flush=True)
    refs={}
    for n,shape in (('nominal','cylinder'),('box_heavy','box'),('sphere','sphere')):
        cohort=[r for r in results if r['scenario']==n and r['window_valid']]
        refs[shape]=dict(relative_offset=np.median([r['window_relative_offset'] for r in cohort],axis=0).tolist() if cohort else None,
            valid_seed_count=len(cohort),window=[8.2,8.9],seeds=[r['seed'] for r in cohort],physics_sha256=results[0]['physics_sha256'])
    (folder/'references.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(refs,ensure_ascii=False),flush=True)


def foundation_suite(folder):
    jobs=[]
    for name in ('nominal','box_heavy','sphere'):
        for family,solref in (('nominal',.020),('compliant',.030)):
            for step in (.0005,.00025):
                jobs.append(dict(name=name,seed=410801,timestep=step,duration=17.,p_overrides={'solref':solref,'contact_family':family}))
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_job,j) for j in jobs]
        for f in as_completed(futures):
            r,a,sub,g=f.result();family=r['physical_parameters']['contact_family'];key=f'{r["scenario"]}_{family}_{r["timestep"]:g}'
            r.update(case=key,family=family);results.append(r);save_probe(folder,key,r,a,sub,g)
            print(json.dumps({k:r[k] for k in ('case','peak_force_N','peak_penetration_m','final_contacts','final_speed','warnings')},ensure_ascii=False),flush=True)
    pairs=[]
    for n in ('nominal','box_heavy','sphere'):
        for family in ('nominal','compliant'):
            r=next(x for x in results if x['scenario']==n and x['family']==family and x['timestep']==.0005)
            s=next(x for x in results if x['scenario']==n and x['family']==family and x['timestep']==.00025)
            changes={k:abs(s[k]-r[k])/max(abs(r[k]),.01 if 'tv' in k else 1e-6) for k in ('peak_force_N','peak_total_N','peak_penetration_m','post_finger_tv_N_per_s')}
            accepted=bool(r['finite'] and s['finite'] and r['warnings']==s['warnings']==0 and r['final_contacts']>=3 and s['final_contacts']>=3
                and max(r['initial_penetration_m'],s['initial_penetration_m'])<=.0005
                and max(r['hold_peak_penetration_m'],s['hold_peak_penetration_m'])<=.003
                and changes['peak_force_N']<=.10 and changes['peak_penetration_m']<=.10
                and abs(s['post_finger_tv_N_per_s']-r['post_finger_tv_N_per_s'])<=.05+abs(r['post_finger_tv_N_per_s'])*.10)
            pairs.append(dict(scenario=n,family=family,relative_changes=changes,accepted=accepted))
    summary=dict(physics_sha256=results[0]['physics_sha256'],trial_count=len(results),pairs=pairs,
        foundation_accepted=all(p['accepted'] for p in pairs),scope='fixed-grasp/hold force probe; severe external collision and free-drop phases require separate results')
    (folder/'acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def identify_local_model(name='nominal',seed=410901):
    s=next(s for s in SCENARIOS if s.name==name);p=parameters(s,seed,False)
    plant,pipe=Plant(s,p),SensorPipeline(s,p,seed)
    for _ in range(round(6./DT)):
        tr=plant.truth();obs=pipe.sample(tr,plant.time);u=np.clip((2.-obs.normal)*.0015,-.003,.003)
        env=schedule(s,p,plant.time);plant.execute(u,obs,env);plant.advance(env)
    anchor=plant.truth();waves=[];details=[]
    for digit in range(5):
        for sign in (-1,1):
            sim=copy.deepcopy(plant);sensor=SensorPipeline(s,{**p,'force_noise':0.,'delay':0.},seed);rows=[]
            for tick in range(61):
                tr=sim.truth();obs=sensor.sample(tr,sim.time);u=np.zeros(5)
                if tick<4:u[digit]=sign*.003
                rows.append([tick*DT,*tr['normal']]);env=schedule(s,p,sim.time);sim.execute(u,obs,env);sim.advance(env)
            a=np.asarray(rows);waves.append(a)
            details.append(dict(digit=digit,sign=sign,final_alpha=sim.alpha.tolist(),settled_normal=a[-20:,1:].mean(axis=0).tolist()))
    jac=np.zeros((5,5));taus=[];errors=[]
    for i in range(5):
        minus,plus=details[2*i:2*i+2];realized=plus['final_alpha'][i]-minus['final_alpha'][i]
        jac[:,i]=(np.array(plus['settled_normal'])-np.array(minus['settled_normal']))/realized
        response=waves[2*i+1][:,i+1]-waves[2*i][:,i+1];t=waves[2*i][:,0];amp=response[-20:].mean()
        values=np.geomspace(.005,.3,150);loss=[np.mean((response-amp*(1-np.exp(-np.maximum(t-.01,0)/v)))**2) for v in values]
        k=int(np.argmin(loss));taus.append(float(values[k]));errors.append(float(np.sqrt(loss[k])))
    result=dict(scenario=name,seed=seed,physics_sha256=hashlib.sha256((ROOT/'physics.py').read_bytes()).hexdigest(),
        linearization_normal=anchor['normal'].tolist(),jacobian_N_per_closure=jac.tolist(),diagonal_gain=np.diag(jac).tolist(),
        time_constant_s=taus,paired_step_model_rmse_N=errors,probes=details,
        scope='central local steps in a held development grasp; cross-contact coupling remains in complete recorded Jacobian')
    return result,np.asarray(waves)


def bare_hand_calibration(timestep=.0005):
    s=next(s for s in SCENARIOS if s.name=='nominal');p=parameters(s,410401,False);plant=Plant(s,p,timestep)
    plant.model.geom_contype[plant.object_geom]=0;plant.model.geom_conaffinity[plant.object_geom]=0;mujoco.mj_forward(plant.model,plant.data)
    pipe=SensorPipeline(s,{**p,'force_noise':0.,'delay':0.},410401);rows=[]
    for _ in range(round(17./DT)):
        t=plant.time
        if t>=1. and plant.phase=='CLOSING':plant.phase='LIFTING';plant.lift_started=t
        tr=plant.truth();obs=pipe.sample(tr,t);target=np.full(5,.48) if t<12. else np.zeros(5);u=np.clip(target-plant.alpha,-.002,.002)
        env=schedule(s,p,t)
        if 9.<=t<12.:env['lift_delta']=.015*np.sin(2*np.pi*.7*(t-9.));env['wrist']=.12*np.sin(2*np.pi*.45*(t-9.))
        rows.append([t,tr['lift'],tr['lift_velocity'],obs.motor_effort,obs.handload_prior,pipe.encoder_acceleration,pipe.load_estimate,obs.load_uncertainty,float(obs.load_valid)])
        plant.execute(u,obs,env,'release' if t>=12. else 'hold');plant.advance(env)
    a=np.asarray(rows);valid=a[:,8]>.5;quiet=(a[:,0]>=6.)&(a[:,0]<8.);res=a[:,3]-a[:,4]
    result=dict(timestep=timestep,bare_hand_weight_N=tr['bare_hand_weight'],known_inertia_kg=tr['known_hand_inertia'],damping_Ns_per_m=tr['lift_damping'],
        static_residual_median_N=float(np.median(res[quiet])),static_residual_p95_N=float(np.percentile(np.abs(res[quiet]),95)),
        valid_filtered_residual_p95_N=float(np.percentile(np.abs(a[valid,6]),95)),valid_filtered_residual_max_N=float(np.abs(a[valid,6]).max()),
        uncertainty_coverage=float(np.mean(np.abs(a[valid,6])<=a[valid,7])),scope='no-object close/lift/transport/release execution calibration')
    return result,a


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--names',nargs='+',default=['nominal','box_heavy','sphere'])
    ap.add_argument('--duration',type=float,default=17.);ap.add_argument('--timestep',type=float,default=.0005);ap.add_argument('--release',action='store_true')
    ap.add_argument('--target',type=float,nargs=5,default=[2.]*5);ap.add_argument('--tag',default='five_pad_foundation')
    for opt in ('reference','suite','identify','bare-hand'):ap.add_argument('--'+opt,action='store_true')
    args=ap.parse_args();folder=ROOT/'review'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    if args.reference:return reference_suite(folder)
    if args.suite:return foundation_suite(folder)
    if args.identify:
        results=[]
        for name in args.names:
            r,w=identify_local_model(name);results.append(r);np.savez_compressed(folder/(name+'.npz'),waves=w)
        (folder/'identification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(results,ensure_ascii=False),flush=True);return
    if args.bare_hand:
        r,a=bare_hand_calibration(args.timestep);(folder/'bare_hand.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(folder/'bare_hand.npz',trace=a);print(json.dumps(r,ensure_ascii=False),flush=True);return
    results=[]
    for name in args.names:
        r,a,s,g=probe(name,timestep=args.timestep,duration=args.duration,release=args.release,target=args.target);results.append(r);save_probe(folder,name,r,a,s,g)
        print(json.dumps(r,ensure_ascii=False),flush=True)
    (folder/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()

"""Constant-reference physical probes, independent of either research policy."""
from pathlib import Path
import argparse,json,time,hashlib,copy
from dataclasses import asdict
import numpy as np
from foundation_config import FoundationConfig
from hand_plant import HandPlant,CommonPreparation,target_profile
from sensors import aggregate_patches

ROOT=Path(__file__).resolve().parent

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.floating,np.integer)):return x.item()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    return x

def constant_probe(config,duration=14.,release_at=10.,out=None,protocol=None):
    plant=HandPlant(config)
    preparation=CommonPreparation(plant)
    lift_start=None;trace=[];stable_state=None;release_supported=False
    configurations=[];velocities=[];controls=[];peak_patch=None;peak_patch_pen=0.
    aggregation_valid=[];aggregation_error=[];world_forces=[];patch_samples=[];references=[]
    began=time.perf_counter()
    for tick in range(round(duration/config.control_dt)+1):
        t=plant.time;truth=plant.truth()
        raw=plant.raw_sensors();measurement,diagnostic=aggregate_patches(raw['patches'])
        aggregation_valid.append(diagnostic['aggregation_valid']);aggregation_error.append(diagnostic['aggregation_error'])
        R=plant.data.xmat[plant.palm].reshape(3,3)
        object_force=sum((R@x['force'] for x in raw['patches'] if x['is_target']),np.zeros(3))
        world_forces.append(object_force)
        if 8.2-1e-9<=t<=8.9+1e-9 or any(abs(t-x)<config.control_dt*.1 for x in (0.,10.,10.5,11.,11.5)):
            patch_samples.append(dict(time=t,patches=raw['patches'],palm_rotation=R.copy(),object_force_world=object_force))
        if truth['penetration']>peak_patch_pen:
            peak_patch_pen=truth['penetration']
            active=[x for x in plant.raw_sensors()['patches'] if x['is_target']]
            peak_patch=dict(time=t,contacts=active)
        if t>=release_at-1e-9:
            if plant.release_started is None:
                release_supported=bool(np.count_nonzero(truth['normal']>.2)>=3 and truth['normal'][0]>.2 and truth['all_hand_object']>.2)
                plant.begin_release()
            delta=-np.minimum(np.maximum(0.,plant.alpha-plant.closure_min),.8*config.control_dt)
            # Four non-thumb fingers extend while the shared thumb-clearance path executes.
            plant.execute_delta(delta)
        else:
            target=preparation.step(truth['skin_normal'])
            lift_start=preparation.lift_started
        references.append(np.zeros(5) if t>=release_at-1e-9 else target.copy())
        row=[t,*truth['normal'],*truth['skin_normal'],truth['palm'],truth['hand_other'],truth['all_hand_object'],
             truth['support'],truth['penetration'],truth['world_penetration'],truth['position'][2],truth['lift'],
             *plant.alpha,truth['warning_count']]
        trace.append(row)
        configurations.append(truth['qpos']);velocities.append(truth['qvel']);controls.append(truth['ctrl'])
        if 7.5-1e-9<=t<7.6 and stable_state is None:stable_state=plant.clone_state()
        if tick<round(duration/config.control_dt):plant.advance()
    a=np.asarray(trace);sub=np.asarray(plant.substep_records)
    hold=(a[:,0]>=8.2)&(a[:,0]<=8.9);release=a[:,0]>=release_at
    support=((a[:,1:6]>.2).sum(axis=1)>=3)&(a[:,1]>.2)&(a[:,13]>.2)
    clearance=a[:,13]<=.2
    released=None;clear_since=None
    for i in np.where(release)[0]:
        if clearance[i]:
            if clear_since is None:clear_since=a[i,0]
            if a[i,0]-clear_since>=.2-1e-9:released=float(a[i,0]-release_at);break
        else:clear_since=None
    result=dict(config=asdict(config),duration=duration,release_at=release_at,
        lift_started=lift_start,held_support_fraction=float(np.mean(support[hold])),
        held_lift_mean=float(np.mean(a[hold,18])),held_object_lift=float(np.mean(a[hold,17])-plant.initial_position[2]),
        held_force_mean=np.mean(a[hold,1:6],axis=0),held_alpha=np.mean(a[hold,19:24],axis=0),
        force_peak=np.max(sub[:,2:7],axis=0),max_penetration=float(np.max(sub[:,-2])),
        warnings=int(np.max(a[:,-1])),release_supported=release_supported,release_latency=released,
        initial_penetration=plant.initial_penetration,hidden_ctrl_clip=plant.max_hidden_clip,
        model_sha256=plant.model_sha256,wall_seconds=time.perf_counter()-began,
        maximum_penetration_snapshot=peak_patch,force_peak_contacts=copy.deepcopy(plant.force_peak_contacts),
        initial_position=plant.initial_position,probe_kind='shared acquisition and holding force reference; no research policy',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py','sensors.py','contracts.py')},
        assets_provenance_sha256=hashlib.sha256((ROOT/'review/source_provenance.json').read_bytes()).hexdigest(),
        actual_world_object_force_static_mean=np.mean(np.asarray(world_forces)[hold],axis=0),
        static_aggregation_valid_fraction=np.mean(np.asarray(aggregation_valid)[hold],axis=0),
        static_aggregation_error_max=np.max(np.asarray(aggregation_error)[hold],axis=0))
    if protocol is not None:
        protocol=Path(protocol)
        result['protocol_path']=str(protocol.resolve())
        result['protocol_sha256']=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if out is not None:
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        source=out/'source_snapshot';source.mkdir()
        for name in result['source_sha256']:(source/name).write_bytes((ROOT/name).read_bytes())
        np.savez_compressed(out/'raw_probe.npz',trace=a,substeps=sub,time=a[:,0],normal=a[:,1:6],skin_normal=a[:,6:11],
            palm=a[:,11],hand_other=a[:,12],all_hand_object=a[:,13],support=a[:,14],penetration=a[:,15],
            world_penetration=a[:,16],object_z=a[:,17],lift=a[:,18],alpha=a[:,19:24],warnings=a[:,24],
            qpos=np.asarray(configurations),qvel=np.asarray(velocities),ctrl=np.asarray(controls),
            force_reference=np.asarray(references),aggregation_valid=np.asarray(aggregation_valid),
            aggregation_error=np.asarray(aggregation_error),object_force_world=np.asarray(world_forces),
            patch_samples_json=json.dumps(jsonable(patch_samples)),
            substep_time=sub[:,0],substep_duration=sub[:,1],substep_normal=sub[:,2:7],substep_skin_normal=sub[:,7:12],
            substep_all_hand_object=sub[:,14],substep_support=sub[:,15],substep_penetration=sub[:,16],
            metadata=json.dumps(jsonable(result)))
        (out/'summary.json').write_text(json.dumps(jsonable(result),ensure_ascii=False,indent=2),encoding='utf-8')
    return result,plant,stable_state,a,sub

def _response_rollout(plant,state,delta,span):
    plant.restore_state(state);base=plant.alpha.copy();plant.substep_records.clear()
    plant.set_closure(base+delta);actual=plant.alpha-base
    values=[]
    for tick in range(round(span/plant.config.control_dt)+1):
        values.append(np.r_[tick*plant.config.control_dt,plant.truth()['normal']])
        if tick<round(span/plant.config.control_dt):plant.advance()
    return np.asarray(values),actual

def response_probe(plant,state,amplitude=.008,span=.25):
    if state is None:raise ValueError('No stable holding checkpoint')
    baseline,_=_response_rollout(plant,state,np.zeros(5),span)
    curves=[];deltas=[]
    for digit in range(5):
        for direction in (-1.,1.):
            delta=np.zeros(5);delta[digit]=direction*amplitude
            curve,actual=_response_rollout(plant,state,delta,span)
            curves.append(curve);deltas.append(actual)
    curves=np.asarray(curves);deltas=np.asarray(deltas)
    times=curves[0,:,0];fit=times<=.080+1e-9
    differential=curves[:,:,1:]-baseline[None,:,1:]
    candidates=np.array([.005,.010,.020,.030,.050,.080,.120,.200])
    losses=[];matrices=[]
    for tau in candidates:
        h=1-np.exp(-times[fit]/tau)
        X=(deltas[:,None,:]*h[None,:,None]).reshape(-1,5)
        Y=differential[:,fit,:].reshape(-1,5)
        matrix=np.linalg.lstsq(X,Y,rcond=1e-10)[0].T
        matrices.append(matrix);losses.append(float(np.mean((X@matrix.T-Y)**2)))
    best=int(np.argmin(losses));tau=float(candidates[best]);matrix=matrices[best]
    predicted=baseline[None,:,1:]+(deltas@matrix.T)[:,None,:]*(1-np.exp(-times[None,:,None]/tau))
    residual=curves[:,:,1:]-predicted
    # Independent validation uses fresh multi-finger actions and a frozen-current
    # baseline; future zero-action measurements are never used in predictions.
    rng=np.random.default_rng(511337);patterns=[]
    for radius in (.02,.04):
        for _ in range(8):patterns.append(rng.uniform(-radius,radius,5))
    holdout=[];holdout_deltas=[];errors=[];initial=baseline[0,1:]
    horizons=np.array([.020,.040,.060]);indices=np.rint(horizons/plant.config.control_dt).astype(int)
    for delta in patterns:
        curve,actual=_response_rollout(plant,state,delta,.060)
        prediction=initial[None,:]+(matrix@actual)[None,:]*(1-np.exp(-horizons[:,None]/tau))
        errors.append(curve[indices,1:]-prediction);holdout.append(curve);holdout_deltas.append(actual)
    errors=np.asarray(errors)
    by_radius=[]
    for i,radius in enumerate((.02,.04)):
        error=errors[i*8:(i+1)*8]
        by_radius.append(dict(radius=radius,rmse=float(np.sqrt(np.mean(error**2))),
                              p95_absolute=float(np.quantile(np.abs(error),.95)),maximum=float(np.max(np.abs(error)))))
    result=dict(force_matrix=matrix,reference_alpha=state['alpha'],force_tau=tau,
        response_error_bound=np.maximum(.05,np.quantile(np.abs(errors[:8]),.95,axis=(0,1))),
        response_error_bound_by_radius={str(radius):np.maximum(.05,np.quantile(np.abs(errors[i*8:(i+1)*8]),.95,axis=(0,1))) for i,radius in enumerate((.02,.04))},
        execution_trust_radius=.02,
        fitted_time_constants=candidates,fit_mse=np.asarray(losses),
        amplitude=amplitude,span=span,fit_horizon=.080,actual_fit_deltas=deltas,
        holdout_by_radius=by_radius,holdout_horizons=horizons,
        calibration_time=state['time'],reference_force=initial,diagonal=np.diag(matrix),
        max_offdiagonal=float(np.max(np.abs(matrix-np.diag(np.diag(matrix))))),
        scope='Signed coupled local physical model; fitted only to +/-8e-3, tested on independent 2e-2/4e-2 actions. No future baseline in holdout prediction.')
    plant.response_extra=dict(baseline=baseline,fit_deltas=deltas,holdout=np.asarray(holdout),
                              holdout_deltas=np.asarray(holdout_deltas),holdout_errors=errors)
    return result,curves

def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',default='capsule');p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--solref',type=float,default=.020);p.add_argument('--out',required=True);p.add_argument('--duration',type=float,default=14.)
    p.add_argument('--response',action='store_true');p.add_argument('--matrix',action='store_true');p.add_argument('--protocol',type=Path);a=p.parse_args()
    if a.matrix:
        destination=ROOT/a.out;destination.mkdir(parents=True,exist_ok=False);rows=[]
        for shape in ('capsule','box','sphere'):
            for solref in (.008,.012):
                for timestep in (.0005,.00025):
                    config=FoundationConfig(shape=shape,solref=solref,timestep=timestep)
                    folder=destination/f'{shape}_s{solref:.3f}_dt{timestep:.4f}'
                    result,plant,state,trace,sub=constant_probe(config,14.,10.,folder,a.protocol)
                    if solref==.008 and timestep==.0005:
                        response,curves=response_probe(plant,state)
                        (folder/'response.json').write_text(json.dumps(jsonable(response),ensure_ascii=False,indent=2),encoding='utf-8')
                        np.savez_compressed(folder/'response_raw.npz',curves=curves,**plant.response_extra)
                    rows.append(dict(folder=str(folder),config=asdict(config),summary=result))
                    print(json.dumps(dict(shape=shape,solref=solref,dt=timestep,penetration=result['max_penetration'],
                        lift=result['held_object_lift'],supported=result['held_support_fraction'],release=result['release_latency']),ensure_ascii=False),flush=True)
        (destination/'manifest.json').write_text(json.dumps(jsonable(rows),ensure_ascii=False,indent=2),encoding='utf-8')
        return
    config=FoundationConfig(shape=a.shape,timestep=a.dt,solref=a.solref)
    result,plant,state,trace,sub=constant_probe(config,a.duration,10.,ROOT/a.out,a.protocol)
    if a.response:
        response,curves=response_probe(plant,state)
        (ROOT/a.out/'response.json').write_text(json.dumps(jsonable(response),ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(ROOT/a.out/'response_raw.npz',curves=curves,**plant.response_extra)
        result['response']=response
    print(json.dumps(jsonable(result),ensure_ascii=False))

if __name__=='__main__':main()

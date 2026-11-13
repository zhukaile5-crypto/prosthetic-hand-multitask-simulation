"""Whole-architecture benchmark: public task, local observations, independent truth.

New source/model changes invalidate the frozen confirmation manifest. Trials and
raw traces are checkpointed individually; resumption requires identical inputs.
"""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, hashlib, json, os, sys, time, platform
import numpy as np
import mujoco
from physics import Plant, SensorPipeline, DT, SUBSTEP_COLUMNS
from controllers import make_controller
from scenarios import SCENARIOS, parameters, schedule, evaluation_events, scenario_manifest
from evaluation import evaluate_trial

ROOT=Path(__file__).resolve().parent
METHODS=('fixed','ratio','independent','unified','unified_no_continuity','unified_no_geometry','unified_fixed_stage')
BENCHMARK_ID='mujoco_finite_mass_five_compliant_contacts_v1'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,p)
def hashes():
    return {name:sha(ROOT/name) for name in ('physics.py','controllers.py','evaluation.py','scenarios.py','run_experiments.py')}

def run_trial(job):
    scenario=next(x for x in SCENARIOS if x.name==job['scenario'])
    p=parameters(scenario,job['seed'],job['split']=='robustness')
    plant=Plant(scenario,p,job['timestep'])
    sensor=SensorPipeline(scenario,p,job['seed'])
    controller=make_controller(job['method'],scenario.limit,scenario.budget,job.get('gains',{}))
    initial=plant.truth()
    refs=job.get('references',{})
    offset=np.array(refs.get(scenario.shape,{}).get('relative_offset',[0.,0.,0.]))
    reference={'relative':(initial['relative']+offset).tolist(),
               'rotation':(np.asarray(initial['rotation'])@np.asarray(refs.get(scenario.shape,{}).get('rotation_offset',np.eye(3)))).tolist(),
               'absolute_z':float(initial['position'][2]+.070),
               'orientation_mode':'free',
               'axis':1 if scenario.shape=='cylinder' else 2}
    trace={k:[] for k in ('time','normal','skin_normal','shear','position','relative','rotation','relative_velocity','angular_velocity','support','palm','hand_other','all_hand_object','collision','penetration','observed_normal','desired','predicted_normal','prediction_at','actions','validity','phase','alpha','qpos','mocap_pos','motion_attained','lift_error','wrist_error')}
    observation_fields=('tangential','slip_local','contact_pos','contact_normal','tangent_basis','joint','joint_velocity','age','lift','motor_effort','load_estimate','load_uncertainty','load_valid')
    for field in observation_fields: trace['observed_'+field]=[]
    for field in ('estimated_normal','delayed_normal','observer_uncertainty','contact_model_active','support_reserve_slack','transport_paused','qp_status','trajectory_qp_status','control_ms'):trace[field]=[]
    start=time.perf_counter(); control_ms=[]
    established=None; arrival=None; failure_at=None; failure_reason=None; loss_since=None; motion_loss_since=None; establish_since=None
    cfg=job.get('evaluation_config',{})
    wrist_id=mujoco.mj_name2id(plant.model,mujoco.mjtObj.mjOBJ_JOINT,'wrist')
    wrist_qadr=int(plant.model.jnt_qposadr[wrist_id])
    for tick in range(round(scenario.duration/DT)+1):
        t=plant.time; true=plant.truth(); obs=sensor.sample(true,t)
        env=schedule(scenario,p,t)
        command='release' if scenario.release_at is not None and t>=scenario.release_at-1e-9 else 'hold'
        began=time.perf_counter_ns()
        action=controller.step(obs,plant.phase,command)
        control_ms.append((time.perf_counter_ns()-began)/1e6)
        # All full-object values below are evaluation-only and never modify plant.
        hand_goal=float(getattr(plant,'task_lift_height',.150))+env['lift_delta']
        lift_error=abs(float(true['lift'])-hand_goal)
        wrist_error=abs(float(plant.data.qpos[wrist_qadr])-env['wrist'])
        motion_attained=lift_error<=cfg.get('hand_motion_tolerance',.010) and wrist_error<=cfg.get('wrist_motion_tolerance',.035)
        lifted=motion_attained and true['position'][2]>=reference['absolute_z']-cfg.get('lift_tolerance',.005) and true['support']<=cfg.get('table_support_max',.05)
        touches=np.asarray(true['normal'])>cfg.get('contact_min',.2)
        contact=np.count_nonzero(touches)>=cfg.get('minimum_contacts',3) and all(touches[i] for i in cfg.get('required_fingers',[0]))
        if established is None:
            if lifted and contact:
                if establish_since is None: establish_since=t
                if t-establish_since>=cfg.get('establishment_dwell',.1)-1e-9: established=t
            else: establish_since=None
        if arrival is None and lifted: arrival=t
        if established is not None and command!='release' and failure_at is None:
            gross_loss=np.linalg.norm(true['relative']-reference['relative'])>.080 or true['position'][2]<initial['position'][2]+.020
            lost=gross_loss or not contact
            if lost:
                if loss_since is None: loss_since=t
                if t-loss_since>=cfg.get('support_loss_dwell',.100)-1e-9:
                    failure_at=t; failure_reason='loss_after_establishment' if gross_loss else 'support_contact_lost'
            else: loss_since=None
            if not motion_attained:
                if motion_loss_since is None: motion_loss_since=t
                if t-motion_loss_since>=cfg.get('transport_loss_dwell',.30)-1e-9 and failure_at is None:
                    failure_at=t; failure_reason='transport_reference_unmet'
            else: motion_loss_since=None
        if established is None and t>=cfg.get('establishment_deadline',8.)-1e-9 and failure_at is None:
            failure_at=t; failure_reason='establishment_failed'
        vals={'time':t,'normal':true['normal'],'skin_normal':true['tactile']['normal'],'shear':true['tangential'],
              'position':true['position'],'relative':true['relative'],'rotation':true['rotation'],
              'relative_velocity':true['relative_velocity'],'angular_velocity':true['angular_velocity'],
              'support':true['support'],'palm':true['palm'],'hand_other':true['hand_other'],
              'all_hand_object':true['all_hand_object'],'collision':true['collision'],'penetration':true['penetration'],
              'observed_normal':obs.normal,'desired':getattr(controller,'desired',np.zeros(5)),
              'predicted_normal':getattr(action,'predicted_normal',np.zeros(5)),
              'prediction_at':(t if command=='release' else float(controller.last_decision))+controller.g.decision_period,
              'actions':np.asarray(action),'validity':obs.valid,'phase':plant.phase,'alpha':plant.alpha,
              'qpos':plant.data.qpos,'mocap_pos':plant.data.mocap_pos,'motion_attained':motion_attained,'lift_error':lift_error,'wrist_error':wrist_error}
        for k,v in vals.items(): trace[k].append(np.asarray(v).copy())
        for field in observation_fields:
            trace['observed_'+field].append(np.asarray(getattr(obs,field,np.nan)).copy())
        trace['estimated_normal'].append(controller.state.normal.copy())
        trace['delayed_normal'].append(controller.state.delayed_normal.copy())
        trace['observer_uncertainty'].append(controller.state.force_uncertainty.copy())
        trace['contact_model_active'].append(controller.state.contact_model_active.copy())
        reserve=getattr(action,'debug',{}).get('support_reserve_slack')
        trace['support_reserve_slack'].append(np.full(controller.g.horizon,np.nan) if reserve is None else np.asarray(reserve,float))
        trace['transport_paused'].append(bool(getattr(action,'pause_transport',False)))
        trace['qp_status'].append(str(getattr(action,'debug',{}).get('qp_status','not_optimized')))
        trace['trajectory_qp_status'].append(str(getattr(action,'debug',{}).get('trajectory_qp_status','not_optimized')))
        trace['control_ms'].append(control_ms[-1])
        if tick==round(scenario.duration/DT): break
        plant.execute(action,obs,env,command)
        plant.advance(env)
        if sum(int(x.number) for x in plant.data.warning)>0 and failure_at is None:
            failure_at=plant.time; failure_reason='numerical_warning'
        if not np.all(np.isfinite(plant.data.qpos)) or not np.all(np.isfinite(plant.data.qvel)):
            if failure_at is None: failure_at=plant.time; failure_reason='nonfinite_dynamics'
            break
    trace={k:np.asarray(v) for k,v in trace.items()}
    # Physics owns the chronological actual-force vectors (including endpoints).
    sub=getattr(plant,'substep_records',None)
    if isinstance(sub,dict):
        trace.update({f'substep_{k}':np.asarray(v) for k,v in sub.items()})
    elif sub is not None:
        sub=np.asarray(sub)
        cols=SUBSTEP_COLUMNS
        if cols is not None:
            for k in ('time','normal','skin_normal','shear','palm','support','hand_other','all_hand_object','collision','penetration','world_penetration','duration'):
                names=([f'{k}_{i}_N' for i in range(5)] if k in ('normal','skin_normal','shear') else
                       [{'time':'solver_time','palm':'palm_N','support':'support_N','hand_other':'hand_other_N','all_hand_object':'all_hand_object_N','collision':'collision_N','penetration':'penetration_m','world_penetration':'world_penetration_m','duration':'duration_s'}[k]])
                if all(x in cols for x in names):
                    ids=[cols.index(x) for x in names]
                    trace[f'substep_{k}']=sub[:,ids] if k in ('normal','skin_normal','shear') else sub[:,ids[0]]
    meta={'scenario':asdict(scenario),'method':job['method'],'seed':job['seed'],'split':job['split'],
          'parameters':p,'reference':reference,'events':evaluation_events(scenario),'physics_timestep':job['timestep'],
          'benchmark_id':BENCHMARK_ID,'model_metadata':plant.model_metadata,
          'desired_semantics':'requested normal-force reference; forecast separately stored in predicted_normal',
          'evaluation_config':cfg,
          'command_at':0.,'mechanical_arrival_at':arrival,'established_at':established,
          'failure_at':failure_at,'failure_reason':failure_reason,
          'numerical_warnings':sum(int(x.number) for x in plant.data.warning),
          'gains':job.get('gains',{}),'controller_final':getattr(controller,'debug',{})}
    metrics=evaluate_trial(trace,meta,job.get('evaluation_config',{}))
    result={**meta,**metrics,'scenario':scenario.name,'scenario_title':scenario.title,
            'boundary':scenario.boundary,'wall_seconds':time.perf_counter()-start,
            'controller_p95_ms':float(np.percentile(control_ms,95)),
            'controller_mean_ms':float(np.mean(control_ms))}
    out=Path(job['output']); key=f'{scenario.name}__{job["method"]}__{job["seed"]}'
    (out/'traces').mkdir(exist_ok=True)
    filename=out/'traces'/f'{key}.npz'
    np.savez_compressed(filename,**trace,metadata=json.dumps(meta,ensure_ascii=False))
    result['trace_file']=str(filename.relative_to(out)); result['trace_sha256']=sha(filename)
    dump(out/'trials'/f'{key}.json',result)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['pilot','development','test','robustness'],default='pilot')
    parser.add_argument('--seeds',type=int,nargs='+',default=[12101])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--scenarios',nargs='+',default=[s.name for s in SCENARIOS])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',required=True)
    parser.add_argument('--settings',type=Path,default=ROOT/'settings.json')
    parser.add_argument('--timestep',type=float,default=.0005)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.workers<1 or len(set(args.seeds))!=len(args.seeds): raise ValueError('invalid workers/seeds')
    if abs(DT/args.timestep-round(DT/args.timestep))>1e-9: raise ValueError('step must divide DT')
    if set(args.scenarios)-{s.name for s in SCENARIOS}: raise ValueError('unknown scenario')
    settings=json.loads(args.settings.read_text(encoding='utf-8')) if args.settings.exists() else {}
    if args.split in ('test','robustness'):
        frozen=json.loads((ROOT/'review'/'confirmation_plan.json').read_text(encoding='utf-8'))
        if frozen['code_sha256']!=hashes() or frozen['settings_sha256']!=sha(args.settings) or frozen['benchmark_id']!=BENCHMARK_ID:
            raise RuntimeError('Confirmation freeze differs; do not retune after test.')
        if args.seeds!=frozen[args.split+'_seeds'] or args.methods!=frozen['methods'] or args.scenarios!=frozen['scenarios']:
            raise RuntimeError('Confirmation matrix differs')
        if args.timestep!=frozen['physics_timestep']:
            raise RuntimeError('Confirmation physics step differs')
    out=ROOT/'output'/args.out; out.mkdir(parents=True,exist_ok=True); (out/'trials').mkdir(exist_ok=True)
    manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'split':args.split,
              'seeds':args.seeds,'methods':args.methods,'scenarios':args.scenarios,'definitions':scenario_manifest(),
              'code_sha256':hashes(),'settings':settings,'physics_timestep':args.timestep,'control_timestep':DT,
              'expected_trials':len(args.seeds)*len(args.methods)*len(args.scenarios),
              'benchmark_id':BENCHMARK_ID,'full_shadow_dynamics':False,
              'python':sys.version,'mujoco':mujoco.__version__,'numpy':np.__version__,'platform':platform.platform(),
              'source_model_sha256':sha(ROOT.parent/'五指假肢抓握仿真'/'model.xml')}
    if (out/'manifest.json').exists():
        if not args.resume: raise RuntimeError('Use a fresh output directory')
        old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        for k in ('split','seeds','methods','scenarios','definitions','code_sha256','settings','physics_timestep','source_model_sha256','benchmark_id'):
            if json.dumps(old[k],sort_keys=True)!=json.dumps(manifest[k],sort_keys=True): raise RuntimeError('resume mismatch '+k)
        manifest=old
    else: dump(out/'manifest.json',manifest)
    results=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'trials').glob('*.json')]
    done={(r['scenario'],r['method'],r['seed']) for r in results}
    jobs=[{'scenario':s,'method':m,'seed':seed,'split':args.split,'output':str(out),
           'timestep':args.timestep,'gains':settings.get('controllers',{}).get(m,{}),
           'references':settings.get('references',{}),'evaluation_config':settings.get('evaluation',{})}
          for s in args.scenarios for seed in args.seeds for m in args.methods if (s,m,seed) not in done]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut={pool.submit(run_trial,j):j for j in jobs}
        for f in as_completed(fut):
            j=fut[f]
            try: r=f.result()
            except Exception as e:
                dump(out/'error.json',{'job':j,'error':repr(e)}); raise
            results.append(r)
            if len(results)%8==0 or len(results)==manifest['expected_trials']:
                dump(out/'trials.json',sorted(results,key=lambda r:(r['scenario'],r['seed'],r['method'])))
                print(json.dumps({'completed':len(results),'expected':manifest['expected_trials'],'elapsed_s':round(time.perf_counter()-start,1)},ensure_ascii=False),flush=True)
    dump(out/'trials.json',sorted(results,key=lambda r:(r['scenario'],r['seed'],r['method'])))
    if hashes()!=manifest['code_sha256']: raise RuntimeError('Code changed during run; this output is development-only')
    dump(out/'execution_complete.json',{'trials':len(results),'wall_seconds':time.perf_counter()-start,'source_unchanged':True})

if __name__=='__main__': main()

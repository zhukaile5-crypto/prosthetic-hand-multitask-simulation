"""Saved six-probe audit plus static FK only. mj_step is forbidden."""
from pathlib import Path
from datetime import datetime,timezone
from types import ModuleType,SimpleNamespace
import sys,json,hashlib,runpy,importlib.util
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'calibration/release_physics_v3'
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(FOUNDATION))
# Snapshot controllers import this path helper but do not call its functions.
bootstrap=ModuleType('research_imports');bootstrap.ROOT=ROOT;bootstrap.FOUNDATION=FOUNDATION
sys.modules['research_imports']=bootstrap
import mujoco
def forbidden_step(*a,**k):raise AssertionError('No new physics in independent audit')
mujoco.mj_step=forbidden_step
from hand_plant import HandPlant
from foundation_config import FoundationConfig
helpers=runpy.run_path(str(ROOT/'review/verify_lifecycle.py'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b,tol=1e-9):return bool(np.allclose(a,b,atol=tol,rtol=0,equal_nan=True))
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
release=load_module('independent_snapshot_release_v3',FOLDER/'sources/lifecycle_release.py')
decoder=load_module('independent_snapshot_decoder_v3',FOLDER/'sources/lifecycle_plant.py')
def obj(d):
    return SimpleNamespace(**{k:(np.asarray(v) if isinstance(v,list) and k!='skin_patches' else v) for k,v in d.items()})
def rel(a,b):return float(np.max(abs(a-b)/np.maximum(.1,np.maximum(abs(a),abs(b)))))

def main():
    summary=read(FOLDER/'summary.json');manifest=read(FOLDER/'manifest.json');checks={};inputs={};records=[];paired={};all_recontacts=[]
    def check(k,v):checks[k]=bool(v)
    for p in [FOLDER/'summary.json',FOLDER/'manifest.json',ROOT/'review/verify_lifecycle.py']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    check('all_six_completed',summary['execution_complete'] and len(summary['results'])==len(manifest['jobs'])==6)
    check('failed_declared_source_gate_preserved',summary['sources_unchanged'] is False)
    for n,h in manifest['sources'].items():
        p=FOLDER/'sources'/n;check('snapshot/'+n,p.exists() and sha(p)==h);inputs[str(p.relative_to(ROOT))]=sha(p)
    actual_dependencies=set(manifest['sources'])-{'run_development.py'}
    for folder in sorted(p for p in FOLDER.iterdir() if p.is_dir() and p.name!='sources'):
        meta=read(folder/'metadata.json');result=read(folder/'result.json');blob=folder/'trace.npz'
        with np.load(blob,allow_pickle=False) as a:z={k:a[k] for k in a.files}
        inputs[str(blob.relative_to(ROOT))]=sha(blob)
        check(folder.name+'/trace_hash',sha(blob)==result['trace_sha256'])
        check(folder.name+'/inline_metadata_equal_except_postsave_trace_hash',json.loads(str(z['metadata']))=={k:v for k,v in meta.items() if k!='trace_sha256'} and meta['trace_sha256']==sha(blob))
        actual=meta['research_source_sha256'];diff={k:dict(start_hash=manifest['sources'].get(k),run_hash=v) for k,v in actual.items() if manifest['sources'].get(k)!=v}
        check(folder.name+'/real_execution_sources_match',all(actual.get(k)==manifest['sources'][k] for k in actual_dependencies-{'validate_release_physics.py'}))
        check(folder.name+'/only_unused_runner_may_differ',set(diff)<={'run_development.py'})
        dt=float(meta['physics_timestep']);t=z['time'];start=z['substep_time'];dur=z['substep_duration'];when=start+dur
        check(folder.name+'/full_clock',len(t)==3201 and same(t,np.arange(3201)*.005,1e-8) and len(start)==round(16/dt) and same(start,np.arange(len(start))*dt,1e-8) and same(dur,dt,1e-12) and abs(when[-1]-16)<1e-8)
        essential=['qpos','qvel','ctrl','ctrl_before','skin_normal','normal','all_hand_object','substep_skin_normal','substep_normal','substep_all_hand_object']
        check(folder.name+'/finite_truth_and_actions',all(np.isfinite(z[k]).all() for k in essential))
        check(folder.name+'/no_mujoco_warning',int(z['warning_count'].max())==0)
        release_at=float(meta['mechanical_release_at']);mask=t>=release_at-1e-9;rawmask=start>=release_at-1e-9
        debugs=[json.loads(str(v)) for v in z['debug_json']];obs=[json.loads(str(v)) for v in z['observation_json']]
        ids=np.flatnonzero(mask);command_ok=rate_ok=monitor_ok=pred_ok=True;max_rate_residual=-float('inf');counts=[]
        names=set(FoundationConfig.__dataclass_fields__);config=FoundationConfig(**{k:v for k,v in meta['config'].items() if k in names});plant=decoder.LifecyclePlant(config)
        aids=np.concatenate(plant.actuator_groups);model=plant.model
        check(folder.name+'/same_model_identity',plant.model_sha256==meta['model_sha256'])
        for i in ids:
            d=debugs[i];before=np.asarray(d['actuator_before']);after=np.asarray(d['actual_actuator_target']);rate=np.asarray(d['actuator_rate_limit']);h=float(d['dt'])
            command_ok &= same(after,z['ctrl'][i,aids]) and same(before,z['ctrl_before'][i,aids]) and np.all(after>=model.actuator_ctrlrange[aids,0]-1e-8) and np.all(after<=model.actuator_ctrlrange[aids,1]+1e-8)
            rate_residual=abs(after-before)-rate*h;max_rate_residual=max(max_rate_residual,float(rate_residual.max()));rate_ok &= bool((rate_residual<=1e-8).all())
            monitor_ok &= d['release_authority'] and d['monitoring_active']
            pred_ok &= not z['prediction_available'][i] and np.isnan(z['predicted_normal'][i]).all()
            counts.append(int(d['observed_recontact_count']))
        check(folder.name+'/actual_18ctrl_bounds_and_full_log',command_ok)
        check(folder.name+'/actual_18ctrl_command_rate',rate_ok)
        # controller_active means an executed authority and includes release;
        # contact_model_active is an observer-state contact mask. Neither is a
        # holding.step/predictor availability flag in the executed runtime.
        check(folder.name+'/release_authority_and_continuous_monitor',monitor_ok and (z['command'][mask]=='release').all() and not z['local_domain_active'][mask].any())
        check(folder.name+'/release_prediction_explicitly_unavailable',pred_ok)
        check(folder.name+'/holds_release_motion_reference',same(z['executed_motion_reference'][mask],np.tile(z['executed_motion_reference'][ids[0]],(len(ids),1)),1e-12))
        skin=np.maximum(z['substep_skin_normal'],0);limits=np.asarray(meta['limits']);budget=float(meta['budget']);local=np.maximum(0,skin-limits);total=np.maximum(0,skin.sum(1)-budget)
        peak=np.maximum(skin.max(0),z['skin_normal'][-1]);guard=bool((skin<=limits+.02).all() and (skin.sum(1)<=budget+.02).all() and (z['skin_normal'][-1]<=limits+.02).all() and z['skin_normal'][-1].sum()<=budget+.02)
        check(folder.name+'/independent_guard_agrees',guard==result['research_metrics']['safety']['guard_pass'] and same(peak,result['research_metrics']['safety']['finger_peak_N']) and abs(float((local.sum(1)*dur).sum())-result['research_metrics']['safety']['local_excess_integral_Ns'])<1e-8)
        recontact=helpers['postclear_recontact'](z,meta);all_recontacts.append(recontact)
        check(folder.name+'/physical_clearance_recomputed',recontact['clear_confirmed_at'] is not None and abs(recontact['clear_confirmed_at']-release_at-result['research_metrics']['release']['physical_clearance_latency_s'])<1e-8)
        # Actual joints are softly limited by MuJoCo, separate from ctrl bounds.
        jids=plant.raw_joint_ids;adr=plant.raw_qadr;dadr=plant.raw_dadr;jrange=model.jnt_range[jids];q=z['qpos'][mask][:,adr];v=z['qvel'][mask][:,dadr]
        over=np.maximum(0,np.maximum(jrange[:,0]-q,q-jrange[:,1]));ii,jj=np.unravel_index(int(np.argmax(over)),over.shape)
        joint_info=dict(maximum_200Hz_range_overshoot_rad=float(over.max()),worst_joint=meta['raw_joint_names'][jj],worst_time_s=float(t[ids[ii]]),maximum_200Hz_actual_speed_rad_s=float(abs(v).max()),per_joint_maximum_200Hz_speed_rad_s=abs(v).max(0).tolist(),actual_joint_speed_limit_registered=False,scope='200Hz actual robot states only; no saved physical-substep joint state and no predeclared qvel limit. Command-rate bounds do not certify measured joint rates.')
        # Independently reconstruct every delayed per-patch geometry packet.
        geometry_ok=True;max_geometry_error=0.;rawpatch=[json.loads(str(x)) for x in z['raw_patches_json']]
        for i,f in enumerate(obs):
            sample=float(f['sample_time']);k=int(np.searchsorted(t,sample+1e-9,side='right')-1)
            if k<0:expected=[]
            else:
                plant.data.qpos[:]=z['qpos'][k];mujoco.mj_kinematics(model,plant.data)
                R=plant.data.xmat[plant.palm].reshape(3,3);origin=plant.data.xpos[plant.palm];press=np.zeros(5);positions=np.zeros((5,3));packets=[]
                for p in rawpatch[k]:
                    digit=int(p['digit']);pressure=max(0,float(p['normal']))
                    if not 0<=digit<5 or pressure<=1e-10:continue
                    body=int(model.geom_bodyid[int(p['hand_geom'])]);Rb=plant.data.xmat[body].reshape(3,3);pb=plant.data.xpos[body];point=np.asarray(p['position']);normal=np.asarray(p['normal_dir'])
                    packets.append((digit,body,Rb.T@(origin+R@point-pb),Rb.T@R@normal,pressure,Rb.T@R));press[digit]+=pressure;positions[digit]+=pressure*point
                positions/=np.maximum(press[:,None],1e-12);expected=[]
                for digit,body,point,normal,pressure,A in packets:
                    if not f['valid'][digit] or not np.isfinite(f['normal'][digit]):continue
                    expected.append(dict(digit=digit,robot_body=body,link_point=point+A@(np.asarray(f['contact_pos'][digit])-positions[digit]),link_normal=normal,normal=max(0,float(f['normal'][digit]))*pressure/max(1e-12,press[digit]),sample_time=float(t[k]),age=float(t[i]-t[k])))
            actualpatch=f['skin_patches'];geometry_ok &= len(expected)==len(actualpatch)
            for a,b in zip(expected,actualpatch):
                geometry_ok &= a['digit']==b['digit'] and a['robot_body']==b['robot_body'] and set(b)==set(a)
                err=max(float(np.max(abs(np.asarray(a[key])-b[key]))) for key in ['link_point','link_normal','normal','sample_time','age']);max_geometry_error=max(max_geometry_error,err);geometry_ok &= err<=1e-8
        check(folder.name+'/every_delayed_patch_reconstructed_without_target_filter',geometry_ok)
        # Static counterfactuals: unrelated object coordinates and diagnostic
        # identity cannot change FK rows or the release QP on the same frame.
        driver=release.LifecycleRelease(plant);f=obj(obs[ids[0]]);driver.request(f,z['ctrl_before'][ids[0],aids],z['executed_motion_reference'][ids[0],0],z['executed_motion_reference'][ids[0],1:])
        a,ad,al=driver.kinematics.contact_rows(f);free=int(np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0]);qadr=int(model.jnt_qposadr[free]);driver.kinematics.data.qpos[qadr:qadr+3]=[23,-31,17];b,bd,bl=driver.kinematics.contact_rows(f)
        check(folder.name+'/FK_robot_encoder_only',np.array_equal(a,b) and ad==bd and np.array_equal(al,bl))
        # Unknown observation after genuine clear must not fabricate recontact.
        blank=dict(obs[ids[-1]]);blank['time']=20.;blank['normal']=[0.]*5;blank['valid']=[True]*5;blank['auxiliary_valid']=True;blank['auxiliary_pressure']=0.;blank['skin_patches']=[]
        driver2=release.LifecycleRelease(plant);driver2.request(obj(blank),z['ctrl_before'][ids[0],aids],.1,np.zeros(2))
        for step in range(42):blank['time']=20+step*.005;driver2.step(obj(blank))
        old=driver2.recontacts;blank['time']+=.005;blank['valid']=[False]*5;blank['auxiliary_valid']=False;unknown=driver2.step(obj(blank))
        check(folder.name+'/unknown_after_clear_not_recontact',driver2.recontacts==old and unknown.stage=='release_observation_uncertain')
        record=dict(folder=folder.name,shape=meta['shape'],dt=dt,declared_identity_diff=diff,actual_execution_dependencies_match=True,mission_success=result['task_success'],physical_release_success=result['research_metrics']['release']['physical_success'],release_latency_s=recontact['clear_confirmed_at']-release_at,whole_force_guard_pass=guard,whole_skin_finger_peak_N=peak.tolist(),release_skin_finger_peak_N=skin[rawmask].max(0).tolist(),release_raw_all_hand_object_peak_N=float(z['substep_all_hand_object'][rawmask].max()),maximum_rate_residual=max_rate_residual,measured_joint_diagnostics=joint_info,postclear_recontact=recontact,observed_recontact_count_final=counts[-1],maximum_delayed_patch_reconstruction_error=max_geometry_error,release_authority_flag_fraction=float(z['controller_active'][mask].mean()),observer_contact_mask_fraction=float(z['contact_model_active'][mask].mean()))
        records.append(record);paired[(meta['shape'],dt)]=(peak,float(skin.sum(1).max()))
        print(folder.name,'raw reviewed',flush=True)
    comparisons=[]
    for shape in ['capsule','sphere','box']:
        coarse,ct=paired[(shape,.0005)];fine,ft=paired[(shape,.00025)];error=max(rel(coarse,fine),rel(np.array(ct),np.array(ft)))
        comparisons.append(dict(shape=shape,max_raw_peak_relative_difference=error,threshold=.15,passed=error<=.15,definition='maximum absolute difference divided by max(0.1,abs(coarse),abs(fine)), over individual finger and total-finger peaks; same as frozen foundation gate'))
    result=dict(schema='independent_release_v3_saved_data_and_static_fk_v1',created_utc=datetime.now(timezone.utc).isoformat(),data_and_interface_checks_passed=all(checks.values()),checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        declared_source_freeze_passed=False,declared_failure_preserved=True,actual_executed_dependency_identity_consistent=all(c['actual_execution_dependencies_match'] for c in records),model_admitted=False,overall_admission_passed=False,
        records=records,step_comparisons=comparisons,postclear_description=helpers['summarize_recontact'](all_recontacts),input_sha256=inputs,reviewer_sha256=sha(__file__),physics_steps_executed=0,
        review_revision=dict(prior_diagnostic_path='review/independent_release_v3_raw_20261104T050149221677Z.json',
            corrections=['Sidecar adds the postsave trace hash and cannot be byte-equivalent to NPZ inline metadata.',
                'controller_active includes the release authority; holding.step absence is established in the separate frozen-source audit.',
                'contact_model_active continues to record observer contact state; actual prediction availability is false and force prediction is NaN throughout release.']),
        limitations=['Six standard nominal development probes are not a formal performance comparison.',
            'The original declared source gate remains failed; the unused runner explains the per-record difference but is not erased from provenance.',
            'Box has force-guard failures and must also satisfy the separately reported step sensitivity gate before release admission.',
            'Actuator command bounds and command-rate bounds are separate from measured soft joint-limit overshoot and actual joint speed. No measured joint-speed safety threshold was registered.',
            'Physical release success uses first clearance dwell. Post-clear raw recontact is separately reported and never replaces the whole raw force guard.',
            'Robot tendon Jacobian uses equal split at coupled joints; delayed planes and ideal known skin-link geometry do not certify collision-free space. Palm/nonfinger geometry is absent from the separation QP.'])
    out=ROOT/'review'/('independent_release_v3_raw_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(data_checks_passed=result['data_and_interface_checks_passed'],checks=len(checks),failed=result['failed_checks'],step_comparisons=comparisons,overall_admission_passed=False,path=str(out),sha256=sha(out)),ensure_ascii=False))

if __name__=='__main__':main()

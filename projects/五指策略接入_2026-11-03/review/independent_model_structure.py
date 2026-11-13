"""Inspect compiled model structure/isolation without stepping any simulation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hand_plant import HandPlant
from foundation_config import FoundationConfig

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--shape',default='capsule')
    parser.add_argument('--solref',type=float,default=.008)
    args=parser.parse_args()
    initial_sources={name:sha(ROOT/name) for name in ('hand_plant.py','foundation_config.py')}
    config=FoundationConfig(shape=args.shape,solref=args.solref,timestep=.001)
    first=HandPlant(config);second=HandPlant(config)
    model=first.model
    body=first.object
    target_geoms=np.flatnonzero(model.geom_bodyid==body)
    joints=np.arange(model.body_jntadr[body],model.body_jntadr[body]+model.body_jntnum[body])
    # Only body equalities (connect/weld) could constrain the free object here.
    object_equalities=[]
    for i in range(model.neq):
        typ=int(model.eq_type[i])
        if typ in (int(mujoco.mjtEq.mjEQ_CONNECT),int(mujoco.mjtEq.mjEQ_WELD)) and body in (int(model.eq_obj1id[i]),int(model.eq_obj2id[i])):
            object_equalities.append(i)
    before=second.model.geom_friction.copy()
    first.set_environment(friction=.173)
    isolated=np.array_equal(before,second.model.geom_friction)
    tree=ET.parse(ROOT/'assets/legacy_model.xml');legacy_root=tree.getroot()
    legacy_root.find('compiler').set('meshdir','')
    assets={p.name:p.read_bytes() for p in (ROOT/'assets/shadow/assets').iterdir() if p.is_file()}
    legacy=mujoco.MjModel.from_xml_string(ET.tostring(legacy_root,encoding='unicode'),assets=assets)
    inertia_match=True
    for b in range(model.nbody):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or ''
        if not name.startswith('rh_'):continue
        old=mujoco.mj_name2id(legacy,mujoco.mjtObj.mjOBJ_BODY,name)
        inertia_match &= old>=0 and abs(float(model.body_mass[b]-legacy.body_mass[old]))<1e-12 and np.allclose(model.body_inertia[b],legacy.body_inertia[old],atol=1e-12,rtol=0)
    checks={
        'full_hand_joint_count':len(first.raw_joint_ids)==24,
        'full_original_hand_actuator_count':sum((mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) or '').startswith('rh_A_') for i in range(model.nu))==20,
        'object_single_free_joint':bool(len(joints)==1 and model.jnt_type[joints[0]]==mujoco.mjtJoint.mjJNT_FREE),
        'object_no_body_equalities':not object_equalities,
        'target_geometry_body_group_complete':set(map(int,target_geoms))==first.object_geoms,
        'declared_object_mass_not_multiplied_by_geometry':abs(float(model.body_mass[body])-float(config.mass))<1e-12,
        'hand_body_mass_and_inertia_preserved':bool(inertia_match),
        'target_geometries_share_environment_friction':bool(np.allclose(model.geom_friction[target_geoms,0],.173,atol=0,rtol=0)),
        'target_contact_dimension_is_three':bool(np.all(model.geom_condim[target_geoms]==3)),
        'all_actuators_explicitly_force_limited':bool(np.all(model.actuator_forcelimited)),
        'all_actuator_force_bounds_finite':bool(np.isfinite(model.actuator_forcerange).all() and np.all(model.actuator_forcerange[:,0]<model.actuator_forcerange[:,1])),
        'closure_domain_finite_nonempty':bool(np.isfinite(first.closure_min).all() and np.isfinite(first.closure_max).all() and np.all(first.closure_min<first.closure_max)),
        'model_instances_independent':bool(isolated and not np.shares_memory(first.model.geom_friction,second.model.geom_friction)),
        'no_dynamics_advanced':first.time==0. and second.time==0.,
        'sources_unchanged_during_check':all(sha(ROOT/name)==value for name,value in initial_sources.items())}
    if args.shape=='box':
        extent=np.array([.018,.060,.026])*config.size
        inertia=config.mass/3.*np.array([extent[1]**2+extent[2]**2,extent[0]**2+extent[2]**2,extent[0]**2+extent[1]**2])
        checks['rounded_box_27_target_geometries']=len(target_geoms)==27
        checks['rounded_box_explicit_envelope_inertia']=bool(np.allclose(model.body_inertia[body],inertia,atol=1e-12,rtol=0))
    result={'schema':'independent-compiled-model-structure-v1','checked_utc':datetime.now(timezone.utc).isoformat(),
            'passed':all(checks.values()),'checks':checks,'model_sha256':first.model_sha256,
            'source_sha256':initial_sources,'configuration':config.to_dict(),
            'dimensions':{'nq':int(model.nq),'nv':int(model.nv),'nu':int(model.nu),'neq':int(model.neq),'hand_joints':len(first.raw_joint_ids),'target_geometry_count':len(target_geoms)},
            'actuator_force_ranges':model.actuator_forcerange.tolist(),'closure_min':first.closure_min.tolist(),'closure_max':first.closure_max.tolist(),
            'limitations':['No simulator step or policy was run. This checks compiled structure and state isolation only.',
                           'Physical holding, release and contact pressure remain subject to independent raw-probe gates.']}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=(args.out or ROOT/f'review/independent_model_structure_{stamp}.json').resolve()
    if not out.is_relative_to(ROOT.resolve()) or out.exists():raise SystemExit('Need a fresh output in the integration directory.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':checks,'dimensions':result['dimensions'],'certificate':str(out)},ensure_ascii=False))

if __name__=='__main__':main()

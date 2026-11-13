"""Public foundation self-check certificate; independent auditor checks raw data."""
from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hand_plant import HandPlant
from foundation_config import FoundationConfig,BENCHMARK_ID

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sustained(t,clear,origin,dwell):
    start=None
    for tick in np.flatnonzero(t>=origin-1e-9):
        if clear[tick]:
            if start is None:start=t[tick]
            if t[tick]-start>=dwell-1e-9:return float(t[tick]-origin)
        else:start=None
    return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--matrix',required=True);parser.add_argument('--protocol',required=True)
    parser.add_argument('--calibration',default='review/foundation_calibration_final.json')
    parser.add_argument('--out',default='review/foundation_physical_acceptance.json');args=parser.parse_args()
    dest=ROOT/args.out
    if dest.exists():raise SystemExit('fresh certificate output required')
    protocol_path=ROOT/args.protocol;protocol=read(protocol_path);g=protocol['physical_foundation']['gates']
    matrix=ROOT/args.matrix;rows=read(matrix/'manifest.json');records=[];checks={};cache={}
    for row in rows:
        folder=Path(row['folder']);path=folder/'raw_probe.npz';summary=read(folder/'summary.json')
        with np.load(path,allow_pickle=False) as z:
            t=z['time'];force=z['normal'];mask=(t>=8.2-1e-9)&(t<=8.9+1e-9)
            contact=force>g['digit_contact_min_N']
            supported=(contact.sum(1)>=g['minimum_supported_digits'])&contact[:,g['required_digit']]
            held=supported&(z['support']<=g['table_support_max_N'])
            held&=np.abs(z['lift']-.1)<=g['lift_error_max_m']
            held&=z['object_z']-summary['initial_position'][2]>=g['minimum_object_lift_from_initial_m']
            release=sustained(t,z['all_hand_object']<=g['release_all_hand_object_force_max_N'],10.,g['release_dwell_s'])
            release_index=int(np.argmin(np.abs(t-10.)))
            cell_checks=dict(finite=all(np.isfinite(z[k]).all() for k in ('normal','skin_normal','qpos','qvel','ctrl','substep_normal')),
                supported_full_static=bool(np.all(held[mask])),warnings=int(np.max(z['warnings']))==0,
                penetration=float(np.max(z['substep_penetration']))<=g['max_hand_object_penetration_m'],
                release_exposed=bool(supported[release_index] and z['all_hand_object'][release_index]>.2),
                release_clearance=release is not None and release<=g['release_deadline_s'],
                public_clock=bool(len(t)==2801 and np.max(np.abs(t-np.arange(len(t))*.005))<1e-9),
                full_physics=bool(len(z['substep_time'])==round(14./row['config']['timestep']) and abs(np.sum(z['substep_duration'])-14.)<1e-8),
                force_partition=bool(np.max(np.abs(force.sum(1)+z['palm']+z['hand_other']-z['all_hand_object']))<1e-8),
                source_bound=all(sha(ROOT/name)==digest for name,digest in summary['source_sha256'].items()))
            record=dict(config=row['config'],checks=cell_checks,passed=all(cell_checks.values()),
                raw_path=str(path),raw_sha256=sha(path),summary_sha256=sha(folder/'summary.json'),
                raw_basis_protocol_sha256=summary['protocol_sha256'],release_latency=release,
                mean_force=np.mean(force[mask],0).tolist(),peak_force=np.max(z['substep_normal'],0).tolist(),
                max_penetration=float(np.max(z['substep_penetration'])),minimum_object_lift=float(np.min((z['object_z']-summary['initial_position'][2])[mask])))
            records.append(record);cache[(row['config']['shape'],row['config']['solref'],row['config']['timestep'])]=record
    expected={(s,c,d) for s in ('capsule','box','sphere') for c in (.008,.012) for d in (.0005,.00025)}
    checks['complete_matrix']=set(cache)==expected and len(records)==12
    checks['all_raw_cells']=all(r['passed'] for r in records)
    pairs=[]
    for shape in ('capsule','box','sphere'):
        for solref in (.008,.012):
            coarse=cache[(shape,solref,.0005)];fine=cache[(shape,solref,.00025)]
            mean=float(np.max(np.abs(np.asarray(coarse['mean_force'])-fine['mean_force'])/np.maximum(.1,fine['mean_force'])))
            peak=float(np.max(np.abs(np.asarray(coarse['peak_force'])-fine['peak_force'])/np.maximum(.1,fine['peak_force'])))
            pair=dict(shape=shape,solref=solref,mean_relative=mean,peak_relative=peak,
                passed=mean<=g['step_halving_static_mean_force_relative_difference_max'] and peak<=g['step_halving_peak_force_relative_difference_max'] and coarse['passed'] and fine['passed'])
            pairs.append(pair)
    checks['all_step_pairs']=all(p['passed'] for p in pairs)
    structures=[]
    for shape in ('capsule','box','sphere'):
        plant=HandPlant(FoundationConfig(shape=shape));m=plant.model
        count=sum((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) or '').startswith('rh_') for i in range(m.nu))
        free=bool(m.body_jntnum[plant.object]==1 and m.jnt_type[m.body_jntadr[plant.object]]==mujoco.mjtJoint.mjJNT_FREE)
        structure=dict(shape=shape,hand_joints=len(plant.raw_joint_ids),hand_actuators=count,
            free_object=free,equality_constraints=int(m.neq),mocap_bodies=int(m.nmocap),object_geom_count=len(plant.object_geoms),
            mass=float(m.body_mass[plant.object]),finite_actuation=bool(np.isfinite(m.actuator_forcerange).all() and np.all(m.actuator_forcelimited)))
        structure['passed']=free and m.neq==0 and m.nmocap==0 and count==20 and len(plant.raw_joint_ids)==24 and structure['finite_actuation'] and abs(structure['mass']-.12)<1e-10
        structures.append(structure)
    checks['compiled_structure']=all(s['passed'] for s in structures)
    cal_path=ROOT/args.calibration;cal=read(cal_path)
    checks['calibration_domains']=all(v['selected_holdout']['passed'] and v['selected_radius'] in (.02,.04) for v in cal['shapes'].values())
    result=dict(passed=all(checks.values()),benchmark_id=BENCHMARK_ID,checks=checks,
        protocol_path=str(protocol_path),protocol_sha256=sha(protocol_path),
        source_sha256={name:sha(ROOT/name) for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py')},
        calibration_path=str(cal_path),calibration_sha256=sha(cal_path),records=records,pairs=pairs,structures=structures,
        certificate_script_sha256=sha(Path(__file__)),scope='Self-check of shared physical apparatus only; independent raw audit and policy admission remain separate.')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(dest),passed=result['passed'],checks=checks),ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()

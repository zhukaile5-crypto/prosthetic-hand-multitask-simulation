"""Audit actual saved physical contact packets without running new dynamics.

Raw patch forces are compared to the recorded physical normal-force samples.
Independent wrench sums verify the distributed and auxiliary representation.
This does not validate controller performance or a global force-response model.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

from sensors import aggregate_patches

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def independent(patches):
    pressure=np.zeros(5);normal_map=np.zeros((6,5));other=np.zeros((5,6))
    total=np.zeros((5,6));aux=np.zeros(6);aux_pressure=0.
    for p in patches:
        d=int(p['digit']);n=max(0.,float(p['normal']))
        if n<=1e-10:continue
        f=np.asarray(p['force'],float);x=np.asarray(p['position'],float)
        tau=np.asarray(p.get('torque',[0.,0.,0.]),float)
        wrench=np.r_[f,np.cross(x,f)+tau]
        if not 0<=d<5:
            aux+=wrench;aux_pressure+=n;continue
        pressure[d]+=n;total[d]+=wrench
        direction=np.asarray(p['normal_dir'],float);direction/=np.linalg.norm(direction)
        fn=n*direction;ft=f-fn
        normal_map[:,d]+=np.r_[fn,np.cross(x,fn)]
        other[d]+=np.r_[ft,np.cross(x,ft)+tau]
    for d in range(5):
        if pressure[d]>1e-10:normal_map[:,d]/=pressure[d]
    return pressure,normal_map,other,total,aux,aux_pressure

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--matrix',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    matrix=(ROOT/args.matrix).resolve();dest=(ROOT/args.out).resolve()
    if not dest.is_relative_to(ROOT) or not matrix.is_relative_to(ROOT):
        raise ValueError('input/output must be in the isolated integration directory')
    if dest.exists():raise FileExistsError('Choose a fresh observation audit destination')
    inputs=sorted(matrix.glob('*/raw_probe.npz'))
    checks={'complete_12_cell_matrix':len(inputs)==12};rows=[]
    input_sha={}
    for path in inputs:
        initial=sha(path);input_sha[str(path.relative_to(ROOT))]=initial
        with np.load(path,allow_pickle=False) as saved:
            packets=json.loads(str(saved['patch_samples_json']))
            t=saved['time'];skin=saved['skin_normal'];saved_support=saved['support'];metadata=json.loads(str(saved['metadata']))
        max_wrench=max_pressure=max_aux=max_world=0.;legacy=[];distributed=[];static=[]
        samples=0;loaded=0;aux_loaded=0;static_table=[]
        for packet in packets:
            timestamp=float(packet['time']);patches=packet['patches']
            R=np.asarray(packet['palm_rotation'],float);up=R.T@np.array([0.,0.,1.])
            m,d=aggregate_patches(patches,support_axis=up)
            p,W,T,total,aux,aux_pressure=independent(patches)
            max_wrench=max(max_wrench,float(np.max(abs(np.asarray(d['normal_wrench_map'])-W))),
                float(np.max(abs(np.asarray(d['finger_non_normal_by_finger'])-T))),
                float(np.max(abs((W*p[None,:]+T.T)-total.T))))
            max_aux=max(max_aux,float(np.max(abs(np.asarray(d['raw_auxiliary_wrench'])-aux))),
                abs(float(d['auxiliary_pressure'])-aux_pressure))
            index=int(np.argmin(abs(t-timestamp)))
            if abs(float(t[index])-timestamp)>1e-8:raise AssertionError('missing physical sample time')
            max_pressure=max(max_pressure,float(np.max(abs(p-skin[index]))))
            target_force=sum((R@np.asarray(q['force']) for q in patches if q['is_target']),np.zeros(3))
            max_world=max(max_world,float(np.max(abs(target_force-np.asarray(packet['object_force_world'])))))
            samples+=1
            if 8.2-1e-8<=timestamp<=8.9+1e-8:
                legacy.append(d['aggregation_error']);distributed.append(d['distributed_valid'])
                static.append(target_force)
                static_table.append(float(saved_support[index]))
                loaded+=int(np.count_nonzero(p>.2)>=3 and p[0]>.2)
                aux_loaded+=int(aux_pressure>.02)
        static=np.asarray(static);legacy=np.asarray(legacy);distributed=np.asarray(distributed)
        row=dict(input=str(path.relative_to(ROOT)),shape=metadata['config']['shape'],
            solref=metadata['config']['solref'],timestep=metadata['config']['timestep'],
            samples=samples,static_samples=len(static),static_loaded_samples=loaded,
            static_auxiliary_contact_samples=aux_loaded,
            max_independent_distributed_wrench_error=max_wrench,
            max_normal_pressure_vs_recorded_error_N=max_pressure,max_auxiliary_wrench_error=max_aux,
            max_target_world_force_reconstruction_error_N=max_world,
            distributed_valid_fraction=np.mean(distributed,axis=0).tolist() if len(static) else [],
            legacy_single_point_error_max=np.max(legacy,axis=0).tolist() if len(static) else [],
            static_hand_target_force_world_mean=np.mean(static,axis=0).tolist() if len(static) else [],
            static_gravity_N=float(metadata['config']['mass']*9.81),
            source_unchanged=sha(path)==initial)
        gravity=float(metadata['config']['mass']*9.81)
        mean_fz=float(np.mean(static[:,2])) if len(static) else float('nan')
        table_mean=float(np.mean(static_table)) if static_table else float('nan')
        balance_tolerance=max(.05,.05*gravity)+table_mean
        row.update(static_target_hand_force_Fz_N=mean_fz,static_table_support_mean_N=table_mean,
            gravity_direction_tolerance_N=balance_tolerance,
            hand_to_target_gravity_direction_pass=bool(mean_fz>0. and abs(mean_fz-gravity)<=balance_tolerance))
        okay=(samples>0 and len(static)>=140 and max_wrench<1e-9 and max_pressure<1e-9 and
              max_aux<1e-9 and max_world<1e-9 and bool(np.all(distributed)) and row['source_unchanged'])
        okay &= row['hand_to_target_gravity_direction_pass']
        row['passed']=okay;checks[path.parent.name]=okay;rows.append(row)
    report=dict(passed=all(checks.values()),checks=checks,cells=rows,input_sha256=input_sha,
        scope='Independent arithmetic on actual saved full-hand physical contact packets; no MuJoCo reconstruction, dynamics, controllers or tuning executed.',
        limits='Static packet representation and source consistency only. Physical task/step-size acceptance and force-model prediction accuracy remain separate gates.',
        source_sha256={name:sha(ROOT/name) for name in ('contracts.py','sensors.py','verify_physical_sensors.py')})
    dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],cells=len(rows),samples=sum(r['samples'] for r in rows),output=str(dest)),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

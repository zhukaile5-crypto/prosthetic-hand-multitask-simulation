"""Header/metadata-only phase3 inventory. Does not read qpos object values."""
from pathlib import Path
import ast,hashlib,json,struct,zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_header(archive,key):
    with archive.open(key+'.npy') as f:
        if f.read(6)!=b'\x93NUMPY':raise ValueError('Not an NPY member')
        version=tuple(f.read(2));width=2 if version==(1,0) else 4
        if version not in ((1,0),(2,0),(3,0)):raise ValueError('Unrecognized NPY header version')
        length=struct.unpack('<H' if width==2 else '<I',f.read(width))[0]
        if length>100000:raise ValueError('Excessive NPY header')
        value=ast.literal_eval(f.read(length).decode('utf-8' if version==(3,0) else 'latin1'))
        return dict(shape=list(value['shape']),dtype=value['descr'],fortran_order=value['fortran_order'])


def run():
    base=ROOT/'calibration/phase3_path_validation_v3';plan=json.loads((base/'plan.json').read_text(encoding='utf-8'))
    required=('time','qpos','qvel','ctrl','alpha','input_reference','robot_lift_position_m','robot_wrist_position_rad',
        'physical_time','physical_interval_start','physical_duration','physical_qpos','physical_qvel','physical_ctrl','physical_input_reference')
    groups=[];executed=0;unavailable=0;incomplete=[]
    for shape in ('capsule','box','sphere'):
        for cp in sorted((base/shape).glob('checkpoint_*')):
            branches=[]
            for child in sorted(p for p in cp.iterdir() if p.is_dir()):
                metadata=child/'metadata.json'
                if not metadata.is_file():continue
                m=json.loads(metadata.read_text(encoding='utf-8'));trace=child/'trace.npz'
                row=dict(name=child.name,metadata_path=str(metadata),trace_path=str(trace),available=m.get('available'),
                    attempted_physics=m.get('attempted_physics'),complete=m.get('complete'),failure=m.get('failure'),
                    declared_trace_sha256=m.get('trace_sha256'),hash_verification='declared existing value; only selected examples rehashed here')
                if m.get('attempted_physics'):
                    executed+=1
                    if not trace.exists() or not m.get('complete'):incomplete.append(str(child))
                    if trace.exists():
                        with zipfile.ZipFile(trace) as archive:
                            present={Path(n).stem for n in archive.namelist()};missing=sorted(set(required)-present)
                            row['missing_robot_response_columns']=missing
                            row['headers']={k:array_header(archive,k) for k in required if k in present}
                        if missing:incomplete.append(str(child))
                else:unavailable+=1
                branches.append(row)
            groups.append(dict(group_id=shape+'/'+cp.name,checkpoint_path=str(cp/'checkpoint.json'),
                train_micro=[r['name'] for r in branches if r['name'].startswith('micro_')],
                heldout_mixed=[r['name'] for r in branches if r['name'].startswith('mixed_')],
                paired_zero=[r['name'] for r in branches if r['name'] in ('zero','zero_replay')],branches=branches))
    cp=base/'capsule/checkpoint_00'
    examples={}
    for name in ('zero','zero_replay','micro_0_minus','mixed_00'):
        trace=cp/name/'trace.npz'
        examples[name]=dict(path=str(trace),sha256=sha(trace) if trace.is_file() else None)
    return dict(schema='phase3_robot_servo_header_inventory_v1',scope='headers and recorded metadata only; no object7 values read, no model fit, no new physics',
        base=str(base),plan_path=str(base/'plan.json'),plan_sha256=sha(base/'plan.json'),
        complete_path=str(base/'execution_complete.json'),complete_sha256=sha(base/'execution_complete.json'),
        source_snapshot_manifest_path=str(base/'source_snapshot_manifest.json'),source_snapshot_manifest_sha256=sha(base/'source_snapshot_manifest.json'),
        checkpoint_groups=len(groups),attempted_physics=executed,unavailable_branches=unavailable,incomplete_or_missing=sorted(set(incomplete)),
        example_pairing=examples,groups=groups,model_admitted=False,
        extraction_rule='Use pinned model raw24 qpos/qvel address map plus transport_lift; never output whole33 qpos or object7.21cmd/8input are references, not instantaneous robot q.')


if __name__=='__main__':
    result=run();path=ROOT/'review/phase3_robot_servo_header_inventory_v1.json'
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    with path.open('x',encoding='utf-8') as f:f.write(payload)
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups',)},ensure_ascii=False,indent=2))

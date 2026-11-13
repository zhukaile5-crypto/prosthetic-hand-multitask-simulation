"""Independent offline contact-output audit; imports no research or physics code."""
from pathlib import Path
import json, hashlib, ast, sys
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/independent_contact_phase2_recompute_20261104.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def shear_all(patches):
    values=np.zeros(5)
    for patch in patches:
        digit=int(patch['digit'])
        if 0<=digit<5:
            tangent=np.asarray(patch['force'])-float(patch['normal'])*np.asarray(patch['normal_dir'])
            values[digit]+=float(np.linalg.norm(tangent))
    return values

def describe_compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return dict(shape=list(a.shape),exact=np.array_equal(a,b),max_abs=float(np.max(np.abs(a-b))) if a.size else 0.)

model_path=ROOT/'calibration/public_contact_model_phase2_v1.json'
script_path=ROOT/'calibration/derive_contact_phase2.py'
old_path=ROOT/'calibration/public_model_phase2_v1.json'
rawroot=ROOT/'calibration/phase2_public_v1'
model=read(model_path); old=read(old_path); plan=read(rawroot/'plan.json')
source=script_path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
shear_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='shear')
keys=sorted({node.slice.value for node in ast.walk(shear_node) if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str)})
# Compile only the small arithmetic shear function, never the calibration module.
isolated={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[shear_node],type_ignores=[])),'<isolated-shear-function>','exec'),isolated)
synthetic=[dict(digit=0,force=[1.,2.,0.],normal=1.,normal_dir=[1.,0.,0.],is_target=False,external_geom=900),dict(digit=0,force=[0.,0.,3.],normal=0.,normal_dir=[0.,1.,0.],is_target=True,external_geom=1)]
synthetic_sum=isolated['shear'](synthetic)
synthetic_flipped=[dict(p,is_target=not p['is_target'],external_geom=-500) for p in synthetic]
synthetic_flipped_sum=isolated['shear'](synthetic_flipped)

result=dict(review_kind='independent_offline_numpy_contact_audit',reviewed_utc=datetime.now(timezone.utc).isoformat(),physics_executed=False,production_modules_imported=False,model_sha256=sha(model_path),source_script_sha256=sha(script_path),source_public_11_model_sha256=sha(old_path),plan_sha256=sha(rawroot/'plan.json'),model_admitted=model['model_admitted'],source_hash_matches=model['source_script_sha256']==sha(script_path),original_hash_matches=model['source_public_11_model_sha256']==sha(old_path),plan_hash_matches=model['source_phase2_plan_sha256']==sha(rawroot/'plan.json'),synthetic_target_invariance=dict(source_patch_fields=keys,all_patch_shear=synthetic_sum.tolist(),flipped_labels=synthetic_flipped_sum.tolist(),expected=[5.,0.,0.,0.,0.],exact=np.array_equal(synthetic_sum,np.array([5.,0.,0.,0.,0.])) and np.array_equal(synthetic_sum,synthetic_flipped_sum)),threshold_declaration=model['new_shear_engineering_reference'],causal_history_note=model['causal_history_note'],shapes=[])
archive=[]
for item in model['raw_archive_manifest']:
    p=Path(item['path']);archive.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),matches=sha(p)==item['archived_sha256']))
result['archive_identity']=dict(count=len(archive),all_match=all(row['matches'] for row in archive),records=archive)
idx=np.arange(26)*4;subset=np.r_[np.arange(5),np.arange(10,16)]
old_by_shape={case['shape']:case for case in old['shapes']}
for case in model['shapes']:
    shape=case['shape'];folder=rawroot/shape;meta={};data={};patch_count=0;non_target_finger_count=0;trace_checks=[]
    for p in sorted(folder.glob('*/metadata.json')):
        label=p.parent.name;m=read(p);meta[label]=m
        if not m['available']:continue
        with np.load(p.with_name('trace.npz'),allow_pickle=False) as z:
            records=[json.loads(str(raw)) for raw in z['raw_patches']]
            tau=np.stack([shear_all(patches) for patches in records])
            for patches in records:
                patch_count+=len(patches)
                non_target_finger_count+=sum(0<=int(q['digit'])<5 and not q.get('is_target',False) for q in patches)
            # Raw original outputs are retained without alteration; only tau is inserted.
            data[label]={'current_palm':np.c_[z['y_palm'][:,:5],tau,z['y_palm'][:,5:]],'initial_fixed_reference':np.c_[z['y'][:,:5],tau,z['y'][:,5:]],'inputs':z['input_reference'].copy()}
        trace_checks.append(dict(label=label,sha_matches=sha(p.with_name('trace.npz'))==m['trace_sha256']))
    shape_result=dict(shape=shape,available_branches=len(data),unavailable_branches=[label for label,m in meta.items() if not m['available']],raw_patch_count=patch_count,non_target_finger_patch_count=non_target_finger_count,all_metadata_trace_sha_match=all(x['sha_matches'] for x in trace_checks),coordinates={})
    for frame in ('current_palm','initial_fixed_reference'):
        dense=np.zeros((101,16,8));identified=np.zeros(8,bool);zero=data['zero'][frame]
        for axis in range(8):
            signs=[sign for sign in ('plus','minus') if meta[f'train_{axis}_{sign}']['available']]
            if not signs:continue
            identified[axis]=True
            if len(signs)==2:
                plus,minus=f'train_{axis}_plus',f'train_{axis}_minus'
                amplitude=data[plus]['inputs'][-1,axis]-data[minus]['inputs'][-1,axis]
                dense[:,:,axis]=(data[plus][frame]-data[minus][frame])/amplitude
            else:
                label=f'train_{axis}_{signs[0]}'
                amplitude=data[label]['inputs'][-1,axis]-data[label]['inputs'][0,axis]
                dense[:,:,axis]=(data[label][frame]-zero)/amplitude
        saved=case['coordinates'][frame];reference=old_by_shape[shape]['coordinates'][frame]
        checks=dict(full_dense=describe_compare(dense,saved['dense_response_to_20ms_interval_increment']),all_20ms_kernel=describe_compare(dense[idx],saved['kernel_response_to_20ms_interval_increment']),five_shear_20ms_kernel=describe_compare(dense[idx,5:10],np.asarray(saved['kernel_response_to_20ms_interval_increment'])[:,5:10]),original_11_unchanged=describe_compare(np.asarray(saved['dense_response_to_20ms_interval_increment'])[:,subset],reference['response_to_20ms_interval_increment']),recomputed_original_11=describe_compare(dense[:,subset],reference['response_to_20ms_interval_increment']),identified_inputs_exact=np.array_equal(identified,saved['identified_inputs']))
        kernel=dense[idx,5:10];all_errors={str(h):[] for h in plan['forecast_horizons_s']};causal_max=0.;causal_count=0;differential=[]
        saved_detail={d['sequence']:d for d in saved['individual_holdout']}
        for q in range(8):
            label=f'holdout_{q:02d}'
            if not meta[label]['available']:continue
            measured=data[label][frame][idx,5:10];path=data[label]['inputs'][idx];change=np.diff(path,axis=0);modeled=np.zeros_like(measured)
            for k in range(1,len(path)):
                for j in range(k):modeled[k]+=kernel[k-j]@change[j]
            residual=measured-modeled;differential.append(measured-zero[idx,5:10]-modeled)
            recorded=saved_detail[q]['complete_causal_errors'];error_rows=[]
            for origin in range(len(path)):
                now=origin*.02;past_time=np.arange(origin+1)*.02;past_values=residual[:origin+1]
                take=past_time>=now-plan['causal_window_s']-1e-10;ts=past_time[take];ys=past_values[take]
                slope=np.zeros(5);level=np.zeros(5)
                for output in range(5):
                    if len(ts)>=4:
                        centered=ts-ts.mean();slope[output]=centered@(ys[:,output]-ys[:,output].mean())/max(1e-12,float(centered@centered));level[output]=ys[:,output].mean()+slope[output]*(now-ts.mean())
                    else:level[output]=ys[-1,output]
                for h in plan['forecast_horizons_s']:
                    target=origin+round(h/.02)
                    if target>=len(path):continue
                    error=level+h*slope+modeled[target]-measured[target];error_rows.append((now,h,error));all_errors[str(h)].append(error)
            assert len(recorded)==len(error_rows)
            for row,(now,h,error) in zip(recorded,error_rows):
                assert row['origin_s']==now and row['horizon_s']==h
                causal_max=max(causal_max,float(np.max(abs(np.asarray(row['error'])[5:10]-error))));causal_count+=5
        causal_summaries={}
        for h,errors in all_errors.items():
            e=np.asarray(errors);s=saved['complete_causal_by_horizon'][h]['groups']['shear_magnitude']
            causal_summaries[h]=dict(vectors=len(e),rmse=describe_compare(np.sqrt(np.mean(e**2,axis=0)),s['rmse_per_output']),p95=describe_compare(np.quantile(abs(e),.95,axis=0),s['p95_abs_per_output']),maximum=describe_compare(np.max(abs(e),axis=0),s['maximum_abs_per_output']))
        checks['causal_shear_error_values']=dict(count=causal_count,max_abs_difference=causal_max,history_used='holdout acquired t0, followed only by residuals at indices <= origin; negative-time shear excluded; no zero trajectory as prehistory')
        checks['causal_summary']=causal_summaries
        e=np.concatenate(differential);s=saved['differential_response_error']['groups']['shear_magnitude'];checks['differential_shear_rmse']=describe_compare(np.sqrt(np.mean(e**2,axis=0)),s['rmse_per_output'])
        shape_result['coordinates'][frame]=checks
    result['shapes'].append(shape_result)
result['source_review_line_references']={'all_patch_shear':'calibration/derive_contact_phase2.py:24-31,64-66,122','original11_insertion_and_comparison':'34,70-79,103-109','causal_no_future_zero':'80-101,124','post_data_exploratory_threshold':'49-51,123','not_admitted':'116-128; public_contact_model_phase2_v1.json:4,366-371'}
result['limitations']=['This verifies offline derivation, not calibration admission, generalization, controller performance or physical safety.','Raw records may contain exclusively target contacts; the isolated arithmetic synthetic test separately verifies inclusion of a non-target patch and invariance to target/external labels.','Future holdout input references in the known planned response are not future output residuals; causal drift estimation uses only acquired t0 and past residuals.']
result['overall_model_admission_issued']=False
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),model_sha256=result['model_sha256'],source_sha256=result['source_script_sha256'],all_archived_raw_hashes_match=result['archive_identity']['all_match'],model_admitted=result['model_admitted'],shape_rows=[dict(shape=x['shape'],coords={k:dict(shear_kernel_max=v['five_shear_20ms_kernel']['max_abs'],original11_max=v['original_11_unchanged']['max_abs'],causal_shear_error_max=v['causal_shear_error_values']['max_abs_difference'],causal_values=v['causal_shear_error_values']['count']) for k,v in x['coordinates'].items()}) for x in result['shapes']]),ensure_ascii=False))

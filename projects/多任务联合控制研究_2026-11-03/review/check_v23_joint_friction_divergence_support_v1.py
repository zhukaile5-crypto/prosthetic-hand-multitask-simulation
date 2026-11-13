"""Supplementary original-normal certificate and provenance checks; no optimization."""
from pathlib import Path
import sys,json,gzip,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'review'))
from v21_original_matrix_kkt_verifier_v1 import verify_stage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=ROOT/'review/v23_joint_friction_first_divergence_v1.json';j=json.loads(p.read_text('utf8'));rows=[];meta={}
for v in ['v22','v23']:
 run=ROOT/f'output/{v}_small_joint_friction_drop_transport_619742_v1/runs/friction_drop__ratio_effort__619742';meta[v]=json.loads((run/'metadata.json').read_text('utf8'))
 with gzip.open(run/'release_decisions.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   t=float(re.search(r'"time_s"\s*:\s*([0-9.eE+-]+)',line).group(1))
   if round(t*200) in [2726,2727,2728]:
    z=json.loads(line);d=z['hierarchy']['diagnostics'];c=d['normal_witness_certified_prefix'][0];hd=c['hard_domain'];x=np.asarray(c['x']);n=len(x);E=np.asarray(hd['fixed_equality_matrix']).reshape((-1,n));rhs=np.asarray(hd['fixed_equality_target']);fr=c['previous_frozen'];E=np.vstack([E]+[np.asarray(a['matrix']) for a in fr]);rhs=np.r_[rhs,*[np.asarray(a['attained']) for a in fr]];k=c['KKT'];pr=dict(kind='least_squares',A=c['matrix'],b=c['target'],G=hd['original_inequality_matrix_with_box'],h=hd['original_inequality_upper_with_box'],E=E,f=rhs,previous_frozen=fr,atol=1e-9,rtol=1e-8);r=verify_stage(pr,x,duals=dict(inequality_unit=k['full_normalized_inequality_dual'],equality_raw=k['equality_raw_dual'],provenance='saved actual normal witness checked independently in original rows'));rows.append(dict(variant=v,time_s=t,normal_independent_KKT=r['kkt_passed'],objective=r.get('objective'),stationarity_inf=r.get('stationarity_inf'),strictbox=bool(np.all(x>=np.asarray(hd['lower'])) and np.all(x<=np.asarray(hd['upper']))),cert_x_equals_saved_xA=np.array_equal(x.view(np.uint64),np.asarray(d['certified_L1_extended_anchor']).view(np.uint64)),original_problem_sha256=r.get('original_problem_sha256')))
   if t>=13.64:break
from v23_task_execution.runtime import source_files
actual_paths=source_files(['ratio_effort'])
old=meta['v22']['research_source_sha256'];new=meta['v23']['research_source_sha256'];pins=dict(old_count179=len(old)==179,new_count186=len(new)==186,old179_subset_new186=all(new.get(k)==v for k,v in old.items()),new186_matches_probe_map=new==j['source_sha256'],both_maps_match_current_file_bytes=all(k in actual_paths and sha(actual_paths[k])==v for k,v in new.items()),old42_and_numeric_identity_same=meta['v22']['numerical_identity']==meta['v23']['numerical_identity']==j['numerical_identity'])
cf=j['local_same_public_input_counterfactuals'];same_backend=[]
for label in ['v22_f304','v23_4b2ac6']:
 a=np.asarray(cf[1]['solutions'][label]['actual_reference21_if_emitted']);b=np.asarray(cf[2]['solutions'][label]['actual_reference21_if_emitted']);same_backend.append(dict(backend=label,changed_only_old_vs_new_saved_input_at13_64=True,reference_difference_inf=float(np.max(abs(a-b)))))
checks=dict(all_six_saved_normal_KKT_and_strictbox=all(z['normal_independent_KKT'] and z['strictbox'] and z['cert_x_equals_saved_xA'] for z in rows),all_18_counterfactual_submitted_certificates_independent_KKT=all(c['independent_KKT'] for row in cf for sol in row['solutions'].values() for c in sol['original_certificates_independently_checked']),all_six_counterfactuals_complete_and_issuable=all(sol['completed_prefix']==3 and sol['fully_converged'] and sol['strictbox'] and not sol['more_than_roundoff_projection'] and sol['emitted_score_available'] for row in cf for sol in row['solutions'].values()),all_three_same_source_saved_reference_exact=all(sol['same_source_saved_reference_difference']['exact'] for row in cf for sol in row['solutions'].values() if sol['same_source_saved_reference_difference'] is not None),all_pins=all(pins.values()))
out=ROOT/'review/v23_joint_friction_divergence_support_v1.json';assert not out.exists();payload=dict(schema='v23_joint_friction_divergence_support_v1',complete=True,passed=all(checks.values()),checks=checks,pins=pins,normal_witness_records=rows,same_backend_across_two_saved_inputs=same_backend,first_divergence_evidence_sha256=sha(p),checker_sha256=sha(__file__),physics_steps=0,model_constructions=0,optimizer_calls=0);out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({'passed':payload['passed'],'checks':checks,'same_backend':same_backend,'out':str(out),'sha256':sha(out)},ensure_ascii=False))

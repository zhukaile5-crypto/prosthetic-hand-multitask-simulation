"""Compare published author tables with independently audited raw quantities."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'review/release_primitive_prospective_v2_independent_v1.json'
AUTHOR=ROOT/'report/release_primitive_prospective_v2_author_v1/readout.json'
OUT=ROOT/'review/release_primitive_prospective_v2_author_crosscheck_v1.json'
if OUT.exists():raise FileExistsError(OUT)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
a=read(AUDIT);b=read(AUTHOR)
if not a['passed'] or a['issues']:raise RuntimeError('Independent raw audit must first pass')
checks=[]
def check(name,value,details=None):checks.append(dict(name=name,passed=bool(value),details=details))
def metric(name,x,y):
 ok=x is y if x is None or y is None else x==y if isinstance(x,(bool,str)) or isinstance(y,(bool,str)) else bool(np.allclose(x,y,rtol=1e-10,atol=1e-10))
 check(name,ok,None if ok else dict(independent=x,author=y))
def latency(value,start):return None if value is None else value-start
def values(r):
 start=r['actual_mechanical_release_s'];coarse=r['clearance_200Hz'];fine=r['clearance_raw_end']
 return dict(mission=r['mission'],full_motion_mission=r['full_motion'],guard=r['safety']['guard_pass'],preparation=r['pre_release_qualifiers']['handoff_passed'],physical_release200Hz=r['release']['physical_success'],task_qualified_release=r['release']['task_qualified_success'],force_peak_N=max(r['safety']['finger_peak_N']),local_excess_Ns=r['safety']['local_excess_integral_Ns'],total_excess_Ns=r['safety']['total_excess_integral_Ns'],initial_stability_s=r['pre_release_qualifiers']['initial_stability_wait_s'],original_first_clear200Hz_s=r['release']['physical_clearance_latency_s'],first_any_frame_s=latency(coarse['first_any_confirmation_s'],start),first_by_deadline_frame_s=latency(coarse['first_by_deadline_confirmation_s'],start),first_any_raw_end_s=latency(fine['first_any_confirmation_s'],start),first_by_deadline_raw_end_s=latency(fine['first_by_deadline_confirmation_s'],start),terminal_low_force_raw_end_s=latency(fine['terminal_clear_confirmation_s'],start),raw_recontact_episodes=len(fine['recontact_episodes']))
ind={key:values(r) for key,r in a['records'].items() if r['status']=='complete'}
check('exact32_author_rows_not_selected_by_success',len(b['rows'])==32 and {r['job']['job_id'] for r in b['rows']}==set(a['records']))
for r in b['rows']:
 key=r['job']['job_id'];ours=a['records'][key]
 metric(key+'/actual_offset_exposure',ours['offset_enabled'],r['offset_enabled'])
 for k,v in ind[key].items():metric(key+'/'+k,v,r['values'][k])
for variant in ['zero','lift_slow']:
 rows=[r for r in a['records'].values() if r['job']['variant']==variant];group=b['groups'][variant]
 metric(variant+'/all_assigned',16,group['assigned']);metric(variant+'/complete',len(rows),group['complete']);metric(variant+'/eligibility',sum(r['eligible'] for r in rows),group['entry_eligible']);metric(variant+'/actual_exposure',sum(r['offset_enabled'] for r in rows),group['offset_enabled'])
 for k in ['mission','full_motion_mission','guard','preparation','physical_release200Hz','task_qualified_release']:
  vals=[ind[r['job']['job_id']][k] for r in rows];metric(variant+'/'+k+'/full_denominator',16,group['binary'][k]['assigned']);metric(variant+'/'+k+'/true',sum(v is True for v in vals),group['binary'][k]['true']);metric(variant+'/'+k+'/unavailable',sum(v is None for v in vals),group['binary'][k]['unavailable'])
 for k in ['force_peak_N','local_excess_Ns','total_excess_Ns','original_first_clear200Hz_s','first_by_deadline_raw_end_s','terminal_low_force_raw_end_s']:
  vals=[ind[r['job']['job_id']][k] for r in rows];good=[v for v in vals if v is not None];metric(variant+'/'+k+'/available',len(good),group['scalar'][k]['available']);metric(variant+'/'+k+'/mean',float(np.mean(good)) if good else None,group['scalar'][k]['mean'])
numeric_full_pairs=[]
for p in b['pairs']:
 prefix=f"{p['scenario']}__{p['motion_mode']}__{p['seed']}__";z=a['records'][prefix+'zero'];s=a['records'][prefix+'lift_slow'];zv=ind[prefix+'zero'];sv=ind[prefix+'lift_slow']
 metric(prefix+'/assigned_strategy_exposure',s['offset_enabled'],p['component_exposed'])
 for k in ['force_peak_N','local_excess_Ns','total_excess_Ns','original_first_clear200Hz_s','first_by_deadline_raw_end_s','terminal_low_force_raw_end_s']:
  delta=None if zv[k] is None or sv[k] is None else sv[k]-zv[k];metric(prefix+'/paired_difference/'+k,delta,p['difference_slow_minus_zero'][k])
 # All observed fallback pairs are genuine native-vs-native executions. Check
 # four complete numeric state/control/pressure channels, not merely totals.
 fields={}
 if not z['offset_enabled'] and not s['offset_enabled']:
  zp=Path(z['job']['directory'])/'record/trace.npz';sp=Path(s['job']['directory'])/'record/trace.npz'
  with np.load(zp,allow_pickle=False) as x,np.load(sp,allow_pickle=False) as y:
   for name in ['qpos','qvel','ctrl','skin_normal']:
    xx,yy=x[name],y[name];fields[name]=xx.dtype==yy.dtype and xx.shape==yy.shape and xx.tobytes()==yy.tobytes()
  check(prefix+'/fallback_whole16s_four_numeric_arrays_exact',all(fields.values()),fields)
 numeric_full_pairs.append(dict(scenario=p['scenario'],motion_mode=p['motion_mode'],seed=p['seed'],offset_enabled=s['offset_enabled'],full16_numeric_fields=fields))
check('two_physical_clusters_not16_independent_subjects',b['scope']['physical_seed_clusters']==2 and b['scope']['assigned_trials']==32 and b['scope']['paired_contexts']==16)
check('author_checker_provenance_stable',b['source']['checker_sha256']==sha(ROOT/'review/readout_release_primitive_prospective_v2.py'))
failure_counts=dict(Counter(r['pre_release_qualifiers']['preparation_failure'] for r in a['records'].values()))
report=dict(schema='release_primitive_prospective_v2_author_crosscheck_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(c['passed'] for c in checks),checks=checks,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [AUDIT,AUTHOR,ROOT/'review/readout_release_primitive_prospective_v2.py']},checker_sha256=sha(Path(__file__)),pre_release_failure_reasons=failure_counts,full16_fallback_pairs=numeric_full_pairs,physics_steps=0,interpretation='All32 fail the calibrated handoff domain, not the physical readiness test. The candidate offset is enabled0/32. These native-vs-native assigned-strategy comparisons test applicability/gating and replay, not efficacy of the active release slowdown. Raw low-force deadline observations are not qualified physical-release success. No seed substitution, successful-only cohort, composite score or force safety claim.',scope='All32 author rows for17 independent raw/mission/clearance scalars/binaries; full16pair whole-safety and timing differences; group denominators/means; four complete numeric arrays per unexposed pair. Other author TV fields are not claimed independently recomputed in this crosscheck.')
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),passed=report['passed'],checks=len(checks),issues=sum(not c['passed'] for c in checks),failure_reasons=failure_counts,physics_steps=0)))

"""Compact facts from complete frozen readouts only; no new metric formulas."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def stat(v):
    known=[float(x) for x in v if x is not None]
    return dict(assigned=len(v),known=len(known),unknown=len(v)-len(known),mean=None if not known else float(np.mean(known)),median=None if not known else float(np.median(known)),min=None if not known else min(known),max=None if not known else max(known))
def select(r):
    v=r['metrics'];o=r['bundle']['diagnostics']['original_result'];m=o['research_metrics'];out={}
    for w in ('static','dynamic','whole_holding'):
        for q in ('mean_total_force_N','finger_TV_rate_N_s','finger_reversal_rate_N_s','total_force_SD_N','total_force_CV'):out[w+'.raw.'+q]=v.get('raw.'+w+'.'+q)
        out[w+'.eligible.TV_raw_N_s']=v.get('eligible.'+w+'.raw_200Hz.finger_TV_rate_N_s')
        out[w+'.eligible.TV_bandlimited_N_s']=v.get('eligible.'+w+'.bandlimited.finger_TV_rate_N_s')
        key=dict(static='static_quality',dynamic='dynamic_quality',whole_holding='entire_holding_quality')[w];q=m.get(key) or {}
        out[w+'.eligible.mean_N']=q.get('mean_actual_finger_force_N') if q.get('eligible') is True else None
    out['body.max_relative_drift_m']=v.get('object.whole_holding.relative_handoff_max_displacement_m')
    for axis,name in enumerate(('lift_m','WRJ2_rad','WRJ1_rad')):out['tracking.holding.RMSE_'+name]=v.get('original.progress.holding_motion_tracking.rmse.'+str(axis))
    for label,stage in [('full',None),('holding','policy_holding_transport'),('release','mechanical_release_and_after')]:
        s=m['safety'] if stage is None else m['safety']['stages'][stage]
        peak=s.get('finger_peak_N');out[label+'.peak_N']=None if peak is None else max(peak)
        out[label+'.dose_Ns']=s.get('local_excess_integral_Ns')
    out['initial_wait_s']=m.get('initial_stability_wait_s');out['release.clearance_s']=m.get('release',{}).get('physical_clearance_latency_s')
    return out
def main():
    path=ROOT/'report/v17b_prospective80_v1/paired_readout_v1.json';auxpath=ROOT/'review/v17b_delivery_auxiliary_readout_v1.json'
    x=read(path);a=read(auxpath)
    if not x['complete'] or not a['complete'] or x['delivery_auxiliary_reference']['sha256']!=sha(auxpath):raise ValueError('Complete exact artifacts required')
    aux={r['record_id']:r for r in a['records']};rows=[]
    for r in x['records']:
        m=r['bundle']['diagnostics']['original_result']['research_metrics'];rel=m['release'];aa=aux[r['record_id']]['auxiliary'];o=r['bundle']['diagnostics']['original_result']
        flags=dict(task=o['task_success'],mission=m['mission_success'],fullmotion=m['full_motion_mission_success'],preparation=m['preparation_passed'],policy=m['policy_exposed'],
            physical_exposure=rel['physical_exposure'],physical_clear=rel['physical_success'],qualified_exposure=rel['task_qualified_exposure'],qualified_clear=rel['task_qualified_success'],guard=m['safety']['guard_pass'],
            floor=aa['ever_floor_contact'])
        rows.append(dict(record_id=r['record_id'],record_key=r['record_key'],role=r['role'],context_key=r['context_key'],flags=flags,values=select(r),failure_reason=m.get('failure_reason'),failure_at=m.get('failure_at')))
    roles=sorted({r['role'] for r in rows});keys=sorted(rows[0]['values'])
    def groups(rr):
        ans={}
        for role in roles:
            rs=[r for r in rr if r['role']==role]
            ans[role]=dict(assigned=len(rs),counts={k:dict(true=sum(r['flags'][k] is True for r in rs),false=sum(r['flags'][k] is False for r in rs),unknown=sum(r['flags'][k] is None for r in rs)) for k in rows[0]['flags']},metrics={k:stat([r['values'][k] for r in rs]) for k in keys})
        return ans
    by={r['record_id']:r for r in rows};pairs=[]
    for p in x['pairs']:
        ar=by[p['candidate_record_id']];br=by[p['reference_record_id']];diff={}
        for k in keys:
            av=ar['values'][k];bv=br['values'][k];ok=av is not None and bv is not None
            diff[k]=dict(delta=av-bv if ok else None,relative=(av-bv)/bv if ok and bv!=0 else None)
        pairs.append(dict(context_key=p['context_key'],contrast=p['contrast'],differences=diff))
    contrasts=sorted({tuple(p['contrast']) for p in pairs})
    def ps(pp):
        return {ar+'__minus__'+br:{k:dict(delta=stat([p['differences'][k]['delta'] for p in pp if p['contrast']==[ar,br]]),relative=stat([p['differences'][k]['relative'] for p in pp if p['contrast']==[ar,br]])) for k in keys} for ar,br in contrasts}
    result=dict(schema='v17b_complete80_compact_facts_v1',complete=True,records=rows,
        all_roles=groups(rows),roles_by_seed={str(s):groups([r for r in rows if r['context_key'][2]==s]) for s in (619741,619742)},
        roles_by_motion={s:groups([r for r in rows if r['context_key'][1]==s]) for s in ('hold','transport')},
        contrasts=ps(pairs),contrasts_by_seed={str(s):ps([p for p in pairs if p['context_key'][2]==s]) for s in (619741,619742)},contrasts_by_motion={s:ps([p for p in pairs if p['context_key'][1]==s]) for s in ('hold','transport')},
        source_readout_sha256=sha(path),auxiliary_sha256=sha(auxpath),reader_sha256=sha(__file__),scope='Descriptive only, all16 each arm, no new success/threshold/weightedscore/CI; eligible unknown and zero-baseline relative preserved; auxiliary stability attribution is separate')
    out=ROOT/'report/v17b_prospective80_v1/compact_facts_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(out=str(out),sha256=sha(out))))
    fields=['static.raw.mean_total_force_N','static.eligible.TV_raw_N_s','dynamic.eligible.TV_raw_N_s','whole_holding.eligible.TV_raw_N_s','whole_holding.raw.finger_TV_rate_N_s','body.max_relative_drift_m','full.peak_N','full.dose_Ns','release.peak_N','release.dose_Ns','release.clearance_s']
    for c in [('full_V17','same_prior_legacy_ratio'),('full_V17','holding_A')]:
        print('CONTRAST',c)
        for k in fields:print(k,json.dumps(result['contrasts'][c[0]+'__minus__'+c[1]][k]))
    for name,g in [('seed'+s,v) for s,v in result['roles_by_seed'].items()]+list(result['roles_by_motion'].items()):
        print('STRATUM',name,{r:{k:v['counts'][k]['true'] for k in ('task','fullmotion','guard','floor')} for r,v in g.items()})
if __name__=='__main__':main()

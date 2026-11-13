"""Six final cached contact-pressure snapshots only; no new success metric."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def end_pressure(run):
    path=run/'delivery_intervals.npz'
    fields=('interval_start_s','interval_end_s','contact_valid','target_hand_normal_N','target_receiver_top_normal_N','target_floor_normal_N')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in fields}
    if not len(a['interval_end_s']):return dict(available=False,reason='no_saved_interval',input_sha256=sha(path))
    i=len(a['interval_end_s'])-1;valid=bool(a['contact_valid'][i])
    forces={k:(float(a[k][i]) if valid and np.isfinite(a[k][i]) else None) for k in fields[3:]}
    return dict(available=valid and all(v is not None for v in forces.values()),
        interval_start_s=float(a['interval_start_s'][i]),interval_end_s=float(a['interval_end_s'][i]),
        **forces,input_sha256=sha(path),scope='offline target-specific summed cached normal contact pressure, not signed support H or fresh-boundary sensor')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    ab=ROOT/'review/v19_motion_ablation_auxiliary_v1.json';full=ROOT/'review/v19_small_auxiliary_v1.json';planpath=ROOT/'review/v19_motion_ablation_plan_v1.json'
    new=read(ab);old=read(full);plan=read(planpath)
    if not new['complete'] or new['issues']:raise ValueError('complete readout required')
    if sha(full)!=new['original_full_v19_auxiliary_sha256'] or sha(planpath)!=new['plan_sha256']:raise ValueError('readout parent chain changed')
    rows=[]
    for r in new['records']:
        s=next(x for x in plan['stages'] if x['name']==r['stage']);reference=next(x for x in old['records'] if x['context_key']==r['context_key'] and x['stage']!='passive_ratio_transport')
        refpath=ROOT/s['full_reference']
        if refpath.resolve()!=Path(reference['run']).resolve():raise ValueError('Full reference mismatch')
        a=end_pressure(Path(r['run']));b=end_pressure(refpath)
        rows.append(dict(stage=r['stage'],context_key=r['context_key'],NoMotion=a,FullV19=b,
            target_hand_pressure_difference_N=None if a['target_hand_normal_N'] is None or b['target_hand_normal_N'] is None else a['target_hand_normal_N']-b['target_hand_normal_N']))
    payload=dict(schema='v19_ablation_terminal_cached_pressure_v1',complete=len(rows)==3 and all(r['NoMotion']['available'] and r['FullV19']['available'] for r in rows),
        records=rows,ablation_auxiliary_sha256=sha(ab),full_auxiliary_sha256=sha(full),plan_sha256=sha(planpath),reader_sha256=sha(__file__),
        physics_steps=0,new_threshold_or_success_label=False,pre12_raw_prefix_checked_here=False,
        scope='final interval only, additional pressure magnitude alongside frozen clearance labels; all6 arrays loaded once, unknown not0')
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False));print(json.dumps(dict(out=str(out),sha256=sha(out))))
if __name__=='__main__':main()

import pathlib,json,numpy as np,csv
root=pathlib.Path(r'D:/生物医学工程原理PBL/多任务联合控制研究_2026-11-03')
base=root/'output/development_v5_physical_all16/runs';out=root/'review/independent_broad_collision_outcome_20261104'
records=[]
for method in ['ratio','joint','ratio_release','joint_release','passive']:
 with np.load(base/f'collision__{method}__610212'/'trace.npz') as z:
  a={k:z[k].copy() for k in ['substep_time','substep_duration','substep_normal']}
 t=a['substep_time']+a['substep_duration'];c=a['substep_normal']>.2;s=c[:,0]&(c.sum(axis=1)>=3);segs=[];first=None;last=None
 for i in np.flatnonzero((t>=9.3-1e-9)&(t<12.-1e-9)):
  if not s[i]:
   if first is None:first=i
   last=i
  elif first is not None:
   segs.append(dict(first_end_time=float(t[first]),last_end_time=float(t[last]),elapsed_s=float(t[last]-t[first]),samples=int(last-first+1)));first=None
 if first is not None:segs.append(dict(first_end_time=float(t[first]),last_end_time=float(t[last]),elapsed_s=float(t[last]-t[first]),samples=int(last-first+1)))
 records.append(dict(method=method,clock='substep_time + substep_duration: after-integration force timestamp',no_support_segments=segs,max_no_support_elapsed_s=max(x['elapsed_s'] for x in segs)))
 if method in ['ratio','joint']:
  with (out/f'collision__{method}__610212_substep_truth_9400_9800.tsv').open('w',encoding='utf-8',newline='') as fp:
   w=csv.writer(fp,delimiter='\t');w.writerow(['substep_index','force_end_time_s','normal_thumb_N','normal_index_N','normal_middle_N','normal_ring_N','normal_little_N','n_contacts_gt_0.2N','thumb_contact','physical_support'])
   for i in np.flatnonzero((t>=9.4-1e-9)&(t<=9.8+1e-9)):
    w.writerow([int(i),f'{t[i]:.4f}',*[f'{v:.12g}' for v in a['substep_normal'][i]],int(c[i].sum()),int(c[i,0]),int(s[i])])
(out/'collision_substep_evidence.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('Substep evidence saved',len(records))

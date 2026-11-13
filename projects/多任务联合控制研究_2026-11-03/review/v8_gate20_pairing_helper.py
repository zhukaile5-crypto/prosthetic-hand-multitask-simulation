"""Read-only V8 terminal source/pairing/numeric-array audit; no app imports."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/development_v8_counterexamples_v1'
OLD=ROOT/'output/development_v7_counterexamples_v1'
OUT=ROOT/'review/v8_gate20_pairing_independent_v1.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def numeric_keys(path):
    names=[]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:raise ValueError('Unsupported NPY header version: '+str(version))
                shape,order,dtype=reader(stream)
            if dtype.kind in 'biufc':names.append(name.removesuffix('.npy'))
    return set(names)

finish=read(RUN/'execution_complete.json')
if finish.get('execution_complete') is not True:raise RuntimeError('Wait for terminal true before reading complete matrix')
manifest=read(RUN/'manifest.json');plan=read(RUN/'plan.json');rows=read(RUN/'trials.json');ledger=read(RUN/'assignment_ledger.json');snap=read(RUN/'source_snapshot_manifest.json')
key=lambda x:(x['scenario'],x['method'],x['seed'])
bykey={key(x):x for x in rows};jobs=manifest['jobs'];checks=[];issues=[];sources=[];records=[];groups=defaultdict(dict);paths={}
def check(name,ok,detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok:issues.append(dict(name=name,detail=detail))
check('exact20assignment_trial_keys',len(jobs)==len(rows)==len(ledger)==20 and len(set(map(key,jobs)))==20 and Counter(map(key,jobs))==Counter(map(key,rows))==Counter(key(x['job']) for x in ledger) and plan['jobs']==jobs)
check('all_assignments_evaluated',all(x['status']=='evaluated' for x in ledger))
check('terminal_file_hashes',finish['trials']==finish['planned']==20 and finish['runtime_failures']==[] and all(finish[k]==sha(RUN/f) for k,f in [('trials_sha256','trials.json'),('assignment_ledger_sha256','assignment_ledger.json'),('plan_sha256','plan.json'),('manifest_sha256','manifest.json')]))
for name,h in manifest['research_source_sha256'].items():
    entry=snap['files'][name];s=RUN/entry['snapshot_path'];p=Path(entry['original_path']);a=sha(s);b=sha(p)
    sources.append(dict(name=name,expected=h,snapshot=a,current=b,matches=a==b==h==entry['sha256'] and s.stat().st_size==p.stat().st_size==entry['bytes']))
check('all80snapshot_current_identities',len(sources)==80 and all(x['matches'] for x in sources))
check('snapshot_set_and_manifest_bindings',set(snap['files'])==set(manifest['research_source_sha256'])=={p.relative_to(RUN/'sources').as_posix() for p in (RUN/'sources').rglob('*') if p.is_file()} and manifest['source_snapshot_manifest_sha256']==sha(RUN/'source_snapshot_manifest.json') and manifest['plan_sha256']==sha(RUN/'plan.json'))
for job in jobs:
    k=key(job);folder=Path(job['destination']);m=read(folder/'metadata.json');r=read(folder/'result.json');paths[k]=folder/'trace.npz';groups[(k[0],k[2])][k[1]]=m
    u=np.random.default_rng(k[2]+511000).normal(size=4)
    caps=[3.5,2.8,2.8,2.8,2.8] if k[0]=='fragile_conflict' else [5.]*5
    record=dict(key=list(k),result_equals_trial=r==bykey[k],metadata_key_matches=m['method']==k[1] and m['scenario']['name']==k[0] and m['seed']==k[2] and m['split']=='robustness',source_matches=m['research_source_sha256']==manifest['research_source_sha256'],parameters_match=m['config']==r['research_metrics']['physical_parameters'],rng_formula_matches=m['config']['mass']==float(.12*np.clip(1+.06*u[0],.85,1.15)) and m['config']['friction']==float(.6*np.clip(1+.08*u[1],.8,1.2)),noise_profile_matches=all(m['noise'][a]==b for a,b in {'force_sigma':.01,'tangential_sigma':.005,'position_sigma':.0001,'slip_sigma':.0002,'effort_sigma':.02,'delay':.01,'dropout_probability':0.,'dropout_windows':[]}.items()),caps_match=m['limits']==caps and m['budget']==(12. if k[0]=='fragile_conflict' else 18.),trace_sha_linkage=r['trace_sha256']==m['trace_sha256']==r['research_metrics']['trace_sha256'],metadata_sha256=sha(folder/'metadata.json'),result_sha256=sha(folder/'result.json'))
    records.append(record)
check('all20results_metadata_actual_parameters',all(all(v for k,v in x.items() if k not in ['key','metadata_sha256','result_sha256']) for x in records))
paired=[]
for g,mm in sorted(groups.items()):
    base=mm['ratio'];fields=['config','noise','limits','budget','model_sha256','initial_position'];eq={f:all(m[f]==base[f] for m in mm.values()) for f in fields}
    public={f:all(m['task_contract'][f]==base['task_contract'][f] for m in mm.values()) for f in ['user_motion_trajectory','motion_reference_bounds','sensor_characteristics','tracking_tolerances']}
    paired.append(dict(scenario=g[0],seed=g[1],physical_noise_caps_exact=eq,public_context_exact=public,baseline_preview_gains_exact=mm['ratio']['gains']==mm['ratio_preview']['gains'] and mm['passive']['gains']==mm['passive_preview']['gains'] and mm['joint']['actual_ratio_task_gains']==mm['ratio']['gains']))
check('all4five_method_pairs',all(all(x['physical_noise_caps_exact'].values()) and all(x['public_context_exact'].values()) and x['baseline_preview_gains_exact'] for x in paired))
baseline=[];exclude={'control_ms'}
for scene in plan['scenarios']:
    for method in ['ratio','passive','ratio_preview','passive_preview']:
        new=paths[(scene,method,610212)];old=OLD/'runs'/f'{scene}__{method}__610212'/'trace.npz';a_keys=numeric_keys(new)-exclude;b_keys=numeric_keys(old)-exclude;different=[];hashes={}
        with np.load(new,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
            for field in sorted(a_keys&b_keys):
                x=a[field];y=b[field];xb=x.tobytes(order='C');yb=y.tobytes(order='C');same=x.dtype==y.dtype and x.shape==y.shape and xb==yb
                hashes[field]=hashlib.sha256(xb).hexdigest()
                if not same:different.append(dict(field=field,numeric_equal=bool(np.array_equal(x,y,equal_nan=True)),shape_a=x.shape,shape_b=y.shape))
        baseline.append(dict(scenario=scene,method=method,numeric_fields_count=len(a_keys),field_set_exact=a_keys==b_keys,all_selected_numeric_arrays_bit_exact=not different and a_keys==b_keys,mismatches=different,numeric_array_sha256=hashes))
check('all16traditional_baselines_all_numeric_arrays_bit_exact',len(baseline)==16 and all(x['all_selected_numeric_arrays_bit_exact'] for x in baseline),[x for x in baseline if not x['all_selected_numeric_arrays_bit_exact']])
motion=[];rates=np.array([.035,.12,.12])
for scene in plan['scenarios']:
    with np.load(paths[(scene,'ratio_preview',610212)],allow_pickle=False) as p,np.load(paths[(scene,'joint',610212)],allow_pickle=False) as j:
        t=j['time'];mask=(t>=7.5-1e-9)&(t<12-1e-9);a=j['executed_motion_reference'];b=p['executed_motion_reference'];target=j['actual_motion_target'];delta=a-np.vstack((a[:1],a[:-1]));td=target-np.vstack((target[:1],target[:-1]));moving=np.any(abs(td)>1e-10,axis=1)&mask;stationary=np.all(abs(delta)<=1e-10,axis=1)
        motion.append(dict(scenario=scene,user_target_exact=np.array_equal(target,p['actual_motion_target']),joint_vs_preview_max_abs_reference_difference=abs(a[mask]-b[mask]).max(axis=0).tolist(),joint_vs_preview_rms_reference_difference=np.sqrt(np.mean((a[mask]-b[mask])**2,axis=0)).tolist(),holding_intervals=int(mask.sum()),reference_stationary_fraction=float(stationary[mask].mean()),user_moving_intervals=int(moving.sum()),stationary_reference_while_user_moves_fraction=None if not moving.any() else float(stationary[moving].mean()),reference_rate_violation_counts=(abs(delta[mask])>rates*.005+1e-9).sum(axis=0).tolist(),pause_definition_limit='Reference stationarity is not a risk-hold flag: static user tasks and legal task-tube minimum intervention can also be stationary. Parent independently audits recovery debug holds.'))
check('user_targets_same_and_joint_public_reference_rates_respected',all(x['user_target_exact'] and x['reference_rate_violation_counts']==[0,0,0] for x in motion))
prep=[]
for scene in ['nominal','fragile_conflict']:
    with np.load(paths[(scene,'ratio',610212)],allow_pickle=False) as a:
        t=a['time'];mask=t<7.5-1e-9;cc=[]
        for method in ['passive','ratio_preview','passive_preview','joint']:
            with np.load(paths[(scene,method,610212)],allow_pickle=False) as b:
                eq={f:bool(np.array_equal(a[f][mask],b[f][mask],equal_nan=True)) for f in ['qpos','qvel','alpha_before','alpha','ctrl_before','ctrl','lift','lift_ref','controller_active']};cc.append(dict(method=method,fields_exact=eq,all_exact=all(eq.values())))
        prep.append(dict(scenario=scene,control_samples=int(mask.sum()),window='t<7.5-1e-9',comparisons=cc))
check('representative_prepare_physics_command_exact',all(c['all_exact'] for x in prep for c in x['comparisons']))
report=dict(schema='v8_gate20_pairing_independent_v1',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,issues=issues,input_sha256={n:sha(RUN/n) for n in ['execution_complete.json','manifest.json','plan.json','trials.json','assignment_ledger.json','source_snapshot_manifest.json']},all80source_snapshot_current_checks=sources,all20result_parameter_checks=records,all4five_method_pairing=paired,all16baseline_full_numeric_exact=baseline,joint_motion_deviation_and_stationarity=motion,representative_preparation=prep,reported_risk_hold_context=dict(source='parent independent raw/recovery debug audit',collision_hold_s=.88,fragile_conflict_hold_s=.04,dynamic_transport_hold_s=0.,interpretation='Reported holds occur in static public-motion scenes; they do not validate moving-risk pause/recovery or explain lower collision exposure by actual halted transport.'),scope=dict(full=['20 assignment/result/meta key/parameter/source checks','80 snapshot and current source byte hashes','16 traditional baseline pairs: every stored numeric/bool array over full16s including raw substeps, excluding control_ms','joint and ratio_preview full3201 public user-target arrays and holding reference-rate/deviation arrays'],representative=['2 preparation scenes x5 methods,9 numeric physical/command fields before7.5s'],not_checked=['full compressed trace byte SHA','full string packet/noise replay','V3 FIR/TV/raw-force safety/physical tracking metrics owned by parent','causal benefit of motion-hold flags']),physics_steps=0,production_or_old_evidence_modified=False,interpretation='One seen physical/noise seed610212; native/strong-preview force baselines reproduce retainedV7 exactly. Joint may legitimately differ in reference/pauses; same public user target is retained. No global model admission or superiority proof.')
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),checks=len(checks),passed=sum(x['passed'] for x in checks),issues=issues,baseline_fields=[x['numeric_fields_count'] for x in baseline],motion=motion),ensure_ascii=False),flush=True)

"""Read-only retained-data audit. No application imports, QP, or physics."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
NAMES=['development_v6_known_counterexamples_v2_contract','development_v6_1_physical_all16']
OUT=ROOT/'review/independent_v6_1_data_audit_20261104.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def key(r):return r['scenario'],r['method'],r['seed']
def close(a,b):
    if a is None or b is None:return a is b
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
def mean(v):return None if not v else sum(v)/len(v)

tree=ast.parse((FOUNDATION/'session.py').read_text(encoding='utf-8-sig'))
scene_node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCENARIOS' for t in n.targets))
SCENES={}
for n in scene_node.elts:
    item={k.arg:ast.literal_eval(k.value) for k in n.keywords};SCENES[item['name']]=item

report={'schema':'independent_v6_1_retained_data_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
    'scope':'108 retained assigned records, all research snapshots/result dictionaries, metadata physical/noise pairing, independent raw-window eligibility and report arithmetic; representative preparation exactness. No FIR/physical-safety/tracking full recomputation, QP, new noise generation or physics.',
    'physics_steps':0,'production_or_original_report_modified':False,'matrices':[],'issues':[]}
all_run_ids=[]
all_rows={};all_metas={};all_paths={}

for name in NAMES:
    run=ROOT/'output'/name;manifest=read(run/'manifest.json');finish=read(run/'execution_complete.json');rows=read(run/'trials.json')
    readout=read(ROOT/'report'/name/'readout.json');markdown=(ROOT/'report'/name/'开发对照.md').read_text(encoding='utf-8-sig')
    jobs=manifest['jobs'];bykey={key(r):r for r in rows};groups=defaultdict(dict);checks=[];issues=[]
    def check(label,ok,detail=None):
        checks.append({'label':label,'passed':bool(ok),'detail':detail})
        if not ok:issues.append({'label':label,'detail':detail})
    check('job_trial_multiset_exact',Counter(map(key,jobs))==Counter(map(key,rows)))
    check('unique_job_and_trial_keys',len(set(map(key,jobs)))==len(jobs) and len(bykey)==len(rows))
    check('execution_counts_and_trials_sha',finish['planned']==finish['trials']==len(rows)==len(jobs) and finish['trials_sha256']==sha(run/'trials.json'))
    check('no_lost_runtime_failures',finish['execution_complete'] is True and finish['runtime_failures']==[])
    check('report_source_identity',readout['matrix_sha256']==sha(run/'trials.json') and readout['source_manifest_sha256']==sha(run/'manifest.json') and readout['execution']==finish)
    check('development_only',manifest['phase']=='development' and manifest['performance_claim_allowed'] is False and readout['performance_superiority_proven'] is False)
    sources=[]
    for source,expected in manifest['research_source_sha256'].items():
        p=run/'sources'/source;actual=sha(p);sources.append({'source':source,'expected':expected,'actual':actual,'passed':actual==expected})
    actual_sources={p.relative_to(run/'sources').as_posix() for p in (run/'sources').rglob('*') if p.is_file()}
    check('all_research_snapshot_hashes',all(x['passed'] for x in sources))
    check('snapshot_file_set_exact',actual_sources==set(manifest['research_source_sha256']))
    check('foundation_admission_identity',sha(FOUNDATION/'review/admission_manifest.json')==manifest['foundation_admission_sha256'])
    check('report_builder_source_hashes',all(sha(ROOT/'review'/p)==h for p,h in readout['report_sources'].items()))
    results=[];raw_eligibility={};metadata_pairing=[];fingerprint_list=[]
    for number,job in enumerate(jobs):
        k=key(job);row=bykey.get(k);folder=Path(job['destination']);result=read(folder/'result.json');meta=read(folder/'metadata.json')
        expected_folder=run/'runs'/f'{k[0]}__{k[1]}__{k[2]}'
        ok=(row==result and folder.resolve()==expected_folder.resolve())
        trace=folder/'trace.npz';trace_hash=meta['trace_sha256']
        with np.load(trace,allow_pickle=False) as z:
            embedded=read_embedded=json.loads(str(z['metadata']))
            side_without_trace={a:b for a,b in meta.items() if a!='trace_sha256'}
            t=z['time'];normal=z['normal'];skin=z['skin_normal'];position=z['position'];support_table=z['support'];active=z['controller_active'].astype(bool)
            dt=float(meta['physics_timestep']);st=z['substep_time'];dur=z['substep_duration']
            truth_keys=['normal','skin_normal','position','relative','relative_velocity','angular_velocity','lift','lift_ref','support']
            complete_control=(len(t)==3201 and np.allclose(t,np.arange(len(t))*.005,rtol=0,atol=1e-8) and all(np.isfinite(z[f]).all() for f in truth_keys))
            physical_grid=(len(st)>0 and len(st)==len(dur) and np.allclose(dur,dt,rtol=0,atol=1e-9) and abs(st[0])<1e-8 and np.allclose(st,np.arange(len(st))*dt,rtol=0,atol=1e-7) and abs(dur.sum()-16.)<1e-8)
            release=12. if meta['release_at'] is None else meta['release_at'];boundaries=sorted({0.,7.5,float(release),16.})
            grid=np.r_[0.,st+dur]
            phase_grid=all(np.any(abs(grid-b)<1e-7) for b in boundaries)
            complete=bool(complete_control and physical_grid and phase_grid and row['research_metrics']['filter_contract']['valid'])
            proto=read(run/'sources/protocol.json');th=proto['thresholds']
            touch=normal>th['finger_contact_min_N'];physical_support=touch[:,0]&(touch.sum(axis=1)>=th['minimum_contacts_including_thumb'])
            raised=position[:,2]>=meta['initial_position'][2]+th['minimum_object_lift_m']
            elig={}
            for q,(lo,hi) in {'static_quality':(8.2,8.9),'dynamic_quality':(9.3,11.7),'entire_holding_quality':(7.5,12.)}.items():
                mask=(t>=lo-1e-9)&(t<=hi+1e-9)
                coverage=min(1.,(t[mask][-1]-t[mask][0])/(hi-lo)) if mask.sum()>1 else 0.
                valid=bool(complete and mask.sum()==round((hi-lo)/.005)+1 and coverage>=1.-1e-8 and np.all(physical_support[mask]&raised[mask]&(support_table[mask]<=.05)&active[mask]))
                expected=row['research_metrics'][q]
                elig[q]={'eligible':valid,'stored_eligible':expected['eligible'],'sample_count':int(mask.sum()),'coverage':float(coverage),'cohort_equal':valid==expected['eligible'],'stored_window_equal':expected['window_s']==[lo,hi]}
                if valid:
                    f=normal[mask];seconds=t[mask][-1]-t[mask][0];tv=float(abs(np.diff(f,axis=0)).sum()/seconds);drift=float(abs(f[-1]-f[0]).sum()/seconds);reverse=max(0.,.5*(tv-drift));raw=expected['raw_200Hz']
                    elig[q]['raw_variation_recomputed']={'TV':tv,'drift':drift,'reversal':reverse}
                    elig[q]['raw_variation_matches']=close(tv,raw['finger_TV_rate_N_s']) and close(drift,raw['finger_net_drift_rate_N_s']) and close(reverse,raw['finger_reversal_rate_N_s'])
                else:elig[q]['missing_metrics_remain_none']=expected['bandlimited'] is None and expected['raw_200Hz'] is None
            raw_eligibility[k]=elig
            config=meta['config'];scene=SCENES[k[0]];u=np.random.default_rng(k[2]+511000).normal(size=4)
            expected_mass=float(scene.get('mass',.12)*np.clip(1.+.06*u[0],.85,1.15));expected_mu=float(scene.get('friction',.60)*np.clip(1.+.08*u[1],.80,1.20))
            physical_expected=(config['mass']==expected_mass and config['friction']==expected_mu and config['shape']==scene['shape'] and config['size']==scene.get('size',1.) and config['solref']==scene.get('solref',.008) and config['timestep']==.0005 and config['control_dt']==.005 and config['seed']==k[2])
            noise=meta['noise'];noise_expected=dict(force_sigma=.08 if k[0]=='sensor_noise' else .01,tangential_sigma=.04 if k[0]=='sensor_noise' else .005,position_sigma=.0001,slip_sigma=.002 if k[0]=='sensor_noise' else .0002,effort_sigma=.02,delay=.01)
            noise_ok=all(noise[a]==b for a,b in noise_expected.items())
            expected_dropout=[[9.3,9.7,[1,2]],[11.3,11.4,[0,1,2,3,4]]] if k[0]=='sensor_dropout' else []
            noise_ok=noise_ok and noise['dropout_windows']==expected_dropout and noise['dropout_probability']==0. and noise['instrument_auxiliary'] is True and noise['geometry_mode']=='distributed_wrench'
            caps=scene.get('limits',[5.,5.,5.,5.,5.]);budget=scene.get('budget',18.)
            record={'key':list(k),'result_equals_trial':ok,'metadata_identity':meta['method']==k[1] and meta['scenario']['name']==k[0] and meta['seed']==k[2] and meta['split']==job['split']=='robustness','trace_sha_linkage_matches':trace_hash==result['trace_sha256']==meta['trace_sha256']==result['research_metrics']['trace_sha256'],'embedded_metadata_equals_sidecar_without_trace_hash':embedded==side_without_trace,'result_physical_equals_metadata':row['research_metrics']['physical_parameters']==config,'metadata_source_subset_matches':all(manifest['research_source_sha256'].get(a)==b for a,b in meta['research_source_sha256'].items()),'physical_formula_matches':physical_expected,'actual_caps_match':meta['limits']==caps and meta['budget']==budget,'noise_profile_matches':noise_ok,'rng_seed_labels_match':row['research_metrics']['physical_seed']==row['research_metrics']['noise_seed']==k[2],'complete_control_and_raw_grid':complete,'quality_eligibility_and_raw_variation':elig,'preparation_passed':meta['handoff_passed'],'failure_reason':row['research_metrics']['failure_reason'],'run_id':row['run_id'],'trace_sha256':trace_hash}
            results.append(record);groups[(k[0],k[2])][k[1]]=meta;all_run_ids.append(row['run_id']);all_metas[(name,k)]=meta;all_paths[(name,k)]=trace
        if (number+1)%24==0:print(name,'read',number+1,'/',len(jobs),flush=True)
    check('all_result_dicts_exact',all(x['result_equals_trial'] for x in results))
    for field in ['metadata_identity','trace_sha_linkage_matches','embedded_metadata_equals_sidecar_without_trace_hash','result_physical_equals_metadata','metadata_source_subset_matches','physical_formula_matches','actual_caps_match','noise_profile_matches','rng_seed_labels_match','complete_control_and_raw_grid']:
        check(field,all(x[field] for x in results),[x['key'] for x in results if not x[field]])
    check('all_raw_window_eligibility_and_raw_variation',all(v['cohort_equal'] and v['stored_window_equal'] and v.get('raw_variation_matches',v.get('missing_metrics_remain_none',False)) for x in results for v in x['quality_eligibility_and_raw_variation'].values()))
    for g,methods in sorted(groups.items()):
        metas=list(methods.values());base=metas[0]
        pairing={'scenario':g[0],'seed':g[1],'methods':sorted(methods),'exact_config':all(m['config']==base['config'] for m in metas),'exact_noise':all(m['noise']==base['noise'] for m in metas),'exact_caps_budget':all((m['limits'],m['budget'])==(base['limits'],base['budget']) for m in metas),'exact_sensor_characteristics':all(m['task_contract']['sensor_characteristics']==base['task_contract']['sensor_characteristics'] for m in metas),'mass':base['config']['mass'],'friction':base['config']['friction'],'admitted_by_method':{k:v['handoff_passed'] for k,v in methods.items()}}
        metadata_pairing.append(pairing)
    check('all_triplets_paired',all(x['methods']==['joint','passive','ratio'] and x['exact_config'] and x['exact_noise'] and x['exact_caps_budget'] and x['exact_sensor_characteristics'] for x in metadata_pairing))
    method_checks=[]
    for summary in readout['methods']:
        method=summary['method'];rr=[r for r in rows if r['method']==method];mm=[r['research_metrics'] for r in rr]
        expected={'assigned':len(rr),'legacy_missions':sum(m['mission_success'] for m in mm),'full_motion_missions':sum(m['full_motion_mission_success'] for m in mm),'raw_force_guard_pass':sum(bool(m['safety'] and m['safety']['guard_pass']) for m in mm),'prepared':sum(m['preparation_passed'] for m in mm),'physical_releases':sum(m['release']['physical_success'] for m in mm),'release_latency_observed':sum(m['release']['physical_clearance_latency_s'] is not None for m in mm),'failure_reasons':dict(Counter(m['failure_reason'] for m in mm if m['failure_reason']))}
        equal=all(summary[k]==v for k,v in expected.items())
        means={}
        for label,q in [('static','static_quality'),('dynamic','dynamic_quality'),('holding','entire_holding_quality')]:
            vv=[m[q] for m in mm if m[q]['eligible']];qexp={'available':len(vv),'assigned':len(rr),'TV_N_s':mean([v['bandlimited']['finger_TV_rate_N_s'] for v in vv]),'reversal_N_s':mean([v['bandlimited']['finger_reversal_rate_N_s'] for v in vv]),'drift_N_s':mean([v['bandlimited']['finger_net_drift_rate_N_s'] for v in vv]),'raw_TV_N_s':mean([v['raw_200Hz']['finger_TV_rate_N_s'] for v in vv]),'mean_force_N':mean([v['mean_actual_finger_force_N'] for v in vv])}
            means[label]={'recomputed':qexp,'matches':all(close(summary[label][k],v) for k,v in qexp.items())}
        extras={'holding_local_excess_Ns':sum(st['local_excess_integral_Ns'] for m in mm if m['safety'] for k,st in m['safety']['stages'].items() if k=='policy_holding_transport'),'initial_stability_wait_s':mean([m['initial_stability_wait_s'] for m in mm]),'whole_force_excess_Ns':sum(m['safety']['local_excess_integral_Ns'] for m in mm if m['safety']),'release_latency_s':mean([m['release']['physical_clearance_latency_s'] for m in mm if m['release']['physical_clearance_latency_s'] is not None])}
        extra_equal=all(close(summary[k],v) for k,v in extras.items()) and summary['event_recovery_statuses']==dict(Counter(e['status'] for m in mm for e in m['event_recovery']))
        method_checks.append({'method':method,'recomputed':expected,'counts_and_failures_match':equal,'quality':means,'extra_values_match':extra_equal,'extra_values':extras})
    check('method_denominators_all_failures_and_aggregate_arithmetic',all(x['counts_and_failures_match'] and x['extra_values_match'] and all(q['matches'] for q in x['quality'].values()) for x in method_checks))
    paired=[]
    for pair in readout['paired_quality']:
        q=pair['window'];b={(r['scenario'],r['seed']):r for r in rows if r['method']==pair['baseline']};c={(r['scenario'],r['seed']):r for r in rows if r['method']==pair['candidate']};assigned=sorted(set(b)|set(c));good=[k for k in assigned if k in b and k in c and raw_eligibility[(k[0],pair['baseline'],k[1])][q]['eligible'] and raw_eligibility[(k[0],pair['candidate'],k[1])][q]['eligible']]
        expected_missing=[list(k) for k in assigned if k not in good];details={};arith=True
        for field,values in pair['contrasts'].items():
            x=[b[k]['research_metrics'][q]['bandlimited'][field] for k in good];y=[c[k]['research_metrics'][q]['bandlimited'][field] for k in good];expected={'baseline_mean':mean(x),'candidate_mean':mean(y),'mean_paired_difference':mean([yy-xx for xx,yy in zip(x,y)])}
            if 'relative_change_percent' in values:expected['relative_change_percent']=None if not expected['baseline_mean'] else 100.*(expected['candidate_mean']/expected['baseline_mean']-1.)
            match=all(close(values[a],v) for a,v in expected.items());details[field]={'recomputed':expected,'matches':match};arith &= match
        paired.append({'baseline':pair['baseline'],'candidate':pair['candidate'],'window':q,'assigned':len(assigned),'common':len(good),'members':good,'unavailable':expected_missing,'membership_counts_match':pair['assigned_pairs']==len(assigned) and pair['common_available_pairs']==len(good) and pair['unavailable_pairs']==expected_missing,'contrast_arithmetic_matches':arith,'contrasts':details})
    check('all_common_window_memberships_and_arithmetic',all(p['membership_counts_match'] and p['contrast_arithmetic_matches'] for p in paired))
    expected_lines=[]
    for s in readout['methods']:
        expected_lines.append(f"|{s['method']}|{s['full_motion_missions']}/{s['assigned']}|{s['raw_force_guard_pass']}/{s['assigned']}|{s['static']['TV_N_s']:.4f}|{s['dynamic']['TV_N_s']:.4f}|{s['holding']['TV_N_s']:.4f}|{s['holding_local_excess_Ns']:.4f}|{s['initial_stability_wait_s']:.4f}|")
    for p in readout['paired_quality']:
        v=p['contrasts']['finger_TV_rate_N_s'];expected_lines.append(f"|joint/{p['baseline']}|{p['window']}|{p['common_available_pairs']}/{p['assigned_pairs']}|{v['baseline_mean']:.4f}|{v['candidate_mean']:.4f}|{v['relative_change_percent']:.2f}%|")
    check('markdown_method_and_common_window_tables_match_json',all(line in markdown for line in expected_lines))
    matrix={'name':name,'manifest_sha256':sha(run/'manifest.json'),'trials_sha256':sha(run/'trials.json'),'readout_sha256':sha(ROOT/'report'/name/'readout.json'),'planned':len(jobs),'trials':len(rows),'group_count':len(groups),'physical_seed_clusters':sorted({r['seed'] for r in rows}),'checks':checks,'source_snapshot_checks':sources,'all_result_record_checks':results,'physical_noise_triplet_pairing':metadata_pairing,'method_aggregate_checks':method_checks,'paired_quality_checks':paired,'issues':issues}
    report['matrices'].append(matrix);all_rows[name]=rows;report['issues'].extend({'matrix':name,**x} for x in issues)
    print(name,'completed integrity/metadata/cohort/report audit; issues',len(issues),flush=True)

report['global_run_ids_unique']=len(set(all_run_ids))==len(all_run_ids)
report['foundation_relevant_source_hashes']={}
for name in ['session.py','sensors.py','foundation_config.py']:
    expected=next(iter(all_metas.values()))['source_sha256'][name];actual=sha(FOUNDATION/name)
    report['foundation_relevant_source_hashes'][name]={'expected':expected,'actual':actual,'matches':actual==expected}
report['rng_scope']={'physical_draw_rule':'default_rng(seed+511000).normal(size=4); mass multiplier clip(1+.06u0,.85,1.15); friction multiplier clip(1+.08u1,.80,1.20)','sensor_rule':'HandSensors default_rng(seed+910000); fixed per-read draw dimensions before validity/contact selection; matching seed/noise/call schedule supplies paired streams, not identical post-policy sensor values.','noise_packets_rebuilt':0,'independence_limit':'32 scenario-seed cells in all16 are only2 shared physical/noise seed clusters. Reused normalized physical/noise streams across scenarios are not32 independent draws. No confirmation or population-superiority inference.'}

representatives=[('nominal',610221),('nominal',610222),('sphere',610221)]
fields=['qpos','qvel','alpha_before','alpha','ctrl_before','ctrl','lift','lift_ref','controller_active']
preparation=[];name=NAMES[1]
for scene,seed in representatives:
    paths={m:all_paths[(name,(scene,m,seed))] for m in ['ratio','passive','joint']};cases=[]
    with np.load(paths['ratio'],allow_pickle=False) as base:
        t=base['time'];mask=t<7.5-1e-9;submask=base['substep_time']+base['substep_duration']<=7.5+1e-9
        for method in ['passive','joint']:
            with np.load(paths[method],allow_pickle=False) as other:
                checks={}
                checks['preparation_time_grid']=np.array_equal(t[mask],other['time'][other['time']<7.5-1e-9])
                for field in fields:
                    a=base[field][mask];b=other[field][mask]
                    checks[field]=bool(np.array_equal(a,b,equal_nan=True)) if a.dtype.kind in 'fc' else bool(np.array_equal(a,b))
                cases.append({'method':method,'fields_exact':checks,'all_exact':all(checks.values())})
        preparation.append({'scenario':scene,'seed':seed,'control_interval':'t<7.5-1e-9','control_samples':int(mask.sum()),'physical_interval':'substep_start+duration<=7.5+1e-9','physical_substeps':int(submask.sum()),'comparisons':cases})
    print('preparation representative',scene,seed,'exact',all(c['all_exact'] for c in cases),flush=True)
report['representative_preparation_exactness']=preparation
report['preparation_scope']='3 named scenario-seed triplets, not all32. Includes admitted/failed and two shapes. Exact9 stored qpos/qvel/alpha/ctrl/lift/controller-active arrays before7.5s plus time grid. No all-noise-packet or substep-force exactness claim.'
report['verification_limits']={'raw_trace_file_bytes_rehashed':0,'raw_trace_sha_linkages_checked':len(all_run_ids),'raw_force_finiteness_FIR_safety_tracking_not_recomputed':True,'window_eligibility_note':'Independently reconstruct control-grid physical support/raised/table/policy windows and substep time grid, conditional on recorded filter_contract.valid. FIR values and all raw force validity are owned by parent stratified review.'}
report['all_reviewed_preparation_fields_exact']=all(c['all_exact'] for p in preparation for c in p['comparisons'])
report['completed_utc']=datetime.now(timezone.utc).isoformat()
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'output':str(OUT),'matrices':[(x['name'],x['planned'],x['trials'],x['group_count']) for x in report['matrices']],'issues':report['issues'],'preparation_exact':report['all_reviewed_preparation_fields_exact'],'raw_trace_bytes_rehashed':0,'raw_trace_sha_linkages_checked':len(all_run_ids)},ensure_ascii=False),flush=True)

"""Check the evidence chain before issuing a real-policy admission manifest.

This program never imports a policy or hand model. Evidence revisions and old
manifests are kept; only the reviewed live pointer inside this new directory
may be atomically replaced after every required check has passed.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,shutil

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK='shadow_hand_strategy_integration_v1'
CORE=('hand_plant.py','foundation_config.py','contracts.py','sensors.py',
      'state_estimator.py','force_model.py','controller_adapter.py','policy_sources.py',
      'session.py','evaluate_runs.py')
PHYSICS=('hand_plant.py','foundation_config.py','calibrate_foundation.py')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inside(path):
    p=Path(path)
    p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT.resolve()):raise ValueError(f'Evidence must stay in the independent directory: {p}')
    return p

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def desc(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}

def build_manifest(paths):
    paths={k:inside(v) for k,v in paths.items()}
    documents={k:read(p) for k,p in paths.items()}
    hashes={k:sha(p) for k,p in paths.items()}
    physical=documents['physical'];independent=documents['independent']
    protocol=documents['protocol'];calibration=documents['calibration'];plan=documents['experiment_plan']
    checks={}
    checks['physical_passed']=physical.get('passed') is True
    checks['independent_passed']=independent.get('passed') is True
    checks['manifest_benchmark_matches']=all(d.get('benchmark_id')==BENCHMARK for d in (physical,independent,protocol,calibration,plan))
    checks['physical_protocol_chain']=physical.get('protocol_sha256')==hashes['protocol']
    checks['calibration_protocol_chain']=calibration.get('protocol_sha256')==hashes['protocol']
    checks['independent_protocol_chain']=independent.get('protocol_sha256')==hashes['protocol']
    checks['independent_physical_chain']=independent.get('physical_certificate_sha256')==hashes['physical']
    checks['independent_calibration_chain']=independent.get('calibration_sha256')==hashes['calibration']
    checks['physical_calibration_chain']=physical.get('calibration_sha256',hashes['calibration'])==hashes['calibration']
    checks['experiment_plan_frozen']=plan.get('status')=='frozen'
    plan_sources=plan.get('source_sha256',{})
    checks['experiment_plan_sources_current']=isinstance(plan_sources,dict) and bool(plan_sources) and all(
        inside(name).is_file() and sha(inside(name))==expected for name,expected in plan_sources.items())
    checks['only_two_current_methods']=plan.get('methods')==['ratio','unified']
    calrows=calibration.get('shapes',{})
    checks['three_shape_calibration_complete']=all(shape in calrows for shape in ('capsule','box','sphere'))
    for shape in ('capsule','box','sphere'):
        row=calrows.get(shape,{})
        row={**row,**row.get('response',{})}
        matrix=row.get('force_matrix',[]);weights=row.get('nominal_force_weights',[])
        errors=row.get('response_error_bound',row.get('response_error_bound_N',[]))
        values=[v for line in matrix if isinstance(line,list) for v in line]
        checks[f'{shape}_signed_matrix']=len(matrix)==5 and all(isinstance(line,list) and len(line)==5 for line in matrix) and len(values)==25 and all(isinstance(v,(float,int)) and math.isfinite(v) for v in values)
        tau=row.get('force_tau',0.)
        checks[f'{shape}_explicit_time_constant']=isinstance(tau,(float,int)) and math.isfinite(tau) and tau>0.
        checks[f'{shape}_response_uncertainty']=len(errors)==5 and all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0. for v in errors)
        checks[f'{shape}_public_force_profile']=len(weights)==5 and all(isinstance(v,(float,int)) and math.isfinite(v) and v>0. for v in weights) and abs(sum(weights)/5.-1.)<1e-9
        radius=row.get('trust_radius',0.)
        admitted=protocol.get('local_execution_model',{}).get('selected_local_domains',{}).get(shape,{})
        selected=row.get('selected_holdout',{})
        declared_radius=admitted.get('selected_radius',0.)
        checks[f'{shape}_validated_local_domain']=isinstance(radius,(float,int)) and math.isfinite(radius) and radius>0. and abs(radius-declared_radius)<1e-12 and abs(float(selected.get('radius',-999.))-radius)<1e-12 and selected.get('passed') is True
        raw=inside(admitted.get('response_raw','__missing__'));summary=inside(admitted.get('response_summary','__missing__'))
        checks[f'{shape}_selected_holdout_sources']=raw.is_file() and summary.is_file() and sha(raw)==admitted.get('response_raw_sha256')==selected.get('source_sha256') and sha(summary)==admitted.get('response_summary_sha256')==selected.get('summary_sha256')
        check_values=admitted.get('selected',{});proxy=check_values.get('p95_error_proxy_per_digit_N',[])
        checks[f'{shape}_selected_error_proxy']=len(errors)==5 and len(proxy)==5 and all(abs(float(a)-float(b))<1e-9 for a,b in zip(errors,proxy))
    for label,doc in [('physical',physical),('calibration',calibration)]:
        sources=doc.get('source_sha256',doc.get('source_export_sha256',{}))
        if label=='calibration' and not isinstance(sources,dict):
            # The aggregate calibration is exported from three independently
            # bound shape records; exporter SHA is not a source mapping.
            checks[f'{label}_source_binding']=all(
                all(name in calrows.get(shape,{}).get('source_sha256',{}) and
                    sha(ROOT/name)==calrows[shape]['source_sha256'][name] for name in PHYSICS)
                for shape in ('capsule','box','sphere'))
        else:
            checks[f'{label}_source_binding']=all(name in sources and (ROOT/name).is_file() and sha(ROOT/name)==sources[name] for name in PHYSICS)
    phases=plan.get('phases',{})
    dev=phases.get('development',{}).get('jobs',[])
    confirm=phases.get('confirmation',{})
    seeds=confirm.get('seeds',[]);scenes=confirm.get('scenarios',[])
    checks['development_six_paired_correctness_runs']=len(dev)==6 and {(j.get('method'),j.get('shape'),j.get('seed')) for j in dev}=={(m,s,520101) for m in ('ratio','unified') for s in ('capsule','box','sphere')}
    checks['confirmation_complete_cartesian_plan']=len(scenes)==16 and len(set(scenes))==16 and len(seeds)==8 and len(set(seeds))==8 and confirm.get('expected_trials')==len(scenes)*len(seeds)*2
    checks['confirmation_fresh_seeds']=all(isinstance(seed,int) and seed>=500000 for seed in seeds) and 520101 not in seeds
    demo=plan.get('demonstration',{}).get('default_seed',plan.get('demo_seed'))
    checks['demonstration_seed_separate']=isinstance(demo,int) and demo not in seeds and demo!=520101
    timing=protocol['matched_comparison']['initial_transfer_timing'];schedule=plan.get('schedule',{})
    checks['declared_timing_consistent']=all(abs(float(schedule.get(k,-999.))-float(timing[v]))<1e-12 for k,v in [('duration','duration_s'),('handoff','handoff_s'),('release_at','release_command_s')])
    endpoints=plan.get('primary_endpoints',{})
    checks['primary_endpoints_consistent']=endpoints.get('finger_tv_rate',{}).get('window')==timing['main_static_window_absolute_s'] and endpoints.get('A_S',{}).get('restricted_horizon')==timing['post_handoff_first_stability_horizon_s'] and endpoints.get('A_S',{}).get('start')==timing['post_handoff_first_stability_origin_s']
    checks['core_sources_exist']=all((ROOT/name).is_file() for name in CORE)
    vendor=ROOT/'vendor/frozen_controllers.py';settings=ROOT/'vendor/frozen_settings.json'
    checks['immutable_vendor_controller']=vendor.is_file() and sha(vendor)=='1277c9656259132c2529964284e58927bf4f6385f09eeabb3f85f3e9581d9805'
    before=read(ROOT/'review/preservation_before.json')
    original_settings=ROOT.parent/'统一人工反射重构/settings.json'
    expected_settings=before['files'].get(str(original_settings),{}).get('sha256')
    checks['immutable_vendor_policy_settings']=settings.is_file() and sha(settings)==expected_settings
    sources=list(ROOT.glob('*.py'))+[vendor,settings]
    sources += [p for p in (ROOT/'vendor').glob('*.py') if p.is_file()]
    sources += [p for p in ROOT.glob('*.ps1') if p.is_file()]
    sources += [p for p in ROOT.glob('*.cmd') if p.is_file()]
    sources += [p for p in (ROOT/'web').rglob('*') if p.is_file() and p.suffix in ('.html','.css','.js')]
    sources += [p for p in (ROOT/'assets').rglob('*') if p.is_file()]
    if (ROOT/'index.html').is_file():sources.append(ROOT/'index.html')
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(sources))}
    return {'schema':'shadow-hand-admission-v1','benchmark_id':BENCHMARK,
            'created_utc':datetime.now(timezone.utc).isoformat(),'passed':all(checks.values()),
            'checks':checks,'evidence':{name:desc(p) for name,p in paths.items()},
            'source_sha256':source_hashes,'admission_builder_sha256':sha(Path(__file__)),
            'scope':'Admit the frozen current-policy held-grasp migration, with shared preparation and simplified full-hand contact model; not next-generation policy or complete autonomous grasp acquisition.',
            'not_performance_evidence':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('physical','independent','protocol','calibration','experiment-plan'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'review/admission_manifest.json')
    args=parser.parse_args()
    paths={name:getattr(args,name) for name in ('physical','independent','protocol','calibration')}
    paths['experiment_plan']=args.experiment_plan
    result=build_manifest(paths)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    certificate=ROOT/f'review/independent_admission_check_{stamp}.json'
    certificate.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if result['passed']:
        output=inside(args.out);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            archive=ROOT/'review/admission_history';archive.mkdir(exist_ok=True)
            previous=archive/f'admission_{sha(output)}.json'
            if previous.exists() and sha(previous)!=sha(output):raise ValueError('Admission archive identity conflict.')
            if not previous.exists():shutil.copy2(output,previous)
        temporary=output.with_name(output.name+'.new')
        if temporary.exists():raise FileExistsError('Existing admission temporary file requires inspection.')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(output)
        print(json.dumps({'passed':True,'manifest':str(output),'checks':len(result['checks']),'sha256':sha(output)},ensure_ascii=False))
    else:
        print(json.dumps({'passed':False,'failed':[name for name,value in result['checks'].items() if not value],
                          'check_certificate':str(certificate),'manifest_not_issued':True},ensure_ascii=False))
        raise SystemExit(1)

if __name__=='__main__':main()

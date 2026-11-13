"""JSON-only supplementary per-run identity/assignment audit; no NPZ reload."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'review/v15_blind_runtime_pin_supplement_v1.json'
    if out.exists():raise FileExistsError(out)
    checks=[]
    def ck(name,v):checks.append(dict(name=name,passed=bool(v)))
    for name,count in [('development_v14_coherent_pilot_v1',105),('v15_passive_blind_bridge_v1',114)]:
        folder=ROOT/'output'/name;manifest=read(folder/'manifest.json');plan=read(folder/'plan.json');ledger=read(folder/'assignment_ledger.json')
        identity=manifest['research_source_sha256'];ck(name+'/count',len(identity)==count)
        expected={('nominal',m,617901,'hold','robustness') for m in ('fixed_effort','ratio_effort','fusion_effort')}
        key=lambda j:(j['scenario'],j['method'],j['seed'],j['motion_mode'],j['split'])
        ck(name+'/exact_manifest_assignment',len(manifest['jobs'])==3 and {key(x) for x in manifest['jobs']}==expected)
        ck(name+'/exact_plan_assignment',len(plan['jobs'])==3 and {key(x) for x in plan['jobs']}==expected)
        ck(name+'/all_assigned_evaluated',len(ledger)==3 and {key(x['job']) for x in ledger}==expected and all(x['status']=='evaluated' for x in ledger))
        for j in manifest['jobs']:
            path=Path(j['destination']);m=read(path/'metadata.json');ck(name+'/'+j['method']+'/actual_run_source_identity',m['research_source_sha256']==identity)
            ck(name+'/'+j['method']+'/seed_config_identity',m['method']==j['method'] and m['seed']==j['seed'] and m['scenario']['name']==j['scenario'] and m['motion_mode']==j['motion_mode'] and m['split']==j['split'])
            if name=='v15_passive_blind_bridge_v1':ck(name+'/'+j['method']+'/blind_legacy_both_job_and_metadata',j['task_policy']==m['task_policy']=='legacy' and j['receiver_policy']==m['receiver_policy']=='blind')
    report=dict(schema='v15_blind_per_run_pin_supplement_independent_v1',passed=all(x['passed'] for x in checks),checks=checks,checker_sha256=sha(__file__),npz_reads=0,physics_steps=0,
        main_evidence=dict(path='review/v15_passive_blind_independent_v1.json',sha256=sha(ROOT/'review/v15_passive_blind_independent_v1.json')),
        native_binary_evidence=dict(path='review/v15_blind_actual_native_environment_pin_v1.json',sha256=sha(ROOT/'review/v15_blind_actual_native_environment_pin_v1.json')))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=len(checks),sha256=sha(out)),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

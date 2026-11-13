"""Actual source/gate reads; memory-only prepare, no registration/dispatch."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse, hashlib, importlib.util, json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main(out):
    if out.exists():raise FileExistsError(out)
    files=['review/prepare_v23_small_v1.py','review/read_v23_small_readout_v1.py','review/verify_v23_v21_full_trace_bridges_v1.py','review/plot_v23_small_scientific_v1.py']
    before={p:sha(ROOT/p) for p in files};prep=load('v23_prepare_readonly_test',ROOT/files[0])
    reader=load('v23_reader_readonly_test',ROOT/files[1]);memory={};checks=[]
    def ck(n,x):checks.append(dict(name=n,passed=bool(x)))
    def payload(p,value):
        p=Path(p).resolve()
        if p.exists() or p in memory:raise FileExistsError(p)
        memory[p]=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    original_sha=prep.sha
    def memory_sha(p):
        key=Path(p).resolve()
        return hashlib.sha256(memory[key].encode('utf-8')).hexdigest() if key in memory else original_sha(p)
    args=SimpleNamespace(kernel_gate='review/v23_finite81_author_v1.json',math_gate='review/v23_numerical_independent_v1.json',
        integration_gate='review/v23_integration_author_final_v1.json',public_gate='review/v23_public_route_author_final_v1.json')
    with patch.object(prep,'write',payload),patch.object(prep,'sha',memory_sha):prep.prepare(args)
    plans=[json.loads(v) for p,v in memory.items() if p.name=='v23_small_plan_v1.json'];assert len(plans)==1
    plan=plans[0]
    ck('actual final software gates and186 interface validated',len(plan['source_sha256'])==186 and len(plan['software_gates'])==4)
    ck('frozen179/170 submaps retained',len(plan['parent179_source_sha256'])==179 and len(plan['original170_source_sha256'])==170 and all(plan['source_sha256'][k]==v for k,v in plan['parent179_source_sha256'].items()) and all(plan['source_sha256'][k]==v for k,v in plan['original170_source_sha256'].items()))
    ck('nine assignments and original nine historical records',len(plan['stages'])==9 and len(plan['historical_v22_full9'])==9 and len(plan['original_v21_bridges'])==3)
    ck('only nine registrations plus plan virtually written',len(memory)==10 and not plan['physics_dispatched'] and all(not p.exists() for p in memory))
    ck('runner module and explicit variants correct',all(s['python_arguments'][2]=='v23_task_execution.run_development' and s['assignment']['release_variant']==s['variant'] for s in plan['stages']))
    ck('final contract is pinned',plan['dependencies']['review/V23高效执行同条件验证合同_v1.md']==sha(ROOT/'review/V23高效执行同条件验证合同_v1.md'))
    ck('fixed metric/helper/bridge/plot dependencies pinned',all(p in plan['dependencies'] for p in ['review/delivery_auxiliary_metrics_v3.py','review/delivery_auxiliary_contract_v3.json','review/read_v21_release_variation_v1.py','review/verify_v23_v21_full_trace_bridges_v1.py','review/plot_v23_small_scientific_v1.py','review/v22_small_execution_v3.json']))
    # Use an empty virtual completion state. No real future/output marker read.
    with patch.object(reader.Path,'exists',return_value=False),patch.object(reader,'module',side_effect=AssertionError('calculator imported before barrier')),patch.object(reader.np,'load',side_effect=AssertionError('NPZ read before barrier')):
        try:reader.collect(plan);blocked=False
        except ValueError as e:blocked='All nine terminal' in str(e)
    ck('incomplete9 cannot import metric calculator or open NPZ',blocked)
    after={p:sha(ROOT/p) for p in files};ck('review files stable during gate',before==after)
    d=dict(schema='v23_review_adapters_software_v1',passed=all(c['passed'] for c in checks),checks=checks,reviewed_source_sha256=after,
        checker_sha256=sha(__file__),production_source_sha256=plan['source_sha256'],numerical_identity=plan['numerical_identity'],
        contract_sha256=plan['dependencies']['review/V23高效执行同条件验证合同_v1.md'],
        all_virtual_registration_paths=[str(p) for p in memory],actual_registration_files_created=0,dispatch_calls=0,
        physics_steps=0,model_constructions=0,new_V23_outcomes_read=False,scope='Actual static source/gate identity and memory-only prepare/partial barrier; not physical execution.')
    out.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=d['passed'],checks=len(checks),out=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if d['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();raise SystemExit(main(a.out))

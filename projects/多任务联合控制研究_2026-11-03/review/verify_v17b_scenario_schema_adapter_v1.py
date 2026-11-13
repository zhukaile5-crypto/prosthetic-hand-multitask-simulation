"""Actual OLD archive schema gate; never read live80 outcomes/physics."""
from pathlib import Path
from copy import deepcopy
import ast,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import read_v17b_delivery_base_v3 as new
import read_v17b_delivery_prospective80_v2 as old
import read_v17b_delivery_prospective80_v4 as entry

EXPECTED_BASE='f9966f5c23fafc79031bae0572d353945ebdaa4495ff903476cfed3490f72120'
CHECKS=[]
def ck(name,value,details=None):CHECKS.append(dict(name=name,passed=bool(value),details=details))
def rejects(fn):
    try:fn()
    except (TypeError,ValueError):return True
    return False
def funcs(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}

def main(out):
    if new.sha(Path(new.__file__))!=EXPECTED_BASE:raise RuntimeError('Stable candidate changed')
    published=new.read(ROOT/'review/delivery_auxiliary_full15_posthoc_v3.json')['records'][0]
    run=Path(published['run']);meta=new.read(run/'metadata.json');original=new.read(run/'result.json')
    item=dict(record_key=['full_V17','fragile_conflict','hold',619732],record_id='["full_V17","fragile_conflict","hold",619732]',
        role='full_V17',scenario='fragile_conflict',motion_mode='hold',physical_seed=619732,method='ratio_effort',
        task_policy='integrated',receiver_policy='coordinated',run=str(run))
    ledger=dict(status='evaluated',job={'destination':str(run)})
    plan={'source_sha256':meta['research_source_sha256']}
    ck('real old metadata descriptor normalizes to independently fixed plan name',type(meta['scenario']) is dict and new.scenario_name(meta['scenario'])=='fragile_conflict'
        and new.scenario_name('fragile_conflict')=='fragile_conflict')
    prior=old.load_assigned(item,ledger,original,plan,[])
    ck('old defect reproduced with genuine plan-string item, retained as failed compatibility',prior['status']=='identity_untrusted' and 'metadata_identity' in prior['issues'] and prior['auxiliary'] is None)
    good=new.load_assigned(item,ledger,original,plan,[])
    ck('new real-schema loader accepts actual known archive with source proof true',good['status']=='computed_complete' and not good['issues']
        and good['execution_source_identity_equal'] is True and good['original_result']==original)
    ck('every frozen v3 auxiliary field equals published historical regression',good['auxiliary']==published['metrics'])
    wrong=deepcopy(item);wrong['scenario']='nominal';r=new.load_assigned(wrong,ledger,original,plan,[])
    ck('different registered scene remains rejected, no broad identity relaxation',r['status']=='identity_untrusted' and 'metadata_identity' in r['issues'] and r['auxiliary'] is None)
    altered=deepcopy(plan);key=next(iter(altered['source_sha256']));altered['source_sha256'][key]='0'*64
    r=new.load_assigned(item,ledger,original,altered,[])
    ck('actual source mismatch still false and untrusted',r['execution_source_identity_equal'] is False and r['auxiliary'] is None and 'record_source_identity' in r['issues'])
    others=[]
    for field,value in (('method','fixed_effort'),('physical_seed',619733),('motion_mode','transport'),('task_policy','legacy'),('receiver_policy','blind')):
        bad=deepcopy(item);bad[field]=value;r=new.load_assigned(bad,ledger,original,plan,[])
        others.append(r['status']=='identity_untrusted' and r['auxiliary'] is None and 'metadata_identity' in r['issues'])
    ck('method seed motion taskpolicy receiverpolicy guards unchanged',all(others))
    bad=deepcopy(item);bad['scenario']=meta['scenario'];r=new.load_assigned(bad,ledger,original,plan,[])
    ck('roster must remain a string: descriptor-in-roster cannot mirror old weak fixture',r['status']=='invalid_or_missing_record'
        and rejects(lambda:new.scenario_name({'name':None})) and rejects(lambda:new.scenario_name({'name':True}))
        and rejects(lambda:new.scenario_name(1)) and rejects(lambda:new.scenario_name({})))
    a=funcs(old.__file__);b=funcs(new.__file__);unchanged=[k for k in a if k not in ('load_assigned','main')]
    ck('all other existing reader functions and numeric/scoring AST unchanged',all(a[k]==b[k] for k in unchanged),unchanged)
    ck('new entry pins exact newbase and unchanged chain validator plus frozen calculator/contract',entry.BASE_SHA==EXPECTED_BASE
        and new.sha(ROOT/'review/v17b_completion_chain_reader_v1.py')=='f9f06eb1367f6219f54cb7976e8303497168b4bdd863fea84c3e5e8958632171'
        and new.sha(Path(new.aux.__file__))==new.CALC_SHA and new.sha(new.aux.CONTRACT_PATH)==new.CONTRACT_SHA
        and 'mujoco' not in sys.modules)
    report=dict(schema='v17b_scenario_metadata_schema_author_gate_v1',passed=all(c['passed'] for c in CHECKS),checks=CHECKS,count=len(CHECKS),
        newbase_sha256=new.sha(Path(new.__file__)),newentry_sha256=new.sha(Path(entry.__file__)),
        original_base_v2_sha256=new.sha(Path(old.__file__)),original_entry_v3_sha256=new.sha(ROOT/'review/read_v17b_delivery_prospective80_v3.py'),
        calculator_sha256=new.CALC_SHA,contract_sha256=new.CONTRACT_SHA,
        historical_input_sha256=good['input_sha256'],scope=dict(old_archive_records=1,live80_outcomes_read=False,physics_steps=0,models=0,
            compatibility_only=True,no_original_metric_or_source_changed=True,original_weak_gates_preserved=True))
    p=ROOT/out
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(p),passed=report['passed'],count=len(CHECKS),failures=[c['name'] for c in CHECKS if not c['passed']],sha256=new.sha(p))))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

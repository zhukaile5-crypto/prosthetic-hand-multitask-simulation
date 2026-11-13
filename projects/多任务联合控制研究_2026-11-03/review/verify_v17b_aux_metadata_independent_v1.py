"""Real old metadata+ledger schema regression; zero new80/physics reads."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import argparse,ast,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--entry',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    bp=ROOT/args.base;ep=ROOT/args.entry;out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    before={str(p.relative_to(ROOT)):sha(p) for p in (bp,ep)}
    sp=importlib.util.spec_from_file_location('aux_metadata_new_subject',bp);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    run=ROOT/'output/v17_transfer_contact_hold_v1/runs/fragile_conflict__ratio_effort__619732';meta=read(run/'metadata.json');result=read(run/'result.json')
    key=['full_V17','fragile_conflict','hold',619732]
    item=dict(record_key=key,record_id=m.compact(key),role=key[0],scenario=key[1],motion_mode=key[2],physical_seed=key[3],method='ratio_effort',stage='old_known_fixture',task_policy='integrated',receiver_policy='coordinated',run=str(run))
    ledger=dict(status='evaluated',job=dict(destination=str(run)));plan=dict(source_sha256=meta['research_source_sha256'])
    checks=[]
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    ck('actual_old_metadata_is_richdict_matching_assigned_string',isinstance(meta['scenario'],dict) and meta['scenario']['name']==item['scenario'])
    ck('explicit_scenario_name_normalizes_actual_mapping',m.scenario_name(meta['scenario'])==item['scenario'])
    ck('explicit_legacy_string_form_supported',m.scenario_name(item['scenario'])==item['scenario'])
    entry_tree=ast.parse(ep.read_text(encoding='utf-8'))
    imports=[n for n in entry_tree.body if isinstance(n,ast.Import) and any(x.name==bp.stem and x.asname=='base' for x in n.names)]
    pins=[ast.literal_eval(n.value) for n in entry_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_SHA' for t in n.targets)]
    ck('actual_entry_imports_and_pins_reviewed_base',len(imports)==1 and pins==[sha(bp)])
    actual=np.load;handles={};columns={};calls=[]
    def cached(path,**kw):
        k=str(Path(path).resolve());calls.append(k)
        if k not in handles:handles[k]=actual(path,**kw);columns[k]={}
        class View:
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def __getitem__(self,key):
                if key not in columns[k]:columns[k][key]=handles[k][key]
                return columns[k][key]
        return View()
    try:
        with patch.object(np,'load',cached):
            good=m.load_assigned(item,ledger,result,plan,[])
            ck('real_dict_metadata_loads_complete_auxiliary',good['status']=='computed_complete' and good['execution_source_identity_equal'] is True and not good['issues'] and good['auxiliary'] is not None,dict(status=good['status'],issues=good['issues']))
            published=read(ROOT/'review/delivery_auxiliary_full15_posthoc_v3.json');prior=next(r for r in published['records'] if Path(r['run']).resolve()==run.resolve())
            ck('every_calculated_field_exact_to_frozen_historical15',good['auxiliary']==prior['metrics'])
            before_negative=len(calls);wrong=deepcopy(item);wrong['scenario']='nominal'
            bad=m.load_assigned(wrong,ledger,result,plan,[])
            ck('wrong_scene_rejected_before_reading_ledger',bad['auxiliary'] is None and 'metadata_identity' in bad['issues'] and len(calls)==before_negative)
            wrongplan=dict(source_sha256={'unknown':'not_original'})
            bad=m.load_assigned(item,ledger,result,wrongplan,[])
            ck('wrong_source_still_rejected',bad['auxiliary'] is None and bad['execution_source_identity_equal'] is False and 'record_source_identity' in bad['issues'])
            wrongtrial=deepcopy(result);wrongtrial['task_success']=not bool(wrongtrial['task_success'])
            bad=m.load_assigned(item,ledger,wrongtrial,plan,[])
            ck('wrong_trial_still_rejected',bad['auxiliary'] is None and 'trial_result_identity' in bad['issues'])
            reader=m.read
            def alternate_meta(value):
                def altered(path):return deepcopy(value) if Path(path).name=='metadata.json' else reader(path)
                return altered
            legacy=deepcopy(meta);legacy['scenario']=item['scenario']
            with patch.object(m,'read',alternate_meta(legacy)):
                legacyrow=m.load_assigned(item,ledger,result,plan,[])
            ck('string_and_mapping_forms_give_identical_calculator_result',legacyrow['auxiliary']==good['auxiliary'] and legacyrow['status']==good['status'])
            malformed=deepcopy(meta);malformed['scenario']={'title':'no declared name'}
            with patch.object(m,'read',alternate_meta(malformed)):bad=m.load_assigned(item,ledger,result,plan,[])
            ck('missing_scenario_name_not_silently_accepted',bad['auxiliary'] is None and bad['status']!='computed_complete')
            ck('only_one_old_ledger_archive_decompressed',len(handles)==1 and all(Path(k).name=='delivery_intervals.npz' for k in handles))
    finally:
        for h in handles.values():h.close()
    calc=Path(m.aux.__file__);contract=m.aux.CONTRACT_PATH
    ck('metric_v3_unchanged_and_pinned',sha(calc)=='0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55' and sha(contract)=='8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979')
    ck('final_base_and_entry_source_stable',before=={str(p.relative_to(ROOT)):sha(p) for p in (bp,ep)})
    report=dict(schema='v17b_aux_metadata_schema_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,check_count=len(checks),
        source_sha256=before,contract_sha256=sha(contract),calculator_sha256=sha(calc),input_sha256={n:sha(run/n) for n in ('metadata.json','result.json','delivery_intervals.npz')},
        old_failure_evidence_sha256=sha(ROOT/'review/v17b_aux_metadata_schema_failed_v1.json'),checker_sha256=sha(__file__),
        scope=dict(known_archives=1,old_physical_seed=619732,new80_results_read=0,physics_steps=0,model_calls=0,production_modifications=0,
            metadata_string_case_mocked_read_only=True,normalization_changes_identity_only_not_metric=True))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],out=str(out),sha256=sha(out))))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

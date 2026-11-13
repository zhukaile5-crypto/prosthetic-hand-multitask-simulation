"""Independent finite review of the V21 AUXILIARY READER, no new results.
Tests original reader functions with fully synthetic filesystem/NPZ/calculator
callbacks. Frozen old complete V20 JSON/metadata supply regression values.
No V21 plan/marker/result/run/NPZ is opened. No real collect/main execution.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import ast,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
SOURCE=ROOT/'review/read_v21_small_auxiliary_v1.py'
VIRTUAL=ROOT/'review/SYNTHETIC_V21_READER_ONLY_NO_FILES'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    out=ROOT/'review/v21_auxiliary_reader_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    before=sha(SOURCE);m=load('v21_aux_reader_under_independent_test',SOURCE);cases=[];reads_actual=[];scopes=[]
    paths,pins=m.frozen_dependencies();projection=load('v21_fixed_projection_independent',paths['metrics']);old20=read(paths['old20']);reads_actual.append(str(paths['old20']))
    oldrows={tuple(v['context_key']):v for v in old20['records'] if v['stage']!='passive_ratio_transport'}
    metas={}
    for key,row in oldrows.items():
        path=Path(row['run'])/'metadata.json';metas[key]=read(path);reads_actual.append(str(path))
    def ck(name,fn):
        try:details=fn();cases.append(dict(name=name,passed=True,details=details))
        except Exception as exc:cases.append(dict(name=name,passed=False,error=repr(exc)))
    def roster():
        stages=[]
        for variant in m.VARIANTS:
            for i,(scene,motion,seed) in enumerate(m.CONTEXTS):
                name=variant+'_'+str(i);ass=dict(methods=['ratio_effort'],scenarios=[scene],seeds=[seed],motion_mode=motion,physical_randomization=True,task_policy='integrated',receiver_policy='coordinated',release_variant=variant)
                stages.append(dict(name=name,variant=variant,planned=1,assignment=ass,output='output/'+name,registration='register/'+name+'.json'))
        plan=dict(planned=9,stages=stages,source_sha256={f's{i}':'0'*64 for i in range(170)},environment={'fixture_only':True},auxiliary_contract_sha256=m.CONTRACT_SHA,auxiliary_calculator_sha256=m.CALC_SHA)
        good=dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],planned=1,trials=1,trials_sha256='trials',assignment_ledger_sha256='assignment_ledger',plan_sha256='plan',manifest_sha256='manifest')
        return plan,{s['name']:deepcopy(good) for s in stages}
    def rejected(fn):
        try:fn()
        except (ValueError,KeyError):return True
        return False
    def frozen_regression():
        assert before=='52474b1d6f8db2ab68b7179e0c162feca4264dc66f0bf27de84ddce1944dd91f'
        assert all(sha(paths[k])==pins[k] for k in paths)
        assert all(projection.metrics(row['auxiliary'])==row['metrics'] for row in old20['records'])
        return dict(pins=pins,old_V20_records_checked=len(old20['records']),independent_projection_exact=True)
    ck('fixed_v3_and_V18_metric_projection_pins_plus_old_complete_V20_regression',frozen_regression)
    def roster_gate():
        p,marks=roster();assert len(m.validate_roster(p))==9
        for change in ('duplicate','missing','wrong_variant'):
            q=deepcopy(p)
            if change=='duplicate':q['stages'][-1]=deepcopy(q['stages'][0])
            elif change=='missing':q['stages'].pop()
            else:q['stages'][0]['assignment']['release_variant']='prefix'
            assert rejected(lambda:m.validate_roster(q))
        return dict(assigned=9,variants=3,contexts=3,roles_not_methods=True)
    ck('exact_nine_unique_variant_context_roster_rejects_missing_or_duplicate',roster_gate)
    def inner_gate():
        p,marks=roster();marks.pop(p['stages'][-1]['name']);called=[]
        def blocked(*args,**kw):called.append(True);raise AssertionError('outcome/dependency load before all9')
        with patch.object(m,'frozen_dependencies',blocked),patch.object(m,'read',blocked),patch.object(m.np,'load',blocked):
            assert rejected(lambda:m.collect(p,marks))
        assert not called;return dict(inner_barrier_precedes_any_dependency_or_result_or_npz_read=True)
    ck('direct_collect_cannot_bypass_all9_terminal_barrier',inner_gate)
    def outer_gate():
        p,marks=roster();pending=p['stages'][-1]['name'];marks.pop(pending);seen=[];vplan=VIRTUAL/'registered_plan.json';objects={str(vplan):p}
        for s in p['stages']:
            if s['name'] in marks:objects[str(VIRTUAL/s['output']/'execution_complete.json')]=marks[s['name']]
        def fake_read(path):
            path=str(path);seen.append(path)
            if path not in objects:raise AssertionError('unexpected real/new outcome read '+path)
            return deepcopy(objects[path])
        exists=Path.exists
        def fake_exists(path):return str(path) in objects if str(path).startswith(str(VIRTUAL)) else exists(path)
        with patch.object(m,'ROOT',VIRTUAL),patch.object(m,'PLAN',vplan),patch.object(m,'read',fake_read),patch.object(Path,'exists',fake_exists),patch.object(m.np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('early NPZ'))):
            barrier,_,_=m.readiness()
        assert not barrier['ready_to_read_metrics'] and barrier['pending']==[pending] and not barrier['metrics_read'];assert all(x.endswith('execution_complete.json') or x==str(vplan) for x in seen)
        # Completed with source/runtime failure remains visibly blocked, never
        # fabricated as successful, failed or zero-valued outcome measurements.
        _,allmarks=roster()
        for field,value in [('sources_unchanged',False),('runtime_failures',[{'error':'fixture'}]),('trials',0),('execution_complete',False)]:
            mm=deepcopy(allmarks);mm[p['stages'][0]['name']][field]=value;assert not m.terminal_barrier(p,mm)['ready_to_read_metrics']
        return dict(only_plan_and_completion_markers_read=len(seen),pending_not_numeric_failure=True)
    ck('outer_readiness_never_reads_outcomes_or_mistakes_pending_for_metrics',outer_gate)
    def metadata_gate():
        p,_=roster();ass=p['stages'][0]['assignment'];meta=dict(scenario={'name':'nominal'},motion_mode='transport',seed=619741,method='ratio_effort',task_policy='integrated',receiver_policy='coordinated',release_variant='v20');m.metadata_context(meta,ass)
        for change in (dict(release_variant='prefix'),dict(scenario={'name':'collision'}),dict(seed=619742),dict(task_policy='legacy'),dict(method='fusion_effort')):
            bad=dict(meta,**change);assert rejected(lambda:m.metadata_context(bad,ass))
        return dict(actual_dict_scenario_supported=True,wrong_same_method_variant_rejected=True)
    ck('actual_dict_metadata_context_and_release_variant_are_explicitly_bound',metadata_gate)
    def simulation(mode=None,param=None):
        p,marks=roster();objects={};hashes={};load_calls=[];calc_calls=[];new_read_calls=[];base_aux={k:deepcopy(v['auxiliary']) for k,v in oldrows.items()}
        oldfake=dict(complete=True,issues=[],records=[]);refs=[]
        for key,row in oldrows.items():
            oldrun=VIRTUAL/'historicalV20'/('__'.join(map(str,key)));orr=deepcopy(row);orr['run']=str(oldrun);oldfake['records'].append(orr);objects[str(oldrun/'metadata.json')]=deepcopy(metas[key])
            for file,digest in row['input_sha256'].items():hashes[str(oldrun/file)]=digest
            for role in m.HISTORICAL_ROLES:
                mm=metas[key];refs.append(dict(role=role,scenario=key[0],motion_mode=key[1],physical_seed=key[2],physical_config=deepcopy(mm['config']),noise=deepcopy(mm['noise']),limits=deepcopy(mm['limits']),budget=mm['budget'],auxiliary=deepcopy(row['auxiliary']),record_id='old_fixture_'+role+'_'+str(key)))
        objects[str(paths['old20'])]=oldfake;objects[str(paths['old80'])]=dict(complete=True,source_identity_passed=True,records=refs)
        for s in p['stages']:
            ass=s['assignment'];key=m.context(ass);variant=s['variant'];folder=VIRTUAL/s['output'];run=folder/'runs'/f'{key[0]}__ratio_effort__{key[2]}';meta=deepcopy(metas[key]);meta.update(scenario={'name':key[0]},release_variant=variant,research_source_sha256=p['source_sha256'])
            result=deepcopy(oldrows[key]['original_result']);result['_synthetic_context']=list(key);result['_synthetic_variant']=variant
            if variant=='prefix' and key==m.CONTEXTS[0] and mode=='null_failure':
                meta['mechanical_release_at']=None;result['task_success']=False;result['research_metrics']['full_motion_mission_success']=False
            if variant=='v21' and key==m.CONTEXTS[0] and mode=='unpaired':
                if param=='budget':meta[param]=float(meta[param])+.125
                elif param=='limits':meta[param]=[float(v)+.125 for v in meta[param]]
                else:meta[param]=dict(meta[param],synthetic_mismatch=True)
            job=dict(status='evaluated',job=dict(destination=str(run)))
            objects[str(folder/'trials.json')]=[result];objects[str(folder/'assignment_ledger.json')]=[job];objects[str(folder/'manifest.json')]=dict(research_source_sha256=p['source_sha256']);objects[str(folder/'environment.json')]=dict(pinned_environment=p['environment']);objects[str(VIRTUAL/s['registration'])]=dict(assignment=ass,source_sha256=p['source_sha256'],environment=p['environment']);objects[str(run/'metadata.json')]=meta;objects[str(run/'result.json')]=result
            for file in ('trials','assignment_ledger','plan','manifest'):hashes[str(folder/(file+'.json'))]=file
        def fake_read(path):
            key=str(path);new_read_calls.append(key)
            if key not in objects:raise AssertionError('UNAUTHORIZED_REAL_OR_NEW_OUTCOME_READ '+key)
            return deepcopy(objects[key])
        def fake_sha(path):
            key=str(path)
            if key in hashes:return hashes[key]
            if key.startswith(str(VIRTUAL)):return 'synthetic_hash_no_file_read'
            if key==str(paths['old80']) or key==str(paths['old20']):return pins['old80'] if key==str(paths['old80']) else pins['old20']
            if key==str(SOURCE):return before
            raise AssertionError('unexpected SHA file read '+key)
        class Archive:
            def __init__(self,path):self.path=path
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __getitem__(self,key):return np.empty(0)
        def fake_load(path,**kwargs):
            assert str(path).startswith(str(VIRTUAL)) and str(path).endswith('delivery_intervals.npz');load_calls.append(str(path));return Archive(path)
        def calculate(arrays,*,release_at_s,episode_end_s,physical_period_s,original_result):
            key=tuple(original_result['_synthetic_context']);variant=original_result['_synthetic_variant'];calc_calls.append((variant,key));x=deepcopy(base_aux[key]);x['original_result']=deepcopy(original_result)
            if mode=='partial' and variant=='prefix' and key==m.CONTEXTS[0]:raise ValueError('synthetic_partial_or_missing_window')
            if mode=='bad_bridge' and variant=='v20' and key==m.CONTEXTS[0]:x['terminal']['linear_speed_m_s']=float(np.nextafter(x['terminal']['linear_speed_m_s'],np.inf))
            return x
        calc=SimpleNamespace(C={'input_fields':['test_array_no_physical_values']},calculate=calculate)
        def fake_module(name,path):return calc if Path(path)==paths['calculator'] else projection
        with patch.object(m,'ROOT',VIRTUAL),patch.object(m,'PLAN',VIRTUAL/'PLAN.json'),patch.object(m,'read',fake_read),patch.object(m,'sha',fake_sha),patch.object(m,'frozen_dependencies',lambda:(paths,pins)),patch.object(m,'load_module',fake_module),patch.object(m.np,'load',fake_load):
            result=m.collect(p,marks)
        scopes.append(dict(mode=mode,param=param,simulation_only=True,records=len(result['records']),within_pairs=len(result['within_variant_pairs']),historical_pairs=len(result['historical_pairs']),bridges=len(result['bridge_auxiliary_equivalence']),NPZ_callback_calls=len(load_calls),each_loaded_at_most_once=len(set(load_calls))==len(load_calls),calc_callback_calls=len(calc_calls),reads_real_new_V21_outcomes=False))
        return result
    def nine_records():
        x=simulation();assert x['complete'] and len(x['records'])==9 and len(x['within_variant_pairs'])==9 and len(x['historical_pairs'])==27 and len(x['bridge_auxiliary_equivalence'])==3;assert all(z['all_non_original_auxiliary_fields_exact'] for z in x['bridge_auxiliary_equivalence']);assert scopes[-1]['each_loaded_at_most_once']
        assert 'two shared physical seed clusters' in x['performance_scope'];return scopes[-1]
    ck('synthetic_collect_keeps_all9_all9pairs_all27historical_and3exact_bridges',nine_records)
    def null_failure():
        x=simulation('null_failure');row=next(r for r in x['records'] if r['variant']=='prefix' and tuple(r['context_key'])==m.CONTEXTS[0]);assert len(x['records'])==9 and x['complete'] and row['original_result']['task_success'] is False and row['metrics'] is None and row['status']=='no_actual_release_clock'
        pair=next(z for z in x['within_variant_pairs'] if z['candidate_variant']=='prefix' and tuple(z['context_key'])==m.CONTEXTS[0]);assert all(v['delta'] is None and not v['paired_available'] for v in pair['differences'].values());return dict(failed_row_kept=True,metrics_unknown_not_zero=True,assigned_9=True)
    ck('original_failure_and_no_exposure_clock_NULL_remain_assigned_not_success_filtered',null_failure)
    def partial():
        x=simulation('partial');assert not x['complete'] and len(x['records'])==9 and x['issues'];row=next(z for z in x['records'] if z['variant']=='prefix' and tuple(z['context_key'])==m.CONTEXTS[0]);assert row['metrics'] is None and row['status']=='diagnostic_unavailable';return dict(all9_kept=True,partial_status=row['status'],issues=x['issues'])
    ck('partial_diagnostic_failure_is_explicit_and_retained_without_complete_claim',partial)
    def pairing():
        details=[]
        for field in ('config','noise','limits','budget'):
            x=simulation('unpaired',field);assert not x['complete'] and len(x['records'])==9 and x['issues'];pairs=[z for z in x['historical_pairs'] if z['variant']=='v21' and tuple(z['context_key'])==m.CONTEXTS[0]];assert all(not z['same_physical_noise_limits'] and all(not v['paired_available'] for v in z['differences'].values()) for z in pairs);details.append(field)
        return dict(mismatched_fields_each_tested=details,unpaired_not_silently_compared=True)
    ck('all_four_physics_noise_material_parameter_groups_gate_within_and_historical_pairs',pairing)
    def exact_bridge():
        x=simulation('bad_bridge');assert not x['complete'] and any('originalV20_aux_differ' in z for z in x['issues']);assert len(x['bridge_auxiliary_equivalence'])==3 and sum(not z['all_non_original_auxiliary_fields_exact'] for z in x['bridge_auxiliary_equivalence'])==1
        z=deepcopy(next(iter(oldrows.values()))['auxiliary']);other=deepcopy(z);other['original_result']={'different_provenance':True};assert projection.same_non_original(z,other);return dict(one_ULP_nonidentity_change_rejected=True,only_original_result_identity_excluded=True)
    ck('V20_three_arm_bridge_requires_all_nonidentity_auxiliary_fields_exact',exact_bridge)
    def numeric_NULL():
        d=m.differences(dict(unknown=None,zero=0.,failure=False),dict(unknown=0.,zero=0.,failure=True));assert d['unknown']['delta'] is None and d['zero']['delta']==0 and d['zero']['relative_delta'] is None and d['failure']['delta']==-1 and d['failure']['relative_delta'] is None
        u=m.differences({'force':1.},{'force':2.},False);assert u['force']['delta'] is None and not u['force']['paired_available'];return dict(NULL_zero_false_separate=True,baseline_zero_relative_is_NULL=True,unpaired_is_not_delta=True)
    ck('NULL_zero_bool_failure_and_zero_baseline_relative_rules_are_distinct',numeric_NULL)
    def input_scope():
        source=SOURCE.read_text(encoding='utf-8');assert 'release_decisions' not in source and 'trace.npz' not in source and 'P0' not in source
        assert not any('/output/v21_' in p.replace('\\','/') for p in reads_actual);assert sha(SOURCE)==before and 'mujoco' not in sys.modules;assert all(s['reads_real_new_V21_outcomes'] is False for s in scopes)
        return dict(no_optimizer_or_P0_count_used_as_performance=True,new_V21_results_read=False,reader_unchanged=True)
    ck('reader_does_not_turn_P0_or_prefix_counts_into_benefit_and_review_reads_no_new_outcomes',input_scope)
    result=dict(schema='v21_auxiliary_reader_independent_v1',passed=all(c['passed'] for c in cases),checks=len(cases),cases=cases,source_sha256={'review/read_v21_small_auxiliary_v1.py':before,'review/read_v18_small_auxiliary_v1.py':sha(paths['metrics']),'review/delivery_auxiliary_metrics_v3.py':sha(paths['calculator']),'review/delivery_auxiliary_contract_v3.json':sha(paths['contract'])},frozen_reference_sha256={'old80':pins['old80'],'old20':pins['old20']},reader_gate_sha256=sha(__file__),synthetic_collect_simulations=scopes,actual_old_inputs_read=reads_actual,
        scope=dict(new_V21_results_read=False,actual_collect_main_run=False,synthetic_IO_only=True,physics=0,models=0,calculators_on_new_results=0),
        limitations=['This is a read-tool gate. Plan/registration/manifest/metadata source dictionaries are bound, but this helper does not hash actual production source snapshots or native binaries; that belongs to execution audit.','The synthetic collect uses old complete V20 auxiliary values and fake IO/NPZ callbacks to test control-flow/denominators, not to claim new9 numerical outcomes.','3 variants times3 repeated contexts with2physical clusters; historical27 comparisons are reused references, not new experiments.','No task/guard/safety/physical placement benefit is certified by this reader review.'])
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');out.with_suffix('.md').write_text('# V21辅助读数工具独立门\n\n'+('PASS' if result['passed'] else 'FAIL')+f"；{sum(c['passed'] for c in cases)}/{len(cases)}限定检查。冻结v3/原V18 metrics和旧完整V20投影核对，内外all9屏障、dict metadata与variant、9/9/27/3分母、参数配对、NULL/失败/partial和精确辅助桥均用独立callback fixture实际运行。\n\n未读新V21结果；synthetic collect不是真新9计算。工具不以P0或优化器计数作收益指标，也不替代actualsource/snapshot/native环境的执行审计。生产源/原指标/结果未修改，0模型/物理。\n",encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(cases),failed=[c for c in cases if not c['passed']],out=str(out),sha256=sha(out),reader_sha256=before),ensure_ascii=False));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

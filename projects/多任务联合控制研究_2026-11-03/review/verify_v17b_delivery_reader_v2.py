"""Reader-only bounded gates, no new80 outcome/NPZ or execution access."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import read_v17b_delivery_prospective80_v1 as subject
CHECKS=[]
def ck(name,value):CHECKS.append(dict(name=name,passed=bool(value)))
def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return True
    return False
def fake_row(role,seed=619741,*,found=False,task=False,known=True):
    m=dict(coverage={'all_valid':known},official_first_stable_receiver={'latency_s':.8} if found else None,
        official_first_stable_by_release_deadline={'latency_s':.8} if found else None,
        official_final_stable_suffix=None,ever_floor_contact=False,terminal={'target_receiver_top_contact':True,
        'moving_receiver_contact':True,'linear_speed_m_s':.01,'angular_speed_rad_s':.2},
        first_receiver_impact20ms=None,first_receiver_impact100ms=None,first_floor_impact100ms=None)
    key=subject.record_key(role,'nominal','hold',seed)
    return dict(record_key=key,record_id=subject.compact(key),role=role,scenario='nominal',motion_mode='hold',
        physical_seed=seed,assignment_status='evaluated',status='computed_complete',auxiliary=m,
        original_result={'task_success':task},physical_config={'mass':.12},noise={'sigma':.01},limits=[5]*5,budget=18.)

def main(out):
    plan=subject.pinned_plan();rows=subject.roster(plan)
    ck('static registered80 roles16 each, ratio3roles do not collide',len(rows)==80 and len({r['record_id'] for r in rows})==80 and len([r for r in rows if r['method']=='ratio_effort'])==48)
    wrong=deepcopy(plan);wrong['stages'][1]['arm']='holding_A'
    ck('duplicate same-role context/destination assignment rejected',rejects(lambda:subject.roster(wrong)))
    completion={s['name']:{'execution_complete':True,'sources_unchanged':True,'trials':s['planned'],'planned':s['planned']} for s in plan['stages']}
    progress={'complete':True,'physics_active':False,'master_plan_sha256':subject.PLAN_SHA}
    pending=deepcopy(completion);pending[plan['stages'][-1]['name']]=None
    ck('a single uncompleted stage cannot become complete80',rejects(lambda:subject.readiness(plan,progress,pending)))
    active=deepcopy(progress);active['physics_active']=True
    ck('active sequence yields pending even if files look complete',subject.readiness(plan,active,completion)['status']=='pending')
    ck('task failure never used as readiness criterion',subject.readiness(plan,progress,completion)['ready'])
    rr=[fake_row('full_V17',found=True,task=False),fake_row('full_V17',619742,found=False,task=False)]
    summary=subject.summarize(rr)
    ck('allassigned failures retained and no-stability never0seconds',summary['assigned']==2 and summary['metrics']['first_stable_found']=={'assigned':2,'known':2,'unknown':0,'true':1,'false':1}
        and summary['metrics']['first_stable_latency_s']['available']==1 and summary['metrics']['first_stable_latency_s']['median']==.8)
    missing=dict(fake_row('full_V17'),auxiliary=None,status='invalid_or_missing_record',assignment_status='runtime_failed')
    sm=subject.summarize([missing])
    ck('allunknown role still exposes complete fixed metric registry and unknown denominator',len(sm['metrics'])==len(subject.METRICS) and sm['metrics']['first_stable_found']['unknown']==1
        and sm['metrics']['first_stable_latency_s']['median'] is None and sm['assignment_status_counts']['runtime_failed']==1)
    pairrows=[fake_row(role) for role in subject.ROLES];pairs=subject.paired_records(pairrows)
    ck('seven predefined five-arm contrasts keep both-notreached latency null',len(pairs)==7 and all(p['differences']['first_stable_latency_s']['delta'] is None for p in pairs)
        and sum(p['primary_same_prior_contrast'] for p in pairs)==1 and sum(p['component_contrast'] for p in pairs)==1)
    pairrows[-1]['physical_config']={'mass':.2};bad=subject.paired_records(pairrows)
    ck('physical mismatch retained labelled unavailable rather than trusted matched contrast',sum(not p['actual_physical_noise_limits_equal'] for p in bad)==2
        and all(p['differences']['terminal_speed_m_s']['delta'] is None for p in bad if not p['actual_physical_noise_limits_equal']))
    stopped={'master_plan_sha256':subject.PLAN_SHA,'complete':False,'physics_active':False,'stopped_on_runtime_failure':True,'ended_utc':'fixture'}
    ck('stopped incomplete cohort is separate archive status never ready',subject.readiness(plan,stopped,pending)['status']=='terminal_incomplete' and not subject.readiness(plan,stopped,pending)['ready'])
    # Fixed first historical v3 record only, to verify caller passes the frozen
    # API faithfully. No new80 files/outcomes are opened by this test.
    previous=subject.read(ROOT/'review/delivery_auxiliary_full15_posthoc_v3.json')['records'][0]
    run=Path(previous['run']);meta=subject.read(run/'metadata.json');original=subject.read(run/'result.json')
    with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in subject.aux.C['input_fields']}
    new=subject.aux.calculate(arrays,release_at_s=meta['mechanical_release_at'],episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=original)
    ck('first old15 archive caller regression remains exact frozenv3',new==previous['metrics'])
    item=dict(record_key=subject.record_key('full_V17',meta['scenario'],meta['motion_mode'],meta['seed']),
        record_id=subject.compact(subject.record_key('full_V17',meta['scenario'],meta['motion_mode'],meta['seed'])),
        role='full_V17',scenario=meta['scenario'],motion_mode=meta['motion_mode'],physical_seed=meta['seed'],method=meta['method'],
        task_policy=meta['task_policy'],receiver_policy=meta['receiver_policy'],run=str(run))
    loaded=subject.load_assigned(item,{'status':'evaluated','job':{'destination':str(run)}},original,
        {'source_sha256':meta['research_source_sha256']},[])
    ck('actual oldrecord loader preserves identity and frozen auxiliary values',loaded['status']=='computed_complete'
        and loaded['execution_source_identity_equal'] and loaded['auxiliary']==new and not loaded['issues'])
    doc=dict(schema='v17b_delivery_reader_software_gate_v2',passed=all(c['passed'] for c in CHECKS),checks=CHECKS,
        count=len(CHECKS),source_sha256=subject.sha(Path(subject.__file__)),plan_sha256=subject.PLAN_SHA,
        calculator_sha256=subject.CALC_SHA,contract_sha256=subject.CONTRACT_SHA,
        scope=dict(new80_outcomes_read=False,models=0,physics_steps=0,old_regression_records=1,no_source_or_frozen_metric_edits=True))
    p=ROOT/out
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(p),passed=doc['passed'],count=len(CHECKS),failures=[c['name'] for c in CHECKS if not c['passed']],sha256=subject.sha(p))))
    return 0 if doc['passed'] else 1
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

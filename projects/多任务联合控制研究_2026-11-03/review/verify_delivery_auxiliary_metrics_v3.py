"""Closed-form temporal/coverage counterexamples, no historical results read."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import delivery_auxiliary_metrics_v3 as subject
CHECKS=[]
def ck(name,value):CHECKS.append(dict(name=name,passed=bool(value)))
def fixture(period=.005,start=11.8,end=16.):
    e=np.arange(round(start/period)+1,round(end/period)+1)*period;n=len(e)
    q=np.zeros((n,7));q[:,3]=1;v=np.zeros((n,6))
    a=dict(interval_start_s=e-period,interval_end_s=e,contact_valid=np.ones(n,bool),object_state_valid=np.ones(n,bool),
        object_free_qpos7_end=q,object_free_qvel6_end=v)
    for name in subject.CATS:a['target_'+name+'_normal_N']=np.zeros(n)
    a['target_receiver_top_normal_N'][:]=1.
    return a
def run(a,period=.005,original=None):return subject.calculate(a,release_at_s=12.,physical_period_s=period,original_result=original)
def near(x,y):return x is not None and abs(x-y)<1e-8

def main():
    a=fixture();r=run(a)
    ck('continuous200ms and prewindow reset: confirmation12.2 not credit before12',r['coverage']['all_valid'] and near(r['official_first_stable_receiver']['confirmation_s'],12.2) and near(r['official_final_stable_suffix']['start_s'],12.))
    a=fixture();remove=np.argmin(abs(a['interval_end_s']-12.1));a={k:np.delete(v,remove,axis=0) for k,v in a.items()};r=run(a)
    ck('one missing interval makes official first unknown and restarts observed dwell',r['official_first_stable_receiver'] is None and r['status']=='incomplete_or_invalid_unknown' and r['observed_stable_candidates']['first']['start_s']>12.1)
    a=fixture();a['target_hand_normal_N'][:]=.3;r=run(a)
    ck('uncleared hand cannot be called stable placement',r['official_first_stable_receiver'] is None and r['no_stability_reason']=='raw_hand_never_cleared')
    a=fixture();a['target_floor_normal_N'][np.argmin(abs(a['interval_end_s']-12.1))]=.1;r=run(a)
    ck('prior floor survives a later quiet receiver dwell',r['ever_floor_contact'] and r['stable_after_previous_floor_event'] and r['official_first_stable_receiver'] is not None and not r['physical_or_permanent_placement_certified'])
    a=fixture();a['object_free_qvel6_end'][a['interval_end_s']<15.6,0]=.01;r=run(a)
    ck('afterdeadline stable any remains separate from bydeadline null',near(r['official_first_stable_receiver']['confirmation_s'],15.8) and r['official_first_stable_by_release_deadline'] is None)
    a=fixture();a['target_receiver_top_normal_N'][:]=0;a['target_receiver_top_normal_N'][-1]=10;r=run(a)
    ck('late contact impact and short terminal suffix rightcensored, no false200ms dwell',r['first_receiver_impact100ms']['right_censored'] and r['first_receiver_impact100ms']['contact']['receiver_top']['peak_N'] is None and r['official_first_stable_receiver'] is None and r['observed_stable_candidates']['final_suffix_insufficient_dwell'])
    a=fixture();a['target_receiver_top_normal_N'][:]=10;r=run(a)
    ck('constant10N receiver impact integral20ms=.2Ns/100ms=1Ns',near(r['first_receiver_load_window20ms']['contact']['receiver_top']['impulse_Ns'],.2) and near(r['first_receiver_load_window100ms']['contact']['receiver_top']['impulse_Ns'],1.) and r['first_floor_impact100ms'] is None)
    a=fixture();last=len(a['interval_end_s'])-1;a={k:np.delete(v,last,axis=0) for k,v in a.items()};r=run(a)
    ck('missing terminal state remains null rather than zero velocity',r['terminal'] is None and r['official_first_stable_receiver'] is None)
    a=fixture();j=np.argmin(abs(a['interval_end_s']-12.1));a['object_state_valid'][j]=False;a['object_free_qvel6_end'][j]=np.nan;r=run(a)
    ck('invalid state never zero filled and full window no official stability',r['status']=='incomplete_or_invalid_unknown' and r['official_first_stable_receiver'] is None and r['contact_coverage']['all_valid'])
    a=fixture();j=np.argmin(abs(a['interval_end_s']-12.5));a['object_free_qvel6_end'][j:,5]=.2;old={'task_success':False,'record_complete':True,'mean_force':None,'guard':False};r=run(a,original=old)
    ck('later movement breaks final stability, original failureNULL kept verbatim',r['official_first_stable_receiver'] is not None and r['official_final_stable_suffix'] is None and r['loss_after_first']['known_condition_loss_episodes']==1 and r['terminal']['moving_receiver_contact'] and not r['terminal']['rolling_identified'] and r['original_result']==old)
    a=fixture();r=run(a)
    ck('preexisting receiver load has load windows but no fabricated new impact',r['receiver_event_provenance']['contact_at_command'] is True and r['first_receiver_impact20ms'] is None and r['first_receiver_load_window20ms']['available'])
    a=fixture();a['target_receiver_top_normal_N'][a['interval_end_s']<=12.1]=0;r=run(a)
    ck('valid contiguous zero-to-positive transition enables a genuine new contact window',r['receiver_event_provenance']['contact_at_command'] is False and r['receiver_event_provenance']['first_observed_new_transition_s'] is not None and r['first_receiver_impact20ms']['available'])
    a=fixture();a['target_floor_normal_N'][:]=1.;r=run(a)
    ck('preexisting floor remains everfloor without calling it a new collision',r['ever_floor_contact'] and r['floor_event_provenance']['contact_at_command'] is True and r['first_floor_impact100ms'] is None)
    a=fixture(period=.03,start=11.7,end=15.99);r=subject.calculate(a,release_at_s=12.,episode_end_s=15.99,physical_period_s=.03)
    suffix=r['official_final_stable_suffix'];end=a['interval_end_s']
    ck('final suffix confirmation uses a real sample when dwell is not integer periods',suffix is not None and np.any(abs(end-suffix['confirmation_s'])<1e-10) and suffix['confirmation_s']-suffix['start_s']>=.2-1e-7)
    a=fixture();a['target_floor_normal_N'][np.argmin(abs(a['interval_end_s']-12.1))]=1.;a['contact_valid'][np.argmin(abs(a['interval_end_s']-12.5))]=False;r=run(a)
    ck('known positive floor evidence survives unrelated later missing contact',r['ever_floor_contact'] is True and r['observed_first_floor_contact_s'] is not None and not r['contact_coverage']['all_valid'])
    a=fixture();a['target_floor_normal_N'][np.argmin(abs(a['interval_end_s']-12.1))]=1.;a['contact_valid'][np.argmin(abs(a['interval_end_s']-12.095))]=False;r=run(a)
    ck('known existing floor with missing prior interval retains event but has unknown onset and no impact',r['ever_floor_contact'] is True and r['floor_event_provenance']['first_positive_onset_status']=='UNKNOWN_PRIOR_CONTACT_OR_ONSET' and r['first_floor_impact100ms'] is None and r['first_floor_load_window100ms'] is not None)
    a=fixture();a['contact_valid'][np.argmin(abs(a['interval_end_s']-12.5))]=False;r=run(a)
    ck('no observed floor plus missing contact stays unknown rather than false',r['ever_floor_contact'] is None and r['floor_event_provenance']['observed_positive_contact'] is False)
    out=Path(__file__).with_name('delivery_auxiliary_metrics_author_v3.json')
    if out.exists():raise FileExistsError(out)
    doc=dict(schema='delivery_auxiliary_metrics_author_v3',passed=all(x['passed'] for x in CHECKS),checks=CHECKS,
        count=len(CHECKS),contract_sha256=hashlib.sha256(subject.CONTRACT_PATH.read_bytes()).hexdigest(),
        calculator_sha256=hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest(),historical_data_read=0,physics_steps=0)
    out.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(out),passed=doc['passed'],failures=[x['name'] for x in CHECKS if not x['passed']])))
    return 0 if doc['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

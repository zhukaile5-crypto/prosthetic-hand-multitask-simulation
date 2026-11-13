"""Independent offline delivery-boundary oracles; no simulator imports/runs.

Tests use synthetic complete sidecars and hand-derived sample/interval answers,
not author test helpers. A positive observed event survives unrelated missingness.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('receiver_top','receiver_otherface','floor','other_nonhand_world','hand','other_nonworld')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(p):
    q=Path(p);return q if q.is_absolute() else ROOT/q
def fixture(period=.01,end=5.):
    n=int(round(end/period));t=np.arange(1,n+1,dtype=float)*period
    q=np.zeros((n,7));q[:,3]=1.
    a=dict(interval_start_s=np.arange(n,dtype=float)*period,interval_end_s=t,
        contact_valid=np.ones(n,bool),object_state_valid=np.ones(n,bool),
        object_free_qpos7_end=q,object_free_qvel6_end=np.zeros((n,6)))
    for k in FIELDS:a['target_'+k+'_normal_N']=np.ones(n) if k=='receiver_top' else np.zeros(n)
    return a
def main():
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--contract',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    modpath=resolve(args.module);contract=resolve(args.contract);out=resolve(args.out)
    if out.exists():raise FileExistsError(out)
    before={str(p.relative_to(ROOT)):sha(p) for p in (modpath,contract)}
    sp=importlib.util.spec_from_file_location('independent_delivery_subject',modpath);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    checks=[]
    def ck(name,condition,detail=None):checks.append(dict(name=name,passed=bool(condition),detail=detail))
    def near(x,y):return x is not None and math.isclose(x,y,rel_tol=0,abs_tol=1e-11)
    def run(a,**kw):return m.calculate(a,release_at_s=kw.pop('release_at_s',1.),episode_end_s=kw.pop('episode_end_s',5.),physical_period_s=kw.pop('physical_period_s',.01),**kw)
    ck('actual_contract_bound_and_threshold_values',m.CONTRACT_PATH.resolve()==contract.resolve() and m.TH==dict(absolute_free_joint_linear_speed_m_s=.005,absolute_free_joint_angular_speed_rad_s=.15,stable_dwell_s=.2,all_target_hand_contact_clearance_N=.2,raw_hand_clearance_dwell_s=.2,original_release_deadline_s=3.5,positive_contact_N=0.,clock_tolerance_s=1e-7))
    a=fixture();a['object_free_qvel6_end'][:,0]=.005;a['object_free_qvel6_end'][:,3]=.15;a['target_hand_normal_N'][:]=.2
    y=run(a);ck('inclusive_exact_speed_hand_thresholds_and_command_dwell',near(y['official_first_stable_receiver']['confirmation_s'],1.2) and near(y['official_final_stable_suffix']['confirmation_s'],1.2),y['official_first_stable_receiver'])
    for name,field,axis,value in [('linear','object_free_qvel6_end',0,.005),('angular','object_free_qvel6_end',3,.15),('hand','target_hand_normal_N',None,.2)]:
        a=fixture()
        if axis is None:a[field][:]=np.nextafter(value,np.inf)
        else:a[field][:,axis]=np.nextafter(value,np.inf)
        y=run(a);ck('oneULP_above_'+name+'_threshold_is_not_stable',y['official_first_stable_receiver'] is None)
        if axis is None:a[field][:]=np.nextafter(value,-np.inf)
        else:a[field][:,axis]=np.nextafter(value,-np.inf)
        ck('oneULP_below_'+name+'_threshold_can_stabilize',run(a)['official_first_stable_receiver'] is not None)
    a=fixture();a['object_free_qvel6_end'][a['interval_end_s']>=4.8,0]=.006;y=run(a)
    ck('late_motion_revokes_final_suffix_without_erasing_first',y['official_first_stable_receiver'] is not None and y['official_final_stable_suffix'] is None and y['terminal']['moving_receiver_contact'] and y['loss_after_first']['known_condition_loss_episodes']==1)
    a=fixture();a['target_receiver_top_normal_N'][a['interval_end_s']<4.9]=0;y=run(a)
    ck('final_short_dwell_is_right_censored',y['official_first_stable_receiver'] is None and y['official_final_stable_suffix']['right_censored'] and y['official_final_stable_suffix']['confirmation_s'] is None)
    a=fixture();e=a['interval_end_s'];hit=(e>=1.3)&(e<=1.5);a['target_floor_normal_N'][hit]=2.;a['target_receiver_top_normal_N'][hit]=0;y=run(a)
    ck('calm_receiver_after_floor_keeps_known_floor_history',y['ever_floor_contact'] is True and y['stable_after_previous_floor_event'] is False and y['official_first_stable_receiver'] is not None and y['official_final_stable_suffix'] is not None,y['ever_floor_contact'])
    # Avoid an earlier stable episode: stable only after the floor event.
    a['target_receiver_top_normal_N'][e<1.6]=0;y=run(a)
    ck('postfloor_first_stable_is_explicitly_tagged_not_new_success',y['ever_floor_contact'] is True and y['stable_after_previous_floor_event'] is True and not y['physical_or_permanent_placement_certified'])
    for other in ('receiver_otherface','floor','other_nonhand_world','other_nonworld'):
        a=fixture();a['target_'+other+'_normal_N'][:]=1e-12;y=run(a)
        ck('quiet_other_support_'+other+'_is_not_receiver_stability',y['official_first_stable_receiver'] is None)
    a=fixture();a['contact_valid'][-20]=False;y=run(a)
    ck('missing_contact_makes_official_stability_unknown',y['status']=='incomplete_or_invalid_unknown' and y['official_first_stable_receiver'] is None and y['official_final_stable_suffix'] is None and y['ever_floor_contact'] is None)
    # Logical existential oracle: one observed positive floor is sufficient,
    # and an unrelated unknown record cannot make that positive fact unknown.
    a=fixture();a['target_floor_normal_N'][130]=2.;a['contact_valid'][-20]=False;y=run(a)
    ck('observed_positive_floor_survives_unrelated_missing_contact',y['ever_floor_contact'] is True,dict(ever_floor_contact=y['ever_floor_contact'],observed_first_floor_contact_s=y['observed_first_floor_contact_s']))
    a=fixture();a['target_receiver_top_normal_N'][a['interval_end_s']>1.05]=0;y=run(a)
    ck('prewindow_good_samples_do_not_credit_release_dwell',y['official_first_stable_receiver'] is None)
    original=dict(task_success=False,research_metrics=dict(release=dict(physical_exposure=False),quality_windows=dict(static=None)),reason='retained_failure')
    frozen=deepcopy(original);y=run(fixture(),original_result=original)
    ck('original_failure_unexposed_and_NULL_preserved',original==frozen and y['original_result']==frozen and y['original_release_physical_exposure'] is False and y['original_metric_or_success_redefined'] is False)
    y['original_result']['task_success']=True;ck('original_record_is_independent_deepcopy',original==frozen)
    y=run(fixture());ck('preexisting_receiver_is_load_not_new_impact',y['receiver_event_provenance']['contact_at_command'] is True and y['first_receiver_impact20ms'] is None and y['first_receiver_load_window20ms']['available'])
    a=fixture();a['target_receiver_top_normal_N'][a['interval_end_s']<=1.3]=0;y=run(a)
    ck('new_contiguous_transition_exact_closedform_impulse',near(y['receiver_event_provenance']['first_observed_new_transition_s'],1.3) and near(y['first_receiver_impact20ms']['contact']['receiver_top']['impulse_Ns'],.02) and y['receiver_event_provenance']['contact_at_command'] is False)
    a=fixture();e=a['interval_end_s'];a['target_receiver_top_normal_N'][e<=1.3]=0;a['contact_valid'][(e>=1.0)&(e<=1.3)]=False;y=run(a)
    ck('positive_after_unknown_without_observed_zero_is_not_new_impact',y['receiver_event_provenance']['contact_at_command'] is None and y['first_receiver_impact20ms'] is None and y['first_receiver_contact_s'] is not None)
    a=fixture();a['target_receiver_top_normal_N'][a['interval_end_s']<4.4]=0;y=run(a)
    ck('late_any_stability_not_backdated_before_release_deadline',y['official_first_stable_receiver'] is not None and y['official_first_stable_by_release_deadline'] is None)
    a=fixture(.03,3.);y=run(a,release_at_s=.3,episode_end_s=3.,physical_period_s=.03)
    ck('confirmation_uses_actual_sample_not_interpolation',near(y['official_first_stable_receiver']['confirmation_s'],.51) and near(y['official_final_stable_suffix']['confirmation_s'],.51))
    a=fixture();idx=200
    for key in a:a[key]=np.delete(a[key],idx,axis=0)
    y=run(a);ck('missing_time_interval_unknown_not_false_complete',not y['coverage']['complete'] and y['official_first_stable_receiver'] is None)
    a=fixture();a['target_receiver_top_normal_N'][a['interval_end_s']<4.96]=0;y=run(a)
    ck('truncated_new_impact_retains_observed_but_null_official100ms',y['first_receiver_impact100ms']['right_censored'] and y['first_receiver_impact100ms']['contact']['receiver_top']['peak_N'] is None and y['first_receiver_impact100ms']['contact']['receiver_top']['observed_partial_peak_N']==1.)
    after={str(p.relative_to(ROOT)):sha(p) for p in (modpath,contract)};ck('subject_source_and_contract_unchanged',before==after)
    doc=dict(schema='delivery_auxiliary_independent_oracles_v1',passed=all(x['passed'] for x in checks),checks=checks,check_count=len(checks),
        contract_path=str(contract.relative_to(ROOT)),contract_sha256=sha(contract),calculator_path=str(modpath.relative_to(ROOT)),calculator_sha256=sha(modpath),reader_sha256=sha(modpath),
        independent_checker_sha256=sha(__file__),scope=dict(physics_steps=0,model_constructions=0,registration_calls=0,tests_use_synthetic_sidecars_only=True,author_test_helpers_imported=False,old_results_modified=False))
    out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=doc['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']],out=str(out),sha256=sha(out))))
    return 0 if doc['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

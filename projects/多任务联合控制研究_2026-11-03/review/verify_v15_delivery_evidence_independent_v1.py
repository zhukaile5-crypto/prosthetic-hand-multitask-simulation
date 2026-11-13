"""Small independent closed-form ledger tests; fake metadata/contacts only."""
from pathlib import Path
import hashlib, importlib.util, json, pickle, sys, traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v15_support.delivery_evidence import DeliveryLedger,CAPTURE_PHASE,CATEGORIES
EXPECTED='1c79a3b258d98af763036d3aa8269bca1bd1e5c7427c2d3adeb9c3402b3d2a65'
# Reuse only Python fake type definitions, not the author's run/main or assertions.
spec=importlib.util.spec_from_file_location('delivery_fake_types',ROOT/'review/verify_v15_delivery_evidence_author_v1.py')
fake=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fake;spec.loader.exec_module(fake)
READS=[]
def reader(model,data,index,out):READS.append(index);out[:]=data.contact[index].wrench
def ledger(window=(12.,12.001),**extra):
    return DeliveryLedger(fake.Model(),target_geom_ids=[3,4],hand_geom_ids=[5],receiver_mount=fake.mount(),
        diagnostic_window_s=window,contact_force_reader=reader,**extra)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def close(name,a,b):check(name,np.allclose(a,b,atol=2e-12,rtol=2e-12),dict(actual=np.asarray(a).tolist(),expected=np.asarray(b).tolist()))
    def reject(name,call):
        try:call()
        except (ValueError,TypeError):check(name,True)
        else:check(name,False)
    global_before=pickle.dumps(np.random.get_state())
    flip=np.diag([-1.,1.,-1.])@np.array([[0,0,1.],[1.,0,0],[0,1.,0]])
    c=fake.contact
    contacts=[c((1,3),2.,(.1,.2,.31)),c((4,1),3.,(-.2,.1,.31),Q=flip),
        c((1,3),5.,(1.,0,.3),Q=np.eye(3)),c((0,4),7.,(0,0,0)),c((2,3),11.),
        c((4,5),13.),c((8,3),17.),c((1,6),19.),c((1,5),23.)]
    L=ledger();data=fake.Data(contacts,12.001);record=L.capture_cached_interval(data,12.,12.001,phase=CAPTURE_PHASE)
    close('mutually_exclusive_multi_geom_classes_closed_form',record.target_normal_N,[5,5,7,11,13,17])
    close('positive_counts_not_geometry_count_or_actor_identity',record.target_positive_contact_count,[2,1,1,1,1,1])
    close('receiver_any_actor_separate_overlapping_total',record.receiver_any_actor_top_normal_N,47.)
    close('receiver_any_actor_wrench_with_two_target_levers',record.receiver_any_actor_top_wrench6,[0,0,-47.,-.7,-.4,0])
    s=L.summary();close('class_impulse_matches_full_1ms_interval',[s['target_contact_classes'][k]['impulse_Ns'] for k in CATEGORIES],np.array([5,5,7,11,13,17])*.001)
    check('floor_post_side_hand_dynamic_categories_distinct',s['floor_event'] is True and s['categories_mutually_exclusive_for_each_target_contact'] and s['receiver_any_actor_top_is_separate_overlapping_total'])
    check('qv_are_independent_integration_end_values',np.array_equal(record.object_free_qpos7_end,data.qpos[5:12]) and np.array_equal(record.object_free_qvel6_end,data.qvel[4:10]))
    n=len(READS);data.contact=[];data.ncon=0;data.time=13.;again=L.capture_cached_interval(data,12.,12.001)
    check('same_data_duplicate_cached_idempotent_without_new_reads',again is record and len(READS)==n and L.summary()['records']==1)
    reject('overlapping_interval_rejected_before_double_count',lambda:L.capture_cached_interval(data,12.0005,12.002))
    reject('rewound_interval_rejected',lambda:L.capture_cached_interval(data,11.,11.5))
    # Previously registered integration oracles, independent of the author's examples.
    clipped=ledger();cd=fake.Data([c((1,3),2.)],12.001);clipped.capture_cached_interval(cd,11.999,12.001)
    close('registered_window_clipping_2N_times_1ms',clipped.summary()['target_contact_classes']['receiver_top']['impulse_Ns'],.002)
    partition=ledger();pd=fake.Data([c((1,4),2.)],12.0005);partition.capture_cached_interval(pd,12.,12.0005);pd.time=12.001;partition.capture_cached_interval(pd,12.0005,12.001)
    close('registered_partition_additivity',partition.summary()['target_contact_classes']['receiver_top']['impulse_Ns'],clipped.summary()['target_contact_classes']['receiver_top']['impulse_Ns'])
    # Known-empty, missing-force and missing-state remain three distinct states.
    zero=ledger();zd=fake.Data([],12.001);zr=zero.capture_cached_interval(zd,12.,12.001);zs=zero.summary()
    check('complete_empty_contact_is_valid_zero_not_unknown',zr.contact_valid and zs['floor_event'] is False and zs['target_contact_classes']['floor']['impulse_Ns']==0.)
    bad=ledger();bd=fake.Data([c((0,3),2.)],12.001);bd.contact[0].wrench[0]=np.nan;br=bad.capture_cached_interval(bd,12.,12.001);bs=bad.summary()
    check('invalid_force_keeps_nan_masks_and_null_complete_statistics',not br.contact_valid and br.object_state_valid and np.isnan(br.target_normal_N).all() and bs['floor_event'] is None and bs['target_contact_classes']['floor']['impulse_Ns'] is None)
    qbad=ledger();qd=fake.Data([],12.001);qd.qpos[5]=np.nan;qr=qbad.capture_cached_interval(qd,12.,12.001)
    check('invalid_object_state_does_not_corrupt_known_contact_zero',qr.contact_valid and not qr.object_state_valid and np.isnan(qr.object_free_qpos7_end).all() and qbad.summary()['floor_event'] is False)
    gap=ledger((12.,12.002));gd=fake.Data([],12.0005);gap.capture_cached_interval(gd,12.,12.0005);gd.time=12.002;gap.capture_cached_interval(gd,12.001,12.002);gs=gap.summary()
    check('gap_is_uncovered_not_zero_filled_absence',not gs['window_coverage_complete'] and not gs['window_contact_evidence_available'] and gs['floor_event'] is None and gs['target_contact_classes']['floor']['peak_N'] is None)
    close('gap_duration_and_valid_observed_integral_preserved',[gs['window_uncovered_duration_s'],gs['target_contact_classes']['floor']['observed_valid_impulse_Ns']],[.0005,0.])
    clock=ledger();cr=clock.capture_cached_interval(fake.Data([],12.002),12.,12.001)
    check('end_clock_mismatch_unknown_contact_and_state',not cr.contact_valid and not cr.object_state_valid and clock.summary()['window_terminal_object_state']['available'] is False)
    reject('sampling_phase_cannot_be_relabelled',lambda:ledger().capture_cached_interval(fake.Data([],12.001),12.,12.001,phase='fresh_after_forward'))
    reject('incomplete_object_geom_list_rejected',lambda:DeliveryLedger(fake.Model(),target_geom_ids=[3],receiver_mount=fake.mount()))
    check('offline_truth_never_exposed_as_online_sensor',s['evaluation_only'] and s['controller_input_permitted'] is False and not hasattr(L,'to_public') and not s['delivery_success_redefined'] and not s['physical_safety_certified'])
    check('cached_contact_and_end_state_role_not_silently_unified','not continuous-time' in s['contact_time_role'] and 'not guaranteed same instantaneous' in s['object_state_time_role'])
    check('no_global_rng_changes_or_mujoco_import',global_before==pickle.dumps(np.random.get_state()) and 'mujoco' not in sys.modules)
    return dict(passed=all(t['passed'] for t in checks),checks=checks,
        scope_limits=['Fake Model/Data/contact definitions reused, but no author main/assertions/run executed; new inputs and closed-form assertions are independent.',
            'All target IDs and free-body state are privileged offline evaluation, not controller information.',
            'Model/data same-episode binding, cached-solve producer timing and physical blind reproduction remain future integration checks.',
            'Window completeness uses the declared 1e-7s clock tolerance; this is not exact continuous-time contact or a delivery/safety certificate.'])
if __name__=='__main__':
    out=ROOT/'review/v15_delivery_evidence_independent_v1.json';source=ROOT/'v15_support/delivery_evidence.py'
    if out.exists():raise FileExistsError(out)
    before=sha(source);assert before==EXPECTED
    try:r=run()
    except Exception as exc:r=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
    r.update(schema='v15_delivery_evidence_independent_v1',role='independent_peer_fake_closed_form',source_sha256=before,source_unchanged=sha(source)==before,
        checker_sha256=sha(__file__),fake_definitions_source_sha256=sha(ROOT/'review/verify_v15_delivery_evidence_author_v1.py'),physics_steps=0,model_constructions=0,mujoco_imported='mujoco' in sys.modules)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r.get('checks',[])),failed=[c['name'] for c in r.get('checks',[]) if not c['passed']],sha256=sha(out),path=str(out)),ensure_ascii=False))
    raise SystemExit(0 if r['passed'] else 1)

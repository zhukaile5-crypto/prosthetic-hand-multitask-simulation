"""Pure synthetic plus optional actual-data-only bank QA. No MuJoCo import."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('servo_bank_test_module',ROOT/'v10/servo_bank.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
AXES=tuple('rh_J'+str(i) for i in range(24))+('transport_lift',)


def fixture(cid,base,gain):
    plus=np.ones(8,bool);plus[5]=False
    return dict(context_id='capsule/'+cid,shape='capsule',axes25=AXES,anchor_ref8=np.full(8,base),anchor_q25=np.zeros(25),anchor_velocity25=np.zeros(25),
        response25x8=np.full((25,8),gain),error25=np.linspace(.001,.002,25),micro_scale8=np.ones(8),proposal_lower8=-np.ones(8),proposal_upper8=np.r_[np.ones(5),0.,1.,1.],
        known_plus8=plus,known_minus8=np.ones(8,bool),mixed_lower_ref8=np.full(8,base-.1),mixed_upper_ref8=np.full(8,base+.1),mixed_reference_points8=np.array([np.full(8,base)]))


def verify(actual=False):
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    bank=m.Bank.from_entries_for_test(AXES,[fixture('a',0.,11.),fixture('b',1.,22.)])
    q=np.arange(25)*.001;v=np.arange(25)*.01;ref=np.r_[np.full(4,.25),np.full(4,.75)]
    selected=bank.select('capsule',q,v,ref)
    check('nearest_uses_single_whole_anchor',selected.debug['context_id']=='capsule/a' and np.array_equal(selected.response25x8,np.full((25,8),11.)))
    check('qfree_true_actual_velocity',np.array_equal(selected.free25,q+.02*v))
    check('not_global_or_sequence_admission',not selected.global_model_admitted and 'UNASSESSED' in selected.debug['sequence_support'])
    check('unknown_contact_mode_explicit','UNKNOWN' in selected.debug['current_contact_mode_assessment'])
    check('future_not_implicitly_known',selected.debug['actual_future_scope']=='NOT_CHECKED')
    bad=np.zeros(8);bad[5]=.1;scope=selected.check_future_increment(bad)
    check('unsupported_positive_lift_not_zero_gain',scope['status']=='UNSUPPORTED_FUTURE' and 'unmeasured_positive_direction_5' in scope['reasons'] and np.all(selected.response25x8[:,5]==11.))
    no=bank.select('capsule',q,v,ref,future_increment8=bad)
    check('select_unsupported_future_has_matrixNone',no.status=='UNSUPPORTED_FUTURE' and no.response25x8 is None)
    good=np.zeros(8);good[5]=-.1;yes=bank.select('capsule',q,v,ref,future_increment8=good)
    check('measured_direction_experimental_not_permission',yes.status=='EMPIRICAL_ANCHOR_LINEARIZATION' and yes.debug['future_component_check']['status']=='EMPIRICAL_COMPONENT_CANDIDATE')
    huge=np.zeros(8);huge[0]=1.1
    check('actual_micro_magnitude_checked',selected.check_future_increment(huge)['status']=='UNSUPPORTED_FUTURE')
    check('missing_velocity_unavailable_not_zero',bank.select('capsule',q,None,ref).response25x8 is None)
    nan=v.copy();nan[-1]=np.nan
    check('missing_lift_velocity_unavailable',bank.select('capsule',q,nan,ref).status=='UNAVAILABLE')
    check('unknown_morph_no_matrix',bank.select('unknown',q,v,ref).response25x8 is None)
    other=bank.select('capsule',q,v,np.full(8,5.))
    check('outside_footprint_explicit_no_fake_cert',other.debug['reference_outside_component_footprint'] and other.status=='EMPIRICAL_ANCHOR_LINEARIZATION')
    check('empirical_error_kept_complete',np.array_equal(selected.error25,np.linspace(.001,.002,25)))
    try:selected.response25x8[0,0]=0.
    except ValueError:check('returned_response_readonly',True)
    else:raise AssertionError('read-only response')
    actual_rows=[]
    if actual:
        root=ROOT/'output/robot_servo_response_all15_v2';first=json.loads((root/'capsule/checkpoint_00/model.json').read_text(encoding='utf-8'))
        axes=tuple(first['output_axes25']);real=m.Bank(axes)
        check('actual_all15_source_load',len(real.entries)==15 and not real.load_errors)
        check('actual62_direct_dependency_fingerprints',len(real.source_identity())==62)
        for entry in real.entries:
            s=real.select(entry['shape'],entry['anchor_q25'],entry['anchor_velocity25'],entry['anchor_ref8'])
            check('anchor_'+entry['context_id'],s.debug['context_id']==entry['context_id'] and s.debug['normalized_reference_distance']==0.)
            check('K20_'+entry['context_id'],np.array_equal(s.response25x8,entry['response25x8']))
            check('A20error_'+entry['context_id'],np.array_equal(s.error25,entry['error25']))
            actual_rows.append(dict(context_id=entry['context_id'],known_plus8=s.known_plus8.tolist(),known_minus8=s.known_minus8.tolist(),
                source_fingerprints=s.debug['source_fingerprints']))
        absent=m.Bank(axes,research_root=ROOT/'nonexistent_bank_fixture')
        check('actual_missing_source_fail_closed',absent.select('capsule',q,v,np.zeros(8)).response25x8 is None)
    return dict(passed=True,new_physics=False,checks=checks,actual_data_only=actual,actual_entries=actual_rows,
        module_sha256=hashlib.sha256((ROOT/'v10/servo_bank.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual',action='store_true');p.add_argument('--out',type=Path);args=p.parse_args();result=verify(args.actual)
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='actual_entries'},indent=2))

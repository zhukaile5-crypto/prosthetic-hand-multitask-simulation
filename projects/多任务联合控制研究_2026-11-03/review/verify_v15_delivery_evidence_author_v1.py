"""Mock-only evaluation-ledger tests. No MuJoCo model/data/physics."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import argparse, hashlib, json, pickle, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v15_support.receiver_sensor import ReceiverMount
from v15_support.delivery_evidence import DeliveryLedger,CAPTURE_PHASE,CATEGORIES

class Model:
    def __init__(self):
        self.ngeom=9;self.nq=20;self.nv=18
        self.names=['floor','support','support_post','target_a','target_b','hand','other_actor','world_obstacle','other_dynamic']
        self.geom_bodyid=np.array([0,0,0,1,1,2,3,0,4]);self.geom_type=np.array([0,6,6,6,6,6,6,6,6])
        self.geom_size=np.ones((9,3));self.geom_size[1]=[1.,1.,.01]
        self.geom_pos=np.zeros((9,3));self.geom_pos[1]=[0.,0.,.3]
        self.geom_quat=np.tile([1.,0.,0.,0.],(9,1))
        self.jnt_type=np.array([3,0]);self.jnt_bodyid=np.array([2,1]);self.jnt_qposadr=np.array([0,5]);self.jnt_dofadr=np.array([0,4])
    def geom(self,name):return SimpleNamespace(id=self.names.index(name))
    def joint(self,name):
        if name!='object_free':raise ValueError('unknown joint')
        return SimpleNamespace(id=1)

def mount():return ReceiverMount('support',('floor','support_post'),[0.,0.,.01],1e-6,.999)
def contact(pair,normal,point=(0.,0.,.31),Q=None,dist=0.,wrench=None):
    if Q is None:Q=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    return SimpleNamespace(geom=np.array(pair),frame=np.asarray(Q).ravel(),pos=np.array(point),dist=dist,wrench=np.array([normal,0.,0.,0.,0.,0.]) if wrench is None else np.array(wrench))
class Data:
    def __init__(self,contacts=(),time=12.0005):
        self.time=time;self.contact=list(contacts);self.ncon=len(contacts);self.qpos=np.arange(20,dtype=float);self.qvel=np.arange(18,dtype=float)
        self.qpos[5:12]=[.2,.3,.4,1.,0.,0.,0.];self.qvel[4:10]=[3.,4.,0.,0.,0.,2.]

def reader(m,d,i,out):out[:]=d.contact[i].wrench
def ledger(model=None,**kwargs):return DeliveryLedger(Model() if model is None else model,target_geom_ids=[3,4],hand_geom_ids=[5],receiver_mount=mount(),contact_force_reader=reader,**kwargs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    source=ROOT/'v15_support/delivery_evidence.py';before=sha(source);checks=[];failures=[];detail=[];rng_before=pickle.dumps(np.random.get_state())
    def ck(name,v,data=None):
        checks.append(dict(name=name,passed=bool(v),detail=data))
        if not v:failures.append(name)
    def close(name,x,y):ck(name,np.allclose(x,y,atol=1e-12,rtol=1e-12),dict(actual=np.asarray(x).tolist(),expected=np.asarray(y).tolist()))
    def rejects(name,fn):
        try:fn();ck(name,False)
        except (ValueError,TypeError,IndexError,AttributeError):ck(name,True)
    try:
        flip=np.array([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])
        cs=[contact((1,3),2.),contact((4,1),3.,Q=flip),contact((1,3),4.,(1.,0.,.3),Q=np.eye(3)),
            contact((0,4),5.,(0.,0.,0.)),contact((2,3),6.),contact((3,5),7.),contact((4,6),8.),
            contact((1,6),11.),contact((1,5),13.),contact((4,7),17.),contact((0,8),19.,(0.,0.,0.)),
            contact((1,3),9.,(0.,0.,.29),Q=flip),contact((1,4),1.5,(1.,1.,.31)),contact((1,4),.25,(1.+3e-6,0.,.31))]
        d=Data(cs);L=ledger(diagnostic_window_s=(12.,12.001));r=L.capture_cached_interval(d,12.,12.0005,phase=CAPTURE_PHASE)
        expected=np.array([6.5,13.25,5.,23.,7.,8.]);close('multi_target_mutually_exclusive_classes',r.target_normal_N,expected)
        close('positive_contact_counts',r.target_positive_contact_count,[3,3,1,2,1,1]);close('top_any_actor_includes_robot_and_other_body',r.receiver_any_actor_top_normal_N,30.5)
        close('any_top_wrench_force_and_moment',r.receiver_any_actor_top_wrench6,[0.,0.,-30.5,-1.5,1.5,0.])
        ck('floor_not_whole_worldbody_or_receiver',r.target_normal_N[2]==5. and r.target_normal_N[3]==23.)
        old_count=len(L.arrays()['interval_end_s']);d.contact=[];d.ncon=0
        duplicate=L.capture_cached_interval(d,12.,12.0005)
        ck('same_interval_returns_first_cached_record_without_new_read',duplicate is r and len(L.arrays()['interval_end_s'])==old_count)
        d.contact=cs;d.ncon=len(cs);d.time=12.001;L.capture_cached_interval(d,12.0005,12.001)
        s=L.summary();ck('full_fixed_window_available',s['window_coverage_complete'] and s['window_contact_evidence_available'])
        for j,name in enumerate(CATEGORIES):
            st=s['target_contact_classes'][name];close('closed_form_impulse_'+name,st['impulse_Ns'],expected[j]*.001)
            close('closed_form_duration_'+name,st['positive_contact_duration_s'],.001)
        close('closed_form_anyactor_impulse',s['receiver_any_actor_top']['impulse_Ns'],.0305)
        ck('floor_event_recorded_not_delivery_success',s['floor_event'] and not s['delivery_success_redefined'] and not s['receiver_contact_proves_target_delivery'])
        close('qpos_uses_declared_free_joint_offset',s['window_terminal_object_state']['qpos7'],[.2,.3,.4,1.,0.,0.,0.])
        close('qvel_uses_declared_dof_offset',s['window_terminal_object_state']['qvel6'],[3.,4.,0.,0.,0.,2.])
        close('linear_speed',s['last_recorded_object_state']['linear_speed_m_s'],5.);close('angular_speed',s['last_recorded_object_state']['angular_speed_rad_s'],2.)
        ck('cached_contact_and_end_state_clock_roles_distinct','not guaranteed same instantaneous' in s['object_state_time_role'] and 'not continuous-time' in s['contact_time_role'])
        rejects('overlap_rejected',lambda:L.capture_cached_interval(d,12.00075,12.00125))
        rejects('rewind_rejected',lambda:L.capture_cached_interval(d,11.,11.5))
        rejects('wrong_phase_rejected',lambda:ledger().capture_cached_interval(Data(),12.,12.0005,phase='boundary_after_forward'))
        rejects('boolean_time_rejected',lambda:ledger().capture_cached_interval(Data(),True,12.0005))
        rejects('string_time_rejected',lambda:ledger().capture_cached_interval(Data(),'12',12.0005))
        rejects('zero_duration_rejected',lambda:ledger().capture_cached_interval(Data(),12.,12.))
        rejects('different_data_binding_rejected',lambda:L.capture_cached_interval(Data(time=12.0015),12.001,12.0015))
        rejects('record_immutable_array',lambda:r.target_normal_N.setflags(write=True))
        aa=L.arrays();aa['target_floor_normal_N'][:]=0.;ck('array_export_does_not_mutate_ledger',L.arrays()['target_floor_normal_N'][0]==5.)
        # Finite window clips an interval, rather than integrating outside it.
        c=contact((1,3),2.);dd=Data([c],12.001);LL=ledger(diagnostic_window_s=(12.,12.001));LL.capture_cached_interval(dd,11.999,12.001)
        close('window_boundary_clipped_closed_form',LL.summary()['target_contact_classes']['receiver_top']['impulse_Ns'],.002)
        zero=ledger(diagnostic_window_s=(12.,12.0005));z=zero.capture_cached_interval(Data([],12.0005),12.,12.0005);zs=zero.summary()
        ck('zero_contacts_valid_zero_not_unknown',z.contact_valid and np.array_equal(z.target_normal_N,np.zeros(6)) and zs['target_contact_classes']['floor']['impulse_Ns']==0. and zs['floor_event'] is False)
        tiny=ledger(diagnostic_window_s=(12.,12.0005));tiny.capture_cached_interval(Data([contact((0,3),1e-12)],12.0005),12.,12.0005)
        ck('no_invented_material_threshold',tiny.summary()['floor_event'] is True)
        # Missing interval or invalid force leaves uncertainty, not false zero.
        gap=ledger(diagnostic_window_s=(12.,12.0015));gd=Data([],12.0005);gap.capture_cached_interval(gd,12.,12.0005);gd.time=12.0015;gap.capture_cached_interval(gd,12.001,12.0015)
        gs=gap.summary();ck('gap_coverage_unavailable_not_zero',not gs['window_coverage_complete'] and gs['target_contact_classes']['floor']['impulse_Ns'] is None and gs['floor_event'] is None)
        bad=ledger(diagnostic_window_s=(12.,12.0005));bd=Data([contact((0,3),1.)],12.0005);bd.contact[0].wrench[0]=np.nan;br=bad.capture_cached_interval(bd,12.,12.0005)
        ck('unknown_force_kept_nan_and_summary_null',not br.contact_valid and br.object_state_valid and np.isnan(br.target_normal_N).all() and bad.summary()['target_contact_classes']['floor']['peak_N'] is None)
        badclock=ledger(diagnostic_window_s=(12.,12.0005));br=badclock.capture_cached_interval(Data([],13.),12.,12.0005)
        ck('clock_mismatch_invalidates_contact_and_state',not br.contact_valid and not br.object_state_valid and badclock.summary()['floor_event'] is None)
        badq=ledger(diagnostic_window_s=(12.,12.0005));dd=Data([],12.0005);dd.qpos[5]=np.nan;br=badq.capture_cached_interval(dd,12.,12.0005)
        ck('unknown_q_not_zero_contact_still_known',br.contact_valid and not br.object_state_valid and np.isnan(br.object_free_qpos7_end).all() and badq.summary()['floor_event'] is False)
        # Midpoint-to-receiver-surface correction is identical to receiver90ba.
        mid=ledger(diagnostic_window_s=(12.,12.0005));md=Data([contact((1,3),2.,(0.,0.,.309),dist=-.002)],12.0005)
        mr=mid.capture_cached_interval(md,12.,12.0005);ck('midpoint_surface_correction_top',mr.target_normal_N[0]==2.)
        nohand=DeliveryLedger(Model(),target_geom_ids=[3,4],receiver_mount=mount(),diagnostic_window_s=(12.,12.0005),contact_force_reader=reader)
        nh=nohand.capture_cached_interval(Data([contact((3,5),7.)],12.0005),12.,12.0005)
        ck('missing_hand_ids_not_world_support',nh.target_normal_N[3]==0. and nh.target_normal_N[4]==0. and nh.target_normal_N[5]==7. and not nohand.summary()['hand_geometry_ids_explicit'])
        mm=Model();mut=ledger(mm,diagnostic_window_s=(12.,12.001));dd=Data([],12.0005);mm.geom_pos[1,0]=1.;rr=mut.capture_cached_interval(dd,12.,12.0005);mm.geom_pos[1,0]=0.;dd.time=12.001;rr2=mut.capture_cached_interval(dd,12.0005,12.001)
        ck('model_mutation_latches_unknown_even_after_restore',not rr.contact_valid and not rr2.contact_valid)
        rejects('target_all_geoms_required',lambda:DeliveryLedger(Model(),target_geom_ids=[3],receiver_mount=mount()))
        rejects('target_id_bool_rejected',lambda:DeliveryLedger(Model(),target_geom_ids=[True,4],receiver_mount=mount()))
        rejects('floor_hand_overlap_rejected',lambda:DeliveryLedger(Model(),target_geom_ids=[3,4],receiver_mount=mount(),hand_geom_ids=[0]))
        m=Model();m.jnt_type[1]=3;rejects('object_joint_must_be_free',lambda:ledger(m))
        m=Model();m.jnt_qposadr[1]=18;rejects('free_coordinate_bounds_checked',lambda:ledger(m))
        m=Model();m.geom_bodyid[1]=2;rejects('receiver_static_top_only',lambda:ledger(m))
        empty=ledger().summary();ck('empty_record_unknown_not_zero',empty['floor_event'] is None and empty['target_contact_classes']['floor']['peak_N'] is None and not empty['window_terminal_object_state']['available'])
        ck('eval_only_no_public_measurement_method',not hasattr(L,'to_public') and s['controller_input_permitted'] is False)
        ck('independent_no_sensor_rng_draws',pickle.dumps(np.random.get_state())==rng_before)
        ck('source_stable',sha(source)==before)
        ck('no_mujoco_model_solver_import','mujoco' not in sys.modules and 'v15_support.task_qp' not in sys.modules)
        detail.append(dict(example=s))
    except Exception:failures.append('exception');detail.append(dict(exception=traceback.format_exc()))
    report=dict(schema='v15_delivery_evidence_mock_author_v1',passed=not failures,checks=checks,failures=failures,details=detail,
        source_sha256=before,receiver_source_sha256=sha(ROOT/'v15_support/receiver_sensor.py'),checker_sha256=sha(__file__),
        scope='Pure mocked contacts/model-address metadata, no real model/data construction, no mj_step/forward/FK/solver, no sensor queue or RNG sampling.',physics_steps=0)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failures=failures,source_sha256=before,out=str(out)),ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()

"""Synthetic contract tests for the articulated-hand observation adapter.

This script verifies measurement math and causal sampling. It is not a
physical acceptance or controller-performance experiment.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import numpy as np

from contracts import SensorNoise
from sensors import aggregate_patches, HandSensors

ROOT = Path(__file__).resolve().parent


def calibration():
    B = np.array([[2., -.4, 0., 0., 0.], [-.1, 3., 0., 0., 0.],
                  [0., 0., -1., .2, 0.], [0., 0., .1, 2.5, 0.], [0., 0., 0., 0., 1.5]])
    return dict(force_matrix=B, force_tau=.015, reference_alpha=np.full(5, .5),
                response_error_bound=np.full(5, .2),
                nominal_force_weights=[1., 1., 1., 1., 1.],
                effort_bias_N=.1, load_error_bound_N=.3)


def patch(digit=0, n=2., position=(.02, 0., 0.), force=None,
          direction=(1., 0., 0.), slip=(0., 0., .01), torque=(0., 0., 0.), target=True):
    if force is None: force=np.asarray(direction)*n+np.array([0., .3, -.2])
    return dict(digit=digit, normal=n, position=list(position), force=np.asarray(force).tolist(),
                normal_dir=list(direction), slip_velocity=list(slip), torque=list(torque),
                is_target=target)


def raw(t=0., patches=None):
    return dict(time=t, patches=[patch()] if patches is None else patches,
                joint=np.full(5, .5+t), joint_velocity=np.zeros(5), alpha=np.full(5, .5+t),
                closure_min=np.full(5, -1.), closure_max=np.full(5, 1.5),
                closure_permission=np.ones(5, bool), raw_joint=np.arange(24, dtype=float),
                raw_joint_velocity=np.zeros(24), lift=.05, lift_ref=.1, lift_velocity=0.,
                motor_effort=11.1, bare_hand_weight=10., known_hand_inertia=1.,
                lift_damping=3., support_axis=[0., 0., 1.], phase='HOLDING',
                # Forbidden values deliberately present in raw evaluation input.
                object_mass=123., object_pose=[9., 9., 9.], friction=.01, future_schedule=[999.])


def main():
    checks={}; details={}
    def check(name, condition, detail=None):
        checks[name]=bool(condition)
        if detail is not None: details[name]=detail
        if not checks[name]: raise AssertionError(name)

    m,d=aggregate_patches([patch()])
    check('single_patch_force_exact',np.max(np.abs(np.asarray(d['raw_force'])-d['equivalent_force']))<1e-12)
    check('single_patch_moment_exact',np.max(np.abs(np.asarray(d['raw_moment'])-d['equivalent_moment']))<1e-12)
    check('single_patch_geometry_eligible', d['aggregation_valid'][0] and d['aggregation_error'][0]<1e-12)
    check('single_patch_local_slip',abs(np.linalg.norm(m['slip_local'][0])-.01)<1e-12)

    angle=.73; R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    p=patch(); r=deepcopy(p)
    for key in ('position','force','normal_dir','slip_velocity','torque'):r[key]=(R@np.asarray(p[key])).tolist()
    mr,dr=aggregate_patches([r])
    check('rotated_patch_force_covariant',np.max(np.abs(np.asarray(dr['equivalent_force'])[0]-R@np.asarray(d['equivalent_force'])[0]))<1e-12)
    check('rotated_patch_moment_covariant',np.max(np.abs(np.asarray(dr['equivalent_moment'])[0]-R@np.asarray(d['equivalent_moment'])[0]))<1e-12)
    check('rotated_patch_slip_norm_invariant',abs(np.linalg.norm(mr['slip_local'][0])-np.linalg.norm(m['slip_local'][0]))<1e-12)

    same=[patch(n=1.,position=(.02,0.,0.),force=(1.,.1,0.)),
          patch(n=3.,position=(.02,.01,0.),force=(3.,.3,0.))]
    _,ds=aggregate_patches(same)
    check('compatible_multi_patch_wrench_exact',ds['aggregation_error'][0]<1e-12 and ds['aggregation_valid'][0])
    opposing=[patch(n=2.,force=(2.,0.,0.),direction=(1.,0.,0.)),
              patch(n=2.,force=(-2.,0.,0.),direction=(-1.,0.,0.))]
    mo,do=aggregate_patches(opposing)
    check('opposing_normals_not_silently_accepted',not do['aggregation_valid'][0] and mo['normal'][0]==4.)
    _,dt=aggregate_patches([patch(torque=(.05,0.,0.))])
    check('unrepresented_contact_torque_invalid',not dt['aggregation_valid'][0] and dt['aggregation_error'][0]>.1)
    npatch=patch(n=3.,force=(0.,3.,0.),direction=(0.,1.,0.),position=(0.,.03,0.))
    _,dn=aggregate_patches([patch(),npatch])
    check('nonparallel_multi_contact_error_reported',not dn['aggregation_valid'][0] and dn['aggregation_error'][0]>.1)
    check('raw_patches_preserved',len(dn['patches'])==2 and dn['patch_count'][0]==2)
    for label,diagnostic in [('single',d),('multi',ds),('opposing',do),('torque',dt),('nonparallel',dn)]:
        W=np.asarray(diagnostic['normal_wrench_map']);N=np.asarray(diagnostic['finger_non_normal_by_finger'])
        pressure=np.zeros(5)
        for p in diagnostic['patches']:
            if 0<=p['digit']<5:pressure[p['digit']]+=p['normal']
        reconstruction=W*pressure[None,:]+N.T
        oracle=np.c_[np.asarray(diagnostic['raw_force']),np.asarray(diagnostic['raw_moment'])].T
        check(f'distributed_{label}_wrench_exact',np.max(abs(reconstruction-oracle))<1e-12 and all(diagnostic['distributed_valid']))
    check('single_point_failure_retained_in_new_representation',not dn['aggregation_valid'][0] and dn['distributed_valid'][0])
    check('normal_map_is_not_reunitized',np.linalg.norm(np.asarray(dn['normal_wrench_map'])[:3,0])<.9)
    up_patches=[patch(n=2.,force=(2.,0.,0.),direction=(1.,0.,0.)),
                patch(n=2.,force=(0.,0.,2.),direction=(0.,0.,1.))]
    _,du=aggregate_patches(up_patches,support_axis=[0.,0.,1.])
    check('support_coefficients_preserve_patch_weights',abs(du['normal_support_coeff'][0]-.5)<1e-12 and abs(du['tangent_support_coeff'][0]-.5)<1e-12)
    _,dd=aggregate_patches([patch(n=2.,force=(0.,0.,-2.),direction=(0.,0.,-1.))])
    check('normal_support_coefficient_keeps_negative_sign',dd['normal_support_coeff'][0]==-1. and dd['tangent_support_coeff'][0]==0.)
    cancellation=[patch(n=2.,force=(2.,.4,0.),slip=(0.,.01,0.)),
                  patch(n=2.,force=(2.,-.4,0.),slip=(0.,-.01,0.))]
    mc,dc=aggregate_patches(cancellation)
    check('opposite_patch_shear_magnitudes_do_not_cancel',abs(dc['shear_magnitude'][0]-.8)<1e-12 and np.linalg.norm(mc['tangential'][0])<1e-12)
    check('opposite_patch_slip_magnitudes_do_not_cancel',abs(dc['slip_magnitude'][0]-.01)<1e-12 and np.linalg.norm(mc['slip_local'][0])<1e-12)

    alt=patch(n=1.,force=(1.,0.,0.),target=False)
    ma,da=aggregate_patches([patch(n=2.,force=(2.,0.,0.)),alt])
    check('external_non_target_contact_in_skin_measurement',ma['normal'][0]==3. and da['target_normal_for_evaluation'][0]==2.)
    palm=patch(digit=-1,n=2.);palm['part']='palm'
    other=patch(digit=-1,n=3.);other['part']='wrist'
    _,dp=aggregate_patches([palm,other])
    check('palm_other_separate_from_finger_channels',dp['palm_normal_for_evaluation']==2. and dp['other_normal_for_evaluation']==3. and sum(dp['patch_count'])==0)
    expected_aux=np.r_[np.asarray(palm['force'])+other['force'],
        np.cross(palm['position'],palm['force'])+np.asarray(palm['torque'])+
        np.cross(other['position'],other['force'])+np.asarray(other['torque'])]
    check('auxiliary_force_and_palm_origin_moment_exact',np.allclose(dp['raw_auxiliary_wrench'],expected_aux,atol=1e-12))
    altered=deepcopy([palm,other]);altered[0]['is_target']=False;altered[1]['is_target']=False
    _,changed=aggregate_patches(altered)
    check('auxiliary_includes_all_external_contacts_without_target_filter',dp['raw_auxiliary_wrench']==changed['raw_auxiliary_wrench'])
    mz,dz=aggregate_patches([])
    check('no_contact_zero_force_zero_direction',not np.any(mz['normal']) and not np.any(mz['contact_normal']) and all(dz['aggregation_valid']))

    sensor=HandSensors(calibration())
    f=sensor.sample(raw())
    check('signed_coupled_matrix_preserved',f.force_matrix[2,2]==-1. and f.force_matrix[0,1]==-.4)
    check('positive_scales_are_labeled_not_physical_diagonal',np.all(f.force_gain>0.) and 'not_signed_response' in f.force_gain_role)
    forbidden={'object_mass','object_pose','friction','future_schedule','is_target','patches','target_normal_for_evaluation'}
    check('controller_frame_has_no_object_or_future_oracle',not (forbidden & set(vars(f))))
    check('raw_joint_and_synergy_units_are_separate',f.raw_joint.shape==(24,) and f.joint.shape==(5,))
    check('current_touch_delay_zero',f.age==0. and f.sample_time==0.)
    f_aux=HandSensors(calibration()).sample(raw(patches=[palm,other]))
    check('auxiliary_frame_matches_raw_wrench',f_aux.auxiliary_valid and np.allclose(f_aux.auxiliary_wrench,expected_aux) and f_aux.auxiliary_pressure==5.)
    check('auxiliary_pressure_not_added_to_finger_channels',not np.any(f_aux.normal))

    invalid=HandSensors(calibration(),SensorNoise(geometry_mode='single_point')).sample(raw(patches=opposing))
    check('strict_aggregation_invalidation',not invalid.valid[0] and np.isnan(invalid.normal[0]) and not invalid.aggregation_valid[0])
    distributed=HandSensors(calibration()).sample(raw(patches=opposing))
    check('distributed_mode_uses_measured_patch_presence',distributed.valid[0] and distributed.contact_present[0] and distributed.distributed_valid[0] and not distributed.aggregation_valid[0])
    no_contact=HandSensors(calibration()).sample(raw(patches=[]))
    check('no_contact_is_not_sensor_dropout',no_contact.valid.all() and not np.any(no_contact.contact_normal))

    ns=SensorNoise(force_sigma=.01,tangential_sigma=.02,position_sigma=.0001,
                   slip_sigma=.001,effort_sigma=.05,dropout_probability=.1)
    a=HandSensors(calibration(),ns,seed=71);b=HandSensors(calibration(),ns,seed=71)
    for k in range(20):
        fa=a.sample(raw(k*.005));fb=b.sample(raw(k*.005,patches=[]))
        check(f'fixed_rng_consumption_tick_{k}',json.dumps(a.rng.bit_generator.state,sort_keys=True)==json.dumps(b.rng.bit_generator.state,sort_keys=True))
    ref=np.random.default_rng(71+910_000)
    for _ in range(20):
        ref.normal(0.,ns.force_sigma,5);ref.normal(0.,ns.tangential_sigma,(5,2))
        ref.normal(0.,ns.position_sigma,(5,3));ref.normal(0.,ns.slip_sigma,(5,2))
        ref.normal(0.,ns.effort_sigma)
        ref.normal(0.,ns.force_sigma,3);ref.normal(0.,ns.force_sigma*.05,3)
        ref.normal(0.,ns.force_sigma);ref.random(6)
    check('exactly_48_gaussian_and_6_uniform_draws',json.dumps(a.rng.bit_generator.state,sort_keys=True)==json.dumps(ref.bit_generator.state,sort_keys=True))
    before=json.dumps(a.rng.bit_generator.state,sort_keys=True);count=a.samples
    dup=a.sample(raw(.095));dup.normal[:]=999.
    again=a.sample(raw(.095))
    check('same_tick_reads_do_not_resample_or_mutate_cache',a.samples==count and before==json.dumps(a.rng.bit_generator.state,sort_keys=True) and not np.any(again.normal==999.))

    delayed=HandSensors(calibration(),SensorNoise(delay=.01))
    delayed_values=[]
    for k in range(6):
        p=patch(n=1.+k,force=(1.+k,0.,0.));f=delayed.sample(raw(k*.005,[p]));delayed_values.append(float(f.normal[0]))
    check('touch_is_delayed_by_exact_sensor_ticks',delayed_values==[1.,1.,1.,2.,3.,4.],delayed_values)
    check('encoders_and_command_remain_current',abs(f.joint[0]-.525)<1e-12 and abs(f.alpha[0]-.525)<1e-12 and abs(f.age-.01)<1e-12)
    delayed=HandSensors(calibration(),SensorNoise(delay=.01));aux_values=[]
    for k in range(6):
        ap=patch(digit=-1,n=1.+k,force=(0.,0.,1.+k),direction=(0.,0.,1.));ap['part']='palm'
        f=delayed.sample(raw(k*.005,[ap]));aux_values.append(float(f.auxiliary_wrench[2]))
    check('auxiliary_and_finger_tactile_have_same_acquisition_age',aux_values==[1.,1.,1.,2.,3.,4.] and abs(f.auxiliary_age-f.age)<1e-12)

    drop=HandSensors(calibration(),SensorNoise(dropout_windows=((.005,.015,(0,2)),)))
    q0=drop.sample(raw(0.));q1=drop.sample(raw(.005));q2=drop.sample(raw(.010));q3=drop.sample(raw(.015))
    check('programmed_dropout_uses_simulation_time',q0.valid.all() and not q1.valid[0] and not q1.valid[2] and not q2.valid[0] and q3.valid.all())
    check('dropout_is_missing_not_zero_load',np.isnan(q1.normal[0]) and not q1.valid[0])
    aux_drop=HandSensors(calibration(),SensorNoise(auxiliary_dropout_windows=((.005,.015),)))
    f0=aux_drop.sample(raw(0.,[palm]));f1=aux_drop.sample(raw(.005,[palm]));f2=aux_drop.sample(raw(.010,[palm]));f3=aux_drop.sample(raw(.015,[palm]))
    check('auxiliary_dropout_is_missing_and_separate_from_fingers',f0.auxiliary_valid and not f1.auxiliary_valid and np.isnan(f1.auxiliary_wrench).all() and f1.valid.all() and not f2.auxiliary_valid and f3.auxiliary_valid)
    disabled=HandSensors(calibration(),SensorNoise(instrument_auxiliary=False)).sample(raw(patches=[palm]))
    check('undeclared_auxiliary_sensor_is_not_zero_observation',not disabled.auxiliary_valid and np.isnan(disabled.auxiliary_wrench).all())

    uncal=calibration();uncal.pop('load_error_bound_N')
    fs=HandSensors(uncal).sample(raw())
    check('uncalibrated_motor_load_has_no_confidence',not fs.load_valid and fs.load_confidence==0.)
    sensor=HandSensors(calibration())
    for k in range(200):f=sensor.sample(raw(k*.005))
    check('motor_load_from_effort_minus_bare_hand_only',f.load_valid and abs(f.load_estimate-1.)<1e-6 and abs(f.handload_prior-10.1)<1e-12)

    rejected=False
    try: HandSensors({'force_tau':.01})
    except (KeyError,ValueError): rejected=True
    check('missing_physical_calibration_rejected',rejected)
    rejected=False
    try: sensor.sample(raw(1.1))
    except ValueError: rejected=True
    check('skipped_sensor_tick_rejected',rejected)
    rejected=False
    try: aggregate_patches([patch(n=-1.)])
    except ValueError: rejected=True
    check('negative_contact_pressure_rejected',rejected)

    certificate={'passed':all(checks.values()),'scope':'Synthetic observation contract only; no MuJoCo dynamics or controller performance executed.',
                 'checks':checks,'details':details,'check_count':len(checks),
                 'sensor_fields':[f.name for f in fields(type(f))],
                 'sources_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('contracts.py','sensors.py','verify_sensors.py')}}
    path=ROOT/'review/observation_contract.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(certificate,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':certificate['passed'],'checks':len(checks),'certificate':str(path)},ensure_ascii=False))

if __name__=='__main__': main()

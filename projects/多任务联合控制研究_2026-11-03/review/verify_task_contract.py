"""Pure synthetic public-task interface checks. No physical simulation."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_contract import build_task_request, select_task_mode, validate_task_request


def fixture(**changes):
    args = dict(time_s=9.5, mode='transport', motion_target=[.09, .04, -.025],
                executed_motion_reference=[.1, .0, .0], deadline_s=12.,
                limits=[5., 5., 5., 5., 5.], budget=18.)
    args.update(changes)
    return build_task_request(**args)


class PublicTaskContractChecks(unittest.TestCase):
    def test_vector_axes_signs_units_and_unchanged_goal(self):
        request = fixture()
        self.assertEqual(request['motion_axes'], ['lift_m', 'WRJ2_rad', 'WRJ1_rad'])
        self.assertAlmostEqual(request['motion_error'][0], -.01)
        self.assertEqual(request['motion_error'][1:], [.04, -.025])
        # An unsupported goal must remain visible for feasibility diagnosis.
        self.assertEqual(fixture(motion_target=[.5, .2, -.3])['motion_target'], [.5, .2, -.3])

    def test_deadline_is_absolute_and_never_automatically_extended(self):
        before = fixture(time_s=11.8)
        at = fixture(time_s=12.)
        after = fixture(time_s=12.2)
        self.assertAlmostEqual(before['remaining_time_s'], .2)
        self.assertEqual(at['remaining_time_s'], 0.)
        self.assertFalse(at['deadline_exceeded'])
        self.assertEqual(after['remaining_time_s'], 0.)
        self.assertTrue(after['deadline_exceeded'])
        self.assertEqual(after['deadline_s'], 12.)

    def test_release_deadline_is_an_explicit_current_stage_request(self):
        request = fixture(time_s=12.1, mode='release', deadline_s=15.5)
        self.assertEqual(request['release_at_s'], 12.)
        self.assertEqual(request['release_deadline_s'], 3.5)
        self.assertAlmostEqual(request['remaining_time_s'], 3.4)
        # A manual early release may have its own explicitly announced deadline.
        manual = fixture(time_s=8., mode='release', deadline_s=11.5)
        self.assertEqual(manual['remaining_time_s'], 3.5)
        self.assertEqual(manual['release_at_s'], 12.)

    def test_source_vectors_are_not_aliased(self):
        target = [.09, .04, -.025]
        limits = [5.] * 5
        request = fixture(motion_target=target, limits=limits)
        target[0] = 123.
        limits[0] = 123.
        self.assertEqual(request['motion_target'][0], .09)
        self.assertEqual(request['limits'][0], 5.)

    def test_json_roundtrip_and_validation_do_not_derive_object_state(self):
        request = fixture()
        serialized = json.dumps(request, allow_nan=False)
        self.assertEqual(validate_task_request(json.loads(serialized)), request)
        self.assertFalse(request['object_state_available'])
        self.assertFalse(request['target_contact_identity_available'])
        self.assertEqual(request['observation_scope'],
                         'online_contact_proxy_separate_from_offline_object_truth')
        self.assertEqual(request['model_domain_transfer'], 'experimental_local_model_not_admitted')
        self.assertEqual(request['force_scope'], 'external_all_instrumented_fingers')
        for forbidden in ('pose', 'actual_mass', 'friction', 'future_events', 'object_wrench'):
            self.assertNotIn(forbidden, request)

    def test_no_unregistered_wrist_task_tolerance(self):
        self.assertEqual(fixture()['tracking_tolerances'], [.01, None, None])
        self.assertEqual(fixture(tracking_tolerances=[.005, .01, .02])['tracking_tolerances'], [.005, .01, .02])

    def test_extra_truth_fields_and_false_derived_fields_rejected(self):
        for key, value in (('object_pose', [0.] * 7), ('friction', .6),
                           ('future_events', [9.3]), ('remaining_time_s', 100.),
                           ('force_scope', 'target_object'), ('object_state_available', True),
                           ('target_contact_identity_available', True),
                           ('model_domain_transfer', 'globally_admitted')):
            with self.subTest(key=key):
                request = fixture()
                request[key] = value
                with self.assertRaises(ValueError):
                    validate_task_request(request)

    def test_bad_time_vector_force_limit_and_mode_rejected(self):
        bad = (dict(time_s=float('nan')), dict(time_s=-1.), dict(time_s=True),
               dict(deadline_s=float('inf')), dict(deadline_s=-1.),
               dict(motion_target=[.1, 0.]), dict(motion_target=[.1, float('nan'), 0.]),
               dict(executed_motion_reference=[.1, True, 0.]),
               dict(limits=[5., -1., 5., 5., 5.]), dict(budget=-.1),
               dict(tracking_tolerances=[.01, -.1, None]), dict(mode='scenario_nominal'))
        for change in bad:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    fixture(**change)

    def test_release_preempts_preparation_recovery_and_transport(self):
        for prepared in (False, True):
            for proxy in ('unknown', 'sliding', 'contact_loss'):
                with self.subTest(prepared=prepared, proxy=proxy):
                    self.assertEqual(select_task_mode('release', prepared, [.1, .04, 0.], [0., 0., 0.],
                        online_contact_proxy=proxy), 'release')
        self.assertEqual(select_task_mode('hold', True, [.1, 0., 0.], [.1, 0., 0.],
                         release_active=True, clear_confirmed=True), 'monitor')

    def test_modes_use_current_intent_and_contact_proxy(self):
        stationary = [.1, 0., 0.]
        moving = [.09, .04, -.025]
        self.assertEqual(select_task_mode('hold', False, moving, stationary), 'preparation')
        self.assertEqual(select_task_mode('hold', True, moving, stationary), 'transport')
        self.assertEqual(select_task_mode('hold', True, stationary, stationary), 'hold')
        self.assertEqual(select_task_mode('hold', True, moving, stationary,
                                         online_contact_proxy='sliding'), 'recovery')
        # Geometric clearance must not be inferred before a release lifecycle.
        self.assertEqual(select_task_mode('hold', True, stationary, stationary, clear_confirmed=True), 'hold')
        with self.assertRaises(ValueError):
            select_task_mode('hold', True, moving, stationary, online_contact_proxy='truth_stable')

    def test_preview_extension_is_copied_and_strict(self):
        from task_trajectory import public_transport_trajectory
        preview=public_transport_trajectory().preview(9.5)
        request=fixture(motion_target=preview['motion_targets'][0],motion_preview=preview,
            motion_reference_bounds=dict(lower=[0.,-.5,-.6],upper=[.1,.2,.4]))
        self.assertEqual(request['schema'],'public-hand-task-request-v2')
        self.assertEqual(validate_task_request(json.loads(json.dumps(request))),request)
        preview['motion_targets'][1][0]=123.
        self.assertLess(request['motion_preview']['motion_targets'][1][0],.2)
        poisoned=copy.deepcopy(request);poisoned['motion_preview']['object_pose']=[0.]*7
        with self.assertRaises(ValueError):validate_task_request(poisoned)
        stale=copy.deepcopy(request);stale['motion_preview']['time_s']=9.4
        with self.assertRaises(ValueError):validate_task_request(stale)

    def test_device_spec_does_not_allow_dropout_schedule_or_true_materials(self):
        device=dict(force_sigma_N=.01,tangential_sigma_N=.005,position_sigma_m=.0001,
            slip_sigma_m_s=.0002,effort_sigma=.02,tactile_delay_s=.01)
        request=fixture(sensor_characteristics=device)
        self.assertEqual(validate_task_request(request),request)
        explicit_aux=fixture(sensor_characteristics=dict(device,auxiliary_force_sigma_N=.02,auxiliary_moment_sigma_Nm=.001))
        self.assertEqual(validate_task_request(explicit_aux),explicit_aux)
        for key in ('dropout_windows','true_friction','future_force'):
            with self.assertRaises(ValueError):fixture(sensor_characteristics=dict(device,**{key:0.}))
        with self.assertRaises(ValueError):fixture(motion_reference_bounds=dict(lower=[.2,0.,0.],upper=[.1,0.,0.]))

    def test_derived_numbers_cannot_be_booleans(self):
        import numpy as np
        for change in (dict(time_s=np.bool_(True)),dict(motion_target=[np.bool_(False),0.,0.])):
            with self.assertRaises(ValueError):fixture(**change)
        request=fixture(motion_target=[1.,0.,0.],executed_motion_reference=[0.,0.,0.])
        request['motion_error']=[True,0.,0.]
        with self.assertRaises(ValueError):validate_task_request(request)
        request=fixture(time_s=11.);request['remaining_time_s']=True
        with self.assertRaises(ValueError):validate_task_request(request)


if __name__ == '__main__':
    unittest.main(verbosity=2)

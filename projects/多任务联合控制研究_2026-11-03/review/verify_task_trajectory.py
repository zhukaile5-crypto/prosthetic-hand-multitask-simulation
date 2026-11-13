"""Public-trajectory intent tests, no plant construction or simulation."""
from pathlib import Path
import math,json,sys,unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from task_trajectory import PublicTaskTrajectory,public_transport_trajectory,constant_trajectory,unknown_future_point,validate_preview


class PublicTrajectoryTests(unittest.TestCase):
    def test_matches_declared_user_transport_before_release(self):
        p=public_transport_trajectory()
        for i in range(2400):
            t=i*.005;u=max(0.,t-9.3)
            expected=[.1-.008*(1.-math.cos(3.*u)),.04*math.sin(3.*u),.025*math.sin(2.*u)]
            for x,y in zip(p.position_at(t),expected):self.assertAlmostEqual(x,y,places=14)

    def test_preview_ramp_endpoint_uses_known_user_future(self):
        p=public_transport_trajectory();v=p.preview(10.,3,.02)
        self.assertEqual(len(v['motion_targets']),4)
        self.assertEqual(v['motion_targets'][1],p.position_at(10.02))
        self.assertNotEqual(v['motion_targets'][1],v['motion_targets'][0])

    def test_release_boundary_never_inserts_early_zero_reset(self):
        p=public_transport_trajectory();v=p.preview(11.98,3,.02)
        self.assertEqual(v['motion_targets'][1],p.position_at(12.))
        self.assertEqual(v['motion_targets'][1:], [p.position_at(12.)]*3)
        self.assertGreater(abs(v['motion_targets'][1][1]),.01)

    def test_unknown_future_does_not_fabricate_preview(self):
        v=unknown_future_point(10.,[.09,.01,.02])
        self.assertFalse(v['future_reference_available']);self.assertEqual(len(v['motion_targets']),1)

    def test_json_roundtrip_and_no_hidden_fields(self):
        p=public_transport_trajectory();v=json.loads(json.dumps(p.as_dict()))
        self.assertEqual(PublicTaskTrajectory.from_dict(v).as_dict(),p.as_dict())
        for field in ('scenario','object_pose','mass','friction','future_force'):
            with self.assertRaises(ValueError):PublicTaskTrajectory.from_dict(dict(v,**{field:0.}))

    def test_constant_is_explicit_known_intent(self):
        p=constant_trajectory(.09,(.01,-.02));v=p.preview(8.)
        self.assertTrue(v['future_reference_available'])
        self.assertEqual(v['motion_targets'],[[.09,.01,-.02]]*4)

    def test_invalid_times_shapes_and_boolean_horizon(self):
        for start,end in ((1.,1.),(-1.,12.),(0.,float('nan'))):
            with self.assertRaises(ValueError):PublicTaskTrajectory(start,end,(.1,0.,0.))
        for horizon,period in ((True,.02),(0,.02),(3,0.),(3,float('inf'))):
            with self.assertRaises(ValueError):public_transport_trajectory().preview(8.,horizon,period)

    def test_unknown_future_cannot_be_extended_with_fabricated_waypoints(self):
        v=unknown_future_point(10.,[.09,.01,.02])
        v['requested_times_s'].append(10.02);v['trajectory_times_s'].append(10.)
        v['motion_targets'].append([.08,.03,.04]);v['active_user_motion'].append(True)
        with self.assertRaises(ValueError):validate_preview(v)

    def test_completed_stage_only_keeps_identical_endpoint(self):
        v=public_transport_trajectory().preview(12.1)
        self.assertEqual(validate_preview(v),v)
        v['motion_targets'][1][0]-=.001
        with self.assertRaises(ValueError):validate_preview(v)

    def test_preview_cannot_reset_after_release_inside_hold_plan(self):
        v=public_transport_trajectory().preview(11.98)
        self.assertEqual(validate_preview(v),v)
        v['motion_targets'][-1]=[.1,0.,0.]
        with self.assertRaises(ValueError):validate_preview(v)


if __name__=='__main__':unittest.main(verbosity=2)

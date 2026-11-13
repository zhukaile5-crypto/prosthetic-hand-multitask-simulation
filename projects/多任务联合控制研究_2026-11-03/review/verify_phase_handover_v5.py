"""Release ownership/phase tests on the actual driver; no physical integration.

Contact-row geometry is deliberately empty here: these tests isolate lifecycle
semantics and actuator limits. Separate actual physics probes remain required.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import research_imports
from foundation_config import FoundationConfig
from lifecycle_plant import LifecyclePlant
from lifecycle_release import LifecycleRelease


class PhaseHandover(unittest.TestCase):
    def fixture(self):
        plant=LifecyclePlant(FoundationConfig())
        driver=LifecycleRelease(plant)
        driver.kinematics.contact_rows=lambda frame:(np.zeros((0,len(driver.aids))),[],np.zeros(0))
        return driver

    @staticmethod
    def frame(t,force=0.,valid=True):
        return SimpleNamespace(time=t,dt=.005,normal=np.array([force,0.,0.,0.,0.]),
            valid=np.full(5,valid),auxiliary_valid=valid,auxiliary_pressure=0.)

    def test_existing_full_open_is_not_restarted_as_thumb_tuck(self):
        d=self.fixture();f=self.frame(12.,valid=False)
        d.request(f,d.full_open,.1,np.zeros(2));out=d.step(f)
        self.assertTrue(d.opening_committed)
        np.testing.assert_allclose(out.target,d.full_open,atol=1e-10)
        self.assertIsNone(d.first_clear)
        self.assertFalse(d.was_clear)

    def test_clearance_then_recontact_preserves_opening_phase(self):
        d=self.fixture();d.request(self.frame(0.),d.clearance_pose,.1,np.zeros(2))
        for i in range(43):d.step(self.frame(i*.005))
        self.assertTrue(d.was_clear);self.assertTrue(d.opening_committed)
        first=d.first_clear
        d.step(self.frame(.215,force=.3));out=d.step(self.frame(.220,force=.3))
        self.assertFalse(d.was_clear);self.assertTrue(d.opening_committed)
        np.testing.assert_array_equal(d.target,d.full_open)
        self.assertEqual(d.first_clear,first);self.assertEqual(d.recontacts,1)
        self.assertEqual(out.debug['release_posture_mode'],'committed_full_open')

    def test_invalid_observation_is_not_recontact(self):
        d=self.fixture();d.request(self.frame(0.),d.full_open,.1,np.zeros(2))
        for i in range(43):d.step(self.frame(i*.005))
        before=d.recontacts;d.step(self.frame(.215,valid=False))
        self.assertEqual(d.recontacts,before)
        self.assertTrue(d.opening_committed)

    def test_repeated_request_cannot_reset_deadline_or_posture(self):
        d=self.fixture();d.request(self.frame(.5),d.full_open,.08,np.array([.01,.02]))
        d.request(self.frame(12.),d.clearance_pose,.1,np.zeros(2))
        self.assertEqual(d.started,.5)
        np.testing.assert_array_equal(d.command,d.full_open)
        self.assertEqual(d.lift_reference,.08)

    def test_phase_transitions_preserve_actuator_bounds_and_rates(self):
        d=self.fixture();initial=np.clip(d.clearance_pose,d.lower,d.upper)
        d.request(self.frame(0.),initial,.1,np.zeros(2));previous=initial
        for i in range(150):
            f=self.frame(i*.005,force=.3 if i>100 else 0.)
            out=d.step(f)
            self.assertTrue(np.all(out.target>=d.lower-1e-9))
            self.assertTrue(np.all(out.target<=d.upper+1e-9))
            self.assertTrue(np.all(abs(out.target-previous)<=d.rate*.005+1e-9))
            previous=out.target.copy()


if __name__=='__main__':unittest.main(verbosity=2)

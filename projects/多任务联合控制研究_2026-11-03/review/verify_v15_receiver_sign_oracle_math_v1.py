"""Algebraic force-direction fixture design; no sensor implementation/physics."""
from pathlib import Path
from importlib.metadata import version
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v15_receiver_force_sign_oracle_design_v1.json'

def main():
    Q=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    D=np.diag([-1.,1.,-1.]);p=np.array([.04,-.02,.005]);fc=np.array([2.,0.,0.]);tc=np.zeros(3)
    force=-Q.T@fc;moment=-Q.T@tc+np.cross(p,force)
    swapped_Q=D@Q;swapped_force=swapped_Q.T@(-D@fc);swapped_moment=swapped_Q.T@(-D@tc)+np.cross(p,swapped_force)
    B=np.array([[.6,0.,.8],[0.,1.,0.],[-.8,0.,.6]]);bf=np.array([5.,1.,-.5]);bt=np.array([.2,-.1,.05]);bp=np.array([.03,-.02,.04])
    F=-B.T@bf;T=-B.T@bt+np.cross(bp,F);SF=(D@B).T@(-D@bf);ST=(D@B).T@(-D@bt)+np.cross(bp,SF)
    checks=dict(top_downward=np.array_equal(force,[0.,0.,-2.]),top_moment=np.allclose(moment,[.04,.08,0.],rtol=0,atol=1e-15),reversed_order_same=np.array_equal(force,swapped_force) and np.array_equal(moment,swapped_moment),oblique_force=np.allclose(F,[-3.4,-1.,-3.7],rtol=0,atol=1e-15),oblique_moment=np.allclose(T,[.034,.075,-.288],rtol=0,atol=1e-15),oblique_reversed_same=np.array_equal(F,SF) and np.array_equal(T,ST),proper_frames=abs(np.linalg.det(Q)-1)<1e-15 and abs(np.linalg.det(B)-1)<1e-15 and abs(np.linalg.det(D@B)-1)<1e-15,Newton_common_origin=np.allclose(np.r_[F,T]+np.r_[-F,-T],0,rtol=0,atol=1e-15))
    checks={k:bool(v) for k,v in checks.items()};assert all(checks.values())
    hp=ROOT.parent/'五指策略接入_2026-11-03/hand_plant.py';assert version('mujoco')=='3.15.0'
    report=dict(schema='v15_receiver_force_sign_oracle_design_v1',role='independent_primary_source_and_algebraic_fixture_design',status='oracle_design_ready_module_not_reviewed',installed_mujoco_version='3.15.0',
      sources=[dict(title='MuJoCo3.15 mj_contactForce returns contact-frame force:torque',url='https://mujoco.readthedocs.io/en/3.15.0/APIreference/APIfunctions.html#mj-contactforce'),dict(title='mjContact.frame normal points geom0 togeom1',url='https://github.com/google-deepmind/mujoco/blob/3.15.0/include/mujoco/mjdata.h#L34-L38'),dict(title='Official contact force transformed and applied minus togeom0 plus togeom1',url='https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_smooth.c#L2349-L2357')],
      local_reference=dict(path=str(hp),sha256=hashlib.sha256(hp.read_bytes()).hexdigest(),lines=[278,294],convention='HandPlant hand->external is opposite to incoming external->receiver.'),
      formulas=dict(receiver_geom0='F=-Q.T@f; torque_contact=-Q.T@tau',receiver_geom1='F=+Q.T@f; torque_contact=+Q.T@tau',moment_origin='torque_origin=torque_contact+(contact_point-origin) cross F',receiver_frame='rotate both force and shifted moment with receiver_rotation.T',layout='mj_contactForce is force:torque; cfrc internals swap to torque:force.'),
      fixtures=[dict(name='top_static_receiver_geom0',frame=Q.tolist(),raw_force_N=fc.tolist(),raw_torque_Nm=tc.tolist(),contact_point_from_origin_m=p.tolist(),expected_receiver_force_N=force.tolist(),expected_receiver_moment_Nm=moment.tolist()),dict(name='swapped_geoms_same_physical_wrench',frame=swapped_Q.tolist(),raw_force_N=(-D@fc).tolist(),raw_torque_Nm=(-D@tc).tolist(),receiver_side=1,expected_receiver_force_N=swapped_force.tolist(),expected_receiver_moment_Nm=swapped_moment.tolist(),note='Use D=diag(-1,+1,-1) to keep a proper frame; raw wrench transforms as -D.'),dict(name='oblique_friction_torque',frame=B.tolist(),raw_force_N=bf.tolist(),raw_torque_Nm=bt.tolist(),contact_point_from_origin_m=bp.tolist(),expected_receiver_force_N=F.tolist(),expected_receiver_moment_Nm=T.tolist()),dict(name='explicit_geometry_identity',expected='Only selected receiver topgeom/declared topface; floor and support_post excluded even when all share worldbody0. Geom name alone does not identify its topface.'),dict(name='any_actor_and_unknown',expected='All actors pressing receiver contribute, without targetID filtering; total load is not proof of intended-object support. Complete no-contact interval can be valid0; missing/incomplete/nonfinite input must remain unknown.')],
      static_math_selfchecks=checks,physics_steps=0,models_constructed=0,mujoco_imports=0,production_receiver_code_written=False,
      limitations=['No receiver module has been imported/reviewed and no physical sensor calibration is claimed.','Signed resultant is not unsigned normal compression; abs(Fz) must not hide wrongface/reversed loads.','Newton wrench pairs must share origin/frame; otherwise transport moments first.'],checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),static_math_checks=len(checks),module_reviewed=False,physics_steps=0),ensure_ascii=False))

if __name__=='__main__':main()

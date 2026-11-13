"""Pre-revision mathematical witness; no model, solver or physical execution."""
from pathlib import Path
import hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v15_support.fusion_tasks import build_fusion_tasks

def main():
    source=ROOT/'v15_support/fusion_tasks.py'
    expected='f3902e626db3e22a8095896f8bf11de27c80d2ed05880753556d0b70345123eb'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=expected:raise RuntimeError('Original f390 source required before witness')
    backup=ROOT/'review/revisions/v15_fusion_tasks_f390_before_representation_revision.py'
    if backup.exists() and backup.read_bytes()!=source.read_bytes():raise RuntimeError('Backup differs')
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_bytes(source.read_bytes())
    common=dict(previous_conditional_force5=np.array([2.,2.,2.,0.,0.]),caps5=np.ones(5)*5.,
        force_cost=.4,reference_continuity=3.,balance_weight=8.,support_contact_floor=.2,characteristic_length=.05)
    p=np.array([[0.,0.,0.],[.03,.02,.01],[-.01,.04,-.02]])
    n=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.]])
    a=build_fusion_tasks(patch_digit_p=np.array([0,1,2]),normal_p=np.ones(3),
        reference_point_p3=p,reference_normal_p3=n,selected_p=np.ones(3,bool),**common)
    T=np.array([[.5,0.,0.],[.5,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    b=build_fusion_tasks(patch_digit_p=np.array([0,0,1,2]),normal_p=T@np.ones(3),
        reference_point_p3=p[[0,0,1,2]],reference_normal_p3=n[[0,0,1,2]],selected_p=np.ones(4,bool),**common)
    za=np.ones(3)*2.;zb=T@za
    eco=lambda spec,z:float(np.square(spec.intentP*(z-spec.base_prior_p)).sum())
    Ba=np.asarray(a.metadata['conditional_digit_sum_map']);Bb=np.asarray(b.metadata['conditional_digit_sum_map'])
    Ja=np.arange(75,dtype=float).reshape(3,25)/100.;Jb=Ja[[0,0,1,2]]
    ca=np.asarray(a.metadata['center_public_point_mean']);cb=np.asarray(b.metadata['center_public_point_mean'])
    result=dict(schema='v15_f390_representation_dependency_witness_v1',source_sha256=digest,
        confirmed=a.available and b.available and eco(a,za)!=eco(b,zb) and not np.array_equal(ca,cb),
        original_economy=eco(a,za),split_economy=eco(b,zb),original_center=ca.tolist(),split_center=cb.tolist(),
        conditional_digit_force_unchanged=bool(np.array_equal(Ba@za,Bb@zb)),
        jt_normal_intent_max_difference=float(np.max(abs(Ja.T@za-Jb.T@zb))),
        original_balance=(np.asarray(a.metadata['unweighted_balance_map'])@za).tolist(),
        split_balance=(np.asarray(b.metadata['unweighted_balance_map'])@zb).tolist(),
        split_transform=T.tolist(),old_normal=[1.,1.,1.],new_normal=[.5,.5,1.,1.],old_z=za.tolist(),new_z=zb.tolist(),
        classification='Original code followed its original specification. This witness motivates a representation-invariant objective specification revision, not a historical implementation bug or a performance-driven parameter change.',
        physics_steps=0,model_imports=0,backup=str(backup),author57_preserved=str(ROOT/'review/v15_fusion_tasks_author_v2.json'))
    out=ROOT/'review/v15_fusion_representation_counterexample_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    assert result['confirmed'] and result['conditional_digit_force_unchanged'] and result['jt_normal_intent_max_difference']<1e-14

if __name__=='__main__':main()

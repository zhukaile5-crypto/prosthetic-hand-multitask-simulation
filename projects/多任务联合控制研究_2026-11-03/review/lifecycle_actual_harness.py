"""Actual release/FK off-policy interface tests; mj_step is forbidden."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
from copy import deepcopy
import hashlib,json,sys,ast,runpy
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import research_imports
import mujoco
from foundation_config import FoundationConfig
from hand_plant import HandPlant
from lifecycle_plant import LifecyclePlant
from lifecycle_release import LifecycleRelease
helpers=runpy.run_path(str(ROOT/'review/verify_lifecycle.py'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def forbidden_step(*args,**kwargs):raise AssertionError('Independent lifecycle review must never step physics')
mujoco.mj_step=forbidden_step
OLD=ROOT.parent/'五指策略接入_2026-11-03/output/confirmation_integration_v2/runs/nominal__capsule__ratio__530101'
with np.load(OLD/'trace.npz',allow_pickle=False) as z:
    frame_dict=json.loads(str(z['observation_json'][2400]));old_ctrl=z['ctrl_before'][2400].copy()
    archived={k:z[k].copy() for k in ('time','qpos','qvel','ctrl_before','alpha_before','executed_motion_reference')}
metadata=json.loads((OLD/'metadata.json').read_text(encoding='utf-8'))
names=set(FoundationConfig.__dataclass_fields__)
config=FoundationConfig(**{k:v for k,v in metadata['config'].items() if k in names})


def frame(data):
    d=deepcopy(data)
    for k,v in list(d.items()):
        if isinstance(v,list):d[k]=np.asarray(v)
    return SimpleNamespace(**d)


class Harness:
    def __init__(self):
        self.plant=LifecyclePlant(config)
        self.driver=LifecycleRelease(self.plant)
        self.command=old_ctrl[self.driver.aids].copy();self.holding_calls=0;self.started=False
    def step(self,data,release_requested=False):
        f=frame(data);holding=False
        if not self.started and not release_requested:
            self.holding_calls+=1;holding=True
            return dict(mode='hold',holding_called=True,monitoring_active=False,clear_confirmed=False,
                actual_command=self.command.copy())
        if release_requested and not self.started:
            self.driver.request(f,self.command,.1,np.zeros(2));self.started=True
        command=self.driver.step(f)
        self.command=command.target.copy();debug=command.debug
        return dict(mode='cleared' if command.stage=='post_clear_monitoring' else 'release',
            holding_called=holding,monitoring_active=debug['monitoring_active'],clear_confirmed=debug['first_clear_confirmation'] is not None,
            actuator_before=debug['actuator_before'],actuator_target=debug['actual_actuator_target'],
            actuator_lower=debug['actuator_lower'],actuator_upper=debug['actuator_upper'],
            actuator_max_rate=debug['actuator_rate_limit'],dt=debug['dt'],actual_command=self.command.copy(),
            observed_recontact_count=debug['observed_recontact_count'],stage=command.stage)


def main():
    source_files=('lifecycle_release.py','lifecycle_plant.py','runtime.py','research_imports.py','evaluation.py','protocol.json')
    before={n:sha(ROOT/n) for n in source_files};checks={};details={}
    frames=[]
    for i in range(65):
        d=deepcopy(frame_dict);d['time']=11.995+i*.005
        if i>=2:d['normal']=[0.]*5;d['auxiliary_pressure']=0.
        if i>=58:d['normal']=[.3,0.,0.,0.,0.]
        frames.append(d)
    harness=Harness();contract=helpers['verify_release_preemption'](harness,frames)
    checks['actual_release_interface_contract']=contract['passed']
    checks['holding_spy_once_before_release']=harness.holding_calls==1
    checks['target_is_18_actuators']=len(harness.driver.aids)==18 and harness.command.shape==(18,)
    invariant=helpers['verify_public_identity_invariance'](Harness,frames,lambda r:r['actual_command'])
    checks['forbidden_identity_diagnostics_invariant']=invariant['passed']
    checks['controller_static_information_boundary']=helpers['static_information_scan'](ROOT/'lifecycle_release.py')['passed']
    # Alter unrelated object generalized coordinates in FK-only MjData. Robot
    # rows must depend on robot encoders and public skin, not object truth.
    kin=harness.driver.kinematics;f=frame(frame_dict);a,da=kin.contact_rows(f)
    object_joint=int(np.flatnonzero((kin.model.jnt_bodyid==harness.plant.object)&(kin.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE))[0])
    object_adr=int(kin.model.jnt_qposadr[object_joint])
    kin.data.qpos[object_adr:object_adr+3]=[50.,-40.,30.]
    b,db=kin.contact_rows(f)
    checks['encoder_FK_ignores_object_generalized_coordinates']=np.array_equal(a,b) and da==db
    # Actual controller currently treats unknown observation as !clear; make
    # that behavior visible instead of calling it a physical recontact.
    invalid=Harness()
    for i,d in enumerate(frames[:55]):invalid.step(d,release_requested=i==1)
    before_count=invalid.driver.recontacts
    bad=deepcopy(frames[55]);bad['valid']=[False]*5;bad['auxiliary_valid']=False
    out=invalid.step(bad)
    checks['invalid_after_clear_not_confirmed_recontact']=out['observed_recontact_count']==before_count
    details['invalid_after_clear']=dict(before_count=before_count,after_count=out['observed_recontact_count'],stage=out['stage'])
    # Compare the actual legacy decoder and the disabled subclass on every
    # archived nominal state, with no integration or new policy execution.
    plain=HandPlant(config);disabled=LifecyclePlant(config);plain_out=[];disabled_out=[]
    anchor=2400
    for i,t in enumerate(archived['time']):
        for plant in (plain,disabled):
            plant.data.qpos[:]=archived['qpos'][i];plant.data.qvel[:]=archived['qvel'][i]
            plant.data.ctrl[:]=archived['ctrl_before'][i];plant.data.time=float(t);plant.control_steps=i
            plant.alpha=archived['alpha_before'][i].copy();plant.lift_ref=float(archived['executed_motion_reference'][i,0]);plant.wrist_ref=archived['executed_motion_reference'][i,1:].copy()
            plant.release_started=None if t<12. else 12.
            if t>=12.:
                plant.release_thumb=archived['ctrl_before'][anchor][plant.actuator_groups[0]].copy()
                plant.release_alpha=archived['alpha_before'][anchor].copy();plant.release_lift=.1
            plant._apply_ctrl()
        plain_out.append(plain.data.ctrl.copy());disabled_out.append(disabled.data.ctrl.copy())
    checks['disabled_decoder_all3201_archived_controls_bitwise']=np.array_equal(plain_out,disabled_out)
    checks['disabled_decoder_lifecycle_targets_unset']=disabled.lifecycle_targets is None
    # Runtime source must call no holding.step in the selected release branch.
    runtime=ast.parse((ROOT/'runtime.py').read_text(encoding='utf-8'))
    selected=[]
    for node in ast.walk(runtime):
        if isinstance(node,ast.If) and 'requested_release and self.release_driver is not None'==ast.unparse(node.test):selected.append(node)
    checks['runtime_release_branch_preempts_holding_step']=len(selected)==1 and not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='self.controller.step' for stmt in selected[0].body for n in ast.walk(stmt))
    checks['sources_frozen_during_interface_review']=all(sha(ROOT/n)==h for n,h in before.items())
    details['release_contract']=contract['checks'];details['public_invariance']=invariant
    result=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=before,details=details,
        old_input_trace_sha256=sha(OLD/'trace.npz'),physics_steps_executed=0,
        scope='Real release algorithm and FK/decoder on archived/off-policy encoder and touch inputs only. Does not validate new physical trajectories, collision safety, release success or measured joint-speed limits.',
        remaining=['Pre-handoff manual release still follows preparation branch unless runtime explicitly changes that priority.',
            'Single equivalent point and equal tendon-joint kinematic split are declared approximations; distributed contacts need physical validation.',
            'Actual joint velocity bounds and full saved-trajectory baseline regression require a separately authorized physical run; this test proves disabled decoder equality on archived states.'])
    target=ROOT/('review/lifecycle_actual_interface_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(target),sha256=sha(target)),ensure_ascii=False))


if __name__=='__main__':main()

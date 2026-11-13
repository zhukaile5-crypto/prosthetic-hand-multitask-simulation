"""Software-only acceptance of the isolated holding adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from identification import v21_fragile_reference_runtime_v1 as adapter
from identification import v21_fragile_reference_program_v1 as programme
from v13_a1.codec import Reference21, EnginePort21
from v13_joint.plant import ExecutionReceipt21

CHECKPOINT=ROOT/'review/v21_fragile_reference_program_offline_gate_v1.json'
ORIGIN=adapter.ORIGIN_S
DT=programme.CONTROL_PERIOD_S


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


class SimulatedPort:
    def __init__(self, carrier, events):
        self.time=ORIGIN
        self.values=np.array(carrier)
        self.lower=self.values-1.
        self.upper=self.values+1.
        self.mode_generation=0
        self.release_started=None
        self.joint_release_entry=None
        self.phase='HOLDING'
        self.model=SimpleNamespace(opt=SimpleNamespace(timestep=.0005))
        self.events=events
        self.receipts=[]
        self.change_once=False

    def reference_snapshot(self):
        return Reference21(self.values,self.time)

    def reference_authority21(self):
        return self.lower.copy(),self.upper.copy()

    def commit_reference21(self, ref, *, mode, expected_generation):
        assert ref.time_s==self.time and mode=='holding' and expected_generation==self.mode_generation
        self.events.append('commit')
        old=self.values
        self.values=ref.values.copy()
        if self.change_once:
            self.values[0]=(self.values[0]+old[0])*.5
            self.change_once=False
        applied=Reference21(self.values,self.time)
        engine=self.values.copy()
        engine[18]+=.004
        receipt=ExecutionReceipt21(ref,applied,EnginePort21(engine,self.time,.004),
                                   mode,self.mode_generation,not np.array_equal(ref.values,self.values),False)
        self.receipts.append(receipt)
        return receipt


class SimulatedState:
    def __init__(self,events):
        self._last_time=ORIGIN-DT
        self.time=self._last_time
        self.events=events

    def update(self,obs,dt):
        assert dt==DT and obs.time>self.time
        self.events.append('state')
        self.time=obs.time
        self._last_time=self.time


class SimulatedController:
    def __init__(self,events):
        self.events=events
        self.state=SimulatedState(events)
        self.debug={}
        self.desired=np.ones(5)
        self.plan_start=9.26
        self.last_decision=9.26
        self.plan_ratio_memory={'integral': 2.}
        self.reconciled=[]

    def set_task_request(self, request):
        assert request['mode']=='hold'
        self.events.append('task')

    def step(self,obs,phase,command):
        assert phase=='HOLDING' and command=='hold'
        self.events.append('nominal')
        self.debug=dict(mode='holding',tag='original')
        return SimpleNamespace(reference21=Reference21(obs.robot_reference21,obs.time),
                               mode='holding',mode_generation=obs.mode_generation)

    def reconcile(self,receipt):
        self.events.append('reconcile')
        self.reconciled.append(receipt)
        self.debug['actual_reference21']=receipt.executed.values.tolist()


class SimulatedSession(adapter.HoldingReferenceTrialSession):
    def __init__(self,carrier):
        # Bypass plant/model construction; call the real adapter methods.
        self._holding_program_gate=None
        self.holding_program_records=[]
        self.events=[]
        self.tick=programme.tick(ORIGIN)
        self.plant=SimulatedPort(carrier,self.events)
        self.controller=SimulatedController(self.events)
        self.sensors=SimpleNamespace(_last_time=ORIGIN-DT)
        self.method='ratio_effort'
        self.release_variant='v21'
        self.task_policy='integrated'
        self.v16_release_policy='coordinated'
        self.scenario_name='fragile_conflict'
        self.mode='scheduled'
        self.motion_mode='hold'
        self.task_request={'mode':'hold'}
        self.running=True
        self.paused=False
        self.handoff_attempted=self.handoff_passed=True
        self.manual_release=False

    def _attach_execution_state(self,obs):
        self.events.append('attach')
        obs.robot_reference21=self.plant.values.copy()
        obs.reference_lower21,obs.reference_upper21=self.plant.reference_authority21()
        obs.robot_control_time_s=self.plant.time
        obs.mode_generation=self.plant.mode_generation

    def frame(self,k=0,missing=False):
        self.tick=programme.tick(ORIGIN)+k
        self.plant.time=self.time
        self.events.clear()
        return SimpleNamespace(time=self.time,dt=DT,sample_time=self.time-.010,
                               normal=np.ones(5),valid=[not missing]*5,contact_present=[True]*5)


def checked(carrier):
    result={}
    g=SimulatedSession(carrier)
    _,_,dbg=g._execute_joint(g.frame(),'hold')
    result['unarmed_unchanged_delegation']=g.events==['attach','task','nominal','commit','reconcile'] and dbg['tag']=='original' and not g.holding_program_records

    for role in programme.ARMS:
        g=SimulatedSession(carrier)
        g.arm_holding_program(role)
        for k in range(len(programme.WAVE)):
            before=len(g.plant.receipts)
            desired,pred,dbg=g._execute_joint(g.frame(k),'hold')
            assert g.events==['attach','task','state','commit','reconcile']
            assert len(g.plant.receipts)==before+1 and g.plant.receipts[-1] is g.controller.reconciled[-1]
            assert g.controller.plan_start is None and g.controller.last_decision==-np.inf
            assert g.controller.plan_ratio_memory is None and np.isnan(desired).all() and np.isnan(pred).all()
            assert dbg['nominal_pressure_task_executed'] is False
        assert g._holding_program_gate.status=='complete'
        assert len(g.holding_program_records)==12 and np.array_equal(g.plant.values,carrier)
        json.dumps(g.holding_program_records,allow_nan=False)
        g._execute_joint(g.frame(12),'hold')
        result[role+'_one_commit_each_tick_and_resume']=g.events==['attach','task','nominal','commit','reconcile']

    g=SimulatedSession(carrier)
    g.arm_holding_program('combined')
    g._execute_joint(g.frame(missing=True),'hold')
    result['missing_public_contact_falls_back_once']=g.events==['attach','task','nominal','commit','reconcile'] and g._holding_program_gate.status=='cancelled_public_contacts_or_clock_unavailable'

    g=SimulatedSession(carrier)
    g.arm_holding_program('combined')
    g.plant.lower[0]=g.plant.values[0]
    g._execute_joint(g.frame(),'hold')
    result['new_authority_falls_back_once']=g.events==['attach','task','nominal','commit','reconcile'] and g._holding_program_gate.status=='cancelled_fresh_authority'

    g=SimulatedSession(carrier)
    g.arm_holding_program('combined')
    g.plant.change_once=True
    g._execute_joint(g.frame(),'hold')
    assert len(g.plant.receipts)==1 and g.plant.receipts[-1] is g.controller.reconciled[-1]
    assert g._holding_program_gate.status=='cancelled_receipt_mismatch'
    g._execute_joint(g.frame(1),'hold')
    result['changed_receipt_never_retries_same_tick']=len(g.plant.receipts)==2 and g.events==['attach','task','nominal','commit','reconcile']

    g=SimulatedSession(carrier)
    g.arm_holding_program('combined')
    with patch.object(adapter.FrozenV21Session,'_lifecycle_step',return_value='frozen-release') as original:
        out=g._lifecycle_step(g.frame())
    result['release_takes_original_owner']=out=='frozen-release' and original.call_count==1 and not g.plant.receipts and g._holding_program_gate.status=='cancelled_release_takeover'

    g=SimulatedSession(carrier)
    g.sensors._last_time=ORIGIN
    try:
        g.arm_holding_program('combined')
    except ValueError:
        result['cannot_arm_after_this_tick_observation']=True
    else:
        result['cannot_arm_after_this_tick_observation']=False

    g=SimulatedSession(carrier)
    g.handoff_passed=False
    try:
        g.arm_holding_program('combined')
    except ValueError:
        result['cannot_arm_failed_handoff']=True
    else:
        result['cannot_arm_failed_handoff']=False
    assert all(result.values()),result
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out
    plan=read(ROOT/'review/v21_complete16_plan_v1.json')
    identity=adapter.source_identity(['ratio_effort'])
    assert len(identity)==173
    assert {k:v for k,v in identity.items() if k not in adapter.SOURCE_NAMES}==plan['source_sha256']
    checkpoint=read(CHECKPOINT)
    assert checkpoint['source_sha256'][Path(programme.__file__).name]==sha(Path(programme.__file__))
    cases=[dict(context=r['context'],checks=checked(np.array(r['carrier21']))) for r in checkpoint['rows']]
    result=dict(schema='v21_fragile_reference_runtime_software_gate_v1',passed=True,physics_steps=0,
                plant_instances=0,actual_runtime_receipts=0,source_sha256=identity,
                prior_offline_gate_sha256=sha(CHECKPOINT),test_sha256=sha(Path(__file__)),cases=cases)
    payload=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)
    with out.open('x',encoding='utf-8') as f:
        f.write(payload+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),checks=sum(len(r['checks']) for r in cases)),ensure_ascii=False))


if __name__=='__main__':
    main()

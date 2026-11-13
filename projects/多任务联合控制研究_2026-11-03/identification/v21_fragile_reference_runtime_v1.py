"""Isolated V21 holding-program adapter. No entry point starts physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from identification.v21_fragile_reference_program_v1 import (
    HoldingProgramGate, build_pair,
)
from v13_a1.codec import Reference21
from v21_stable_handoff.runtime import ReleaseSession as FrozenV21Session
from v21_stable_handoff.runtime import source_files as v21_source_files
from v25_response_identification.forks import _boundary as pre_observe_boundary


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'v21_fragile_reference_runtime_v1'
ORIGIN_S = 9.280
ACTIVE = ('ready','running')
SOURCE_NAMES = ('identification/v21_fragile_reference_program_v1.py',
                'identification/v21_fragile_reference_runtime_v1.py',
                'v25_response_identification/forks.py')


def source_files(methods=None):
    files = dict(v21_source_files(methods))
    files.update({name:ROOT/name for name in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in source_files(methods).items()}


class HoldingReferenceTrialSession(FrozenV21Session):
    """Hold the fixed experimental carrier under the original single commit."""

    def __init__(self, *args, **kwargs):
        self._holding_program_gate = None
        self.holding_program_records = []
        super().__init__(*args,**kwargs)
        self.research_sources = source_identity([self.method])

    def arm_holding_program(self, arm):
        if self._holding_program_gate is not None:
            raise ValueError('session already assigned to a holding programme')
        if (self.method!='ratio_effort' or self.release_variant!='v21'
                or self.task_policy!='integrated' or self.v16_release_policy!='coordinated'
                or self.scenario_name!='fragile_conflict' or self.mode!='scheduled'
                or self.motion_mode not in ('hold','transport')):
            raise ValueError('only the registered V21 fragile holding context is supported')
        if (not self.running or self.paused or not self.handoff_attempted or not self.handoff_passed
                or self.manual_release or self.plant.release_started is not None
                or self.plant.joint_release_entry is not None
                or str(self.plant.phase).upper()!='HOLDING'):
            raise ValueError('a successful active hold with no release request is required')
        now = pre_observe_boundary(self)
        if abs(now-ORIGIN_S)>1e-9:
            raise ValueError('expected the fixed 9.280 s PRE_OBSERVE boundary')
        current=self.plant.reference_snapshot()
        lower,upper=self.plant.reference_authority21()
        gate=HoldingProgramGate(build_pair(current,lower,upper,
                                           generation=self.plant.mode_generation),arm)
        self._holding_program_gate=gate
        return gate.program

    def _cancel_holding_program(self, reason):
        gate=self._holding_program_gate
        if gate is not None and gate.status in ACTIVE:
            if gate.pending is not None:
                raise RuntimeError('a holding submission must reconcile before mode change')
            gate.status=reason
            self.holding_program_records.append(dict(event='cancelled',time_s=float(self.time),
                                                      arm=gate.arm,status=reason))

    def _lifecycle_step(self, obs):
        self._cancel_holding_program('cancelled_release_takeover')
        return super()._lifecycle_step(obs)

    def _normal_after_attach(self, obs, command):
        """Exact frozen holding path after attachment has already run once."""
        result=self.controller.step(obs,self.plant.phase,command)
        receipt=self.plant.commit_reference21(result.reference21,mode=result.mode,
                                               expected_generation=result.mode_generation)
        self.controller.reconcile(receipt)
        return self.controller.desired.copy(),np.full(5,np.nan),deepcopy(self.controller.debug)

    @staticmethod
    def _public_contacts_available(obs):
        pressure=np.asarray(obs.normal,dtype=float)
        valid=np.asarray(obs.valid,dtype=bool)
        contact=np.asarray(obs.contact_present,dtype=bool)
        return (pressure.shape==valid.shape==contact.shape==(5,) and np.isfinite(pressure).all()
                and bool(np.all(valid & contact)) and obs.sample_time is not None
                and abs(float(obs.time)-float(obs.sample_time)-.010)<=1e-9
                and abs(float(obs.robot_control_time_s)-float(obs.time))<=1e-9)

    def _execute_joint(self, obs, command):
        gate=self._holding_program_gate
        if gate is None or gate.status not in ACTIVE:
            return super()._execute_joint(obs,command)
        if (command!='hold' or self.manual_release or self.plant.release_started is not None
                or self.plant.joint_release_entry is not None):
            # Real dispatch goes through the existing release lifecycle first.
            self._cancel_holding_program('cancelled_release_takeover')
            if self.plant.joint_release_entry is None:
                raise ValueError('release takeover must begin in the existing lifecycle')
            return self._lifecycle_step(obs)
        if (not self.handoff_passed or str(self.plant.phase).upper()!='HOLDING'
                or self.task_request['mode'] not in ('hold','transport')):
            self._cancel_holding_program('cancelled_holding_not_authorized')
            return super()._execute_joint(obs,command)

        self._attach_execution_state(obs)
        self.controller.set_task_request(self.task_request)
        if not self._public_contacts_available(obs):
            self._cancel_holding_program('cancelled_public_contacts_or_clock_unavailable')
            return self._normal_after_attach(obs,command)
        current=Reference21(obs.robot_reference21,obs.time)
        chosen=gate.select(current,obs.reference_lower21,obs.reference_upper21,
                           mode='holding',generation=obs.mode_generation)
        if chosen is None:
            self.holding_program_records.append(dict(event='cancelled',time_s=float(obs.time),
                                                      arm=gate.arm,status=gate.status))
            return self._normal_after_attach(obs,command)

        # The programme replaces the solver's output, rather than pretending
        # that the solver's force target was actually applied.
        if gate.offset==0:
            self.controller.plan_start=None
            self.controller.last_decision=-np.inf
            self.controller.plan_ratio_memory=None
        now=float(obs.time)
        if self.controller.state._last_time is None or self.controller.state.time<now-1e-9:
            self.controller.state.update(obs,float(obs.dt))
        self.controller.debug=dict(mode='holding',label='experimental_fixed_holding_reference',
                                   nominal_pressure_task_executed=False,pressure_prediction_available=False,
                                   plan_snapshot_scope='fixed programme; no nominal QP or force certificate',
                                   task_request_modified=False)
        receipt=self.plant.commit_reference21(chosen,mode='holding',
                                               expected_generation=gate.program.generation)
        self.controller.reconcile(receipt)
        gate.accept(receipt)
        record=dict(event='receipt',arm=gate.arm,time_s=now,
                    precommit_reference21=current.values.tolist(),
                    requested_reference21=receipt.requested.values.tolist(),
                    executed_reference21=receipt.executed.values.tolist(),
                    executed_engine21=receipt.engine.values.tolist(),lift_bias_m=receipt.engine.lift_bias_m,
                    mode=receipt.mode,generation=receipt.mode_generation,
                    execution_changed=receipt.execution_changed,stale=receipt.stale_plan_replaced,
                    status=gate.status,nominal_proposal_computed=False)
        self.holding_program_records.append(record)
        self.controller.debug['holding_program']=deepcopy(record)
        self.controller.last_debug=deepcopy(self.controller.debug)
        # No new force target was produced by a pressure solver on this tick.
        return np.full(5,np.nan),np.full(5,np.nan),deepcopy(self.controller.debug)

    def save(self,directory):
        meta=super().save(directory)
        gate=self._holding_program_gate
        if gate is not None:
            payload=dict(schema=SCHEMA,arm=gate.arm,status=gate.status,records=self.holding_program_records,
                         origin_s=ORIGIN_S,reference_authority_is_not_pressure_certificate=True)
            path=Path(directory)/'holding_reference_program_v1.json'
            with path.open('x',encoding='utf-8') as f:
                f.write(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        return meta

"""Independent release integration; frozen V15 holding/execution/evaluation.

The cooperative bootstrap inserts one new plant factory into SupportSession's
normal super chain. No module globals or existing classes/instances are patched.
All policies receive the same existing devices and public robot-only adapter.
"""
from pathlib import Path
from copy import deepcopy
import hashlib, gzip, json, time
import numpy as np
from v15_support.runtime import SupportSession, METHODS, MOTION_MODES
from v15_support.runtime import source_files as frozen_source_files
from v14_coord.design import DECISION_PERIOD_S
from session import jsonable
from .bootstrap import ReleaseBootstrap
from .public_adapter import PublicReleaseAdapter
from .transfer_observer import TransferObserver
from .supervisor import ReleaseSupervisor

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCE_NAMES = ('bootstrap.py','plant.py','runtime.py','run_development.py',
    'geometry_tasks.py','transfer_observer.py','hierarchy.py','supervisor.py','public_adapter.py')


def source_files(methods=None):
    files = dict(frozen_source_files(methods))
    files.update({'v16_release_geometry/'+n:ROOT/'v16_release_geometry'/n for n in NEW_SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in source_files(methods).items()}


class ReleaseSession(SupportSession, ReleaseBootstrap):
    def __init__(self, method='ratio_effort', **kwargs):
        policy = kwargs.pop('receiver_policy', 'coordinated')
        if policy not in ('blind','guarded','coordinated'):
            raise ValueError('Unknown independent release policy')
        self.v16_release_policy = policy
        self.release_adapter = None; self.v16_supervisor = None
        super().__init__(method=method, receiver_policy='blind' if policy=='coordinated' else policy, **kwargs)
        self.release_adapter = PublicReleaseAdapter(self.plant.model,
            open18=np.concatenate(self.plant.open_pose), receiver_mount=self.receiver_mount)
        self.research_sources = source_identity([method])

    def observe_public(self):
        result = super().observe_public()
        started = time.perf_counter_ns()
        if self.release_adapter is not None:
            self.release_adapter.observe(result[2])
        extra_ms = (time.perf_counter_ns()-started)/1e6
        return (*result[:-1],result[-1]+extra_ms)

    def _attach_execution_state(self, obs):
        super()._attach_execution_state(obs)
        graph = self.v16_supervisor
        if graph is not None:
            obs.receiver_release_tasks = deepcopy(graph.public_state)

    def _lifecycle_step(self, obs):
        if self.v16_release_policy != 'coordinated':
            return super()._lifecycle_step(obs)
        entry = self.plant.joint_release_entry
        if entry is None:
            raise ValueError('Release request must preempt before coordinated action')
        if self.v16_supervisor is None:
            prior = self.release_adapter.begin_release(entry.started_s)
            observer = TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3, prior=prior)
            self.v16_supervisor = ReleaseSupervisor(entry,open18=np.concatenate(self.plant.open_pose),observer=observer)
            self.release_driver = self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        graph = self.v16_supervisor
        if self.controller.state._last_time is None or self.controller.state.time < float(obs.time)-1e-9:
            self.controller.state.update(obs,float(obs.dt))
        current = self.plant.reference_snapshot().values.copy()
        graph.observe(frame=obs,transfer_inputs=self.release_adapter.transfer_inputs(self.latest_receiver_measurement),current21=current)
        self.plant.sync_supervisor_generation()
        self._attach_execution_state(obs)
        reference, debug = graph.action(frame=obs,adapter=self.release_adapter,current21=current,
            lower21=obs.reference_lower21,upper21=obs.reference_upper21,generation=self.plant.mode_generation)
        receipt = self.plant.commit_reference21(reference,mode='release',expected_generation=self.plant.mode_generation)
        graph.reconcile(receipt)
        debug.update(actual_reference21=receipt.executed.values.tolist(),execution_changed_plan=receipt.execution_changed,
            stale_plan_replaced=receipt.stale_plan_replaced,task_request_modified=False)
        self.controller.motion_reference = receipt.executed.values[18:].copy()
        self.controller.desired[:] = 0.
        self.controller.debug = deepcopy(debug)
        return np.zeros(5),np.full(5,np.nan),debug

    def metadata(self):
        value = super().metadata()
        value.update(schema='v16_public_transfer_geometry_trial_v1',benchmark_id='v16_public_transfer_geometry_development',
            receiver_policy=self.v16_release_policy,
            release_mode='coordinated_public_transfer_and_geometry' if self.v16_release_policy=='coordinated' else self.v16_release_policy,
            research_source_sha256=self.research_sources,
            holding_source='unchanged_v15_controller_or_v14_legacy_as_declared',
            frozen_v15_source_sha256={key:val for key,val in self.research_sources.items() if not key.startswith('v16_release_geometry/')},
            v16_adapter_scope='same public devices and robot-only algebra/FK; no target identity, pose, mass or future state',
            v16_release_scope='experimental virtual geometry proposal, not future servo/contact/placement certification',
            v16_release_original_task_deadlines_unchanged=True)
        value['coordination_contract']['release_policy'] = self.v16_release_policy
        value['controller_scope'] = 'Frozen holding coordination, separate unified transfer/OPEN geometry/monitoring release' if self.v16_release_policy=='coordinated' else 'Frozen V15 policy through independent passive V16 bridge'
        return value

    def save(self, directory):
        meta = super().save(directory)
        path = Path(directory)/'release_decisions.jsonl.gz'
        records = [] if self.v16_supervisor is None else self.v16_supervisor.decision_evidence
        with gzip.open(path,'xt',encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(jsonable(record),ensure_ascii=False,allow_nan=False)+'\n')
        manifest = dict(schema='v16_release_decision_evidence_v1',records=len(records),file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),placement_certified=False,
            holding_joint_decisions_separate=True)
        (Path(directory)/'release_decision_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return meta

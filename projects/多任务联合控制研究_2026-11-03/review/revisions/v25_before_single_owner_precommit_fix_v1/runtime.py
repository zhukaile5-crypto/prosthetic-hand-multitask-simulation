"""Passive V24 observer tap and one precommit probe selector.

No engine writer, sensor producer or physics checkpoint is installed here.
Passive mode keeps the actual frozen V24 canonical_joint supervisor. Probe
mode changes only the reference returned before the parent's single commit.
"""
from pathlib import Path
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
from copy import deepcopy
import gzip, hashlib, json, time
import numpy as np
from v24_canonical_task_fusion.runtime import ReleaseSession as FrozenSession
from v24_canonical_task_fusion.runtime import source_files as frozen_files, METHODS, MOTION_MODES
from v22_joint_task_fusion.numerics import numerical_identity
from v13_a1.codec import Reference21, finite_array
from v14_coord.design import reference_rates
from .design import SOURCE_NAMES, SCHEMA, ProbeSpec, CONTROL_PERIOD_S

ROOT = Path(__file__).resolve().parents[1]


def source_files(methods=None):
    value = dict(frozen_files(methods))
    value.update({'v25_response_identification/'+n: ROOT/'v25_response_identification'/n for n in SOURCE_NAMES})
    return dict(sorted(value.items()))


def source_identity(methods=None):
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source_files(methods).items()}


def plain(value):
    if hasattr(value, 'to_dict'):
        return plain(value.to_dict())
    if hasattr(value, 'to_public'):
        return plain(value.to_public())
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Mapping):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class ProbeCoordinator:
    """Nominal may compute a proposal; only the parent's actor can commit it.

    Its historical nominal certificate is never relabelled a probe certificate.
    The raw fixed target is retained even when a later real domain clips it.
    """
    def __init__(self, nominal, spec):
        if not isinstance(spec, ProbeSpec):
            raise TypeError('Declared ProbeSpec required')
        self.nominal_supervisor = nominal
        self.spec = spec
        self.records = []
        self.carrier21 = None
        self.target21 = None
        self.origin_generation = None
        self.terminal_status = None

    def __getattr__(self, name):
        nominal = self.__dict__.get('nominal_supervisor')
        if nominal is None:
            raise AttributeError(name)
        return getattr(nominal, name)

    def observe(self, **kwargs):
        return self.nominal_supervisor.observe(**kwargs)

    def action(self, **kwargs):
        proposed, debug = self.nominal_supervisor.action(**kwargs)
        now = float(kwargs['frame'].time)
        if now < self.spec.origin_s-1e-9:
            return proposed, debug
        if self.terminal_status is None and self.carrier21 is None:
            if abs(now-self.spec.origin_s) > 1e-9:
                raise ValueError('Probe may only begin at its registered origin')
            nominal_record = self.nominal_supervisor.decision_evidence[-1]
            lo = finite_array(nominal_record['actual_reference_lower21'], (21,), 'actualTASKlower')
            hi = finite_array(nominal_record['actual_reference_upper21'], (21,), 'actualTASKupper')
            self.carrier21 = proposed.values.copy()
            self.target21 = self.carrier21.copy()
            self.origin_generation = int(kwargs['generation'])
            axis = self.spec.axis
            if axis is not None:
                margin = hi[axis]-self.carrier21[axis] if self.spec.sign > 0 else self.carrier21[axis]-lo[axis]
                amplitude = self.spec.amplitude_fraction*min(reference_rates('release')[axis]*CONTROL_PERIOD_S, max(0.,float(margin)))
                self.target21[axis] += self.spec.sign*amplitude
                if amplitude == 0 or self.target21[axis] == self.carrier21[axis]:
                    self.terminal_status = 'UNSUPPORTED_ZERO_ACTUAL_ORIGIN_AUTHORITY'
        if self.terminal_status is None and int(kwargs['generation']) != self.origin_generation:
            self.terminal_status = 'CANCELLED_MODE_GENERATION_CHANGED'
        if self.terminal_status is None and now >= self.spec.origin_s+self.spec.hold_s-1e-9:
            self.terminal_status = 'PROBE_WINDOW_COMPLETED'
        active = self.terminal_status is None
        nominal_record = self.nominal_supervisor.decision_evidence[-1]
        lo = finite_array(nominal_record['actual_reference_lower21'], (21,), 'currentTASKlower')
        hi = finite_array(nominal_record['actual_reference_upper21'], (21,), 'currentTASKupper')
        value = np.clip(self.target21,lo,hi) if active else proposed.values.copy()
        emitted = Reference21(value, now)
        changed = value.tobytes()!=proposed.values.tobytes()
        nominal_certificate = bool(nominal_record.get('geometry_proposal_accepted') is True
            and (nominal_record.get('emitted_hierarchy_score') or {}).get('available') is True)
        row = dict(time_s=now,arm=self.spec.arm,origin_s=self.spec.origin_s,
            active=active,status='PROBE_ACTIVE' if active else self.terminal_status,
            origin_generation=self.origin_generation,current_generation=int(kwargs['generation']),
            carrier_at_origin21=plain(self.carrier21),raw_probe_target21=plain(self.target21),
            nominal_proposal_reference21=proposed.values.tolist(),action_emitted_reference21=value.tolist(),
            actual_TASK_lower21=lo.tolist(),actual_TASK_upper21=hi.tolist(),
            requested_target_clipped_by_fresh_TASK=active and not np.array_equal(value,self.target21),
            nominal_reference_unchanged=not changed,nominal_task_certificate_available=nominal_certificate,
            nominal_task_certificate_applies_to_emitted_reference=nominal_certificate and not changed,
            probe_future_response_prediction_available=False,actual_receipt=None)
        self.records.append(row)
        result = deepcopy(debug)
        result['response_probe'] = {k:v for k,v in row.items() if k not in ('actual_TASK_lower21','actual_TASK_upper21')}
        return emitted, result

    def reconcile(self, receipt):
        self.nominal_supervisor.reconcile(receipt)
        if self.records and abs(self.records[-1]['time_s']-receipt.executed.time_s) <= 1e-9:
            row = self.records[-1]
            row['actual_receipt'] = dict(actual_reference21=receipt.executed.values.tolist(),
                actual_engine21=receipt.engine.values.tolist(),mode=receipt.mode,
                mode_generation=receipt.mode_generation,execution_changed=receipt.execution_changed,
                stale_plan_replaced=receipt.stale_plan_replaced)
            row['nominal_task_certificate_applies_to_actual_reference'] = bool(
                row['nominal_task_certificate_applies_to_emitted_reference']
                and receipt.executed.values.tobytes()==np.asarray(row['action_emitted_reference21'],float).tobytes())


class ResponseSession(FrozenSession):
    def __init__(self,*args,**kwargs):
        role = kwargs.pop('release_variant','canonical_joint')
        if role != 'canonical_joint':
            raise ValueError('This protocol uses frozenV24 canonical_joint only')
        if kwargs.get('receiver_policy','coordinated') != 'coordinated':
            raise ValueError('This response protocol requires the original coordinated release chain')
        self.response_observer = None
        self.probe_coordinator = None
        super().__init__(*args,release_variant='canonical_joint',**kwargs)
        from .observer import PublicMotionObserver
        self.response_observer = PublicMotionObserver(stream_id='v25_public_response',epoch_id=self.run_id)
        self.research_sources = source_identity([self.method])
        self.response_numerical_identity = numerical_identity()
        self.response_log_errors = []

    def observe_public(self):
        result = super().observe_public()
        began = time.perf_counter_ns()
        if self.response_observer is not None:
            self.response_observer.observe(result[2],self.release_adapter)
        return (*result[:-1],result[-1]+(time.perf_counter_ns()-began)/1e6)

    def _precommit(self,obs):
        return dict(time_s=float(obs.time),reference21=self.plant.reference_snapshot().values.copy(),
            engine21=self.plant.engine_snapshot().values.copy(),
            robot_position25=np.asarray(obs.robot_position25).copy(),
            robot_velocity25=np.asarray(obs.robot_velocity25).copy(),
            robot_actuator_axes21=list(obs.robot_actuator_axes21))

    def _receipt(self,snapshot):
        receipt=self.plant.last_execution_receipt21
        if receipt is None or abs(receipt.executed.time_s-snapshot['time_s'])>1e-9:
            raise RuntimeError('No real same-tick receipt after execution')
        self.response_observer.on_receipt(receipt,snapshot)

    def _execute_joint(self,obs,command):
        snapshot=self._precommit(obs)
        result=super()._execute_joint(obs,command)
        self._receipt(snapshot)
        return result

    def _lifecycle_step(self,obs):
        snapshot=self._precommit(obs)
        result=super()._lifecycle_step(obs)
        self._receipt(snapshot)
        return result

    def arm_probe(self,spec):
        if not isinstance(spec,ProbeSpec) or abs(self.time-spec.origin_s)>1e-9:
            raise ValueError('Arm only at the registered current PRE_OBSERVE boundary')
        last=getattr(self.sensors,'_last_time',None)
        if last is not None and abs(float(last)-(self.time-CONTROL_PERIOD_S))>1e-9:
            raise ValueError('Probe must be armed before this tick sensor acquisition')
        if self.v16_supervisor is None or self.plant.joint_release_entry is None or self.probe_coordinator is not None:
            raise ValueError('Exactly one probe requires an existing actual release supervisor')
        self.probe_coordinator=ProbeCoordinator(self.v16_supervisor,spec)
        self.v16_supervisor=self.probe_coordinator
        self.release_driver=self.probe_coordinator
        self.plant.bind_supervisor(self.probe_coordinator)

    def metadata(self):
        value=super().metadata()
        value.update(schema=SCHEMA,research_source_sha256=self.research_sources,
            frozen_v24_source_sha256={k:self.research_sources[k] for k in frozen_files([self.method])},
            numerical_identity=self.response_numerical_identity if hasattr(self,'response_numerical_identity') else self.numerical_sources,
            response_identification=dict(passive_observer_only=self.probe_coordinator is None,
                probe=None if self.probe_coordinator is None else plain(self.probe_coordinator.spec),
                proposal_is_not_executed_input=True,only_actual_receipts_are_input_events=True,
                common_preparation_has_public_snapshots_but_no_fabricated_typed_receipts=True,
                full_response_model_available=False,placement_certified=False))
        return value

    def save(self,directory):
        result=super().save(directory)
        destination=Path(directory);records={}
        streams={'response_observations.jsonl.gz':self.response_observer.packets(),
                 'response_receipts.jsonl.gz':self.response_observer.receipts(),
                 'response_input_intervals.jsonl.gz':self.response_observer.input_intervals(),
                 'probe_decisions.jsonl.gz':() if self.probe_coordinator is None else self.probe_coordinator.records}
        for name,events in streams.items():
            count=0
            with gzip.open(destination/name,'wt',encoding='utf-8') as f:
                for event in events:
                    f.write(json.dumps(plain(event),ensure_ascii=False,allow_nan=False)+'\n');count+=1
            records[name]=dict(records=count,sha256=hashlib.sha256((destination/name).read_bytes()).hexdigest())
        (destination/'response_manifest.json').write_text(json.dumps(dict(schema=SCHEMA,streams=records,
            no_future_or_zero_oracle_exposed_to_policy=True,response_model_admitted=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return result


ReleaseSession = ResponseSession

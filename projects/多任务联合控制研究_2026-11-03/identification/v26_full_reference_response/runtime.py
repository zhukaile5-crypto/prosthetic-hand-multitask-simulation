"""V26 passive V25 session plus independent selector and public origin history."""
from pathlib import Path
import hashlib,json
from v25_response_identification.runtime import ResponseSession as ParentSession,plain
from v25_response_identification.runtime import source_files as parent_files
from .design import ReferenceProbeSpec,SOURCE_NAMES,SCHEMA,CONTROL_PERIOD_S,protocol
from .coordinator import FullReferenceCoordinator

ROOT=Path(__file__).resolve().parents[2]
PREFIX='identification/v26_full_reference_response/'
def source_files(methods=None):
    value=dict(parent_files(methods));value.update({PREFIX+n:ROOT/PREFIX/n for n in SOURCE_NAMES})
    return dict(sorted(value.items()))
def source_identity(methods=None):return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source_files(methods).items()}

class ResponseSession(ParentSession):
    def __init__(self,*args,**kwargs):
        self.first_transfer_read_s=None;self.transfer_window_interrupted=False
        self.public_origin_history=[];self._origin_last_read=None
        super().__init__(*args,**kwargs)
        self.research_sources=source_identity([self.method])

    def step_one_sensor_tick(self):
        result=super().step_one_sensor_tick()
        frame=self.last_obs;graph=self.v16_supervisor
        if frame is not None and graph is not None:
            now=float(frame.time)
            if self._origin_last_read is None or now>self._origin_last_read+1e-9:
                phase=graph.public_state.get('phase')
                if self.first_transfer_read_s is None and phase=='TRANSFER':self.first_transfer_read_s=now
                elif self.first_transfer_read_s is not None and phase!='TRANSFER':self.transfer_window_interrupted=True
                self.public_origin_history.append(dict(read_time_s=now,phase=phase,
                    receiver_fresh_streak=graph.public_state.get('receiver_fresh_streak'),
                    first_transfer_read_s=self.first_transfer_read_s,
                    transfer_window_interrupted=self.transfer_window_interrupted))
                self._origin_last_read=now
        return result

    def arm_probe(self,spec):
        if type(spec) is not ReferenceProbeSpec or abs(self.time-spec.origin_s)>1e-9:raise ValueError('Arm declared current PRE_OBSERVE V26 program only')
        last=getattr(self.sensors,'_last_time',None)
        if last is not None and abs(float(last)-(self.time-CONTROL_PERIOD_S))>1e-9:raise ValueError('Program must precede this tick acquisition')
        if self.v16_supervisor is None or self.plant.joint_release_entry is None or self.probe_coordinator is not None:raise ValueError('One selector requires actual release entry')
        if self.release_driver is not self.v16_supervisor or self.plant.v16_supervisor is not self.v16_supervisor:raise ValueError('Single real lifecycle owner required')
        self.probe_coordinator=FullReferenceCoordinator(self.v16_supervisor,spec)

    def metadata(self):
        value=super().metadata();value.update(schema=SCHEMA,research_source_sha256=self.research_sources,
            frozen_v25_source_sha256={k:self.research_sources[k] for k in parent_files([self.method])},
            full_reference_identification=protocol(),public_origin_history_role='read-clock public phase only; no target truth')
        return value

    def save(self,directory):
        result=super().save(directory);directory=Path(directory)
        manifest=directory/'response_manifest.json';value=json.loads(manifest.read_text(encoding='utf-8'))
        value.update(schema=SCHEMA,full_reference_protocol=protocol())
        manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (directory/'public_origin_history.json').write_text(json.dumps(plain(self.public_origin_history),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return result

ReleaseSession=ResponseSession

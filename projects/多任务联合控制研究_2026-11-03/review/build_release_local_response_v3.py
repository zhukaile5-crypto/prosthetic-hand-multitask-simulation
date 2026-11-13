"""Create only the fresh V3 package; leave frozen V2 and every old run unchanged."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'identification/release_local_response_v2';NEW=ROOT/'identification/release_local_response_v3'
if NEW.exists():raise FileExistsError(NEW)
NEW.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,text):
    with (NEW/name).open('x',encoding='utf-8') as f:f.write(text)
def replace(text,a,b):
    if text.count(a)!=1:raise RuntimeError('Expected exactly one patch anchor: '+a)
    return text.replace(a,b)
decoder=(OLD/'decoder.py').read_text(encoding='utf-8')
decoder=replace(decoder,'PROBE_DELAY_TICKS=4',"ORIGINS={'phi_150ms':30,'phi_350ms':70}")
decoder=replace(decoder,'release_control_tick,release_time_s):','release_control_tick,release_time_s,probe_origin_tick):')
decoder=replace(decoder," if variant not in VARIANTS:raise ValueError('Registered one-sided direction required')"," if variant not in VARIANTS:raise ValueError('Registered one-sided direction required')\n if isinstance(probe_origin_tick,(bool,np.bool_)) or not isinstance(probe_origin_tick,(int,np.integer)) or probe_origin_tick not in ORIGINS.values():raise ValueError('Registered30/70 tick native stage required')")
decoder=replace(decoder,'relative_control_tick-PROBE_DELAY_TICKS','relative_control_tick-probe_origin_tick')
decoder=replace(decoder,'def make_plant_class():','def make_plant_class(probe_origin_tick):\n if isinstance(probe_origin_tick,(bool,np.bool_)) or not isinstance(probe_origin_tick,(int,np.integer)) or probe_origin_tick not in ORIGINS.values():raise ValueError(\'Registered probe origin required\')')
decoder=replace(decoder,'self.probe_variant=variant;self.probe_decoder=None;self.probe_release_tick=None','self.probe_variant=variant;self.probe_origin_tick=int(probe_origin_tick);self.probe_decoder=None;self.probe_release_tick=None')
decoder=replace(decoder,'release_time_s=self.release_started)','release_time_s=self.release_started,probe_origin_tick=self.probe_origin_tick)')
put('decoder.py',decoder)
put('__init__.py','"""Two later release-stage measurements; no response model or control admission."""\n')
provenance=(OLD/'provenance.py').read_text(encoding='utf-8')
provenance=replace(provenance,'"""Frozen230 parent plus all new pilot inputs; no hydration dependency."""','"""Frozen237 V2 closure plus seven fresh V3 inputs; old sources stay unchanged."""')
provenance=replace(provenance,'from identification.release_basis_v2 import source_pinning as parent','from identification.release_local_response_v2 import provenance as parent')
provenance=replace(provenance,'identification/release_basis_v2/dependency_lock.json','identification/release_local_response_v2/dependency_lock.json')
provenance=replace(provenance,"raise RuntimeError('Immutable230 parent changed')","raise RuntimeError('Immutable237 V2 source closure changed')")
provenance=replace(provenance,"f'research/identification/release_local_response_v2/{p.name}'","f'research/identification/release_local_response_v3/{p.name}'")
put('provenance.py',provenance)
summary=json.loads((ROOT/'output/release_local_response_capsule_v2/summary.json').read_text(encoding='utf-8'))
plan=json.loads((OLD/'plan.json').read_text(encoding='utf-8'))
plan.update(schema='release_local_response_two_stages_v3',status='registered_for_root_pure_review_before_physics',planned=14,
    probe_origins=[dict(stage='phi_150ms',relative_control_tick=30,native_progress_s=.150,origin_control_time_s=12.150,full_input_control_time_s=12.170,first_full_input_physical_end_s=12.1705),dict(stage='phi_350ms',relative_control_tick=70,native_progress_s=.350,origin_control_time_s=12.350,full_input_control_time_s=12.370,first_full_input_physical_end_s=12.3705)],
    ramp_contract='Per stage, origin has offset0. Four following5ms ticks produce1/4,1/2,3/4,full20ms slowdown offset. Reference is held over physical500us substeps. First full-input q-end is origin+20.5ms, not the control application at origin+20ms.',
    entry_pairing='Independent deterministic original prep in every branch. Full integration/RNG/queue/observer/FK snapshots at release12.000 and stage-specific12.150/12.350, before sensor/control action; also local decoder/TV/origin state. Within each stage all7 signatures and every physical/public prefix field through origin inclusive must match its executed zero. No hydration/reset-dt clone claim.',
    zero_reproduction='Each new zero must reproduce BOTH retained V2 zero and old native full16s40 actual physical fields bitexact, public observations exact. Two new zeros must match each other. Old7 V2 are references only, never rerun/overwrite/refill.',
    input_denominator='Actual phi and actual lift reference minus matched stage zero at identical control/application/physical clocks; native_carrier_command21 must equal executed zero. Requested fractions are not realized outputs. Command start and q-end are separate.',
    fit_scope='Measurements only. Each stage has3 training and4 strict holdout branches. Assess stage-specific fixed superposition on20/40/100/200ms windows and complete release separately; no fitting/controller/force kernel/safety bound. Later contact modes and current external motion remain unknown online.',
    terminal_semantics='By13.000 only offset ramp establishment and actual commanded lift floor are required, not phi catch-up or final finger opening. Lift floor latest12.620. Phi delay up to-.020 persists; original thumb path ends at native progress2.2, hence full-slow command finalization around14.220 (floating/tick phase recorded). No progress rate>1 or jump is permitted. Full16s retained.',
    frozen_reference=dict(v2_output='output/release_local_response_capsule_v2',summary_sha256=sha(ROOT/'output/release_local_response_capsule_v2/summary.json'),v2_dependency_lock_sha256=sha(OLD/'dependency_lock.json'),v2_zero_trace_sha256=sha(ROOT/'output/release_local_response_capsule_v2/runs/capsule__zero/record/trace.npz'),old_native_trace_sha256=sha(ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz'),old_020_stage_not_rerun=True),
    sources='Frozen237 V2 complete closure plus seven new V3 sources/plan/readme=244 planned files. Root reviews, real-model zero-step preflight and source pin before sole launch. No servo/holding-controller source change.',
    wallclock_reference=dict(measured_v2_seven_jobs_three_workers_wall_s=summary['wall_seconds'],measured_v2_complete_worker_s=[min(x['elapsed_s'] for x in summary['results']),max(x['elapsed_s'] for x in summary['results'])],new14_at_three_workers_waves=5,estimate_status='not_measured_and_not_a_runtime_promise; root chooses schedule and IO/QA overhead remains unknown'))
plan.pop('probe_origin_s',None)
put('plan.json',json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(created=str(NEW),files=['decoder.py','provenance.py','__init__.py','plan.json'],remaining=['run_pilot.py','verify_pure.py','README.md']),ensure_ascii=False))

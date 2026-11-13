"""Matched-reference numerical diagnostic. Never a new closed-loop comparison.

--freeze-plan performs only saved-file checks and writes a fresh prospective plan.
--run executes exactly the six registered probes; use only after root authorization.
No admitted/research source or original record is changed by either mode.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import research_imports
from research_imports import FOUNDATION, source_identity
from session import sha, jsonable

METHODS = ('ratio', 'passive', 'joint')
DTS = (.0005, .00025)
SEED = 610212
SOURCE = ROOT/'output/development_v5_physical_all16/runs'
SELF = Path(__file__).resolve()
PERIOD = .005
HANDOFF = 7.5


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Diagnostic writes must stay inside the research directory')
    return path


def source_records():
    records = {}
    for method in METHODS:
        directory = SOURCE/f'nominal__{method}__{SEED}'
        meta = load_json(directory/'metadata.json')
        result = load_json(directory/'result.json')
        if not result.get('record_complete') or not meta.get('completed'):
            raise ValueError(f'Incomplete source record: {directory}')
        if meta['seed'] != SEED or meta['split'] != 'robustness' or meta['scenario']['name'] != 'nominal':
            raise ValueError('The actual randomized 610212 nominal record is mandatory')
        if sha(directory/'trace.npz') != meta['trace_sha256']:
            raise ValueError('Original source trace identity mismatch')
        with np.load(directory/'trace.npz', allow_pickle=False) as saved:
            times = saved['time']
            if not np.allclose(times, np.arange(3201)*PERIOD, atol=1e-10, rtol=0):
                raise ValueError('Source does not cover the exact 0..16 s public clock')
            if saved['alpha'].shape != (3201, 5) or saved['executed_motion_reference'].shape != (3201, 3):
                raise ValueError('Source executed-reference dimensions are incomplete')
        records[method] = dict(directory=str(directory), config=meta['config'],
            input_sha256={name:sha(directory/name) for name in ('trace.npz','metadata.json','result.json')},
            research_sources=meta['research_source_sha256'],
            foundation_sources=meta['source_sha256'], noise=meta['noise'],
            model_sha256=meta['model_sha256'])
    reference = records[METHODS[0]]
    for r in records.values():
        if r['config'] != reference['config'] or r['noise'] != reference['noise']:
            raise ValueError('Source methods do not share actual physical parameters/noise settings')
    if abs(reference['config']['mass']-.11854323381711594)>1e-14 or abs(reference['config']['friction']-.5704155406240315)>1e-14:
        raise ValueError('Unexpected 610212 randomization; do not substitute nominal mass/friction')
    return records


def prospective_plan():
    records = source_records()
    live = source_identity(list(METHODS))
    # Freeze all executed dependencies, including the admitted foundation, and
    # require current research dependencies to match each source that declared it.
    for method, r in records.items():
        for name, expected in r['research_sources'].items():
            if name in live and live[name] != expected:
                raise ValueError(f'Original executed source changed: {method}/{name}')
    foundation = records[METHODS[0]]['foundation_sources']
    for r in records.values():
        if r['foundation_sources'] != foundation:
            raise ValueError('Different foundation identities in source records')
    for name, expected in foundation.items():
        if sha(FOUNDATION/name) != expected:
            raise ValueError(f'Foundation identity changed: {name}')
    assets={str(path.relative_to(FOUNDATION)).replace('\\','/'):sha(path)
        for path in (FOUNDATION/'assets/shadow/assets').iterdir() if path.is_file()}
    assets['assets/legacy_model.xml']=sha(FOUNDATION/'assets/legacy_model.xml')
    return dict(schema='nominal_610212_dt_matched_reference_plan_v1',
        phase='prospective_development_numerical_diagnostic', seed=SEED, scenario='nominal',
        mode='matched_saved_reference_not_closed_loop_performance',
        matrix=[dict(method=m, dt=dt) for m in METHODS for dt in DTS],
        preparation='Each probe independently starts at reset and runs the unchanged common preparation to 7.5 s; no saved physical state is restored.',
        replay='At each 5 ms tick >=7.5, command that source method\'s saved actual alpha and motion reference. Finite common execution and original release override stay active. Record every mismatch rather than changing or clipping the saved source.',
        original_reference='Saved original 610212 robustness records; controller outputs/predictions are not recomputed. Policy names identify only source input paths.',
        config_variation='Only timestep differs. No mass/friction/solref/contact/clock changes.',
        clocks=dict(sensor_period_s=PERIOD,duration_s=16.,handoff_s=HANDOFF,
            substep_force='At the END of [substep_time,substep_time+duration]. Raw contact packets use physical data.time at that same end.',
            noise='Same sensor seed/config and fixed sample clock; record RNG state digest before and after every sample. This proves same random stream, not identical physical measurements.'),
        retention='All original records remain unchanged. Six full raw records, all-substep complete MuJoCo contacts and hand/external patches, warnings, penetration, and original raw force guards are retained including release.',
        analysis_windows_s=dict(full=[0.,16.],holding=[7.5,12.],static=[8.2,8.9],late=[11.,11.7]),
        spectrum=dict(signal='200Hz actual target finger normal, demeaned separately by window',
            window='Hann',normalization='one-sided density N^2/Hz',bands_Hz=[[.5,5.],[5.,20.],[20.,50.]],
            note='Report both absolute and relative changes; short-window frequency bins limit interpretation. Spectrum never replaces unfiltered force safety.'),
        prospective_flags=dict(role='descriptive_sensitivity_flags_not_new_admission_or_safety_thresholds',
            mean_force_relative_change=.10,raw_peak_relative_change=.15,
            static_or_late_finger_TV_rate_relative_change=.20,
            LF_band_power_relative_change=.25,LF_dominant_frequency_change_bins=1.,
            relative_denominator_floor=1e-12,
            uncertainty='A flag indicates numerical sensitivity to investigate; absence is not a certified material/physical oscillation model. No solref tuning is authorized.'),
        comparison_requirements=['All configs identical except timestep',
            'Exact 3201 sensor ticks and 16s substep duration',
            'Matched alpha/motion/actuator references after handoff; mismatches invalidate the strictly matched-input interpretation',
            'Per-tick RNG stream digests equal',
            'Preparation handoff/state and 7.5s reference-switch increments reported',
            '0.5ms source reproduction errors reported, never assumed exact',
            'Warnings/finite state/penetration and complete phase guard outcomes retained'],
        sources=live, foundation_sources=foundation, asset_sources=assets, records=records,
        diagnostic_script_sha256=sha(SELF), no_confirmatory_superiority_claim=True)


class SavedReferenceController:
    """No feedback policy is run; use public observed state only for diagnostics."""
    def __init__(self, original, arrays):
        self.g = original.g
        self.state = original.state
        self.last_decision = -np.inf
        self.desired = np.zeros(5)
        self.predicted = np.full(5,np.nan)
        self.motion_reference = np.zeros(3)
        self.arrays = arrays
    def set_task_request(self, request):
        self.request = request
    def set_motion_request(self, *args):
        pass
    def step(self, frame, phase, command):
        k = int(round(float(frame.time)/PERIOD))
        if abs(float(self.arrays['time'][k])-float(frame.time))>1e-8:
            raise ValueError('Replay sample clock mismatch')
        self.state.update(frame, PERIOD)
        self.last_decision = float(frame.time)
        self.motion_reference = self.arrays['executed_motion_reference'][k].copy()
        delta = self.arrays['alpha'][k]-frame.alpha
        return SimpleNamespace(delta_closure=delta, predicted_normal=self.predicted.copy(),
            pause_transport=False, debug=dict(saved_reference_replay=True,
            source_time_s=float(self.arrays['time'][k]), prediction_available=False,
            prediction_unavailable_reason='Numerical reference replay does not run or retimestamp a model prediction'))


def spectral_summary(t, signal, lo, hi):
    mask=(t>=lo-1e-9)&(t<hi-1e-9)
    y=np.asarray(signal)[mask]
    if len(y)<4:
        return dict(available=False)
    dt=float(np.median(np.diff(t)))
    w=np.hanning(len(y));center=y-y.mean(axis=0)
    psd=np.abs(np.fft.rfft(center*w[:,None],axis=0))**2/(1./dt*np.sum(w*w))
    if len(y)%2==0:psd[1:-1]*=2.
    else:psd[1:]*=2.
    f=np.fft.rfftfreq(len(y),dt);df=float(f[1]-f[0]);positive=(f>=.5)&(f<=50.)
    power=[(psd[(f>=a)&(f<b)].sum(axis=0)*df).tolist() for a,b in ((.5,5.),(5.,20.),(20.,50.))]
    peak=f[positive][np.argmax(psd[positive],axis=0)].tolist()
    tv=np.abs(np.diff(y,axis=0)).sum(axis=0)
    drift=np.abs(y[-1]-y[0])
    return dict(available=True,samples=len(y),frequency_resolution_Hz=df,
        mean_N=y.mean(axis=0).tolist(),per_finger_TV_N=tv.tolist(),
        finger_TV_rate_N_s=float(tv.sum()/(t[mask][-1]-t[mask][0])),
        per_finger_net_drift_N=drift.tolist(),per_finger_reversal_N=(.5*(tv-drift)).tolist(),
        bands_Hz=[[.5,5.],[5.,20.],[20.,50.]],band_power_N2=power,
        dominant_frequency_Hz=peak,frequency_Hz=f.tolist(),PSD_N2_Hz=psd.tolist())


def max_difference(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return dict(comparable=False,shape_a=list(a.shape),shape_b=list(b.shape))
    return dict(comparable=True,max_abs=float(np.max(abs(a-b))) if a.size else 0.,exact=bool(np.array_equal(a,b)))


def probe(job):
    import mujoco
    from runtime import ResearchSession
    from hand_plant import HandPlant,CommonPreparation
    from skin_geometry import DelayedSkinGeometry
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research, force_band, read_protocol
    target=Path(job['target']);packet_path=Path(job['packet_path'])
    source=Path(job['source']);meta=load_json(source/'metadata.json')
    with np.load(source/'trace.npz',allow_pickle=False) as archive:
        saved={key:archive[key].copy() for key in archive.files if key!='metadata'}
    began=time.perf_counter();packet_count=0;last_packet=-1.
    packet_path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(packet_path,'xt',encoding='utf-8',compresslevel=3) as packet_stream:
        class RecordedPlant(HandPlant):
            def _contacts(self,patches=True):
                nonlocal packet_count,last_packet
                result=super()._contacts(patches)
                # The first contact aggregation immediately AFTER each mj_step.
                # Contact force reads do not call forward/step or change warmstart.
                physical=float(self.data.time)
                if physical>last_packet+1e-10 and physical>1e-12:
                    last_packet=physical
                    full=result if patches else super()._contacts(True)
                    all_contacts=[]
                    for i in range(self.data.ncon):
                        c=self.data.contact[i];force=np.zeros(6)
                        mujoco.mj_contactForce(self.model,self.data,i,force)
                        all_contacts.append(dict(geom1=int(c.geom1),geom2=int(c.geom2),
                            distance_m=float(c.dist),position_world=c.pos.tolist(),
                            frame=c.frame.reshape(3,3).tolist(),force_contact_frame=force.tolist()))
                    packet=dict(substep_end_s=physical,substep_start_s=physical-job['dt'],
                        duration_s=job['dt'],warning_count=int(sum(int(x.number) for x in self.data.warning)),
                        palm_rotation=self.data.xmat[self.palm].reshape(3,3).tolist(),
                        patches=full['patches'],all_contacts=all_contacts)
                    packet_stream.write(json.dumps(jsonable(packet),separators=(',',':'),allow_nan=False)+'\n')
                    packet_count+=1
                return result
        class NumericalSession(ResearchSession):
            def metadata(self):
                value=super().metadata()
                value.update(numerical_probe=dict(mode='matched_saved_reference',
                    source_method=job['method'],plan_sha256=job['plan_sha256'],
                    adaptive_policy_executed=False,source_directory=str(source)),
                    controller_scope='Numerical probe replaying saved executed references; not new closed-loop performance')
                return value
        s=NumericalSession(method=job['method'],scenario='nominal',seed=SEED,split='robustness')
        original_config=asdict(s.config)
        if original_config!=meta['config']:
            raise ValueError('Reconstructed actual randomized config differs from saved source')
        s.config=replace(s.config,timestep=job['dt'])
        s.plant=RecordedPlant(s.config)
        s.obstacle_geom=int(s.plant.model.geom('integration_obstacle_geom').id)
        s.preparation=CommonPreparation(s.plant);s.last_truth=s.plant.truth()
        s.skin_geometry=DelayedSkinGeometry(s.plant)
        s.controller=SavedReferenceController(s.controller,saved)
        rng_before=[];rng_after=[];switch=None
        while s.running:
            if abs(s.time-HANDOFF)<1e-9:
                switch=dict(time_s=s.time,alpha_before=s.plant.alpha.copy(),
                    saved_alpha_after=saved['alpha'][1500],
                    requested_switch_delta=saved['alpha'][1500]-s.plant.alpha,
                    qpos_before=s.plant.data.qpos.copy(),qvel_before=s.plant.data.qvel.copy(),
                    saved_alpha_before=saved['alpha_before'][1500],
                    saved_qpos=saved['qpos'][1500],saved_qvel=saved['qvel'][1500])
            rng_before.append(digest(s.sensors.rng.bit_generator.state))
            s.step_one_sensor_tick()
            rng_after.append(digest(s.sensors.rng.bit_generator.state))
        s.save(target)
    # Packet logger is now closed; all identities below describe complete files.
    result=evaluate_trace(target/'trace.npz')
    result['research_metrics']=evaluate_research(target/'trace.npz')
    probe_meta=dict(schema='nominal_dt_saved_reference_probe_v1',method_label_role='source saved input, not executed adaptive controller',
        mode='matched_reference',seed=SEED,dt=job['dt'],plan_sha256=job['plan_sha256'],
        source_inputs={name:sha(source/name) for name in ('trace.npz','metadata.json','result.json')},
        config_original=original_config,config_executed=asdict(s.config),
        rng_before=rng_before,rng_after=rng_after,reference_switch=switch,
        packet_path=str(packet_path),packet_sha256=sha(packet_path),packet_count=packet_count,
        packet_clock='mj_step end; physical floating data.time; public event clock remains integer 5ms',
        posthandoff_input_match={},source_reproduction={},spectra={},
        wall_seconds=time.perf_counter()-began,closed_loop_performance_claim=False)
    with np.load(target/'trace.npz',allow_pickle=False) as output:
        tr={key:output[key] for key in output.files if key!='metadata'}
    hold=(tr['time']>=HANDOFF-1e-9)&(tr['time']<12.-1e-9)
    post=tr['time']>=HANDOFF-1e-9
    for key in ('alpha','ctrl','executed_motion_reference'):
        probe_meta['posthandoff_input_match'][key]=max_difference(tr[key][post],saved[key][post])
    for key in ('qpos','qvel','normal','skin_normal','alpha','ctrl'):
        probe_meta['source_reproduction'][key]=max_difference(tr[key][hold],saved[key][hold])
    band,band_status=force_band(tr,load_json(target/'metadata.json'),read_protocol())
    probe_meta['force_band_contract']=band_status
    for name,(lo,hi) in (('holding',(7.5,12.)),('static',(8.2,8.9)),('late',(11.,11.7)),('full',(0.,16.))):
        probe_meta['spectra'][name]=spectral_summary(tr['time'],tr['normal'],lo,hi)
        if band is not None:
            probe_meta['spectra'][name]['band50_finger_TV_rate_N_s']=spectral_summary(tr['time'],band,lo,hi)['finger_TV_rate_N_s']
    probe_meta['integrity']=dict(sensor_samples=len(tr['time']),
        complete_public_clock=bool(np.allclose(tr['time'],np.arange(3201)*PERIOD,atol=1e-10,rtol=0)),
        substep_count=len(tr['substep_time']),packet_count_matches_substeps=packet_count==len(tr['substep_time']),
        full_duration_s=float(tr['substep_duration'].sum()),
        warning_count=int(np.max(tr['warning_count'])),
        max_hand_object_penetration_m=float(np.max(tr['substep_penetration'])),
        max_world_object_penetration_m=float(np.max(tr['substep_world_penetration'])),
        finite_raw_state=bool(all(np.isfinite(tr[k]).all() for k in ('qpos','qvel','normal','skin_normal','substep_skin_normal'))))
    write_fresh(target/'result.json',result)
    write_fresh(target/'probe.json',probe_meta)
    return dict(method=job['method'],dt=job['dt'],directory=str(target),
        trace_sha256=sha(target/'trace.npz'),metadata_sha256=sha(target/'metadata.json'),
        result_sha256=sha(target/'result.json'),probe_sha256=sha(target/'probe.json'))


def relative_difference(a,b):
    return (np.asarray(b)-np.asarray(a))/np.maximum(np.abs(np.asarray(a)),1e-12)


def summarize(target, rows, plan):
    pairs={}
    for method in METHODS:
        a=next(r for r in rows if r['method']==method and r['dt']==DTS[0])
        b=next(r for r in rows if r['method']==method and r['dt']==DTS[1])
        x=load_json(Path(a['directory'])/'probe.json');y=load_json(Path(b['directory'])/'probe.json')
        rx=load_json(Path(a['directory'])/'result.json');ry=load_json(Path(b['directory'])/'result.json')
        config_x=x['config_executed'].copy();config_y=y['config_executed'].copy()
        config_x.pop('timestep');config_y.pop('timestep')
        entry=dict(config_identical_except_dt=config_x==config_y,
            rng_stream_identical=x['rng_before']==y['rng_before'] and x['rng_after']==y['rng_after'],
            source_input_match=[x['posthandoff_input_match'],y['posthandoff_input_match']],
            reference_switches=[x['reference_switch'],y['reference_switch']],
            integrity=[x['integrity'],y['integrity']],
            task_success=[rx['task_success'],ry['task_success']],
            original_phase_force_guards=[rx.get('research_metrics',{}).get('safety'),ry.get('research_metrics',{}).get('safety')],
            full_safety=[rx['safety'],ry['safety']],windows={})
        for name in x['spectra']:
            sx=x['spectra'][name];sy=y['spectra'][name]
            if not sx.get('available') or not sy.get('available'):
                entry['windows'][name]=dict(available=False);continue
            tv=float(relative_difference(sx['finger_TV_rate_N_s'],sy['finger_TV_rate_N_s']))
            powers=relative_difference(np.array(sx['band_power_N2'])[:,4],np.array(sy['band_power_N2'])[:,4])
            shift=abs(sy['dominant_frequency_Hz'][4]-sx['dominant_frequency_Hz'][4])/sx['frequency_resolution_Hz']
            entry['windows'][name]=dict(raw200Hz_TV_relative_change=tv,
                band50_TV_relative_change=float(relative_difference(sx['band50_finger_TV_rate_N_s'],sy['band50_finger_TV_rate_N_s'])) if 'band50_finger_TV_rate_N_s' in sx and 'band50_finger_TV_rate_N_s' in sy else None,
                mean_relative_change=relative_difference(sx['mean_N'],sy['mean_N']).tolist(),
                LF_band_power_relative_change=powers.tolist(),LF_dominant_frequency_shift_bins=float(shift),
                descriptive_flags=dict(TV_change_over_20percent=abs(tv)>.20,
                    LF_band_power_change_over_25percent=bool(np.any(abs(powers)>.25)),
                    LF_dominant_frequency_shift_over_one_bin=bool(shift>1.)),
                interpretation='Sensitivity flags only. No classification as real material oscillation or safety certification.')
        pairs[method]=entry
    return dict(schema='nominal_610212_dt_matched_summary_v1',execution_complete=len(rows)==6,
        plan_sha256=sha(target/'plan.json'),rows=rows,pairs=pairs,
        interpretation='Matched saved-input numerical probes, not closed-loop performance. A source-reproduction or reference mismatch must be explained before interpreting numerical sensitivity.',
        sources_unchanged=all(sha(ROOT/n)==v for n,v in plan['sources'].items()),
        foundation_sources_unchanged=all(sha(FOUNDATION/n)==v for n,v in plan['foundation_sources'].items()),
        asset_sources_unchanged=all(sha(FOUNDATION/n)==v for n,v in plan['asset_sources'].items()),
        original_inputs_unchanged=all(sha(Path(r['directory'])/n)==v for r in plan['records'].values() for n,v in r['input_sha256'].items()))


def self_test():
    t=np.arange(0.,2.,.005)
    y=np.column_stack([np.sin(2*np.pi*7*t)]*5)
    summary=spectral_summary(t,y,0.,2.)
    assert np.allclose(summary['dominant_frequency_Hz'],[7.]*5,atol=1e-10,rtol=0)
    assert abs(sum(row[0] for row in summary['band_power_N2'])-.5)<.01
    assert max_difference(np.zeros((2,5)),np.zeros((2,5)))['exact']
    assert max_difference(np.zeros((2,5)),np.zeros((3,5)))['comparable'] is False
    assert abs(float(relative_difference(2.,2.4))-.2)<1e-12
    print('5 pure numerical/interface checks passed; no plant or physics constructed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-plan',help='Fresh plan filename; validates existing saved files only')
    group.add_argument('--run',help='Previously frozen plan file; root-authorized physics execution')
    group.add_argument('--self-test',action='store_true')
    p.add_argument('--out',help='Fresh research output directory, required with --run')
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.freeze_plan:
        plan=prospective_plan();path=checked_path(a.freeze_plan);write_fresh(path,plan)
        print(json.dumps(dict(plan=str(path),sha256=sha(path),probes=6,physics_started=False)));return
    if not a.out:p.error('--run requires --out')
    plan_path=checked_path(a.run);plan=load_json(plan_path)
    if plan!=prospective_plan():raise ValueError('Frozen plan/script/input/source identity no longer matches')
    target=checked_path(a.out)
    if target.exists():raise ValueError('Fresh output is required; previous failures cannot be overwritten')
    target.mkdir(parents=True);shutil.copy2(plan_path,target/'plan.json')
    for base,names,label in ((ROOT,plan['sources'],'research'),(FOUNDATION,plan['foundation_sources'],'foundation')):
        for name in names:
            destination=target/'sources'/label/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(base/name,destination)
    for name in plan['asset_sources']:
        destination=target/'sources/foundation'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(FOUNDATION/name,destination)
    shutil.copy2(SELF,target/'sources'/SELF.name)
    jobs=[dict(method=m,dt=dt,source=plan['records'][m]['directory'],
        target=str(target/'runs'/f'nominal__{m}__610212__dt{dt:.5f}'),
        packet_path=str(target/'packets'/f'{m}__dt{dt:.5f}.jsonl.gz'),
        plan_sha256=sha(target/'plan.json')) for m in METHODS for dt in DTS]
    write_fresh(target/'execution_started.json',dict(jobs=jobs,source_identity=plan['sources'],complete=False))
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    rows=[];errors=[]
    with ProcessPoolExecutor(max_workers=max(1,min(6,a.workers))) as pool:
        futures={pool.submit(probe,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                row=future.result();rows.append(row)
                print(json.dumps(dict(completed=len(rows),method=row['method'],dt=row['dt'])),flush=True)
            except Exception as exc:
                errors.append(dict(job=job,error=repr(exc)))
                print(json.dumps(errors[-1]),flush=True)
    if errors:
        write_fresh(target/'execution_failed.json',dict(completed=rows,errors=errors,execution_complete=False))
        raise RuntimeError('Diagnostic failures retained; no rerun or selection performed')
    summary=summarize(target,rows,plan);write_fresh(target/'summary.json',summary)
    if not all(summary[k] for k in ('sources_unchanged','foundation_sources_unchanged','asset_sources_unchanged','original_inputs_unchanged')):
        raise RuntimeError('Source or original input changed; diagnostic not valid')


if __name__=='__main__':main()

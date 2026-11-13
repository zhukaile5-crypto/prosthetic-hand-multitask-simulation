"""Development-only tactile friction-cone probe; no core files are modified.

The sliding shear/normal ratio is a friction proxy, not an identified physical
coefficient. Only sustained fast slip with independent tactile force evidence
activates an experimental lower-normal-force request. Aggregate contact forces,
sensor delay, load transfer and partially sticking contacts can bias the proxy.
This is an engineering experiment related to existing adaptive cone control,
not a claim of methodological novelty or a formal-evaluation result.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace
import traceback

import numpy as np

import run_experiments as runner
from controllers import CoordinatedRatio, Gains, box_budget_projection
from physics import DT, ROOT


SEEDS = (1101,1102,1103)
CASES = ('contact_robustness','friction_drop','three_pulses','local_limit','sensor_noise')
VARIANTS = {
    'cone_fast': dict(safety_margin=.85,confirm_seconds=.030,fast_slip_threshold=.015,
                      ratio_drop_fraction=.20,load_increase_fraction=.25),
    'cone_balanced': dict(safety_margin=.80,confirm_seconds=.050,fast_slip_threshold=.020,
                          ratio_drop_fraction=.20,load_increase_fraction=.30),
    'cone_cautious': dict(safety_margin=.90,confirm_seconds=.080,fast_slip_threshold=.025,
                          ratio_drop_fraction=.25,load_increase_fraction=.35),
}
METHODS = ('ratio','coordinated','cone_fast','cone_balanced','cone_cautious')
CORE_FILES = ('controllers.py','physics.py','scenarios.py','run_experiments.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_hashes():
    return {name:sha256(ROOT/name) for name in CORE_FILES}


class TactileConeProbe(CoordinatedRatio):
    """SensorFrame-only cone request with inherited pressure and release guards."""
    def __init__(self,name,limits,budget,gains,configuration):
        super().__init__(name,limits,budget,gains)
        self.probe_config = dict(configuration)
        self.stable_tau = None
        self.stable_shear_load = None
        self.fast_slip_seconds = 0.
        self.sliding_tau = deque(maxlen=8)
        self.mu_proxy = None
        self.active_until = -1.
        self.activations = []
        self.maximum_extra_request = 0.
        self.maximum_mode_force = 0.

    def acquire(self,frame,phase,valid):
        # An acquisition slip is expected contact establishment, not a friction
        # estimate. The inherited acquisition pressure guard remains active.
        self.fast_slip_seconds = 0.
        self.sliding_tau.clear()
        self.mu_proxy = None
        self.active_until = -1.
        return super().acquire(frame,phase,valid)

    def hold(self,frame,valid):
        previous_output = self.desired.copy()
        baseline_output = super().hold(frame,valid)
        fresh = valid & (frame.age <= .10)
        supported = fresh & (self.normal > .25) & (frame.normal > .15)
        supported_count = int(np.count_nonzero(supported))
        load = float(np.sum(self.shear[supported]))
        normal_sum = float(np.sum(self.normal[supported]))
        tau = load/max(.25,normal_sum)
        enough = supported_count >= 3
        fast = enough and self.slip_filtered > self.probe_config['fast_slip_threshold']
        # Let the inherited tactile low-pass settle before defining a load
        # reference; otherwise its startup ramp looks like a new shear load.
        if enough and not fast and self.control_ticks >= 20:
            if self.stable_tau is None:
                self.stable_tau,self.stable_shear_load = tau,load
            else:
                self.stable_tau = .995*self.stable_tau+.005*tau
                self.stable_shear_load = .995*self.stable_shear_load+.005*load
        ratio_evidence = (self.stable_tau is not None and
                          tau < self.stable_tau*(1.-self.probe_config['ratio_drop_fraction']))
        load_evidence = (self.stable_shear_load is not None and
                         load > self.stable_shear_load*(1.+self.probe_config['load_increase_fraction']) and
                         load-self.stable_shear_load > .30)
        force_evidence = ratio_evidence or load_evidence
        if fast and force_evidence:
            self.fast_slip_seconds += DT
            if .025 <= tau <= 1.:
                self.sliding_tau.append(tau)
        else:
            self.fast_slip_seconds = 0.
            self.sliding_tau.clear()
        trigger = (fast and force_evidence and len(self.sliding_tau) >= 3 and
                   self.fast_slip_seconds >= self.probe_config['confirm_seconds']-1e-10)
        was_active = self.mu_proxy is not None and frame.time < self.active_until
        if trigger and not was_active:
            # Fix the estimate for this episode. Continually lowering it when
            # the commanded normal force rises would create positive feedback.
            self.mu_proxy = float(np.clip(.95*np.quantile(self.sliding_tau,.25),.05,1.))
            self.active_until = frame.time+.40
            if len(self.activations) < 32:
                self.activations.append(dict(time=float(frame.time),mu_proxy=self.mu_proxy,
                                             tau=tau,stable_tau=self.stable_tau,
                                             shear_load_N=load,stable_shear_load_N=self.stable_shear_load,
                                             ratio_evidence=bool(ratio_evidence),
                                             load_evidence=bool(load_evidence)))
        if was_active and fast:
            self.active_until = frame.time+.40
        active = self.mu_proxy is not None and frame.time < self.active_until and enough
        normal_lower = baseline_output.copy()
        result = baseline_output.copy()
        if active:
            denominator = self.mu_proxy*self.probe_config['safety_margin']
            normal_lower[supported] = self.shear[supported]/max(.01,denominator)
            proposal = np.maximum(baseline_output,normal_lower)
            upper = np.minimum(self.limits-.18,self.guard_caps)
            upper[~supported] = np.minimum(upper[~supported],baseline_output[~supported])
            upper[~fresh] = np.minimum(upper[~fresh],previous_output[~fresh])
            budget = min(self.budget,self.total_cap)
            # One fixed response rate for all three settings. Any pressure cap
            # can unload immediately; this rate only limits positive requests.
            proposal = np.minimum(proposal,previous_output+40.*DT)
            result = box_budget_projection(proposal,upper,budget)
            self.label = 'research_cone_request'
            self.maximum_extra_request = max(self.maximum_extra_request,
                                              float(np.sum(np.maximum(0.,result-baseline_output))))
            self.maximum_mode_force = max(self.maximum_mode_force,float(np.sum(result)))
        if frame.time >= self.active_until:
            self.mu_proxy = None
        self.debug.update(probe_active=bool(active),mu_proxy=self.mu_proxy,
                          aggregate_tau=tau,stable_tau=self.stable_tau,
                          stable_shear_load_N=self.stable_shear_load,
                          supported_contacts=supported_count,
                          force_evidence=bool(force_evidence),normal_lower_N=normal_lower.tolist(),
                          probe_activations=self.activations,
                          probe_max_extra_request_N=self.maximum_extra_request,
                          probe_max_total_request_N=self.maximum_mode_force)
        return result


def synthetic_checks(gains):
    def frame(t,normal=None,shear=None,slip=0.,valid=None):
        return SimpleNamespace(time=t,normal=np.full(5,2.) if normal is None else np.asarray(normal,dtype=float),
                               shear=np.full(5,.8) if shear is None else np.asarray(shear,dtype=float),
                               slip=slip,valid=np.ones(5,dtype=bool) if valid is None else np.asarray(valid,dtype=bool),
                               age=.01,collision=0.)
    evidence = []
    for name,configuration in VARIANTS.items():
        controller = TactileConeProbe(name,[5.]*5,22.,gains,configuration)
        for k in range(250):
            controller.step(frame(8.+k*DT),'HOLDING','hold')
        # The false-slip signal has no changed force evidence. It must never
        # activate the cone, even though it exceeds the fast-slip threshold.
        for k in range(40):
            controller.step(frame(9.25+k*DT,slip=.045),'HOLDING','hold')
        if controller.activations or controller.maximum_extra_request > 0.:
            raise AssertionError(f'{name}: unsupported false slip activated the cone')
        for k in range(100):
            result = controller.step(frame(9.45+k*DT,shear=[.25]*5,slip=.045),'HOLDING','hold')
            if not (np.all(np.isfinite(result)) and np.all(result >= 0.) and
                    np.all(result <= 5.+1e-6) and np.sum(result) <= 22.+1e-6):
                raise AssertionError(f'{name}: prototype request is infeasible')
        if not controller.activations or controller.maximum_extra_request <= 0.:
            raise AssertionError(f'{name}: supported sustained fast-slip evidence failed to activate')
        previous = controller.desired.copy()
        result = controller.step(frame(9.95,normal=[np.nan,2,2,2,2],
                                       shear=[np.nan,.25,.25,.25,.25],slip=.045,
                                       valid=[False,True,True,True,True]),'HOLDING','hold')
        if result[0] > previous[0]+1e-7:
            raise AssertionError(f'{name}: invalid channel obtained extra force')
        if np.any(controller.step(frame(9.955),'RELEASING','release')):
            raise AssertionError(f'{name}: prototype blocked release')
        evidence.append(dict(method=name,false_slip_without_force_change_rejected=True,
                             sustained_slip_with_force_change_activated=True,
                             request_bounds=True,invalid_channel_no_increase=True,release_zero=True))
    return evidence


def execute_job(job):
    if core_hashes() != job['expected_core_hashes']:
        raise RuntimeError('Core files changed after the development-only probe was frozen')
    original_factory = runner.make_controller
    try:
        if job['method'] in VARIANTS:
            configuration = job['prototype_configuration']
            runner.make_controller = lambda name,limits,budget,gains: TactileConeProbe(
                name,limits,budget,gains,configuration)
        result = runner.run_episode(job)
        result['research_only'] = True
        if job['method'] in VARIANTS:
            result['prototype_configuration'] = dict(job['prototype_configuration'])
        return result
    finally:
        runner.make_controller = original_factory


def summarize(results):
    summary = {}
    for method in METHODS:
        rows = [row for row in results if row['method']==method]
        measured = [row for row in rows if not row.get('execution_error')]
        summary[method] = dict(episodes=len(rows),safe_completed=sum(row.get('safe_completed',False) for row in rows),
                              completed=sum(row.get('completed',False) for row in rows),
                              dropped=sum(row.get('dropped',False) for row in rows),
                              setup_failed=sum(not row.get('setup_ok',False) for row in rows),
                              execution_errors=len(rows)-len(measured),
                              mean_total_force_excess_Ns=float(np.mean([row['total_force_excess_Ns'] for row in measured])) if measured else None,
                              mean_force_N=float(np.mean([row['mean_force_N'] for row in measured])) if measured else None,
                              maximum_physics_force_N=max((row['physics_peak_force_N'] for row in measured),default=None),
                              maximum_physics_penetration_mm=max((row['physics_peak_penetration_mm'] for row in measured),default=None),
                              activations=sum(len(row['final_debug'].get('probe_activations',[])) for row in measured))
    per_scenario = {}
    for scenario in CASES:
        per_scenario[scenario] = {method:dict(
            safe_completed=sum(row.get('safe_completed',False) for row in results if row['scenario']==scenario and row['method']==method),
            dropped=sum(row.get('dropped',False) for row in results if row['scenario']==scenario and row['method']==method))
            for method in METHODS}
    paired = {}
    keyed = {(row['scenario'],row['seed'],row['method']):row for row in results}
    for method in VARIANTS:
        paired[method] = {}
        for reference in ('ratio','coordinated'):
            changes = []
            for scenario in CASES:
                for seed in SEEDS:
                    candidate,baseline = keyed.get((scenario,seed,method)),keyed.get((scenario,seed,reference))
                    if candidate is None or baseline is None:
                        continue
                    if candidate.get('safe_completed',False) != baseline.get('safe_completed',False):
                        changes.append(dict(scenario=scenario,seed=seed,
                                            candidate_safe=candidate.get('safe_completed',False),
                                            reference_safe=baseline.get('safe_completed',False),
                                            candidate_dropped=candidate.get('dropped',False),
                                            reference_dropped=baseline.get('dropped',False)))
            paired[method][reference] = changes
    return dict(scope='Development-only exploratory probe; not formal evaluation',
                methods=summary,scenarios=per_scenario,paired_safe_completion_changes=paired,
                limitations=['Only three development seeds and five scenarios were tested.',
                             'The aggregate sliding force ratio is an imperfect friction proxy.',
                             'No novel-method, physical-damage, or held-out advantage claim is supported.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2,choices=(1,2))
    parser.add_argument('--out',type=Path,default=ROOT/'output'/'friction_probe')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    # Seeds and scenarios are fixed here, rather than permitting accidental
    # use of the fresh test/robustness directories or evaluation seed ranges.
    selected_path = ROOT/'selected_gains.json'
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    immutable = dict(seeds=list(SEEDS),scenarios=list(CASES),methods=list(METHODS),
                     prototype_configurations=VARIANTS,core_sha256=core_hashes(),
                     prototype_source_sha256=sha256(__file__),
                     selected_gains_sha256=sha256(selected_path),
                     selected_gains=selected,physics_timestep=.0005,workers=args.workers,
                     trials_expected=len(SEEDS)*len(CASES)*len(METHODS))
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError('Preserve existing probe results; use --resume for an identical frozen probe')
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['immutable'] != immutable:
            raise RuntimeError('Cannot resume a different prototype/core/gains configuration')
    else:
        out.mkdir(parents=True,exist_ok=True)
        manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                        research_only=True,immutable=immutable,
                        runtime=dict(python=platform.python_version(),numpy=np.__version__),
                        synthetic_checks=synthetic_checks(Gains(**selected['coordinated'])),
                        statement='No fresh held-out results are read. The prototype uses SensorFrame only, without true mass/friction/pose/future schedule.')
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        snapshot = out/'snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in (*CORE_FILES,'selected_gains.json',Path(__file__).name):
            shutil.copy2(ROOT/name,snapshot/name)
    results_path = out/'trials.json'
    results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
    completed_keys = {(row['scenario'],row['seed'],row['method']) for row in results}
    jobs = []
    for method in METHODS:
        gains = selected['ratio'] if method=='ratio' else selected['coordinated']
        for scenario in CASES:
            for seed in SEEDS:
                if (scenario,seed,method) in completed_keys:
                    continue
                job = dict(scenario=scenario,seed=seed,method=method,gains=gains,
                           timestep=.0005,save_trace=True,output=str(out),
                           expected_core_hashes=immutable['core_sha256'])
                if method in VARIANTS:
                    job['prototype_configuration'] = VARIANTS[method]
                jobs.append(job)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(execute_job,job):job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(scenario=job['scenario'],seed=job['seed'],method=job['method'],
                              research_only=True,execution_error=str(error),traceback=traceback.format_exc(),
                              completed=False,safe_completed=False,setup_ok=False,dropped=False)
            results.append(result)
            results.sort(key=lambda row:(row['method'],row['scenario'],row['seed']))
            runner.write_results(out,results)
            print(json.dumps(dict(completed=len(results),expected=immutable['trials_expected'],
                                  scenario=result['scenario'],method=result['method'],seed=result['seed'],
                                  safe_completed=result.get('safe_completed'),
                                  dropped=result.get('dropped'),error=result.get('execution_error')),
                             ensure_ascii=False),flush=True)
    if core_hashes() != immutable['core_sha256']:
        raise RuntimeError('Core files changed while the probe was running')
    final = summarize(results)
    final['all_core_files_preserved'] = True
    final['all_expected_trials_present'] = len(results)==immutable['trials_expected']
    (out/'summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=str((out/'summary.json').resolve()),methods=final['methods']),
                     ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

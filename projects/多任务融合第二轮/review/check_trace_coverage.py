"""Read-only coverage audit of every formal solver sample and sync endpoint.

No physics or controller is imported or rerun. Existing NPZ files are read;
only the separate review report is written. Positive-duration solver samples
and zero-duration endpoints have separate strictly increasing clocks. At each
control boundary the endpoint and the following solver sample share a time,
and both force observations are retained rather than deduplicated.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CLOCK_ATOL_S = 1e-8
DURATION_ATOL_S = 1e-12
EXPECTED_SUBSTEP_COLUMNS = ('solver_time','max_force_N','penetration_m',
                            'force_excess_N','budget_excess_N','duration_s')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_arrays(substeps,columns,control_times,control_dt,physics_dt,duration,
                   qpos_times=None,qpos_rows=None):
    report = dict(passed=True,checks={},errors=[],clock_tolerance_s=CLOCK_ATOL_S,
                  duration_tolerance_s=DURATION_ATOL_S)
    def check(name,condition):
        value = bool(condition)
        report['checks'][name] = value
        if not value:
            report['passed'] = False
            report['errors'].append(name)
        return value
    ratio = control_dt/physics_dt
    n_control = round(duration/control_dt)
    n_solver_per_control = round(ratio)
    n_solver = n_control*n_solver_per_control
    n_endpoint = n_control
    width = n_solver_per_control+1
    expected_rows = n_control*width
    report.update(expected_control_rows=n_control,expected_solver_rows=n_solver,
                  expected_endpoint_rows=n_endpoint,expected_substep_rows=expected_rows,
                  observed_control_rows=len(control_times),observed_substep_rows=len(substeps),
                  solver_points_per_control=n_solver_per_control)
    check('integer_timestep_ratio',abs(ratio-n_solver_per_control)<1e-12)
    check('integer_duration_control_ratio',abs(duration/control_dt-n_control)<1e-9)
    if not check('substep_columns_exact',tuple(columns)==EXPECTED_SUBSTEP_COLUMNS):
        return report
    if not check('substep_matrix_shape',substeps.ndim==2 and substeps.shape[1]==len(columns)):
        return report
    check('substep_values_finite',np.all(np.isfinite(substeps)))
    complete = check('complete_substep_row_count',len(substeps)==expected_rows)
    control_complete = check('complete_control_row_count',len(control_times)==n_control)
    check('control_clock_finite',np.all(np.isfinite(control_times)))
    if control_complete:
        control_error = float(np.max(abs(control_times-np.arange(n_control)*control_dt)))
        report['maximum_control_grid_error_s'] = control_error
        check('control_integer_clock_grid',control_error<=CLOCK_ATOL_S)
        check('control_clock_strictly_increasing',np.all(np.diff(control_times)>0.))
    if not complete:
        return report
    times = substeps[:,columns.index('solver_time')]
    durations = substeps[:,columns.index('duration_s')]
    grouped_times = times.reshape(n_control,width)
    grouped_durations = durations.reshape(n_control,width)
    solver_times = grouped_times[:,:n_solver_per_control].reshape(-1)
    endpoints = grouped_times[:,-1]
    solver_duration_error = float(np.max(abs(grouped_durations[:,:n_solver_per_control]-physics_dt)))
    endpoint_duration_error = float(np.max(abs(grouped_durations[:,-1])))
    solver_grid_error = float(np.max(abs(solver_times-np.arange(n_solver)*physics_dt)))
    endpoint_grid_error = float(np.max(abs(endpoints-np.arange(1,n_endpoint+1)*control_dt)))
    solver_spacing_error = float(np.max(abs(np.diff(solver_times)-physics_dt)))
    endpoint_spacing_error = float(np.max(abs(np.diff(endpoints)-control_dt)))
    boundary_alignment_error = float(np.max(abs(endpoints[:-1]-grouped_times[1:,0])))
    total_duration = float(np.sum(durations))
    total_duration_error = abs(total_duration-duration)
    differences = np.diff(times)
    same_time_indices = np.flatnonzero(abs(differences)<=CLOCK_ATOL_S)
    expected_same_time = np.arange(width-1,expected_rows-1,width)
    report.update(maximum_solver_duration_error_s=solver_duration_error,
                  maximum_endpoint_duration_error_s=endpoint_duration_error,
                  maximum_solver_grid_error_s=solver_grid_error,
                  maximum_endpoint_grid_error_s=endpoint_grid_error,
                  maximum_solver_spacing_error_s=solver_spacing_error,
                  maximum_endpoint_spacing_error_s=endpoint_spacing_error,
                  maximum_boundary_alignment_error_s=boundary_alignment_error,
                  total_recorded_exposure_s=total_duration,total_duration_error_s=total_duration_error,
                  same_time_boundary_count=len(same_time_indices),
                  expected_same_time_boundary_count=n_control-1,
                  first_solver_time_s=float(solver_times[0]),last_solver_time_s=float(solver_times[-1]),
                  last_endpoint_time_s=float(endpoints[-1]),
                  minimum_global_clock_difference_s=float(np.min(differences)))
    check('solver_durations_equal_physics_step',solver_duration_error<=DURATION_ATOL_S)
    check('endpoint_durations_zero',endpoint_duration_error<=DURATION_ATOL_S)
    check('durations_nonnegative',np.all(durations>=0.))
    check('positive_duration_solver_count',np.count_nonzero(durations>DURATION_ATOL_S)==n_solver)
    check('zero_duration_endpoint_count',np.count_nonzero(abs(durations)<=DURATION_ATOL_S)==n_endpoint)
    check('solver_clock_strictly_increasing',np.all(np.diff(solver_times)>0.))
    check('endpoint_clock_strictly_increasing',np.all(np.diff(endpoints)>0.))
    check('global_clock_nondecreasing_with_tolerance',np.all(differences>=-CLOCK_ATOL_S))
    check('solver_full_uniform_grid',solver_grid_error<=CLOCK_ATOL_S and solver_spacing_error<=CLOCK_ATOL_S)
    check('endpoint_full_uniform_grid',endpoint_grid_error<=CLOCK_ATOL_S and endpoint_spacing_error<=CLOCK_ATOL_S)
    check('boundary_duplicates_only_at_sync_points',np.array_equal(same_time_indices,expected_same_time))
    check('sync_endpoint_matches_next_solver',boundary_alignment_error<=CLOCK_ATOL_S)
    check('whole_episode_duration',total_duration_error<=CLOCK_ATOL_S)
    check('first_solver_at_zero',abs(solver_times[0])<=CLOCK_ATOL_S)
    check('terminal_endpoint_at_episode_end',abs(endpoints[-1]-duration)<=CLOCK_ATOL_S)
    if control_complete:
        sync_error = float(np.max(abs(endpoints-(control_times+control_dt))))
        starts_error = float(np.max(abs(grouped_times[:,0]-control_times)))
        report.update(maximum_control_to_solver_start_error_s=starts_error,
                      maximum_control_to_endpoint_error_s=sync_error,
                      terminal_endpoint_minus_last_control_s=float(endpoints[-1]-control_times[-1]))
        check('every_solver_block_matches_control_start',starts_error<=CLOCK_ATOL_S)
        check('every_endpoint_matches_control_end',sync_error<=CLOCK_ATOL_S)
    if qpos_times is not None:
        expected_qpos = np.arange(0,n_control,10)*control_dt
        qpos_complete = check('complete_saved_state_clock',len(qpos_times)==len(expected_qpos))
        check('saved_state_rows_match_clock',qpos_rows==len(qpos_times))
        if qpos_complete:
            qpos_error = float(np.max(abs(qpos_times-expected_qpos)))
            report['maximum_saved_state_grid_error_s'] = qpos_error
            check('saved_state_clock_grid',qpos_error<=CLOCK_ATOL_S)
            check('saved_state_clock_strictly_increasing',np.all(np.diff(qpos_times)>0.))
    return report


def synthetic_arrays():
    control_dt,physics_dt,duration = .005,.0005,17.
    count = round(duration/control_dt)
    blocks = np.zeros((count,11,len(EXPECTED_SUBSTEP_COLUMNS)))
    blocks[:,:10,0] = np.arange(count*10).reshape(count,10)*physics_dt
    blocks[:,-1,0] = np.arange(1,count+1)*control_dt
    blocks[:,:10,-1] = physics_dt
    return blocks.reshape(-1,len(EXPECTED_SUBSTEP_COLUMNS)),np.arange(count)*control_dt


def self_tests():
    rows,controls = synthetic_arrays()
    def inspect(data=rows,clock=controls):
        return inspect_arrays(data,EXPECTED_SUBSTEP_COLUMNS,clock,.005,.0005,17.,controls[::10],len(controls[::10]))
    positive = inspect()
    if not positive['passed']:
        raise AssertionError(positive)
    corrupted = {}
    corrupted['missing_solver_row'] = rows[np.arange(len(rows))!=7]
    corrupted['duplicate_row'] = np.insert(rows,7,rows[7],axis=0)
    wrong_duration = rows.copy(); wrong_duration[10,-1]=.0005
    corrupted['endpoint_integrated_twice'] = wrong_duration
    wrong_time = rows.copy(); wrong_time[12,0]+=.0002
    corrupted['off_grid_solver_time'] = wrong_time
    wrong_terminal = rows.copy(); wrong_terminal[-1,0]-=.005
    corrupted['missing_terminal_endpoint'] = wrong_terminal
    wrong_order = rows.copy(); wrong_order[[10,11]]=wrong_order[[11,10]]
    corrupted['endpoint_solver_roles_swapped'] = wrong_order
    negative = []
    for name,data in corrupted.items():
        result = inspect(data)
        if result['passed']:
            raise AssertionError(f'Coverage checker accepted {name}')
        negative.append(dict(case=name,rejected=True,errors=result['errors']))
    wrong_control = controls.copy(); wrong_control[-1]-=.005
    result = inspect(clock=wrong_control)
    if result['passed']:
        raise AssertionError('Coverage checker accepted the incorrect control terminal time')
    negative.append(dict(case='incorrect_control_terminal_clock',rejected=True,errors=result['errors']))
    return dict(passed=True,scope='Pure synthetic arrays; no physical simulation or formal result read',
                positive_complete_grid=positive,negative_corruption_cases=negative)


def source_contract(path):
    text = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(text)
    plant = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Plant')
    advance = next(node for node in plant.body if isinstance(node,ast.FunctionDef) and node.name=='advance')
    calls = sorted((node for node in ast.walk(advance) if isinstance(node,ast.Call) and
                    isinstance(node.func,ast.Attribute) and node.func.attr=='record_contact_diagnostics'),
                   key=lambda node:node.lineno)
    if len(calls)!=2 or ast.unparse(calls[0].args[0])!='solver_time' or ast.unparse(calls[0].args[1])!='self.model.opt.timestep':
        raise AssertionError('Frozen solver recording contract differs from the reviewed implementation')
    if ast.unparse(calls[1].args[0])!='float(self.data.time)' or ast.literal_eval(calls[1].args[1])!=0.:
        raise AssertionError('Frozen endpoint recording contract differs from the reviewed implementation')
    return dict(source_sha256=sha256(path),advance_source_sha256=hashlib.sha256(
        ast.get_source_segment(text,advance).encode('utf-8')).hexdigest(),
        description='Ten positive-duration solver starts followed by one zero-duration sync endpoint per 5ms; adjacent blocks share the endpoint/start timestamp')


def audit_suite(directory,plan):
    directory = directory.resolve()
    manifest_path,trials_path = directory/'manifest.json',directory/'trials.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    trials = json.loads(trials_path.read_text(encoding='utf-8'))
    seeds = plan['test_seeds'] if manifest['suite']=='test' else plan['robustness_seeds']
    if manifest['suite'] not in ('test','robustness'):
        raise AssertionError('Only the two declared formal suites are accepted')
    if sorted(manifest['seeds'])!=seeds or set(manifest['methods'])!=set(plan['methods']):
        raise AssertionError('Formal suite seed/method declaration changed')
    if manifest['code_sha256']!=plan['code_sha256'] or manifest['final_code_sha256']!=plan['code_sha256']:
        raise AssertionError('Formal source differs from the prespecified frozen plan')
    if not manifest['code_unchanged_during_run'] or manifest['gains']!=plan['gains']:
        raise AssertionError('Formal runtime was not the frozen declared configuration')
    if manifest['physics_timestep']!=.0005 or manifest['control_timestep']!=.005:
        raise AssertionError('The audited formal sampling contract is 0.5ms physics / 5ms control')
    expected={(s,m,seed) for s in manifest['scenarios'] for m in plan['methods'] for seed in seeds}
    observed={(r['scenario'],r['method'],r['seed']) for r in trials}
    if len(trials)!=len(expected) or observed!=expected or manifest['completed_trials']!=manifest['expected_trials'] or len(trials)!=manifest['expected_trials']:
        raise AssertionError('Formal Cartesian trial matrix is incomplete')
    definitions={s['name']:s for s in manifest['scenario_definitions']}
    source = directory/'snapshot'/'physics.py'
    if sha256(source)!=plan['code_sha256']['physics.py']:
        raise AssertionError('Frozen physics snapshot does not match the declared plan')
    contract = source_contract(source)
    rows=[]
    for index,trial in enumerate(trials,1):
        record=dict(scenario=trial['scenario'],method=trial['method'],seed=trial['seed'])
        try:
            trace_path=(directory/trial['trace_file']).resolve()
            if not trace_path.is_relative_to(directory):
                raise AssertionError('Trace path leaves the formal results directory')
            original_hash=sha256(trace_path)
            record.update(path=str(trace_path),npz_sha256=original_hash,bytes=trace_path.stat().st_size)
            with np.load(trace_path,allow_pickle=False) as saved:
                columns=saved['columns'].tolist()
                control_times=saved['trace'][:,columns.index('time')].copy()
                substeps=saved['substeps'].copy()
                subcols=saved['substep_columns'].tolist()
                qpos_times=saved['qpos_time'].copy()
                qpos_rows=saved['qpos'].shape[0]
                metadata=json.loads(str(saved['metadata']))
            if (metadata['scenario'],metadata['method'],metadata['seed'])!=(trial['scenario'],trial['method'],trial['seed']):
                raise AssertionError('NPZ metadata differs from its declared trial')
            duration=definitions[trial['scenario']]['duration']
            if duration!=17.:
                raise AssertionError('The frozen formal episode is not the declared full 17 seconds')
            record.update(inspect_arrays(substeps,subcols,control_times,.005,.0005,duration,qpos_times,qpos_rows))
            record['file_unchanged_during_read']=sha256(trace_path)==original_hash
            if not record['file_unchanged_during_read']:
                record['passed']=False; record['errors'].append('npz_changed_during_read')
        except Exception as error:
            record.update(passed=False,exception=str(error))
        rows.append(record)
        if index%100==0 or index==len(trials):
            print(json.dumps(dict(suite=manifest['suite'],checked=index,expected=len(trials),
                                  failures=sum(not item['passed'] for item in rows))),flush=True)
    return dict(suite=manifest['suite'],directory=str(directory),passed=all(r['passed'] for r in rows),
                trials=len(rows),failed_trials=sum(not r['passed'] for r in rows),
                manifest_sha256=sha256(manifest_path),trials_sha256=sha256(trials_path),
                recording_contract=contract,records=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--results',nargs='+',type=Path,
                        default=[ROOT/'output'/'test_holdout_v2',ROOT/'output'/'robustness_holdout_v2'])
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    out=(args.out or ROOT/'review'/('trace_coverage_self_tests.json' if args.self_test else 'trace_coverage_audit.json')).resolve()
    if any(out.is_relative_to(directory.resolve()) for directory in args.results):
        raise AssertionError('Coverage output must not modify either formal results directory')
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                script_sha256=sha256(__file__),scope='Read-only timing/cardinality coverage, not a change to force metrics',
                runtime=dict(python=platform.python_version(),numpy=np.__version__),self_tests=self_tests())
    if args.self_test:
        report['passed']=report['self_tests']['passed']
    else:
        plan_path=ROOT/'review'/'formal_plan.json'
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        report['formal_plan_sha256']=sha256(plan_path)
        report['suites']=[audit_suite(directory,plan) for directory in args.results]
        report['passed']=all(suite['passed'] for suite in report['suites'])
        report['checked_trials']=sum(suite['trials'] for suite in report['suites'])
        report['full_declared_pair_of_suites_checked']=sorted(suite['suite'] for suite in report['suites'])==['robustness','test']
        if not report['full_declared_pair_of_suites_checked']:
            report['passed']=False
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],output=str(out),checked_trials=report.get('checked_trials',0)),ensure_ascii=False),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())

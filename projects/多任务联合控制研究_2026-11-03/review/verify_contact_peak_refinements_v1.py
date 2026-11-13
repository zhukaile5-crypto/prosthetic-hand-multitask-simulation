"""Independent saved-array timestep diagnostics, never imports a simulator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from verify_contact_peak_convergence_bridge_v1 import ROOT,PROTOTYPE,METHODS,path_for,read,sha,exact,scalar


def resolved(path):
    p=Path(path);return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def recompute(a,meta):
    start=a['substep_time'];dur=a['substep_duration'];end=start+dur
    force=a['substep_skin_normal'];limits=np.asarray(meta['limits']);budget=float(meta['budget'])
    local=np.maximum(force-limits,0.);total=np.maximum(force.sum(1)-budget,0.)
    terminal=bool(np.any(a['skin_normal'][-1]>limits+.02) or a['skin_normal'][-1].sum()>budget+.02)
    full=dict(force_protection_pass=bool(not terminal and np.all(force<=limits+.02) and np.all(force.sum(1)<=budget+.02)),
        local_excess_integral=float(np.sum(local*dur[:,None])),total_excess_integral=float(total@dur),
        finger_peak=np.maximum(force.max(0),a['skin_normal'][-1]).tolist(),
        palm_peak=float(max(a['substep_palm'].max(),a['palm'][-1])))
    stages={};peaks={}
    for name,lo,hi in [('preparation',0.,7.5),('holding',7.5,12.),('collision',9.2,10.4),('release',12.,16.)]:
        mask=(start>=lo-1e-9)&(start<hi-1e-9);indices=np.flatnonzero(mask)
        row,digit=np.unravel_index(int(np.argmax(force[mask])),force[mask].shape);k=int(indices[row])
        nontarget=np.maximum(force[mask]-a['substep_normal'][mask],0).sum(1)
        contact=np.flatnonzero(nontarget>1e-10)
        peak=dict(peak_N=float(force[k,digit]),digit=int(digit),peak_start_s=float(start[k]),peak_end_s=float(end[k]),
            target_same_digit_N=float(a['substep_normal'][k,digit]),nontarget_same_digit_N=float(max(0.,force[k,digit]-a['substep_normal'][k,digit])),
            target_palm_normal_sum_N=float(a['substep_palm'][k]),table_normal_sum_N=float(a['substep_support'][k]),
            local_excess_Ns=float(np.sum(local[mask]*dur[mask,None])),budget_excess_Ns=float(total[mask]@dur[mask]),
            first_nontarget_finger_contact_end_s=float(end[indices[contact[0]]]) if len(contact) else None,
            maximum_hand_object_penetration_m=float(a['substep_penetration'][mask].max()),
            maximum_world_penetration_m=float(a['substep_world_penetration'][mask].max()),maximum_warning_count=int(a['substep_warning_count'][mask].max()),
            actual_ctrl_at_peak=a['substep_actual_ctrl'][k].tolist(),numerical_qpos_at_peak_evaluation_only=a['substep_qpos'][k].tolist())
        peaks[name]=peak
        stages[name]=dict(peak_N=peak['peak_N'],digit=digit,peak_end_s=peak['peak_end_s'],
            duration_s=float(dur[mask].sum()),local_excess_Ns=peak['local_excess_Ns'],budget_excess_Ns=peak['budget_excess_Ns'],
            above_cap_any_finger_duration_s=float(dur[mask & np.any(force>limits,axis=1)].sum()),
            above_guard_any_finger_duration_s=float(dur[mask & np.any(force>limits+.02,axis=1)].sum()),
            target_same_digit_N=peak['target_same_digit_N'],nontarget_same_digit_N=peak['nontarget_same_digit_N'],
            first_nontarget_finger_contact_end_s=peak['first_nontarget_finger_contact_end_s'],
            maximum_world_penetration_m=peak['maximum_world_penetration_m'],maximum_warning_count=peak['maximum_warning_count'])
    return full,stages,peaks


def main():
    p=argparse.ArgumentParser();p.add_argument('--refine',type=Path,default=ROOT/'output/contact_peak_convergence_refined_v2')
    p.add_argument('--bridge',type=Path,default=ROOT/'output/contact_peak_convergence_bridge_v1')
    p.add_argument('--out',type=Path,default=ROOT/'review/contact_peak_refinements_independent_v1.json');args=p.parse_args()
    run=args.refine.resolve();bridge=args.bridge.resolve();out=args.out.resolve()
    if not (run/'execution_complete.json').exists():print('pending: no terminal marker');return 3
    if out.exists():raise FileExistsError(out)
    finish=read(run/'execution_complete.json');manifest=read(run/'manifest.json');lock=read(run/'source_lock.json')
    bfinish=read(bridge/'execution_complete.json');bmanifest=read(bridge/'manifest.json')
    prior=read(ROOT/'review/contact_peak_convergence_bridge_independent_v1.json')
    checks=[];issues=[];rows=[];rng_reference=None
    def check(name,ok,detail=None):
        item=dict(name=name,passed=bool(ok),detail=detail);checks.append(item)
        if not ok:issues.append(item)
    check('independent_three_bridge_precondition',prior['passed'] is True and prior['bridge_gate_passed_for_samelock_refinements'] is True)
    check('six_unique_complete_assignments',len(manifest['jobs'])==len(finish['results'])==6 and
        {(j['method'],j['dt']) for j in manifest['jobs']}=={(m,dt) for m in METHODS for dt in (.00025,.000125)} and
        finish['execution_complete'] is True and finish['phase']=='refine' and all(r['complete'] for r in finish['results']))
    check('source_and_rng_execution_gates',finish['sources_unchanged'] is True and
        finish['identical_sensor_rng_final_per_method']=={m:True for m in METHODS})
    locksha=sha(run/'source_lock.json')
    check('same_lock_as_live_and_complete_bridge',locksha==sha(PROTOTYPE/'source_lock.json')==sha(bridge/'source_lock.json')==
        manifest['source_lock_sha256']==finish['source_lock_sha256']==prior['source_lock_sha256'] and bfinish['bridge_all_exact'] is True)
    check('source88',len(lock['sources'])==88)
    for key,value in lock['sources'].items():check('source/'+key,sha(path_for(key))==sha(run/'sources'/key)==value)
    check('same_original_inputs',manifest['input_sources']==lock['source_inputs']==bmanifest['input_sources'])
    wanted={'time','qpos','qvel','ctrl_before','ctrl','alpha','phase','executed_motion_reference','source_environment_json',
        'rng_state_before_sample','rng_state_after_sample','skin_normal','palm','warning_count',
        'substep_time','substep_duration','substep_skin_normal','substep_normal','substep_palm','substep_support',
        'substep_penetration','substep_world_penetration','substep_state_end_time','substep_qpos','substep_qvel','substep_actual_ctrl','substep_warning_count'}
    for method in METHODS:
        inp=lock['source_inputs'][method];oldpath=Path(inp['path']);oldmeta=read(oldpath.with_name('metadata.json'))
        check(method+'/old_trace_and_metadata_unchanged',sha(oldpath)==inp['trace_sha256'] and sha(oldpath.with_name('metadata.json'))==inp['metadata_sha256'])
        with np.load(oldpath,allow_pickle=False) as z:
            original={k:z[k] for k in ('time','ctrl_before','ctrl','alpha','phase','executed_motion_reference','environment_json')}
        import xml.etree.ElementTree as ET
        names=[a.attrib['name'] for a in ET.parse(run/'sources/foundation/assets/legacy_model.xml').getroot().find('actuator')]
        ids=[names.index(n) for n in oldmeta['robot_actuator_telemetry']['axes21']];other=sorted(set(range(22))-set(ids))
        expectedctrl=original['ctrl'][:-1].copy()
        expectedctrl[:,other[0]]=[float(np.clip(json.loads(str(e))['obstacle_x'],-.4,.1)) for e in original['environment_json'][:-1]]
        jobs=[next(j for j in bmanifest['jobs'] if j['method']==method)]+[j for j in manifest['jobs'] if j['method']==method]
        for job in jobs:
            dt=float(job['dt']);folder=resolved(job['destination']);path=folder/'trace.npz';meta=read(folder/'metadata.json');result=read(folder/'result.json')
            prefix=method+f'/{dt:.6f}'
            check(prefix+'/trace_identity',sha(path)==meta['trace_sha256']==result['trace_sha256'] and result['source_trace_sha256']==inp['trace_sha256'])
            parent_result=next(r for r in (bfinish['results'] if dt==.0005 else finish['results']) if r['method']==method and r['dt']==dt)
            check(prefix+'/terminal_record_identical',result==parent_result)
            expectedconfig=dict(oldmeta['config'],timestep=dt)
            check(prefix+'/actual_config_only_timestep_changed',meta['config']==expectedconfig and meta['physics_timestep']==dt and
                meta['original_trace_sha256']==inp['trace_sha256'] and meta['new_feedback_decisions']==0 and meta['actual_q_restored_after_reset'] is False)
            check(prefix+'/model_identity_consistent_with_input',dt!=.0005 or meta['model_sha256']==oldmeta['model_sha256'])
            with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in wanted}
            for key in ('time','ctrl_before','ctrl','alpha','phase','executed_motion_reference'):
                check(prefix+'/same_actual_boundary_input/'+key,exact(a[key],original[key]))
            check(prefix+'/same_current_environment',exact(a['source_environment_json'],original['environment_json']))
            n=int(round(16./dt));per=int(round(.005/dt));st=a['substep_time'];dur=a['substep_duration'];end=a['substep_state_end_time']
            check(prefix+'/complete_clock',len(a['time'])==3201 and len(st)==n and np.all(dur==dt) and
                len(end)==n and abs(float(dur.sum())-16.)<1e-9 and np.all(np.diff(st)>0) and
                abs(float(st[0]))<1e-9 and abs(float(end[-1])-16.)<1e-8 and np.max(abs(end-st-dur))<1e-9 and
                np.max(abs(np.diff(st)-dt))<1e-9)
            check(prefix+'/all_selected_numeric_finite',all(np.isfinite(value).all() for key,value in a.items() if value.dtype.kind in 'fiu'))
            check(prefix+'/all_actual_micro22_inputs',a['substep_actual_ctrl'].shape==(n,22) and exact(a['substep_actual_ctrl'],np.repeat(expectedctrl,per,axis=0)))
            check(prefix+'/micro_qv_shape',a['substep_qpos'].shape==(n,a['qpos'].shape[1]) and a['substep_qvel'].shape==(n,a['qvel'].shape[1]))
            before=a['rng_state_before_sample'];after=a['rng_state_after_sample']
            check(prefix+'/RNG3201_chain',len(before)==len(after)==3201 and exact(before[1:],after[:-1]) and str(after[-1])==result['noise_rng_final'])
            if rng_reference is None:rng_reference=(before.copy(),after.copy())
            else:check(prefix+'/same_draws_all_methods_and_steps',exact(before,rng_reference[0]) and exact(after,rng_reference[1]))
            full,stages,peaks=recompute(a,oldmeta);savedguard=read(folder/'original_force_guards.json');savedpeak=read(folder/'peak_diagnostics.json')
            check(prefix+'/entire_original_force_guard_recomputed',all(full[k]==savedguard[k] for k in full))
            check(prefix+'/all_stage_peak_meta_recomputed',peaks==savedpeak)
            for phase,key in [('preparation','preparation'),('holding','policy_holding'),('release','mechanical_release_and_after')]:
                saved=savedguard['stages'][key];stage=stages[phase]
                check(prefix+'/original_guard_stage/'+phase,stage['duration_s']==saved['duration_s'] and
                    stage['local_excess_Ns']==saved['local_excess_integral_Ns'] and stage['budget_excess_Ns']==saved['total_excess_integral_Ns'] and
                    stage['peak_N']==max(saved['finger_peak_N']))
            local_sum=sum(stages[s]['local_excess_Ns'] for s in ('preparation','holding','release'))
            check(prefix+'/nonoverlapping_stages_integral_identity',abs(local_sum-full['local_excess_integral'])<1e-12)
            rows.append(dict(method=method,dt=dt,force_guard=full,stages=stages,
                trace_sha256=result['trace_sha256'],model_sha256=meta['model_sha256'],micro_records=n,
                warnings_max=max(int(a['warning_count'].max()),int(a['substep_warning_count'].max())),
                no_policy_outcome_exported=True))
            print(json.dumps(dict(method=method,dt=dt,records_checked=len(rows),issues=len(issues))),flush=True)
            del a
        del original
    sensitivity=[]
    for method in METHODS:
        ordered=sorted([r for r in rows if r['method']==method],key=lambda r:-r['dt'])
        for a,b in zip(ordered[:-1],ordered[1:]):
            item=dict(method=method,coarse_dt=a['dt'],fine_dt=b['dt'],stages={})
            for name in ('preparation','holding','collision','release'):
                x=a['stages'][name];y=b['stages'][name]
                item['stages'][name]=dict(coarse_peak_N=x['peak_N'],fine_peak_N=y['peak_N'],
                    peak_change_percent=100*(y['peak_N']-x['peak_N'])/x['peak_N'] if x['peak_N'] else None,
                    peak_time_shift_s=y['peak_end_s']-x['peak_end_s'],coarse_local_Ns=x['local_excess_Ns'],fine_local_Ns=y['local_excess_Ns'],
                    local_Ns_change_percent=100*(y['local_excess_Ns']-x['local_excess_Ns'])/x['local_excess_Ns'] if x['local_excess_Ns'] else None)
            sensitivity.append(item)
    incident=ROOT/'review/contact_peak_refine_launcher_incident_v1.json'
    report=dict(schema='contact_peak_timestep_refinements_independent_readonly_v1',passed=not issues,checks=checks,issues=issues,
        rows=rows,adjacent_step_descriptive_sensitivity=sensitivity,complete_bridge_records=3,complete_fresh_refinement_records=6,
        source_lock_sha256=locksha,checker_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'review/verify_contact_peak_convergence_bridge_v1.py'),
        failed_launch_incident_path=str(incident),failed_launch_incident_sha256=sha(incident),
        failed_launch_scope='Prior refine_v1 Windows multiprocessing launch failures retained separately; no completed physical refinement record pooled from it.',
        source_context='old V12 collision/hold/617902; fixed actual22 references from reset. No current native48 input.',
        new_physics_steps=0,simulator_imported='mujoco' in sys.modules,
        limitations=['Fixed-input numerical sensitivity, not new closed-loop policy efficacy.',
            'Historical handoff/task metadata is not treated as newly achieved policy admission or mission success.',
            'No automatic convergence/safety threshold added; no contact parameter changed.',
            'Old per-substep q/v unavailable. New q/v checked finite/coverage only.',
            'Raw pressure attribution at reported peaks is independently recomputed from saved target/all-skin channels; patch gzip not exhaustively decoded.',
            'Collision window nests holding and is never added to the three disjoint phases.'])
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,default=scalar)
    print(json.dumps(dict(passed=not issues,checks=len(checks),issues=len(issues),output=str(out),sha256=sha(out),new_physics_steps=0)))
    return 0 if not issues else 2


if __name__=='__main__':raise SystemExit(main())

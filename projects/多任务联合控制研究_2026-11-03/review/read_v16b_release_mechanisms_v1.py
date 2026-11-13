"""Append goal-corrected V16b3 to frozen V16/A/ratio nine-record readout.

No physics/production import. Run only after root reports all stages terminal.
All force/contact/decision formulas are the unchanged prior mechanism reader.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'review/v16_release_mechanisms_full9_v1.json'
OLD_SHA='6ca249c66e1b3d0de8859d3042a2600723e70d66540100b15cc61db1d8a872bb'
HELPER=ROOT/'review/read_v16_release_mechanisms_v1.py'
HELPER_SHA='571fbdab19ec01f00b8923f68023be5db9774a4de6df1e5106e79d010b103c4d'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


class ArrayCache:
    """Local offline helper proxy: each NPZ column decompressed at most once."""
    def __init__(self):self.files={};self.arrays={}
    def load(self,path,**kwargs):
        key=str(Path(path).resolve())
        if key not in self.files:self.files[key]=np.load(path,**kwargs);self.arrays[key]={}
        parent=self
        class View:
            def __enter__(self):return self
            def __exit__(self,*unused):return False
            def __getitem__(self,name):
                if name not in parent.arrays[key]:parent.arrays[key][name]=parent.files[key][name]
                return parent.arrays[key][name]
        return View()
    def close(self):
        for f in self.files.values():f.close()


class NumpyProxy:
    def __init__(self,cache):self.cache=cache
    def __getattr__(self,name):return self.cache.load if name=='load' else getattr(np,name)


def stage_alignment(row,arrays,public_helper):
    if not row.get('measurements_computed') or not row.get('release_object'):return None
    with arrays.load(Path(row['run'])/'trace.npz',allow_pickle=False) as z:
        t=z['time'];obs=z['observation_json'];ref=z['executed_reference21'];lift=z['lift']
    history={int(i):json.loads(str(obs[i])) for i in np.flatnonzero(t>=11.75-1e-9)}
    events={'release_entry':row['release_object']['entry'],'record_end16':row['release_object']['end16']}
    for label,category,key in [('first_receiver_top','top','first_positive_state'),('last_hand_contact','hand','last_positive_state'),('first_floor','floor','first_positive_state')]:
        events[label]=(row['release_contact_ledger'].get(category) or {}).get(key)
    goal=np.asarray(row['goal_comparison']['declared_goal18'])
    stages={}
    for label,state in events.items():
        if state is None:stages[label]=None;continue
        end=state['integration_end_time_s'];i=min(len(t)-1,int(np.searchsorted(t,end-1e-9)));o=history[i]
        h=public_helper.project_public_hand(o,history);receiver=o['receiver_surface'];cop=public_helper.cop_proxy(receiver)
        signedR=None if not receiver['valid'] else -float(receiver['wrench6'][2])
        stages[label]=dict(physical_event=state,first_public_read_s=float(t[i]),public_acquisition_s=o['sample_time'],
            hand_support_proxy=h,receiver_signed_support_N=signedR,receiver_compression_N=receiver['normal_load_N'],
            signedR_role='minusFz in the declared horizontal fixed receiver frame; any actor, not target-only',cop=cop,
            incoming_public_reference21=o['robot_reference21'],post_action_reference21=ref[i],robot_lift_encoder_m=float(lift[i]),
            finger_reference_distance_to_declared_goal_inf=float(np.max(abs(ref[i,:18]-goal))),
            public_phase=o.get('receiver_release_tasks'),
            clocks='cached-contact event uses integrated end label; public observation is next200Hz read and10ms delayed, no instantaneous force match assumed')
    mask=(t>=12-1e-9)&(t<=16+1e-9);rr=ref[mask];error=np.max(abs(rr[:,:18]-goal),axis=1)
    tol=128*np.finfo(float).eps*max(1.,float(np.max(abs(goal))),float(np.max(abs(rr[:,:18]))))
    hits=np.flatnonzero(error<=tol);delta=np.diff(rr,axis=0)
    return dict(events=stages,first_declared_goal_reference_at_numerical_roundoff_s=None if not len(hits) else float(t[mask][hits[0]]),
        goal_distance_roundoff_allowance_rad=tol,reference_TV_finger18_rad=float(abs(delta[:,:18]).sum()),
        reference_TV_lift_m=float(abs(delta[:,18]).sum()),reference_TV_wrist2_rad=float(abs(delta[:,19:]).sum()),
        exact_nonzero_5ms_reference_steps=int(np.count_nonzero(np.any(delta!=0,axis=1))),
        numerical_nonzero_5ms_reference_steps=int(np.count_nonzero(np.max(abs(delta),axis=1)>tol)),
        goal_attainment_not_physical_clearance=True)


def goal_diagnostics(row,helper):
    if not row.get('measurements_computed'):return row
    reference=row['reference'];end=np.asarray(reference['end21'])[:18]
    historical=np.r_[[-1.,.1,0.,0.,0.],helper.OPEN18[5:]]
    role=row['role'];source='frozen_complete_release_terminal18'
    actual_goal=historical
    if role=='new_V16':actual_goal=helper.OPEN18;source='V16_preparation_OPEN18_binding_preserved_as_failed_attempt'
    elif role=='corrected_V16b':
        metadata=read(Path(row['run'])/'metadata.json');contract=metadata.get('release_clearance_goal')
        if contract is None:
            row['goal_comparison']=dict(available=False,reason='missing_actual_goal_contract_not_substituted')
            return row
        actual_goal=np.asarray(contract['reference18'],float)
        if actual_goal.shape!=(18,) or not np.isfinite(actual_goal).all():
            row['goal_comparison']=dict(available=False,reason='invalid_actual_goal_contract_not_substituted')
            return row
        source=contract['source'];row['corrected_goal_contract']=contract
    row['goal_comparison']=dict(available=True,declared_goal_source=source,declared_goal18=actual_goal.tolist(),
        reference_end_distance_to_this_declared_goal_inf=float(np.max(abs(end-actual_goal))),
        historical_clearance18=historical.tolist(),reference_end_distance_to_historical_clearance_inf=float(np.max(abs(end-historical))),
        reference_end_distance_to_preparation_OPEN_inf=float(np.max(abs(end-helper.OPEN18))),
        task_goal_is_not_object_clearance_or_success=True,old_labels_unchanged=True)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args(argv)
    out=Path(args.out);out=out if out.is_absolute() else ROOT/out
    if out.exists():raise FileExistsError(out)
    if sha(OLD)!=OLD_SHA or sha(HELPER)!=HELPER_SHA:raise RuntimeError('frozen readout/formula helper identity changed')
    spec=importlib.util.spec_from_file_location('v16_frozen_mechanism_reader',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    public_path=ROOT/'review/read_v16_release_geometry_six_cases_v1.py'
    specp=importlib.util.spec_from_file_location('v16_frozen_public_diagnostic',public_path);public=importlib.util.module_from_spec(specp);specp.loader.exec_module(public)
    prior=read(OLD);rows=deepcopy(prior['records'])
    plan_path=ROOT/'review/v16b_release_small_plan_v1.json';plan=read(plan_path)
    stages={x['motion']:ROOT/x['output'] for x in plan['stages'] if x['policy']=='coordinated'}
    cache=ROOT/'review/v16b_release_mechanism_cache_v1';cache.mkdir(exist_ok=True)
    for motion,scene in [('hold','fragile_conflict'),('transport','nominal'),('transport','friction_drop')]:
        path=stages[motion]/'runs'/f'{scene}__ratio_effort__619732'
        arrays=ArrayCache();helper.np=NumpyProxy(arrays)
        try:
            new=goal_diagnostics(helper.run_record(path,'corrected_V16b',(motion,scene),cache),helper)
            new['stage_alignment']=stage_alignment(new,arrays,public)
            rows.append(new)
        finally:arrays.close();helper.np=np
    rows=[goal_diagnostics(row,helper) for row in rows]
    rows.sort(key=lambda r:(r['context']['motion_mode'],r['context']['scenario'],r['role']))
    complete=len(rows)==12 and all(r.get('measurements_computed') and all(r.get('coverage',{}).get(k) for k in ('full16','ledger_release_known','raw_force_complete')) for r in rows)
    if not complete and not args.allow_partial:raise RuntimeError('all12 complete inputs required; no result finalization from partial runs')
    pairs=[]
    for new in [r for r in rows if r['role']=='corrected_V16b']:
        same=[r for r in rows if r['context']==new['context'] and r['role']!='corrected_V16b']
        for old in same:
            def metrics(r):
                if not r.get('measurements_computed'):return None
                return dict(original_task_success=r['original']['legacy_task_success'],original_release=r['original']['release'],
                    release_force=r['release_force'],floor=r['release_contact_ledger']['floor'],
                    receiver_top=r['release_contact_ledger']['top'],object_end=r['release_object']['end16'],
                    reference_end21=r['reference']['end21'],goal_comparison=r.get('goal_comparison'),
                    technical=r['release_decisions'])
            pairs.append(dict(context=new['context'],candidate_role=new['role'],reference_role=old['role'],candidate=metrics(new),reference=metrics(old)))
    document=dict(schema='v16b_goal_correction_twelve_release_mechanisms_v1',complete=complete,planned=12,
        computed=sum(r.get('measurements_computed',False) for r in rows),records=rows,pairs=pairs,
        source_sha256={Path(__file__).name:sha(__file__),HELPER.name:HELPER_SHA,public_path.name:sha(public_path)},
        input_frozen_V16_nine_readout_sha256=OLD_SHA,corrected_plan_sha256=sha(plan_path),
        scope=dict(physics_steps=0,models_constructed=0,controllers_run=0,old_sources_or_labels_changed=False,
            new_V16b_results_not_assumed_successful=True,
            goal_correction_not_algorithm_innovation=True,
            continuous_load_or_no_floor_not_placement=True,
            commanded_terminal_reference_not_physical_clearance=True,
            three_seen_contexts_one_physical_seed_not_twelve_independent_draws=True,
            terminal_goal_extraction_clock_distinct_from_actual_command_clock=True))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(helper.plain(document),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(complete=complete,computed=document['computed'],planned=12,pairs=len(pairs),path=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':main()

"""Independent first-five causal primitive reader. Never executes a model.

Without --execute, only checks a terminal marker and refuses branch data.
All five assigned arms remain in the report; no response model is fitted.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=('zero','lift_plus','lift_minus','wrj1_plus','wrj1_minus')
HORIZONS=(.005,.010,.020,.040,.100,.200)
TIMING={'control_ms','actor_callback_ms','full_control_loop_ms','observation_ms','physics_advance_ms'}
LEAVES=('qp','native_inverse','whole_decision')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def gz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(s) for s in f]
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind==b.dtype.kind=='U':return all(str(x).encode('utf8')==str(y).encode('utf8') for x,y in zip(a.flat,b.flat))
    return a.dtype==b.dtype and a.tobytes()==b.tobytes()
def debug(v):
    d=json.loads(str(v));t=d.get('solver_timing_ms')
    if t is not None:
        if not isinstance(t,dict):raise ValueError('invalid debug timing container')
        for k in LEAVES:
            if k in t:
                x=t[k]
                if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<0:raise ValueError('invalid debug timing leaf')
                t[k]='__finite_nonnegative_wall_time__'
    return json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=True).encode('utf8')
def prefix_equal(a,b,t):
    if set(a)!=set(b):return False,{'column_keys':False}
    ia=np.asarray(a['time'])<t-1e-9;ib=np.asarray(b['time'])<t-1e-9;checks={}
    for k in a:
        if k=='metadata':continue
        x,y=a[k],b[k]
        # Trace columns have the leading time dimension; metadata is scalar.
        if x.ndim==0 or y.ndim==0:checks[k]=bits(x,y);continue
        x,y=x[ia],y[ib]
        if k in TIMING:checks[k]=x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all() and (x>=0).all() and (y>=0).all()
        elif k=='debug_json':checks[k]=x.shape==y.shape and all(debug(u)==debug(v) for u,v in zip(x.flat,y.flat))
        else:checks[k]=bits(x,y)
    return bool(all(checks.values())),checks
def index_time(rows,key):return {int(round(float(r[key])/.005)):r for r in rows}
def footprint(rows,zero,t):
    a=index_time(rows,'issue_time_s');z=index_time(zero,'issue_time_s');ks=sorted(k for k in a if k in z and k*.005>=t-1e-9)
    d=np.asarray([np.asarray(a[k]['executed_reference21'],float)-np.asarray(z[k]['executed_reference21'],float) for k in ks],float).reshape(-1,21)
    if not ks:return dict(available=False,paired_issues=0)
    maximum=np.max(np.abs(d),axis=0)
    return dict(available=True,paired_issues=len(ks),times_s=[k*.005 for k in ks],delta_reference21=d.tolist(),max_abs21=maximum.tolist(),signed_reference_time_integral21=(np.sum(d,axis=0)*.005).tolist(),nonzero_axes21=np.flatnonzero(maximum>0).tolist(),role='ACTUAL_EXECUTED_REFERENCE_DIFFERENCE_RELATIVE_TO_OFFLINE_ZERO_ALL21_NOT_SINGLE_AXIS_GRADIENT')
def horizon_record(packets,zero,origin,h):
    target=origin+h;tick=int(round(target/.005))
    candidates=[p for p in packets if int(p['acquisition_tick'])==tick]
    zs=[p for p in zero if int(p['acquisition_tick'])==tick]
    if not candidates:return dict(horizon_s=h,acquisition_time_s=target,available=False,reason='MISSING_ACQUISITION_PACKET')
    p=min(candidates,key=lambda v:v['available_at_s']);z=min(zs,key=lambda v:v['available_at_s']) if zs else None
    digits=[]
    for d in p['digits']:
        zd=None if z is None else z['digits'][d['digit']]
        e=d['external_projection2_m_s'];ze=None if zd is None else zd['external_projection2_m_s']
        samebasis=e is not None and ze is not None and bits(np.asarray(d['projection_world2x3']),np.asarray(zd['projection_world2x3']))
        digits.append(dict(digit=d['digit'],status=d['status'],zero_status=None if zd is None else zd['status'],reasons=d['reasons'],instrument_valid=d['instrument_valid'],observed_normal_N=d['observed_normal_N'],noise_group=d['noise_group'],patch_count=len(d['patch_support']),pressure_shares=d['pressure_shares'],projection_world2x3=d['projection_world2x3'],conditional_C2x6=d['conditional_C2x6'],external_projection2_m_s=e,zero_external_projection2_m_s=ze,same_projection_basis_exact=samebasis,device_coordinate_difference_only=None if e is None or ze is None else (np.asarray(e)-np.asarray(ze)).tolist(),common_fixed_physical_output_difference_available=samebasis,actor_membership_verified=False))
    return dict(horizon_s=h,acquisition_time_s=p['acquisition_time_s'],read_time_s=p['read_time_s'],available_at_s=p['available_at_s'],age_s=p['age_s'],available=p['available'],known_empty=p['known_empty'],status=p['status'],row_keys=p['row_keys'],zero_row_keys=None if z is None else z['row_keys'],rowmask_same=z is not None and p['row_keys']==z['row_keys'],digits=digits)
def static_trigger_audit():
    p=ROOT/'v25_response_identification/run_identification.py';tree=ast.parse(p.read_text(encoding='utf8'))
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)};f=funcs['public_trigger']
    attrs=sorted({n.attr for n in ast.walk(f) if isinstance(n,ast.Attribute)})
    prohibited={'object_position','object_velocity','target_id','object_geoms','true_support','mass','friction','qpos','qvel'}
    return dict(passed=not prohibited.intersection(attrs),attributes=attrs,source_sha256=sha(p),scope='current registered trigger static public-field inspection; truth-rich replay evaluation files are not controller inputs')
def execute(planpath):
    master=read(planpath);folder=ROOT/master['output'];marker=folder/'execution_complete.json'
    # Barrier: do not read branch outcomes before all five assignments terminal.
    if not marker.exists():return dict(complete=False,passed=False,pending=True,reason='NO_TERMINAL_MARKER',outcomes_read=False)
    done=read(marker)
    if done.get('execution_complete') is not True or done.get('mode')!='probe-five':return dict(complete=False,passed=False,pending=True,reason='EXECUTION_NOT_COMPLETE',outcomes_read=False,terminal_marker_sha256=sha(marker))
    checks=[];issues=[];inputs={str(planpath.relative_to(ROOT)):sha(planpath)}
    def ck(n,v,detail=None):
        checks.append(dict(name=n,passed=bool(v),details=detail))
        if not v:issues.append(n)
    def pin(p):inputs[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    for n in ('execution_complete.json','plan.json','manifest.json','assignment_ledger.json','runtime_failures.json','source_snapshot_manifest.json','origin_public.json'):pin(folder/n)
    plan=read(folder/'plan.json');manifest=read(folder/'manifest.json');ledger=read(folder/'assignment_ledger.json');identity=master['source_sha256']
    regpath=Path(plan['registration']);reg=read(regpath);pin(regpath)
    ck('registration_and_completion_source198_exact',len(identity)==198 and identity==plan['source_sha256']==done['source_sha256']==manifest['source_sha256']==reg['source_sha256'] and sha(regpath)==master['registration_sha256']==plan['registration_sha256'])
    ck('complete_five_assignment_accounting',master['assigned']==list(ARMS) and done['planned']==5 and len(ledger)==5 and len({r['key'] for r in ledger})==5 and {r['key'] for r in ledger}==set(ARMS) and done['rows']==ledger and done['all_assigned_accounted'] is True)
    ck('sources_environment_numerics_and_chain',done['sources_unchanged'] is True and done['numerical_unchanged'] is True and plan['numerical_identity']==done['numerical_identity']==master['numerical_identity']==manifest['numerical_identity']==reg['numerical_identity'] and plan['environment']==done['environment']==master['environment']==reg['environment'] and len(done['environment']['native_binaries_sha256'])==42 and done['plan_sha256']==sha(folder/'plan.json')==manifest['plan_sha256'] and done['manifest_sha256']==sha(folder/'manifest.json') and done['assignment_ledger_sha256']==sha(folder/'assignment_ledger.json'))
    snapshot=read(folder/'source_snapshot_manifest.json');snapshot_ok=set(snapshot['files'])==set(identity)
    for k,row in snapshot['files'].items():snapshot_ok &= row['sha256']==identity.get(k) and sha(folder/row['snapshot_path'])==identity.get(k) and sha(Path(row['original_path']))==identity.get(k)
    ck('all198_live_and_snapshot_bytes',snapshot_ok)
    for k,v in reg.get('dependencies',{}).items():ck('registered_dependency_'+k,sha(ROOT/k)==v)
    for k,v in reg.get('required_bridge_artifact_sha256',{}).items():ck('pinned_bridge_artifact_'+k,sha(ROOT/k)==v)
    bridgepin=reg['required_bridge_gate'];bridge=read(ROOT/bridgepin['path'])
    ck('actual_complete_remaining16_bridge_required',sha(ROOT/bridgepin['path'])==bridgepin['sha256'] and bridge.get('passed') is True and bridge.get('full_remaining_episode_compared') is True and bridge.get('source_sha256')==identity)
    expected=dict(mode='probe-five',method='ratio_effort',scenario='nominal',seed=619741,motion_mode='transport',split='robustness',origin_trigger='receiver_transfer')
    trigger=static_trigger_audit();origin=read(folder/'origin_public.json');t=origin.get('time_s')
    ck('fixed_context_and_public_trigger',master['assignment']==reg['assignment']==plan['assignment']==expected and trigger['passed'] and origin.get('trigger')=='receiver_transfer' and origin.get('next_PRE_OBSERVE_boundary') is True and origin.get('phase')=='TRANSFER' and origin.get('available') is True and abs(t-origin['trigger_read_time_s']-.005)<1e-9)
    # Normal unexposed/failed ledgers stay visible, never converted to zero response.
    bykey={r['key']:r for r in ledger};records={};forks=[]
    for arm in ARMS:
        row=bykey[arm];base=folder/'branches'/arm;record=dict(arm=arm,assignment=row,status=row['status'],missing_files=[]);records[arm]=record
        if not base.exists():continue
        names=('metadata.json','trace.npz','response_manifest.json','response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz','probe_decisions.jsonl.gz','opaque_fork_metadata.json')
        record['missing_files']=[n for n in names if not (base/n).exists()]
        for n in names:
            if (base/n).exists():pin(base/n)
        if record['missing_files']:continue
        meta=read(base/'metadata.json');fm=read(base/'opaque_fork_metadata.json');forks.append(fm)
        scene=meta['scenario'];scene=scene.get('name') if isinstance(scene,dict) else scene
        ck(arm+'_metadata_identity',scene=='nominal' and meta['seed']==619741 and meta['motion_mode']=='transport' and meta['method']=='ratio_effort' and meta['release_variant']=='canonical_joint' and meta['research_source_sha256']==identity and meta['response_identification']['probe']['arm']==arm and meta['response_identification']['probe']['origin_s']==t)
        rm=read(base/'response_manifest.json');streamnames=('response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz','probe_decisions.jsonl.gz')
        for n in streamnames:ck(arm+'_stream_pin_'+n,rm['streams'][n]['sha256']==sha(base/n))
        record['packets']=gz(base/streamnames[0]);record['receipts']=gz(base/streamnames[1]);record['intervals']=gz(base/streamnames[2]);record['probes']=gz(base/streamnames[3])
        with np.load(base/'trace.npz',allow_pickle=False) as z:record['trace']={k:z[k] for k in z.files}
        ck(arm+'_opaque_same_origin',fm['passed'] is True and abs(fm['time_s']-t)<1e-9 and fm['checkpoint_state_exposed_to_policy'] is False and all(c['passed'] for c in fm['checks']))
        times=record['trace']['time'];ck(arm+'_trace_coverage',bits(times,np.arange(len(times))*.005) and abs(times[-1]+.005-row['recorded_until_s'])<1e-8)
        receipt_by=index_time(record['receipts'],'issue_time_s');okay=True
        for p in record['probes']:
            tick=int(round(p['time_s']/.005));actual=p.get('actual_receipt');r=receipt_by.get(tick)
            okay &= r is not None and actual is not None
            if r is None or actual is None:continue
            u=np.asarray(actual['actual_reference21'],float);lo=np.asarray(p['actual_TASK_lower21']);hi=np.asarray(p['actual_TASK_upper21']);eps=64*np.finfo(float).eps*np.maximum(1,np.abs(u))
            okay &= bool(np.isfinite(u).all() and np.all(u>=lo-eps) and np.all(u<=hi+eps) and bits(u,np.asarray(r['executed_reference21'])) and actual['mode_generation']==r['mode_generation'])
            okay &= not p['nominal_task_certificate_applies_to_actual_reference'] or bits(u,np.asarray(p['nominal_proposal_reference21']))
        ck(arm+'_actual_all21_receipts_and_bounds',okay)
        record['probe_status_counts']={s:sum(p['status']==s for p in record['probes']) for s in sorted({p['status'] for p in record['probes']})}
    ck('all_assigned_runtime_failures_retained',read(folder/'runtime_failures.json')==done['runtime_failures'])
    ck('all_available_forks_same_opaque_integration',len({f['opaque_integration_sha256'] for f in forks})<=1)
    zero=records['zero'];ztrace=zero.get('trace');zpack=zero.get('packets',[]);zreceipts=zero.get('receipts',[])
    if ztrace is not None:
        obs=[json.loads(str(v)) for v in ztrace['observation_json']];before=[o for o in obs if abs(float(o['time'])-origin['trigger_read_time_s'])<1e-9]
        ck('saved_public_trigger_not_truth_selected',len(before)==1 and before[0].get('receiver_surface',{}).get('valid') is True)
    for arm,r in records.items():
        if 'trace' not in r:continue
        equal,fields=prefix_equal(r['trace'],ztrace,t);ck(arm+'_full_trace_preorigin_exact',equal,dict(fields=len(fields),failed=[k for k,v in fields.items() if not v]))
        for name,key in (('packets','read_time_s'),('receipts','issue_time_s')):
            a=[p for p in r[name] if p[key]<t-1e-9];b=[p for p in zero[name] if p[key]<t-1e-9]
            ck(arm+'_public_'+name+'_prefix_exact',len(a)==len(b) and all(canonical(x)==canonical(y) for x,y in zip(a,b)))
        r['actual_input_footprint']=footprint(r['receipts'],zreceipts,t)
        r['horizons']=[horizon_record(r['packets'],zpack,t,h) for h in HORIZONS]
        ck(arm+'_acquisition_read_availability',all(p['available_at_s']==p['read_time_s'] and abs(p['age_s']-(p['read_time_s']-p['acquisition_time_s']))<1e-9 and p['read_time_s']>=p['acquisition_time_s']-1e-9 for p in r['packets']))
        first=r['probes'][0] if r['probes'] else None
        if first:
            axis=None if arm=='zero' else 18 if arm.startswith('lift') else 20
            target=np.asarray(first['raw_probe_target21']);carrier=np.asarray(first['carrier_at_origin21']);delta=target-carrier
            r['origin_probe']=dict(axis=axis,requested_delta21=delta.tolist(),zero_amplitude=not np.any(delta),status=first['status'],exposed_reported=bykey[arm].get('exposed'))
            ck(arm+'_requested_axis_only_at_origin',np.all(delta==0) if axis is None else not np.any(np.delete(delta,axis)))
        for name in ('trace','packets','receipts','intervals','probes'):r.pop(name,None)
    return dict(schema='v25_first5_execution_independent_v1',complete=True,passed=all(c['passed'] for c in checks),checks=checks,issues=issues,assigned_count=5,physical_seed_clusters=1,production_source_sha256=identity,numerical_identity=master['numerical_identity'],environment=master['environment'],input_sha256=inputs,origin_public=origin,trigger_audit=trigger,records=[records[a] for a in ARMS],physics_steps=0,new_physical_runs=0,source_unchanged=True,full21_response_model_available=False,only_two_assigned_input_columns=True,zero_role='OFFLINE_PAIRED_COUNTERFACTUAL_NOT_ONLINE_FREE_EVOLUTION',scope='All five assigned short branches, opaque hash/prefix/log identity, actual all21 inputs and fixed acquisition horizons; not causal model admission, common-actor proof, full16 task success, derivative from ideal single-axis request, or independent five-sample evidence.')
def selftest():
    rows=[dict(issue_time_s=.1,executed_reference21=[0.]*21),dict(issue_time_s=.105,executed_reference21=[0.]*21)];z=json.loads(json.dumps(rows));rows[1]['executed_reference21'][2]=.1;rows[1]['executed_reference21'][18]=.01
    fp=footprint(rows,z,.1)
    tests=[dict(name='all21_footprint_keeps_incidental_coordinate',passed=fp['nonzero_axes21']==[2,18]),dict(name='missing_acquisition_remains_unknown',passed=horizon_record([],[],.1,.02)['available'] is False),dict(name='registered_trigger_static_public_fields',passed=static_trigger_audit()['passed'])]
    return dict(schema='v25_first5_reader_software_v1',passed=all(x['passed'] for x in tests),checks=tests,physics_steps=0,outcomes_read=False,checker_sha256=sha(__file__))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--execute',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError('Fresh evidence required')
    try:
        if args.self_test:result=selftest()
        elif args.execute:result=execute((ROOT/args.plan).resolve())
        else:result=dict(complete=False,pending=True,outcomes_read=False,reason='Explicit --execute only after root terminal notice')
    except Exception:result=dict(schema='v25_first5_execution_independent_v1',complete=False,passed=False,error=traceback.format_exc(),physics_steps=0)
    result['checker_sha256']=sha(__file__);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(dict(out=str(out),sha256=sha(out),passed=result.get('passed'),complete=result.get('complete'),issues=result.get('issues'),error=result.get('error')),ensure_ascii=False));raise SystemExit(0 if result.get('passed',result.get('pending',False)) else 1)

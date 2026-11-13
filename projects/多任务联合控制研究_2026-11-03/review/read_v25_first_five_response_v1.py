"""Read only the complete, registered V25 five-arm experiment. No model fit."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,gzip,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
ARMS=('zero','lift_plus','lift_minus','wrj1_plus','wrj1_minus')
HORIZONS=(.005,.010,.020,.040,.100,.200)
CONTRACT='review/V25首五支公开响应读数与解释合同_v1.md'
TOL=1e-9

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def records(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def check(ok,reason,issues):
    if not ok:issues.append(reason)
    return bool(ok)
def tick(t):return int(round(float(t)/.005))
def equal(a,b):return canonical(a)==canonical(b)
def verify_nested_pins(value,issues):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            path=(ROOT/value['path']).resolve()
            check(path.is_relative_to(ROOT.resolve()) and sha(path)==value['sha256'],'artifact_pin:'+value['path'],issues)
        for x in value.values():verify_nested_pins(x,issues)
    elif isinstance(value,list):
        for x in value:verify_nested_pins(x,issues)
def array(x,shape):
    a=np.asarray(x,float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid finite array '+str(shape))
    return a
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def terminal_barrier(plan,complete):
    """No branch stream may be opened until this pure barrier succeeds."""
    issues=[]
    check(plan.get('mode')=='probe-five' and plan.get('planned')==5 and plan.get('assigned')==list(ARMS),'plan_not_fixed_five',issues)
    check(complete.get('execution_complete') is True and complete.get('passed') is True,'five_execution_not_terminal_passed',issues)
    check(complete.get('mode')=='probe-five' and complete.get('planned')==complete.get('completed')==5 and complete.get('all_assigned_accounted') is True,'incomplete_assignment_counts',issues)
    rows=complete.get('rows',[])
    check(len(rows)==5 and {r.get('key') for r in rows}==set(ARMS),'missing_or_duplicate_arm',issues)
    check(all(r.get('status')=='short_window_recorded' and r.get('full_requested_window_recorded') is True for r in rows),'short_window_incomplete_or_failed',issues)
    check(complete.get('runtime_failures')==[] and complete.get('sources_unchanged') is True and complete.get('numerical_unchanged') is True,'runtime_source_numerical_failure',issues)
    for key in ('source_sha256','environment','numerical_identity'):
        check(complete.get(key)==plan.get(key),'completion_identity:'+key,issues)
    return not issues,issues


def select_label(packet,*,target_acq,recorded_until):
    if packet is None:return dict(available=False,reason='exact_acquisition_packet_missing')
    a=float(packet['acquisition_time_s']);r=float(packet['read_time_s']);v=float(packet['available_at_s'])
    if abs(a-target_acq)>TOL or abs(v-r)>TOL or a>v+TOL or v>recorded_until+TOL:
        return dict(available=False,reason='label_clock_or_public_availability_invalid')
    return dict(available=True,read_time_s=r,acquisition_time_s=a,available_at_s=v,
        age_s=packet['age_s'],status=packet['status'],known_empty=packet['known_empty'],
        row_keys=packet['row_keys'],digits=packet['digits'])


def common_function(origin,zero,arm):
    """Recover a common CONDITIONAL C row function; never complete a twist.

    Nullspace tests use one fixed IEEE/SVD rule, not a physical threshold.
    Common actor membership remains unknown even if the intersection is full.
    """
    if any(d is None or d.get('external_projection2_m_s') is None or d.get('conditional_C2x6') is None for d in (origin,zero,arm)):
        return dict(available=False,reason='conditional_common_function_missing_rows')
    Cs=[array(d['conditional_C2x6'],(2,6)) for d in (origin,zero,arm)]
    complements=[]
    for C in Cs:
        _,s,vh=np.linalg.svd(C,full_matrices=False)
        tol=128*np.finfo(float).eps*max(C.shape)*max(1.,float(s[0]))
        Q=vh[s>tol];complements.append(np.eye(6)-Q.T@Q)
    K=np.vstack(complements);_,s,vh=np.linalg.svd(K,full_matrices=False)
    tol=128*np.finfo(float).eps*max(K.shape)*max(1.,float(s[0]))
    L=vh[np.sum(s>tol):]
    if not len(L):return dict(available=False,reason='no_common_observed_function_in_origin_zero_arm_C_rowspaces',svd_tolerance=tol)
    weights=[np.linalg.lstsq(C.T,L.T,rcond=None)[0].T for C in Cs]
    residual=max(float(np.max(np.abs(W@C-L))) for W,C in zip(weights,Cs))
    if residual>tol:return dict(available=False,reason='common_function_not_representable_at_fixed_numeric_tolerance',row_residual=residual,svd_tolerance=tol)
    z=weights[1]@array(zero['external_projection2_m_s'],(2,));a=weights[2]@array(arm['external_projection2_m_s'],(2,))
    return dict(available=True,dimension=len(L),rows6=L.tolist(),zero_value=z.tolist(),arm_value=a.tolist(),
        difference=(a-z).tolist(),reconstruction_error=residual,svd_tolerance=tol,
        role='CONDITIONAL_COMMON_RIGID_VELOCITY_FIELD_FUNCTION_ONLY',
        actor_association='UNASSIGNED',same_actor_verified=False,
        interpretation='Common linear rows under fixed world m/rad/anchor convention; no rigid actor, full twist, dissipation or causal response certificate')


def paired_digit(origin,zero,arm):
    result=dict(available=False,role='OFFLINE_DIFFERENCE_OF_EACH_DEVICE_COORDINATE_OUTPUT',
        same_actor_verified=False,physical_fixed_plane_velocity_difference_certified=False)
    if zero is None or arm is None:
        result['reason']='missing_acquisition_label';return result
    if zero.get('external_projection2_m_s') is None or arm.get('external_projection2_m_s') is None:
        result.update(reason='one_or_both_digit_projections_unknown',zero_status=zero['status'],arm_status=arm['status']);return result
    z=array(zero['external_projection2_m_s'],(2,));a=array(arm['external_projection2_m_s'],(2,))
    rawz=zero.get('raw_relative_slip2_m_s');rawa=arm.get('raw_relative_slip2_m_s')
    result.update(available=True,device_external_coordinate_difference_m_s=(a-z).tolist(),
        raw_relative_slip_coordinate_difference_m_s=None if rawz is None or rawa is None else (array(rawa,(2,))-array(rawz,(2,))).tolist(),
        acquisition_basis_equal=equal(zero.get('basis_palm2x3'),arm.get('basis_palm2x3')),
        world_projection_equal=equal(zero.get('projection_world2x3'),arm.get('projection_world2x3')),
        pressure_shares_equal=equal(zero.get('pressure_shares'),arm.get('pressure_shares')),
        support_equal=equal(zero.get('patch_support'),arm.get('patch_support')),
        conditional_common_function=common_function(origin,zero,arm))
    return result


def observed_rank(matrix):
    X=np.asarray(matrix,float)
    if X.size==0:return dict(rows=0,columns=21,rank=None,singular_values=[],reason='no_matching_actual_receipt_differences')
    X=X.reshape(-1,21);s=np.linalg.svd(X,compute_uv=False)
    tol=max(X.shape)*np.finfo(float).eps*(float(s[0]) if len(s) else 0.)
    return dict(rows=len(X),columns=21,rank=int(np.sum(s>tol)),singular_values=s.tolist(),svd_tolerance=tol,
        nonzero_observed_columns=np.flatnonzero(np.any(X!=0.,axis=0)).tolist(),
        not_a_causal_or_full21_identification_admission=True)


def scan_branch(folder,origin,row,issues):
    meta=read(folder/'metadata.json');manifest=read(folder/'response_manifest.json');fork=read(folder/'opaque_fork_metadata.json')
    files={};clocks=[];selected={};receipt_window={};program=[];phases={};counts={};prefix={}
    targets={tick(origin+h) for h in (0.,)+HORIZONS}
    end=float(row['recorded_until_s'])
    for name in ('response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz','probe_decisions.jsonl.gz'):
        path=folder/name;files[name]=sha(path);d=manifest.get('streams',{}).get(name,{})
        check(d.get('sha256')==files[name],str(folder.name)+':stream_hash:'+name,issues)
        number=0;prefix_hash=hashlib.sha256();prefix_count=0
        for x in records(path):
            number+=1
            if name=='response_observations.jsonl.gz':
                a=float(x['acquisition_time_s']);r=float(x['read_time_s']);v=float(x['available_at_s'])
                check(abs(v-r)<=TOL and abs(x['age_s']-(r-a))<=TOL and a<=r+TOL,str(folder.name)+':observation_clock',issues)
                rowkeys=[tuple(k) for k in x['row_keys']];observed_digits={k[0] for k in rowkeys}
                expected={(d,a) for d in observed_digits for a in (0,1)}
                check([d['digit'] for d in x['digits']]==list(range(5)) and len(rowkeys)==len(set(rowkeys)) and set(rowkeys)==expected and observed_digits<=set(range(5)),str(folder.name)+':per_digit_unique_pair',issues)
                if a<=origin+TOL:prefix_hash.update(canonical(x)+b'\n');prefix_count+=1
                if tick(a) in targets and abs(a-tick(a)*.005)<=TOL:
                    old=selected.get(tick(a))
                    if old is None or v<float(old['available_at_s']):selected[tick(a)]=x
                clocks.append(r)
            elif name=='response_receipts.jsonl.gz':
                t=float(x['issue_time_s'])
                check(abs(float(x['available_at_s'])-t)<=TOL,str(folder.name)+':receipt_clock',issues)
                if t<origin-TOL:prefix_hash.update(canonical(x)+b'\n');prefix_count+=1
                if origin-TOL<=t<=end+TOL:
                    k=tick(t)
                    check(k not in receipt_window,str(folder.name)+':duplicate_actual_receipt',issues)
                    receipt_window[k]=x
            elif name=='probe_decisions.jsonl.gz':program.append(x)
        counts[name]=number
        check(d.get('records')==number,str(folder.name)+':stream_count:'+name,issues)
        if name in ('response_observations.jsonl.gz','response_receipts.jsonl.gz'):
            prefix[name]=dict(count=prefix_count,sha256=prefix_hash.hexdigest(),
                boundary='acquisition<=origin' if name.startswith('response_observations') else 'issue<origin')
    path=folder/'release_decisions.jsonl.gz';files[path.name]=sha(path)
    for x in records(path):
        t=float(x['time_s'])
        if origin-.015-TOL<=t<=end+TOL:phases[tick(t)]=x.get('phase')
    check(bool(clocks) and abs(max(clocks)+.005-end)<=TOL,str(folder.name)+':last_public_read_boundary',issues)
    check(fork.get('passed') is True and fork.get('physics_steps')==0 and fork.get('sensor_samples_added')==0 and abs(fork['time_s']-origin)<=TOL,str(folder.name)+':opaque_fork_origin',issues)
    fork_checks={c['name']:c for c in fork.get('checks',[])}
    check(bool(fork_checks) and all(c.get('passed') is True for c in fork_checks.values()),str(folder.name)+':opaque_copy_checks',issues)
    check(fork_checks.get('opaque_integration_exact_and_original_unchanged',{}).get('passed') is True,str(folder.name)+':base_integration_unchanged',issues)
    rng_checks=[n for n in fork_checks if n.startswith(('generator_state_','randomstate_state_'))]
    check(bool(rng_checks),str(folder.name)+':origin_rng_support_evidence_missing',issues)
    labels={}
    for h in (0.,)+HORIZONS:
        label=select_label(selected.get(tick(origin+h)),target_acq=origin+h,recorded_until=end)
        label['phase_at_acquisition_boundary']=phases.get(tick(origin+h))
        label['phase_at_public_delivery']=phases.get(tick(label['read_time_s'])) if label['available'] else None
        labels[str(h)]=label
    for name in ('metadata.json','trace.npz','response_manifest.json','opaque_fork_metadata.json'):
        files[name]=sha(folder/name)
    return dict(meta=meta,opaque=fork,origin_rng_checks=rng_checks,files=files,counts=counts,prefix=prefix,
        labels=labels,receipts=receipt_window,program=program,phases=phases)


def collect(planpath):
    p=read(planpath);folder=ROOT/p['output'];marker=folder/'execution_complete.json'
    if not marker.exists():return dict(complete=False,issues=['execution_complete_marker_missing'],physics_steps=0,assigned=list(ARMS))
    c=read(marker);ready,issues=terminal_barrier(p,c)
    if not ready:return dict(complete=False,issues=issues,physics_steps=0,assigned=list(ARMS),execution_summary=c)
    # Only after the all-five barrier are branch metadata and streams opened.
    from v25_response_identification.runtime import source_identity
    from v22_joint_task_fusion.numerics import numerical_identity
    from v13_a1.run_bridge import environment
    from v14_coord.design import reference_rates
    identity=source_identity();numeric=numerical_identity();env=environment()
    check(len(identity)==198 and identity==p['source_sha256'],'current198_source_identity',issues)
    check(env==p['environment'] and len(env['native_binaries_sha256'])==42 and numeric==p['numerical_identity'],'current42_numerical_identity',issues)
    regpath=ROOT/p['registration'];reg=read(regpath)
    check(sha(regpath)==p['registration_sha256'] and reg['assignment']==p['assignment'],'registration_assignment_pin',issues)
    verify_nested_pins(p['software_gates'],issues)
    for n,digest in p['dependencies'].items():check(sha(ROOT/n)==digest,'dependency_changed:'+n,issues)
    for n,digest in reg['required_bridge_artifact_sha256'].items():check(sha(ROOT/n)==digest,'bridge_input_changed:'+n,issues)
    check(sha(ROOT/reg['required_bridge_gate']['path'])==reg['required_bridge_gate']['sha256'],'actual_bridge_gate_pin',issues)
    rp=read(folder/'plan.json');manifest=read(folder/'manifest.json');snap=read(folder/'source_snapshot_manifest.json')
    check(rp['registration_sha256']==sha(regpath) and rp['assignment']==p['assignment'],'actual_run_assignment',issues)
    check(rp['source_sha256']==manifest['source_sha256']==identity and rp['numerical_identity']==manifest['numerical_identity']==numeric and rp['environment']==env,'actual_run_source_environment_identity',issues)
    check(c['plan_sha256']==sha(folder/'plan.json')==manifest['plan_sha256'] and c['manifest_sha256']==sha(folder/'manifest.json') and c['assignment_ledger_sha256']==sha(folder/'assignment_ledger.json'),'actual_completion_chain',issues)
    check(equal(read(folder/'assignment_ledger.json'),c['rows']) and read(folder/'runtime_failures.json')==[],'actual_assignment_ledger',issues)
    check(set(snap['files'])==set(identity),'all198_snapshot_keys',issues)
    check(manifest['source_snapshot_manifest_sha256']==sha(folder/'source_snapshot_manifest.json'),'snapshot_manifest_pin',issues)
    for k,x in snap['files'].items():
        path=(folder/x['snapshot_path']).resolve()
        check(path.is_relative_to(folder.resolve()) and x['sha256']==identity.get(k) and sha(path)==identity.get(k),'snapshot_changed:'+k,issues)
    origin_record=read(folder/'origin_public.json');origin=float(origin_record['time_s'])
    check(origin_record.get('available') is True and origin_record.get('trigger')=='receiver_transfer' and origin_record.get('next_PRE_OBSERVE_boundary') is True and abs(origin-float(origin_record['trigger_read_time_s'])-.005)<=TOL,'fixed_public_receiver2_origin',issues)
    if issues:
        return dict(complete=False,issues=list(dict.fromkeys(issues)),physics_steps=0,model_fits=0,
            assigned=list(ARMS),execution_rows=c['rows'],branch_streams_opened=0,
            reader_sha256=sha(__file__),plan_sha256=sha(planpath),contract_sha256=sha(ROOT/CONTRACT))
    rows={x['key']:x for x in c['rows']};branches={}
    for arm in ARMS:
        b=scan_branch(folder/'branches'/arm,origin,rows[arm],issues);m=b['meta'];branches[arm]=b
        scene=m.get('scenario');scene=scene.get('name') if isinstance(scene,dict) else scene
        check((scene,m.get('motion_mode'),m.get('seed'),m.get('method'))==('nominal','transport',619741,'ratio_effort'),arm+':actual_context',issues)
        check(m.get('research_source_sha256')==identity and m.get('numerical_identity')==numeric,arm+':source_numeric',issues)
        spec=m['response_identification']['probe']
        check(spec==dict(arm=arm,origin_s=origin,hold_s=.2,amplitude_fraction=.5),arm+':fixed_actual_probe_spec',issues)
    zero=branches['zero'];base_fingerprint=zero['opaque']['opaque_integration_sha256']
    for arm,b in branches.items():
        check(b['opaque']['opaque_integration_sha256']==base_fingerprint,arm+':common_opaque_origin_integration',issues)
        check(b['prefix']==zero['prefix'],arm+':same_origin_public_measurement_and_prior_receipts',issues)
        for key in ('config','noise','limits','budget','gains','task_policy','receiver_policy','release_variant'):
            check(equal(b['meta'].get(key),zero['meta'].get(key)),arm+':same_actual_input:'+key,issues)
    branch_records={};pairs=[];all_inputs=[]
    origin_digits=zero['labels']['0.0'].get('digits')
    for arm,b in branches.items():
        differences=[];input_rows=[]
        for k,event in sorted(b['receipts'].items()):
            if origin-TOL<=event['issue_time_s']<origin+.2-TOL and k in zero['receipts']:
                z=zero['receipts'][k];delta=array(event['executed_reference21'],(21,))-array(z['executed_reference21'],(21,))
                input_rows.append(dict(time_s=event['issue_time_s'],actual_executed_difference_from_zero21=delta.tolist(),
                    actual_reference_increment21=event['actual_reference_increment21'],mode=event['mode'],
                    mode_generation=event['mode_generation'],execution_changed=event['execution_changed'],stale_plan_replaced=event['stale_plan_replaced']))
                differences.append(delta)
                if arm!='zero':all_inputs.append(delta)
        X=np.asarray(differences).reshape(-1,21) if differences else np.empty((0,21))
        program_statuses=Counter(x['status'] for x in b['program'])
        branch_records[arm]=dict(assignment=rows[arm],input_sha256=b['files'],prefix=b['prefix'],stream_counts=b['counts'],
            origin_opaque_integration_sha256=b['opaque']['opaque_integration_sha256'],origin_rng_state_check_names=b['origin_rng_checks'],
            origin_rng_support_scope='Per-fork clone/source RNG equality checks plus same public pre-origin streams; raw final RNG states not persisted',
            program=b['program'],program_status_counts=dict(program_statuses),actual_receipt_input_rows=input_rows,
            first_actual_executed_difference_from_zero21=None if not len(X) else X[0].tolist(),
            max_abs_actual_executed_difference_from_zero21=None if not len(X) else np.max(np.abs(X),axis=0).tolist(),
            labels=b['labels'],unsupported=any('UNSUPPORTED' in x for x in program_statuses),
            no_unmeasured_input_columns_filled_with_response_zero=True)
        if arm=='zero':continue
        for h in HORIZONS:
            zl=zero['labels'][str(h)];al=b['labels'][str(h)];digits=[]
            for d in range(5):
                od=None if origin_digits is None else origin_digits[d]
                zd=None if not zl['available'] else zl['digits'][d]
                ad=None if not al['available'] else al['digits'][d]
                digits.append(dict(digit=d,**paired_digit(od,zd,ad)))
            pairs.append(dict(arm=arm,horizon_s=h,target_acquisition_time_s=origin+h,
                zero_available_at_s=zl.get('available_at_s'),arm_available_at_s=al.get('available_at_s'),
                unsupported_program=branch_records[arm]['unsupported'],
                no_local_response_fit_claim=True,digits=digits))
    scales=np.asarray(reference_rates('release'))*.005
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Original positive rate scales unavailable')
    matrix=np.asarray(all_inputs).reshape(-1,21) if all_inputs else np.empty((0,21))
    ranks=dict(raw_reference_units=observed_rank(matrix),original_5ms_rate_scaled=observed_rank(matrix/scales if len(matrix) else matrix),
        intended_tested_axes=[18,20],unmeasured_axes=[i for i in range(21) if i not in (18,20)],
        interpretation='Empirical actual program support/rank only. Endogenous fresh bounds and unsupported adaptive branch can create other columns; not independent direction interventions or full21 gains.')
    check(source_identity()==identity,'sources_changed_during_read',issues)
    return dict(schema='v25_first_five_readout_v1',complete=not issues,issues=list(dict.fromkeys(issues)),
        physics_steps=0,model_fits=0,source_sha256=identity,numerical_identity=numeric,environment=env,
        plan_sha256=sha(planpath),reader_sha256=sha(__file__),contract_sha256=sha(ROOT/CONTRACT),
        execution_complete_sha256=sha(marker),origin_public=origin_record,assigned=list(ARMS),
        branches=branch_records,paired_device_labels=pairs,actual_program_support=ranks,
        physical_seed_clusters=1,full21_model_admitted=False,force_safety_certified=False,
        original_full16_task_metrics_available=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='All five assigned arms; archived public measurements/actual receipts only for response labels; opaque/eval metadata solely audit. Zero future is offline paired oracle, never online model. No fit, selection, new score or strategy success claim.')


def markdown(result):
    lines=['# V25首五支公开响应只读统一读数','',f"完整读取：{result.get('complete')}；问题：{len(result.get('issues',[]))}。",'',
        '设备坐标差与条件共同函数分列；未认证actor，不填正常分量或未测19列。zero仅离线配对，未拟合模型。','']
    if not result.get('complete'):return '\n'.join(lines+[json.dumps(result.get('issues'),ensure_ascii=False)])+'\n'
    lines+=['| 分支 | 实际终态 | 未支持 | 窗内匹配回执 | |Δu|最大值 |','|---|---|---|---:|---|']
    for arm,b in result['branches'].items():
        x=b['max_abs_actual_executed_difference_from_zero21'];mx=None if x is None else max(x)
        lines.append(f"| {arm} | {b['assignment'].get('terminal_probe_status')} | {b['unsupported']} | {len(b['actual_receipt_input_rows'])} | {mx} |")
    lines+=['',f"实际原速率归一输入rank：{result['actual_program_support']['original_5ms_rate_scaled']['rank']}，仅描述本次程序支持，不认证全21模型。",'','逐horizon、逐指原值/基面/份额/mask/phase及缺测原因完整保存在JSON；不筛选幸存接触。']
    return '\n'.join(lines)+'\n'


def self_test(out):
    checks=[]
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    p=dict(mode='probe-five',planned=5,assigned=list(ARMS),source_sha256={},environment={},numerical_identity={})
    c=dict(execution_complete=True,passed=True,mode='probe-five',planned=5,completed=5,all_assigned_accounted=True,
        rows=[dict(key=a,status='short_window_recorded',full_requested_window_recorded=True) for a in ARMS],
        runtime_failures=[],sources_unchanged=True,numerical_unchanged=True,source_sha256={},environment={},numerical_identity={})
    ck('all-five barrier accepts fixed complete synthetic assignment',terminal_barrier(p,c)[0])
    bad=json.loads(json.dumps(c));bad['rows'].pop();ck('missing branch stops before branch stream reads',not terminal_barrier(p,bad)[0])
    bad=json.loads(json.dumps(c));bad['execution_complete']=False;ck('partial marker is not readable result cohort',not terminal_barrier(p,bad)[0])
    ck('missing exact acquisition is unknown not nearby/future interpolation',not select_label(None,target_acq=1.,recorded_until=2.)['available'])
    x=dict(acquisition_time_s=1.,read_time_s=1.01,available_at_s=1.01,age_s=.01,status='AVAILABLE',known_empty=False,row_keys=[],digits=[])
    ck('label unavailable before delivery even if acquisition exists',not select_label(x,target_acq=1.,recorded_until=1.005)['available'] and select_label(x,target_acq=1.,recorded_until=1.01)['available'])
    C=np.c_[np.eye(2,3),np.zeros((2,3))];d=dict(status='OBSERVED_PROJECTION',external_projection2_m_s=[1.,2.],conditional_C2x6=C.tolist())
    e=dict(d,external_projection2_m_s=[2.,4.]);f=common_function(d,d,e)
    ck('common row function recovered without completing unknown twist',f['available'] and f['dimension']==2 and not f['same_actor_verified'])
    a=dict(d,conditional_C2x6=np.c_[np.zeros((2,3)),np.eye(2,3)].tolist())
    ck('disjoint observable functions remain unknown',not common_function(d,d,a)['available'])
    u=dict(d,external_projection2_m_s=None,status='UNKNOWN_PROJECTION')
    ck('missing digit does not become zero effect',not paired_digit(d,d,u)['available'])
    q=dict(d,raw_relative_slip2_m_s=[0.,0.],basis_palm2x3=np.eye(2,3).tolist(),projection_world2x3=np.eye(2,3).tolist(),pressure_shares=[1.],patch_support=[])
    r=dict(q,basis_palm2x3=[[0.,1.,0.],[1.,0.,0.]])
    z=paired_digit(q,q,r)
    ck('basis change preserved and device difference never certified physical',z['available'] and not z['acquisition_basis_equal'] and not z['physical_fixed_plane_velocity_difference_certified'])
    X=np.zeros((4,21));X[0,18]=1.;X[1,20]=1.
    rank=observed_rank(X);ck('two observed columns do not become full21 gain matrix',rank['rank']==2 and rank['nonzero_observed_columns']==[18,20] and rank['not_a_causal_or_full21_identification_admission'])
    write(out,dict(passed=all(c['passed'] for c in checks),checks=checks,physics_steps=0,model_fits=0,new_results_read=0,
        reader_sha256=sha(__file__),contract_sha256=sha(ROOT/CONTRACT),scope='Finite synthetic barrier/acquisition/coordinate/rowspace/rank checks only'))
    print(json.dumps(dict(passed=all(c['passed'] for c in checks),checks=len(checks),out=str(out)),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--execute',action='store_true');g.add_argument('--self-test',action='store_true')
    ap.add_argument('--plan',default='review/v25_probe_five_plan_v1.json');ap.add_argument('--out',required=True);ap.add_argument('--md')
    args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError('Preserve earlier attempts; fresh readout required')
    if args.self_test:self_test(out)
    else:
        if args.md and (ROOT/args.md).exists():raise FileExistsError('Fresh Markdown output required')
        try:result=collect(ROOT/args.plan)
        except Exception:
            result=dict(complete=False,issues=['reader_or_input_validation_exception'],
                reader_error=traceback.format_exc(),physics_steps=0,model_fits=0,
                reader_sha256=sha(__file__),contract_sha256=sha(ROOT/CONTRACT))
        write(out,result)
        if args.md:(ROOT/args.md).write_text(markdown(result),encoding='utf-8')
        print(json.dumps(dict(complete=result['complete'],issues=result['issues'],out=str(out)),ensure_ascii=False))
        raise SystemExit(0 if result['complete'] else 1)

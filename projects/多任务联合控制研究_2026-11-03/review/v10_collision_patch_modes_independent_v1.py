"""Frozen hold/collision 20 accepted plans: patch averages and offline surface motion.
No controller, servo predictor, simulation step, or live producer is executed.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys, types, json, hashlib, importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/development_v10_hold_v1'; SRC=RUN/'sources'
PARENT=ROOT/'review/v10_hold20_mechanisms_v1/readout.json'
TRACE=RUN/'runs/collision__joint__610212/trace.npz'
OUT=ROOT/'review/v10_collision_patch_modes_independent_v1.json'
if OUT.exists():raise FileExistsError(OUT)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,value,detail=None):
    checks.append(dict(name=name,passed=bool(value),detail=detail))
    if not value:raise AssertionError(name+': '+str(detail))
complete=read(RUN/'execution_complete.json')
check('terminal_hold_batch',complete['execution_complete'] and complete['trials']==20 and complete['sources_unchanged'])
parent=read(PARENT); pair=next(x for x in parent['pairs'] if x['scenario']=='collision')
plans=pair['mechanisms']['accepted_plans'];check('twenty_accepted_plans',len(plans)==20)
check('trace_identity_matches_parent_plan_inventory',sha(TRACE)==pair['mechanisms']['trace_sha256'])
manifest=read(RUN/'source_snapshot_manifest.json')['files']
used=['v10/controller.py','v10/robot_contact_space.py','skin_geometry.py','foundation/foundation_config.py','foundation/hand_plant.py']
used+=sorted(k for k in manifest if k.startswith('foundation/assets/'))
source_hashes={}
for k in used:
    digest=sha(SRC/k);check('snapshot_hash:'+k,digest==manifest[k]['sha256']);source_hashes[k]=digest
with np.load(TRACE,allow_pickle=False) as z:
    t=z['time']; qpos=z['qpos']; observations=[json.loads(str(x)) for x in z['observation_json']]
    debug=[json.loads(str(x)) for x in z['debug_json']]; raw=[json.loads(str(x)) for x in z['raw_patches_json']]
    metadata=json.loads(str(z['metadata'])); rawpressure=z['skin_normal']

# Frozen model compilation and position kinematics only. Refuse every Python dynamics entry.
import mujoco
dynamics_calls={k:0 for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_inverseSkip')}
def refuse(name):
    def block(*a,**kw):
        dynamics_calls[name]+=1
        raise RuntimeError('Dynamics prohibited in this offline review: '+name)
    return block
for k in dynamics_calls:
    if hasattr(mujoco,k):setattr(mujoco,k,refuse(k))
sys.dont_write_bytecode=True
sys.modules['research_imports']=types.ModuleType('research_imports')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
configmod=module('foundation_config',SRC/'foundation/foundation_config.py')
plantmod=module('offline_frozen_hand_plant',SRC/'foundation/hand_plant.py')
geometrymod=module('offline_frozen_robot_contact_space',SRC/'v10/robot_contact_space.py')
model,xmlhash=plantmod._model(configmod.FoundationConfig(**metadata['config']))
check('compiled_frozen_model_xml_identity',xmlhash==metadata['model_sha256'],xmlhash)
geometry=geometrymod.RobotContactSpace(model,metadata['raw_joint_names'])
check('same_public_robot_axis_order',list(geometry.axes25)==parent['robot_axes25'])
fkdata=mujoco.MjData(model)
def full_fk(index):
    mujoco.mj_resetData(model,fkdata);fkdata.qpos[:]=qpos[index];mujoco.mj_kinematics(model,fkdata)
    return fkdata.xpos.copy(),fkdata.xmat.reshape(-1,3,3).copy()
def index_at(clock):
    i=int(np.argmin(abs(t-clock)));check('record_clock:'+format(clock,'.9f'),abs(t[i]-clock)<1e-9)
    return i
def sample(index):
    f=observations[index]
    return geometrymod.RobotJointSample.from_public_frame(types.SimpleNamespace(**f),geometry.raw_joint_names,lift_velocity_m_s=f['lift_velocity_m_s'])
def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
def rotation_angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def stats(values):
    a=np.asarray(values,float)
    return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max())) if a.size else None
def ratio(numerator,denominator):return None if denominator<1e-15 else float(numerator/denominator)
def categories(patches,digit):
    out={}
    for p in patches:
        if p['digit']!=digit:continue
        label=('target' if p['is_target'] else 'other_external')+':'+str(p['external_geom'])
        x=out.setdefault(label,dict(count=0,pressure_N=0.));x['count']+=1;x['pressure_N']+=float(p['normal'])
    return out

records=[]
for plan in plans:
    now=plan['start_time_s'];i=index_at(now);j=index_at(now+.02);f=observations[i];d=debug[i]
    check('accepted_debug_at_start:'+str(now),d.get('status')=='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP')
    check('context_identity:'+str(now),d['servo_selection']['context_id']==plan['context_id'])
    current=sample(i);end=sample(j)
    check('recorded_actual20_public_q_exact:'+str(now),np.array_equal(end.q25,np.asarray(plan['actual_robot25_at_20ms'])))
    predicted=geometrymod.RobotJointSample(now+.02,geometry.raw_joint_names,np.asarray(plan['predicted_robot25'])[:24],plan['predicted_robot25'][24])
    xp0,R0=full_fk(i);xp1,R1=full_fk(j)
    # All material groups actually requested in this decision, not all digit contacts.
    risk=np.asarray(d['material']['raw_local_excess_evidence_N'])
    budget=d['material']['raw_budget_excess_evidence_N']
    digits=sorted({int(p['digit']) for p in f['skin_patches'] if p['normal']>0 and (risk[int(p['digit'])]>0 or budget is not None and budget>0)})
    groups=[]
    for digit in digits:
        patches=[p for p in f['skin_patches'] if p['digit']==digit and p['normal']>0 and p['age']<=.12]
        total=sum(p['normal'] for p in patches);rows=[];normals=[];weights=[];details=[];used_raw=set()
        for k,p in enumerate(patches):
            patch=geometrymod.PublicSkinPatch.from_public(p);projection=geometry.linearize(current,patch,maximum_age_s=.12)
            check('usable_public_patch:'+str((now,digit,k)),projection.available)
            n=projection.acquisition_normal_world;w=p['normal']/total
            kp=geometry.point_kinematics(predicted,patch);ka=geometry.point_kinematics(end,patch);kc=projection.kinematics
            predicted_n=float(n@(kp.point_world-kc.point_world));actual_n=float(n@(ka.point_world-kc.point_world))
            a=index_at(p['sample_time']);xpa,Ra=full_fk(a)
            check('public_acquisition_clock_not_current:'+str((now,digit,k)),abs(now-p['sample_time']-.01)<1e-9)
            # Match at the acquisition packet only; known link + normal + within-digit pressure share.
            palm=int(model.body('rh_palm').id);Rpalma=Ra[palm]
            candidates=[];raw_digit_total=sum(max(0.,x['normal']) for x in raw[a] if x['digit']==digit)
            for ri,r in enumerate(raw[a]):
                if r['digit']!=digit or int(model.geom_bodyid[r['hand_geom']])!=p['robot_body']:continue
                normal_world=Rpalma@np.asarray(r['normal_dir'])
                share=float(r['normal'])/raw_digit_total
                if angle(n,normal_world)<1e-4 and abs(share-w)<1e-8:candidates.append(ri)
            matched=len(candidates)==1 and candidates[0] not in used_raw
            offline=None
            if matched:
                ri=candidates[0];used_raw.add(ri);r=raw[a][ri];ebody=int(model.geom_bodyid[r['external_geom']])
                # Body-fixed material point from the old raw contact, not an asserted persistent contact ID.
                extlocal=Ra[ebody].T@(np.asarray(r['world_position'])-xpa[ebody])
                ex0=xp0[ebody]+R0[ebody]@extlocal;ex1=xp1[ebody]+R1[ebody]@extlocal
                extdelta=ex1-ex0;external_n=float(n@extdelta)
                offline=dict(match='unique acquisition link_normal_pressure_share',is_target=bool(r['is_target']),
                    external_geom=int(r['external_geom']),external_geom_name=model.geom(r['external_geom']).name,
                    external_body_name=model.body(ebody).name,hand_geom=int(r['hand_geom']),
                    acquisition_pressure_N=float(r['normal']),normal_angle_match_deg=angle(n,Rpalma@np.asarray(r['normal_dir'])),
                    external_body_rotation_20ms_deg=rotation_angle(R1[ebody],R0[ebody]),
                    external_body_rotation_since_acquisition_deg=rotation_angle(R0[ebody],Ra[ebody]),
                    advected_external_material_point_delta_world_m=extdelta.tolist(),external_old_normal_delta_m=external_n,
                    relative_robot_minus_advected_external_old_normal_delta_m=actual_n-external_n,
                    external_motion_scope='offline only; old body-fixed material point, not contact persistence/current gap/pressure prediction')
            else:
                offline=dict(match='ambiguous_or_missing',candidate_indices=candidates,
                    acquisition_digit_categories=categories(raw[a],digit))
            rows.append(projection.normal_jacobian25);normals.append(n);weights.append(w)
            details.append(dict(public_patch_index=k,robot_body=int(p['robot_body']),robot_link=model.body(p['robot_body']).name,
                public_pressure_N=float(p['normal']),weight=w,sample_time_s=p['sample_time'],age_s=p['age'],
                normal_world=n.tolist(),link_point_m=p['link_point'],predicted_point_delta_world_m=(kp.point_world-kc.point_world).tolist(),
                actual_point_delta_world_m=(ka.point_world-kc.point_world).tolist(),predicted_old_normal_delta_m=predicted_n,
                actual_old_normal_delta_m=actual_n,point_prediction_error_m=float(np.linalg.norm(ka.point_world-kp.point_world)),
                offline_evaluation=offline))
        w=np.asarray(weights);n=np.asarray(normals);row=np.asarray(rows);meanrow=w@row
        pn=np.array([x['predicted_old_normal_delta_m'] for x in details]);an=np.array([x['actual_old_normal_delta_m'] for x in details])
        angles=[angle(n[a],n[b]) for a in range(len(n)) for b in range(a+1,len(n))]
        average_predicted=float(w@pn);average_actual=float(w@an)
        g=dict(digit=digit,patch_count=len(patches),public_pressure_sum_N=total,
            pairwise_normal_angles_deg=angles,normal_resultant_ratio=float(np.linalg.norm(w@n)),
            angular_row24_resultant_ratio=ratio(np.linalg.norm(meanrow[:24]),sum(w*np.linalg.norm(row[:,:24],axis=1))),
            lift_row_resultant_ratio=ratio(abs(meanrow[24]),sum(w*abs(row[:,24]))),
            mean_predicted_old_normal_delta_m=average_predicted,mean_actual_old_normal_delta_m=average_actual,
            predicted_directional_cancellation_ratio=ratio(abs(average_predicted),float(w@abs(pn))),
            actual_directional_cancellation_ratio=ratio(abs(average_actual),float(w@abs(an))),
            predicted_patch_inward_count=int(np.count_nonzero(pn>1e-10)),actual_patch_inward_count=int(np.count_nonzero(an>1e-10)),
            predicted_mean_noninward_but_some_patch_inward=bool(average_predicted<=1e-10 and np.any(pn>1e-10)),
            actual_mean_noninward_but_some_patch_inward=bool(average_actual<=1e-10 and np.any(an>1e-10)),
            raw_current_digit_categories=categories(raw[i],digit),raw_end20_digit_categories=categories(raw[j],digit),patches=details)
        groups.append(g)
    check('weighted_predicted_exact_reproduces_frozen_debug:'+str(now),np.allclose([g['mean_predicted_old_normal_delta_m'] for g in groups],d['exact_predicted_robot_normal_delta_m'],rtol=0,atol=1e-12))
    records.append(dict(start_time_s=now,context_id=plan['context_id'],public_sample_time_s=f['sample_time'],
        public_pressure5_N=f['normal'],actual_raw_pressure5_N=rawpressure[i].tolist(),actual_raw_pressure5_at20ms_N=rawpressure[j].tolist(),groups=groups))
allgroups=[g for r in records for g in r['groups']];allpatches=[p for g in allgroups for p in g['patches']]
matched=[p for p in allpatches if p['offline_evaluation']['match'].startswith('unique')]
summary=dict(plan_count=len(records),group_count=len(allgroups),patch_count=len(allpatches),uniquely_matched_acquisition_patches=len(matched),
    context_ids=sorted(set(r['context_id'] for r in records)),digits=sorted(set(g['digit'] for g in allgroups)),
    patch_count_per_group=stats([g['patch_count'] for g in allgroups]),
    normal_resultant_ratio=stats([g['normal_resultant_ratio'] for g in allgroups]),
    maximum_pairwise_normal_angle_deg=stats([max(g['pairwise_normal_angles_deg'],default=0.) for g in allgroups]),
    angular_row24_resultant_ratio=stats([g['angular_row24_resultant_ratio'] for g in allgroups]),
    lift_row_resultant_ratio=stats([g['lift_row_resultant_ratio'] for g in allgroups if g['lift_row_resultant_ratio'] is not None]),
    predicted_mean_noninward_but_some_patch_inward_groups=sum(g['predicted_mean_noninward_but_some_patch_inward'] for g in allgroups),
    actual_mean_noninward_but_some_patch_inward_groups=sum(g['actual_mean_noninward_but_some_patch_inward'] for g in allgroups),
    predicted_inward_patch_max_m=max(p['predicted_old_normal_delta_m'] for p in allpatches),
    actual_inward_patch_max_m=max(p['actual_old_normal_delta_m'] for p in allpatches),
    groups_containing_target_and_other_external=sum(len({p['offline_evaluation'].get('is_target') for p in g['patches']})>1 for g in allgroups),
    external_body_rotation_20ms_deg=stats([p['offline_evaluation']['external_body_rotation_20ms_deg'] for p in matched]),
    external_old_normal_delta_m=stats([p['offline_evaluation']['external_old_normal_delta_m'] for p in matched]),
    relative_old_normal_delta_m=stats([p['offline_evaluation']['relative_robot_minus_advected_external_old_normal_delta_m'] for p in matched]))
check('zero_dynamics_calls',sum(dynamics_calls.values())==0,dynamics_calls)
for k,h in source_hashes.items():check('snapshot_still_identical:'+k,sha(SRC/k)==h)
result=dict(schema='v10_collision_patch_modes_independent_v1',created_utc=datetime.now(timezone.utc).isoformat(),
    status='descriptive_frozen_hold_collision20_only',scope=dict(batch='hold',scenario='collision',method='joint',seed=610212,
        accepted_plan_scope='all20 accepted plan starts and exact20ms endpoints; no other scene or live transport',
        producer_imports='snapshot model loader and robot contact FK only; no plant/session/controller constructor',
        simulation_steps=0,model_compile_count=1,physics_or_frozen_source_writes=False,
        stage_A_servo_and_four_quarters='not recomputed; parent inventory used only for qpred/actual20 and plan list'),
    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [PARENT,TRACE,RUN/'execution_complete.json',RUN/'source_snapshot_manifest.json']},
    source_sha256=source_hashes,compiled_model_xml_sha256=xmlhash,dynamics_guard_calls=dynamics_calls,
    signs=dict(positive_old_normal_delta='robot point displacement along old acquired normal; controller treats as inward',
        negative_old_normal_delta='withdrawal relative to old acquired plane',
        actual_relative_proxy='robot point minus advected external body-fixed old material point; offline only'),
    summary=summary,records=records,checks=checks,limitations=[
        'No actual baseline future counterfactual is reconstructed; exact_model_geometry_cost_before is null for every accepted parent plan.',
        'Per-digit pressure-weighted mean is neither per-patch guarantee nor pressure response. Separate24 angular and lift row ratios avoid mixing units.',
        'Public geometry is delayed by10ms; raw identity matching uses the same acquisition packet, never the current raw packet.',
        'External truth fields appear only in offline evaluation after public plan selection; no controller input/proposal/eligibility uses them.',
        'Advected external material points are not persistent contact IDs. Separation, sliding, recontact, normal changes and compliance prevent interpreting this proxy as measured current gap or pressure.',
        'Source/trace checks bind this finite descriptive scope; neither model admission nor material-force safety certification is claimed.'])
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
md=['# V10 collision：平均接触方向与外物运动离线分解','',
    '范围：冻结 hold 批次 collision/joint/610212 的全部20个已接受计划。只编译一次冻结模型、进行机器人与离线外物位置运动学；所有动力学接口设拒绝守卫，0次物理步。',
    '', '|量|结果|','|---|---|']
for key,value in summary.items():md.append('|'+key+'|'+json.dumps(value,ensure_ascii=False)+'|')
md+=['','正值表示机器人点沿旧采样法向的运动。平均非向内不保证每个patch非向内。类别由同一采样时刻原始记录、机器人link和法向/压力比例唯一匹配后才离线解释。',
    '', '外物项只跟踪旧接触点对应的外物body局部材料点，不宣称接触始终存在或当前真实gap已知。它不进入控制器。没有重算parent的A项或比较真实baseline未来动作；不能将平均FK cost当成真实力下降。',
    '', 'JSON SHA256 `'+sha(OUT)+'`。']
OUT.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),checks=len(checks),summary=summary),ensure_ascii=False))

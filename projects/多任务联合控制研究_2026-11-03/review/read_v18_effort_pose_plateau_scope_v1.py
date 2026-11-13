"""Read archived finite-value flat native maps/pose surrogate, no model or QP.

This is source+execution mechanism description, not a component ablation or
an explanation of floor outcomes. Reviewer authored the effort_task component.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,gzip
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'review/v18_effort_pose_plateau_scope_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def main():
    if any(STEM.with_suffix(s).exists() for s in ('.json','.md')):raise FileExistsError(STEM)
    plan=read(ROOT/'review/v18_small_plan_v1.json');core=read(ROOT/'review/v18_small_execution_independent_v2.json')
    if not core['passed'] or not core['all_assigned_checked'] or not all(read(ROOT/s['output']/'execution_complete.json')['execution_complete'] is True for s in plan['stages']):raise RuntimeError('All4 terminal + successful execution gate required')
    rows=[];inputs={}
    for stage in plan['stages'][1:]:
        run=next((ROOT/stage['output']/'runs').iterdir());old=ROOT/stage['original_same_condition'];meta=read(run/'metadata.json');oldmeta=read(old/'metadata.json')
        for p in (run/'trace.npz',run/'release_decisions.jsonl.gz',old/'trace.npz'):inputs[str(p)]=sha(p)
        with gzip.open(run/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(s) for s in f]
        count=Counter();per_axis=np.zeros(21,int);examples=[];tangent_ages=[];leak=[];tangent_unknown=Counter()
        for q in records:
            count['decisions']+=1;count['accepted']+=q['effort_goal_realized'];current=np.asarray(q['current_reference21']);pose=np.asarray(q['posture_reference21'])
            count['pose_reference_diff_finger_axis_ticks']+=int(np.count_nonzero(pose[:18]!=current[:18]))
            domain=q['domain'];p=q['qp_problem'];native=q['native']
            if domain['status']=='KNOWN_DISJOINT_NATIVE_EFFORT_BOX':
                pl,pu=np.asarray(domain['effort_lower21']),np.asarray(domain['effort_upper21']);lo,hi=np.asarray(domain['reference_lower21']),np.asarray(domain['reference_upper21'])
                fixed_ref=lo==hi;finite_flat=(pl==pu)&~fixed_ref
                count['flat_because_reference_fixed_axis_ticks']+=int(fixed_ref.sum())
                count['flat_on_nonzero_reference_interval_axis_ticks']+=int(finite_flat.sum())
                per_axis+=finite_flat.astype(int)
                target=np.asarray(p['extra_task_rows']['target']) if p is not None else None
                for axis in range(18):
                    if not finite_flat[axis] or pose[axis]==current[axis]:continue
                    count['finger_flat_despite_requested_pose_change']+=1
                    invisible=target is not None and target[axis]==0.
                    count['finger_flat_pose_effort_target_zero']+=bool(invisible)
                    if q['effort_goal_realized'] and native is not None:
                        same=np.asarray(q['proposed_reference21'])[axis]==current[axis]
                        chosen=native['diagnostics']['axis_selections'][axis]['family']
                        clipped=native['diagnostics']['native_force_clipped21'][axis]
                        count['accepted_flat_finger_ref_held']+=bool(same)
                        count['accepted_flat_current_family']+=chosen=='current'
                        count['accepted_flat_force_clipped']+=bool(clipped)
                        if len(examples)<6:examples.append(dict(time_s=q['time_s'],axis=axis,current_reference=current[axis],posture_reference=pose[axis],effort_interval=[pl[axis],pu[axis]],reference_interval=[lo[axis],hi[axis]],soft_effort_target=target[axis],native_family=chosen,force_clipped=clipped,proposed_reference=q['proposed_reference21'][axis]))
            diag=q['contact_intents']
            for record in diag['tangent_records']:
                count['active_digit_tangent_records']+=1;tangent_ages.append(record['age_s']);leak.extend(record['patch_normal_leakage_N'])
            for item in diag['tangent_unavailable']:tangent_unknown[str(item['reason'])]+=1
        archive=[]
        for path in (run,old):
            with np.load(path/'trace.npz',allow_pickle=False) as z:
                archive.append(dict(time=z['time'],reference=z['executed_reference21'],observation=z['observation_json']))
        samples=[];summaries=[]
        for data,metadata in zip(archive,(meta,oldmeta)):
            t=data['time'];pick=np.flatnonzero(t>=12.);entry=json.loads(str(data['observation'][2400]));refs=np.vstack((entry['robot_reference21'],data['reference'][pick]))
            goal=np.asarray(metadata['release_clearance_goal']['reference18'])
            summaries.append(dict(finger_reference_TV18_rad=np.sum(abs(np.diff(refs[:,:18],axis=0)),axis=0),
                finger_reference_net18_rad=refs[-1,:18]-refs[0,:18],final_reference18=refs[-1,:18],final_reference_error_to_declared_clearance_goal18_rad=refs[-1,:18]-goal,
                initial_reference_before_action21=entry['robot_reference21']))
        for query in (12.,12.4,12.8,13.2,14.,16.):
            i=round(query/.005);nf=json.loads(str(archive[0]['observation'][i]));of=json.loads(str(archive[1]['observation'][i]))
            nq,oq=np.asarray(nf['robot_position25']),np.asarray(of['robot_position25'])
            samples.append(dict(time_s=query,finger_reference_difference18_rad=archive[0]['reference'][i,:18]-archive[1]['reference'][i,:18],
                angular_robot_encoder_difference24_rad=nq[:24]-oq[:24],lift_encoder_difference_m=float(nq[24]-oq[24]),
                new_slip_acquisition_s=nf['sample_time'],old_slip_acquisition_s=of['sample_time'],
                new_public_device_slip2_m_s=nf['slip_local'],old_public_device_slip2_m_s=of['slip_local'],
                new_public_slip_valid5=nf['slip_valid5'],old_public_slip_valid5=of['slip_valid5'],
                slip_not_current_per_patch_velocity=True))
        rows.append(dict(stage=stage['name'],run=str(run),old_v17_same_context=str(old),counts=dict(count),finite_flat_by_axis21=per_axis,
            first_examples=examples,new_v18_reference_summary=summaries[0],old_v17_reference_summary=summaries[1],samples=samples,
            tangent_record_age_range_s=None if not tangent_ages else [min(tangent_ages),max(tangent_ages)],
            intent_patch_normal_leakage_abs_max_N=None if not leak else max(abs(np.asarray(leak))),
            tangent_unavailable_reasons=dict(tangent_unknown)))
    report=dict(schema='v18_effort_pose_plateau_scope_v1',role='effort_task author descriptive source/execution diagnosis, not component causal identification',
        core_execution_audit_sha256=sha(ROOT/'review/v18_small_execution_independent_v2.json'),plan_sha256=sha(ROOT/'review/v18_small_plan_v1.json'),input_sha256=inputs,rows=rows,
        source_sha256={k:v for k,v in plan['source_sha256'].items() if k.startswith('v18_effort_transfer/')},
        source_facts=['At fixed q/v, flat native effort over a nonzero reference interval can erase a posture-reference difference in the effort-only soft row.',
                      'Realizer effort ties prioritize nearest current reference. A singleton imposed by hard reference/TASK authority is reported separately, not called a saturation plateau.',
                      'The module does not command geometric pose directly: posture is converted to native(posture)-fixedp0; delta regularizer remains an additional compromise.',
                      'Tangential effort is one delayed device-plane intention per digit, pressure-shared onto current robot point Jacobians; not measured actual contact effort or current per-patch tangential velocity.'],
        limitations=['Observed paired postrelease reference/encoder differences identify the changed package, not isolated soft-pose/plateau causation of floor outcomes.',
                     'A nonzero device-plane normal leakage value is an intention projection, not measured pressure or actual tangential motion/work.',
                     'No FK/contact/dynamics replay; public device slip remains delayed mixed-actor aggregate. No target-specific current slip, current actor twist, per-patch work or global contact geometry certificate.',
                     'Fixedq/v effort/torque and geometric motion must be coordinated explicitly in a later registered design; lower impact or legal effort realization alone is not multitask improvement.'],
        physics_steps=0,model_constructions=0,optimizer_replays=0)
    STEM.with_suffix('.json').write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V18努力软姿态与平台：实际执行边界','', '全4执行门已通过。本文由effort_task作者整理，是源码和记录描述，不是单独机制消融或放置安全认证。', '',
        '| 条件 | 非零reference区间内finite-flat轴次 | 手指欲改姿态但flat轴次 | 其中soft effort目标0 | 接受后仍保持current参考 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        c=r['counts'];lines.append(f"| {r['stage']} | {c.get('flat_on_nonzero_reference_interval_axis_ticks',0)} | {c.get('finger_flat_despite_requested_pose_change',0)} | {c.get('finger_flat_pose_effort_target_zero',0)} | {c.get('accepted_flat_finger_ref_held',0)} |")
    lines += ['', '硬权限/TASK将reference区间固定成单点的情况另计，不冒充forceclip平台。若表中实际平台数为0，平台只是源代码层的潜在局限，不能当本轮退步的实证原因。', '',
        '努力软姿态比较的是当前q/v下native(posture)−p0，未直接约束机器人几何或OPEN参考距离；相同努力的参考方向可能对该代价不可见，且有限逆映射的努力并列会优先closest current。与此同时原δ正则仍保留，普通非平台分支也会和posture目标折中。', '',
        'JSON逐例保存与已配对V17的12–16秒参考TV/net、6个预定时点的18参考及24角encoder差异，lift米单列。命令与encoder差异不能直接证明逐patch切向移动或导致落地；未运行FK或接触重建。', '',
        '公开slip只在延迟采集时的每指设备平面中可得，并混合全部actor。记录的tau/normal-leakage是努力意图，不是实际接触力、当前逐patch切向速度或耗散功。因而“原生努力精确兑现/接收冲击较低”不能替代几何清离、滑移和稳定驻留的联合验收。', '',
        f'[完整数值与来源]({STEM.name}.json)。']
    STEM.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(STEM.with_suffix('.json')),sha256=sha(STEM.with_suffix('.json')),md_sha256=sha(STEM.with_suffix('.md')),counts=[dict(stage=r['stage'],counts=r['counts']) for r in rows]),ensure_ascii=False))


if __name__=='__main__':main()

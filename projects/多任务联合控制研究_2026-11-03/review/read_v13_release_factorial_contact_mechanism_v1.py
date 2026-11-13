"""Offline geometry/contact interpretation of four completed fixed-input replays."""
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v13_release_factorial_contact_mechanism_v1.json'
MD=ROOT/'review/v13_release_factorial_contact_mechanism_v1.md'
BRIDGE=ROOT/'output/v13_release_factorial_bridge_v1'
OTHER=ROOT/'output/v13_release_factorial_other3_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x
def rotate(q,v):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);v=np.broadcast_to(v,q[...,1:].shape)
    return v+2.*np.cross(q[...,1:],np.cross(q[...,1:],v)+q[...,:1]*v)
def capsule_box_gap(center,axis,half,radius,lo,hi):
    """Distance(segment,box)-radius; negative is overlap proxy, not solver dist."""
    knots=[-half,half]
    for k in range(3):
        if axis[k]!=0:
            knots.extend(float((b-center[k])/axis[k]) for b in (lo[k],hi[k]) if -half<(b-center[k])/axis[k]<half)
    knots=sorted(set(knots));candidates=list(knots)
    for l,u in zip(knots,knots[1:]):
        point=center+.5*(l+u)*axis;where=np.where(point<lo,lo,np.where(point>hi,hi,point));active=(point<lo)|(point>hi)
        den=float(axis[active]@axis[active])
        if den:candidates.append(float(np.clip(-axis[active]@(center[active]-where[active])/den,l,u)))
    distance=min(float(np.linalg.norm(np.maximum(lo-(center+t*axis),0)+np.maximum((center+t*axis)-hi,0))) for t in candidates)
    return distance-radius

def main():
    complete=[load(p/'execution_complete.json') for p in (BRIDGE,OTHER)]
    assert all(x['execution_complete'] and all(v['complete'] for v in x['results']) for x in complete)
    folders={'both':BRIDGE/'both',**{v:OTHER/v for v in ('lift_only','fingers_only','neither')}}
    metas={v:load(p/'metadata.json') for v,p in folders.items()};assert len({m['model_sha256'] for m in metas.values()})==1
    sys.path.insert(0,str(ROOT));import research_imports,mujoco
    forbidden={};saved={}
    for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_jac','mj_collision','mj_contactForce'):
        if hasattr(mujoco,n):
            saved[n]=getattr(mujoco,n);forbidden[n]=0
            def deny(*a,_n=n,**k):forbidden[_n]+=1;raise AssertionError('forbidden '+_n)
            setattr(mujoco,n,deny)
    from foundation_config import FoundationConfig
    import hand_plant
    ctor=hand_plant.HandPlant.__init__;constructors=[]
    def no_plant(*a,**k):constructors.append(True);raise AssertionError('no plant construction')
    hand_plant.HandPlant.__init__=no_plant
    try:
        model,model_sha=hand_plant._model(FoundationConfig(**metas['both']['config']));assert model_sha==metas['both']['model_sha256']
        j=model.joint('object_free').id;qa=int(model.jnt_qposadr[j]);va=int(model.jnt_dofadr[j]);liftqa=int(model.jnt_qposadr[model.joint('transport_lift').id])
        obj=model.geom('object_geom').id;radius,half=model.geom_size[obj,:2];localaxis=rotate(model.geom_quat[obj],[0.,0.,1.]);assert np.array_equal(model.geom_pos[obj],np.zeros(3))
        boxes={}
        for name in ('support','support_post'):
            g=model.geom(name).id;assert np.array_equal(model.geom_quat[g],[1.,0.,0.,0.])
            boxes[name]=(model.geom_pos[g]-model.geom_size[g],model.geom_pos[g]+model.geom_size[g])
        floor_z=float(model.geom_pos[model.geom('floor').id,2]);table_lo,table_hi=boxes['support']
        rows={};entry=[];inputs={};classified={}
        for variant,folder in folders.items():
            result=load(folder/'result.json');diag=load(folder/'diagnostics.json');assert result['complete']
            assert sha(folder/'cached_contact_packets.jsonl.gz')==result['cached_contact_packet_sha256']
            inputs[variant]=dict(folder=str(folder),result_sha256=sha(folder/'result.json'),metadata_sha256=sha(folder/'metadata.json'),trace_sha256_declared=result['trace_sha256'],cached_packet_sha256=result['cached_contact_packet_sha256'])
            with np.load(folder/'trace.npz',allow_pickle=False) as z:
                t=z['substep_state_end_time'];q=z['substep_qpos'];vel=z['substep_qvel'];skin=z['substep_skin_normal'];support=z['substep_support'];penetration=z['substep_world_penetration'];allhand=z['substep_all_hand_object'];ctrl=z['substep_actual_ctrl']
                assert len(t)==len(z['substep_time']) and np.max(abs(t-(z['substep_time']+z['substep_duration'])))<1e-8
                center=q[:,qa:qa+3];axis=rotate(q[:,qa+3:qa+7],localaxis);bottom=center[:,2]-radius-half*abs(axis[:,2]);omega=np.linalg.norm(vel[:,va+3:va+6],axis=1)
                release=np.flatnonzero(t>=12.-1e-8);lf=release[np.argmax(skin[release,4])];pen=release[np.argmax(penetration[release])];force=release[np.argmax(support[release])]
                onset=release[support[release]>1e-8];entry_idx=int(np.argmin(abs(t-12.)));entry.append((q[entry_idx].tobytes(),vel[entry_idx].tobytes(),ctrl[entry_idx].tobytes()))
                events={'entry':entry_idx,'LF_peak':lf,'world_penetration_peak':pen,'external_support_peak':force}
                if len(onset):events['first_positive_external_support']=int(onset[0])
                for qt in (12.18,12.229,12.6,13.,13.937,14.2,14.609,14.6215,16.):events[f't{qt:g}']=int(np.argmin(abs(t-qt)))
                samples={}
                for name,k in events.items():
                    gaps={n:capsule_box_gap(center[k],axis[k],half,radius,lo,hi) for n,(lo,hi) in boxes.items()}
                    samples[name]=dict(end_s=float(t[k]),cached_interval_start_s=float(z['substep_time'][k]),LF_skin_N=float(skin[k,4]),hand_object_N=float(allhand[k]),external_support_aggregate_N=float(support[k]),cached_world_penetration_m=float(penetration[k]),
                        object_center_end_m=center[k],object_quaternion_end=q[k,qa+3:qa+7],object_translation_velocity_end_m_s=vel[k,va:va+3],object_angular_speed_end_rad_s=float(omega[k]),capsule_axis_end=axis[k],capsule_bottom_floor_gap_end_m=float(bottom[k]-floor_z),capsule_box_gap_end_m=gaps,
                        whole_capsule_right_of_table_x=bool(center[k,0]-radius-half*abs(axis[k,0])>table_hi[0]),robot_lift_position_end_m=float(q[k,liftqa]))
                # Only hand-vs-external contacts exist in this cached packet stream.
                selected_packets={};event_times={name:float(t[k]) for name,k in events.items()}
                with gzip.open(folder/'cached_contact_packets.jsonl.gz','rt',encoding='utf-8') as stream:
                    for line in stream:
                        packet=json.loads(line)
                        names=[n for n,et in event_times.items() if abs(packet['end_s']-et)<1e-8]
                        if not names:continue
                        patches=[]
                        for p in packet['patches']:
                            p=dict(p);g=int(p['hand_geom']);e=int(p['external_geom']);p['hand_body_offline']=model.body(int(model.geom_bodyid[g])).name;p['hand_geom_type_offline']=int(model.geom_type[g]);p['external_name_offline']=model.geom(e).name
                            patches.append(p);classified[str(g)]=dict(body=p['hand_body_offline'],type=p['hand_geom_type_offline'])
                        for name in names:selected_packets[name]=dict(start_s=packet['start_s'],end_s=packet['end_s'],patches=patches)
                rows[variant]=dict(recorded_diagnostic_summary=diag,events=plain(samples),selected_cached_contact_packets=plain(selected_packets),
                    contact_query_scope='Cached solve-stage force/position/frame; microq/v are post-integration end. No forward/FK/contact recomputation. Table/floor contacts are absent from packet list; external_support is aggregate and world component assignment uses separate end-pose geometry proximity.',
                    treatment={'source_lift_trajectory_enabled':variant in ('both','lift_only'),'source_all18_finger_trajectory_enabled':variant in ('both','fingers_only'),'inactive_channel':'reference frozen at incoming12s entry; not qpos or force held directly'})
        assert all(x==entry[0] for x in entry)
        assert not any(forbidden.values()) and not constructors
        model_info=dict(xml_sha256=model_sha,object_free_qposadr=qa,object_free_dofadr=va,lift_qposadr=liftqa,capsule_radius_m=float(radius),capsule_segment_half_length_m=float(half),capsule_local_axis=localaxis,
                        table_box_lo_m=table_lo,table_box_hi_m=table_hi,floor_plane_z_m=floor_z,classified_hand_geoms=classified,model_compile_count_this_reader=1,prior_exploratory_compile_count=1)
    finally:
        hand_plant.HandPlant.__init__=ctor
        for n,fn in saved.items():setattr(mujoco,n,fn)
    graph=[
      dict(event='release_requested',must=['cancel holding-force/support goals immediately','invalidate prior holding plans from actual21 entry'],may_not=['wait for a placement gate before activating release','regrip through a secondary return task']),
      dict(task='unload_and_clear_hand',primary='Continue bounded robot release progress using current public tactile/robot state and last-contact history; absent sampled contact means clearance unknown, not support confirmed.',secondary='Return/lower lift is coordinated and subordinate to contact-clearance task, not an automatic concurrent priority.'),
      dict(optional_task='safe_placement',preconditions=['Explicit user/task receiver objective','Shared public receiver-support sensing or declared receiver geometry with uncertainty; existing offline support truth is unavailable online'],order=['approach/receive support while holding is still requested','transfer load with measured support evidence','unload/clear hand','secondary robot return'],override='An immediate release request still cancels holding and bypasses waiting; do not silently call an uncontrolled drop a successful placement.'),
      dict(acceptance='All outcome components retained',metrics=['all hand force caps/dose','object-world impact/support force and penetration','clearance with raw/coarse clocks separated','object velocity/spin and workspace retention','task/return tracking and deadline'],weights='No posthoc utility weights or success-loss tolerances are invented.')]
    report=plain(dict(schema='v13_release_factorial_contact_mechanism_v1',role='independent_offline_contact_geometry_causal_interpretation',terminal_gates=True,input_provenance=inputs,entry_micro_qv_ctrl_exact=True,model_info=model_info,variants=rows,task_graph_constraints=graph,
        clock_contract='Cached integration force START+duration=END. New q/v recorded at integration END; cached contact geometry/force is not recomputed END geometry. Treat substep co-occurrence with this0.5ms phase qualification.',
        conclusions=[
          'Both: transient LF-metacarpal/object contact while object has external support; fixed simultaneous trajectories create a force/rotation/contact-transition interaction. Not evidence of false native effort arithmetic.',
          'Lift_only: frozen gripping references plus descent bring supported object into persistent hand/external-support loading; no low-force clearance.',
          'Fingers_only: hand clearance can occur, but object later leaves the support and hits the floor;12.8mm max world penetration occurs at floor, not tabletop. Lower hand force alone is not comprehensive safety.',
          'Neither: held references keep the object in hand with no external support and no clearance. Logical release phase alone is not physical unloading.'],
        limitations=['Fixed-input factors are source recorded lift/all18finger channels, not idealized generic descent/opening laws or isolated single joints. All other commands/environment retained.',
          'One capsule/nominal/hold/617901 entry; treatment effects are local input counterfactuals, not a new feedback strategy or population safety proof.',
          'is_target, object free pose/velocity, geometry labels and external-support truth are used only in this offline report, never proposed as online inputs.',
          'Packet stream has no object-table/floor contact points; component assignment is geometry-supported inference, not invented per-contact normals/forces.',
          'Parent/foundation own full source/prefix/RNG/full metrics audit. This reader checks one common microentry and selected contact episodes, not the full108 closure or all16s forceguard.'],
        physics_steps=0,robot_FK_calls=0,forbidden_pipeline_calls=forbidden,plant_constructors=0,reader_sha256=sha(Path(__file__))))
    assert not OUT.exists() and not MD.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    b=rows['both'];l=rows['lift_only'];f=rows['fingers_only'];n=rows['neither'];floor=f['events']['world_penetration_peak'];bp=b['events']['LF_peak'];lfpacket=b['selected_cached_contact_packets']['LF_peak']['patches']
    lines=[]
    for name,v in rows.items():
        d=v['recorded_diagnostic_summary'];x=d['release'];c=d['low_force_clearance'];lines.append(f"| {name} | {x['peak_N']:.6f} | {x['local_excess_Ns']:.8f} | {str(c['raw_integration_end_first_by_deadline'])} | {x['max_penetration_m']*1000:.3f} |")
    text=f"""# 固定输入四分支的接触链解释

最重要的修正：`fingers_only`的12.8mm最大世界穿透发生在物体离开支撑台后落到地板，不能写成安全落台，也不能只凭较低手指力称它全面更优。`both`短LF峰则出现在物体同时受小指掌骨与外界支持的接触转换中；它不是末节胶囊本身的峰。

四分支已完整终态，来自同一个12s入口、同一原始输入。这里另核一个共同micro q/v/ctrl入口逐字节相同。完整来源、前缀、RNG和全指标由其他独审负责；本报告只解释几何机制。数值表为已存诊断，未重算全部门槛。

| 固定输入分支 | 释放段全指峰 N | 局部超限 Ns | raw-end低力清离首次时刻 s | 最大world穿透 mm |
|---|---:|---:|---:|---:|
"""+'\n'.join(lines)+f"""

这里“清离”仍是既定0.2N/0.2s低力接触判据，不是几何间隙或永久清离证书。`fingers_only`200Hz首次时刻13.92s与raw-end13.937s应分列，不能混用。

## 确定的接触/运动链

* **both**：12.229s的LF峰{bp['LF_skin_N']:.6f}N与缓存外界支持{bp['external_support_aggregate_N']:.6f}N并存；唯一手部斑块为hand_geom47 / `rh_lfmetacarpal`盒几何对object_geom65，位置约[.02277,−.03638,.36399]m，穿透约14.2μm。它是小指掌骨接触，而非末节。对象姿态变化与台面接触阶段重叠，下降与打开同时执行后出现短时重新接触；数据支持接触链耦合，但不能据此指定某一个LF关节或某个TH参数为单因。
* **lift_only**：所有18指参考冻结在入口而升降沿原记录下降，物体在手部约束下接触支撑台，之后支持力约20N级且手部不清离。它显示“下降”在未解除抓握约束时会把负载传成手—物—台内力；不是下降自动完成安全放置。
* **fingers_only**：升降参考保持入口，全部18指沿原打开/拇指路径运动。可以降低手部局部峰并最终低力清离，但物体随后离开支撑台水平范围，转动并落地。最大world穿透在{floor['end_s']:.4f}s，物体中心约{np.round(floor['object_center_end_m'],5).tolist()}m，vz≈{floor['object_translation_velocity_end_m_s'][2]:.3f}m/s；台面x范围[.025,.165]m、顶面z=.337m，地板z=0，物体此时已在台面右侧。几何与支持脉冲共同指向地板冲击，而非台面12.8mm穿透。
* **neither**：冻结升降与18指参考，仍维持夹持；外界支持为0，未出现低力清离。仅把生命周期标为RELEASING或把逻辑holding目标撤掉，并不等于原生参考已物理卸载。

## 时间与几何证据的边界

新增微状态是每次积分后的q/v；力、contact.pos/frame和slip来自该子步求解缓存，没有再forward/FK。因此不能把END姿态重算的间隙与缓存接触穿透强行逐数相等，也不能把±0.5ms差异说成新的因果先后。

接触包只包含手部对外接触，并没有对象—台面/地板的逐接触点。`support`是对象对非手部的聚合力。报告用离线物体free pose、静态支撑盒/地板几何及解析胶囊间隙辨别哪一外部表面相容；这是有证据的离线推断，未虚构桌面法向、支撑点或在线传感器。静态模型XML SHA与运行metadata一致；本reader编译1次（先前探查另1次），所有step/forward/FK/contactForce/plant均为0。

## 下一整体生命周期的偏序约束

1. **释放请求立即生效**：立即取消holding力/承载目标、废弃旧holding计划，以actual21进入释放。不能等“先放好”门控才开始响应用户release。
2. **卸载/清手是当前主任务；升降返回是次任务**：手指路径进度、实际原生努力和升降应共同协调。不能让独立下降默认优先于仍有接触的手部清离，也不能把单关节“向open”误当所有patch都向外。
3. **若要求safe placement，必须另列任务和公共观测**：在用户仍要求holding时可先接近明确接收面，再验证接收支持、转移负载、卸载清手、最后返回。当前公共流没有离线对象—台面支持真值；仅靠手内力降低/当前无patch不能区分已支撑与正在下落。新增接收支持传感或声明几何必须同等给baseline。即时release始终可以抢占上述可选放置流程，不能恢复holding或偷偷把drop判成place成功。
4. **保留综合代价，但不事后加权**：手指峰/Ns、物体外界冲击/穿透、物体速度/旋转/工作区留存、清离时延与机器人返回任务要并列。fingers_only的低手指力不能洗掉落地代价，both更小的world穿透也不能洗掉LF过力。

本2×2改变的是固定记录的升降与全部18指输入，非新反馈策略；保留的腕参考和环境不变。它支持局部输入因果比较，不支持跨物体、独立总体安全或某个关节/增益的单独因果结论。所有对象姿态、is_target和台面身份只用于离线解释，不能直接喂下一控制器。

结构化证据：[JSON](v13_release_factorial_contact_mechanism_v1.json)。
"""
    MD.write_text(text,encoding='utf-8');print(json.dumps(dict(json=str(OUT),json_sha256=sha(OUT),markdown=str(MD),markdown_sha256=sha(MD),physics_steps=0),ensure_ascii=False))

if __name__=='__main__':main()

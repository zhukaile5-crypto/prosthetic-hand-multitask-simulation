"""Nine archived runs: new descriptive delivery diagnostic, no score revision."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/native_v13_v14_release_delivery_diagnostic_v1.json'
MD=ROOT/'review/native_v13_v14_release_delivery_diagnostic_v1.md'
BATCHES={'native':'development_v12_native_pilot_v1','V13':'development_v13_joint_pilot_v1','V14':'development_v14_coherent_pilot_v1'}
METHODS=('ratio_effort','fixed_effort','fusion_effort')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    manifests={};meta={};results={};paths={};provenance={}
    for cohort,batch in BATCHES.items():
        folder=ROOT/'output'/batch;done=load(folder/'execution_complete.json');assert done['execution_complete'] and done['sources_unchanged'] and done['trials']==done['planned']==3 and not done['runtime_failures']
        assert sha(folder/'trials.json')==done['trials_sha256'];manifests[cohort]=load(folder/'manifest.json')
        provenance[cohort]=dict(terminal_sha256=sha(folder/'execution_complete.json'),trials_sha256=done['trials_sha256'])
        for method in METHODS:
            key=cohort+'/'+method;p=folder/'runs'/f'nominal__{method}__617901';paths[key]=p;meta[key]=load(p/'metadata.json');results[key]=load(p/'result.json')
            scenario=meta[key]['scenario'];scenario=scenario['name'] if isinstance(scenario,dict) else scenario
            assert meta[key]['completed'] and scenario=='nominal' and meta[key]['seed']==617901 and meta[key]['motion_mode']=='hold'
    first=meta['native/fusion_effort'];config_text=json.dumps(first['config'],sort_keys=True)
    assert all(json.dumps(m['config'],sort_keys=True)==config_text for m in meta.values())
    assert len({m['model_sha256'] for m in meta.values()})==1
    # Reuse only pure analytical quaternion/capsule-distance functions, not its main/physics guard.
    helper_path=ROOT/'review/read_v13_release_factorial_contact_mechanism_v1.py';spec=importlib.util.spec_from_file_location('delivery_geometry_helpers',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    sys.path.insert(0,str(ROOT));import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    saved={};forbidden={}
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_jac','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            saved[name]=getattr(mujoco,name);forbidden[name]=0
            def deny(*a,_name=name,**k):forbidden[_name]+=1;raise AssertionError('forbidden '+_name)
            setattr(mujoco,name,deny)
    init=hand_plant.HandPlant.__init__;plant_calls=[]
    def no_plant(*a,**k):plant_calls.append(True);raise AssertionError('no plant constructor')
    hand_plant.HandPlant.__init__=no_plant
    try:
        model,model_sha=hand_plant._model(FoundationConfig(**first['config']));assert model_sha==first['model_sha256']
        object_joint=model.joint('object_free').id;qa=int(model.jnt_qposadr[object_joint]);va=int(model.jnt_dofadr[object_joint]);geom=model.geom('object_geom').id
        radius,half=model.geom_size[geom,:2];local_axis=helper.rotate(model.geom_quat[geom],[0.,0.,1.]);receiver=model.geom('support').id
        lo=model.geom_pos[receiver]-model.geom_size[receiver];hi=model.geom_pos[receiver]+model.geom_size[receiver];floor_z=float(model.geom_pos[model.geom('floor').id,2]);rows={}
        for key,p in paths.items():
            result=results[key];m=meta[key];cohort,method=key.split('/');r=result['research_metrics']
            with np.load(p/'trace.npz',allow_pickle=False) as z:
                t=z['time'];position=z['position'];q=z['qpos'];v=z['qvel'];rel_rotation=z['rotation'];allhand=z['all_hand_object'];support200=z['support']
                assert np.array_equal(position,q[:,qa:qa+3]);quat=q[:,qa+3:qa+7];axis=helper.rotate(quat,local_axis);bottom=position[:,2]-radius-half*abs(axis[:,2]);rotation_error=float(np.max(abs(np.einsum('tji,tjk->tik',rel_rotation,rel_rotation)-np.eye(3))))
                extents=radius+half*abs(axis);extent_margin=np.minimum(position-extents-lo,hi-(position+extents));center_margin=np.minimum(position-lo,hi-position)
                st=z['substep_time'];dt=z['substep_duration'];se=st+dt;ss=z['substep_support'];wp=z['substep_world_penetration'];release=np.flatnonzero(st>=12.-1e-9)
                peak=release[np.argmax(ss[release])];pen=release[np.argmax(wp[release])]
                def sample(i):
                    return dict(time_s=float(t[i]),position_world_m=position[i],free_qpos_world_m=q[i,qa:qa+3],free_quaternion_world=quat[i],object_axis_world=axis[i],free_translation_velocity_world_m_s=v[i,va:va+3],free_angular_speed_rad_s=float(np.linalg.norm(v[i,va+3:va+6])),
                        capsule_floor_gap_m=float(bottom[i]-floor_z),capsule_receiver_gap_m=helper.capsule_box_gap(position[i],axis[i],half,radius,lo,hi),center_xy_edge_margin_m=float(center_margin[i,:2].min()),full_capsule_AABB_xy_margin_m=float(extent_margin[i,:2].min()),hand_object_N=float(allhand[i]),external_support_aggregate_N=float(support200[i]))
                def raw_event(i):
                    j=int(np.argmin(abs(t-se[i])));return dict(start_s=float(st[i]),end_s=float(se[i]),external_support_N=float(ss[i]),cached_world_penetration_m=float(wp[i]),nearest200Hz_pose=sample(j),pose_minus_raw_end_s=float(t[j]-se[i]))
                row=dict(cohort=cohort,method=method,metadata_sha256=sha(p/'metadata.json'),result_sha256=sha(p/'result.json'),declared_trace_sha256=result['trace_sha256'],model_sha256=m['model_sha256'],
                    position_equals_free_qpos_all3201=True,quaternion_norm_max_error=float(np.max(abs(np.linalg.norm(quat,axis=1)-1.))),relative_rotation_orthogonality_max_error=rotation_error,
                    final=sample(len(t)-1),release_external_support_peak=raw_event(peak),release_world_penetration_peak=raw_event(pen),
                    original_scores_unchanged=dict(task_success=result['task_success'],mission_success=r['mission_success'],full_motion_mission_success=r['full_motion_mission_success'],release=result['release_success'],release_V3=r['release'],force_guard=r['safety']['guard_pass']),
                    stored_static_quality=dict(eligible=r['static_quality']['eligible'],mean_N=r['static_quality']['mean_actual_finger_force_N'],raw200_TV_N_s=r['static_quality']['raw_200Hz']['finger_TV_rate_N_s']))
                if key=='native/fusion_effort':
                    floor_samples=np.flatnonzero((t>=12.)&(bottom<=floor_z));assert len(floor_samples);j=int(floor_samples[0]);assert j>0
                    # Coarse pose brackets floor geometry. Raw force onset is located inside a stated wider matching neighborhood, not claimed as a fresh end contact query.
                    neighborhood=np.flatnonzero((se>=t[j-1]-.005)&(se<=t[j]+.005)&(ss>1e-8));first_raw=int(neighborhood[0]) if len(neighborhood) else None
                    snapshots={str(x):sample(int(np.argmin(abs(t-x)))) for x in (12.,12.5,13.,13.5,14.,14.5,15.,15.5,16.)}
                    row['old_fusion_floor_timeline']=dict(first_sampled_floor_overlap_bracket_s=[float(t[j-1]),float(t[j])],last_before=sample(j-1),first_after=sample(j),
                      first_positive_external_support_in_floor_neighborhood=None if first_raw is None else raw_event(first_raw),snapshots=snapshots,
                      interpretation='Floor association is supported by world pose and receiver separation; raw support is not geometrically labelled.200Hz pose only brackets0.5ms raw force events; this is not a micro-pose replay.')
                rows[key]=plain(row)
        assert not any(forbidden.values()) and not plant_calls
        model_info=plain(dict(model_xml_sha256=model_sha,all9_same_config_and_model=True,object_free_qposadr=qa,object_free_dofadr=va,receiver_geom_id=receiver,receiver_name='support',receiver_box_lo=lo,receiver_box_hi=hi,floor_geom_id=int(model.geom('floor').id),floor_plane_z=floor_z,capsule_radius=radius,capsule_halfsegment=half,body_local_object_geom_quaternion=model.geom_quat[geom]))
    finally:
        hand_plant.HandPlant.__init__=init
        for name,fn in saved.items():setattr(mujoco,name,fn)
    definitions=dict(position='HandPlant.truth position is data.xpos[object], world object-body origin; equals object_free qpos XYZ here.',rotation='trace.rotation is palm_R.T @ object_R, a PALM-RELATIVE rotation; use freejoint quaternion for world capsule orientation.',velocity='trace.relative_velocity/angular_velocity are palm-relative; freejoint translational qvel gives world translation, angular-speed norm is coordinate independent.',support='HandPlant._contacts support sums target-object contact normal against ANY nonhand geom. It can include floor, receiver support and post; it is not receiver-identified delivery evidence.',
        success='Retention failure search ends at actual release/12s. Release success is qualified physical exposure plus <=.2N hand-object force dwell before release deadline; mission adds completion/handoff/pre-release failure/warnings. Neither receiver retention nor postrelease floor impact/workspace is a delivery criterion.',
        source_locations=[dict(file=str(ROOT/'output/development_v12_native_pilot_v1/sources/foundation/hand_plant.py'),lines=[262,296,314,323]),dict(file=str(ROOT/'output/development_v12_native_pilot_v1/sources/evaluation.py'),lines=[242,264]),dict(file=str(ROOT/'output/development_v12_native_pilot_v1/sources/foundation/evaluate_runs.py'),lines=[125,137])])
    report=plain(dict(schema='native_v13_v14_release_delivery_diagnostic_v1',role='independent_new_postrelease_geometry_diagnostic_not_rescoring',terminal_cohorts=provenance,model_info=model_info,rows=rows,definitions=definitions,
        conclusions=['Old native fusion ends on the floor and still rolling; world position/free-qpos agreement and static geometry rule out body/frame confusion.','V13 fusion is at receiver/table height with a much smaller final translational/angular speed.','V14 fusion is at table height but near right edge and still moving outward; centre margin is not a certified stability margin and object overhang is not itself an original failure rule.','Original task/release scores remain true under their frozen clearance semantics; adding delivery diagnosis does not retrospectively alter them.','Old fusion static-force advantages remain local static-force evidence; they do not establish comprehensive lifecycle or delivery superiority.'],
        limits=['One seen nominal/hold/capsule/seed617901 context; not9 independent physical contexts or population-level delivery reliability.','No new safety/placement pass threshold or utility weight. Endpoint/edge/overhang/velocity are descriptive new diagnostics.','No continuation beyond16s; outward velocity near edge is a risk observation, not an asserted future fall.','Raw support/penetration are cached solve-stage outputs; matched pose is200Hz boundary state with explicitly reported time difference, not exact same-query contact geometry.','Full source/RNG/prefix and V3 metric audits belong to prior independent evidence; this diagnostic reads the relevant9 archives and preserves stored scores.'],
        geometry_helper_sha256=sha(helper_path),reader_sha256=sha(Path(__file__)),physics_steps=0,static_model_compile_count=1,robot_FK_calls=0,forbidden_pipeline_calls=forbidden,production_or_score_edits=0))
    assert not OUT.exists() and not MD.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    table=[]
    for key,row in rows.items():
        f=row['final'];xyz=f['position_world_m'];table.append(f"| {key} | {xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f} | {f['free_translation_velocity_world_m_s'][0]:.6f} | {f['free_angular_speed_rad_s']:.6f} | {f['center_xy_edge_margin_m']*1000:.3f} |")
    old=rows['native/fusion_effort'];timeline=old['old_fusion_floor_timeline'];v14=rows['V14/fusion_effort']['final'];fp=old['release_external_support_peak'];pen=old['release_world_penetration_peak']
    text=f"""# 九条归档的释放后交付诊断（不改原评分）

结论：旧native fusion的16s末态确实在地板上，仍以约0.279m/s向+x滚动；不是把相对位姿或手部body读成物体。V13 fusion留在支撑台高度且运动较小；V14 fusion仍在台面高度，但中心距右缘仅约{v14['center_xy_edge_margin_m']*1000:.3f}mm，仍向外运动。不能据此倒推任何版本总体更优；旧fusion静态force更优的事实，与释放后的交付缺陷必须同时保留。

## 九条末态

| 版本/方法 | object world XYZ at16s m | vx m/s | angular speed rad/s | centre XY edge margin mm |
|---|---|---:|---:|---:|
"""+'\n'.join(table)+f"""

三代九条metadata/config及XML model SHA完全相同。`trace.position`与真实object_free qpos[25:28]在全部3201个边界样本逐值相同。静态receiver顶面z=.337m、x范围[.025,.165]m；地板z=0；capsule半径.028m。旧fusion末态centre z=.027921m与地板软接触高度相符，而台上其他末态z=.364921m相符。对象free quaternion用于世界方向；trace.rotation本身是相对手掌的旋转，不能当世界旋转。原relative_velocity也不能当世界速度，本报告取freejoint平移速度与角速度范数。

中心边缘裕量只是点到矩形边界的描述，不是经验证的稳定裕量。完整胶囊投影可有overhang，overhang也不能自动改判原任务失败。V14fusion的末态vx约{v14['free_translation_velocity_world_m_s'][0]:.6f}m/s/角速度{v14['free_angular_speed_rad_s']:.6f}rad/s说明尚非静止交付；没有运行16s以后的物理，不能断言它将落地。

## 旧fusion落地时序与外界代价

200Hz解析胶囊几何首次跨入floor的时间括在{timeline['first_sampled_floor_overlap_bracket_s']}s。原始0.5ms支持峰{fp['external_support_N']:.6f}N位于end={fp['end_s']:.6f}s；世界穿透峰{pen['cached_world_penetration_m']*1000:.6f}mm位于end={pen['end_s']:.6f}s。其附近200Hz姿态显示对象已离开接收台，靠近地板；每个原始事件和最近边界姿态的时间差都在JSON明列。这不是同一微时刻fresh FK/contact查询，因此不以几何间隙和缓存穿透逐位相等作证明。

`support`字段并未漏掉对象—地板的接触。源`HandPlant._contacts`将目标物体与任何非手部几何的接触法向量累加到这个旧命名变量，其中可以包括floor。它是对象特定的离线非手接触聚合，**不是指定receiver传感器，也不等于交付到台面**。后续公共receiver设备必须显式绑定接收几何，且不能沿用这个target过滤真值当在线传感器。

## 为什么原分数仍能成功

原V3保留任务门：holding/transport的失持检索止于release时刻；释放判据是合格曝光后，在时限内达到手—物低接触力并维持dwell。mission再要求完整记录、handoff、无先前失败和无warning；full-motion补充运动跟踪。**它们没有检查释放后物体留在指定receiver、落地冲击或16s的工作区交付。**

因此旧fusion低力清离后滚出台面，仍可能按原合同task/release成功。这是任务覆盖边界，不是本次改阈值的理由。本报告逐条保留原task/mission/fullmotion/release/guard与静态质量，不改任何原result、score或门槛，也不把新增单项作为事后综合权重。

下一阶段若目标包括安全交付，应预先定义receiver任务、共享公共观测及物体/环境冲击与留存诊断；释放请求仍立即取消holding，不能以未知placement阻塞用户release。当前结论只适用于这个已见开发条件，不能称九个独立总体样本或全生命周期优越证明。

结构化证据：[JSON](native_v13_v14_release_delivery_diagnostic_v1.json)。纯读取加一次静态模型编译；step/forward/FK/contactForce全部0，无新物理、无原评分修改。
"""
    MD.write_text(text,encoding='utf-8');print(json.dumps(dict(json=str(OUT),json_sha256=sha(OUT),markdown=str(MD),markdown_sha256=sha(MD),rows=9,physics_steps=0),ensure_ascii=False))

if __name__=='__main__':main()

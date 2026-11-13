"""Write a self-contained V24 stage report only after all physical evidence."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, re, shutil, statistics
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pct(x,y):return (x/y-1)*100
def display(x,digits=3):return '未达到/未知' if x is None else f'{x:.{digits}f}'

def main():
    d=read('review/v24_small_readout_v1.json');a=read('review/v24_small_execution_v1.json');p=read('review/v24_small_plan_v1.json')
    assert d['complete'] and d['issues']==[] and len(d['records'])==9
    assert a['passed'] and a['issues']==[] and a['all_assigned_checked']
    assert d['plan_sha256']==a['plan_sha256']==sha(ROOT/'review/v24_small_plan_v1.json')
    assert len(p['source_sha256'])==192
    for n,h in p['source_sha256'].items():
        actual=ROOT.parent/'五指策略接入_2026-11-03'/n[11:] if n.startswith('foundation/') else ROOT/n
        assert sha(actual)==h,n
    roles=('v23_joint','canonical_backend','canonical_joint');labels=('冻结V23联合','规范参照层级','规范参照联合')
    contexts=[tuple(c) for c in p['contexts']];scene_labels=('名义运输','易碎冲突持握','摩擦变化运输')
    idx={(r['variant'],*r['context_key']):r for r in d['records']}
    old={v:{(r['variant'],*r['context_key']):r for r in read(f'review/{v}_small_readout_v1.json')['records']} for v in ('v22','v23')}
    pi={tuple(r['context_key']):r['reference_metrics'] for r in d['historical_pairs'] if r['variant']=='canonical_joint' and r['reference_role']=='same_prior_legacy_ratio'}
    def count(role,key):return sum((pi[c] if role=='PI' else idx[(role,*c)]['metrics'])[key] is True for c in contexts)
    summary='\n'.join('|'+label+'|'+'|'.join(f'{count(role,key)}/3' for role in ('PI',*roles))+'|' for label,key in (
        ('原任务完成','original_task_success'),('物理清离','original_physical_clearance_success'),('全程力限通过','original_full_guard_pass'),
        ('释放窗曾落地↓','ever_floor'),('首次低运动驻留','first_low_motion_found'),('末段低运动驻留','final_low_motion_suffix_found')))
    performance=[];times=[];effects=[]
    for label,c in zip(scene_labels,contexts):
        for role in roles:
            r=idx[(role,*c)];m=r['metrics'];w=r['observed_control_walltime']
            performance.append(f'|{label}|{labels[roles.index(role)]}|{r["release_TV"]["all_actor_skin_TV_N"]:.3f}|{m["end_angular_speed_rad_s"]:.6f}|{m["end_linear_speed_m_s"]:.6f}|{display(m["receiver_first20ms_peak_N"])}|')
            times.append(f'|{label}|{labels[roles.index(role)]}|{display(m["original_200Hz_clearance_latency_s"])}|{display(m["first_low_motion_latency_s"])}|{display(m["final_low_motion_duration_s"])}|{w["p95_ms"]:.3f}|{w["maximum_ms"]:.3f}|')
        candidate=idx[('canonical_joint',*c)]
        for role,ref in [('同轮规范层级',idx[('canonical_backend',*c)]),('同轮冻结V23',idx[('v23_joint',*c)]),('历史V22联合',old['v22'][('joint',*c)]),('历史V21',old['v23'][('v21',*c)])]:
            effects.append(f'|{label}|{role}|{pct(candidate["release_TV"]["all_actor_skin_TV_N"],ref["release_TV"]["all_actor_skin_TV_N"]):+.2f}%|{pct(candidate["metrics"]["end_angular_speed_rad_s"],ref["metrics"]["end_angular_speed_rad_s"]):+.2f}%|')
    tv_b=statistics.median(pct(idx[('canonical_joint',*c)]['release_TV']['all_actor_skin_TV_N'],idx[('canonical_backend',*c)]['release_TV']['all_actor_skin_TV_N']) for c in contexts)
    omega_b=statistics.median(pct(idx[('canonical_joint',*c)]['metrics']['end_angular_speed_rad_s'],idx[('canonical_backend',*c)]['metrics']['end_angular_speed_rad_s']) for c in contexts)
    crole='canonical_joint'
    execution_rows=[]
    for role in roles:
        counts=Counter()
        for row in a['rows']:
            if row['variant']==role:counts.update(row['counts'])
        execution_rows.append(f'|{labels[roles.index(role)]}|{counts["direct"]}|{counts["accepted_full"]}|{counts["accepted_prefix"]}|{counts["candidate_not_committed_fallback"]}|')
    expansion=all([count(crole,'original_task_success')==3,count(crole,'original_physical_clearance_success')==3,
        count(crole,'ever_floor')==0,count(crole,'original_full_guard_pass')>=2,count(crole,'final_low_motion_suffix_found')>=1])
    now=datetime.now(timezone.utc).isoformat();name='V24规范参照多任务融合九条对照与后续路线_20261106.md'
    assert not (ROOT/name).exists()
    analysis='review/V24多接触运动观测与响应建模整体分析_v1.md'
    assert (ROOT/analysis).is_file(),'Require the whole-framework analysis before publishing the stage'
    body=f'''# 规范参照多任务融合：九条同条件物理结果与整体路线

V24已完成独立六文件实现、全部九条16秒物理运行、完整参考桥和保存执行审查。规范联合任务原任务{count(crole,'original_task_success')}/3，物理清离{count(crole,'original_physical_clearance_success')}/3，力限{count(crole,'original_full_guard_pass')}/3，释放窗落地{count(crole,'ever_floor')}/3，末段低运动{count(crole,'final_low_motion_suffix_found')}/3；同轮规范层级末段低运动{count('canonical_backend','final_low_motion_suffix_found')}/3，冻结V23为{count('v23_joint','final_low_motion_suffix_found')}/3。相对规范层级，联合版逐指TV配对变化中位{tv_b:+.2f}%，末角速中位{omega_b:+.2f}%。这些是三个已见开发条件的具体取舍，不能单独证明整体或泛化优势。

报告UTC：{now}。九条已全部终态、当前无物理仿真后台运行。原平台、父186来源和全部既有实验及负结果保持；主目标active，尚未证明完成。

## 全景方案

控制结构是：公共触觉/机器人观测及有效性→原机械/速率/当前状态权限→唯一法向参照→可观测运动预算→实际多任务融合→交付退出进度与全手路径→原问题及最终发出动作核验→原物理执行反馈。

原法向最优参照不唯一，却会决定后续运动预算，导致求解实现改变控制问题。V24先保持原压力加权纠偏最优值，再在其达到集合内选归一21维参考最小范数的唯一点，用它定义预算。该参照选择是任务合同的一部分，不是新力学模型、物理能量或未来安全保证。

参照两层须全部认证；第二层只确定预算，不能把其I21达到值提交到实际任务中，否则会锁死21维退出自由度。规范层级仅提交原法向第一层，再退出；规范联合不提交参照层，而联合协调法向纠偏与观测运动，最后进度与路径。选择失败则明确预算未建立，使用原合法fallback。未知不填零，P0保留原退出记忆。

持握、运输、物理模型、随机流、噪声/延迟/丢包、原有限后端、权限、容差、参数、成功定义和评价阈值均未改变。实现中的索引记录及审查脚本符号零问题已保留在验收材料，不作为算法创新或额外效果。

## 同条件比较与固定综合评价

A为冻结V23联合，B为规范参照层级，C为规范参照联合。A与C比较预算来源重构；B与C比较相同规范预算下的任务融合。九条只有三个开发情境、两个物理种子簇，不是九个独立重复。A三条完整16秒桥各92项通过后才运行B/C；全九条终态后统一读数，未在中途筛选或修改source。

|指标|历史PI同条件|冻结V23联合|规范参照层级|规范参照联合|
|---|---:|---:|---:|---:|
{summary}

PI是论文启发力比控制的迁移基线，不是论文全系统复现。PI与新版本的持握差异已有历史实验，不能把全部力或冲击收益归功于本轮释放融合模块。既有完整16条件与本轮三条件分开，不拼成新的完整矩阵。

|条件|角色|释放逐指TV N↓|末角速度rad/s↓|末线速度m/s↓|首触接收20ms峰N↓|
|---|---|---:|---:|---:|---:|
{chr(10).join(performance)}

|条件|角色|清离延迟s↓|首次低运动延迟s↓|末段低运动时长s↑|控制p95 ms|控制最大ms|
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(times)}

所有指标使用冻结v3，低运动驻留要求接收面有载、手低力清离、无其它支撑，v≤0.005m/s且ω≤0.15rad/s持续0.2秒。未达到或未知保留NULL，不视为0延迟，也不只比较成功者的平均时间。驻留不是永久稳定或完整几何容纳证书。

TV为12–16秒801个原始200Hz样本的逐指绝对增量总和，五指全部外接触、不含掌部，含正常卸力；不是纯振荡或合力向量范数。更低TV与更低物体转动应并列检查。计算墙钟不推进固定仿真时钟，观察p95/最大值不等于硬件200Hz实时能力。

|规范联合相对参考|参考|TV变化|末角速变化|
|---|---|---:|---:|
{chr(10).join(effects)}

扩展最低资格（任务/清离3、落地0、力限≥2、末段低运动≥1）本轮为 **{'通过' if expansion else '未通过'}**。该资格仅决定是否扩大开发矩阵；实际是否整体更优仍需力、冲击、时间和稳定性取舍及完整/未见确认。没有用事后权重把一种代价隐藏成单分数收益。

## 执行证据与仍待解决的物理问题

作者35项、独立264项、三角色零步21项、六条公共流38项均通过，最终192来源及父186/原42和数值身份保持。新独审从真实保存公共输入重建normal、minnorm、实际任务与全部hard G/E并独立回验KKT，不用旧81题冒充新任务门。保存执行审查{len(a['checks'])}项通过，覆盖全部九条与实际21维发出/回执、真实权限、任务证据角色及错误前缀；这些证明实现与记录一致，不能替代物理性能结论。

全部7209个释放决策均核对，每臂2403；并非所有任务求解都成功：

|角色|直接阶段|任务全层|部分前缀|透明fallback|
|---|---:|---:|---:|---:|
{chr(10).join(execution_rows)}

规范层级9次fallback为法向第一层2次、参照选择层7次未获得原问题证据；规范联合14次中，法向5次、参照选择4次，其余5次为实际联合层未获得原KKT。未完整参照时不建立预算，实际联合失败时也不把参照证据当成已执行任务。全三臂实际提交有限证书共8341份按各自历史点原KKT复核，actual/emitted/receipt另核，changed/stale均0；803份规范参照证据与206份旧参照证据单列。通过审查不是把这些失败改成全层成功。

当前触觉运动表示与机器人参考几何并非“动作如何改变物体速度”的响应模型。多点切向观测在同一刚体、同龄与足够几何秩条件下可能提供完整运动信息；当前all-actor压力平均投影没有给出这种归属与一致性保证，手清离后也不再提供物体运动反馈。不能靠把未知方向设0或直接使用离线物体真值补齐控制。

后续整体路线应将接触归属/一致性与可观测子空间、真实指令到接触运动响应、接触交接和任务阶段统一设计。在已验证观测子空间内协调防滑、限力、低冲击与低运动交付；对不可观测/不可控方向明确保留未知和降级，不将几何代理最优误作真实物体稳定。规范参照只解决预算定义，不预设解决上述动力学缺口。详见[整体分析]({analysis})。

## 复核材料

- [物理前整体合同](review/V24规范参照多任务融合与同条件验证合同_v1.md)
- [完整九条固定读数](review/v24_small_readout_v1.json)
- [全执行审查](review/v24_small_execution_v1.json)
- [冻结参考完整桥](review/v24_original_V23_joint_full16_bridges_v1.json)
- [新任务独立验收](review/V24规范witness与实际退出任务独立数值验收_v1.md)
- [三组科学时序与图注](report/v24_small_v1/release_curves_v1/figures.md)
- [V23负结果与结构原因](V23高效执行闭环九条结果与规范任务重构_20261106.md)

![名义运输](report/v24_small_v1/release_curves_v1/01_nominal_transport_619741.png)

![易碎冲突持握](report/v24_small_v1/release_curves_v1/02_fragile_conflict_hold_619741.png)

![摩擦变化运输](report/v24_small_v1/release_curves_v1/03_friction_drop_transport_619742.png)

读数SHA256 `{sha(ROOT/'review/v24_small_readout_v1.json')}`；执行审查 `{sha(ROOT/'review/v24_small_execution_v1.json')}`；计划 `{sha(ROOT/'review/v24_small_plan_v1.json')}`。原始失败/NULL与旧尝试均保留。
'''
    assert all((ROOT/n).is_file() for n in re.findall(r'\]\(([^)]+)\)',body))
    (ROOT/name).write_text(body,encoding='utf-8')
    hist=ROOT/'report/progress_history';hist.mkdir(parents=True,exist_ok=True);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for n in ('研究进度.json','当前进度与阶段汇报.md','review/v24_small_progress_v1.json'):shutil.copyfile(ROOT/n,hist/(stamp+'_'+Path(n).name))
    state=read('研究进度.json')
    state.update(updated_utc=now,goal_status='active',superiority_proven=False,latest_report=name,latest_stage_report=name,latest_algorithm_report=name,
        current_work='V24 independent canonical task192 implemented; fixed9 physical/readout/execution complete. Stage tradeoffs retained; whole contact-motion response and observability architecture under evidence-based analysis.',
        current_physics_processes=[],current_physics_jobs=[],active_physics_jobs=[],
        previous_goal_turn_classification='progress: published V23 negative evidence, completed canonical task contract with actual gates, registered and executed9, exact reference bridges, fixed composite readout and actual execution review.')
    state['v24_current'].update(status='small9_complete_readout_execution_passed',completed_physical_runs=9,physical_runs=9,physics_active=False,physics_sessions=[],
        readout='review/v24_small_readout_v1.json',execution_audit='review/v24_small_execution_v1.json',execution_checks=len(a['checks']),report=name,
        expansion_gate_passed=expansion,superiority_proven=False,execution_review_session=None,execution_audit_pending=False)
    save('研究进度.json',state)
    progress=read('review/v24_small_progress_v1.json');progress.update(updated_utc=now,phase='readout_and_execution_complete',completed=9,all_terminal=True,
        physics_active=False,physics_sessions=[],results_read=True,readout='review/v24_small_readout_v1.json',execution_audit='review/v24_small_execution_v1.json',report=name,execution_review_session=None)
    save('review/v24_small_progress_v1.json',progress)
    (ROOT/'当前进度与阶段汇报.md').write_text(f'''# V24规范参照融合九条完成

更新UTC：{now}。当前无物理仿真后台运行，主目标active、综合优势尚未证明完成。

规范联合任务{count(crole,'original_task_success')}/3、清离{count(crole,'original_physical_clearance_success')}/3、力限{count(crole,'original_full_guard_pass')}/3、落地{count(crole,'ever_floor')}/3，末段低运动{count(crole,'final_low_motion_suffix_found')}/3；规范层级末段{count('canonical_backend','final_low_motion_suffix_found')}/3。相对规范层级TV配对中位{tv_b:+.2f}%、末角速{omega_b:+.2f}%，取舍完整保留。

三条参考完整桥各92通过，保存执行审查{len(a['checks'])}项通过，全部九条终态，192来源/父186与原依赖保持。下一步依据多接触观测与响应模型分析推进整体融合，不移动评价阈值或只报告胜例。

[完整全景报告]({name})；[九条读数](review/v24_small_readout_v1.json)；[三组时序](report/v24_small_v1/release_curves_v1/figures.md)；[整体分析]({analysis})。
''',encoding='utf-8')
    print(json.dumps(dict(report=name,completed=9,execution_passed=True,expansion_gate_passed=expansion,superiority_proven=False),ensure_ascii=False))

if __name__=='__main__':main()

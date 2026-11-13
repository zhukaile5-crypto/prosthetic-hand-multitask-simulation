"""Final nine-run report from complete fixed readout and passed execution audit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil,statistics
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
d=read('review/v22_small_readout_v1.json');audit=read('review/v22_small_execution_v3.json');plan=read('review/v22_small_plan_v1.json')
assert d['complete'] and not d['issues'] and audit['passed'] and not audit['issues']
assert len(d['records'])==9 and len(d['within_pairs'])==9 and len(d['historical_pairs'])==27
for k,v in plan['source_sha256'].items():
    p=ROOT.parent/'五指策略接入_2026-11-03'/k[11:] if k.startswith('foundation/') else ROOT/k
    assert sha(p)==v,k
idx={(r['variant'],*r['context_key']):r for r in d['records']}
contexts=[tuple(k) for k in plan['contexts']]
labels=['名义运输','易碎冲突持握','摩擦变化运输']
roles=['v21','backend','joint']
pi={tuple(p['context_key']):p['reference_metrics'] for p in d['historical_pairs'] if p['variant']=='joint' and p['reference_role']=='same_prior_legacy_ratio'}
tvold=read('review/v21_release_variation_v1.json')
pi_tv={tuple(r['context_key']):r['reference_PI_metrics']['all_actor_skin_TV_N'] for r in tvold['records']}
def count(role,key):return sum((pi[k] if role=='PI' else idx[(role,*k)]['metrics'])[key] is True for k in contexts)
def pct(a,b):return (a/b-1)*100
aggregate='\n'.join('|'+name+'|'+'|'.join(str(count(v,key))+'/3' for v in ['PI',*roles])+'|' for name,key in [
    ('原任务完成','original_task_success'),('物理清离','original_physical_clearance_success'),('全程力限通过','original_full_guard_pass'),
    ('释放窗曾落地↓','ever_floor'),('末段低运动驻留','final_low_motion_suffix_found')])
table=[];effects=[];latency=[]
for label,k in zip(labels,contexts):
    for role in roles:
        r=idx[(role,*k)];m=r['metrics'];w=r['observed_control_walltime']
        table.append(f'|{label}|{role}|{r["release_TV"]["all_actor_skin_TV_N"]:.3f}|{m["end_angular_speed_rad_s"]:.6f}|{m["original_200Hz_clearance_latency_s"]:.3f}|{m["receiver_first20ms_peak_N"]:.3f}|')
        latency.append(f'|{label}|{role}|{w["median_ms"]:.3f}|{w["p95_ms"]:.3f}|{w["maximum_ms"]:.3f}|{w["above_5ms_count"]}/{w["samples"]}|')
    a,b,c=(idx[(v,*k)] for v in roles)
    effects.append(f'|{label}|{pct(c["release_TV"]["all_actor_skin_TV_N"],a["release_TV"]["all_actor_skin_TV_N"]):+.2f}%|{pct(c["metrics"]["end_angular_speed_rad_s"],a["metrics"]["end_angular_speed_rad_s"]):+.2f}%|{pct(c["release_TV"]["all_actor_skin_TV_N"],b["release_TV"]["all_actor_skin_TV_N"]):+.2f}%|{pct(c["metrics"]["end_angular_speed_rad_s"],b["metrics"]["end_angular_speed_rad_s"]):+.2f}%|')
tv_pi_changes=[pct(idx[('joint',*k)]['release_TV']['all_actor_skin_TV_N'],pi_tv[k]) for k in contexts]
now=datetime.now(timezone.utc).isoformat();name='V22联合任务闭环九条对照与后续路线_20261105.md'
assert not (ROOT/name).exists()
body=f'''# 联合任务闭环：已有实测收益，融合贡献与计算代价仍须分开

本轮完成独立V22闭环接线、九条实际物理运行和完整执行审查。联合任务版相对V21的三个条件末角速度均下降，力变化总量两降一升，原任务/清离3/3、力限2/3、零落地及末段低运动1/3保持。与仅换后端对照相比，融合版在摩擦变化条件更稳定，但另外两条件末角速度更高；不能把全部收益归因于联合任务结构，也不能称综合目标已完成。

报告记录UTC：{now}。原平台、原170来源、V21完整16及全部旧实验保留，V22固定179来源未变。当前九条全部终态、无运行错误、无物理进程在后台运行；主目标继续active。

## 全景方案

控制链为：公共触觉/机器人状态及有效性→真实21维机械/速率/当前状态权限→联合纠偏与可观测运动→退出进度及全手路径→原问题与实际发出动作验收→执行和反馈。

持握、运输、物理模型、随机流、观测噪声/延迟/丢包和材料评价定义均沿原版本。改进集中在退出决策：将几何法向纠偏slack与当前触觉运动投影作为同一层软任务，按声明尺度共同求解，避免先固定几何代理最优值再要求运动任务让路。它们不是未来接触力/物体速度预测。后续退出进度与路径使用联合层剩余能力，未知不填零，P0保留原退出记忆。

求解候选和接受证据分离：每个候选必须通过原矩阵/原约束/原容差KKT与严格真实参考边界，再按最终发出参考重评分。失败保留最后认证前缀或实际合法anchor，normal witness不能冒充joint目标已完成。软件曾发现的越界候选、读数脚本类型错误及所有失败尝试均保留，不作为科学创新或额外收益。

## 同条件辨别

A=v21，冻结原策略；B=backend，保持原V21任务规则，仅换数值后端；C=joint，与B同后端，改为联合任务。三个条件为nominal/transport/619741、fragile_conflict/hold/619741、friction_drop/transport/619742。

九条均为本轮新运行。先完成A三条与旧V21完整16秒逐字段桥，每条92项通过，只有原计时白名单例外；再运行B/C各三条，全部终态后才统一读指标。三个条件复用两个物理种子簇，不是九个独立重复或未见确认。

B后端可改变非唯一normal最优点的选择，后续motion预算由该点构造，因此B−A包含这种预算传递影响，不能简单称为修复数值报错；C−B才是同后端下任务协调结构的比较。

|指标|论文启发力比PI历史同条件|A 原V21|B 仅后端|C 联合任务|
|---|---:|---:|---:|---:|
{aggregate}

PI是论文启发的迁移基线，并非论文全系统复现。PI全程指峰/接收冲击较高的既有差距还来自共享持握方案，不能据此宣称本轮新增融合模块单独带来全部优势。

## 具体收益与代价

|条件|角色|释放五指力TV N↓|末角速度rad/s↓|清离延迟s↓|触台首20ms峰N↓|
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

|条件|C相对A：TV|C相对A：末角速|C相对B：TV|C相对B：末角速|
|---|---:|---:|---:|---:|
{chr(10).join(effects)}

C相对A末角速配对变化中位−36.44%，TV配对变化中位−7.43%。易碎条件TV增加10.68%必须并列；不能只展示摩擦变化这一胜例。名义与易碎条件仍未达到固定末段低运动门槛，不能把速度下降当驻留成功。

相对同条件PI历史参考，C释放TV三例分别下降{abs(tv_pi_changes[0]):.2f}%、{abs(tv_pi_changes[1]):.2f}%、{abs(tv_pi_changes[2]):.2f}%，中位−{abs(statistics.median(tv_pi_changes)):.2f}%。这些是固定12–16秒801个未滤波200Hz样本的逐指总变差，含正常卸力，不全部等同于不必要噪声；五指全部外接触不含掌部。历史PI TV来自既有冻结同条件记录，未修改公式或原物理数据。

与B相比，C恢复摩擦变化条件的末段低运动（末角速0.58334→0.06803rad/s），代价是该条件TV增加31.36%；B在名义条件已有低运动而C没有。这个结果说明更低力TV与更稳定交付不必同时出现，综合评价是必要的，当前没有一个版本在所有目标上占优。

## 计算开销是明确短板

|条件|角色|完整释放窗控制中位ms|p95 ms|最大ms|超过5ms次数|
|---|---|---:|---:|---:|---:|
{chr(10).join(latency)}

这些是同机并行批次观察值，包含直接状态阶段，低中位可能掩盖优化阶段大开销；不能作为独占硬件速度排名。控制计算不推进仿真传感时钟，因此当前结果是固定200Hz离线闭环仿真，不能声称200Hz实时可用。各episode16秒的实际运行与归档，A约114–116秒，B约444–504秒，C约529–693秒。

独立旧历史联合层资源诊断进一步揭示：当前f304在16题中12题用满400个FW方向，15/16获得原问题KKT；中位单层约1.30秒。原65题全部通过不能外推新联合层任意输入都可求解。该缺口、实际fallback和计算成本均保留。

## 完整执行证据

独立审查126/126通过，全部7209个释放决策完整核验。每臂2403个，其中direct1133、任务1270。

|角色|任务全层|部分前缀|透明fallback|有压全层/有压总数|
|---|---:|---:|---:|---:|
|A 原V21|1152|118|0|109/227|
|B 仅后端|1265|4|1|188/193|
|C 联合任务|1261|5|4|199/208|

C的4次fallback都是joint目标无KKT见证而未提交witness，不把normal已解说成joint成功；B唯一fallback为L1未认证。B/C共5649个有限证书在各自当时cert.x复核完整原KKT，最终actual另核任务输出/硬域；execution_changed与stale均0。B/C释放前完整37列及公共观测UTF8与原条件精确相同。

## 后续整体路线

当前已达到原三条件扩展资格，但结合新联合层求解与计算缺口，不直接扩大昂贵f304运行。先完成任务定义与数值执行的清晰分离，再在相同指标下验证。

第一步是高效且可认证的执行实现：独立V23数学原型已完成14作者项及固定81问题原KKT/严格box，联合16题无需前置FW，单层中位约9.73ms。它未接入runtime、没有新物理结果，不可替换或美化本轮数据；仍需独立全门、实际发出检查和同条件闭环。

第二步是检验运动目标的物理有效性。局部设备平面投影不是完整物体转动，reference几何不是实际对象响应；本轮更多任务层求解完成也没有使C稳定超过B。应核查可观测范围、观测/执行延迟及预算anchor对任务的影响，再选择统一任务表示。保留多任务联合方向，不能靠调权重、放宽力限或移动驻留阈值解释未达到的结果。

快速实现通过小规模对照后，再决定固定完整16与未见物理种子确认。不能将这三个开发条件与原V21完整16拼接成当前V22完整矩阵，也不能以软件门或单一指标结束主目标。

## 完整材料

- [全部九条固定读数](review/v22_small_readout_v1.json)
- [独立全执行审查](review/v22_small_execution_v3.json)
- [三条完整V21桥](review/v22_original_v21_full_bridges_v1.json)
- [三组完整时序图](report/v22_small_v1/release_curves_v2/figures.md)
- [物理前实施合同](review/V22联合任务闭环实施与小规模对照合同_v1.md)
- [数值后端实际方法与边界](review/V22有限原问题后端作者说明_v1.md)
- [当前联合层资源与可解性诊断](review/V22历史联合层数值与资源诊断_v1.md)
- [未接入物理的高效执行原型](review/V23候选组织原型与固定81题验收_v1.md)

![名义运输完整时序](report/v22_small_v1/release_curves_v2/01_nominal_transport_619741.png)

![易碎冲突持握完整时序](report/v22_small_v1/release_curves_v2/02_fragile_conflict_hold_619741.png)

![摩擦变化运输完整时序](report/v22_small_v1/release_curves_v2/03_friction_drop_transport_619742.png)

本报告主要输入SHA256：readout `{sha(ROOT/'review/v22_small_readout_v1.json')}`；execution `{sha(ROOT/'review/v22_small_execution_v3.json')}`；plan `{sha(ROOT/'review/v22_small_plan_v1.json')}`。所有旧尝试、失败/NULL和来源快照保持。
'''
(ROOT/name).write_text(body,encoding='utf-8')
assert all((ROOT/n).is_file() for n in re.findall(r'\]\(([^)]+)\)',body))
hist=ROOT/'report/progress_history';hist.mkdir(parents=True,exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
for n in ('研究进度.json','当前进度与阶段汇报.md'):shutil.copyfile(ROOT/n,hist/(stamp+'_'+n))
p=read('研究进度.json');p.update(updated_utc=now,goal_status='active',superiority_proven=False,
    latest_report=name,latest_stage_report=name,latest_algorithm_report=name,
    current_work='V22 nine runs complete/readout and126 execution accepted; partial physical gains and high computation cost retained. V23 math-only81 prototype ready, runtime/same-condition verification still required.',
    current_physics_processes=[],current_physics_jobs=[],active_physics_jobs=[],
    previous_goal_turn_classification='progress: implemented full179 runtime, resolved strict actuation, executed all9, exact3 bridges, frozen composite metrics and all7209 decisions audited, plus independent efficient81-task prototype.')
p['v22_current'].update(status='small9_complete_readout_execution126_passed',physics_active=False,physics_sessions=[],
    completed_physical_runs=9,execution_audit_pending=False,execution_audit='review/v22_small_execution_v3.json',
    execution_checks=126,report=name,current_original65_KKT=65,current_joint_historical16_KKT=15,
    current_joint_historical16_probe='review/v22_joint_level_resource_probe_v1.json',superiority_proven=False)
p['v23_current']=dict(status='mathematical_prototype81_passed_not_runtime',source_namespace='v23_task_execution',
    math_evidence='review/v23_finite81_author_v1.json',runtime_connected=False,physics_active=False,physical_runs=0,
    scope='Fixed original65 + historical joint16, not physical outcome or realtime proof',next='Independent math+runtime+actual emission acceptance, then same-condition comparison')
save('研究进度.json',p)
progress=read('review/v22_small_progress_v1.json');progress.update(updated_utc=now,completed=9,all_terminal=True,
    phase='readout_and_execution_complete',physics_active=False,physics_sessions=[],execution_audit='review/v22_small_execution_v3.json')
save('review/v22_small_progress_v1.json',progress)
(ROOT/'当前进度与阶段汇报.md').write_text(f'''# V22九条闭环对照完成，下一步解决高效执行与任务有效性

更新UTC：{now}。当前没有物理仿真在后台运行，主目标active。

联合任务相对V21：三条件末角速度均降低（−36.4%、−56.7%、−16.9%），力TV为−7.4%、+10.7%、−27.6%。任务3/3、清离3/3、力限2/3、零落地、末段低运动1/3均保持。相对仅后端版本仍存在取舍，不能把全部收益归于融合。

三条完整V21桥各92通过，独立执行126项/7209决策通过，source179及旧170/42、数值依赖保持。计算开销显著，不能称实时可用。V23独立数学原型81题已过，尚未接入runtime或新物理。

[完整全景方案、九条结果与后续路线]({name})；[三组完整曲线](report/v22_small_v1/release_curves_v2/figures.md)；[固定读数](review/v22_small_readout_v1.json)；[执行审查](review/v22_small_execution_v3.json)。
''',encoding='utf-8')
print(json.dumps({'report':name,'source179_unchanged':True,'physical_runs_terminal':9,'execution_passed':True,'goal_status':'active'},ensure_ascii=False))

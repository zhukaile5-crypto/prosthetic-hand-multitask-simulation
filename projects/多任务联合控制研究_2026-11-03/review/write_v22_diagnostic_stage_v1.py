"""Write a source-backed stage report and update progress, with no new physics."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,statistics,shutil,re
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
aux=read('review/v21_complete16_auxiliary_v1.json')
for k,v in aux['source_sha256'].items():
    p=ROOT.parent/'五指策略接入_2026-11-03'/k[11:] if k.startswith('foundation/') else ROOT/k
    assert sha(p)==v,k
oracle=read('review/v21_same_problem_oracle_results_v1.json')
quotient=read('review/v21_affine_quotient_oracle_results_v1.json')
order=read('review/v21_motion_priority_counterfactual_v1.json')
tradeoff=read('review/v21_normal_motion_tradeoff_counterfactual_v1.json')
joint=read('review/v22_joint_bundle_offline_v1.json')
assert oracle['assigned']==oracle['assessed']==65 and len(joint['rows'])==16
assert len(order['rows'])==len(tradeoff['rows'])==16
motion_change=statistics.median((r['motion_task_loss_after']/r['motion_task_loss_before']-1)*100 for r in joint['rows'])
normal_changes=[(r['normal_task_loss_after']/r['normal_task_loss_before']-1)*100 for r in joint['rows']]
max_order_delta=max(abs(r['counterfactual_motion_loss']-r['original_motion_loss']) for r in order['rows'])
now=datetime.now(timezone.utc).isoformat()
name='V22联合任务重构依据与离线原型结果_20261105.md'
assert not (ROOT/name).exists(),'Preserve the previous stage report.'
text=f'''# V22联合任务重构：从顺序让路改为共同协调

本轮已定位到可验证的设计限制，并写入独立V22联合任务代数原型。固定16个历史输入上，原型的运动代理误差全部下降，配对降幅中位约{abs(motion_change):.2f}%；几何纠偏软任务代价上升{min(normal_changes):.2f}%–{max(normal_changes):.2f}%。这是局部参考任务的数学取舍，不是物体真实速度、接触力或抓握完成率改善。新原型尚未接入物理运行。

报告时间UTC：{now}。V21完整16物理结果保持冻结，原平台和全部旧结果保留。没有新增物理试验，没有安装数值包或修改全局环境。

## 当前实测基础

V21完整16条件相对力比PI迁移基线：任务14/16对12/16、全程力限8/16对3/16、释放曾落地0/16对4/16；末段低运动驻留5/16对5/16。相对V20，末段驻留从0/16改善到5/16、落地2/16降到0/16，但释放力TV配对中位+9.53%。只有两个物理种子簇，这仍是开发证据。

完整1377个有压决策中686全层、686部分前缀、5回退。固定规则先选择每个(context,失败层)第一条，共49个失败样本（L1/L2/L3/L4为1/16/16/16）及16条每context第一条全成功对照。没有按求解结果筛样。其余4个L1失败保留在全1377分类及原日志，未宣称65代表1377全部问题。

## 数值失败与任务能力分别核查

独立原问题方法生成候选后，用原矩阵、硬行、前层结果、原容差检查可行性与KKT，不能只看库success。65条中35条通过；显式等式仿射消元的另一候选表示为37条通过，尚不足以替换完整后端。

原L2的16首失败中13个原点已经最优，原L3的16首失败原点全部已经最优。独立审查进一步从原日志重建A/b及E/f，并用另一套对偶可行性LP验证16个L3原点KKT，排除了错误冻结当前运动层的解释。L3的A在E零空间仍有4–10的秩，不能说运动矩阵被E完全消掉；不等式活跃约束也必须计入。

16条原成功对照指其真正末层成功；其L3初值不是末层解。抽取两个真实保存末层点及原对偶映射的正对照均通过。因此当前SLSQP候选在姿态层失败，不能推翻原问题可解，也不能直接把库当可靠替代。

## 两个明确区分的任务反事实

第一个反事实只取消L2退出进度的已达成等式，保留L1最优slack冻结、全部原normal/motion/机械/rate域。16条仍有可行线段，但运动目标最大绝对变化仅{max_order_delta:.3g}，属于舍入量级。对这些输入，单纯交换退出与运动层序没有提供有效改善方向。

第二个反事实继续允许几何纠偏slack偏离L1最优值，恢复其原L1推导上下界；保留前21维原机械/rate边界、全部normal不等式和原motion预算。16条都存在运动目标下降的可行参考方向，同时原几何纠偏目标代价增加。作者实际对方向LP验KKT，并对线段逐一回原域验可行；独立代入审查的范围另在审查文件列出，不把未保存的LP对偶数组冒称可独立复签。

几何slack不是接触压力。这个反事实没有改材料阈值或原forceguard，也没有证明未来压力仍安全；原几何QP本来就没有以预测接触力N表达的硬上限。安全代价必须用后续相同物理情境实测。

## 整体架构与已实现原型

整体逻辑为：公共观测及其可观测范围→真实执行权限→联合任务协调→实际动作与原问题验收→交付证据。

1. 观测只使用既有机器人状态、接触几何和触觉投影；报告其时间、有效性与秩。当前每指2个投影不等于物体6维运动。P0时运动反馈未知，不能当零速度或完成稳定交付。
2. 实际机械、速率、执行代际与原生执行限幅继续由公共执行链约束。材料力限沿原定义评估；不把几何约束伪称力安全证明。
3. 几何纠偏与可观测运动作为同一层的软任务共同求解，保留纠偏slack原界限与法向行，不先冻结其最优值。退出进度与末端路径作为后续任务，只使用联合层剩余能力；若清离受阻就保留失败，不能暗中加力或改评价阈值。
4. 求解器负责给候选，完整原问题的可行性、KKT和实际发出动作负责决定是否接受。已认证部分与全部完成分开；不能因内部库报成功就跳过验收。
5. 沿用V21清手后的退出路径记忆，避免恢复整手OPEN扫描。稳定交付仍由固定v3证据评价；未知反馈不补零，不新增有利总分。

新文件位于独立v22_joint_task_fusion，原170来源未变。当前实现仅为联合代数任务bundle，尚无V22 runtime接线、完整退出层或可视化切换。

原型变量z=[x21,ηP]，联合目标为：

`J(z)=||A_normal z-b_normal||² + (1/M)||A_motion z-b_motion||²`。

normal项保留原压力份额平方根因子；motion项按已观测行取均方，两项系数固定为1。它们具有明确的无量纲参考动作尺度，但不表示物理效用相等；这是公开候选选择，未扫描权重或按结果再选系数。原型保持V21同一motion预算，以区分联合协调与预算变化的影响。未观测运动的分支尚未实现，不能用空行补成静止任务。

固定16历史输入的首轮原型结果：16/16原域可行，8/16严格KKT通过，8/16仅可行未获最优性确认；所有未通过项保留。另有一个可解析的冲突任务小例验证共同最优为0.5。数值可靠性尚未达接入门，不能把代理目标的下降直接写成物理优势。

## 接入与同条件对照

先完成同问题求解完整度及实际发出动作的验收，再接入物理。计划三个角色：A保留冻结V21；B仅更换通过验收的后端，保持V21任务定义；C与B同后端但使用联合任务层。先沿原nominal/transport/619741、fragile/hold/619741、friction/transport/619742三个条件，A三条明确作为历史保留，B/C六条新运行。记录不同合法L1锚点造成的后层预算变化，不能假定B全部后层问题逐值相同。

评价沿原任务、力限、清离、低运动驻留、冲击、耗时、全程峰/超限和释放TV；不筛失败或改阈值。沿原扩展门：候选任务3/3、清离3/3、floor0、forceguard至少2/3、末段驻留至少1/3；通过只允许扩大开发，不证明综合胜出。仍须看相对V21和PI的完整收益及代价，再决定完整矩阵和未见种子确认。

## 可检查证据

- [V21完整16物理结果](V21完整16条件综合对照与数值重构依据_20261105.md)
- [固定语料与选择分母](review/V21同问题数值语料schema_v1.md)
- [同问题数值对照65条](review/v21_same_problem_oracle_results_v1.json)
- [仿射表示对照65条](review/v21_affine_quotient_oracle_results_v1.json)
- [L3原点与真实成功末点独立核查](review/V21运动层原点与已成功末层正对照核查_v1.md)
- [只取消退出任务冻结](review/v21_motion_priority_counterfactual_v1.json)
- [纠偏与运动取舍反事实](review/v21_normal_motion_tradeoff_counterfactual_v1.json)
- [联合任务原型源码](v22_joint_task_fusion/task_bundle.py)
- [联合原型固定16离线结果](review/v22_joint_bundle_offline_v1.json)

数值接口核对参考SciPy官方[linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)及[SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。本机实际SciPy1.18.1，文档页1.18.0；使用接口另经本机实际调用核验。仿真venv没有SciPy，离线运行使用已有research312并仅在该子进程前置其Library/bin；裸解释器原生退出及隔离模式import失败的旧尝试保留，未安装或全局修改环境。
'''
(ROOT/name).write_text(text,encoding='utf-8')
assert all((ROOT/p).is_file() for p in re.findall(r'\]\(([^)]+)\)',text) if not p.startswith('https://'))
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
hist=ROOT/'report/progress_history';hist.mkdir(parents=True,exist_ok=True)
for n in ('研究进度.json','当前进度与阶段汇报.md'):shutil.copyfile(ROOT/n,hist/(stamp+'_'+n))
d=read('研究进度.json')
d.update(updated_utc=now,goal_status='active',superiority_proven=False,latest_stage_report=name,
    latest_report=name,latest_algorithm_report=name,current_work='V22 joint-task algebra prototype implemented and offline16 tested; strict original-problem certification/backend remains incomplete, no V22 physics dispatched.',
    previous_goal_turn_classification='progress: full16 physical evidence frozen; fixed65 numerical comparison and16+16 counterfactuals changed architecture, independent dual proof and new joint prototype implemented.')
d['v22_current']=dict(status='joint_task_prototype_offline16_complete_backend_certification_in_progress',
    implemented_algebra_prototype=True,runtime_connected=False,physics_active=False,physical_runs=0,
    source_namespace='v22_joint_task_fusion',source_sha256=joint['prototype_source_sha256'],
    historical_source170_unchanged=True,offline_samples=16,offline_primal_passed=16,offline_KKT_passed=8,
    offline_proxy_motion_change_median_pct=motion_change,offline_result='review/v22_joint_bundle_offline_v1.json',
    report=name,superiority_proven=False)
save('研究进度.json',d)
(ROOT/'当前进度与阶段汇报.md').write_text(f'''# V22联合任务原型已实现，正在解决严格数值验收

更新UTC：{now}。V21完整16物理结果已冻结，无物理仿真在后台运行，主目标active。

本轮65条固定同问题对照及32个任务反事实表明：只换求解库或交换退出/运动层序不足以解释问题；应让几何纠偏软任务与可观测运动共同协调。

独立V22 task_bundle原型已写入，固定16历史输入16可行、8严格KKT，运动代理误差配对下降中位约{abs(motion_change):.1f}%，几何纠偏代理代价小幅增加。它尚未接入物理，不能说完成率或真实力波动已进一步改善。

[本轮完整重构依据与离线结果]({name})；[V21完整16物理结果](V21完整16条件综合对照与数值重构依据_20261105.md)。

原170来源、旧结果和原进度字节备份均保留。下一步先完成原问题验收及完整任务bundle接线，再按同后端/同条件分离数值与联合任务贡献。
''',encoding='utf-8')
print(json.dumps({'report':name,'source170_unchanged':True,'goal_status':'active','physics_active':False,'offline16_KKT':8},ensure_ascii=False))

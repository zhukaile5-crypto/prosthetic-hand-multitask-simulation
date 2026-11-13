"""Build the resumed-run scientific report from the immutable verified summary."""
from pathlib import Path
import json, hashlib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/confirmation_combined/analysis/summary.json'
S = json.loads(SOURCE.read_text(encoding='utf-8'))
BINDINGS = []

def link(label, relative):
    return f'[{label}]({(ROOT / relative).as_posix()})'

def cmp(split, endpoint, cohort=None):
    rows = [r for r in S['comparisons'] if r['split'] == split and
            r['domain'] == 'main' and r['method'] == 'unified' and
            r['endpoint'] == endpoint and r['role'] != 'sensitivity']
    if cohort is not None:
        rows = [r for r in rows if r['cohort'] == cohort]
    assert len(rows) == 1, (split, endpoint, rows)
    BINDINGS.append(rows[0])
    return rows[0]

def pair(split, endpoint, places=3):
    r = cmp(split, endpoint)
    return f"{r['baseline_mean']:.{places}f} → {r['candidate_mean']:.{places}f}"

def counts(split, metric, denominator='trials'):
    d = S['summaries'][split]['main']
    return ' → '.join(f"{d[m][metric]}/{d[m][denominator]}" for m in ('ratio', 'unified'))

def ci(split, endpoint, scale=1., places=3):
    r = cmp(split, endpoint)
    return f"{r['difference']*scale:+.{places}f} [{r['ci_low']*scale:+.{places}f}, {r['ci_high']*scale:+.{places}f}]"

def percent(split, endpoint):
    return abs(cmp(split, endpoint)['relative_percent'])

def build():
    doc = f'''# 统一人工反射：整体重构与独立仿真报告

本轮整体决策重构、2240次同条件确认、综合评价和独立核验均已完成。当前方案降低了平均握力和超力累积，却带来更大的逐指力波动，任务完成率非劣也未成立。**因此，这一轮得到的是明确的性能权衡，不能表述为整体优于基线。** 用户提出的“力波动可能更低”是合理的待检验假设，本轮数据没有支持它。

这份报告先说明项目的统一逻辑，再呈现实验全貌和下一步整体路线。运行代码、参数及评价定义继续冻结；本次恢复工作分析已有确认轨迹，没有重跑或在确认结果上调参。

## 1. 项目逻辑：把支持、保护、平稳与意图放进一个闭环

项目的目标是让抓握在多种任务同时发生时保持协调：既能建立和维持支持，又避免过度施力，在扰动后恢复，并服从用户释放。成功率是必要门槛，平稳、响应和施力是同时需要评价的品质。

当前已经实施的控制链为：

**用户任务 → 共同接触状态估计 → 统一短时力轨迹协调 → 共同柔顺执行 → 新触觉反馈。**

共同估计器用执行历史对齐延迟触觉，区分请求参考力、当前估计力与实际物理力。没有接触时保持搜索状态，不将自由闭合当作已经产生握力；载荷由触觉和经裸手标定的搬运电机观测融合。控制器不读取物体真值位姿、真实质量与摩擦或未来扰动。

统一协调器每20 ms规划三个时刻的五指力参考，将少施力、参考/实际预测力/动作连续性和接触法向合力及力矩协调放进同一个目标，受逐指压力、总力、动作域、支持接触和摩擦储备约束。支持储备可记录未满足量，但不放松限力与动作域，也不将松弛量称作物理不可达证明。释放具有最高执行权限。公共执行器将参考变成真实闭合动作；评价使用未经裁剪的物理接触力。

约束作用于预测模型，不能保证真实碰撞中绝不超限；独立评价会完整记录实际超限。这已是统一状态、统一决策与共同执行的整体实现。当前短时预测主要覆盖法向力与执行响应，接触运动预测的不足在第6节讨论。方程及实现边界见{link('控制实现说明', 'controller_implementation.md')}和{link('实施与实验协议', '实施与实验协议.md')}。

## 2. 仿真和对照的实际范围

公共模型是MuJoCo中**五个有限质量柔顺接触垫与自由刚体**，覆盖柱体、盒体和球体；物理积分2000 Hz、执行与真值200 Hz、决策50 Hz。它已经通过物理接触、响应、饱和、释放与步长敏感性验收，详见{link('物理验收', 'review/physical_acceptance.md')}。完整Shadow模型尚未通过共同物理验收，其失败记录保留，未进入本轮确认。本轮结果不能替代完整多关节假肢验证，也不能与此前87.3%等不同物理模型的绝对完成率直接比较。

七种方法包括固定握力、摩擦比基线、独立反射叠加、完整统一协调，以及去连续性、去几何耦合、固定任务权重三项消融。摩擦比基线是已有文献原理在公共五触点模型中的迁移，属于概念对照，不是原论文装置和数值结果的精确复现。各方法共享观测、动作权限、执行内环与释放权；各主方法同样开发预算选择参数，消融继承完整方案参数。

确认前已冻结全部方法、场景、参数、种子与评价。测试域16场景×7方法×12种子=1344次，参数随机化鲁棒域16×7×8=896次，共2240次；对应方法间实际物理参数、指令、噪声、延迟与丢包序列逐一配对。主比较包含15个常规场景，预先定义的极低摩擦高载荷能力边界单列，仍保留在全场景表中；碰撞等失败场景没有从主域删除。

## 3. 综合评价：先看门槛，再看品质

没有用可相互抵消的加权总分。任务完成、实际限力、释放和有效窗口覆盖单独判断；在这些条件公开的基础上比较响应、力波动与平均施力。

| 评价轴 | 具体定义与解释 |
|---|---|
| 任务与保持 | 完整任务完成、建立失败和建立后失抓；暂停或降级不能算原任务成功 |
| 实际力保护 | 全物理子步逐指/总力合格，另列逐指超限积分之和及总预算超限积分；两个积分有重叠，不相加 |
| 响应 | 首次稳定受限等待A_S，截止8.9 s；未首次稳定贡献完整期限。它不是整段任务失稳时长，首次稳定后失抓仍由任务门槛记录 |
| 持握平稳 | 固定8.2–8.9 s共同有效窗的实际逐指TV，另列逐指和总力SD；逐指TV避免指间波动互相抵消 |
| 施力与意图 | 静态平均总法向力、用户命令后释放完成和延迟；失抓后的零力不算平稳或释放成功 |

TV按预定50 Hz/50 ms离线对称FIR处理完整实际力，再统一200 Hz计量；峰值、稳定判定与超限积分不滤波。释放必须在命令前仍有有效支持，随后连续解除目标物接触，所有预定释放任务保留为分母。

置信区间使用5000次完整种子簇bootstrap，两个主要品质端点A_S和TV在各参数域进行精确簇符号置换与Holm校正。预定工程非劣容差为完成、限力、释放、覆盖最多退化5个百分点，超限积分分别最多增加0.02 N·s；它们不是临床或材料安全阈值。只有12/8个独立种子簇，结论属于本公共基准的初轮确认。

## 4. 主结果：降低施力，但平稳和可靠性仍付出代价

下表顺序均为**摩擦比基线 → 完整统一方案**。任务等分母为测试180、鲁棒120；力品质比较为双方共同有效的180/117对，鲁棒三次统一方案未建立抓握仍留在全任务与A_S分母中。

| 指标 | 测试域 | 鲁棒域 |
|---|---:|---:|
| 完整任务完成 | {counts('test','successes')} | {counts('robustness','successes')} |
| 完成率 | {pair('test','task_success',4)} | {pair('robustness','task_success',4)} |
| 全子步实际力合格 | {counts('test','force_protection_passes')} | {counts('robustness','force_protection_passes')} |
| 首次稳定受限等待A_S，s | {pair('test','restricted_unstable_time')} | {pair('robustness','restricted_unstable_time')} |
| 静态逐指TV，N/s | {pair('test','static.finger_tv_rate')} | {pair('robustness','static.finger_tv_rate')} |
| 静态平均总握力，N | {pair('test','static.mean_force')} | {pair('robustness','static.mean_force')} |
| 逐指超限积分之和均值，N·s | {pair('test','safety.local_excess_integral',4)} | {pair('robustness','safety.local_excess_integral',4)} |
| 总预算超限积分均值，N·s | {pair('test','safety.total_excess_integral',4)} | {pair('robustness','safety.total_excess_integral',4)} |
| 固定静态窗口覆盖 | {counts('test','static_eligible')} | {counts('robustness','static_eligible')} |
| 有效暴露后的释放完成 / 全预定释放 | {counts('test','release_successes','release_trials')} | {counts('robustness','release_successes','release_trials')} |

完成率一行用比例表示。平均握力分别减少{percent('test','static.mean_force'):.2f}%和{percent('robustness','static.mean_force'):.2f}%；总力超限累积减少{percent('test','safety.total_excess_integral'):.2f}%和{percent('robustness','safety.total_excess_integral'):.2f}%。平均施力收益基于共同有效静态窗，超限积分基于主域全部完整任务；两者不能解释为能耗、材料损伤改善或每个场景都有收益。严格实际力合格次数反而略低，因为它检查整次任务是否出现任何规定超限，和超限累积量回答不同问题。

**逐指TV约为基线的4.45倍和4.50倍，波动更大。** 逐指与总力SD也都增加。当前结果没有更快稳定优势；测试A_S相同，鲁棒点估计更慢。主要差异与区间如下，差值均为统一方案减基线：

| 差异及95%簇bootstrap区间 | 测试域 | 鲁棒域 |
|---|---:|---:|
| 完成率，百分点 | {ci('test','task_success',100,2)} | {ci('robustness','task_success',100,2)} |
| 首次稳定等待，s | {ci('test','restricted_unstable_time')} | {ci('robustness','restricted_unstable_time')} |
| 逐指TV，N/s | {ci('test','static.finger_tv_rate')} | {ci('robustness','static.finger_tv_rate')} |
| 平均总握力，N | {ci('test','static.mean_force')} | {ci('robustness','static.mean_force')} |

完成率区间均越过预定−5个百分点容差，非劣尚未获得支持。实际力合格的工程非劣门槛得到支持，仍不能说合格率提高。TV的Holm p分别为0.00098、0.01563；鲁棒A_S差值区间虽在零以上，其8簇精确检验经Holm后p=0.25，不能只选其中一种方法宣称显著变化。

双方完整任务均成功且主窗有效的敏感性队列中，TV差仍为+0.824/+1.059 N/s；预定8.3–8.8 s短窗口为+0.703/+0.900 N/s。波动结论没有因两种敏感性口径而改变。全部恢复事件与失败结局见{link('完整统计附表', 'output/confirmation_combined/analysis/综合评价结果.md')}，没有用恢复成功者的条件均值代替全任务结果。

![配对响应与力涨落权衡]({(ROOT / 'output/confirmation_combined/figures_verified/响应与力涨落权衡.png').as_posix()})

七方法全景图采用各方法有效窗描述值，正式差异使用上表共同队列，鲁棒域不同队列均值不能直接相减：{link('测试全景图', 'output/confirmation_combined/figures_verified/全景五轴_test.png')}、{link('鲁棒全景图', 'output/confirmation_combined/figures_verified/全景五轴_robustness.png')}；同目录有SVG导出。

## 5. 失败场景与消融：完整保留不利证据

以下为所有16场景的完整任务完成数，每格均为基线 → 统一方案；测试每方法每场景12次，鲁棒8次。

| 场景 | 测试 / 12 | 鲁棒 / 8 |
|---|---:|---:|
'''
    plan = json.loads((ROOT/'review/confirmation_plan.json').read_text(encoding='utf-8'))
    for scene in plan['scenarios']:
        cells = []
        for split in ('test', 'robustness'):
            d = S['scenarios'][split][scene]
            cells.append(f"{d['ratio']['successes']} → {d['unified']['successes']}")
        title = S['scenarios']['test'][scene]['ratio']['title']
        if scene == 'infeasible':
            title += '（预定边界，单列）'
        doc += f"| {title} | {cells[0]} | {cells[1]} |\n"
    doc += '''
碰撞、柔顺接触组合扰动在两个参数域双方均全部失败；防滑限力并发、局部受限拉动和重盒体造成进一步差额。鲁棒重盒体统一方案3次未按期限建立，不能记作稳定零力。能力边界双方也全部失败。把边界纳入全16场景后，测试为142/192 → 137/192，鲁棒101/128 → 95/128；常规场景的失败没有被主分析掩盖。

三项消融检查组件作用，下面的TV和平均力是各方法自身有效窗描述值，未构成新的确认性多方法优越检验：

| 方法 | 测试完成 / 180 | 鲁棒完成 / 120 | 测试TV / 鲁棒TV，N/s | 测试均力 / 鲁棒均力，N |
|---|---:|---:|---:|---:|
'''
    titles = {'unified':'完整统一方案','unified_no_continuity':'去连续性代价','unified_no_geometry':'去接触几何耦合','unified_fixed_stage':'固定任务权重','independent':'独立反射叠加'}
    for method, title in titles.items():
        t = S['summaries']['test']['main'][method]
        r = S['summaries']['robustness']['main'][method]
        doc += f"| {title} | {t['successes']} | {r['successes']} | {t['finger_tv_eligible_mean']:.3f} / {r['finger_tv_eligible_mean']:.3f} | {t['mean_force_eligible_mean']:.3f} / {r['mean_force_eligible_mean']:.3f} |\n"
    doc += f'''
完整方案的有效窗TV描述均值低于去连续性版本，支持连续性机制的作用线索，但仍高于基线。固定权重版本波动明显较小，鲁棒完成数也较高，同时施力更多，提示阶段参考与目标组织值得整体重审。几何消融方向混合，不能声称各组件都有独立普遍收益。独立反射在均力、波动与限力上也有竞争力，平均少力并非统一协调独有的成果。

## 6. 从证据到下一版整体路线

正式盒体13101的指定静态窗中，统一方案实际逐指TV为7.2365 N/s、基线0.14445；当前力估计对实际力RMSE仅0.00568 N，内外求解均正常、五接触持续有效，仍伴随请求力参考反复变化。该例统一方案平均力12.882 N，还高于基线9.997 N。盒体约贡献全域TV差额的59.6%/62.9%，其余场景仍有同向差异。这是观测与算术分解，不能直接当作机制因果证明。

源码与消融提示两个相互关联的结构不足：持握阶段缺少持续的稳定平衡参考，最小施力目标可能沿不断更新的防滑需求边界重新分配；预测传播了法向力与动作，却没有联合预测切向负载迁移和相对运动。估计准、优化可解，都不能替代对持续稳定接触的表达。证据与假说边界详见{link('机制与架构诊断', 'review/机制与架构诊断.md')}。

下一版只沿一条整体路线设计：**共同接触运动估计 → 稳健稳定支持走廊 → 走廊内连续力轨迹 → 次级少力优化。** 这里“走廊”是可持续支持的接触与运动状态范围。

1. 在共同时间模型中融合压力、切向力、接触几何、已知触点运动、局部滑移与电机载荷；仅估计可观测的相对运动分量，其余保留不确定范围，禁止引入评价真值。
2. 联合预测动作引起的法向力、切向接触响应、载荷再分配和相对运动；依据可信平衡及摩擦/载荷不确定范围建立稳定支持条件，而非追随当前指力重分配。
3. 在同一优化中表达支持、任务运动误差、相对速度、实际力及动作边界，并明确终端持续支持要求。目标顺序是先保证任务与稳定可行，再维持连续分配，最后降低施力；偏离范围时统一恢复，用户释放继续优先。
4. 先做共同观测模型的可辨识与响应标定，再以全新开发种子检验整个生命周期；基线共享相同新观测和执行能力，同预算选参后冻结，用新独立种子复核原综合门槛及品质。现有2240次结果作为已公开诊断历史，不能重用为调参后的独立确认。

**该下一版尚未实施或测试。** 本次交付是当前整体重构的完整确认与下一版体系设计依据。完整Shadow迁移还需要独立完成多关节可达性、接触响应、运动观测和全手动力学校准。

## 7. 证据、交付与复现边界

独立核验逐一检查2240条记录与轨迹哈希、实际配对参数和噪声序列，独立重算10836个描述值与456项配对统计；原始力波形计量抽样64条，全部140条预定释放原始轨迹另行核验。冻结源、设置、18份快照、登记的147个旧项目文件和60个物理证据文件哈希未变化。64条力波形抽样不是全部2240条波形的独立重算，147项也不是旧目录每一文件的扫描。

此前开发中V1基线18条校准波形及摘要曾被误覆盖，现按{link('开发记录完整性说明', 'review/development_record_integrity.json')}公开为丢失/替代；本次核验通过不表示恢复。旧正式结果、360条V1比较及本轮冻结确认不受该覆盖影响。

- {link('独立核验结论', 'review/final_result_audit.md')}与{link('逐项机器证据', 'review/final_result_audit.json')}。
- {link('冻结协议与源码清单', 'review/confirmation_plan.json')}；{link('恢复后输入核验', 'review/resumed_input_verification.json')}。
- {link('正式数据摘要', 'output/confirmation_combined/analysis/summary.json')}、{link('完整统计附表', 'output/confirmation_combined/analysis/综合评价结果.md')}与{link('合并来源', 'output/confirmation_combined/combined_manifest.json')}；原始两域JSON/NPZ保持原样。
- {link('失败与覆盖图', 'output/confirmation_combined/figures_verified/失败与覆盖.png')}，{link('图表数据和来源哈希', 'output/confirmation_combined/figures_verified/figure_provenance.json')}。
- {link('保存状态几何回放', 'output/confirmation_replay_nominal_13101/paired_saved_state_replay.gif')}只恢复已保存状态，不执行新控制/动力学，不独立证明算法效能。
- {link('完整复现入口', 'reproduce_all.ps1')}用于另开输出目录复现冻结矩阵；本次恢复没有运行它，也未启动第三版测试。

报告表格直接绑定冻结summary中的主配对数值，绑定来源见{link('报告数字来源', 'review/report_numeric_bindings.json')}。
'''
    (ROOT / '全景实验报告.md').write_text(doc, encoding='utf-8')
    manifest = {'summary_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'report_sha256': hashlib.sha256((ROOT/'全景实验报告.md').read_bytes()).hexdigest(),
                'source': SOURCE.as_posix(), 'comparisons_used': BINDINGS,
                'descriptive_source': 'summaries[split].main[method] and scenarios[split][scene][method]',
                'scope': 'render only; no new simulation, parameter selection, or statistic definition'}
    (ROOT/'review/report_numeric_bindings.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report':(ROOT/'全景实验报告.md').as_posix(),'characters':len(doc),'bindings':len(BINDINGS)},ensure_ascii=False))

if __name__ == '__main__':
    build()

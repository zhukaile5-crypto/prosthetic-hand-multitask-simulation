"""Create a source-backed project report from completed, frozen analysis."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
ANALYSIS=ROOT/'output/final_analysis'


def number(value,digits=2):
    return '缺失' if value is None else f'{value:.{digits}f}'


def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |',
                       '| '+' | '.join(['---']*len(header))+' |',
                       *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def main():
    summary=json.loads((ANALYSIS/'summary.json').read_text(encoding='utf-8'))
    effects_file=json.loads((ANALYSIS/'paired_effects.json').read_text(encoding='utf-8'))
    effects=effects_file['effects']
    metadata=summary['metadata'];titles=metadata['method_titles']
    assert metadata==effects_file['metadata']
    plan=ROOT/'review_2026-10-13/formal_plan.json'
    frozen_plan=json.loads(plan.read_text(encoding='utf-8'))
    assert metadata['analysis_script_sha256']==frozen_plan['analysis_sha256']
    audits={};raw={}
    for suite,name in (('test','test_holdout_v1'),('robustness','robustness_holdout_v1')):
        directory=ROOT/'output'/name
        audits[suite]=json.loads((directory/'audit.json').read_text(encoding='utf-8'))
        audit=audits[suite]
        assert audit['passed'] and audit['repeatability']['identical_physical_and_control_results']
        assert audit['results']['independently_recomputed_trace_integrals']==(2560 if suite=='test' else 1280)
        assert hashlib.sha256((directory/'trials.json').read_bytes()).hexdigest()==metadata['inputs'][suite]['trials_sha256']
        raw[suite]=json.loads((directory/'trials.json').read_text(encoding='utf-8'))
    recovery=ROOT/'output/resume_recovery_v3/audit.json'
    assert json.loads(recovery.read_text(encoding='utf-8'))['passed']
    rows=summary['domains']
    get=lambda suite,method,domain='main': next(r for r in rows if (r['suite'],r['method'],r['domain'])==(suite,method,domain))
    select=lambda suite,comparator,metric,condition='all': next(r for r in effects if (r['suite'],r['domain'],r['comparator'],r['metric'],r['condition'])==(suite,'main',comparator,metric,condition))
    primary=[select('test',m,'safe_completed') for m in ('constant','event','ratio','adaptive')]
    advantages=[r for r in primary if r['estimate']>0 and r['p_value_holm']<.05]
    disadvantages=[r for r in primary if r['estimate']<0 and r['p_value_holm']<.05]
    uncertain=[r for r in primary if r['p_value_holm']>=.05]
    if len(advantages)==4:
        conclusion='本轮预先声明的主指标上，融合版对四个基线均有正向配对差值，且通过四比较 Holm 校正。该结果仅覆盖当前模拟模型、场景与参数范围。'
    elif advantages:
        conclusion='本轮融合版对'+ '、'.join(titles[r['comparator']] for r in advantages)+'的主指标有经校正的优势；对其余基线不能据此宣称全面胜出。'
    else:
        conclusion='本轮正式主指标未证明完整融合全面优于四个基线。已完成可复现的基线、融合实现与同条件比较，优势和代价应按下列分指标结果解释。'
    if disadvantages:
        conclusion+=' 同一主指标上，融合版相对'+ '、'.join(titles[r['comparator']] for r in disadvantages)+'有经校正的劣势，必须保留这一退步。'
    if uncertain:
        conclusion+=' 与'+ '、'.join(titles[r['comparator']] for r in uncertain)+'的差异尚未由本轮检验确定。'
    chunks=['# 多任务融合控制的改进与对照结果',
            '日期：2026年10月13日。'+conclusion,
            '本项目将防滑、五指限力和用户释放放入同一任务框架。主要比较是全程五指限力合格任务完成率；判据为任一指超过上限 0.25 N 或五指合力超过预算 0.5 N，持续 30 ms。短时高峰仍可能被这个二元指标判合格，因此峰值和超限积分必须并报。滑移、超力暴露和响应时间分别报告，不用任意加权总分替代失败类型。',
            '## 完成的工作',
            '扫描并核对开题方案与现有实验后，保留原演示、PPT及参考文献。修正公共内环的 1.45 N 姿态跳变和掌部摩擦遗漏；修正跨阶段超限与姿态 RMS；加强触觉有效性、稳定初始化与耦合抗饱和。Matos 基线补齐实时五样本力比反馈。具体机制与下一阶段路线见 [改进路线与实现](改进路线与实现.md)。',
            '文献对照采用 Matos（滑移／摩擦比）、Gautham（事件增力）、Zhang（逐指自适应摩擦锥），并保留固定握力和三个消融。各论文已有部分融合或逐指控制；当前贡献聚焦冲突协调的工程质量。硬件、传感器与执行器均经过迁移，不能称原论文完整复现。见 [文献基线核对](review_2026-10-13/文献基线核对.md)。',
            '## 实验范围与冻结',
            '开发阶段先保留旧结果，再完成修订后的 36 候选首轮校准和 84 候选扩展校准。最终固定／事件／摩擦比／逐指／融合的候选数分别为 4／8／24／24／24，对应 72／144／432／432／432 次开发试验，计算预算不相等。扩展为后三类方法补上同样的低基础力搜索，参数维度不同仍使搜索密度不同；所有候选及结果完整保存，不能称等计算预算或各方法全局最优。',
            '整定先比较全程限力合格完成，再比较掉落／初始失败；第三级平局排序用逐指与预算两项超限积分之和，最后比较中位偏移。这个和仅是开发工程惩罚，分量有重叠，不是独立物理冲量；正式分析把两项分开。正式或扰动结果没有参与参数选择。',
            '正式测试为 16 场景 × 20 种子 × 8 方法 = 2560 次；额外参数扰动为 16 × 10 × 8 = 1280 次，共 3840 次。主域预先排除一个极低摩擦高载荷边界，得到每方法正式 300 次、扰动 150 次；边界和全 16 场景完整另报。每个种子跨场景共享随机参数结构，区间以种子作为配对簇，不能将 300 次当作 300 个独立人体样本。',
            '所有方法共用 Shadow Hand 接触动力学、连续力内环、逐采样噪声／延迟／丢包、释放指令及限力配置。物体自由运动，无手物焊接。物理步长 1 ms，控制周期 5 ms。控制器不接收质量、真实摩擦、未来扰动或评价位姿。真值只用于共同理想化传感器生成和独立评价。',
            '## 正式与扰动总体结果']
    for suite,label in (('test','正式留出'),('robustness','额外参数扰动')):
        entries=[]
        for method in metadata['methods']:
            r=get(suite,method);c=r['counts'];v=r['continuous']
            entries.append([titles[method],f"{c['safe_completed']}/{r['n']} ({r['rates_pct']['safe_completed']:.1f}%)",
                            f"{c['completed']}/{r['n']}",c['drop'],c['setup_failure'],
                            number(v['max_slip_mm']['median']),number(v['total_force_excess_Ns']['mean'],4),
                            number(v['total_budget_excess_Ns']['mean'],4)])
        chunks += [f'### {label}主域',table(['方法','限力合格完成','完成','掉落','初始失败','最大偏移中位数 mm','逐指超限均值 Ns','预算超限均值 Ns'],entries)]
    chunks += ['超限均值涵盖抓握与持握全程，逐指与预算两种积分有重叠，不能把它们加成一个物理冲量。偏移中位数仅来自抓握建立的样本；失败分母保留在完成率中。低力可能来自失去接触，必须连同掉落和初始失败一起解释。',
               '## 主指标的配对检验',
               '下表差值为融合减基线，单位是百分点。95% 区间为 10000 次配对种子簇重采样，区间本身未做多重校正；p 值为种子层面方法交换检验的四基线 Holm 校正，该检验假设零假设下配对方法可交换。扰动集属于次要验证。']
    for suite,label in (('test','正式'),('robustness','扰动')):
        entries=[]
        for method in ('constant','event','ratio','adaptive'):
            r=select(suite,method,'safe_completed');lo,hi=r['ci95']
            entries.append([titles[method],f"{r['estimate']:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]',f"{r['p_value_holm']:.4g}"])
        chunks += [f'### {label}配对结果',table(['比较基线','融合差值 pp','95% 区间 pp','校正 p'],entries)]
    chunks += ['## 连续指标及消融',
               '以下均为解释性分析。对偏移仅比较两方法均建立抓握的配对样本；对超限同时给出全试验与两方法均完成任务的结果。两种条件都受算法结果影响、存在选择偏差，不能以这些筛选样本取代总体完成率或推断全部任务的因果效果。']
    entries=[]
    for method in ('constant','event','ratio','adaptive'):
        slip=select('test',method,'max_slip_mm','both_setup')
        force=select('test',method,'total_force_excess_Ns')
        matched=select('test',method,'total_force_excess_Ns','jointly_completed')
        entries.append([titles[method],f"{number(slip['estimate'])} [{number(slip['ci95'][0])}, {number(slip['ci95'][1])}] (n={slip['n_pairs']})",
                        f"{number(force['estimate'],4)} [{number(force['ci95'][0],4)}, {number(force['ci95'][1],4)}]",
                        f"{number(matched['estimate'],4)} [{number(matched['ci95'][0],4)}, {number(matched['ci95'][1],4)}] (n={matched['n_pairs']})"])
    chunks += [table(['基线','配对平均偏移差 mm','全程逐指超限差 Ns','共同完成的逐指超限差 Ns'],entries)]
    entries=[]
    for method in ('constant','event','ratio','adaptive'):
        full=select('test',method,'total_budget_excess_Ns')
        matched=select('test',method,'total_budget_excess_Ns','jointly_completed')
        entries.append([titles[method],f"{number(full['estimate'],4)} [{number(full['ci95'][0],4)}, {number(full['ci95'][1],4)}]",
                        f"{number(matched['estimate'],4)} [{number(matched['ci95'][0],4)}, {number(matched['ci95'][1],4)}] (n={matched['n_pairs']})"])
    chunks += [table(['基线','全程预算超限差 Ns','共同完成的预算超限差 Ns'],entries)]
    entries=[]
    for method in ('fusion_no_prediction','fusion_no_distribution','fusion_no_filter'):
        safe=select('test',method,'safe_completed');slip=select('test',method,'max_slip_mm','both_setup')
        entries.append([titles[method],f"{safe['estimate']:+.2f} [{safe['ci95'][0]:+.2f}, {safe['ci95'][1]:+.2f}]",
                        f"{number(slip['estimate'])} [{number(slip['ci95'][0])}, {number(slip['ci95'][1])}]"])
    chunks += [table(['完整融合对照消融','限力合格完成差 pp','偏移差 mm'],entries),
               '消融沿用完整融合的选定参数，未针对各消融重新调参；结论仅是本冻结配置下的净收益，不能断言该机制普遍有效或无效。分配消融改用固定指力比例，但仍保留逐指上限及总预算投影。',
               '## 分场景完整结果']
    titles_by_scene={s['name']:s['title'] for s in metadata['scenario_definitions']}
    for suite,label in (('test','正式'),('robustness','扰动')):
        entries=[]
        for scene in metadata['scenario_definitions']:
            values=[]
            for method in ('constant','event','ratio','adaptive','fusion'):
                r=next(r for r in summary['scenarios'] if (r['suite'],r['scenario'],r['method'])==(suite,scene['name'],method))
                values.append(f"{r['counts']['safe_completed']}/{r['n']}")
            entries.append([scene['title']+('（边界）' if scene['boundary'] else ''),*values])
        chunks += [f'### {label}五指限力合格完成数',table(['场景','固定','事件','摩擦比','逐指','融合'],entries)]
    chunks += ['## 释放响应、接触负荷与计算耗时',
               '释放延迟表只使用已建立抓握、释放前未确认掉落且测得清除的试验，列出有效数量并保留全部释放任务完成数。它不代表安全落置速度。控制耗时是“单次试验内 p95”的跨试验中位数／最大值，来自并行 CPU 计算，不能当作总体采样 p95 或硬件最坏闭环延迟。']
    for suite,label in (('test','正式'),('robustness','扰动')):
        entries=[]
        for method in ('constant','event','ratio','adaptive','fusion'):
            release=[r for r in raw[suite] if r['scenario']=='release_slip' and r['method']==method]
            latencies=[r['release_latency_s'] for r in release if r['setup_ok'] and not r['dropped'] and r['release_latency_s'] is not None]
            stat=get(suite,method)['continuous']['controller_p95_ms']
            entries.append([titles[method],f"{sum(r['completed'] for r in release)}/{len(release)}",
                            f'{len(latencies)}/{len(release)}',number(float(np.median(latencies)) if latencies else None,3),
                            f"{number(stat['median'],3)} / {number(stat['maximum'],3)}"])
        chunks += [f'### {label}释放与计算',table(['方法','释放任务完成','有效延迟数量','延迟中位数 s','试验内控制 p95 中位／最大 ms'],entries)]
        entries=[]
        for method in ('constant','event','ratio','adaptive','fusion'):
            value=get(suite,method)['continuous']
            collision=next(r for r in summary['scenarios'] if (r['suite'],r['scenario'],r['method'])==(suite,'collision',method))
            cv=collision['continuous']
            entries.append([titles[method],number(value['palm_peak_N']['mean']),number(value['palm_force_Ns']['mean']),
                            f"{collision['counts']['setup']}/{collision['n']}",number(cv['collision_peak_N']['mean']),number(cv['collision_force_Ns']['mean'])])
        chunks += [f'### {label}预算外接触负荷',table(['方法','主域掌峰均值 N','主域掌力积分均值 Ns','碰撞场景抓握建立','碰撞场景峰值均值 N','碰撞场景积分均值 Ns'],entries)]
    chunks += ['上述接触负荷也受初始失败及掉落影响，低值不单独构成优势；掌部负荷没有纳入五指限力合格判据。',
               '掉落计数不包含初始失败，两者的联合损失在原始统计中单独保存。数值失败数量如下：'+
               '；'.join(f"{label}全部场景：非有限轨迹 {sum(r['nonfinite'] for r in raw[suite])}，含求解器警告试验 {sum(r['numerical_warnings']>0 for r in raw[suite])}" for suite,label in (('test','正式'),('robustness','扰动')))+'。']
    entries=[]
    for method in ('constant','event','ratio','adaptive','fusion'):
        values=[max(r['initial_force_peak_N'],r['peak_force_N']) for r in raw['test'] if r['method']==method and not r['boundary']]
        entries.append([titles[method],number(float(np.median(values))),number(float(np.max(values)))])
    chunks += ['### 包括抓握阶段的瞬时指力峰值',
               table(['方法','每试验峰值的中位数 N','所有主域试验的最大值 N'],entries),
               '原始接触求解器力仍有极短的高峰，未因求解器无警告而排除。代表种子3001的限力冲突试验中，融合版初始单指峰值为 136.34 N，但持续超限不足 30 ms，最终被当前二元判据判为合格。这说明本轮完成率优势不能解释为避免瞬时冲击或真实损伤。下一阶段应优先标定接触模型和执行器内环，并另设峰值／冲量约束，保留全部异常峰，而不是只继续增加融合模块。']
    chunks += ['## 验证证据与可复现文件',
               '正式与扰动审计核对完整试验笛卡尔矩阵、配对环境、冻结源码与模型资产；从全部保存轨迹独立重算法向超限、预算、连续安全、释放、姿态和掌部／碰撞指标，并各复跑一条试验检验确定性。中断恢复副本验证了复用已完成结果与仅补跑缺失试验。',
               '- [正式逐次数据](output/test_holdout_v1/trials.csv) · [冻结信息](output/test_holdout_v1/manifest.json) · [独立审计](output/test_holdout_v1/audit.json)\n- [扰动逐次数据](output/robustness_holdout_v1/trials.csv) · [独立审计](output/robustness_holdout_v1/audit.json)\n- [配对效应与区间](output/final_analysis/paired_effects.csv) · [分场景统计](output/final_analysis/scenario_summary.csv)\n- [使用说明](使用说明.md) · [实验协议](实验协议.md)',
               '## 解释边界与后续路线',
               '当前力限制只覆盖五指法向力与总预算；掌部、局部压强、软物变形及真实损伤均未受完整验证。碰撞响应位于公共任务层，相关差异不能视作融合独有碰撞反射。球体完整 SO(3) 和轴对称物体绕长轴旋转只作诊断；本轮不主张通用姿态精度改善。释放代表清除接触，尚非安全落置。',
               '随机化覆盖规定的质量、尺寸、摩擦、扰动强度、接触参数、噪声与延迟；扰动时刻使用固定时间表，初始姿态和基本抓型固定，未验证随机事件时刻或意图识别延迟。仿真使用理想化触觉信号，不验证真实传感器物理响应、人体 EMG 识别、认知负荷下降或临床效益。短时预测和投影约束没有形式化稳定性或动力学安全保证。运行时统计来自并行 CPU 实验，不能当作硬件最坏实时延迟。',
               '本轮摩擦比强基线的限力合格完成率略高于完整融合，且融合相对它的全程超力暴露均值更高。下一轮宜以稳健的摩擦比反馈作为主干，仅在明确局部限力冲突时启用分配，重新检验预测通路是否值得保留；同时用共用初始持握状态隔离抓握阶段与持握协调的贡献。这些是下一轮待实施验证的路线，不属于本轮已证实改进。',
               '随后再增加滑移方向、接触位置、支撑力矩和掌部限制，将安全落置、软物变形与触觉硬件误差纳入任务。任何后续调参都要另建测试集，不能继续使用本轮留出证明改进。',
               '## 科学图与实际轨迹示例']
    for filename,label in [('paired_safe_completion','配对完成率差与区间'),('completion_test','正式完整场景矩阵'),('completion_robustness','扰动完整场景矩阵'),('trajectory_fragile_conflict_3001','固定配对种子3001的冲突轨迹'),('trajectory_three_pulses_3001','固定配对种子3001的重复拉动轨迹')]:
        chunks.append(f'![{label}]({(ROOT/"output/final_figures"/(filename+".png")).as_posix()})')
    chunks += ['[三次拉动的真实轨迹回放](output/final_replay/three_pulses__event_vs_fusion__3001.gif) · [限力冲突的真实轨迹回放](output/final_replay/fragile_conflict__event_vs_fusion__3001.gif)。固定种子3001仅作观测示例，不替代完整矩阵。']
    report='\n\n'.join(chunks)+'\n'
    (ROOT/'结果与改进报告.md').write_text(report,encoding='utf-8')
    evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (ANALYSIS/'summary.json',ANALYSIS/'paired_effects.json',Path(__file__),plan,recovery,
               ROOT/'output/test_holdout_v1/audit.json',ROOT/'output/robustness_holdout_v1/audit.json')}
    (ANALYSIS/'report_sources.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(conclusion,flush=True)


if __name__=='__main__':
    main()

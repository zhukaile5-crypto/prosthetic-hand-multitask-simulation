"""Reproducible descriptive readout of the retained first development matrix."""
import json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'output/development_v1/trials.json'
rows=json.loads(source.read_text(encoding='utf-8'))
summary=[]
for method in ('ratio','unified','joint'):
    group=[r for r in rows if r['method']==method]
    summary.append(dict(method=method,trials=len(group),missions=sum(r['task_success'] for r in group),
        static_TV_N_s=float(np.mean([r['finger_tv_rate'] for r in group])),
        mean_finger_force_N=float(np.mean([r['mean_force'] for r in group])),
        whole_force_passes=sum(r['safety']['force_protection_pass'] for r in group),
        holding_local_excess_Ns=float(np.mean([r['safety']['stages']['policy_holding']['local_excess_integral_Ns'] for r in group])),
        physical_releases=sum(r['research_metrics']['release']['physical_success'] for r in group)))
dest=ROOT/'report/development_v1'
dest.mkdir(parents=True,exist_ok=True)
record=dict(phase='development',independent_confirmation=False,physical_seeds=1,
    scenarios=6,rows=summary,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    conclusions=['Joint progress eliminates the old unified transport failure in this development case.',
        'The joint candidate does not outperform the traditional baseline overall; force variation is worse.',
        'Whole-task safety remains poor owing to the inherited release path.',
        'Known uncertainty/pressure-domain errors prevent any formal safety interpretation.'])
(dest/'readout.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 首轮联合控制开发结果','',
    '这是6个代表情景、1个专用开发种子、3种方法的18次开发试验。全部完成，无运行错误，运行期间源码身份一致。不能据此作独立确认或统计泛化。','',
    '| 方法 | 完成数 | 静态逐指TV N/s | 平均指力 N | 全程限力通过 | 持握局部超限均值 N·s | 物理释放 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['method']} | {r['missions']}/{r['trials']} | {r['static_TV_N_s']:.4f} | {r['mean_finger_force_N']:.4f} | {r['whole_force_passes']}/{r['trials']} | {r['holding_local_excess_Ns']:.6f} | {r['physical_releases']}/{r['trials']} |")
lines+=['','新联合决策恢复了本开发运输条件的完成能力，并减少平均施力；其力变化明显高于传统基线，尚未达到总体更优。旧融合版的运输失败仍完成物理释放，表中不把它重复算成机械释放失败。','',
    '已证实常规场景新增TV主要为持续降力。模型的短时力预测误差较小，因此不能简单归咎于预测器噪声；需要核查完整手作用力目标、被动掌反力的响应，以及载荷估计暖机如何进入任务目标。','',
    '独立审查另外发现公开误差向下截断与预测负压力抵扣预算两项正确性缺口，见review/development_v1_review.md。修订必须使用新源码快照与新输出，本轮全部保留。','',
    '下一阶段先补齐数学域约束和反例验证，再通过公共观测的小幅因果标定建立手指动作、被动掌部与完整手作用力之间的关系。正式矩阵、统计和策略尚未冻结，目标保持进行中。']
(dest/'开发结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))

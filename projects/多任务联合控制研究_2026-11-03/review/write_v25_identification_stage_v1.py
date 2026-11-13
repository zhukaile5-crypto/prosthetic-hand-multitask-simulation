"""Publish the actual identification foundation stage, without a policy-win claim."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    data=read('review/v25_first_five_response_v1.json')
    audit=read('review/v25_first5_execution_independent_v1.json')
    plan=read('review/v25_first_five_plan_v2.json')
    complete=read('output/v25_first_five_v2/execution_complete.json')
    bridge=read('output/v25_bridge_fork_v3/bridge_complete.json')
    assert data['complete'] and not data['issues'] and audit['passed'] and not audit['issues']
    assert len(audit['checks'])==105 and all(x['passed'] for x in audit['checks'])
    assert complete['passed'] and complete['execution_complete'] and complete['completed']==complete['planned']==5
    assert bridge['passed'] and bridge['full_remaining_episode_compared']
    assert data['source_sha256']==plan['source_sha256']==complete['source_sha256']==bridge['source_sha256']
    assert len(data['source_sha256'])==198
    for name,digest in data['source_sha256'].items():
        path=ROOT.parent/'五指策略接入_2026-11-03'/name[11:] if name.startswith('foundation/') else ROOT/name
        assert sha(path)==digest,name
    active=('lift_plus','lift_minus','wrj1_minus')
    matrix=np.asarray([row['actual_executed_difference_from_zero21'] for arm in active for row in data['branches'][arm]['actual_receipt_input_rows']])
    rank=int(np.linalg.matrix_rank(matrix));columns=np.flatnonzero(np.any(matrix!=0,axis=0)).tolist()
    assert rank==2 and columns==[18,20]
    availability={arm:sum(z['available'] for pair in data['paired_device_labels'] if pair['arm']==arm for z in pair['digits']) for arm in ('lift_plus','lift_minus','wrj1_plus','wrj1_minus')}
    now=datetime.now(timezone.utc).isoformat();name='V25公开观测与响应识别基础阶段报告_20261106.md'
    assert not (ROOT/name).exists()
    body=f'''# 公开观测与响应识别基础：首组真实受控样本已完成

本轮已实现独立六文件V25基础版本，完成当前来源198的完整16秒被动接入桥、复制剩余轨迹桥，以及首组五支实际受控实验和105项独立证据核查。原平台、父192、设备与物理/数值环境保持。已取得可追溯的公开观测—真实动作样本，尚未拟合或准入响应模型，也尚无新融合策略的性能胜负结论。

报告UTC：{now}。所有本轮物理及读数工具已终态，当前无相关物理仿真后台运行。最终综合优越性目标继续active，未标记完成。

## 全景方案与本轮实施位置

整体路线是：同龄公共观测 → 已确认的实际动作历史 → 经验证的力—运动—接收载荷响应 → 一次多任务联合决策 → 唯一21维执行器 → 实际回执与观测反馈。

```mermaid
flowchart LR
    A[公共触觉、机器人、接收面] --> B[已实现：同龄测量与有效性]
    B --> C[已实现：实际动作与响应样本]
    C -. 待识别与留出验证 .-> D[力与运动响应模型]
    D -. 待实施 .-> E[响应驱动多任务决策]
    E --> F[已有：唯一21维执行器]
    F --> G[已实现：真实回执]
    G --> C
```

本轮完成的是观测与识别基础。测量提取与当前几何提案、B矩阵和控制clip资格分开；一指只用原有一对滑移测量，不把多patch当多份独立信息。保留读出/采集/可用时钟、逐指压力、原slip及补偿同龄机器人速度后的外部投影。未知、部分几何、失接触和演员归属未认证均保持原义。

实际动作只从唯一commit后的typed receipt进入历史。实验目标在原控制器提交前选择，原graph、driver和plant绑定始终属于同一个控制器；原路径/状态记忆与新观测历史使用同一真实回执。名义QP证据仅认证其原候选，改变为探针目标后不冒签实际动作。

后续模型将同时面向公开压力、设备运动投影和接收载荷，将限力、稳定与交接进度放进同一决策问题。当前几何M仍不是实际物体响应。完整路线见[响应驱动融合实施路线](review/V26响应驱动多任务融合整体实施路线_v1.md)。

## 基础接入是否改变原行为

固定情境为nominal / transport / 619741，冻结参考为V24 canonical_joint。公开TRANSFER及receiver fresh streak达到2的读时刻为13.935秒，复制/干预起点固定在下一PRE_OBSERVE边界13.940秒，没有用物体真值或成绩挑入口。

- 当前版本新增观测记录后的完整16秒轨迹，对冻结V24参考92项核对通过。
- 从13.940秒复制的分支继续到16秒，与原分支的完整轨迹92项核对通过。
- 3201个公开运动包、1701个真实回执、1700个相邻回执区间逐项相同。尾端没有凭空补造后续保持区间。
- 仅沿原五个墙钟列和三个debug计时叶保留计时差异，其它物理数值/公共UTF8内容精确一致；来源provenance单列。

这些证明基础接入与本次复制可靠，不证明所有未来对象/接触下的预测或安全。完整198与父192及原42/数值依赖在登记、执行和读数时保持，未安装软件、修改原模型或移动评价阈值。

## 首五支实际结果

全部五支在同一实验起点，记录到14.155秒，覆盖5/10/20/40/100/200毫秒六个预登记采集时点，均在采集后10毫秒公开可读。四支完成200毫秒保持程序，其中一支是固定carrier的zero；第五支原方向权限为0，明确未暴露。

|分支|相对固定zero的实际初始扰动|实际处理|逐指配对可用数|
|---|---:|---|---:|
|zero|0|固定carrier对照|本支六时点29/30|
|lift+|+0.450毫米|完成保持|{availability['lift_plus']}/30|
|lift−|−0.007299毫米|完成保持，原余量很小|{availability['lift_minus']}/30|
|WRJ1+|0|权限为0，未暴露，回到原策略|{availability['wrj1_plus']}/30，仅保留不算轴效应|
|WRJ1−|−0.000300弧度|完成保持|{availability['wrj1_minus']}/30|

三个有效扰动在[13.94,14.14)保持窗内，除所分配的升降/腕轴外，其余坐标相对zero全部精确为0；实际方向支持rank2。200毫秒结束后的恢复/flush三tick另列，不能混入固定保持程序。

未支持的WRJ1+回到原自适应策略，而zero仍保持固定carrier，二者会产生多轴差。若把这一支混进输入矩阵，可得到rank16；这不代表识别了16个方向，也不能作为WRJ1+梯度。首组只有两个分配轴、一个已见物理cluster，其余19列未独立测量，不能设零gain。

## 这批数据改变了什么判断

固定zero与三个有效扰动的逐指配对共有87/90可用；连同未暴露分支的原始文件比较则是105/120。缺的是部分指端投影domain，六时点的整包均已送达，不能把缺行当零运动。全部程序阶段仍为TRANSFER，说明仅以阶段名划分模型也不足以保证接触观测结构固定。

105个可配对设备输出中，仅5个保持完全相同的basis及世界投影；按原点/zero/arm共同条件C的固定函数检查，0/105可恢复。因而直接相减的二维数值只能称设备自身坐标输出变化，不能升级为同一固定物理速度差或完整对象转动反馈。当前未认证这些接触来自同一刚体。

下一模型必须事先明确输出：预测设备实际测量函数时，把基面、接触结构和mask变化纳入定义与误差；若预测固定空间运动函数，则必须另证明该函数在所用观察子空间和归属条件下可恢复。自由演化与动作响应分别验证，未来zero只作离线配对标签，不能喂给在线控制。

这支持继续采用响应驱动的整体路线，尚不足以把两列局部样本接成全21模型。接下来应另登记完整21方向、混合留出及不同公共模式/独立物理seed，验证误差和可迁移范围，再实施联合力—运动—交接决策。

## 验收范围和未完成项

最终有限门包括观测作者38/独审10、fork作者27/独审16、提交链作者45/独审6、完整198零步10项。实际首五支独立审查105项通过，覆盖全分配、同起点、全部200Hz/2kHz前缀、实际21维回执与编码、采集/可用时钟和缺测。两次早期实际失败分别为已移除旧字段的复制要求、探针误换单一owner及其异常保存；原件保留，修正后重新登记与验证，未将这些工程修复计作算法收益。

尚未完成：全21响应与因果自由演化模型准入、新融合决策实现、完整多情境/未见seed性能对照。因此本轮不新增成功率/稳放优势声明，也不把短识别记录与此前完整任务结果拼接。原固定v3、力TV、力限、接收冲击、稳定达到时间和末端运动指标继续保留。

## 可复核材料

- [当前完整轨迹桥](output/v25_bridge_fork_v3/bridge_complete.json)
- [实际五支终态与分配](output/v25_first_five_v2/execution_complete.json)
- [统一公开响应读数](review/v25_first_five_response_v1.json)
- [技术摘要与有效方向范围](review/v25_first_five_technical_summary_v1.json)
- [独立实际证据审查](review/v25_first5_execution_independent_v1.json)
- [首轮固定实验合同](review/V25公开接触投影响应识别首轮固定实验合同_v1.md)
- [因果识别与模型支持域](review/V25外部投影响应因果识别与模型支持域设计_v1.md)
- [上一轮完整策略对照](V24规范参照多任务融合九条对照与后续路线_20261106.md)

主要输入SHA256：readout `{sha(ROOT/'review/v25_first_five_response_v1.json')}`；independent `{sha(ROOT/'review/v25_first5_execution_independent_v1.json')}`；probe plan `{sha(ROOT/'review/v25_first_five_plan_v2.json')}`。所有失败、unsupported和unknown均保留。
'''
    assert all((ROOT/link).is_file() for link in re.findall(r'\]\(([^)]+)\)',body))
    (ROOT/name).write_text(body,encoding='utf-8')
    hist=ROOT/'report/progress_history';hist.mkdir(exist_ok=True);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for old in ('研究进度.json','当前进度与阶段汇报.md','review/v25_identification_progress_v1.json'):shutil.copyfile(ROOT/old,hist/(stamp+'_'+Path(old).name))
    state=read('研究进度.json')
    state.update(updated_utc=now,goal_status='active',superiority_proven=False,latest_report=name,latest_stage_report=name,
        latest_foundation_report=name,current_work='V25 source198 observer/actual receipts/fork foundation accepted; exact full16 bridges and fixed five intervention records with105 independent checks complete. Effective assigned input rank2; no fitted/admitted response model or new policy-performance claim.',
        current_physics_processes=[],current_physics_jobs=[],active_physics_jobs=[],
        previous_goal_turn_classification='progress: implemented6 independent sources; passed actual198 gates and full16 passive/fork bridges; executed all5 controlled arms, preserved failures/unsupported, fixed acquisition-clock readout and105 execution audit, defined response-driven whole fusion route.')
    state['v25_current'].update(status='foundation_first_five_complete_readout_and_independent105_passed',source_count_verified=True,
        physics_active=False,physics_sessions=[],physical_runs=5,probe_five_terminal=True,probe_five_passed=True,
        readout='review/v25_first_five_response_v1.json',execution_audit='review/v25_first5_execution_independent_v1.json',
        execution_checks=105,model_admitted=False,model_fits=0,superiority_proven=False,report=name,
        supported_independently_tested_input_columns=columns,physical_seed_clusters=1)
    save('研究进度.json',state)
    progress=read('review/v25_identification_progress_v1.json');progress.update(updated_utc=now,
        phase=state['v25_current']['status'],physics_active=False,physics_sessions=[],bridge_terminal=True,bridge_passed=True,
        probe_five_terminal=True,probe_five_passed=True,readout=state['v25_current']['readout'],independent_audit=state['v25_current']['execution_audit'],
        report=name,model_admitted=False)
    save('review/v25_identification_progress_v1.json',progress)
    (ROOT/'当前进度与阶段汇报.md').write_text(f'''# V25基础完成：已取得首组真实响应样本，模型与新策略待验证

更新UTC：{now}。当前无相关物理仿真后台运行，主目标active。

六文件198基础版本完成同龄公开观测、实际receipt历史与受控识别；原V24完整16秒被动桥和复制剩余桥各92项通过，3公开流3201/1701/1700记录一致。首五支全部终态，独立实际105项通过，父192/原42与数值身份保持。

有效扰动为升降+0.450mm/−7.30µm、WRJ1−0.000300rad；WRJ1+权限0明确未暴露。有效保持窗只支持两输入列；全assigned rank16含未支持支回原策略的混合，不能据此宣称16方向识别。设备基面/有效行变化保留，未知不补0。

当前未拟合/准入响应模型，尚无新策略性能对照；下一步按全21/混合留出/不同公共模式和seed验证，再实施统一力—运动—交接决策。原多指标评价不变。

[完整阶段报告]({name})；[真实读数](review/v25_first_five_response_v1.json)；[整体实施路线](review/V26响应驱动多任务融合整体实施路线_v1.md)；[独立105项审查](review/v25_first5_execution_independent_v1.json)。
''',encoding='utf-8')
    print(json.dumps(dict(report=name,source_count=198,actual_probe_assignments=5,independent_checks=105,model_admitted=False,goal_status='active'),ensure_ascii=False))

if __name__=='__main__':main()

"""Persist the already-terminal V21 readout; never run a controller or solver."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, data):
    (ROOT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

aux_name = 'review/v21_complete16_auxiliary_v1.json'
audit_name = 'review/v21_complete16_execution_v1.json'
tv_name = 'review/v21_complete16_release_variation_v1.json'
aux, audit, tv = map(read, (aux_name, audit_name, tv_name))
assert aux['complete'] and audit['complete'] and audit['passed'] and tv['complete']
assert aux['combined_count'] == 16 and tv['assigned_count'] == 16
assert not aux['issues'] and not audit['issues'] and not tv['issues']
for name, expected in aux['source_sha256'].items():
    path = ROOT.parent/'五指策略接入_2026-11-03'/name[11:] if name.startswith('foundation/') else ROOT/name
    assert sha(path) == expected, name
assert len(aux['source_sha256']) == 170

roles = ['same_prior_legacy_ratio', 'full_V20', 'full_V21']
fields = ['original_task_success', 'original_physical_clearance_success', 'original_full_guard_pass', 'ever_floor', 'first_low_motion_found', 'final_low_motion_suffix_found']
labels = ['原任务完成', '原物理清离', '全程力限通过', '释放窗曾落地↓', '首次低运动驻留', '末段低运动驻留']
candidate = [p['candidate_metrics'] for p in aux['pairs'] if p['reference_role'] == roles[0]]
by_role = {role: [p['reference_metrics'] for p in aux['pairs'] if p['reference_role'] == role] for role in roles[:-1]}
by_role['full_V21'] = candidate
assert all(len(rows) == 16 for rows in by_role.values())
counts = {role:{key:sum(r[key] is True for r in rows) for key in fields} for role,rows in by_role.items()}
qualified = {role:sum(r['final_low_motion_suffix_found'] is True and r['original_task_qualified_release_exposure'] is True for r in rows) for role,rows in by_role.items()}
table = '\n'.join('|'+label+'|'+'|'.join(str(counts[role][key])+'/16' for role in roles)+'|' for label,key in zip(labels,fields))
now = datetime.now(timezone.utc).isoformat()
report_name = 'V21完整16条件综合对照与数值重构依据_20261105.md'
assert not (ROOT/report_name).exists(), 'Preserve a previous report rather than overwrite it.'
report = f'''# V21完整16条件：低冲击与清离后运动改善，求解完整度仍待重构

完整16条件结果与执行独审已结束。相对力比PI迁移基线，V21保留任务、限力、低冲击收益，并使释放窗落地从4/16降至0/16；相对V20，末段低运动驻留由0/16增至5/16，但释放力变化总量中位增加9.53%。这支持继续以多任务融合推进整体改进，尚未证明全面或泛化优越。

本报告依据已终态的完整16读数，记录时间UTC：{now}。原3条FullV21保留，新补13条；加上三臂开发的9条，本版本共22条物理运行均终态，无运行错误。执行独审201/201通过；此结果仅证明执行证据一致，不替代任务表现。

## 方案与比较范围

V21在独立命名空间实现接触法向纠偏、同一纠偏锚点上的运动预算、退出进度、全手退出路径记忆与分层认证。低层失败时只提交确已认证的前层点；全部任务完成另行报告。原平台、原基线、V17/V20、评价定义与旧输出保留，170项执行来源未变。

完整矩阵为4场景×hold/transport×2物理种子（619741、619742）。这是开发条件复用，不能视为16个独立物理抽样或未见条件确认。力比PI为论文启发的迁移基线，不是论文全部系统的复现。所有任务失败、缺失值与回退保留，不筛成功样本；没有新增加权总分或调整成功阈值。

## 完整分母结果

|指标|力比PI|V20|V21|
|---|---:|---:|---:|
{table}

末段低运动与原任务合格释放相交：PI {qualified[roles[0]]}/16、V20 {qualified[roles[1]]}/16、V21 {qualified[roles[2]]}/16。低运动驻留沿用固定v3：接收面有载荷、手已低力清离、无其它支撑，并满足既定线/角速度持续0.2秒。它不是永久稳定放置或完整几何容纳证书。V21仍有11/16在末段未满足此要求。

## 同条件配对变化

下列百分比都是逐对变化的中位数，不是组中位数之比。

|指标|V21相对力比PI|V21相对V20|
|---|---:|---:|
|全程最大指接触力|−26.29%|0.00%|
|首次触台20毫秒接收峰|−64.64%|0.00%|
|首次触台20毫秒接收冲量|−38.25%|+0.20%|
|末线速度|−56.66%|−74.74%|
|原200Hz清离延迟|+1.4025秒|+0.0300秒|
|释放阶段五指力变化总量TV|−43.25%|+9.53%|

TV固定取12–16秒801个原始200Hz样本，无滤波，含正常卸力与动作变化，不能全部称为不必要的振荡。V21相对PI为15降/1增，相对V20为6降/10增；五指目标接触与五指全部外接触两通道在本批读数相同，不是一般恒等，也不含掌部/辅助接触。未用PI的优势遮盖相对V20的代价。

种子差异保留：619741末段驻留1/8，619742为4/8；相对V20的TV配对变化中位分别−3.22%、+17.63%。全局中位不能代表两种子均有相同收益。

## 数值完成度及失败

完整16的12816个5毫秒释放记录中：直接SEARCH/CAPTURE 5519；有压1377；已知无指接触5919；几何未知1。有压1377次为686全层、686部分认证前缀、5零前缀回退；已知无接触5919全层接受，未知1合法回退。实际执行被替换与过期计划计数均0。

有压失败层：L1纠偏5，L2退出111，L3运动285，L4姿态290，原始状态均为DEPENDENT_BLOCKER_NUMERICAL_FAILURE。可用前缀不等于全部融合任务已优化。这个数值证据使下一步应先分离求解器退化与控制目标冲突，再决定架构；不能仅从物理失败推断目标错误。

原任务失败仍为collision/hold/619741、collision/transport/619742，发生在释放前，释放策略不能追溯改写其成功。8/16全程力限仍未通过；碰撞场景早期指峰需与退出任务分开治理。

## 下一轮整体重构依据

先将真实保存的每层数学问题交给独立数值实现，保持原目标、硬约束、前层结果与容差，回原坐标核验可行性、目标和KKT，不把库success标志当证据。现有环境已有SciPy1.18.1，不需安装。诊断仅离线重解数学问题，不作为物理收益，不更换冻结V21后端。

同时核查公共运动任务的可观测范围，以及最终接触消失前是否真正完成运动收敛。下一候选应统一任务表示、稳定交接与求解执行，不以放宽力限、修改指标或局部参数扫描追求表面胜利。新版本物理测试需同条件对照并保留所有代价。

## 证据

- [完整16辅助读数](review/v21_complete16_auxiliary_v1.json)
- [执行独审](review/V21完整16条件执行独审_v1.md)
- [释放TV逐对读数](review/v21_complete16_release_variation_v1.json)
- [释放TV全16逐对表](review/v21_complete16_release_variation_tables_v1.md)
- [V21物理前实施合同](review/V21稳定交接实施合同_v1.md)
- [九条三臂开发结果](V21整体稳定交接与九条三臂开发结果_20261105.md)

输入SHA256：

|文件|SHA256|
|---|---|
|辅助读数|{sha(ROOT/aux_name)}|
|执行独审|{sha(ROOT/audit_name)}|
|释放TV|{sha(ROOT/tv_name)}|
'''
(ROOT/report_name).write_text(report, encoding='utf-8')
history = ROOT/'report/progress_history'; history.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
for name in ('研究进度.json', '当前进度与阶段汇报.md', 'review/v21_complete16_progress_v1.json'):
    p = ROOT/name
    if p.exists(): shutil.copyfile(p, history/(stamp+'_'+p.name))
progress = read('研究进度.json')
progress.update(updated_utc=now, goal_status='active', superiority_proven=False,
    implementation='V21 source170 frozen; all22 physical runs terminal; complete16 auxiliary, force TV and independent execution201 passed. Offline numerical and public-motion diagnosis now active.',
    latest_report=report_name, latest_stage_report=report_name, latest_algorithm_report=report_name,
    previous_goal_turn_classification='progress: completed full16 readouts and actual independent execution201, changing the next action to numerical-completeness and motion-observability diagnosis',
    current_work='Same-problem offline numerical oracle and public-motion observability diagnosis; no new physical run.',
    current_physics_processes=[], current_physics_jobs=[], active_physics_jobs=[],
    next_required_work=['Original-matrix offline independent numerical solve and KKT verification', 'Public motion observability evidence', 'Evidence-based unified next architecture and controlled physical comparisons'])
progress['v21_current'].update(status='complete16_terminal_aux_tv_execution201_complete_diagnosis_active',
    physics_active=False, physics_sessions=[], new_physical_runs=22, completed=22,
    candidate_full_count=16, candidate_task_count=14, candidate_clearance_count=16, candidate_guard_count=8,
    candidate_floor_count=0, candidate_final_residence_count=5, execution_audit=audit_name,
    execution_checks=201, auxiliary=aux_name, release_variation=tv_name,
    expanded13_completed=13, combined_completed=16, report=report_name, numerical_diagnostic_active=True)
write('研究进度.json', progress)
p16 = read('review/v21_complete16_progress_v1.json')
p16.update(updated_utc=now, new_completed=13, combined_completed=16, all_new_terminal=True,
    physics_active=False, physics_sessions=[], current_stage='complete16_all_readouts_and_execution201_terminal',
    all_assigned_readout_pending=False, performance_claim_allowed=True, superiority_proven=False,
    execution_audit=audit_name, auxiliary=aux_name, release_variation=tv_name)
write('review/v21_complete16_progress_v1.json', p16)
(ROOT/'当前进度与阶段汇报.md').write_text(f'''# V21完整16条件已结束，离线数值与运动诊断进行中

更新UTC：{now}。原平台与旧实验保留，source170不变；本版本22条物理运行均终态，当前没有物理仿真在后台运行。主目标保持active。

|指标|力比PI|V20|V21|
|---|---:|---:|---:|
{table}

V21相对PI：全程指峰配对中位−26.3%、接收首20ms峰−64.6%、释放TV−43.3%，清离延迟+1.4025秒。相对V20：末线速−74.7%、floor2→0、末段驻留0→5，但释放TV+9.53%。完整16执行201/201通过，不能据此宣称全面胜出。

1377次有压决策：686全层、686部分前缀、5回退。下一步保持V21结果冻结，离线对比同一数学问题并核查公共运动可观测性，再设计统一新架构。当前尚未开始下一版本物理试验。

[完整阶段报告]({report_name})；[完整辅助读数](review/V21补齐13合并16固定辅助读数_v1.md)；[执行独审](review/V21完整16条件执行独审_v1.md)；[释放TV](review/v21_complete16_release_variation_v1.md)。

原进度文件已按原字节备份至report/progress_history；历史报告与全部输出保持。
''', encoding='utf-8')
assert all(sha(ROOT.parent/'五指策略接入_2026-11-03'/k[11:] if k.startswith('foundation/') else ROOT/k) == v for k,v in aux['source_sha256'].items())
print(json.dumps({'report':report_name,'source170_unchanged':True,'goal_status':'active','physics_active':False,'counts':counts}, ensure_ascii=False))

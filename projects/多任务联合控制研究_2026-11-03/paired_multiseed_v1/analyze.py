"""Summarize all 80 frozen assignments without running any simulation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .statistics import (read, sha, dump, csv_dump, flatten, aggregate, SCENES, SCENE_NAMES,
                         ARMS, ARM_NAMES, STAGES, STAGE_NAMES, LABELS, BINARY, np)


def number(value, digits=3):
    return '未达成/无数据' if value is None else f'{value:.{digits}f}'


def count(metric):
    return f'{metric["successes"]}/{metric["trials"]}（{100*metric["mean"]:.1f}%）'


def change(pair):
    value = pair['relative_change_pct']
    return '不适用' if value is None else f'{value:+.1f}%'


def ci_text(pair):
    value = pair['ci95']
    return '无法估计' if value is None else f'[{value[0]:.3f}, {value[1]:.3f}]'


def report_text(plan, summary, groups, validation):
    overall = groups['overall']
    arms = overall['arms']
    lines = ['固定策略的多随机种子对照测试',
        '10 个新种子 × 4 个场景 × 2 种策略，共 80 次；每次物理时长 16 秒。',
        '基线和宏观版从 0 秒独立控制，沿用已有全控制版本的策略、参数和物理模型。',
        '本轮只运行和统计，不生成图片或视频，不覆盖原平台及已有实验结果。', '',
        f'仿真及配对验收耗时：{summary["wall_seconds"]/60:.2f} 分钟；并行进程：{plan["workers"]}。',
        f'完整验收通过：{validation["passed"]}；完整记录：{validation["valid_records"]}/80；同条件配对：{validation["paired_condition_checks_passed"]}/40。',
        '种子：' + ', '.join(str(s) for s in plan['seeds']), '',
        '总体结果（每种策略 40 次；统计推断按 10 个独立种子分组）',
        '指标\t基线\t宏观版\t宏观版相对变化']
    for key in BINARY:
        a, b = [arms[arm]['metrics'][key] for arm in ARMS]
        lines.append(f'{LABELS[key]}\t{count(a)}\t{count(b)}\t{100*(b["mean"]-a["mean"]):+.1f} 个百分点')
    main = ('full__total_force_reversal_rate_N_s', 'full__total_force_SD_N',
            'pickup_stable_time_s', 'pickup_deadline_capped_s', 'release_latency_s',
            'release_deadline_capped_s', 'steady_release_latency_s',
            'force_exceedance_duration_s', 'maximum_force_excess_N', 'force_excess_impulse_N_s')
    for key in main:
        a, b = [arms[arm]['metrics'][key] for arm in ARMS]
        p = overall['pairs'][key]
        # Arm means can have different attainment populations: paired deltas always use matched observations.
        means = [f'{number(x["mean"])} ± {number(x["sd"])}（n={x["measured_n"]}）' for x in (a,b)]
        lines.append(f'{LABELS[key]}\t{means[0]}\t{means[1]}\t配对 {change(p)}，n={p["paired_n"]}')
    lines += ['', '分场景结果：每种策略、每个场景均为 10 次。',
              '场景\t完整任务成功 基线/宏观\t限力通过 基线/宏观\t全程往复变化率 基线/宏观（N/s）\t全程标准差 基线/宏观（N）\t稳定抓起时间 基线/宏观（s）']
    for scene in SCENES:
        g = groups[scene]
        a,b = [g['arms'][arm]['metrics'] for arm in ARMS]
        def pair(k): return f'{number(a[k]["mean"])}/{number(b[k]["mean"])}'
        lines.append(f'{SCENE_NAMES[scene]}\t{a["terminal_success"]["successes"]}/10 : {b["terminal_success"]["successes"]}/10'
            f'\t{a["force_limit_pass"]["successes"]}/10 : {b["force_limit_pass"]["successes"]}/10'
            f'\t{pair("full__total_force_reversal_rate_N_s")}\t{pair("full__total_force_SD_N")}\t{pair("pickup_stable_time_s")}')
    lines += ['', '分阶段总指力变化：均使用固定时间窗，包含任务失败的有效物理记录。',
              '阶段\t往复变化率 基线/宏观（N/s）\t宏观版变化\t标准差 基线/宏观（N）\t宏观版变化']
    for stage in STAGES:
        p = overall['pairs'][f'{stage}__total_force_reversal_rate_N_s']
        q = overall['pairs'][f'{stage}__total_force_SD_N']
        lines.append(f'{STAGE_NAMES[stage]}\t{number(p["baseline_mean"])}/{number(p["macro_mean"])}\t{change(p)}'
            f'\t{number(q["baseline_mean"])}/{number(q["macro_mean"])}\t{change(q)}')
    lines += ['', '同等有效抓握配对子集：两种策略都稳定抓起，并保持到放下指令。',
              f'共 {overall["functional_pair_n"]}/40 对，覆盖 {overall["functional_seed_n"]}/10 个种子。',
              '场景\t配对数\t全程往复变化率 基线/宏观（N/s）\t宏观版变化\t差值的95%区间']
    for scene in (*SCENES, 'overall'):
        g = groups[scene]; p = g['functional_pairs']['full__total_force_reversal_rate_N_s']
        lines.append(f'{SCENE_NAMES[scene]}\t{g["functional_pair_n"]}\t{number(p["baseline_mean"])}/{number(p["macro_mean"])}\t{change(p)}\t{ci_text(p)}')
    lines += ['', '主要配对差值（宏观版减基线；负值表示该数值降低）',
              '指标\t差值\t95%种子分组重采样区间\t配对数']
    for key in ('terminal_success', 'force_limit_pass', 'full__total_force_reversal_rate_N_s',
                'full__total_force_SD_N', 'pickup_deadline_capped_s', 'release_deadline_capped_s',
                'force_exceedance_duration_s'):
        p = overall['pairs'][key]
        if key in BINARY:
            interval=p['ci95']
            interval_text='无法估计' if interval is None else f'[{100*interval[0]:.1f}, {100*interval[1]:.1f}] 个百分点'
            lines.append(f'{LABELS[key]}\t{100*p["macro_minus_baseline"]:.1f} 个百分点\t{interval_text}\t{p["paired_n"]}')
        else:
            lines.append(f'{LABELS[key]}\t{number(p["macro_minus_baseline"])}\t{ci_text(p)}\t{p["paired_n"]}')
    lines += ['', '失败原因（原因可重叠，同一试验不重复计入成功分母）']
    reasons = dict(pickup_not_stable_by_7_5s='抓握未在7.5秒前达到稳定判据',
        object_not_retained_until_release='运输中未能保持物体',
        transport_tracking_outside_tolerance='运输位置跟踪超出容差',
        no_valid_raised_grasp_at_release_command='放下指令时缺少有效抬升抓握',
        placed_and_clear_not_confirmed_by_15_5s='未在15.5秒前确认放好并离手',
        physical_record_invalid='物理记录无效', technical_failure='运行或校验异常',
        terminal_receiver_or_clearance_not_retained='曾完成放置，但结束时未保持留台离手')
    for scene in (*SCENES, 'overall'):
        for arm in ARMS:
            items=groups[scene]['arms'][arm]['failure_counts']
            text='；'.join(f'{reasons.get(k,k)} {v} 次' for k,v in items.items()) or '无任务失败'
            lines.append(f'{SCENE_NAMES[scene]} / {ARM_NAMES[arm]}：{text}')
    lines += ['', '指标与解释',
        '1. 完整任务成功：稳定抓起、运输保持、按要求跟踪、放好并离手均达成，且最后0.2秒持续留在接收台上、离手且未落地。阶段完成率保留旧口径，二者分别报告。',
        '2. 总指力指五指作用于目标物体的法向力之和。往复变化率以既有50Hz滤波信号计算：每秒反复增加与减少的力变化量；标准差反映整段力的离散程度。二者都包含有意的任务动作，不能直接解释为纯噪声。',
        '3. 时间窗固定为抓握0–7.5秒、保持及运输7.5–12秒、放下12–16秒、全程0–16秒。全程指标会受任务阶段和失物影响，因此同时给出分阶段结果及双侧有效抓握子集。子集属于条件性分析，不能替代全体成功率。',
        '4. 抓握稳定要求连续0.2秒满足接触、抬升、速度和位置判据。未达成的试验不按0秒计算；达成者均值须连同n解释。另给出截止时间封顶统计，保留失败代价。放置稳定另含物体速度约束，不等于仅离手或仅留台。',
        '5. 限力用未滤波的全部外部指部受力，在每个0.5毫秒物理步及端点检查单指上限和总力预算，容差0.02N；这与只画目标物体总指力的信号不同。超力时间即任一约束越界的累计时长。',
        '6. 10个种子在运行前抽取并排除历史使用/规划种子。沿用既有随机规则：质量相对波动的标准差为6%，限制在名义值的85%–115%；摩擦相对波动的标准差为8%，限制在80%–120%，并使用该种子的传感噪声。同一种子下两种策略及四场景共享随机对象条件。',
        '统计区间按种子成组重采样20000次，不把四场景当作40个独立种子。实际质量与摩擦取值见random_conditions.csv。',
        '7. 相对变化在配对有效记录上计算；若某指标仅一方达成，不把缺失时间补成0。分场景二项比例给出Wilson区间，详见统计表。区间为描述性估计，未开展多指标显著性筛选。',
        '8. 本轮验证现有模型及随机条件下的重复性，未增加物体类型、扩大场景族或调整策略。原始元数据沿用旧版development标签；本轮冻结计划明确记录10个新种子的重复测试范围。', '',
        '文件：per_trial_statistics.csv为逐次记录；scenario_statistics.csv为分场景统计；paired_comparisons.csv为配对差值；functional_force_comparisons.csv为有效抓握子集；statistics_summary.json保存完整结果与验收。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out', required=True);args=parser.parse_args()
    out=Path(args.out).resolve();plan=read(out/'plan.json');summary=read(out/'summary.json')
    expected={j['id']:j for j in plan['jobs']};records=summary['jobs']
    assert len(records)==len(expected)==80
    assert len({r['job']['id'] for r in records})==80
    assert all(r['job']==expected[r['job']['id']] for r in records)
    assert len(plan['seeds'])==len(set(plan['seeds']))==10
    assert not (set(plan['seeds']) & set(plan['historical_seeds']))
    rows=sorted((flatten(r) for r in records),key=lambda r:(plan['seeds'].index(r['seed']),SCENES.index(r['scene']),ARMS.index(r['arm'])))
    # Verify saved arrays/metadata against the hashes recorded by the unchanged evaluator.
    artifact_checks={}
    for record in records:
        job=record['job'];folder=out/'runs'/job['id'];r=record.get('result')
        artifact_checks[job['id']]=bool(r and all(sha(folder/name)==digest for name,digest in r['artifacts_sha256'].items()))
    conditions=[]
    for seed in plan['seeds']:
        meta=read(out/'runs'/f'baseline__nominal__{seed}'/'metadata.json')
        conditions.append(dict(seed=seed,mass_kg=meta['config']['mass'],friction=meta['config']['friction']))
    assert len({(c['mass_kg'],c['friction']) for c in conditions})==10
    groups,compact,pairs,functional=aggregate(rows)
    validation=dict(batch_audit_passed=summary['audit']['passed'],expected_assignments=80,retained_assignments=len(rows),
        paired_condition_checks_passed=sum(p['passed'] for p in summary['audit']['pairs'].values()),
        valid_records=sum(r['valid_record'] for r in rows),every_assigned_seed_new=True,
        unique_random_physical_conditions=10,artifact_hash_checks=artifact_checks,
        per_scene_arm_denominators=dict(Counter(f'{r["scene"]}/{r["arm"]}' for r in rows)))
    validation['passed']=bool(summary['audit']['passed'] and validation['valid_records']==80 and all(artifact_checks.values())
        and set(validation['per_scene_arm_denominators'].values())=={10})
    destination=out/'statistics';destination.mkdir(exist_ok=False)
    csv_dump(destination/'per_trial_statistics.csv',rows)
    csv_dump(destination/'scenario_statistics.csv',compact)
    csv_dump(destination/'paired_comparisons.csv',pairs)
    csv_dump(destination/'functional_force_comparisons.csv',functional)
    csv_dump(destination/'random_conditions.csv',conditions)
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(out/'plan.json'),
        batch_summary_sha256=sha(out/'summary.json'),statistics_spec=plan['statistics_spec'],
        simulation_wall_seconds=summary['wall_seconds'],groups=groups,validation=validation,random_conditions=conditions,
        metric_labels=LABELS,code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    dump(destination/'statistics_summary.json',value)
    (destination/'统计结果报告.txt').write_text(report_text(plan,summary,groups,validation),encoding='utf-8-sig')
    for name in ('analyze.py','statistics.py'):
        shutil.copy2(Path(__file__).parent/name,destination/name)
    overall=groups['overall']['arms']
    print({'statistics_valid':validation['passed'],'output':str(destination),
        'terminal_success':{a:overall[a]['metrics']['terminal_success']['successes'] for a in ARMS},
        'force_limit_pass':{a:overall[a]['metrics']['force_limit_pass']['successes'] for a in ARMS}})


if __name__=='__main__':main()

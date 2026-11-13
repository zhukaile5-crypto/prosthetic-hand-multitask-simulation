"""Write a visible stage snapshot from completed result files, not predictions.

The caller owns live-process verification. --running-session is supplied only
after polling that exact live unified-exec handle. This script starts no work.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--running-session',type=int)
    args=parser.parse_args();now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    plan=json.loads((ROOT/'review/v12_native_development_registration_v1.json').read_text(encoding='utf-8'))
    records=[];errors=[];groups=[];seen=set()
    for group in plan['groups']:
        directory=ROOT/group['output'];local=[]
        for path in sorted(directory.glob('runs/*/result.json')):
            r=json.loads(path.read_text(encoding='utf-8'))
            key=(r['method'],r['scenario'],r['seed'],r['motion_mode'])
            if key in seen:raise ValueError('Duplicate result cannot advance progress')
            seen.add(key);local.append(r);records.append(r)
        errors.extend(str(p.relative_to(ROOT)) for p in directory.glob('runs/*/runtime_error.json'))
        finish=directory/'execution_complete.json'
        terminal=json.loads(finish.read_text(encoding='utf-8')) if finish.exists() else None
        groups.append(dict(output=group['output'],completed=len(local),planned=group['records'],
                           execution_complete=None if terminal is None else terminal['execution_complete']))
    n=len(records);assert n<=48
    audit_path=ROOT/'review/v12_native_full48_readonly_acceptance_v1.json'
    if n==48 and args.running_session is None and audit_path.exists() and json.loads(audit_path.read_text(encoding='utf-8'))['passed']:
        from finalize_visible_native48_v1 import publish
        publish()
        return
    old=json.loads((ROOT/'report/v12_full48_v1/metric_readout.json').read_text(encoding='utf-8'))['records']
    labels={'ratio_effort':'摩擦比对照','fixed_effort':'固定力对照','fusion_effort':'融合策略'}
    lines=['# 当前进度与阶段汇报','',f'更新：{now}。','',
        '本文件保存最新阶段快照。每个阶段结束还会在聊天中单独汇报。','',
        '## 已确认的物理结果：V12原版48条','',
        '胶囊体、四环境、静态/运输两任务、两个物理种子。每法16个条件；全部完整记录，来源冻结。','',
        '|指标|摩擦比对照|固定力对照|融合策略|','|---|---:|---:|---:|']
    vals=[[r for r in old if r['method']==m] for m in labels]
    lines+=['|完整任务完成|'+'|'.join(f"{sum(x['full_motion_mission_success'] for x in a)}/16" for a in vals)+'|',
        '|稳态平均总指力 N|'+'|'.join(f"{np.mean([x['static_force_N'] for x in a]):.2f}" for a in vals)+'|',
        '|稳态原200Hz指力变化率 N/s|'+'|'.join(f"{np.mean([x['static_raw200Hz']['finger_TV_rate_N_s'] for x in a]):.2f}" for a in vals)+'|',
        '|全程力限通过|'+'|'.join(f"{sum(x['force_guard'] for x in a)}/16" for a in vals)+'|','',
        '旧版已见多指标开发收益；融合仍有12/16未通过全程力限，部分峰值和全持握标准差更差。只有两个物理种子，不能外推为普遍或真实硬件优势。','',
        '## 本轮：统一原生执行计算后的完整复验','',
        f'**已完成 {n}/48 条完整记录。** '+('同一冻结来源的余下记录正在运行。' if args.running_session else '当前没有已确认存活的物理运行句柄。'),
        '新旧48条分别统计，不混池；本轮力目标、任务、阈值和三种策略保持原合同。','',
        '|方法|当前任务通过/已完成|最终计划|','|---|---:|---:|']
    for method,label in labels.items():
        a=[r for r in records if r['method']==method]
        lines.append(f"|{label}|{sum(r['task_success'] for r in a)}/{len(a)}|16|")
    lines+=['','以上是进行中的完成记录计数，不是整批成功率结论。',
        f'已保存运行异常记录：{len(errors)}。','',
        '|核查步骤|状态|','|---|---|',
        '|新旧外层逻辑和实际控制接口|通过|',
        '|675次旧决策离线回归|通过；599次保留、76次数值误拒恢复，目标与约束不变|',
        '|关键边界反例|18项通过，保留真实残差与不支持状态|',
        '|首批3条接入/共同前缀|312项通过|',
        '|整批48条原始记录验收与综合比较|'+('等待整批终态' if n<48 else '记录已齐，等待最终只读验收')+'|','',
        '评价继续保留任务、力限、均力、绝对波动、跟踪和释放；另记录相对波动和计算耗时。共同可评价窗配对，NULL不补零，不设置新综合权重。','',
        '## 详细资料','',
        '- [原48条完整对比](D:/生物医学工程原理PBL/多任务联合控制研究_2026-11-03/V12多场景开发对比_20261105.md)',
        '- [本轮复验方案](D:/生物医学工程原理PBL/多任务联合控制研究_2026-11-03/V12原生执行一致性复验_20261105.md)',
        '- [相对波动说明](D:/生物医学工程原理PBL/多任务联合控制研究_2026-11-03/相对力波动与阶段指标说明_20261105.md)','']
    rendered='\n'.join(lines)
    old_start=rendered.index('## 已确认的物理结果')
    current_start=rendered.index('## 本轮：')
    details_start=rendered.index('## 详细资料')
    static_path=ROOT/'report/v12_native_static24_v1/summary.json'
    stage=''
    if static_path.is_file():
        a=json.loads(static_path.read_text(encoding='utf-8'))['records']
        stage='## 本轮已完成的静态24条\n\n每法8个条件，仅静态阶段，不能代替运输或整批结论。\n\n'
        stage+='|指标|摩擦比对照|固定力对照|融合策略|\n|---|---:|---:|---:|\n'
        stage+='|完整任务|'+'|'.join(f"{x['full_motion']}/8" for x in a)+'|\n'
        stage+='|稳态均力 N|'+'|'.join(f"{x['static_mean_N']:.2f}" for x in a)+'|\n'
        stage+='|稳态原200Hz指力变化率 N/s|'+'|'.join(f"{x['static_raw_TV_N_s']:.2f}" for x in a)+'|\n'
        stage+='|全程力限通过|'+'|'.join(f"{x['force_guard']}/8" for x in a)+'|\n\n'
    rendered=rendered[:old_start]+rendered[current_start:details_start]+stage+rendered[old_start:current_start]+rendered[details_start:]
    (ROOT/'当前进度与阶段汇报.md').write_text(rendered,encoding='utf-8')
    snapshot=dict(updated_utc=datetime.now(timezone.utc).isoformat(),completed=n,planned=48,
                  running_session=args.running_session,groups=groups,runtime_error_files=errors)
    p=ROOT/'研究进度.json';d=json.loads(p.read_text(encoding='utf-8'))
    d['updated_utc']=snapshot['updated_utc'];d['v12_native_current']['progress_snapshot']=snapshot
    d['v12_native_current']['new48_completed']=n
    d['v12_native_current']['status']='remaining45_running' if args.running_session else 'awaiting_final_audit' if n==48 else 'between_batches'
    d['current_physics_processes']=[] if args.running_session is None else [dict(session_id=args.running_session,kind='V12Native remaining45 registered development',planned=45)]
    d['current_work']=f'V12Native {n}/48 complete; '+('registered remaining45 live in session'+str(args.running_session) if args.running_session else 'no confirmed live physics handle')+'; old48 and frozen80 sources preserved.'
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (ROOT/'阶段进度记录.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(snapshot,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(snapshot,ensure_ascii=False))


if __name__=='__main__':main()

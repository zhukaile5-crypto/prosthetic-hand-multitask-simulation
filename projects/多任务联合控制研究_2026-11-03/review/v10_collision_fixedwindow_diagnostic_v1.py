"""Descriptive fixed-window forces even when original V3 quality is NULL.

Original eligibility/results are preserved and never backfilled.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation import force_band, force_variation


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows=[];protocol=read(ROOT/'protocol.json')
    for motion in ('hold','transport'):
        batch=ROOT/f'output/development_v10_{motion}_v1';finish=read(batch/'execution_complete.json')
        if not finish['execution_complete']:raise ValueError('No partial-cohort diagnostic')
        manifest=read(batch/'manifest.json')
        for name in ('evaluation.py','protocol.json'):
            if sha(ROOT/name)!=manifest['research_source_sha256'][name]:raise ValueError('Frozen evaluation identity mismatch')
        for method in ('ratio_preview','joint'):
            directory=batch/'runs'/f'collision__{method}__610212';metadata=read(directory/'metadata.json');result=read(directory/'result.json')['research_metrics']
            with np.load(directory/'trace.npz',allow_pickle=False) as z:
                trace={k:z[k].copy() for k in ('time','normal','skin_normal','substep_time','substep_duration','substep_normal','substep_skin_normal')}
            band,information=force_band(trace,metadata,protocol)
            if not information['valid']:raise ValueError('Original frozen force-band coverage unavailable')
            t=trace['time'];mask=(t>=9.3-1e-9)&(t<=11.7+1e-9)
            if mask.sum()!=481:raise ValueError('Full fixed481-point event window required')
            raw=force_variation(t[mask],trace['normal'][mask]);filtered=force_variation(t[mask],band[mask])
            allskin=force_variation(t[mask],trace['skin_normal'][mask])
            per={}
            for label,signal in (('raw_200Hz',trace['normal'][mask]),('band50Hz',band[mask])):
                tv=abs(np.diff(signal,axis=0)).sum(axis=0);net=abs(signal[-1]-signal[0]);reverse=np.maximum(0.,(tv-net)/2)
                per[label]=dict(TV_N_by_finger=tv.tolist(),net_drift_N_by_finger=net.tolist(),reversal_N_by_finger=reverse.tolist())
            rows.append(dict(motion=motion,method=method,window_s=[9.3,11.7],samples=481,diagnostic_only=True,
                original_v3_quality_eligible=result['dynamic_quality']['eligible'],original_v3_bandlimited=result['dynamic_quality']['bandlimited'],
                original_v3_raw=result['dynamic_quality']['raw_200Hz'],physical_support_fraction=result['dynamic_quality']['physical_support_fraction'],
                unqualified_fixed_window_target_force_raw=raw,unqualified_fixed_window_target_force_band50=filtered,
                unqualified_fixed_window_all_external_skin_raw=allskin,per_finger_target_force=per,
                actual_target_force_mean_N=float(trace['normal'][mask].sum(axis=1).mean()),
                whole_guard=result['safety']['guard_pass'],holding_excess_Ns=result['safety']['stages']['policy_holding_transport']['local_excess_integral_Ns'],
                force_band_definition=information,trace_sha256=sha(directory/'trace.npz')))
    result=dict(schema='v10_collision_fixedwindow_descriptive_force_v1',rows=rows,new_physics=False,
        source_sha256=sha(__file__),evaluation_sha256=sha(ROOT/'evaluation.py'),protocol_sha256=sha(ROOT/'protocol.json'),
        scope='Full fixed9.3..11.7s descriptive event window; not an eligible stable quality cohort. Original NULLs/3-of4 common-window means remain unchanged. Target normal truth used only offline; all-external skin raw separately labeled.')
    out=ROOT/'review/v10_collision_fixedwindow_diagnostic_v1.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# V10碰撞：完整固定事件窗的额外波动诊断','',
        '原V3品质仍为NULL；以下使用固定9.3–11.7秒的完整481点，不称合格稳定段，不回填原3/4共同品质均值。TV为逐指路径变化率；净漂移与往复另存JSON。','',
        '| 运动 | 方法 | 原品质eligible | 支持样本比例 | target raw TV N/s | target 50Hz TV N/s | 全外接触skin raw TV N/s | 持握超限 N·s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['motion']} | {row['method']} | {int(row['original_v3_quality_eligible'])} | {row['physical_support_fraction']:.6%} | {row['unqualified_fixed_window_target_force_raw']['finger_TV_rate_N_s']:.6f} | {row['unqualified_fixed_window_target_force_band50']['finger_TV_rate_N_s']:.6f} | {row['unqualified_fixed_window_all_external_skin_raw']['finger_TV_rate_N_s']:.6f} | {row['holding_excess_Ns']:.6f} |")
    lines+=['','target normal是离线被抓物逐指接触真值；skin是全部外接触压力，包含障碍。二者不混为同一在线信号。raw峰值和安全积分仍使用原完整未滤波口径；带限诊断调用冻结V3 force_band/force_variation，保持子步end-clock、50Hz FIR和阶段边界反射。','',
        '这些额外描述回答失去完整支持时实际波动如何变化，不改变失败、覆盖或安全结论，也不能用波动单项抵消任务/材料代价。']
    md=ROOT/'review/v10_collision_fixedwindow_diagnostic_v1.md';md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),rows=[dict(motion=r['motion'],method=r['method'],support=r['physical_support_fraction'],raw=r['unqualified_fixed_window_target_force_raw']['finger_TV_rate_N_s'],band=r['unqualified_fixed_window_target_force_band50']['finger_TV_rate_N_s']) for r in rows]),ensure_ascii=False))


if __name__=='__main__':main()

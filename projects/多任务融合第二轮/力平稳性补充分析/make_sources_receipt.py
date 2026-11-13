"""Project safe, relevant recorded evidence for the native inline Sources receipt."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent


def main():
    effects=json.loads((HERE/'output/paired_effects.json').read_text(encoding='utf-8'))
    meta=json.loads((HERE/'output/metadata.json').read_text(encoding='utf-8'))
    rows=[]
    for r in effects:
        if r['holm_family'] is not None:
            rows.append(dict(suite=r['suite'],domain=r['domain'],window=r['condition'],candidate=r['focal'],
                metric=r['metric'],paired_trials=r['n_pairs'],candidate_mean=r['focal_mean'],baseline_mean=r['comparator_mean'],
                difference=r['estimate'],ci_low=r['ci95'][0],ci_high=r['ci95'][1],holm_p=r['p_value_holm']))
    assert len(rows)==24
    source=dict(label='五指仿真原始轨迹与平稳性补充分析',
        files=[{'label':'paired_effects.json'},{'label':'分析计划.json'},{'label':'analyze_smoothness.py'},{'label':'独立核查.json'}],
        metricDefinitions=[
            {'label':'合力标准差','definition':'合力标准差是每次试验窗口内，五指原始法向力幅值之和的时间标准差，单位N；再在配对试验上取均值。'},
            {'label':'逐指总变差率','definition':'逐指总变差率是五指各自相邻力变化绝对值的总和除以实际窗口跨度，单位N/s。'},
            {'label':'快速变化代理','definition':'快速变化代理是合力减去51点中心移动均值后的有效内区残差RMS，单位N。'}],
        filters=['对照：同场景、同种子摩擦比基线',
                 '共同完成窗口：六个无机械外扰场景，双方均完成任务，8至17秒',
                 '预扰动窗口：15个非边界场景，双方已建立持握且在8.2至8.9秒未达到保守失抓边界'],
        caveats=['指标是在看到成功率后提出的，全部结果属于事后探索。',
                 '共同完成窗口筛选了成功样本，不能代表所有试验。',
                 '原始完整逐指力只有200Hz采样，不能据此推断所有0.5ms瞬态峰值。',
                 '六个无机械外扰场景包括触觉噪声及丢包，并非完全无干扰。',
                 '置信区间为未校正的95%种子簇区间；Holm检验单独校正24项探索性比较。'],
        executedAt=meta['created_utc'],
        evidenceFlow=[{'kind':'calculation','title':'统计单位','detail':'重采样整组随机种子，保留组内所有同种子场景；10000次配对种子簇bootstrap。'},
            {'kind':'validation','title':'保存的计算核查','detail':f"计算读取了{meta['trials']}条原始轨迹，逐文件匹配先前覆盖审计哈希；原冻结控制代码和参数保持一致。"},
            {'kind':'validation','title':'独立复算','detail':'另用独立中心矩与累计和算法复算120条预定原轨迹和452个原始配对窗口，核对10829项配对资格、3456行效应均值及24项检验，全部通过。'}])
    payload=dict(schemaVersion=1,items=[dict(id='force-variability',title='当前数据是否支持低力波动优势？',queries=[dict(
        id='paired-smoothness',source=source,reportingPeriod='仿真持握8–17秒与预扰动8.2–8.9秒',
        columns=[{'field':field,'label':label} for field,label in [
            ('suite','测试集合'),('domain','场景域'),('window','条件窗口'),('candidate','候选方法'),
            ('metric','指标及单位'),('paired_trials','配对试验数'),('candidate_mean','候选均值'),
            ('baseline_mean','基线均值'),('difference','候选减基线'),('ci_low','95%下界'),('ci_high','95%上界'),('holm_p','Holm p')]],
        rows=rows,preview=dict(kind='aggregate',note='24项固定探索性比较；每行是配对试验指标的聚合，时间点不是独立样本。',totalRows=24),
        methods=[{'language':'calculation','code':'正式共同完成队列：逐指变差率 (26.29181332962461 / 31.864719595970215 − 1) × 100 = −17.489268184397844%\n扰动共同完成队列：逐指变差率 (30.00306424708827 / 32.39022420367781 − 1) × 100 = −7.37000133613922%\n旧融合正式共同完成队列：合力标准差 (0.7588834862083154 / 0.937199277690598 − 1) × 100 = −19.02645421597847%'}])])])
    (HERE/'inline_sources.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()

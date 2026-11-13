"""Standalone scientific plot of all21 seen development profiles, not validation."""
from pathlib import Path
import os,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'review/.mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=ROOT/'report/release_development_tradeoffs_v1'
    if target.exists():raise FileExistsError('Retain the old figure; choose a new revision')
    audit_path=ROOT/'review/release_clearance_resolution_independent_v2.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'));assert audit['passed']
    records=[];sources={str(audit_path):sha(audit_path)}
    for record in audit['records']:
        trace=Path(record['trace_path']);assert sha(trace)==record['input_sha256'][record['trace_path']]
        sources.update(record['input_sha256'])
        result_path=trace.parent.parent/'result.json'
        result=json.loads(result_path.read_text(encoding='utf-8'));m=result['research_metrics']
        sources[str(result_path)]=sha(result_path)
        records.append(dict(stage=record['stage'],variant=record['variant'],
            peak_N=max(m['safety']['finger_peak_N']),excess_Ns=m['safety']['local_excess_integral_Ns'],
            legacy_confirmation_s=record['sampled_200Hz']['first_dwell_latency_s'],
            raw_end_confirmation_s=record['raw_end_labels']['first_dwell_latency_s'],
            guard_pass=m['safety']['guard_pass'],mission=m['full_motion_mission_success']))
    assert len(records)==21 and not any(x['guard_pass'] for x in records)
    baselines=[r for r in records if r['variant']=='zero'];base=baselines[0]
    assert all(r['excess_Ns']==base['excess_Ns'] for r in baselines)
    assert all(abs(r['raw_end_confirmation_s']-base['raw_end_confirmation_s'])<1e-10 for r in baselines)
    for r in records:
        r['raw_confirmation_change_ms']=1000*(r['raw_end_confirmation_s']-base['raw_end_confirmation_s'])
        r['legacy_confirmation_change_ms']=1000*(r['legacy_confirmation_s']-base['legacy_confirmation_s'])
        r['excess_reduction_percent']=100*(1-r['excess_Ns']/base['excess_Ns'])
    font_file=Path('C:/Windows/Fonts/msyh.ttc');assert font_file.exists()
    fontManager.addfont(str(font_file));font=FontProperties(fname=str(font_file))
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    colors={'phi_020ms':'#276FBF','phi_150ms':'#D9822B','phi_350ms':'#8F5AA8'}
    names={'phi_020ms':'起点0.02秒','phi_150ms':'起点0.15秒','phi_350ms':'起点0.35秒'}
    fig,(left,right)=plt.subplots(1,2,figsize=(12.4,5.4),gridspec_kw={'width_ratios':[1.2,1]})
    fig.suptitle('释放协调的多目标权衡（开发数据）',fontsize=16,fontweight='bold',y=.98)
    for stage,color in colors.items():
        for role,marker in [('train','^'),('holdout','o')]:
            rows=[r for r in records if r['stage']==stage and r['variant']!='zero'
                  and ((r['variant'] in ('phi_slow','lift_slow'))==(role=='train'))]
            left.scatter([r['raw_confirmation_change_ms'] for r in rows],
                         [r['excess_reduction_percent'] for r in rows],s=55,marker=marker,
                         edgecolor=color,facecolor=color if role=='train' else 'white',linewidth=1.3,
                         label=names[stage] if role=='train' else None,zorder=3)
    left.scatter([0],[0],marker='+',s=100,color='#333333',linewidth=2,zorder=4,label='原生参考（重复点合并）')
    selected=next(r for r in records if r['stage']=='phi_020ms' and r['variant']=='lift_slow')
    left.scatter([selected['raw_confirmation_change_ms']],[selected['excess_reduction_percent']],
                 marker='*',s=240,color=colors['phi_020ms'],edgecolor='#102A43',linewidth=1,zorder=6)
    left.annotate('待前瞻验证的训练轴向候选\n超限积分−40.2%，确认时间+11 ms',
                  xy=(11,selected['excess_reduction_percent']),xytext=(-12,54),fontsize=9,
                  arrowprops={'arrowstyle':'->','color':'#445566','lw':.9})
    left.axhline(0,color='#A0A7AE',lw=.8);left.axvline(0,color='#A0A7AE',lw=.8)
    left.set(xlabel='完整积分记录：低力确认时间相对原生的变化（ms）',
             ylabel='全程力超限积分降低（%）',xlim=(-16,29),ylim=(-53,76))
    left.set_title('收益与代价同时保留',fontsize=12,pad=12)
    left.grid(alpha=.17);left.legend(loc='lower right',fontsize=8,framealpha=.9)
    labels=[('phi_020ms',1,'0.02秒起点'),('phi_350ms',0,'0.35秒起点')]
    for stage,y,label in labels:
        r=next(r for r in records if r['stage']==stage and r['variant']=='lift_slow')
        old,new=r['legacy_confirmation_change_ms'],r['raw_confirmation_change_ms']
        right.plot([old,new],[y,y],color=colors[stage],lw=1.7,zorder=2)
        right.scatter([old],[y],marker='o',s=65,facecolor='white',edgecolor=colors[stage],zorder=3)
        right.scatter([new],[y],marker='D',s=52,color=colors[stage],zorder=3)
        right.annotate(f'{old:+.0f} ms',(old,y),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10)
        right.annotate(f'{new:+.0f} ms',(new,y),xytext=(0,-22),textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
    right.axvline(0,color='#A0A7AE',lw=.8)
    right.set(yticks=[0,1],yticklabels=['0.35秒起点\n下降轴延迟','0.02秒起点\n下降轴延迟'],
              ylim=(-.55,1.65),xlim=(-129,89),xlabel='低力确认时间相对原生的变化（ms）')
    right.set_title('评价时点不同，会改变时间收益判断',fontsize=12,pad=12)
    right.scatter([],[],marker='o',s=50,facecolor='white',edgecolor='#555',label='200Hz边界查询（原V3）')
    right.scatter([],[],marker='D',s=45,color='#555',label='2000Hz积分记录（终点标签）')
    right.legend(loc='upper left',fontsize=8,framealpha=.9)
    right.grid(axis='x',alpha=.17)
    fig.text(.075,.095,'▲ 训练单轴分支    ○ 混合留出分支（仅诊断）    ★ 训练候选；并非新种子验证结论',fontsize=9)
    fig.text(.075,.055,'同一胶囊体与物理种子的21条开发分支，不是21次独立重复；全程力限均未通过。',fontsize=9,color='#555555')
    fig.text(.075,.018,'低力判据为总手–目标接触力≤0.2 N并维持0.2 s；仅指记录内判据，不等同几何分离或连续时间证明。',fontsize=8,color='#555555')
    fig.subplots_adjust(left=.075,right=.985,bottom=.225,top=.85,wspace=.32)
    target.mkdir(parents=True)
    for ext in ('png','svg'):fig.savefig(target/f'多目标权衡.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    payload=dict(schema='release_development_tradeoffs_v1',data_role='seen development; no independent confirmation',
        sources=sources,plotter_sha256=sha(__file__),records=records,baseline=base,
        input_files_unchanged=all(sha(p)==v for p,v in sources.items()))
    assert payload['input_files_unchanged']
    with (target/'data_and_provenance.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'output':str(target),'records':21,'physics_steps':0},ensure_ascii=False))
if __name__=='__main__':main()

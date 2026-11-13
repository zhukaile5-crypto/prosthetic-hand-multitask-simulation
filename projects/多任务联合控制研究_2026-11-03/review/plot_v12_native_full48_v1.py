"""Reproducible standalone figure for the existing local Markdown report."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'review/.mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    directory=ROOT/'report/v12_native_full48_v1'
    summary_path=directory/'summary.json'
    audit_path=ROOT/'review/v12_native_full48_readonly_acceptance_v1.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))['records']
    audit=json.loads(audit_path.read_text(encoding='utf-8'));assert audit['passed']
    records=audit['aggregate_full_assigned']
    methods=['ratio_effort','fixed_effort','fusion_effort']
    assert [r['method'] for r in summary]==methods
    labels=['摩擦比对照','固定力对照','多任务融合']
    colors=['#496F95','#8B969D','#B68A2F']
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#59646C','text.color':'#263239','axes.labelcolor':'#263239',
        'xtick.color':'#34444E','ytick.color':'#34444E','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12.6,8.4))
    fig.subplots_adjust(left=.078,right=.97,bottom=.19,top=.85,hspace=.44,wspace=.24)
    fig.suptitle('多任务融合：同条件开发结果',x=.078,y=.97,ha='left',fontsize=20,fontweight='bold')
    fig.text(.078,.918,'胶囊体 · 4环境 × 2任务 × 2物理种子 · 每法16个条件，48条完整记录',fontsize=12)
    panels=[('稳态平均总指力（N）','static_force_N','%.2f'),
            ('稳态指力变化率（N/s，原200 Hz）','static_raw200TV_N_s','%.3f'),
            ('全程超限积分合计（N·s，16条）','local_Ns_sum','%.3f')]
    x=np.arange(3)
    plotted={}
    for ax,(title,key,fmt) in zip(axes.flat,panels):
        values=[r[key] for r in summary]
        ax.bar(x,values,width=.58,color=colors,edgecolor='#394A54',linewidth=.6,zorder=3)
        ax.set_title(title,loc='left',pad=12,fontsize=12,fontweight='bold')
        ax.set_xticks(x,labels);ax.set_ylim(0,max(values)*1.23)
        ax.grid(axis='y',color='#E0E5E8',linewidth=.7,zorder=0)
        ax.tick_params(axis='both',length=3)
        for i,value in enumerate(values):ax.text(i,value+max(values)*.035,fmt%value,ha='center',fontsize=12)
        plotted[key]=values
    ax=axes[1,1]
    median=[records[m]['control_ms_decision_grid']['median_ms'] for m in methods]
    p95=[records[m]['control_ms_decision_grid']['p95_ms'] for m in methods]
    maximum=[records[m]['control_ms_decision_grid']['max_ms'] for m in methods]
    ax.bar(x,median,width=.58,color=colors,edgecolor='#394A54',linewidth=.6,zorder=3)
    ax.scatter(x,p95,marker='D',color='#27343D',s=31,zorder=4)
    ax.axhline(20,color='#45545E',linestyle='--',linewidth=1)
    ax.text(2.5,20.4,'20 ms决策周期',ha='right',fontsize=9,color='#45545E')
    ax.set_xticks(x,labels);ax.set_ylim(0,24)
    ax.set_title('决策计算耗时（ms）',loc='left',pad=12,fontsize=12,fontweight='bold')
    ax.grid(axis='y',color='#E0E5E8',linewidth=.7,zorder=0)
    for i,value in enumerate(median):
        ax.text(i,value*.52,f'{value:.2f}',ha='center',va='center',fontsize=12,
            fontweight='bold',color='white' if i==0 else '#202020')
    ax.legend(handles=[Patch(facecolor='#8B969D',label='中位数'),
        Line2D([],[],marker='D',linestyle='none',color='#27343D',label='95分位')],
        loc='upper left',frameon=False,ncol=2,fontsize=9)
    fig.text(.078,.115,'完整任务：14/16 · 16/16 · 16/16     全程力限：0/16 · 0/16 · 4/16（均按图中方法顺序）',fontsize=11)
    fig.text(.078,.077,'稳态窗为8.2–8.9秒；两个物理种子及共享前缀，不能当作16次独立物理重复。',fontsize=10,color='#4B5B64')
    fig.text(.078,.042,'耗时来自3进程并行的控制分支；最大值为25.9 / 21.3 / 32.7 ms，未作实时保证。',fontsize=10,color='#4B5B64')
    for suffix in ('png','svg'):
        path=directory/f'综合收益与计算代价.{suffix}'
        fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    provenance=dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path,audit_path)},
        methods=methods,labels=labels,plotted_values=plotted,control_median_ms=median,control_p95_ms=p95,control_max_ms=maximum,
        source='unchanged V3 fixed-window/equal16 completed conditions and independent raw/source audit',
        inference='descriptive two physical clusters; no invented intervals or utility weights',
        selected_surface='existing local Markdown report; standalone scientific figure',
        visual_inspection_pending=True)
    (directory/'figure_data_and_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Figure saved with exact reviewed values and provenance')


if __name__=='__main__':main()

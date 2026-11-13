"""Scientific development comparison: display benefits and costs together."""
from pathlib import Path
import json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'report/.mplconfig_v5')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder=ROOT/'report/development_v5_pilot';out=folder/'品质与代价.png'
    if out.exists():raise FileExistsError(out)
    data=json.loads((folder/'readout.json').read_text(encoding='utf-8'))
    rows={r['method']:r for r in data['methods']};track={r['method']:r for r in data['transport_tracking']}
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.7),layout='constrained')
    colors=['#50677D','#25848B'];width=.34
    def bars(ax,labels,first,second,ylabel):
        x=np.arange(len(labels))
        a=ax.bar(x-width/2,first,width,color=colors[0],label='摩擦比基线')
        b=ax.bar(x+width/2,second,width,color=colors[1],label='新联合策略')
        ax.set_xticks(x,labels);ax.set_ylabel(ylabel);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
        ax.bar_label(a,fmt='%.3f',padding=3,fontsize=9);ax.bar_label(b,fmt='%.3f',padding=3,fontsize=9)
        ax.set_ylim(0,max([*first,*second])*1.22)
    bars(axes[0],['静态窗口','动态窗口'],
        [rows['ratio'][q]['TV_N_s'] for q in ('static','dynamic')],
        [rows['joint'][q]['TV_N_s'] for q in ('static','dynamic')],'逐指力变化率 TV（N/s）')
    axes[0].set_title('持握品质：同批六场景均值')
    axes[0].legend(frameon=False,loc='upper left')
    axes[1].bar([0,1],[rows['ratio_release']['raw_force_guard_pass'],rows['joint_release']['raw_force_guard_pass']],color=colors)
    axes[1].set_xticks([0,1],['基线＋新释放','新策略＋新释放'])
    axes[1].set_ylim(0,6.8);axes[1].set_yticks(range(7));axes[1].set_ylabel('通过局数 / 6')
    axes[1].set_title('全程限力：相同新释放强对照')
    for i,n in enumerate([rows['ratio_release']['raw_force_guard_pass'],rows['joint_release']['raw_force_guard_pass']]):
        axes[1].text(i,n+.12,f'{n}/6',ha='center',fontweight='bold',color='#994C3C' if i else colors[0])
    axes[1].set_axisbelow(True);axes[1].grid(axis='y',alpha=.18)
    bars(axes[2],['升降\nmm','腕轴1\nmrad','腕轴2\nmrad'],
        (np.array(track['ratio']['rmse'])*1000).tolist(),(np.array(track['joint']['rmse'])*1000).tolist(),
        '实际运动跟踪 RMSE（各组单位见标签）')
    axes[2].set_title('运输跟踪：仍有精度代价')
    fig.suptitle('V5整体重构：品质改善与尚未解决的代价',fontsize=16,fontweight='bold')
    fig.supxlabel('36条开发试验，6场景×6方法，单一新种子；双方任务完成均6/6。未做独立确认，不能宣称总体优越。',fontsize=10)
    fig.savefig(out,dpi=180);fig.savefig(folder/'品质与代价.svg');plt.close(fig)
    print(str(out))


if __name__=='__main__':main()

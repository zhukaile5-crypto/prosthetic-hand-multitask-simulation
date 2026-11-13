"""Export scientific figures from frozen trials; representative seed is prespecified."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from controllers import METHOD_TITLES

ROOT = Path(__file__).resolve().parent
FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.unicode_minus':False,'svg.fonttype':'path'})
COLORS = {'event':'#3274A1','ratio':'#B36A2C','adaptive':'#718047','fusion':'#745196'}
STYLES = {'event':'-','ratio':'--','adaptive':'-.','fusion':':'}
SHORT = ['固定握力','事件规则','摩擦比','逐指自适应','完整融合','去预测','去分配','去中值']


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def heatmap(directory,out,suite):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    rows = json.loads((directory/'trials.json').read_text(encoding='utf-8'))
    methods,names = manifest['methods'],manifest['scenarios']
    titles = {s['name']:s['title'] for s in manifest['scenario_definitions']}
    matrix = np.array([[sum(r['safe_completed'] for r in rows if r['method']==m and r['scenario']==s)
                        for m in methods] for s in names])
    n = len(manifest['seeds'])
    fig,ax=plt.subplots(figsize=(12.5,10))
    im=ax.imshow(matrix/n*100,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(methods)),SHORT)
    ax.set_yticks(range(len(names)),[titles[s]+('（边界）' if s=='infeasible' else '') for s in names])
    for i in range(len(names)):
        for j in range(len(methods)):
            ax.text(j,i,f'{matrix[i,j]}/{n}',ha='center',va='center',
                    color='white' if matrix[i,j]/n>.55 else '#202A34',fontsize=11)
    ax.set_title(f'{suite}：全程五指限力合格任务完成数',loc='left',pad=20,fontsize=16)
    ax.set_xticks(np.arange(-.5,len(methods)),minor=True)
    ax.set_yticks(np.arange(-.5,len(names)),minor=True)
    ax.grid(which='minor',color='white',linewidth=1)
    ax.tick_params(which='minor',bottom=False,left=False)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.025,label='完成率（%）')
    fig.text(.01,.02,'每一列使用相同环境与噪声种子；限力判据包括抓握阶段。掌部压力与真实材料损伤未验证。',fontsize=10)
    fig.subplots_adjust(left=.30,bottom=.10,top=.92,right=.92)
    save(fig,out,'completion_'+manifest['suite'])


def trajectory(directory,out,name,seed):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    results=json.loads((directory/'trials.json').read_text(encoding='utf-8'))
    scenario=next(s for s in manifest['scenario_definitions'] if s['name']==name)
    limits=np.array(scenario['limit'])
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for method in COLORS:
        result=next(r for r in results if (r['scenario'],r['method'],r['seed'])==(name,method,seed))
        with np.load(directory/result['trace_file']) as data:
            x=data['trace'];cols=data['columns'].tolist()
        get=lambda col:x[:,cols.index(col)]
        t=get('time');normal=x[:,[cols.index(f'normal_{i}') for i in range(5)]]
        kwargs=dict(color=COLORS[method],linestyle=STYLES[method],linewidth=1.7,label=METHOD_TITLES[method])
        axes[0].plot(t,np.sum(normal,axis=1),**kwargs)
        axes[1].plot(t,np.max(normal/(limits+.25),axis=1),**kwargs)
        displacement=get('position_error_m')*1000
        displacement[t<8.]=np.nan
        if scenario['release_at'] is not None:
            displacement[t>=scenario['release_at']-1e-8]=np.nan
        axes[2].plot(t,displacement,**kwargs)
        if result['drop_time_s'] is not None:
            i=np.argmin(abs(t-result['drop_time_s']))
            axes[2].scatter(t[i],displacement[i],marker='x',s=50,color=COLORS[method],zorder=5)
    axes[0].axhline(scenario['budget'],color='#444444',linewidth=1,linestyle='--',label='五指预算')
    axes[0].axhline(scenario['budget']+.5,color='#999999',linewidth=1,linestyle=':')
    axes[1].axhline(1.,color='#444444',linewidth=1,linestyle='--')
    for ax in axes:
        ax.axvline(8.,color='#9DA5AE',linewidth=.8)
        ax.grid(axis='y',color='#DFE3E7',linewidth=.6)
        ax.set_xlim(0,17)
        if scenario['release_at'] is not None:
            ax.axvline(scenario['release_at'],color='#444444',linewidth=.8)
    axes[0].set_ylabel('五指法向力之和（N）')
    axes[1].set_ylabel(r'最大 $n_i/(L_i+0.25\,N)$')
    axes[2].set_ylabel('相对位置偏移（mm）')
    axes[2].set_xlabel('仿真时间（s）')
    axes[0].legend(ncol=3,loc='upper left',fontsize=10)
    fig.suptitle(f'{scenario["title"]} · 预先选定配对种子 {seed}',fontsize=17,y=.98)
    fig.text(.01,.015,'8 s 开始评价持握；叉号为确认掉落。掉落后曲线保留自由运动，释放后的偏移不填零；轨迹仅作示例。',fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,top=.91,bottom=.09,hspace=.20)
    save(fig,out,f'trajectory_{name}_{seed}')


def paired_intervals(analysis,out):
    effects=json.loads((analysis/'paired_effects.json').read_text(encoding='utf-8'))['effects']
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharex=True)
    selected=[r for r in effects if r['domain']=='main' and r['metric']=='safe_completed'
              and r['condition']=='all' and r['comparator'] in ('constant','event','ratio','adaptive')]
    magnitude=max(3.,max(abs(v) for r in selected for v in r['ci95']))*1.25
    for ax,suite,title in zip(axes,('test','robustness'),('正式留出 · 20 配对种子','额外参数扰动 · 10 配对种子')):
        rows=[next(r for r in selected if r['suite']==suite and r['comparator']==m)
              for m in ('constant','event','ratio','adaptive')]
        for y,row in enumerate(rows):
            low,high=row['ci95'];estimate=row['estimate']
            ax.plot([low,high],[y,y],color='#745196',linewidth=2)
            ax.plot([low,low],[y-.08,y+.08],color='#745196',linewidth=1)
            ax.plot([high,high],[y-.08,y+.08],color='#745196',linewidth=1)
            ax.scatter(estimate,y,s=65,edgecolor='#745196',
                       facecolor='#745196' if row['p_value_holm']<.05 else 'white',zorder=4)
            ax.text(.99,y+.23,f'{estimate:+.1f} [{low:+.1f}, {high:+.1f}]；p={row["p_value_holm"]:.3g}',
                    transform=ax.get_yaxis_transform(),ha='right',fontsize=9,color='#40464C')
        ax.set_yticks(range(4),['融合 − '+SHORT[i] for i in range(4)])
        ax.set_ylim(3.60,-.35)
        ax.axvline(0,color='#666666',linewidth=1,linestyle='--')
        ax.set_xlim(-magnitude,magnitude)
        ax.set_title(title,fontsize=14,loc='left',pad=15)
        ax.set_xlabel('限力合格任务完成率差（百分点）')
        ax.grid(axis='x',color='#DFE3E7',linewidth=.6)
    fig.suptitle('融合相对基线的配对差值与 95% 区间',fontsize=17,y=1.01)
    fig.text(.01,.01,'15 个非边界场景；区间按种子跨场景簇重采样，未做多重区间校正。p 为四基线 Holm 校正；空心点 p ≥ 0.05。',fontsize=10)
    fig.subplots_adjust(left=.14,right=.99,bottom=.16,top=.88,wspace=.48)
    save(fig,out,'paired_safe_completion')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--robustness',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--analysis',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    heatmap(args.test,args.out,'正式留出测试')
    heatmap(args.robustness,args.out,'额外参数扰动测试')
    for scene in ('three_pulses','friction_drop','fragile_conflict','release_slip','infeasible','contact_robustness'):
        trajectory(args.test,args.out,scene,3001)
    if args.analysis:
        paired_intervals(args.analysis,args.out)
    inputs=[args.test/'trials.json',args.robustness/'trials.json',Path(__file__)]
    if args.analysis:
        inputs.append(args.analysis/'paired_effects.json')
    source={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for pattern in ('*.png','*.svg') for p in args.out.glob(pattern)}
    (args.out/'figure_sources.json').write_text(json.dumps(dict(sources=source,outputs=outputs,representative_seed=3001),indent=2),encoding='utf-8')
    print('Saved scientific figures as PNG and editable SVG',flush=True)


if __name__=='__main__':
    main()

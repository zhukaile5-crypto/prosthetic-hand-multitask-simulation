"""Static scientific figures from saved, reviewed post-hoc metrics and original traces."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    meta=read(HERE/'output/metadata.json'); assert meta['passed']
    path=HERE/'output/paired_effects.json'; assert sha(path)==meta['output_sha256'][path.name]
    effects=read(path)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#28333b','text.color':'#28333b','xtick.color':'#28333b','ytick.color':'#28333b',
        'savefig.facecolor':'white','svg.fonttype':'path'})
    out=HERE/'figures'; out.mkdir(exist_ok=True)
    sources={str(path):sha(path),str(HERE/'output/metadata.json'):sha(HERE/'output/metadata.json'),
             str(Path(__file__)):sha(__file__)}
    artifacts=[]
    def save(fig,name):
        for extension in ('png','svg'):
            target=out/f'{name}.{extension}'; fig.savefig(target,dpi=190,bbox_inches='tight')
            artifacts.append({'path':str(target),'sha256':sha(target)})
        plt.close(fig)
    rows=[('test','coordinated'),('robustness','coordinated'),('test','fusion'),('robustness','fusion')]
    labels=[]
    metrics=[('force_total_sd_N','合力时间标准差差值（N）'),
             ('force_finger_tv_Nps','逐指总变差率差值（N/s）'),
             ('force_total_highpass_rms_N','合力快速变化代理差值（N）')]
    fig,axes=plt.subplots(1,3,figsize=(16,5.6),sharey=True)
    for axis,(metric,title) in zip(axes,metrics):
        for y,(suite,focal) in enumerate(rows):
            match=[r for r in effects if r['suite']==suite and r['focal']==focal and r['comparator']=='ratio'
                   and r['domain']=='quiet' and r['condition']=='jointly_completed' and r['metric']==metric]
            assert len(match)==1; r=match[0]; estimate=r['estimate']; lo,hi=r['ci95']
            color='#326b87' if focal=='coordinated' else '#bd693f'
            axis.plot([lo,hi],[y,y],color=color,lw=2)
            axis.plot(estimate,y,'o',color=color,ms=7)
            axis.plot([lo,lo],[y-.07,y+.07],color=color,lw=1.5)
            axis.plot([hi,hi],[y-.07,y+.07],color=color,lw=1.5)
            if metric==metrics[0][0]:
                labels.append(('正式' if suite=='test' else '扰动')+' · '+('新协调' if focal=='coordinated' else '旧融合')+f"  n={r['n_pairs']}")
        axis.axvline(0,color='#71797e',ls='--',lw=1)
        axis.set_xlabel(title); axis.grid(axis='x',color='#e3e7e9',lw=.8); axis.set_ylim(3.6,-.6)
        axis.set_title('负值表示候选低于摩擦比基线',fontsize=11,pad=13)
    axes[0].set_yticks(range(4),labels)
    fig.suptitle('力平稳性：六个无机械外扰场景中的共同完成队列',fontsize=18,x=.06,ha='left',y=1.01)
    fig.text(.06,-.06,'每点比较相同场景/种子、双方均完成的8–17秒窗口；六场景含四种形状、触觉噪声与丢包。\n'
             '横线为95%种子簇置信区间（未作多重校正）；全部是事后探索，成功样本筛选可能产生偏差。',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.93]); save(fig,'force_smoothness_effects')
    directory=ROOT/'output/test_holdout_v2'
    manifest_path=directory/'manifest.json'; trials_path=directory/'trials.json'
    sources.update({str(manifest_path):sha(manifest_path),str(trials_path):sha(trials_path)})
    trials=read(trials_path); indexed={(r['scenario'],r['method'],r['seed']):r for r in trials}
    coverage=read(ROOT/'review/trace_coverage_audit.json')
    covered={r['path']:r['npz_sha256'] for s in coverage['suites'] for r in s['records']}
    fig,axes=plt.subplots(3,2,figsize=(16,11.8),sharex=True)
    styles={'ratio':('#353e44','摩擦比基线'), 'fusion':('#bd693f','旧融合'), 'coordinated':('#326b87','新协调')}
    for col,(scene,title) in enumerate((('friction_drop','摩擦突降'),('local_limit','局部限力＋三次拉动'))):
        for method,(color,label) in styles.items():
            trial=indexed[(scene,method,8001)]; trace_path=directory/trial['trace_file']
            sources[str(trace_path)]=sha(trace_path); assert sources[str(trace_path)]==covered[str(trace_path.resolve())]
            with np.load(trace_path,allow_pickle=False) as data: trace=data['trace']; columns=data['columns'].tolist()
            idx={name:i for i,name in enumerate(columns)}; t=trace[:,idx['time']]; mask=(t>=8)&(t<17)
            force=trace[:,[idx[f'normal_{i}'] for i in range(5)]].sum(axis=1)
            individual_command=trace[:,[idx[f'desired_{i}'] for i in range(5)]]
            command=individual_command.sum(axis=1)
            finger_change=np.r_[0.,np.abs(np.diff(individual_command,axis=0)).sum(axis=1)/.005]
            axes[0,col].plot(t[mask],force[mask],color=color,lw=1.1,label=label)
            axes[1,col].plot(t[mask],command[mask],color=color,lw=1.1,label=label)
            axes[2,col].plot(t[mask],finger_change[mask],color=color,lw=1.1,label=label)
            drop=trial.get('drop_time_s')
            if drop is not None:
                for axis in axes[:,col]: axis.axvline(drop,color=color,ls=':',lw=1)
        axes[0,col].set_title(title+' · 固定种子8001',fontsize=13,pad=13)
        for axis in axes[:,col]:
            axis.axvspan(8.2,8.9,color='#d8dde0',alpha=.4)
            if scene=='local_limit':
                for onset in (9.,11.,13.): axis.axvspan(onset,onset+.45,color='#c4cbd0',alpha=.25)
            else: axis.axvline(9.,color='#888e92',ls='--',lw=1)
            axis.grid(color='#e3e7e9',lw=.6); axis.set_xlim(8.,17.); axis.set_ylim(bottom=0)
            axis.set_xticks(np.arange(8,18)); axis.set_xlabel('时间（s）')
        axes[0,col].set_ylabel('真实五指法向力之和（N）'); axes[1,col].set_ylabel('五指握力请求之和（N）')
        axes[2,col].set_ylabel('瞬时逐指命令变化速率（N/s）')
    axes[0,0].legend(loc='upper left',frameon=False,ncol=3,fontsize=10)
    fig.suptitle('真实力与握力命令：原正式计划固定种子案例',fontsize=18,x=.06,ha='left',y=.99)
    fig.text(.06,-.02,'使用原200 Hz记录；不做平滑、不裁剪峰值。灰区：预扰动窗口／拉动时段；彩色虚线：确认失抓时刻（如有）。\n'
             '该两案例沿用原轮预定回放，不证明总体优势；0.5 ms子步峰值与接触模型可信度另见原报告。',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.94]); save(fig,'force_command_waveforms')
    assert all(sha(path)==expected for path,expected in sources.items())
    (out/'figure_sources.json').write_text(json.dumps(dict(sources=sources,artifacts=artifacts),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=2,artifacts=len(artifacts),sources=len(sources)),ensure_ascii=False))


if __name__=='__main__': main()

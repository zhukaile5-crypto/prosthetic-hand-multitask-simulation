"""All three registered failure-case trajectories; raw saved data only."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from review.run_v16_release_small_v1 import previous_A

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':10})
records=json.loads((ROOT/'report/v15A_prospective_v2/paired_readout_v1.json').read_text(encoding='utf-8'))['records']
contexts=[('fragile_conflict','hold','脆弱物体冲突 / 静态'),('nominal','transport','常规 / 运输'),('friction_drop','transport','摩擦下降 / 运输')]
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
for row,(scene,motion,title) in enumerate(contexts):
    ratio=next(r for r in records if r['role']=='baseline' and r['force_preference']=='ratio_effort' and r['context_key']==[scene,motion,619732])
    group='coordinated_hold' if motion=='hold' else 'coordinated_transport'
    paths=[('旧候选A',previous_A(scene,motion,619732),'#8152a5'),
        ('V16释放原型',ROOT/f'output/v16_release_{group}_v1/runs/{scene}__ratio_effort__619732','#cf543b'),
        ('力比基线',Path(ratio['bundle']['run_directory']),'#247bb0')]
    for label,path,color in paths:
        with np.load(path/'trace.npz',allow_pickle=False) as data:
            mask=data['time']>=12.-1e-9;t=data['time'][mask]
            axes[row,0].plot(t,data['all_hand_object'][mask],label=label,color=color,lw=1.5)
            axes[row,1].plot(t,data['position'][mask,2],label=label,color=color,lw=1.5)
    axes[row,0].axhline(.2,color='#777',ls='--',lw=.8)
    axes[row,0].set_ylabel(title+'\n手–目标接触合量 / N')
    axes[row,1].set_ylabel('物体高度 / m')
    for ax in axes[row]:ax.grid(alpha=.18);ax.set_xlim(12,16)
axes[0,0].legend(loc='upper right',fontsize=9)
axes[0,0].set_title('低落地事件必须与清手接触一起解释')
axes[0,1].set_title('V16三条无落地，但仍未完成物理清离')
axes[-1,0].set_xlabel('时间 / s');axes[-1,1].set_xlabel('时间 / s')
fig.suptitle('V16首轮：三条已见失败条件的完整释放记录',fontsize=15,y=.985)
fig.text(.05,.015,'seed619732开发反例，非独立确认；全部三条保留。V16误用预抓OPEN终点，已达到参考目标但仍接触目标。\n曲线来自保存状态，未新跑物理或改评分；虚线为0.2 N参考。原判据下V16清离0/3、落地0/3。',fontsize=10)
fig.tight_layout(rect=(0,.075,1,.96))
out=ROOT/'report/v16_small_v1';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'释放清离与落地的实际代价.png',dpi=160);plt.close(fig)
print(str(out/'释放清离与落地的实际代价.png'))

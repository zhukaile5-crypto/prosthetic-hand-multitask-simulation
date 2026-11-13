from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report/v15_stage_v1';OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');records=json.loads((OUT/'readout_root_v1.json').read_text(encoding='utf-8'))['records']
cells=['旧策略+被动观测','单一持握任务','共享释放任务图','完整组合'];short=['旧策略','仅新持握','仅新释放','完整组合'];colors=['#929fac','#8f78bd','#d8944d','#188c89'];methods=['ratio_effort','fixed_effort','fusion_effort'];names=['摩擦比偏好','固定力偏好','融合偏好'];xx=np.arange(3);width=.18
fig,axes=plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
for ax,key,title in zip(axes.flat,['static_mean_N','static_TV_N_s','peak_N','local_dose_Ns'],['原固定稳态窗平均总指力（N）','原固定稳态窗逐指力变化率（N/s）','全16秒最高单指力（N）','全16秒局部超限积分（mN·s）']):
 allvals=[]
 for j,cell in enumerate(cells):
  values=[next(r[key] for r in records if r['cell']==cell and r['method']==m) for m in methods];values=[None if v is None else v*(1000. if key=='local_dose_Ns' else 1.) for v in values];numeric=[np.nan if v is None else v for v in values];allvals+=numeric
  bars=ax.bar(xx+(j-1.5)*width,numeric,width,label=short[j],color=colors[j])
  for b,v in zip(bars,values):
   if v is not None:ax.annotate(f'{v:.3g}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom',fontsize=8,xytext=(0,3),textcoords='offset points')
  for x,v in zip(xx+(j-1.5)*width,values):
   if v is None:ax.text(x,0.,'N/A',rotation=90,ha='center',va='bottom',fontsize=9,color=colors[j])
 ax.set_title(title,fontproperties=fp,fontsize=12);ax.set_xticks(xx,names,fontproperties=fp);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(v for v in allvals if np.isfinite(v))*1.23 if any(np.isfinite(v) for v in allvals) else 1)
axes[0,0].legend(prop=fp,ncol=2,frameon=False,loc='upper right')
fig.suptitle('V15已完成12条：手部力指标有收益，接触资格与交付仍有缺口',fontproperties=fp,fontsize=16)
fig.text(.5,-.02,'同一已见标称条件。N/A来自原三指接触资格不足，不能视为零或直接判定物体掉落；全程峰/剂量仍保留失败记录。',fontproperties=fp,fontsize=10,ha='center')
fig.savefig(OUT/'多任务消融的收益与资格代价.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,layout='constrained')
for col,(method,title) in enumerate(zip(methods,names)):
 for j,cell in enumerate(cells):
  item=next(v for v in records if v['cell']==cell and v['method']==method)
  with np.load(ROOT/'output'/item['output']/'runs'/f"nominal__{method}__617901"/'trace.npz',allow_pickle=False) as z:t=z['time'];p=z['position']
  mask=t>=12.-1e-9
  axes[0,col].plot(t[mask],p[mask,0],color=colors[j],lw=1.7,label=short[j]);axes[1,col].plot(t[mask],p[mask,2],color=colors[j],lw=1.7)
 axes[0,col].axhspan(.025,.165,color='#a2b4ab',alpha=.22,label='接收台x范围（仅中心）');axes[0,col].set_title(title,fontproperties=fp,fontsize=13)
 axes[1,col].axhline(.337,color='#65736b',ls='--',lw=1);axes[1,col].text(12.03,.344,'台顶高度',fontproperties=fp,fontsize=9,color='#65736b')
 for ax in axes[:,col]:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_xlim(12,16)
 axes[1,col].set_xlabel('时间（s）',fontproperties=fp)
axes[0,0].set_ylabel('物体中心 x（m）',fontproperties=fp);axes[1,0].set_ylabel('物体中心 z（m）',fontproperties=fp)
axes[0,0].legend(prop=fp,fontsize=9,frameon=False,loc='upper left')
fig.suptitle('触台与低手力均不代表稳定交付：新增释放图三法随后都落地',fontproperties=fp,fontsize=16)
fig.text(.5,-.025,'离线物体真值仅用于评价，未进入控制。阴影仅为中心x范围，并非全物体在台面或稳定放置证书；所有原成功标签保留。',fontproperties=fp,fontsize=10,ha='center')
fig.savefig(OUT/'触台后的落点与交付代价.png',dpi=150,bbox_inches='tight');plt.close(fig)
print('saved two source-backed static figures')

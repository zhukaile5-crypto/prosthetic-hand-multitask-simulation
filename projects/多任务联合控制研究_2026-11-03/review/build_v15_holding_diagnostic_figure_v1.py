from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report/v15_stage_v1';os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
methods=['ratio_effort','fixed_effort','fusion_effort'];names=['摩擦比偏好','固定力偏好','融合偏好'];colors=['#929fac','#188c89']
fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,layout='constrained')
for col,(method,title) in enumerate(zip(methods,names)):
 for j,(label,folder) in enumerate([('旧持握','v15_passive_blind_bridge_v1'),('新持握任务','v15_holding_task_ablation_v1')]):
  path=ROOT/'output'/folder/'runs'/f'nominal__{method}__617901'/'trace.npz'
  with np.load(path,allow_pickle=False) as z:t=z['time'];force=z['normal'];relative=z['relative']
  i=int(np.flatnonzero(abs(t-7.5)<1e-9)[0]);mask=(t>=7.5-1e-9)&(t<12.-1e-9)
  drift=np.linalg.norm(relative-relative[i],axis=1)*1000.
  axes[0,col].plot(t[mask],force[mask].sum(axis=1),color=colors[j],lw=1.1,label=label)
  axes[1,col].plot(t[mask],drift[mask],color=colors[j],lw=1.4)
 axes[0,col].set_title(title,fontproperties=fp,fontsize=13)
 for ax in axes[:,col]:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_xlim(7.5,12)
 axes[1,col].set_xlabel('时间（s）',fontproperties=fp)
axes[0,0].set_ylabel('实际总指力（N）',fontproperties=fp);axes[1,0].set_ylabel('物体相对接管位置漂移（mm）',fontproperties=fp)
axes[0,0].legend(prop=fp,frameon=False)
fig.suptitle('未筛选持握诊断：新融合更省力，力波动与物体扰动却明显增加',fontproperties=fp,fontsize=16)
fig.text(.5,-.025,'六条唯一持握轨迹的后验诊断；原资格/成功/NULL不改，无掉落事件也不等于稳定性相同。物体位置仅用于离线评价。',ha='center',fontproperties=fp,fontsize=10)
fig.savefig(OUT/'六条持握轨迹的力与真实漂移.png',dpi=150,bbox_inches='tight');plt.close(fig)
print('saved six-trajectory diagnostic figure')

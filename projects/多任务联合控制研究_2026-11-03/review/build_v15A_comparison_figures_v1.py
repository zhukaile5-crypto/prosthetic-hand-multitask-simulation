"""Source-backed final64 figures. Refuses incomplete final readouts."""
from pathlib import Path
import argparse,json,os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report/v15A_prospective_v2';os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
COLORS={'nominal':'#386c9a','friction_drop':'#c0833d','fragile_conflict':'#9b6f9d','collision':'#2a8b7d'}
SCENE_NAMES={'nominal':'常规','friction_drop':'摩擦变化','fragile_conflict':'易损冲突','collision':'接触碰撞'}
ARMS=[('candidate_A','ratio_effort','多任务协调A','#188b84'),('baseline','ratio_effort','力比PI基线','#4a719b'),('baseline','fixed_effort','固定力基线','#99a4ad'),('baseline','fusion_effort','旧融合','#c69b43')]
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='report/v15A_prospective_v2/paired_readout_v1.json');args=ap.parse_args()
 source=ROOT/args.input;data=json.loads(source.read_text(encoding='utf-8-sig'))
 if not data['complete']:raise ValueError('Final figures require all64 assignments complete')
 OUT.mkdir(parents=True,exist_ok=True)
 pairs=[p for p in data['pairs'] if p['primary_same_preference_contrast']]
 if len(pairs)!=16:raise ValueError('Exactly16 same-force-prior paired contexts required')
 fields=[('raw.static.mean_total_force_N','稳态平均总指力（N）',1.,None),('raw.static.finger_TV_rate_N_s','稳态逐指力变化率（N/s）',1.,.1),('raw.dynamic.finger_TV_rate_N_s','动态窗逐指力变化率（N/s）',1.,.1),('raw.whole_holding.finger_TV_rate_N_s','全持握逐指力变化率（N/s）',1.,.1),('object.whole_holding.relative_handoff_max_displacement_m','最大掌相对位移（mm）',1000.,.05),('original.safety.local_excess_integral_Ns','全程局部超限积分（N·s）',1.,.001)]
 fig,axes=plt.subplots(2,3,figsize=(15.5,10),layout='constrained');plotted={}
 for ax,(key,title,scale,linearpart) in zip(axes.flat,fields):
  values=[];n=0
  for p in pairs:
   d=p['differences'][key];x=d['baseline'];y=d['candidate']
   if not finite(x) or not finite(y):continue
   x*=scale;y*=scale;values.extend([x,y]);n+=1
   scene,motion,seed=p['context_key'];marker='o' if motion=='hold' else '^'
   ax.scatter(x,y,s=45,marker=marker,color=COLORS[scene],edgecolors='white',linewidths=.5,alpha=.88)
  top=max(values)*1.12 if values else 1.;top=max(top,1e-9)
  if linearpart is not None:ax.set_xscale('symlog',linthresh=linearpart);ax.set_yscale('symlog',linthresh=linearpart)
  ax.plot([0,top],[0,top],ls='--',color='#9aa5a5',lw=1,zorder=0);ax.set_xlim(0,top);ax.set_ylim(0,top)
  ax.set_title(title+f' · {n}/16对',fontproperties=fp,fontsize=12);ax.set_xlabel('力比PI基线',fontproperties=fp);ax.set_ylabel('多任务协调A',fontproperties=fp)
  ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False);plotted[key]=n
 from matplotlib.lines import Line2D
 handles=[Line2D([],[],marker='o',ls='',color=c,label=SCENE_NAMES[s]) for s,c in COLORS.items()]+[Line2D([],[],marker='o',ls='',color='#475d64',label='原地持握'),Line2D([],[],marker='^',ls='',color='#475d64',label='运输')]
 axes[0,0].legend(handles=handles,prop=fp,loc='upper left',ncol=2,frameon=False,fontsize=9)
 fig.suptitle('同一力比先验：多任务协调在16个配对条件中的影响',fontproperties=fp,fontsize=17)
 fig.text(.5,-.018,'原固定窗的全分配原始200Hz统计；低于虚线表示该维度数值下降。两个物理种子，不是16次独立随机重复；任务和交付代价另列。',ha='center',fontproperties=fp,fontsize=10)
 fig.savefig(OUT/'同力比先验的十六条件配对.png',dpi=155,bbox_inches='tight');plt.close(fig)
 groups=[]
 for role,pref,label,color in ARMS:
  rr=[r for r in data['records'] if r['role']==role and r['force_preference']==pref]
  if len(rr)!=16:raise ValueError('Each arm must retain16 assigned records')
  groups.append((label,color,rr))
 def values(records,key):return [r['metrics'].get(key) for r in records]
 def avg(records,key):
  v=[x for x in values(records,key) if finite(x)]
  return (float(np.mean(v)) if v else None,len(v))
 def count(records,key):
  data=values(records,key)
  if any(type(x) is not bool for x in data):raise ValueError('Count metric unavailable or wrong schema key: '+key)
  return sum(x is True for x in data)
 xx=np.arange(4);labels=[g[0] for g in groups];colors=[g[1] for g in groups]
 fig,axes=plt.subplots(2,3,figsize=(15.5,9.5),layout='constrained')
 for ax,keys,title in [(axes[0,0],[('original.legacy.task_success','原任务'),('original.full_motion_mission_success','完整运动')],'任务完成（各16条）'),(axes[0,1],[('original.safety.guard_pass','全程力限通过'),('contact.release_and_after.floor.any_positive_contact','释放后地板事件')],'手力限与落地分别计数')]:
  for j,(key,name) in enumerate(keys):
   vals=[count(g[2],key) for g in groups];bars=ax.bar(xx+(j-.5)*.32,vals,.32,label=name,color='#4a7f72' if j==0 else '#c08e69')
   for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.25,str(v),ha='center',fontsize=10)
  ax.set_ylim(0,21. if ax is axes[0,0] else 18.6);ax.set_title(title,fontproperties=fp);ax.legend(prop=fp,fontsize=9,frameon=False)
 scalar=[(axes[0,2],'raw.whole_holding.mean_total_force_N','全持握平均总指力（N）',1.),(axes[1,0],'original.safety.local_excess_integral_Ns','全程超限积分：条件均值（N·s）',1.),(axes[1,1],'object.whole_holding.relative_handoff_max_displacement_m','最大掌相对位移：条件均值（mm）',1000.),(axes[1,2],'timing.decision.control_ms.median','各条件决策分支中位耗时：均值（ms）',1.)]
 for ax,key,title,scale in scalar:
  data_vals=[avg(g[2],key) for g in groups];vv=[np.nan if v is None else v*scale for v,n in data_vals];bars=ax.bar(xx,vv,.58,color=colors)
  for b,(val,n) in zip(bars,data_vals):
   if val is not None:ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{val*scale:.3g}\n({n}/16)',ha='center',va='bottom',fontsize=9)
  ax.set_title(title,fontproperties=fp,fontsize=12);ax.set_ylim(0,max(v for v in vv if np.isfinite(v))*1.25)
 for ax in axes.flat:ax.set_xticks(xx,labels,fontproperties=fp,fontsize=10);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('四种策略的综合取舍：保持、用力、波动与交付同时报告',fontproperties=fp,fontsize=17)
 fig.text(.5,-.02,'全部64分配保留，旧任务与品质NULL不改。辅助连续指标不生成新成功标签；较低手力不代表稳定交付，耗时不构成硬实时保证。',ha='center',fontproperties=fp,fontsize=10)
 fig.savefig(OUT/'四种策略的综合收益与代价.png',dpi=155,bbox_inches='tight');plt.close(fig)
 (OUT/'figure_readout_v1.json').write_text(json.dumps(dict(source=str(source),same_prior_pairs=len(pairs),plotted_available=plotted,arms=[dict(label=g[0],n=len(g[2])) for g in groups],scope='Descriptive fullassigned64,16contexts in2 physicalseed clusters'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('saved final64 paired and multi-criterion figures')
if __name__=='__main__':main()


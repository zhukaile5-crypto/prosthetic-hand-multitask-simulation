from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'report/v14_coherent_stage_v1';out.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');records=[]
cohorts=[('原native','development_v12_native_pilot_v1'),('V13','development_v13_joint_pilot_v1'),('V14','development_v14_coherent_pilot_v1')]
for version,cohort in cohorts:
 for x in json.loads((ROOT/'output'/cohort/'trials.json').read_text(encoding='utf-8-sig')):
  m=x['research_metrics'];q=m['static_quality'];h=m['entire_holding_quality'];dy=m['dynamic_quality'];safe=m['safety']
  branch=fullmed=fullmax=decision_count=None
  if version!='原native':
   folder=ROOT/'output'/cohort/'runs'/f"nominal__{x['method']}__617901"
   with np.load(folder/'trace.npz',allow_pickle=False) as a:
    # Each compressed array is materialized once, never inside a row loop.
    times=a['time'];debug=a['debug_json'];control=a['control_ms'];full=a['full_control_loop_ms']
   ids=[i for i,(t,text) in enumerate(zip(times,debug)) if abs(json.loads(str(text)).get('decision_time_s',-1.)-t)<1e-9]
   assert len(ids)==426
   decision_count=len(ids);branch=float(np.median(control[ids]));fullmed=float(np.median(full[ids]));fullmax=float(np.max(full[ids]))
   del debug
  records.append(dict(version=version,method=x['method'],task=x['task_success'],full_motion=m['full_motion_mission_success'],force_guard=safe['guard_pass'],static_mean_N=q['mean_actual_finger_force_N'],static_TV_N_s=q['raw_200Hz']['finger_TV_rate_N_s'],static_total_SD_N=q['raw_200Hz']['total_force_SD_N'],dynamic_TV_N_s=dy['raw_200Hz']['finger_TV_rate_N_s'],whole_hold_TV_N_s=h['raw_200Hz']['finger_TV_rate_N_s'],peak_N=max(safe['finger_peak_N']),local_dose_Ns=safe['local_excess_integral_Ns'],release_latency_s=m['release']['physical_clearance_latency_s'],decision_count=decision_count,control_branch_median_ms=branch,full_loop_median_ms=fullmed,full_loop_max_ms=fullmax))
  print(json.dumps(dict(read=version,method=x['method']),ensure_ascii=False),flush=True)
with (out/'readout_root.json').open('w',encoding='utf-8') as f:json.dump(dict(scope='one seen nominal hold physical seed; oldnative timing not analyzed in this readout; no broad superiority proof',records=records),f,ensure_ascii=False,indent=2,allow_nan=False)
methods=['ratio_effort','fixed_effort','fusion_effort'];labels=['摩擦比意图','固定力意图','融合意图'];colors=['#9caebe','#d5a063','#167c86'];xx=np.arange(3)
fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
for ax,key,title in zip(axes.flat,['static_mean_N','static_TV_N_s','local_dose_Ns','control_branch_median_ms'],['稳态平均总指力（N）','稳态原200Hz逐指TV（N/s）','全程局部超限积分（N·s）','全部20ms决策控制分支中位耗时（ms）']):
 versions=['V13','V14'] if key=='control_branch_median_ms' else [v for v,c in cohorts];width=.24 if len(versions)==3 else .31
 for j,version in enumerate(versions):
  ci=[v for v,c in cohorts].index(version);vals=[next(x[key] for x in records if x['method']==m and x['version']==version) for m in methods]
  bars=ax.bar(xx+(j-(len(versions)-1)/2)*width,vals,width,color=colors[ci],label=version)
  for b,val in zip(bars,vals):ax.annotate(f'{val:.3g}',(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom',fontsize=9,xytext=(0,3),textcoords='offset points')
 ax.set_title(title,fontproperties=fp,fontsize=12);ax.set_xticks(xx,labels,fontproperties=fp);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_ylim(top=ax.get_ylim()[1]*1.18)
axes.flat[0].legend(prop=fp,frameon=False)
fig.suptitle('V14首轮闭环：求解与计算成本改善，融合力品质仍有取舍',fontproperties=fp,fontsize=16)
fig.text(.5,-.025,'同一已见标称条件；三法任务均完成、全程力限仍均失败。原评价窗/阈值不变；V13/V14各426次决策，耗时排除物理积分。',ha='center',fontproperties=fp,fontsize=10)
fig.savefig(out/'协调结构与实际代价.png',dpi=155,bbox_inches='tight');plt.close(fig)
factors=[]
for folder in ['v13_release_factorial_bridge_v1','v13_release_factorial_other3_v1']:
 terminal=json.loads((ROOT/'output'/folder/'execution_complete.json').read_text(encoding='utf-8'));assert terminal['passed']
 for item in terminal['results']:
  p=ROOT/'output'/folder/item['variant']/'trace.npz';q=item['diagnostics']['release'];cl=item['diagnostics']['low_force_clearance']
  with np.load(p,allow_pickle=False) as a:
   times=a['substep_time'];support=a['substep_support'];position=a['position'];mask=times>=12.-1e-9
   peak=float(np.max(support[mask]));z0=float(position[2400,2]);z1=float(position[-1,2])
  factors.append(dict(variant=item['variant'],finger_peak_N=q['peak_N'],finger_local_dose_Ns=q['local_excess_Ns'],external_support_peak_N=peak,max_world_penetration_m=q['max_penetration_m'],object_z_change_m=z1-z0,raw_lowforce_confirmation_s=cl['raw_integration_end_first_by_deadline']))
with (out/'factorial_readout_root.json').open('w',encoding='utf-8') as f:json.dump(dict(role='causal fixed-input diagnosis; not feedback success comparison',records=factors),f,ensure_ascii=False,indent=2,allow_nan=False)
fig,axes=plt.subplots(2,2,figsize=(11.4,7.4),layout='constrained')
for ax,(variant,title) in zip(axes.flat,[('both','张手＋下降'),('fingers_only','张手＋升降保持'),('lift_only','手姿态保持＋下降'),('neither','手姿态与升降都保持')]):
 x=next(x for x in factors if x['variant']==variant);clear='无低力确认' if x['raw_lowforce_confirmation_s'] is None else f"首次低力确认：{x['raw_lowforce_confirmation_s']-12.:.3f}s"
 note='物体离台落地，不能算较优释放' if variant=='fingers_only' else '持续压夹，未完成清手' if variant=='lift_only' else '未执行清手' if variant=='neither' else '首次低力确认后仍有再接触'
 text=f"最高单指力：{x['finger_peak_N']:.3f} N\n指部超限积分：{x['finger_local_dose_Ns']:.5g} N·s\n外界支撑峰（含地面）：{x['external_support_peak_N']:.2f} N\n最大世界穿透：{1000*x['max_world_penetration_m']:.2f} mm\n{clear}\n{note}"
 ax.set_facecolor('#f6f8fa');ax.text(.05,.9,title,transform=ax.transAxes,fontproperties=fp,fontsize=14,weight='bold');ax.text(.05,.75,text,transform=ax.transAxes,va='top',fontproperties=fp,fontsize=12,linespacing=1.65);ax.set_xticks([]);ax.set_yticks([]);ax.spines[['top','right','bottom','left']].set_visible(False)
fig.suptitle('同一释放入口的四组固定输入反事实',fontproperties=fp,fontsize=17)
fig.text(.5,-.025,'只改变12秒后的手指／升降输入；四条均完整16秒、环境与噪声相同。无新反馈，不生成新算法成功率。',ha='center',fontproperties=fp,fontsize=10)
fig.savefig(out/'释放输入因果与遗漏代价.png',dpi=155,bbox_inches='tight');plt.close(fig)
print(json.dumps(dict(records=len(records),factorial_cells=len(factors),completed=True),ensure_ascii=False),flush=True)

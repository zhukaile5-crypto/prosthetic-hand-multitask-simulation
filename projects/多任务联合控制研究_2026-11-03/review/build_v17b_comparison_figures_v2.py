"""Scientific figures from completed, audited80 only; no new metric or score.

Retains all assigned contexts and original NULL. Existing repository figures
and colors are preserved; exports are separate research PNG/SVG artifacts.
"""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
ARMS=[('full_V17','完整V17','#0b615f'),('holding_A','仅持握A','#188b84'),
      ('same_prior_legacy_ratio','力比基线','#4a719b'),('baseline_fixed','固定力','#99a4ad'),
      ('legacy_fusion','旧融合','#c69b43')]
SCENES={'nominal':('常规','#386c9a'),'friction_drop':('摩擦变化','#c0833d'),
        'fragile_conflict':('易损冲突','#9b6f9d'),'collision':('接触碰撞','#2a8b7d')}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and np.isfinite(v)
def release_peak(metrics):
    vals=[metrics.get('original.safety.stages.mechanical_release_and_after.finger_peak_N.'+str(i)) for i in range(5)]
    return max(vals) if all(finite(x) for x in vals) else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='report/v17b_prospective80_v1/paired_readout_v1.json')
    p.add_argument('--auxiliary',default='review/v17b_delivery_auxiliary_readout_v1.json')
    p.add_argument('--audit',default='review/v17b_prospective80_execution_independent_v1.json')
    p.add_argument('--out',default='report/v17b_prospective80_v1');a=p.parse_args()
    audit=read(ROOT/a.audit)
    if not (audit.get('passed') is True and audit.get('all_assigned_checked') is True
            and audit.get('assigned_count')==80 and audit.get('plan_sha256')==PLAN_SHA):
        raise ValueError('All80 independent execution audit required')
    data=read(ROOT/a.input);aux=read(ROOT/a.auxiliary)
    if not data.get('complete') or data['assigned']!=80 or data['issues'] or data['provenance']['plan_sha256']!=PLAN_SHA:
        raise ValueError('Completed same-cohort paired readout required')
    if (not aux.get('complete') or not aux.get('source_identity_passed') or aux['plan_sha256']!=PLAN_SHA
        or len(aux.get('records',[]))!=80 or aux.get('record_error_count')!=0):
        raise ValueError('Completed trusted auxiliary80 required')
    rows=data['records'];by={tuple(r['record_key']):r for r in rows};ab={tuple(r['record_key']):r for r in aux['records']}
    if len(by)!=80 or set(by)!=set(ab) or len(data['pairs'])!=112:
        raise ValueError('Exact five-arm80 four-part record join required')
    if any(len([r for r in rows if r['role']==role])!=16 for role,_,_ in ARMS):raise ValueError('Each arm retains16')
    if any(not r['measurements_ready'] for r in rows):raise ValueError('Unavailable original record retained; figures withheld')
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':10})
    manifest=dict(schema='v17b_scientific_comparison_figures_v2',plan_sha256=PLAN_SHA,
        inputs={a.input:sha(ROOT/a.input),a.auxiliary:sha(ROOT/a.auxiliary),a.audit:sha(ROOT/a.audit)},
        plotter_sha256=sha(Path(__file__)),assigned=80,contexts=16,physical_seed_clusters=2,
        no_CI_or_weighted_score=True,no_original_eligibility_changed=True,figures={})
    def save(fig,name,records):
        name=name+'_v2'
        png=out/(name+'.png');svg=out/(name+'.svg')
        if png.exists() or svg.exists():raise FileExistsError('Keep existing rendered artifacts: '+name)
        fig.savefig(png,dpi=170,bbox_inches='tight');fig.savefig(svg,bbox_inches='tight');plt.close(fig)
        manifest['figures'][name]=dict(png_sha256=sha(png),svg_sha256=sha(svg),plotted_rows=records)
    def clean_axes(ax):
        ax.grid(alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    # Original outcomes stay separate from supplemental low-motion states.
    counts=[('完整运动任务通过','original.full_motion_mission_success',None),
        ('原物理清离通过','original.release.physical_success',None),
        ('全程原力限通过','original.safety.guard_pass',None),
        ('释放窗发生落地（越少越好）','contact.release_and_after.floor.any_positive_contact',None),
        ('首次低运动接收驻留（辅助状态）',None,'first'),
        ('末段≥0.2秒低运动接收驻留（辅助状态）',None,'final')]
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained');count_rows=[]
    for ax,(title,key,kind) in zip(axes.flat,counts):
        values=[]
        for role,label,color in ARMS:
            rr=[r for r in rows if r['role']==role];vv=[]
            for r in rr:
                if kind is None:
                    if key not in r['metrics']:raise KeyError(key)
                    v=r['metrics'][key]
                else:
                    m=ab[tuple(r['record_key'])].get('auxiliary')
                    if m is None or not m['coverage']['all_valid']:v=None
                    elif kind=='first':v=m['official_first_stable_receiver'] is not None
                    else:
                        suffix=m['official_final_stable_suffix'];v=bool(suffix and not suffix['right_censored'])
                vv.append(v)
            ntrue=sum(v is True for v in vv);unknown=sum(type(v) is not bool for v in vv)
            values.append((ntrue,unknown));count_rows.append(dict(metric=key or kind,role=role,assigned=16,true=ntrue,unknown=unknown))
        bars=ax.bar(np.arange(5),[v[0] for v in values],color=[x[2] for x in ARMS],width=.62)
        bars[1].set_hatch('//')
        for bar,(n,u) in zip(bars,values):
            text=f'{n}/16'+(f'\n未知{u}' if u else '')
            ax.text(bar.get_x()+bar.get_width()/2,n+.25,text,ha='center',va='bottom',fontsize=10)
        ax.set_ylim(0,18.6);ax.set_yticks([0,4,8,12,16]);ax.set_xticks(range(5),[x[1] for x in ARMS],fontsize=10)
        ax.set_title(title,fontsize=12);ax.set_ylabel('次数 / 16条分配');clean_axes(ax)
    fig.suptitle('五种策略的任务、接触风险与低运动状态',fontsize=18)
    fig.text(.5,-.012,'80条均物理暴露，分属2个物理种子簇。后两项为辅助低运动状态，不等同完整任务成功；原失败与未达标保留。',ha='center',fontsize=11)
    save(fig,'五法任务与接触状态全分配',count_rows)
    # Six old fixed-window dimensions. Actual absolute pairs avoid zero-base ratios.
    fields=[('raw.static.mean_total_force_N','静态均力（N）',1.,None),
        ('raw.static.finger_TV_rate_N_s','静态逐指力变化率（N/s）',1.,.1),
        ('raw.dynamic.finger_TV_rate_N_s','动态窗逐指力变化率（N/s）',1.,.1),
        ('raw.whole_holding.finger_TV_rate_N_s','全持握逐指力变化率（N/s）',1.,.1),
        ('object.whole_holding.relative_handoff_max_displacement_m','最大掌相对位移（mm）',1000.,.05),
        ('original.safety.local_excess_integral_Ns','全程局部超力剂量（N·s）',1.,.001)]
    pairs=[q for q in data['pairs'] if q['contrast']==['full_V17','same_prior_legacy_ratio']]
    if len(pairs)!=16:raise ValueError('Exactly16 same-prior contrasts required')
    fig,axes=plt.subplots(2,3,figsize=(16,10.5),layout='constrained');paired_rows=[]
    for ax,(key,title,scale,linear) in zip(axes.flat,fields):
        all_values=[];n=0
        for q in pairs:
            d=q['differences'][key];x,y=d['baseline'],d['candidate'];scene,motion,seed=q['context_key']
            paired_rows.append(dict(metric=key,context=q['context_key'],reference=x,candidate=y,scale=scale))
            if not finite(x) or not finite(y):continue
            x*=scale;y*=scale;all_values.extend([x,y]);n+=1
            color=SCENES[scene][1];marker='o' if motion=='hold' else '^'
            ax.scatter(x,y,s=50,marker=marker,facecolors=color if seed==619741 else 'none',edgecolors=color,linewidths=1.2,alpha=.9)
        upper=max(max(all_values,default=0.)*1.14,1e-6)
        if linear is not None:ax.set_xscale('symlog',linthresh=linear);ax.set_yscale('symlog',linthresh=linear)
        ax.plot([0,upper],[0,upper],ls='--',color='#899798',lw=1,zorder=0);ax.set_xlim(0,upper);ax.set_ylim(0,upper)
        ax.set_title(f'{title} · {n}/16对',fontsize=12);ax.set_xlabel('力比PI基线');ax.set_ylabel('完整V17');clean_axes(ax)
    legend=[Line2D([],[],marker='o',ls='',color=c,label=label) for label,c in SCENES.values()]
    legend += [Line2D([],[],marker='o',ls='',color='#536367',label='静态'),Line2D([],[],marker='^',ls='',color='#536367',label='搬运'),
        Line2D([],[],marker='o',ls='',color='#536367',label='种子619741'),Line2D([],[],marker='o',ls='',color='#536367',markerfacecolor='none',label='种子619742')]
    axes[0,0].legend(handles=legend,loc='upper left',ncol=2,frameon=False,fontsize=8)
    fig.suptitle('完整V17相对同力比基线的16条件配对',fontsize=18)
    fig.text(.5,-.015,'原固定窗全分配200Hz读数；低于虚线仅表示该项更低。部分坐标用symlog展示近零值；两个随机簇，不画独立n16置信区间。',ha='center',fontsize=11)
    save(fig,'同力比基线的十六条件配对',paired_rows)
    # Controlled release-package contrast: holding and all pre12 packets match.
    release_fields=[('original.release.physical_clearance_latency_s','原清离延迟（s）',1.),
        (None,'释放段最大指峰力（N）',1.),('endstate.record_end16.linear_speed_m_s','16秒末线速（mm/s）',1000.),
        ('endstate.record_end16.angular_speed_rad_s','16秒末角速（rad/s）',1.)]
    contexts=sorted({tuple(r['context_key']) for r in rows});fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained');release_rows=[]
    for ax,(key,title,scale) in zip(axes.flat,release_fields):
        vals=[];n=0
        for context in contexts:
            cand=by[('full_V17',*context)]['metrics'];ref=by[('holding_A',*context)]['metrics']
            x,y=(release_peak(ref),release_peak(cand)) if key is None else (ref.get(key),cand.get(key))
            release_rows.append(dict(metric=key or 'maximum_of_original_release_finger_peaks',context=list(context),reference=x,candidate=y,scale=scale))
            if not finite(x) or not finite(y):continue
            x*=scale;y*=scale;vals.extend([x,y]);n+=1;scene,motion,seed=context;color=SCENES[scene][1]
            ax.scatter(x,y,s=53,marker='o' if motion=='hold' else '^',facecolors=color if seed==619741 else 'none',edgecolors=color,linewidths=1.2)
        upper=max(max(vals,default=0)*1.14,1e-6)
        if key in ('endstate.record_end16.linear_speed_m_s','endstate.record_end16.angular_speed_rad_s'):
            threshold=5. if scale==1000. else .15
            ax.set_xscale('symlog',linthresh=threshold);ax.set_yscale('symlog',linthresh=threshold)
        ax.plot([0,upper],[0,upper],ls='--',color='#899798',lw=1)
        ax.set_xlim(0,upper);ax.set_ylim(0,upper);ax.set_title(f'{title} · {n}/16对',fontsize=12)
        ax.set_xlabel('仅持握A＋原释放');ax.set_ylabel('完整V17接触转移');clean_axes(ax)
    axes[0,0].legend(handles=legend,loc='upper left',ncol=2,frameon=False,fontsize=8)
    fig.suptitle('同一持握前缀：释放任务包的收益与代价',fontsize=18)
    fig.text(.5,-.015,'16个同条件对照均保留；末态速度用symlog展示低速与落地高速，线性区分别为5mm/s与0.15rad/s。清离NULL不补零，低末速不等于成功。',ha='center',fontsize=11)
    save(fig,'释放任务包的十六条件取舍',release_rows)
    receipt=out/'figure_readout_v2.json'
    if receipt.exists():raise FileExistsError(receipt)
    receipt.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=3,source_records=80,receipt_sha256=sha(receipt))))


if __name__=='__main__':main()

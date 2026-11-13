"""Fixed all9 V23 release curves. Prepare/self-test does not read new outcomes.

Execute only after root confirms all9 terminal and the pinned readout complete.
No simulation/controller/model imports; no filtering, outcome selection or score.
"""
from pathlib import Path
from unittest.mock import patch
import argparse, hashlib, importlib.util, json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v23_small_plan_v1.json'
READOUT_CODE=ROOT/'review/read_v23_small_readout_v1.py'
VARIANTS=('v21','backend','joint')
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
LABELS={'v21':'原V21','backend':'仅数值后端替换','joint':'联合任务'}
COLORS={'v21':'#8B969D','backend':'#496F95','joint':'#B68A2F'}
STYLES={'v21':'-','backend':'--','joint':'-.'}
TITLES={'nominal':'名义场景','fragile_conflict':'脆弱限力冲突','friction_drop':'摩擦变化'}
EPS=1e-9

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b,name):
    if a!=b:raise ValueError(name+' differs')
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write_new(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def force_curves(time,skin):
    """Same per-digit absolute differences as frozen raw200Hz release TV."""
    t=np.asarray(time,float);f=np.asarray(skin,float)
    if t.ndim!=1 or f.shape!=(len(t),5) or not np.isfinite(t).all():
        return dict(status='unknown_shape_or_clock',time=np.empty(0),total=np.empty(0),tv=np.empty(0),final_TV_N=None)
    mask=(t>=12.-EPS)&(t<=16.+EPS);x=t[mask];y=f[mask]
    expected=12.+np.arange(801)*.005
    if x.shape!=(801,) or np.max(abs(x-expected),initial=0.)>EPS:
        return dict(status='unknown_fixed_window_coverage',time=x,total=np.full(len(x),np.nan),tv=np.full(len(x),np.nan),final_TV_N=None)
    valid=np.isfinite(y).all(axis=1)
    total=np.full(len(x),np.nan);total[valid]=y[valid].sum(axis=1)
    tv=np.full(len(x),np.nan)
    # Unknown never becomes zero and no difference bridges a missing sample.
    first_bad=np.flatnonzero(~valid);stop=int(first_bad[0]) if len(first_bad) else len(x)
    if stop:
        tv[0]=0.
        if stop>1:tv[1:stop]=np.cumsum(np.abs(np.diff(y[:stop],axis=0)).sum(axis=1))
    return dict(status='complete_raw200Hz' if valid.all() else 'unknown_force_samples',
        time=x,total=total,tv=tv,final_TV_N=float(tv[-1]) if np.isfinite(tv[-1]) else None)

def angular_curve(start,end,state_valid,qvel,period):
    """Offline object free-joint angular-speed norm, integration END clock."""
    a=np.asarray(start,float);b=np.asarray(end,float);valid=np.asarray(state_valid);v=np.asarray(qvel,float)
    if a.ndim!=1 or b.shape!=a.shape or valid.shape!=a.shape or valid.dtype.kind!='b' or v.shape!=(len(a),6):
        return dict(status='unknown_ledger_shape',time=np.empty(0),angular=np.empty(0))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not math.isfinite(period) or period<=0:
        return dict(status='unknown_ledger_clock',time=np.empty(0),angular=np.empty(0))
    mask=(b>=12.-EPS)&(b<=16.+EPS);x=b[mask];vel=v[mask];vv=valid[mask]
    overlaps=(b>12.+EPS)&(a<16.-EPS)
    aa=a[overlaps];bb=b[overlaps]
    complete=bool(len(aa) and abs(aa[0]-12.)<=EPS and abs(bb[-1]-16.)<=EPS
        and np.all(bb>aa) and np.max(abs((bb-aa)-period),initial=0.)<=EPS
        and np.max(abs(aa[1:]-bb[:-1]),initial=0.)<=EPS)
    y=np.full(len(x),np.nan);known=vv&np.isfinite(vel).all(axis=1)
    y[known]=np.linalg.norm(vel[known,3:],axis=1)
    if not complete:y[:]=np.nan
    return dict(status='complete_END_state' if complete and known.all() else 'unknown_state_or_coverage',time=x,angular=y)

def ready(plan,report,markers):
    """No curve source may be opened until every fixed assignment is terminal."""
    keys={(v,*c) for v in VARIANTS for c in CONTEXTS}
    actual=[(s['variant'],*s['context_key']) for s in plan.get('stages',[])]
    rows=[(r['variant'],*r['context_key']) for r in report.get('records',[])]
    if len(actual)!=9 or len(set(actual))!=9 or set(actual)!=keys:raise ValueError('Exact planned3x3 roster required')
    if report.get('complete') is not True or report.get('issues')!=[] or report.get('assigned')!=9:
        raise ValueError('Pinned full readout complete with no issues required')
    if len(rows)!=9 or len(set(rows))!=9 or set(rows)!=keys:raise ValueError('All9 readout records including failures required')
    for s in plan['stages']:
        m=markers.get(s['name'])
        if not m or not (m.get('execution_complete') is True and m.get('sources_unchanged') is True
            and m.get('numerical_unchanged') is True and m.get('runtime_failures')==[] and m.get('planned')==m.get('trials')==1):
            raise ValueError('All9 genuinely terminal before curve/result reads')

def load_complete(readout_path):
    plan=read(PLAN);report=read(readout_path)
    markers={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in plan['stages']
        if (ROOT/s['output']/'execution_complete.json').is_file()}
    ready(plan,report,markers)
    eq(report['plan_sha256'],sha(PLAN),'readout plan identity')
    eq(report['reader_sha256'],sha(READOUT_CODE),'actual readout code identity')
    eq(plan['dependencies']['review/read_v23_small_readout_v1.py'],sha(READOUT_CODE),'registered readout code')
    for path,digest in plan['dependencies'].items():eq(sha(ROOT/path),digest,'frozen definition '+path)
    rows={(r['variant'],*r['context_key']):r for r in report['records']};inputs=[]
    # Verify ALL input hashes/context before loading the first compressed array.
    for stage in plan['stages']:
        key=(stage['variant'],*stage['context_key']);r=rows[key]
        run=ROOT/stage['output']/'runs'/f'{key[1]}__ratio_effort__{key[3]}'
        eq(Path(r['run']).resolve(),run.resolve(),'assigned run')
        for name in ('trace.npz','delivery_intervals.npz','metadata.json','result.json'):
            eq(sha(run/name),r['input_sha256'][name],'readout input '+name)
        meta=read(run/'metadata.json');scene=meta['scenario'];scene=scene['name'] if isinstance(scene,dict) else scene
        eq((meta['release_variant'],scene,meta['motion_mode'],meta['seed']),key,'actual context')
        eq(meta['research_source_sha256'],plan['source_sha256'],'actual186 source')
        eq(meta['numerical_identity'],plan['numerical_identity'],'actual numerical identity')
        inputs.append((stage,r,run,meta))
    data=[]
    for stage,row,run,meta in inputs:
        with np.load(run/'trace.npz',allow_pickle=False) as z:
            time=z['time'];skin=z['skin_normal']
        force=force_curves(time,skin)
        with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:
            start=z['interval_start_s'];end=z['interval_end_s'];valid=z['object_state_valid'];v=z['object_free_qvel6_end']
        angular=angular_curve(start,end,valid,v,float(meta['physics_timestep']))
        frozen=row['release_TV']['all_actor_skin_TV_N'];drawn=force['final_TV_N']
        if drawn is not None and frozen is not None:
            tol=64*np.finfo(float).eps*max(1.,abs(drawn),abs(frozen))
            if abs(drawn-frozen)>tol:raise ValueError('Cumulative final TV differs from frozen raw200Hz TV')
        data.append(dict(variant=stage['variant'],context_key=stage['context_key'],force=force,angular=angular,
            original_task_success=row['original_result']['task_success'],frozen_final_skin_TV_N=frozen,
            input_sha256=row['input_sha256'],run=str(run)))
    return plan,report,data,{s['name']:sha(ROOT/s['output']/'execution_complete.json') for s in plan['stages']}

def execute(readout_path,outdir):
    if outdir.exists():raise FileExistsError('Fresh figure directory required')
    plan,report,data,markers=load_complete(readout_path)
    # Matplotlib is intentionally imported AFTER all barriers and input checks.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'svg.fonttype':'none','svg.hashsalt':'v23-fixed-scientific-v1',
        'font.size':10,'axes.labelsize':10,'axes.titlesize':12,'legend.fontsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    outdir.mkdir(parents=True,exist_ok=False)
    index={(r['variant'],*r['context_key']):r for r in data}
    maxima=[]
    for kind,field in [('force','total'),('force','tv'),('angular','angular')]:
        finite=[r[kind][field][np.isfinite(r[kind][field])] for r in data]
        maxima.append(max([float(np.max(a)) for a in finite if len(a)]+[0.]))
    plots=[];saved_arrays={};series=[]
    for ci,key in enumerate(CONTEXTS):
        fig,axes=plt.subplots(3,1,figsize=(11.6,8.5),sharex=True,gridspec_kw={'hspace':.23})
        for variant in VARIANTS:
            r=index[(variant,*key)];style=dict(color=COLORS[variant],linestyle=STYLES[variant],linewidth=1.45,label=LABELS[variant])
            force=r['force'];angular=r['angular'];prefix=f'c{ci}_{variant}'
            axes[0].plot(force['time'],force['total'],**style)
            axes[1].plot(force['time'],force['tv'],**style)
            axes[2].plot(angular['time'],angular['angular'],**style)
            for name,a in [('force_time',force['time']),('five_finger_skin_total_N',force['total']),('cumulative_per_finger_TV_N',force['tv']),('object_END_time',angular['time']),('object_angular_speed_rad_s',angular['angular'])]:saved_arrays[prefix+'_'+name]=a
            series.append(dict(variant=variant,context_key=list(key),original_task_success=r['original_task_success'],
                force_status=force['status'],angular_status=angular['status'],final_TV_N=force['final_TV_N'],
                frozen_final_skin_TV_N=r['frozen_final_skin_TV_N'],run=r['run'],input_sha256=r['input_sha256']))
            for ax,kind in [(axes[0],'force'),(axes[2],'angular')]:
                if r[kind]['status'].startswith('unknown'):
                    offset=VARIANTS.index(variant)
                    ax.text(.01,.94-.08*offset,LABELS[variant]+'：存在缺测/覆盖未知',transform=ax.transAxes,color=COLORS[variant],va='top')
        for i,ax in enumerate(axes):
            ax.set_xlim(12.,16.);ax.set_xticks(np.arange(12.,16.01,.5));ax.set_ylim(0.,1.06*maxima[i] if maxima[i]>0 else 1.)
            ax.grid(axis='y',color='#DDE1E3',linewidth=.6);ax.tick_params(labelsize=9)
        axes[0].set_ylabel('五指接触法向力之和 / N')
        axes[1].set_ylabel('累计逐指总变差 / N')
        axes[2].set_ylabel('物体角速度模 / (rad/s)');axes[2].set_xlabel('记录时间 / s')
        motion='静态持握' if key[1]=='hold' else '运输'
        fig.suptitle(f'{TITLES[key[0]]} · {motion} · 物理种子 {key[2]}',x=.11,y=.98,ha='left',fontsize=14)
        axes[0].legend(loc='upper right',ncol=3,frameon=False)
        fig.text(.11,.025,'完整12–16秒；五指全部外接触，不含掌部。200 Hz未滤波；TV含正常卸力。\n角速度来自独立离线物体状态账本（积分END时钟），未作为控制输入；缺测不补0。',fontsize=9,ha='left')
        fig.subplots_adjust(left=.11,right=.98,top=.92,bottom=.13)
        name=f'{ci+1:02d}_{key[0]}_{key[1]}_{key[2]}'
        for ext in ('png','svg'):
            dest=outdir/(name+'.'+ext)
            fig.savefig(dest,dpi=180 if ext=='png' else None,metadata={'Date':None} if ext=='svg' else None)
            plots.append(dict(path=dest.name,sha256=sha(dest),context_key=list(key)))
        plt.close(fig)
    np.savez_compressed(outdir/'plotted_series.npz',**saved_arrays)
    manifest=dict(schema='v23_fixed_release_scientific_curves_v1',complete=True,assigned=9,all_success_filter_applied=False,
        window_s=[12.,16.],force_cadence_Hz=200,filtering='none',force_scope='sum of five fingers all-actor skin normal, excludes palm',
        TV_definition='C(t_k)=sum_d sum_{j=1..k}|skin_normal_d(t_j)-skin_normal_d(t_{j-1})|, includes intentional unloading; unitN notN/s',
        angular_scope='offline object free-joint qvel6_end[3:] norm; integration END clock; not controller observable/input',
        figure_limits_common_across_contexts=True,styles={v:dict(label=LABELS[v],color=COLORS[v],linestyle=STYLES[v]) for v in VARIANTS},
        plan_sha256=sha(PLAN),readout_sha256=sha(readout_path),plotter_sha256=sha(__file__),
        completion_marker_sha256=markers,plotted_series_sha256=sha(outdir/'plotted_series.npz'),series=series,plots=plots,
        visual_export_inspected=False,physics_steps=0,new_scores_or_thresholds=False,
        scope='Three seen development contexts/two physical seed clusters; curve comparison is not independent confirmation or a causal decomposition.')
    write_new(outdir/'figure_manifest.json',manifest)
    lines=['# V23三臂完整释放窗口时序图','',
        '按固定三条件逐条对照，保留原任务失败及不可评记录，不按清离或成功筛选。三种颜色只表示原V21、仅数值后端替换和联合任务实现。',
        '', '每图依次为五指接触法向力之和、累计逐指总变差、离线物体角速度模；均保留12–16秒完整窗口。五指曲线不含掌部，累计TV包含正常卸力，并非振荡评分。角速度来自物理积分END账本，仅为评价真值，控制器不可读取。', '']
    for key in CONTEXTS:
        p=next(p for p in plots if p['context_key']==list(key) and p['path'].endswith('.png'))
        lines += [f"## {TITLES[key[0]]} / {key[1]} / {key[2]}",'',f"![三臂完整释放曲线]({p['path']})",'',f"[矢量图]({Path(p['path']).with_suffix('.svg').name})",'']
    lines += ['未增加评分、阈值或成功定义；图不能替代原全程力限、交付辅助指标、覆盖/资格和实际执行独审。没有把不同场景或重复时间点当独立样本。','',
        '[完整来源、曲线定义与缺测状态](figure_manifest.json)。PNG/SVG尚须实际导出后人工目视核验；没有宣称已完成该步骤。']
    (outdir/'figures.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(outdir=str(outdir),figures=3,files=6,manifest_sha256=sha(outdir/'figure_manifest.json')),ensure_ascii=False))

def selftest(out):
    if out.exists():raise FileExistsError(out)
    t=12.+np.arange(801)*.005;z=np.zeros((801,5));f=z.copy();f[400:,0]=2.
    cancellation=z.copy();cancellation[400:,0]=1.;cancellation[:400,1]=1.
    checks=[]
    def ck(n,x):checks.append(dict(name=n,passed=bool(x)))
    ck('known constant total and TV zero',force_curves(t,z)['final_TV_N']==0.)
    ck('single digit step agrees raw TV',force_curves(t,f)['final_TV_N']==2.)
    r=force_curves(t,cancellation);ck('per-digit TV retained when total force cancels',r['final_TV_N']==2. and np.all(r['total']==1.))
    bad=f.copy();bad[400,0]=np.nan;r=force_curves(t,bad)
    ck('unknown pressure not zero or bridged in cumulative TV',r['final_TV_N'] is None and np.isnan(r['tv'][400:]).all())
    ck('truncated fixed window unavailable',force_curves(t[:-1],f[:-1])['final_TV_N'] is None)
    stages=[dict(name=f'{v}_{i}',variant=v,context_key=list(c)) for v in VARIANTS for i,c in enumerate(CONTEXTS)]
    report=dict(complete=True,issues=[],assigned=9,records=[dict(variant=s['variant'],context_key=s['context_key']) for s in stages])
    good=dict(execution_complete=True,sources_unchanged=True,numerical_unchanged=True,runtime_failures=[],planned=1,trials=1)
    marks={s['name']:dict(good) for s in stages};marks.pop(stages[-1]['name'])
    try:ready(dict(stages=stages),report,marks);blocked=False
    except ValueError:blocked=True
    ck('one missing terminal blocks all curve reads',blocked)
    e=12.+np.arange(1,8001)*.0005;a=e-.0005;v=np.zeros((8000,6));v[:,5]=2.
    q=angular_curve(a,e,np.ones(8000,bool),v,.0005)
    ck('angular END-state norm and clock independent of contact',q['status']=='complete_END_state' and np.all(q['angular']==2.))
    vv=np.ones(8000,bool);vv[10]=False;q=angular_curve(a,e,vv,v,.0005)
    ck('unknown object state is a visible gap',np.isnan(q['angular'][10]))
    write_new(out,dict(schema='v23_plot_preparation_software_v1',passed=all(x['passed'] for x in checks),checks=checks,
        plotter_sha256=sha(__file__),new_outcomes_read=False,NPZ_reads=0,physics_steps=0,rendered=False))
    print(json.dumps(dict(passed=all(x['passed'] for x in checks),checks=len(checks),out=str(out),sha256=sha(out))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute',action='store_true');group.add_argument('--self-test',action='store_true')
    ap.add_argument('--readout',type=Path);ap.add_argument('--outdir',type=Path);ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.self_test:
        if args.out is None:raise ValueError('--out required for self-test')
        selftest(args.out)
    else:
        if args.readout is None or args.outdir is None:raise ValueError('--readout and fresh --outdir required')
        execute(args.readout,args.outdir)

"""Frozen-record video production. No controller and no physics integration."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco

PROJECT=Path(__file__).resolve().parents[2] / 'projects' / '多任务联合控制研究_2026-11-03'
REPORT=Path(__file__).resolve().parent / 'original_report_not_included'
BATCH=PROJECT/'output/midterm_core_macro_ablation_v1_verified_20261111'
OUT=REPORT/'模拟动画对照'
WORK=REPORT/'.制作与校验/动画制作_2026-11-12'
for d in (OUT,WORK,OUT/'封面'):d.mkdir(parents=True,exist_ok=True)
SCENES={'nominal':'正常运输','friction_drop':'摩擦变化','fragile_conflict':'易碎限力','collision':'碰撞扰动'}
FILENAMES={s:f'{i:02d}_{name}_基线与宏观版.mp4' for i,(s,name) in enumerate(SCENES.items(),1)}
W,H,FPS=1920,1080,30
PW,PH=916,460
CAMERA=dict(azimuth=145.,elevation=-18.,distance=1.10,lookat=(.16,0.,.24))
COLORS={'pi':'#2767A8','core':'#128376'}
INK='#142636';MUTED='#566775';GRID='#DCE4EB';BG='#F5F7FA'
FONTS={}
def font(size,bold=False):
    key=(size,bold)
    if key not in FONTS:FONTS[key]=ImageFont.truetype(r'C:\Windows\Fonts\msyhbd.ttc' if bold else r'C:\Windows\Fonts\msyh.ttc',size)
    return FONTS[key]
def text(d,xy,s,size=24,fill=INK,bold=False,anchor=None):
    d.text(xy,str(s),font=font(size,bold),fill=fill,anchor=anchor)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
PLAN=read(BATCH/'plan.json');LEDGER=read(BATCH/'assignment_ledger.json')
assert read(BATCH/'execution_complete.json')['execution_complete']
CORE_METRICS={r['job']['scene']:r['metrics'] for r in LEDGER if r['job']['arm']=='core'}
BLOCKED_CALLS={}
def disallow(name):
    BLOCKED_CALLS[name]=0
    def f(*a,**k):
        BLOCKED_CALLS[name]+=1
        raise RuntimeError('Forbidden during archived visual playback: '+name)
    return f
for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_collision','mj_contactForce'):
    if hasattr(mujoco,n):setattr(mujoco,n,disallow(n))
PRESERVED={}
for p,h in PLAN['preserved_sources'].items():
    assert sha(p)==h,'Preserved source changed: '+p
    PRESERVED[p]=h
for p in (BATCH/'plan.json',BATCH/'assignment_ledger.json',BATCH/'execution_complete.json',PROJECT/'midterm_core_v1/验证汇总.json'):
    PRESERVED[str(p)]=sha(p)

def load_snapshot(folder,metadata):
    batch=folder.parent.parent
    source=metadata['research_source_sha256']
    for name,h in source.items():
        if name.startswith('foundation/'):
            p=batch/'sources'/name
            assert p.is_file() and sha(p)==h, str(p)
            PRESERVED[str(p)]=h
    base=batch/'sources/foundation'
    suffix=hashlib.sha256(str(batch).encode()).hexdigest()[:12]
    def load(name,path):
        if name in sys.modules:return sys.modules[name]
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m;spec.loader.exec_module(m);return m
    cfg=load('_demo_config_'+suffix,base/'foundation_config.py')
    prev=sys.modules.get('foundation_config');sys.modules['foundation_config']=cfg
    try:plant=load('_demo_plant_'+suffix,base/'hand_plant.py')
    finally:
        if prev is None:sys.modules.pop('foundation_config',None)
        else:sys.modules['foundation_config']=prev
    model,identity=plant._model(cfg.FoundationConfig(**metadata['config']))
    assert identity==metadata['model_sha256'],'Visual model identity mismatch'
    return model

def record(scene,role):
    f=Path(PLAN['references'][scene]['pi']['path']) if role=='pi' else BATCH/'runs'/f'core__{scene}__619741'
    m=read(f/'metadata.json');metric=PLAN['references'][scene]['pi']['metrics'] if role=='pi' else CORE_METRICS[scene]
    expected=PLAN['references'][scene]['pi']['hashes'] if role=='pi' else metric['input_sha256']
    for name,h in expected.items():assert sha(f/name)==h,f'{scene}/{role}/{name}'
    for p in f.iterdir():
        if p.is_file():PRESERVED[str(p)]=sha(p)
    assert m['scenario']['name']==scene and m['seed']==619741 and m['motion_mode']=='transport'
    if role=='core':assert m.get('experiment_arm')=='core'
    else:assert m['task_policy']=='legacy' and m['method']=='ratio_effort'
    assert m['trace_sha256']==sha(f/'trace.npz')
    with np.load(f/'trace.npz',allow_pickle=False) as z:
        a={k:z[k] for k in ['time','qpos','qvel','ctrl','normal','position','phase']}
    assert a['time'].shape==(3201,) and np.allclose(a['time'],np.arange(3201)*.005,atol=1e-8,rtol=0)
    for k in ['qpos','qvel','ctrl','normal','position']:assert np.isfinite(a[k]).all()
    with np.load(f/'delivery_intervals.npz',allow_pickle=False) as z:
        a['receiver_t']=z['interval_end_s'];a['receiver_force']=z['target_receiver_top_normal_N']
        assert z['contact_valid'].all() and len(a['receiver_t'])==32000
    assert np.isfinite(a['receiver_force']).all()
    a['finger_total']=a['normal'].sum(axis=1)
    assert a['finger_total'].max()<25 and a['receiver_force'].max()<10,'Would clip shared plot scale'
    model=load_snapshot(f,m)
    assert a['qpos'].shape[1]==model.nq and a['qvel'].shape[1]==model.nv and a['ctrl'].shape[1]==model.nu
    for value in a.values():value.flags.writeable=False
    return dict(scene=scene,role=role,folder=f,meta=m,metric=metric,arrays=a,model=model)

RECORDS={s:{r:record(s,r) for r in ['pi','core']} for s in SCENES}
for pair in RECORDS.values():
    assert pair['pi']['meta']['config']==pair['core']['meta']['config']
    assert pair['pi']['meta']['model_sha256']==pair['core']['meta']['model_sha256']
    assert np.array_equal(pair['pi']['arrays']['time'],pair['core']['arrays']['time'])
dump(WORK/'输入与保护清单.json',dict(files=PRESERVED,camera=CAMERA,scenes=list(SCENES),seed=619741,
    left='frozen literature-inspired force-ratio PI',right='midterm_core_v1/core',new_physics_steps=0,
    controls='Physics/contact solver calls are blocked; only geometry transforms are refreshed'))
print('INPUTS VERIFIED: eight records; frozen state, force, model and source bindings passed',flush=True)

class Visuals:
    def __init__(self,pair):
        self.pair=pair;self.renderers={};self.data={}
        for role,r in pair.items():
            # Each side has separate MjData, even when the immutable model is shared.
            model=r['model']
            model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),PW)
            model.vis.global_.offheight=max(int(model.vis.global_.offheight),PH)
            self.data[role]=mujoco.MjData(model)
            self.renderers[role]=mujoco.Renderer(model,width=PW,height=PH)
        self.camera=mujoco.MjvCamera()
        self.camera.lookat[:]=CAMERA['lookat']
        for k in ['azimuth','elevation','distance']:setattr(self.camera,k,CAMERA[k])
        self.opt=mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False
        self.body_errors=[]
    def draw(self,role,index):
        r=self.pair[role];a=r['arrays'];model=r['model'];data=self.data[role]
        data.qpos[:]=a['qpos'][index];data.qvel[:]=a['qvel'][index];data.ctrl[:]=a['ctrl'][index];data.time=float(a['time'][index])
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        mujoco.mj_camlight(model,data);mujoco.mj_tendon(model,data)
        assert np.array_equal(data.qpos,a['qpos'][index]),'Rendering altered saved position'
        err=float(np.max(np.abs(data.xpos[model.body('object').id]-a['position'][index])))
        self.body_errors.append(err);assert err<1e-10,'Visible object differs from saved object'
        ren=self.renderers[role];ren.update_scene(data,camera=self.camera,scene_option=self.opt)
        return Image.fromarray(ren.render())
    def close(self):
        for r in self.renderers.values():r.close()

def nearest(t):
    return int(np.clip(np.floor(float(t)/.005+.5),0,3200))
def current_stage(t):
    return '共同准备' if t<7.5 else ('策略持握与运输' if t<12 else '放下与交接')
def status(metric):
    o=metric['original']
    return [('任务完成','是' if o['full_motion_mission_success'] else '否',bool(o['full_motion_mission_success'])),
            ('全程限力','通过' if o['force_guard_pass'] else '未通过',bool(o['force_guard_pass'])),
            ('物体触地','有' if metric['ever_target_floor_contact'] else '无',not metric['ever_target_floor_contact'])]
def chart_base(scene,start):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    rects=[]
    for j,(key,ymax,title,subtitle) in enumerate([
        ('finger_total',25.,'五指对物体合力','原始 200 Hz 记录'),
        ('receiver_force',10.,'物体与接收面接触力','原始 2 kHz 记录')]):
        ox=32+j*940
        text(d,(ox,730),title,27,bold=True)
        text(d,(ox,766),subtitle,18,MUTED)
        for l,(role,label) in enumerate([('pi','基线'),('core','宏观版')]):
            lx=ox+555+l*160;d.line((lx,750,lx+32,750),fill=COLORS[role],width=5)
            text(d,(lx+42,733),label,23,COLORS[role])
        x0,y0,x1,y1=ox+65,810,ox+887,982
        d.rectangle((x0,y0,x1,y1),fill='white',outline=GRID,width=1)
        for value in np.linspace(0,ymax,6):
            yy=y1-(y1-y0)*value/ymax;d.line((x0,yy,x1,yy),fill=GRID,width=1)
            text(d,(x0-12,yy),f'{value:g}',18,MUTED,anchor='rm')
        for t in np.linspace(start,16,5):
            xx=x0+(x1-x0)*(t-start)/(16-start);d.line((xx,y0,xx,y1),fill=GRID,width=1)
            text(d,(xx,y1+9),f'{t:g}',18,MUTED,anchor='mt')
        text(d,(x0-40,y0-22),'N',19,MUTED)
        text(d,(x1,y1+9),'',18,MUTED)
        for role,r in RECORDS[scene].items():
            a=r['arrays'];tt=a['time'] if key=='finger_total' else a['receiver_t'];values=a[key]
            mask=(tt>=start-1e-9)&(tt<=16+1e-9)
            pts=np.column_stack((x0+(tt[mask]-start)/(16-start)*(x1-x0),y1-values[mask]/ymax*(y1-y0)))
            d.line([tuple(p) for p in pts],fill=COLORS[role],width=2)
        # The plots always show the full saved window; only the replay cursor moves.
        rects.append((key,x0,y0,x1,y1,ymax))
    return im,rects

def frame(scene,vis,t,mode,background,rects):
    im=background.copy();d=ImageDraw.Draw(im);i=nearest(t);actual=float(RECORDS[scene]['pi']['arrays']['time'][i])
    name=SCENES[scene]
    text(d,(32,20),'五指多任务融合控制  '+name,40,bold=True)
    text(d,(32,78),'同条件归档回放  ·  左侧力比 PI 基线 / 右侧中期宏观版  ·  同一固定视角',23,MUTED)
    text(d,(1888,24),f'实验时刻  {actual:05.2f} s',34,bold=True,anchor='rt')
    tag={'full':'完整过程  1×','slow':'交接重播  0.25×','end':'本场景结束画面'}[mode]
    text(d,(1888,77),tag+'  |  '+current_stage(actual),23,MUTED,anchor='rt')
    for j,role in enumerate(['pi','core']):
        x=32+j*940;r=RECORDS[scene][role];metric=r['metric'];col=COLORS[role]
        text(d,(x,121),'力比 PI 基线' if role=='pi' else '中期宏观策略',30,col,True)
        touch=metric['first_target_top_contact_s']
        text(d,(x+PW,129),f'触台时刻  {touch:.3f} s',21,MUTED,anchor='rt')
        im.paste(vis.draw(role,i),(x,166));d=ImageDraw.Draw(im)
        d.rectangle((x,166,x+PW-1,166+PH-1),outline='#D6E0E8',width=1)
        peak=metric['first_target_top_impact20ms_peak_N'];rms=metric['touch_200_to_500ms_rms']['value_N']
        latency=metric['original']['clearance_latency_s']
        line=f'首峰 {peak:.2f} N    后段波动 {rms:.3f} N    清离 {latency:.2f} s' if latency is not None else f'首峰 {peak:.2f} N    后段波动 {rms:.3f} N    清离未完成'
        text(d,(x,640),line,23,col)
        text(d,(x,677),'全程结果',19,MUTED)
        sx=x+102
        for label,value,good in status(metric):
            text(d,(sx,675),f'{label}：{value}',22,INK if good else '#AD3D33')
            sx+=255
    start=0. if mode=='full' else 12.
    for key,x0,y0,x1,y1,ymax in rects:
        xx=x0+(actual-start)/(16-start)*(x1-x0)
        d.line((xx,y0,xx,y1),fill='#243747',width=2)
        for role,r in RECORDS[scene].items():
            a=r['arrays']
            if key=='finger_total':value=float(a[key][i])
            else:
                n=int(np.clip(np.searchsorted(a['receiver_t'],actual,side='right')-1,0,len(a[key])-1));value=float(a[key][n])
            yy=y1-value/ymax*(y1-y0)
            d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=COLORS[role],outline='white',width=1)
    text(d,(32,1027),'横轴：实验时间（s）  ·  曲线显示完整记录，竖线指示当前时刻；原始数据未平滑',20,MUTED)
    text(d,(1888,1027),'首峰：触台后 20 ms  |  后段波动：触台后 200–500 ms',20,MUTED,anchor='rt')
    text(d,(32,1053),'任务完成不等于稳定放置或全程不过力；五指合力不能代替单指限力检查。',18,MUTED)
    text(d,(1888,1053),'四场景 / 同一已见种子 619741 · 小规模开发证据',18,MUTED,anchor='rt')
    return im,actual,i

def preview():
    results=[]
    for scene,times in [('nominal',[8.5,12.35,14.2,16.]),('friction_drop',[16.]),('collision',[9.6])]:
        vis=Visuals(RECORDS[scene])
        try:
            for t in times:
                mode='full' if t<12 else 'slow';bg,rects=chart_base(scene,0 if mode=='full' else 12)
                im,actual,i=frame(scene,vis,t,mode,bg,rects)
                path=WORK/f'preview_{scene}_{t:.2f}.png';im.save(path)
                results.append(str(path))
        finally:vis.close()
    dump(WORK/'预览记录.json',dict(images=results,blocked_calls=BLOCKED_CALLS))
    print(json.dumps({'previews':results,'blocked_calls':BLOCKED_CALLS},ensure_ascii=False),flush=True)

def timeline():
    # Both sides share these exact archived indices. No temporal event realignment.
    for k in range(481):yield k/30.,'full'
    for k in range(481):yield 12+k/120.,'slow'
    for k in range(90):yield 16.,'end'

def produce(scene):
    ff=shutil.which('ffmpeg');assert ff
    dest=OUT/FILENAMES[scene];temporary=WORK/(scene+'_encoding.mp4');logfile=WORK/(scene+'_ffmpeg.log')
    command=[ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',str(temporary)]
    vis=Visuals(RECORDS[scene]);bgs={k:chart_base(scene,s) for k,s in [('full',0.),('slow',12.),('end',12.)]}
    mappings=[];start_wall=time.time()
    with logfile.open('wb') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            last_image=None
            for n,(t,mode) in enumerate(timeline()):
                if mode=='end' and last_image is not None and n>962:im=last_image;actual=16.;i=3200
                else:
                    im,actual,i=frame(scene,vis,t,mode,*bgs[mode])
                    if mode=='end':last_image=im
                if n in [0,255,370,440,480,630,800,961,962]:
                    im.save(WORK/f'{scene}_frame_{n:04d}.jpg',quality=93)
                if n==800:im.save(OUT/'封面'/(scene+'.jpg'),quality=93)
                proc.stdin.write(im.tobytes())
                mappings.append(dict(frame=n,video_time_s=n/FPS,segment=mode,requested_sim_time_s=t,recorded_sim_time_s=actual,left_index=i,right_index=i))
                if n%90==0:print(f'{scene}: {n}/1052 frames; {time.time()-start_wall:.1f}s elapsed',flush=True)
            proc.stdin.close();rc=proc.wait(timeout=120)
            if rc:raise RuntimeError(logfile.read_text(encoding='utf-8',errors='replace'))
        finally:
            if proc.poll() is None:proc.kill()
            vis.close()
    assert len(mappings)==1052 and max(vis.body_errors)<1e-10 and not any(BLOCKED_CALLS.values())
    temporary.replace(dest)
    dump(WORK/(scene+'_frame_mapping.json'),mappings)
    result=dict(scene=scene,title=SCENES[scene],path=str(dest),frames=len(mappings),fps=FPS,duration_s=len(mappings)/FPS,
        camera=CAMERA,frame_size=[W,H],no_physics_or_contact_solve=not any(BLOCKED_CALLS.values()),
        visual_object_max_position_error_m=max(vis.body_errors),time_pairing='same recorded timestamp and index; no event shift',
        segments=[dict(name='full',simulation_seconds=[0,16],speed=1),dict(name='slow',simulation_seconds=[12,16],speed=.25),dict(name='end',duration_s=3)],
        sha256=sha(dest),bytes=dest.stat().st_size,wall_seconds=time.time()-start_wall,
        methods={role:dict(folder=str(r['folder']),trace_sha256=r['meta']['trace_sha256'],metrics=r['metric']) for role,r in RECORDS[scene].items()})
    dump(WORK/(scene+'_export.json'),result)
    print('FINISHED '+scene+' '+str(dest.stat().st_size)+' bytes',flush=True)
    return result

def finish():
    results=[read(WORK/(s+'_export.json')) for s in SCENES]
    assert all(sha(p)==h for p,h in PRESERVED.items()),'Source files changed'
    dump(WORK/'最终导出核对.json',dict(status='passed',outputs=results,protected_files=len(PRESERVED),
        source_files_unchanged=True,blocked_physics_calls=BLOCKED_CALLS,controllers_loaded=False,
        plots='200 Hz saved five-finger target force sum and 2 kHz saved target-receiver-top normal force; no smoothing',
        axes=dict(finger_force_N=[0,25],receiver_force_N=[0,10]),all_scenes_included=True))
    print('ALL FOUR FINISHED AND ORIGINALS VERIFIED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--scene',choices=list(SCENES));parser.add_argument('--all',action='store_true');parser.add_argument('--verify-final',action='store_true')
    args=parser.parse_args()
    if args.preview:preview()
    elif args.scene:produce(args.scene)
    elif args.all:
        for s in SCENES:produce(s)
        finish()
    elif args.verify_final:finish()

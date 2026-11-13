"""Render saved cold-start trajectories. Never execute a controller or step physics."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, shutil, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco

sys.dont_write_bytecode=True
REPORT=Path(__file__).resolve().parent / 'original_report_not_included'
PROJECT=REPORT.parent/'多任务联合控制研究_2026-11-03'
FOUNDATION=REPORT.parent/'五指策略接入_2026-11-03'
BATCH=PROJECT/'output/cold_start_take_pilot_v1_20261112'
WORK=Path(__file__).resolve().parent
OUT=WORK/'待发布'
for p in (OUT,OUT/'封面',WORK/'画面检查'):p.mkdir(parents=True,exist_ok=True)
SCENES={'nominal':'正常运输','friction_drop':'摩擦变化','fragile_conflict':'易碎限力','collision':'碰撞扰动'}
FILENAMES={s:f'{i:02d}_{name}_基线与宏观版.mp4' for i,(s,name) in enumerate(SCENES.items(),1)}
W,H,FPS=1920,1080,30
PW,PH=916,460
CAMERA=dict(azimuth=145.,elevation=-18.,distance=1.25,lookat=(.19,0.,.22))
COLORS={'baseline':'#2767A8','macro':'#128376'}
INK='#142636';MUTED='#566775';GRID='#DCE4EB';BG='#F5F7FA'
FONTS={}

def font(size,bold=False,latin=False):
    key=(size,bold,latin)
    if key not in FONTS:
        name=('timesbd.ttf' if bold else 'times.ttf') if latin else ('msyhbd.ttc' if bold else 'msyh.ttc')
        FONTS[key]=ImageFont.truetype(str(Path(r'C:\Windows\Fonts')/name),size)
    return FONTS[key]

def text(d,xy,s,size=24,fill=INK,bold=False,anchor=None):
    # Consistent Chinese UI lettering; explicit Times New Roman Latin/numbers.
    import re
    pieces=[(x,font(size,bold,x.isascii())) for x in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',str(s))]
    widths=[d.textlength(x,font=f) for x,f in pieces]
    x,y=xy;anchor=anchor or 'lt'
    if anchor[0]=='r':x-=sum(widths)
    elif anchor[0]=='m':x-=sum(widths)/2
    va=anchor[1]
    # Shared top line or shared baseline avoids platform fallback font changes.
    for (part,f),width in zip(pieces,widths):
        d.text((x,y),part,font=f,fill=fill,anchor='l'+va)
        x+=width

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
PLAN=read(BATCH/'plan.json')
ENDPOINT=read(BATCH/'terminal_delivery_addendum.json')
assert read(BATCH/'audit.json')['passed']
assert ENDPOINT['plan_sha256']==sha(BATCH/'plan.json')
BLOCKED_CALLS={}
def disallow(name):
    BLOCKED_CALLS[name]=0
    def f(*a,**k):
        BLOCKED_CALLS[name]+=1
        raise RuntimeError('Forbidden in saved-state playback: '+name)
    return f
for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_collision','mj_contactForce'):
    if hasattr(mujoco,n):setattr(mujoco,n,disallow(n))

PRESERVED={}
for name,expected in PLAN['sources_sha256'].items():
    p=FOUNDATION/name.removeprefix('foundation/') if name.startswith('foundation/') else PROJECT/name
    assert p.is_file() and sha(p)==expected,'Frozen source differs: '+str(p)
    PRESERVED[str(p)]=expected
for p in BATCH.rglob('*'):
    if p.is_file():PRESERVED[str(p)]=sha(p)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
CFG=load_module('_cold_demo_config',FOUNDATION/'foundation_config.py')
previous=sys.modules.get('foundation_config');sys.modules['foundation_config']=CFG
try:PLANT=load_module('_cold_demo_geometry',FOUNDATION/'hand_plant.py')
finally:
    if previous is None:sys.modules.pop('foundation_config',None)
    else:sys.modules['foundation_config']=previous
PLANT.HandPlant.__init__=disallow('HandPlant.__init__')
PLANT.CommonPreparation.__init__=disallow('CommonPreparation.__init__')
PLANT.CommonPreparation.step=disallow('CommonPreparation.step')

def record(scene,arm):
    folder=BATCH/'runs'/f'{arm}__{scene}__619741'
    meta=read(folder/'metadata.json');result=read(folder/'result.json')
    assert meta['schema']=='cold_start_take_trial_v1' and meta['experiment_arm']==arm
    assert meta['control_start_s']==0 and meta['common_preparation_used'] is False
    assert meta['seed']==619741 and meta['scenario']['name']==scene
    assert meta['trace_sha256']==sha(folder/'trace.npz')
    for name,h in result['artifacts_sha256'].items():assert sha(folder/name)==h
    assert meta['research_source_sha256']==PLAN['sources_sha256']
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        a={k:z[k] for k in ('time','qpos','qvel','ctrl','normal','position','phase')}
        a['force_time']=np.r_[0.,z['substep_time']+z['substep_duration']]
        a['finger_total']=np.vstack([z['normal'][0],z['substep_normal']]).sum(axis=1)
    assert np.allclose(a['time'],np.arange(3201)*.005,atol=1e-8,rtol=0)
    assert np.allclose(a['force_time'],np.arange(32001)*.0005,atol=1e-8,rtol=0)
    for k,v in a.items():
        if v.dtype.kind in 'fi':assert np.isfinite(v).all()
        v.flags.writeable=False
    assert 0<=a['finger_total'].min() and a['finger_total'].max()<25
    assert abs(a['finger_total'].max()-result['total_force']['full']['raw_total_peak_N'])<1e-10
    model,model_id=PLANT._model(CFG.FoundationConfig(**meta['config']))
    assert model_id==meta['model_sha256']
    assert a['qpos'].shape[1]==model.nq and a['qvel'].shape[1]==model.nv and a['ctrl'].shape[1]==model.nu
    return dict(folder=folder,meta=meta,result=result,arrays=a,model=model,endpoint=ENDPOINT['rows'][scene][arm])

RECORDS={s:{a:record(s,a) for a in ('baseline','macro')} for s in SCENES}
for pair in RECORDS.values():
    b,m=pair['baseline'],pair['macro']
    assert b['meta']['config']==m['meta']['config']
    assert np.array_equal(b['arrays']['time'],m['arrays']['time'])
    assert np.array_equal(b['arrays']['qpos'][0],m['arrays']['qpos'][0])
    assert np.max(abs(b['arrays']['qpos'][:1500]-m['arrays']['qpos'][:1500]))>1e-4
    assert np.max(abs(b['arrays']['finger_total'][:15000]-m['arrays']['finger_total'][:15000]))>1e-4
dump(WORK/'输入与保护清单.json',dict(files=PRESERVED,source_batch=str(BATCH),seed=619741,
    methods=['baseline','macro'],camera=CAMERA,states_Hz=200,raw_total_finger_force_Hz=2000,
    current_pickup_is_independent=True,physical_step_calls_blocked=True,
    force_source='initial normal plus all saved substep_normal sums; no smoothing or recomputation'))
print('CURRENT EIGHT TRIALS VERIFIED; independent pickup and immutable sources confirmed',flush=True)

class Visuals:
    def __init__(self,pair):
        self.pair=pair;self.data={};self.renderers={};self.body_errors=[]
        for arm,r in pair.items():
            model=r['model']
            model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),PW)
            model.vis.global_.offheight=max(int(model.vis.global_.offheight),PH)
            self.data[arm]=mujoco.MjData(model)
            self.renderers[arm]=mujoco.Renderer(model,width=PW,height=PH)
        self.camera=mujoco.MjvCamera();self.camera.lookat[:]=CAMERA['lookat']
        for key in ('azimuth','elevation','distance'):setattr(self.camera,key,CAMERA[key])
        self.option=mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False
    def draw(self,arm,index):
        r=self.pair[arm];a=r['arrays'];model=r['model'];data=self.data[arm]
        data.qpos[:]=a['qpos'][index];data.qvel[:]=a['qvel'][index];data.ctrl[:]=a['ctrl'][index]
        data.time=float(a['time'][index])
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        mujoco.mj_camlight(model,data);mujoco.mj_tendon(model,data)
        assert np.array_equal(data.qpos,a['qpos'][index])
        err=float(np.max(abs(data.xpos[model.body('object').id]-a['position'][index])))
        assert err<1e-10
        self.body_errors.append(err)
        renderer=self.renderers[arm]
        renderer.update_scene(data,camera=self.camera,scene_option=self.option)
        return Image.fromarray(renderer.render())
    def close(self):
        for renderer in self.renderers.values():renderer.close()

def nearest(t):return int(np.clip(np.floor(float(t)/.005+.5),0,3200))
def stage(t):
    if t<7.5:return '拿取与抬升'
    if t<9.3:return '持稳'
    if t<12:return '运输'
    return '放下与交接'

def chart_base(scene,start):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(32,726),'总指力变化',28,bold=True)
    text(d,(220,732),'五指对目标物的法向力之和 · 原始 2 kHz 记录',22,MUTED)
    for j,arm in enumerate(('baseline','macro')):
        lx=1520+j*190
        d.line((lx,746,lx+38,746),fill=COLORS[arm],width=4)
        text(d,(lx+48,732),'基线' if arm=='baseline' else '宏观版',23,COLORS[arm])
    x0,y0,x1,y1,ymax=96,799,1873,977,25.
    d.rectangle((x0,y0,x1,y1),fill='white',outline=GRID,width=1)
    for v in range(0,26,5):
        yy=y1-(y1-y0)*v/ymax;d.line((x0,yy,x1,yy),fill=GRID,width=1)
        text(d,(x0-13,yy),str(v),20,MUTED,anchor='rm')
    for t in np.linspace(start,16,9):
        xx=x0+(x1-x0)*(t-start)/(16-start)
        d.line((xx,y0,xx,y1),fill=GRID,width=1)
        text(d,(xx,y1+8),f'{t:g}',20,MUTED,anchor='mt')
    text(d,(x0-36,y0-24),'N',20,MUTED)
    markers=[(7.5,'拿取评价节点'),(9.3,'运输指令'),(12.,'放下指令')]
    for t,label in markers:
        if t<start:continue
        xx=x0+(x1-x0)*(t-start)/(16-start)
        for yy in range(y0,y1,12):d.line((xx,yy,xx,min(y1,yy+6)),fill='#8E9DA9',width=1)
        text(d,(xx,y0-27),label,19,MUTED,anchor='mt' if t>start else 'lt')
    for arm in ('baseline','macro'):
        a=RECORDS[scene][arm]['arrays'];tt=a['force_time'];vv=a['finger_total']
        keep=(tt>=start-1e-8)&(tt<=16+1e-8)
        pts=list(zip(x0+(tt[keep]-start)/(16-start)*(x1-x0),y1-vv[keep]/ymax*(y1-y0)))
        if arm=='macro':d.line(pts,fill='white',width=4)
        d.line(pts,fill='#80A8CF' if arm=='baseline' else COLORS[arm],width=2)
    return im,(x0,y0,x1,y1,ymax)

def frame(scene,vis,t,mode,background,rect):
    im=background.copy();d=ImageDraw.Draw(im);idx=nearest(t)
    actual=float(RECORDS[scene]['baseline']['arrays']['time'][idx])
    text(d,(32,20),'全流程独立控制  '+SCENES[scene],40,bold=True)
    text(d,(32,78),'左侧基线 / 右侧宏观版 · 从 0 秒各自控制 · 同视角、同一实验时钟',23,MUTED)
    text(d,(1888,24),f'实验时刻  {actual:05.2f} s',34,bold=True,anchor='rt')
    tag={'full':'完整过程  1×','slow':'放下重播  0.25×','end':'最终画面停留'}[mode]
    text(d,(1888,78),tag+'  |  '+stage(actual),23,MUTED,anchor='rt')
    for j,arm in enumerate(('baseline','macro')):
        x=32+j*940;r=RECORDS[scene][arm];result=r['result'];color=COLORS[arm]
        text(d,(x,121),'力比 PI 基线' if arm=='baseline' else '中期宏观策略',30,color,True)
        force=float(r['arrays']['finger_total'][idx*10])
        text(d,(x+PW,129),f'当前总指力  {force:.2f} N',22,MUTED,anchor='rt')
        im.paste(vis.draw(arm,idx),(x,166));d=ImageDraw.Draw(im)
        d.rectangle((x,166,x+PW-1,166+PH-1),outline='#D6E0E8',width=1)
        pickup=result['pickup_stable_time_s'];rev=result['total_force']['full']['total_force_reversal_rate_N_s']
        latency=result['release_latency_s']
        text(d,(x,641),f'拿稳 {pickup:.2f} s    全程往复变化 {rev:.3f} N/s    清离 {latency:.3f} s',23,color)
        text(d,(x,679),'整次结果',19,MUTED)
        passed=result['force_limit_pass'];end=r['endpoint']['terminal_15_8_to_16s_on_receiver_and_hand_clear']
        text(d,(x+104,676),'限力：'+('通过' if passed else '未通过'),22,INK if passed else '#AD3D33')
        text(d,(x+345,676),'末态留台：'+('是' if end else '否'),22,INK if end else '#AD3D33')
        exceed=result['force_exceedance_duration_s']*1000
        text(d,(x+615,676),f'超力 {exceed:.1f} ms',22,INK if passed else '#AD3D33')
    x0,y0,x1,y1,ymax=rect;start=0 if mode=='full' else 12
    xx=x0+(actual-start)/(16-start)*(x1-x0)
    d.line((xx,y0,xx,y1),fill='#243747',width=2)
    for arm in ('baseline','macro'):
        f=float(RECORDS[scene][arm]['arrays']['finger_total'][idx*10]);yy=y1-f/ymax*(y1-y0)
        d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=COLORS[arm],outline='white',width=1)
    text(d,(32,1020),'横轴：实验时间（s） · 曲线展示完整记录，竖线指示当前时刻 · 原始数据未平滑',21,MUTED)
    text(d,(1888,1020),'限力按所有外部接触与每指上限检查',20,MUTED,anchor='rt')
    text(d,(32,1052),'末态留台为结束前 0.2 秒的补充核查；清离耗时从 12 秒放下请求起计。',18,MUTED)
    text(d,(1888,1052),'四场景 · 同一个开发种子 619741 · 仅回放已有记录',18,MUTED,anchor='rt')
    return im,actual,idx

def preview():
    paths=[]
    for scene,times in [('nominal',[0.,1.5,6.,12.4,15.2,16.]),('friction_drop',[16.]),('fragile_conflict',[15.2]),('collision',[9.435,9.6,16.])]:
        vis=Visuals(RECORDS[scene])
        try:
            for t in times:
                mode='full' if t<12 else 'slow'
                im,actual,idx=frame(scene,vis,t,mode,*chart_base(scene,0 if mode=='full' else 12))
                p=WORK/'画面检查'/f'preview_{scene}_{t:.3f}.png';im.save(p);paths.append(str(p))
        finally:vis.close()
    assert not any(BLOCKED_CALLS.values())
    dump(WORK/'预览核对.json',dict(images=paths,blocked_calls=BLOCKED_CALLS))
    print(json.dumps({'previews':paths,'blocked_calls':BLOCKED_CALLS},ensure_ascii=False),flush=True)

def timeline():
    for k in range(481):yield k/30.,'full'
    for k in range(481):yield 12+k/120.,'slow'
    for k in range(90):yield 16.,'end'

def produce(scene):
    ff=shutil.which('ffmpeg');assert ff
    dest=OUT/FILENAMES[scene];temporary=WORK/(scene+'_encoding.mp4');logfile=WORK/(scene+'_ffmpeg.log')
    cmd=[ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
         '-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264',
         '-preset','medium','-crf','18','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',
         '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',str(temporary)]
    vis=Visuals(RECORDS[scene]);bgs={k:chart_base(scene,s) for k,s in [('full',0),('slow',12),('end',12)]}
    mappings=[];start_wall=time.monotonic()
    with logfile.open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            last=None
            for n,(t,mode) in enumerate(timeline()):
                if mode=='end' and last is not None:im=last;actual=16.;idx=3200
                else:
                    im,actual,idx=frame(scene,vis,t,mode,*bgs[mode])
                    if mode=='end':last=im
                if n in [0,45,180,282,370,460,480,800,962]:im.save(WORK/'画面检查'/f'{scene}_{n:04d}.jpg',quality=93)
                if n==370:im.save(OUT/'封面'/(scene+'.jpg'),quality=94)
                proc.stdin.write(im.tobytes())
                mappings.append(dict(frame=n,video_time_s=n/FPS,segment=mode,requested_sim_time_s=t,
                    recorded_sim_time_s=actual,left_index=idx,right_index=idx,force_index=idx*10))
                if n%180==0:print(f'{scene}: {n}/1052 frames; {time.monotonic()-start_wall:.1f}s',flush=True)
            proc.stdin.close();rc=proc.wait(timeout=60)
            if rc:raise RuntimeError(logfile.read_text(encoding='utf-8',errors='replace'))
        finally:
            if proc.poll() is None:proc.kill()
            vis.close()
    assert len(mappings)==1052 and not any(BLOCKED_CALLS.values())
    temporary.replace(dest)
    dump(WORK/(scene+'_frame_mapping.json'),mappings)
    result=dict(scene=scene,title=SCENES[scene],path=str(dest),frames=1052,fps=FPS,duration_s=1052/FPS,
        frame_size=[W,H],camera=CAMERA,no_physics_steps=True,no_contact_force_recomputation=True,
        blocked_calls=BLOCKED_CALLS.copy(),visual_object_max_position_error_m=max(vis.body_errors),
        original_timestamps_shared=True,force_sampling_Hz=2000,force_curve_samples=32001,
        segments=[dict(name='full',simulation_seconds=[0,16],speed=1),dict(name='slow',simulation_seconds=[12,16],speed=.25),dict(name='end',duration_s=3)],
        sha256=sha(dest),bytes=dest.stat().st_size,wall_seconds=time.monotonic()-start_wall,
        methods={a:dict(folder=str(r['folder']),trace_sha256=r['meta']['trace_sha256'],metrics=r['result'],endpoint=r['endpoint']) for a,r in RECORDS[scene].items()})
    dump(WORK/(scene+'_export.json'),result)
    print('FINISHED '+scene,flush=True)
    return result

def finish():
    rows=[read(WORK/(s+'_export.json')) for s in SCENES]
    assert all(sha(p)==h for p,h in PRESERVED.items())
    assert not any(BLOCKED_CALLS.values())
    dump(WORK/'导出核对.json',dict(status='passed',outputs=rows,source_files_unchanged=True,
        protected_files=len(PRESERVED),physics_or_controller_calls=BLOCKED_CALLS,
        raw_force_y_axis_N=[0,25],all_scenes_included=True,source_batch=str(BATCH)))
    print('ALL FOUR EXPORTED; ORIGINAL SOURCE HASHES UNCHANGED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    if args.preview:preview()
    elif args.all:
        for s in SCENES:produce(s)
        finish()

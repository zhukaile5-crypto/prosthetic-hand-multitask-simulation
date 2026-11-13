"""Synthetic/import-only checks. No archive NPZ, server, GL or MuJoCo calls."""
from pathlib import Path
from datetime import datetime,timezone
from types import MethodType
import argparse,ast,hashlib,json,sys,shutil,subprocess
from html.parser import HTMLParser
import numpy as np
from .reader import ResultStore,metric_card,METHODS
from . import server,rendering


def run():
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    m=dict(full_motion_mission_success=False,safety=dict(guard_pass=False,finger_peak_N=[0.]*5,local_excess_integral_Ns=0.))
    m.update(static_quality=dict(eligible=False,mean_actual_finger_force_N=9.,bandlimited=dict(finger_TV_rate_N_s=8.)),
        dynamic_quality=dict(eligible=True,mean_actual_finger_force_N=0.,bandlimited=dict(finger_TV_rate_N_s=0.)),
        entire_holding_quality=dict(eligible=True,mean_actual_finger_force_N=3.,bandlimited=dict(finger_TV_rate_N_s=999.),raw_200Hz=dict(finger_TV_rate_N_s=2.)))
    card=metric_card(dict(research_metrics=m))
    check('ineligible_quality_stays_NULL',card['static']['mean_N'] is None and card['static']['tv_N_per_s'] is None)
    check('genuine_zero_stays_zero',card['dynamic']['mean_N']==card['dynamic']['tv_N_per_s']==0.)
    check('failure_not_hidden',card['full_task'] is False and card['force_guard'] is False)
    check('wholeholding_does_not_import_release_FIR_future',card['holding']['tv_N_per_s']==2.)
    check('pressure_peak_zero_not_missing',card['max_finger_peak_N']==0.)
    trace=dict(time=np.arange(3201)*.005,skin_normal=np.ones((3201,5)),phase=np.full(3201,'HOLDING'))
    check('nearest_record_no_interpolation',ResultStore.index_at(trace,7.501)==1500)
    for value in (True,float('nan'),-1.,16.1,'1'):
        try:ResultStore.index_at(trace,value)
        except ValueError:pass
        else:raise AssertionError('Invalid recorded time accepted')
    check('invalid_times_rejected',True)
    store=object.__new__(ResultStore);store.records={('nominal','hold',617901,method):dict(available=True) for method in METHODS}
    store.load_trace=MethodType(lambda self,*args:trace,store)
    sample=store.sample('nominal','hold',617901,7.501)
    check('three_methods_same_saved_clock',sample['time_s']==7.5 and len(sample['methods'])==3)
    check('display_force_is_saved_force',all(v['total_force_N']==5. for v in sample['methods'].values()))
    store.records[('nominal','hold',617901,'fixed_effort')]['available']=False
    sample=store.sample('nominal','hold',617901,7.5)
    check('unavailable_method_never_filled_zero',sample['methods']['fixed_effort']['available'] is False and 'total_force_N' not in sample['methods']['fixed_effort'])
    try:store.validate('nominal','hold',617901,'../runtime')
    except ValueError:pass
    else:raise AssertionError('Invalid method accepted')
    check('method_whitelist',True)
    check('no_engine_imported','mujoco' not in sys.modules)
    for name in ('reader.py','rendering.py','server.py'):
        tree=ast.parse((Path(__file__).parent/name).read_text(encoding='utf-8'))
        bad=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                func=node.func;called=func.attr if isinstance(func,ast.Attribute) else func.id if isinstance(func,ast.Name) else ''
                if called in {'mj_step','mj_step1','mj_step2','mj_forward','HandPlant','TrialSession','step_one_sensor_tick'}:bad.append(called)
        check(name+'/no_control_dynamics_or_forward_calls',not bad)
    check('three_visual_camera_choices',set(rendering.CAMERAS)=={'front','side','oblique'})
    class Page(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.external=[];self.inscript=False;self.scripts=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            for name in ('src','href'):
                if a.get(name,'').startswith(('http:','https:','//')):self.external.append(a[name])
            if tag=='script':self.inscript=True
        def handle_endtag(self,tag):
            if tag=='script':self.inscript=False
        def handle_data(self,data):
            if self.inscript:self.scripts.append(data)
    html=(Path(__file__).parent/'web/index.html').read_text(encoding='utf-8');page=Page();page.feed(html)
    check('page_ids_unique',len(page.ids)==len(set(page.ids)))
    check('page_assets_remain_local',not page.external)
    check('page_explicit_recorded_results','已运行结果回放' in html and '不启动新的实验' in html)
    check('page_api_contract',all(name in html for name in ('/api/catalog','/api/context','/api/sample','/api/frame')))
    node=shutil.which('node')
    syntax='not_checked_node_unavailable'
    if node:
        result=subprocess.run([node,'--check'],input='\n'.join(page.scripts),text=True,encoding='utf-8',capture_output=True,timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        check('javascript_syntax_only_no_execution',result.returncode==0);syntax='passed_node_check_no_execution'
    return dict(schema='recorded_visual_replay_pure_checks_v1',passed=True,checks=checks,actual_archive_npz_read=False,
        server_started=False,renderer_created=False,new_physics_steps=0,
        frontend_syntax_check=syntax,
        limitations=['No rendered pixel or browser behavior has been tested.','Root must separately schedule real archive/renderer/UI validation after the source80 measurement run.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);args=p.parse_args();report=run()
    report['created_utc']=datetime.now(timezone.utc).isoformat()
    folder=Path(__file__).parent
    paths=sorted(folder.glob('*.py'))+[folder/'README.md',folder/'web/index.html']
    report['source_sha256']={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=True,checks=len(report['checks']),actual_archive_npz_read=False,server_started=False,renderer_created=False,new_physics_steps=0)))


if __name__=='__main__':main()

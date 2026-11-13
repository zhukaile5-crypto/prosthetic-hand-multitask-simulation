"""Small pure interface gate: synthetic role/NULL/timing and pending fence.
No real result/NPZ/model/renderer/server/browser is used by these checks.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,json,sys
import numpy as np
from . import reader as r

def main():
    checks=[]
    def ck(name,value):checks.append(dict(name=name,passed=bool(value)))
    ck('same_underlying_ratio_has_two_distinct_roles',r.role_for('candidate_A','ratio_effort')=='candidate_A' and r.role_for('baseline','ratio_effort')=='baseline_ratio')
    for arm,method in [('candidate_A','fusion_effort'),('unknown','ratio_effort')]:
        try:r.role_for(arm,method);okay=False
        except ValueError:okay=True
        ck('reject_unregistered_'+arm+'_'+method,okay)
    result=dict(task_success=False,research_metrics=dict(full_motion_mission_success=False,safety=dict(guard_pass=True),
        static_quality=dict(eligible=False,mean_actual_finger_force_N=0.,raw_200Hz=dict(finger_TV_rate_N_s=0.)),
        dynamic_quality=dict(eligible=True,mean_actual_finger_force_N=2.,raw_200Hz=dict(finger_TV_rate_N_s=3.),bandlimited=dict(finger_TV_rate_N_s=99.))))
    m=r.original_metrics(result)
    ck('original_NULL_preserved_real200Hz_selected',m['static']['mean_N'] is None and m['static']['tv_N_per_s'] is None and m['dynamic']['tv_N_per_s']==3. and m['full_task'] is False and m['force_guard'] is True)
    report=dict(measurements=dict(static=dict(unselected_force=dict(available=True,mean_total_force_N=0.,finger_TV_rate_N_s=0.))),floor=dict(release_and_after12to16=dict(available=False,any_positive_contact=False)))
    d=r.diagnostic_metrics(report)
    ck('allassigned_zero_distinct_from_unknown_floor',d['windows']['static']['mean_N']==0. and d['windows']['static']['available'] and d['delivery']['floor_contact'] is None)
    data=dict(time=np.arange(3201)*.005)
    ck('record_time_nearest_and16_endpoint',r.ResultStore.index_at(data,8.5)==1700 and r.ResultStore.index_at(data,16.)==3200 and r.ResultStore.index_at(data,.0025)==0)
    for value in (True,np.bool_(False),float('nan'),-.1,16.1,'8.5'):
        try:r.ResultStore.index_at(data,value);okay=False
        except ValueError:okay=True
        ck('invalid_time_'+str(value),okay)
    store=r.ResultStore.__new__(r.ResultStore)
    try:store.validate('nominal','hold',619731,'ratio_effort');okay=False
    except ValueError:okay=True
    ck('underlying_method_not_accepted_as_role',okay)
    plan=r.read(r.DEFAULT_PLAN);original_read=r.read;access=[]
    with TemporaryDirectory(prefix='v15A_replay_pure_') as tmp:
        def guarded_read(path):
            access.append(str(path))
            if Path(path).resolve()==r.DEFAULT_PLAN.resolve():return deepcopy(plan)
            if Path(path).name=='execution_complete.json':return dict(execution_complete=False)
            raise AssertionError('Partial archive cannot be read: '+str(path))
        for stage in plan['stages']:
            p=Path(tmp)/stage['output'];p.mkdir(parents=True);(p/'execution_complete.json').write_text('{}',encoding='utf-8')
        with patch.object(r,'ROOT',Path(tmp)),patch.object(r,'read',guarded_read):
            pending=r.ResultStore(r.DEFAULT_PLAN);catalog=pending.catalog()
        ck('pending_fence_before_any_result_orNPZ_read',catalog['status']=='pending' and len(catalog['contexts'])==16 and not pending.records and all(Path(p).name in ('v15A_prospective_multiscene_plan_v2.json','execution_complete.json') for p in access))
    root=Path(__file__).parent
    for name in ('reader.py','rendering.py','server.py'):
        tree=ast.parse((root/name).read_text(encoding='utf-8'))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        ck('no_dynamics_calls_'+name,not set(calls)&{'mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'})
    html=(root/'web/index.html').read_text(encoding='utf-8')
    ck('four_roles_responsive_and_metric_scope_controls',all(x in html for x in r.METHODS) and all(x in html for x in ('repeat(4,minmax(0,1fr))','repeat(2,minmax(0,1fr))','grid-template-columns:1fr','全分配连续诊断','原评价 · 合格窗口','12–16秒落地','当前皮肤总指力','target_total_force_N')))
    ck('imports_do_not_create_model_or_server','mujoco' not in sys.modules and 'v15A_replay_v1.rendering' not in sys.modules and 'v15A_replay_v1.server' not in sys.modules)
    out=dict(schema='v15A_replay_pure_v1',passed=all(c['passed'] for c in checks),checks=checks,
        plan_sha256=r.PLAN_SHA,continuous_reader_sha256=r.DIAGNOSTIC_SHA,source_sha256={p.relative_to(root).as_posix():r.sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.html')},
        actual_model_created=False,renderer_started=False,server_started=False,browser_opened=False,physics_steps=0,
        scope='Small synthetic interface and AST gate only; no actual rendering, browser or physical performance assertion')
    path=root/'pure_checks_v1.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(path),sha256=r.sha(path)),ensure_ascii=False))
    raise SystemExit(0 if out['passed'] else 1)
if __name__=='__main__':main()

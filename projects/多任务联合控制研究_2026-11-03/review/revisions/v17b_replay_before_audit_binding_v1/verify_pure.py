"""Five-arm identity, pending fence and auxiliary display checks, no physics."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,json,sys
import numpy as np
from . import reader as r


def main():
    checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    ck('same_ratio_three_distinct_roles',[
        r.role_for(a,'ratio_effort') for a in ('full_V17','holding_A','baselines')
        ]==['full_V17','holding_A','baseline_ratio'])
    for arm,method in [('full_V17','fusion_effort'),('holding_A','fixed_effort'),('baseline','ratio_effort')]:
        try:r.role_for(arm,method);ok=False
        except ValueError:ok=True
        ck('reject_wrong_role_'+arm+'_'+method,ok)
    original=dict(task_success=False,research_metrics=dict(full_motion_mission_success=False,
        safety=dict(guard_pass=True),static_quality=dict(eligible=False,mean_actual_finger_force_N=0.,
            raw_200Hz=dict(finger_TV_rate_N_s=0.)),dynamic_quality=dict(eligible=True,
            mean_actual_finger_force_N=2.,raw_200Hz=dict(finger_TV_rate_N_s=3.))))
    metric=r.original_metrics(original)
    ck('original_failure_NULL_preserved',metric['full_task'] is False and metric['static']['mean_N'] is None
        and metric['static']['tv_N_per_s'] is None and metric['dynamic']['tv_N_per_s']==3. and metric['force_guard'] is True)
    data=dict(time=np.arange(3201)*.005)
    ck('nearest_record_time_and_terminal',r.ResultStore.index_at(data,16.)==3200
        and r.ResultStore.index_at(data,.0025)==0)
    for t in [True,float('nan'),-.01,16.01,'12']:
        try:r.ResultStore.index_at(data,t);ok=False
        except ValueError:ok=True
        ck('reject_invalid_time_'+str(t),ok)
    plan=r.read(r.DEFAULT_PLAN);access=[]
    with TemporaryDirectory(prefix='v17b_replay_fence_') as temp:
        def guarded(path):
            access.append(Path(path).name)
            if Path(path).resolve()==r.DEFAULT_PLAN.resolve():return deepcopy(plan)
            if Path(path).name=='execution_complete.json':return dict(execution_complete=False)
            raise AssertionError('Pending comparison read a result: '+str(path))
        for stage in plan['stages']:
            p=Path(temp)/stage['output'];p.mkdir(parents=True);(p/'execution_complete.json').write_text('{}')
        with patch.object(r,'ROOT',Path(temp)),patch.object(r,'read',guarded):
            store=r.ResultStore();catalog=store.catalog()
        ck('all80_pending_fence_before_result_orNPZ',catalog['status']=='pending' and len(catalog['contexts'])==16
            and len(catalog['methods'])==5 and not store.records and set(access)<=
            {'v17b_prospective80_plan_v1.json','execution_complete.json'})
    root=Path(__file__).parent
    for name in ('reader.py','server.py','rendering.py'):
        tree=ast.parse((root/name).read_text(encoding='utf-8'))
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        ck('no_physical_calls_'+name,not calls&{'mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'})
    html=(root/'web/index.html').read_text(encoding='utf-8')
    ck('five_roles_and_auxiliary_controls',all(role in html for role in r.METHODS) and all(x in html for x in
        ('repeat(5,minmax(0,1fr))','repeat(2,minmax(0,1fr))','grid-template-columns:1fr',
         '首次低运动驻留','末段低运动驻留','触台20ms峰 / 冲量','无可评新接触事件','未达到','619741','619742')))
    ck('import_is_not_renderer_or_physics','mujoco' not in sys.modules and 'v17b_replay_v1.rendering' not in sys.modules)
    report=dict(schema='v17b_replay_pure_v1',passed=all(c['passed'] for c in checks),checks=checks,
        source_sha256={str(p.relative_to(root)):r.sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.html')},
        plan_sha256=r.PLAN_SHA,physics_steps=0,renderer_started=False,server_started=False,
        actual_new_results_read=False,browser_checked=False)
    out=root/'pure_checks_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']])))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

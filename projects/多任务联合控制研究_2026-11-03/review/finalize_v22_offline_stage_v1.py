"""Read back every saved refinement witness and persist the final stage state."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'review'))
from v21_original_matrix_kkt_verifier_v1 import verify_stage
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
rname='review/v22_joint_active_face_refinement_v1.json'
d=read(rname);old=read('review/v22_joint_bundle_offline_v1.json')
assert d['complete'] and d['assigned']==d['assessed']==16 and not d['errors']
assert sha(ROOT/'review/v22_joint_bundle_offline_v1.json')==d['original_candidate_file_sha256']
checks=[]
for row in d['rows']:
    witness=row['after_same_method']['dual_witness']
    v=verify_stage(row['original_joint_problem'],row['after_x'],duals=witness,comparison_x=row['before_x'])
    assert v['kkt_passed'] and v['original_problem_sha256']==row['original_problem_sha256'],row['context_key']
    checks.append(dict(context_key=row['context_key'],original_problem_sha256=v['original_problem_sha256'],
        passed=v['kkt_passed'],primal_inf=v['primal']['inequality_positive_violation_inf'],
        stationarity_inf=v['stationarity_inf'],no_optimizer_replay=True))
for k,v in read('review/v21_complete16_auxiliary_v1.json')['source_sha256'].items():
    p=ROOT.parent/'五指策略接入_2026-11-03'/k[11:] if k.startswith('foundation/') else ROOT/k
    assert sha(p)==v,k
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');now=datetime.now(timezone.utc).isoformat()
report='V22联合任务重构依据与离线原型结果_20261105.md';body=(ROOT/report).read_text(encoding='utf-8')
assert '## 本轮收尾：联合主任务16个原问题验收通过' not in body
hist=ROOT/'report/progress_history';hist.mkdir(parents=True,exist_ok=True)
for n in ('研究进度.json','当前进度与阶段汇报.md',report):shutil.copyfile(ROOT/n,hist/(stamp+'_'+n))
body+='''
## 本轮收尾：联合主任务16个原问题验收通过

首轮8个未获KKT确认的候选与全部旧结果保留。随后对全部16候选，按原坐标容差选择一次紧约束面，求原联合LS在该面的解，再回完整原G/E/目标核验。没有修改目标、边界、验收容差或反复选择活动阈值。细化前后使用同一独立对偶可行性方法，严格KKT从8/16增至16/16。主程序再逐条读取保存的原问题/新点/对偶，16项复核通过，未重解优化器。

这主要是数值表示改善：前后联合目标差仅−3.97e−13至+2.84e−14，5例浮点上升也保留，不能称额外物理或优化收益。最大原问题stationarity低于1e−15、primal低于1e−15；没有5毫秒实时保证。

联合主任务的这批有限输入已通过，但完整退出进度/路径层、未知或无运动反馈分支、闭环执行适配、全场景与未见输入仍未验收。当前没有V22物理结果，下一步完成这些接线后再按同后端三臂方案测试。

- [一次活动面细化与全部16原问题证据](review/v22_joint_active_face_refinement_v1.json)
- [细化范围及数值量级](review/V22联合任务一次活动面细化诊断_v1.md)
- [主程序逐条读回核验](review/v22_joint_refinement_receipt_v1.json)
- [两个任务反事实独立代入审查](review/v21_motion_tradeoff_cross_independent_v2.json)
- [依据新证据更新的整体设计评审](review/v22_task_priority_design_review_v2.md)
'''
(ROOT/report).write_text(body,encoding='utf-8')
save('review/v22_joint_refinement_receipt_v1.json',dict(complete=True,assigned=16,passed=sum(c['passed'] for c in checks),
    checks=checks,refinement_sha256=sha(ROOT/rname),verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'),
    physics_steps=0,optimizer_replays=0,source170_unchanged=True))
p=read('研究进度.json');p.update(updated_utc=now,goal_status='active',superiority_proven=False,
    implementation='V21 physical source170 frozen; V22 independent joint-task algebra prototype plus offline16 primary-task certification, full runtime not connected.',
    current_work='V22 joint primary finite tasks verified16/16 after once-only active-face refinement; full closed-loop hierarchy and same-condition three-arm physics remain required.')
p['v22_current'].update(status='joint_primary_offline16_certified_full_runtime_pending',initial_candidate_KKT_passed=8,
    offline_KKT_passed=16,refinement=rname,refinement_receipt='review/v22_joint_refinement_receipt_v1.json',
    runtime_connected=False,physics_active=False,physical_runs=0,
    remaining=['Generalized candidate/refinement/certificate backend', 'Complete joint/exit/path layers and unknown/P0 branches',
        'Physical adapter and actual emitted-task validation', 'Same-condition backend-only and joint-task arms', 'Full matrix and unseen confirmation'])
save('研究进度.json',p)
(ROOT/'当前进度与阶段汇报.md').write_text(f'''# V22联合主任务原型已通过16个离线原问题验收

更新UTC：{now}。主目标active，V21完整16物理结果冻结，无物理仿真在后台运行。

本轮固定65条同问题数值对照、两个各16条任务反事实表明：只更换库或交换退出/运动层序不足以解决当前限制。新V22将几何纠偏与有限运动投影共同协调；原型已写入独立目录。

首轮16/16可行、8/16 KKT；全部16一次活动面细化后，原完整问题KKT16/16，旧失败保留。固定历史输入运动代理目标配对降幅中位54.7%，几何纠偏代理代价增加0.2%–1.6%。这些是任务代数结果，不能说真实力波动或完成率已进一步改善。

完整退出/路径/未知分支、实际执行适配及新物理对照尚未完成；下一步完成完整接线，再做同后端的数值单改与联合任务两臂对照。

[本轮完整重构依据与结果]({report})；[V21完整16物理结果](V21完整16条件综合对照与数值重构依据_20261105.md)；[16项原问题读回核验](review/v22_joint_refinement_receipt_v1.json)。
''',encoding='utf-8')
print(json.dumps({'report':report,'refinement_original_KKT_readback':16,'source170_unchanged':True,
    'goal_status':'active','runtime_connected':False,'physics_active':False},ensure_ascii=False))

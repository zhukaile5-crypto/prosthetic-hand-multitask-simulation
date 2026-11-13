"""Read-only V19 post-release branch/goal mechanism evidence, not a new solve.
All3 complete candidate decision archives are read once. Last12.5..16.0s is
exhaustive (701 decisions each). Fixed times declared before calculation:
12.5,13,13.5,14,14.5,15,15.5,16. No model/FK/physics/native/optimizer calls.
Reviewer authored task_qp; this is algebraic execution diagnosis, not its
independent mathematical acceptance or an intervention on a physical system.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v19_small_plan_v1.json'
TIMES=(12.5,13.,13.5,14.,14.5,15.,15.5,16.)
SIGN=1<<63;MASK=(1<<64)-1

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def stats(values):
    a=np.asarray(values,float)
    return None if not a.size else dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()))
def ordered(x):
    bits=int(np.float64(x).view(np.uint64));return (~bits)&MASK if bits&SIGN else bits|SIGN
def ulps(x,lo,hi):
    return dict(from_lower=ordered(x)-ordered(lo),to_upper=ordered(hi)-ordered(x),inside=bool(lo<=x<=hi))
def restore(p):
    if type(p['Jpx25']) is list and p['Jpx25']==[] and all(type(p[k]) is list and not p[k] for k in ('y_p','patch_digit_p')) and p['scales']['intentP']==[]:
        p=dict(p);p['Jpx25']=np.empty((0,25));return p
    return p

def losses(problem,du,z,gamma,actual_delta=None):
    p=restore(problem);a=lambda k:np.asarray(p[k],float);s=p['scales']
    affine=a('offset21')+a('gain21')*np.asarray(du)
    d=affine if actual_delta is None else np.asarray(actual_delta,float)
    correction=a('Jpx25').T@(np.asarray(z)-a('y_p'))+a('tau_extra25')
    r=a('A21x25').T@d-correction
    wr=np.asarray(s['residual25'])*r;wd=np.asarray(s['delta21'])*d
    wu=np.asarray(s['reference21'])*(np.asarray(du)-gamma*a('open_step21')-a('motion_step21'))
    wz=np.asarray(s['intentP'])*(np.asarray(z)+gamma*p['unload_fraction']*a('y_p')-a('y_p'))
    wg=s['progress']*(gamma-1.)
    terms=dict(mechanical=float(wr@wr),normal=float(wz@wz),reference=float(wu@wu),delta=float(wd@wd),progress=float(wg*wg))
    # Preserve raw kernel addition order, independently written from the equation.
    total=float(wr@wr+wd@wd+wu@wu+wz@wz+wg*wg)
    feasible=bool(np.all(np.asarray(du)>=a('delta_lower21')) and np.all(np.asarray(du)<=a('delta_upper21')) and p['gamma_lower']<=gamma<=p['gamma_upper'] and np.all(np.asarray(z)>=0) and sum(z)<=p['total_budget']+p.get('atol',1e-8) and all(sum(np.asarray(z)[np.asarray(p['patch_digit_p'])==i])<=p['caps5'][i]+p.get('atol',1e-8) for i in range(5)))
    return dict(terms=terms,total=total,feasible=feasible,actual_native_delta_used=actual_delta is not None,
        residual_inf=float(np.max(abs(r))),du21=np.asarray(du).tolist(),zP=list(z),gamma=gamma)

def samefloat(x,y):return np.float64(x).tobytes()==np.float64(y).tobytes()
def segments(qs):
    out=[]
    for q in qs:
        if not out or out[-1]['phase']!=q['phase']:
            out.append(dict(phase=q['phase'],first_decision_s=q['time_s'],last_decision_s=q['time_s'],count=1))
        else:out[-1]['last_decision_s']=q['time_s'];out[-1]['count']+=1
    return out

def snapshot(q,axes,goal):
    p=restore(q['qp_problem']);qp=q['qp'];b=q['branch'];u=np.asarray(q['current_reference21']);ex=np.asarray(q['execution_receipt']['actual_reference21'])
    bl=np.asarray(b['reference_lower21']);bh=np.asarray(b['reference_upper21']);lo=np.asarray(q['actual_reference_lower21']);hi=np.asarray(q['actual_reference_upper21'])
    preferred=np.asarray(q['preferred_reference21']);opened=np.asarray(q['open_reference21']);anchor=q['proposal_acceptance']['anchor']
    raw=losses(p,qp['du21'],qp['zP'],qp['gamma'])
    actual=losses(p,ex-u,qp['zP'],qp['gamma'],q['native_actual']['actual_delta_effort21'])
    av=losses(p,anchor['du21'],anchor['zP'],anchor['gamma'])
    ideal=np.asarray(p['motion_step21'])+qp['gamma']*np.asarray(p['open_step21'])
    domain_ideal=np.minimum(np.asarray(p['delta_upper21']),np.maximum(np.asarray(p['delta_lower21']),ideal))
    iv=losses(p,domain_ideal,qp['zP'],qp['gamma'])
    upper_gamma=losses(p,qp['du21'],qp['zP'],p['gamma_upper'])
    allaxes=[]
    for j,name in enumerate(axes):
        fullgoal=float(goal[j]) if j<18 else None
        allaxes.append(dict(axis=j,name=name,unit='m' if j==18 else 'rad',current=float(u[j]),actual=float(ex[j]),local_open_endpoint=float(opened[j]),preferred=float(preferred[j]),terminal_hand_goal=fullgoal,
            task_lower=float(lo[j]),task_upper=float(hi[j]),branch_lower=float(bl[j]),branch_upper=float(bh[j]),task_width=float(hi[j]-lo[j]),branch_width=float(bh[j]-bl[j]),
            kind=b['branch_kind21'][j],gain=b['gain21'][j],offset=b['offset21'][j],actual_du=float(ex[j]-u[j]),
            current_branch_ULPs=ulps(u[j],bl[j],bh[j]),actual_branch_ULPs=ulps(ex[j],bl[j],bh[j]),preferred_branch_ULPs=ulps(preferred[j],bl[j],bh[j]),open_endpoint_branch_ULPs=ulps(opened[j],bl[j],bh[j])))
    return dict(decision_id=q['decision_id'],time_s=q['time_s'],phase=q['phase'],gamma=qp['gamma'],gamma_domain=q['gamma_domain'],reaction_share=q['reaction_share'],
        accepted=q['reference_effort_goal_accepted'],qp_status=qp['status'],low_hand=q['public_process']['hand_low_evidence'],governor=q['governor'],
        objective_parts=dict(raw_affine_candidate=raw,actual_native_candidate=actual,original_feasible_anchor=av,
            same_z_gamma_closest_reference_preference_within_branch=iv,same_du_z_upper_gamma=upper_gamma),
        raw_objective_bytes_equal=samefloat(raw['total'],qp['objective']),actual_objective_bytes_equal=samefloat(actual['total'],q['actual_full_task_score']['objective']),anchor_objective_bytes_equal=samefloat(av['total'],q['proposal_acceptance']['anchor_objective']),
        original_reference_preference_inside_branch=bool(np.all(ideal>=np.asarray(p['delta_lower21'])) and np.all(ideal<=np.asarray(p['delta_upper21']))),axes21=allaxes)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if not args.execute:raise ValueError('Use --execute only after explicit all4 terminal authorization')
    plan=read(PLAN)
    if not all(read(ROOT/s['output']/'execution_complete.json').get('execution_complete') is True for s in plan['stages']):raise ValueError('PENDING')
    out=ROOT/'review/v19_goal_branch_mechanism_v1.json'
    if out.exists():raise FileExistsError(out)
    inputs={str(PLAN):sha(PLAN)};rows=[];checks=[]
    for stage in plan['stages']:
        if stage['assignment']['receiver_policy']=='blind':continue
        a=stage['assignment'];run=ROOT/stage['output']/'runs'/f"{a['scenarios'][0]}__{a['methods'][0]}__{a['seeds'][0]}"
        meta=read(run/'metadata.json');result=read(run/'result.json');dm=read(run/'release_decision_manifest.json')
        for p in (run/'metadata.json',run/'result.json',run/'release_decision_manifest.json',run/dm['file']):inputs[str(p)]=sha(p)
        with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:full=[json.loads(v) for v in f]
        qs=[q for q in full if 12.5<=q['time_s']<=16.];axes=meta['robot_actuator_telemetry']['axes21'];goal=np.asarray(meta['release_clearance_goal']['reference18'],float)
        ck=dict(stage=stage['name'],full_records=len(full),window_records=len(qs),full801=len(full)==801,last3p5_701=len(qs)==701,archive_hash_matches=inputs[str(run/dm['file'])]==dm['sha256'])
        axisrows=[];gtime=[];kindstats=Counter();all_gamma=[];loss_agg={key:[] for key in ('mechanical','normal','reference','delta','progress')};bad=[]
        for q in qs:
            u=np.asarray(q['current_reference21']);ex=np.asarray(q['execution_receipt']['actual_reference21']);qp=q['qp'];b=q['branch'];p=restore(q['qp_problem'])
            l=losses(p,qp['du21'],qp['zP'],qp['gamma']);actual=losses(p,ex-u,qp['zP'],qp['gamma'],q['native_actual']['actual_delta_effort21'])
            if not samefloat(l['total'],qp['objective']) or not samefloat(actual['total'],q['actual_full_task_score']['objective']):bad.append(q['time_s'])
            for key in loss_agg:loss_agg[key].append(actual['terms'][key])
            all_gamma.append(qp['gamma']);gtime.append(dict(time_s=q['time_s'],phase=q['phase'],gamma=qp['gamma'],domain=q['gamma_domain'],reaction_share=q['reaction_share'],open18_actual_inf_error=float(np.max(abs(ex[:18]-goal))),low_hand_pressure_N=q['public_process']['low_hand_pressure_N'],low_hand_confirmed=q['public_process']['low_hand_pressure_confirmed'],lift_ref_m=float(ex[18])))
            kindstats.update(b['branch_kind21'])
        for j,name in enumerate(axes):
            tw=[];bw=[];ratio=[];counts=Counter();prefgap=[];endpointgap=[];gains=[];offs=[];actualdu=[];cases=[]
            for q in qs:
                b=q['branch'];lo=q['actual_reference_lower21'][j];hi=q['actual_reference_upper21'][j];bl=b['reference_lower21'][j];bh=b['reference_upper21'][j]
                w=hi-lo;v=bh-bl;cur=q['current_reference21'][j];pref=q['preferred_reference21'][j];end=q['open_reference21'][j];act=q['execution_receipt']['actual_reference21'][j]
                tw.append(w);bw.append(v);gains.append(b['gain21'][j]);offs.append(b['offset21'][j]);actualdu.append(act-cur)
                if w>0:ratio.append(v/w)
                counts['task_singleton']+=w==0;counts['branch_singleton']+=v==0;counts['task_mobile_branch_singleton']+=w>0 and v==0
                counts['known_plateau_mobile']+=v>0 and b['gain21'][j]==0;counts['current_outside_selected_branch']+=not bl<=cur<=bh
                counts['preferred_outside_branch']+=not bl<=pref<=bh;counts['open_endpoint_outside_branch']+=not bl<=end<=bh
                counts['actual_at_boundary']+=act==bl or act==bh;counts['actual_exactly_current']+=act==cur
                prefgap.append(max(bl-pref,0,pref-bh));endpointgap.append(max(bl-end,0,end-bh))
                if w>0 and (v==0 or not bl<=pref<=bh) and len(cases)<3:cases.append(dict(time_s=q['time_s'],kind=b['branch_kind21'][j],task=[lo,hi],branch=[bl,bh],current=cur,preferred=pref,open=end,actual=act))
            axisrows.append(dict(axis=j,name=name,unit='m' if j==18 else 'rad',counts=dict(counts),kind_counts=dict(Counter(q['branch']['branch_kind21'][j] for q in qs)),task_width=stats(tw),branch_width=stats(bw),width_fraction_if_task_mobile=stats(ratio),preferred_excluded_distance=stats(prefgap),open_endpoint_excluded_distance=stats(endpointgap),gain=stats(gains),offset=stats(offs),actual_increment=stats(actualdu),first_examples=cases))
        shots=[snapshot(next(q for q in qs if q['time_s']==t),axes,goal) for t in TIMES]
        ck.update(all_1402_raw_and_actual_totals_byte_equal=not bad,fixed_anchors_totals_byte_equal=all(s['anchor_objective_bytes_equal'] for s in shots),all_fixed_original_scores_byte_equal=all(s['raw_objective_bytes_equal'] and s['actual_objective_bytes_equal'] for s in shots));checks.append(ck)
        end=qs[-1];actual=np.asarray(end['execution_receipt']['actual_reference21']);err=actual[:18]-goal;imax=int(np.argmax(abs(err)))
        rows.append(dict(stage=stage['name'],run=str(run),original_task_success=result['task_success'],original_release_success=result['release_success'],original_failure_reason=result['failure_reason'],failure_at=result['failure_at'],
            full_release_phase_segments=segments(full),last3p5_phase_segments=segments(qs),last3p5_gamma=stats(all_gamma),last3p5_gamma_exact_zero_count=sum(g==0 for g in all_gamma),last3p5_gamma_exact_one_count=sum(g==1 for g in all_gamma),
            last3p5_all_accepted=all(q['reference_effort_goal_accepted'] for q in qs),last3p5_all_converged=all(q['qp']['status']=='converged' for q in qs),
            terminal=dict(time_s=end['time_s'],phase=end['phase'],gamma=end['qp']['gamma'],gamma_domain=end['gamma_domain'],actual_reference21=actual.tolist(),clearance_goal18=goal.tolist(),hand_reference_error18=err.tolist(),hand_reference_error_inf_rad=float(abs(err).max()),maximum_error_axis=axes[imax],low_hand_evidence=end['public_process']['hand_low_evidence'],receiver_event_at=end['public_process']['receiver_event_at'],governor=end['governor']),
            per_axis=axisrows,gamma_and_progress_time_series=gtime,fixed_snapshots=shots,descriptive_loss_terms_per5ms_not_integrated= {k:stats(v) for k,v in loss_agg.items()},branch_kind_axis_frame_counts=dict(kindstats)))
    passed=all(c['full801'] and c['last3p5_701'] and c['archive_hash_matches'] and c['all_1402_raw_and_actual_totals_byte_equal'] and c['fixed_anchors_totals_byte_equal'] and c['all_fixed_original_scores_byte_equal'] for c in checks)
    report=dict(schema='v19_branch_goal_execution_mechanism_v1',passed=passed,role='task_qp author algebraic cross-execution diagnosis, not independent kernel proof',plan_sha256=sha(PLAN),reader_sha256=sha(__file__),source_sha256={k:v for k,v in plan['source_sha256'].items() if k.startswith('v19_contact_reference/')},input_sha256=inputs,
        scope=dict(candidate_runs=3,full_decisions_read=2403,last3p5_decisions_checked=2103,raw_and_actual_objective_recomputations=4206,fixed_times_s=TIMES,solver_calls=0,model_constructions=0,physics_steps=0),checks=checks,rows=rows,
        units_and_limits=['ULP counts are signed ordered-binary64 distances to each selected branch endpoint, not physical slack or safety margins.','All interval widths have actual reference-axis units; no mixed-axis sum is used.','Normal z is N, mechanical residual generalized effort, delta regularizer actuator effort, reference rows rad/m, progress dimensionless; reported loss terms already contain preregistered scales.','Fixed anchor and upper-gamma/reference-preference comparisons are same-goal algebra, not a new solver, native law beyond the selected branch, pressure model or physical intervention.','A preferred endpoint outside the selected affine cell proves only algorithmic restriction; it does not prove the robot could safely cross that kink.','OPEN reference agreement does not imply actual finger configuration, loss of contact or target delivery.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V19支域与软目标执行诊断','',f'数据/代数核验：{passed}。完整三条各801决策；12.5–16s逐条701，共2103。固定时刻八点预先声明，未求解或运行物理。','', '| 条目 | 原清离成功 | 末阶段 | γ | OPEN18参考最大误差rad | 最大误差轴 | 末手压N |','|---|---|---|---:|---:|---|---:|']
    for row in rows:
        t=row['terminal'];lines.append(f"| {row['stage']} | {row['original_release_success']} | {t['phase']} | {t['gamma']:.8g} | {t['hand_reference_error_inf_rad']:.9g} | {t['maximum_error_axis']} | {t['low_hand_evidence']['pressure_N']:.9g} |")
    lines+=['','两失败末帧γ均被UNSUPPORTED_EXIT固定为1，OPEN参考已接近目标。由此不能把终端不清离直接归于γ选择0或显著OPEN命令缺失。具体较早支域排除/平台/单点和损失折中见JSON的per_axis与fixed_snapshots。','', '末状态手压低证据未成立与失败同时存在，是日志事实；接触为何持续、提高或改变运动能否清离仍需几何/动态证据，不能仅凭本代数诊断归因。','', '本审者为QP作者；这里是同目标分项/实际执行诊断，不是数学独立证明或新策略/安全性能证明。所有来源、原失败与unknown均保留。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=passed,out=str(out),sha256=sha(out),checks=checks),ensure_ascii=False));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())

"""Read-only derived-feature/actual-force association digest; no model import."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GROUPS={'TH':[0,1,2,3,4],'FF':[5,6,7],'MF':[8,9,10],'RF':[11,12,13],
        'LF':[14,15,16,17],'lift':[18],'WRJ2':[19],'WRJ1':[20]}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();out=Path(args.out)
    if out.exists() or out.with_suffix('.md').exists():raise FileExistsError('Fresh evidence required')
    source=ROOT/'review/v10_hold_collision_servo_effort_mechanism_v1.json';main=js(source);derived=source.with_suffix('.npz')
    if sha(derived)!=main['derived_timeseries_sha256']:raise ValueError('Derived identity changed')
    with np.load(derived,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    t=a['joint__time'];kp=np.asarray(main['kp21']);kv=np.asarray(main['kv21'])
    dt=.005;ends=np.array([int(np.argmin(abs(t-w['end_s']))) for w in main['plan_windows']]);active=np.unique(np.concatenate([np.arange(k-4,k) for k in ends]))
    du=a['joint__actual_before21']-a['ratio_preview__actual_before21']
    dl=a['joint__length']-a['ratio_preview__length'];dv=a['joint__speed']-a['ratio_preview__speed']
    de=a['joint__effort']-a['ratio_preview__effort'];dp=a['joint__strain_proxy']-a['ratio_preview__strain_proxy']
    ds=a['joint__skin']-a['ratio_preview__skin'];dn=a['joint__normal']-a['ratio_preview__normal']
    dd=ds-dn
    if not np.allclose(de,du*kp-dl*kp-dv*kv,atol=1e-12,rtol=1e-10):raise AssertionError('Unsaturated exact effort decomposition')
    groups={}
    for name,g in GROUPS.items():
        ref=(du[active][:,g]*kp[g]).sum(1);position=(-dl[active][:,g]*kp[g]).sum(1);damping=(-dv[active][:,g]*kv[g]).sum(1)
        q1=main['sections']['twenty_plan_windows_union']['joint']['groups'][name]['strain_proxy_mean_J'];q0=main['sections']['twenty_plan_windows_union']['ratio_preview']['groups'][name]['strain_proxy_mean_J']
        groups[name]=dict(active_reference_elastic_drive_difference_mean=float(ref.mean()),
            active_actual_transmission_deflection_drive_difference_mean=float(position.mean()),
            active_damping_effort_difference_mean=float(damping.mean()),active_net_effort_difference_mean=float(de[active][:,g].sum(1).mean()),
            strain_proxy_mean_relative_change=(q1/q0-1) if q0>0 else None,
            window_end_positive_strain_proxy_difference_count=int((dp[ends][:,g].sum(1)>1e-12).sum()),
            unit='N' if name=='lift' else 'Nm; signed sum in actuator transmission coordinates, not a grasp-closing scalar')
    actuator=[]
    for k,name in enumerate(main['actual_actuator_names21']):
        actuator.append(dict(name=name,reference_component_mean=float((du[active,k]*kp[k]).mean()),
            position_component_mean=float((-dl[active,k]*kp[k]).mean()),damping_component_mean=float((-dv[active,k]*kv[k]).mean()),
            effort_difference_mean=float(de[active,k].mean()),strain_proxy_difference_mean_J=float(dp[active,k].mean()),
            actual_reference_max_abs_difference=float(abs(du[:,k]).max()),
            unit='N' if k==18 else 'Nm'))
    raw={};sources={}
    for method in ['joint','ratio_preview']:
        path=ROOT/'output/development_v10_hold_v1/runs'/f'collision__{method}__610212/trace.npz'
        expected=main['input_sha256'][method][str(path)]
        if sha(path)!=expected:raise ValueError('Original trace changed')
        with np.load(path,allow_pickle=False) as z:
            rt=z['substep_time'];rd=z['substep_duration'];skin=z['substep_skin_normal'];normal=z['substep_normal']
        window=(rt>=7.5-1e-9)&(rt<12.-1e-9);actual_active=np.zeros(len(rt),bool)
        for w in main['plan_windows']:actual_active|=(rt>=w['start_s']-1e-9)&(rt<w['end_s']-1e-9)
        raw[method]=dict(t=rt,dt=rd,skin=skin,normal=normal,hold=window,active=actual_active)
        sources[str(path)]=expected
    x=raw['joint'];b=raw['ratio_preview']
    if not np.array_equal(x['t'],b['t']) or not np.array_equal(x['dt'],b['dt']):raise AssertionError('Raw clocks differ')
    force_splits={}
    for name,mask in [('full_holding',x['hold']),('twenty_plan_windows',x['active'])]:
        deltas=x['skin'][mask]-b['skin'][mask];targets=x['normal'][mask]-b['normal'][mask]
        force_splits[name]=dict(substeps=int(mask.sum()),duration_s=float(x['dt'][mask].sum()),
            skin_pressure_difference_mean5_N=deltas.mean(0).tolist(),
            target_object_pressure_difference_mean5_N=targets.mean(0).tolist(),
            non_target_external_pressure_difference_mean5_N=(deltas-targets).mean(0).tolist(),
            skin_pressure_difference_integral5_N_s=(deltas*x['dt'][mask,None]).sum(0).tolist())
    rawmask=x['hold'];over1=(x['skin']>5.)&rawmask[:,None];over0=(b['skin']>5.)&rawmask[:,None]
    force_splits['overlimit_masks_identical']=bool(np.array_equal(over1,over0))
    peaks={}
    for method,r in raw.items():
        k=np.flatnonzero(r['hold'])[int(np.argmax(r['skin'][r['hold'],0]))]
        peaks[method]=dict(thumb_peak_N=float(r['skin'][k,0]),raw_start_label_s=float(r['t'][k]),raw_end_label_s=float(r['t'][k]+r['dt'][k]),
            before_first_intervention=bool(r['t'][k]+r['dt'][k]<9.54-1e-9))
    output=dict(schema='v10_hold_collision_servo_effort_association_v1',new_physics=False,script_sha256=sha(__file__),
        input_sha256={str(source):sha(source),str(derived):sha(derived),**sources},groups=groups,actuators21=actuator,
        active_200Hz_pressure=dict(mean_skin_difference5_N=ds[active].mean(0).tolist(),mean_target_pressure_difference5_N=dn[active].mean(0).tolist(),
            mean_non_target_pressure_difference5_N=dd[active].mean(0).tolist(),
            plan_end_positive_skin_difference_count5=(ds[ends]>1e-12).sum(0).tolist(),
            thumb_higher_pressure_but_lower_strain_proxy_end_count=int(((ds[ends,0]>1e-12)&(dp[ends,:5].sum(1)<-1e-12)).sum())),
        full_raw_pressure_split=force_splits,raw_peaks=peaks,
        interpretation='Computed actuator effort is a deterministic function of the same robot q/v/actual reference. Associations are useful state features but not independent force observations or an identified contact-force Jacobian. Post-divergence trajectories differ in contact geometry; no pressure causal gain is estimated.')
    with out.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# V10碰撞：伺服顺应性机制与可辨识边界','',
        '两条同seed610212的完整实际轨迹已固定；本报告使用控制前机器人25轴q/v、实际21执行器端口参考，调用冻结f067模块计算内部执行器量。没有假设未来均匀斜坡，没有新仿真，没有使用外物位姿/速度/实际质量/摩擦作为模型输入。目标物/非目标压力标签仅用于离线解释已保存结果。','',
        '## 已确认的时序与幅值','',
        '首次实际命令分歧在9.540s；控制前参考、机器人q/v和压力首次分歧均在9.545s。两法拇指原始峰值相同，均发生在首次干预之前。额外持握超限积分由0.802843588增至1.011945757 N·s（+26.04%），全部来自拇指。',
        f'超5N子步的二值时序是否完全相同：{force_splits["overlimit_masks_identical"]}。这不能将差异归咎于首次峰值或总超限时间增长；差异主要体现为同一事件中后续压力幅值。','',
        '两法各900个200Hz持握帧中，21执行器均无控制值裁剪、无驱动力裁剪。固定上限处的控制kink会另外记录，不能将“未被裁剪”误称尚有增加权限。','',
        '|分组|20窗口平均内部驱动力差|应变代理均值变化|解释限制|','|---|---:|---:|---|']
    for name,g in groups.items():
        lines.append(f'|{name}|{g["active_net_effort_difference_mean"]:.8g} {"N" if name=="lift" else "Nm"}|{100*g["strain_proxy_mean_relative_change"]:+.3f}%|有符号执行器坐标和，非接触力/闭合标量|')
    f=force_splits['twenty_plan_windows']
    lines+=['',
        f'20个实际plan窗的完整0.5ms记录中，拇指皮肤压力差均值为{f["skin_pressure_difference_mean5_N"][0]:+.6f}N，其中目标物接触差{f["target_object_pressure_difference_mean5_N"][0]:+.6f}N、非目标外接触差{f["non_target_external_pressure_difference_mean5_N"][0]:+.6f}N。几何回撤减少了障碍压力，同时增加了原目标接触压力，二者不能只用总法向距离一个符号解释。',
        f'按200Hz窗口终点，{output["active_200Hz_pressure"]["thumb_higher_pressure_but_lower_strain_proxy_end_count"]}/20个窗口出现“拇指压力更高但拇指应变代理更低”。FF应变代理也降低；LF代理增加，但本例额外材料超限并非LF。因此没有证据把全部额外拇指压力归因于主动提高拇指弹性驱动。','',
        '## 执行器顺应性特征能提供什么','',
        '可直接得到：每个实际传动坐标的目标−长度误差、kp误差驱动、kv速度阻尼、有限驱动力、控制/力限幅状态，以及实际驱动力与传动速度的模型功率。误差必须保留真实腱和关节坐标，不虚构24个独立q_des。',
        '本例可精确分解：Δeffort = kp·Δactual_target − kp·Δtransmission_length − kv·Δtransmission_velocity（此处无力饱和）。TH参考驱动差很小，驱动力差主要来自实际接触姿态改变；腕参考/实际运动及升降误差则显著改变。','',
        '未限幅代理0.5kp(error)^2不是实际储能：参考可主动移动，含偏置，系统有阻尼和力限幅。升降实际端口包含已知机器人重力补偿，参考偏置为0.02481276m、对应37.21914N；不能把该偏置误当外物接触载荷。JSON同时给出扣除该已知偏置的升降代理。','',
        '## 下一整体模型的证据要求','',
        '执行器顺应性值得纳入统一接触状态特征，因为它能区分目标变化、真实传动挠曲、阻尼与限幅；本例也反证了“位移退让/应变下降必然减少全部材料压力”。它没有单独提供力预测能力。',
        '下一步需要在同一时钟上联合：机器人传动误差/实际参考及速度、逐patch机器人局部位置与法向/归属假设、共享指级法向与切向、完整掌/手作用及接触切换。新障碍和原抓握接触应分别保留响应假设，同时全部皮肤受材料上限保护。',
        '只有在明确的接触模式/姿态/执行参考范围内，通过有限真实扰动实测“命令→传动挠曲→接触压力分配”的响应和误差，才可用于约束规划。现有robot K、几何J、代理和本例相关性均不能替代该识别；本报告未拟合压力梯度、未宣称安全或全域模型准入。','',
        '来源、21个执行器的参考/位置/阻尼分解、20个窗口完整结果与所有原始SHA见同名JSON及主伺服机制JSON/NPZ。']
    with out.with_suffix('.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(out=str(out),sha256=sha(out),active_raw_pressure_split=f,raw_peaks=peaks),ensure_ascii=False))

if __name__=='__main__':main()

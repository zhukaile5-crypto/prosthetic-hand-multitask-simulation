"""Create a scientific figure and local research report from numerical readout."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'review/.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

def main():
    source=ROOT/'review/contact_peak_fixed_input_sensitivity_v1.json';data=json.loads(source.read_text(encoding='utf-8'))
    directory=ROOT/'report/contact_peak_convergence_v1';directory.mkdir(exist_ok=False)
    methods=('ratio_effort','fixed_effort','fusion_effort');labels=('摩擦比对照','固定力对照','融合记录输入');colors=('#496F95','#8B969D','#B68A2F')
    rows={(x['method'],x['dt']):x for x in data['rows']};dts=(.0005,.00025,.000125)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12.8,5.4),sharey=True)
    fig.subplots_adjust(left=.07,right=.98,top=.77,bottom=.28,wspace=.22)
    fig.suptitle('相同实际输入下，碰撞与释放峰值如何随步长变化',x=.07,y=.965,ha='left',fontsize=18,fontweight='bold')
    fig.text(.07,.89,'一个旧V12碰撞条件 · 三法各3档步长 · 无新闭环决策、无接触参数调整',fontsize=11)
    for ax,method,label,color in zip(axes,methods,labels,colors):
        ax.set_title(label,fontsize=12,fontweight='bold',loc='left',color=color,pad=10)
        for phase,marker,style,offset in (('collision','o','-',.9),('release','s','--',-2.1)):
            values=[rows[(method,dt)]['metrics'][phase]['peak_N'] for dt in dts]
            ax.plot(range(3),values,color=color,marker=marker,linestyle=style,linewidth=2,markersize=6)
            for i,v in enumerate(values):ax.text(i,v+offset,f'{v:.2f}',ha='center',fontsize=10)
        ax.set_ylim(0,34);ax.set_xlim(-.25,2.25);ax.set_xticks(range(3),('0.5','0.25','0.125'));ax.set_xlabel('积分步长（ms）');ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('原始子步最高单指力（N）')
    fig.legend(handles=[Line2D([],[],color='#34444E',marker='o',label='碰撞窗 9.2–10.4 s'),Line2D([],[],color='#34444E',marker='s',linestyle='--',label='释放窗 12–16 s')],loc='lower left',bbox_to_anchor=(.062,.105),frameon=False,ncol=2,fontsize=11)
    fig.text(.07,.073,'碰撞最后两档差异约0.7%；释放含上游轨迹分歧，不将其全部解释为局部采样误差。',fontsize=10,color='#4B5B64')
    fig.text(.07,.027,'0.5 ms三条完整轨迹与旧数据逐字节一致；本图只描述数值敏感性，不是新的算法胜率比较。',fontsize=10,color='#4B5B64')
    for suffix in ('png','svg'):fig.savefig(directory/f'固定输入步长敏感性.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    provenance=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source=str(source.relative_to(ROOT)),methods=list(methods),timesteps_s=list(dts),plotted_peak_N={m:{phase:[rows[(m,dt)]['metrics'][phase]['peak_N'] for dt in dts] for phase in ('collision','release')} for m in methods},visual_inspection_pending=True,reader=str(Path(__file__).relative_to(ROOT)))
    (directory/'figure_data_and_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 接触峰数值验证与下一阶段方案','',
      '本轮完成三条0.5 ms完整物理桥接、六条0.25/0.125 ms固定实际输入试验。碰撞首峰在较细两档间仅变化约0.7%，接近29 N的大峰仍存在；释放结果更敏感，不能仅依据一档积分步长评价释放方案。没有在本轮运行新的闭环策略，也不新增算法胜率。','',
      '![固定输入步长敏感性](report/contact_peak_convergence_v1/固定输入步长敏感性.png)','',
      '## 试验边界','',
      '输入来自旧V12同一个胶囊体/collision/hold/617902条件，三种方法全部保留。固定原实际22路执行输入（21机器人端口及按原顺序更新的环境障碍端口）、环境、噪声抽样时钟、原接触参数和完整16秒时长；只改变积分步长和每个5 ms内的积分次数。源码/资产88项及42个原生二进制锁定。','',
      '先完成0.5 ms三条全程桥接。39个同名可比较字段逐字节一致；独立核查加上环境字段映射共40项，260项通过。包括共同传感、3201个边界和32000个子步力，不能仅用峰相同替代全轨迹复现。旧数据缺完整初始warmstart及旧子步q/v，不假称这些缺失字段也有逐字节对照。','',
      '六条细化均完整结束，同一锁、输入和RNG序列保留，数值警告均为0。首次细化采用了不兼容Windows并行的stdin入口，6个worker在运行前失败、物理积分为0；失败目录与ledger完整保留，随后标准模块入口的v2目录单独执行同一六条，不把失败启动当成算法失效或增加样本数。','',
      '## 原始子步结果','',
      '|原输入方法|步长 ms|碰撞峰 N|碰撞局部超限 N·s|释放峰 N|释放局部超限 N·s|','|---|---:|---:|---:|---:|---:|']
    for method,label in zip(methods,labels):
        for dt in dts:
            c=rows[(method,dt)]['metrics']['collision'];r=rows[(method,dt)]['metrics']['release']
            lines.append(f"|{label}|{dt*1000:g}|{c['peak_N']:.4f}|{c['local_excess_Ns']:.6f}|{r['peak_N']:.4f}|{r['local_excess_Ns']:.6f}|")
    lines += ['',
      '碰撞0.25→0.125 ms：三法峰值分别降低0.660%、0.660%、0.679%；超限积分分别降低1.54%、1.33%、1.29%。融合记录输入在三档中的碰撞最高峰均略高于两种对照，原反例保留。这仅是一个固定输入条件的描述，未事后设置收敛容差，也不构成全域接触真实性证明。','',
      '释放0.25→0.125 ms：摩擦比峰增加1.09%，固定力增加28.90%，融合记录输入增加4.04%；固定力超限积分增加约117%。融合记录输入的释放超限积分在三档均为0，但不因此宣称新闭环方案更优。细化从相同reset重放完整16秒，释放时状态包含之前接触引起的轨迹分歧；所以不能把这些差异全归为释放峰的一两个采样点，也不能把不同步长的状态当作同一个接触状态。','',
      '所有数据使用原子步START及END=START+duration，峰时按END展示。200Hz边界额外求解与积分缓存不是同一查询，不混池。qpos包含米、弧度、四元数等不同量，逐列最大差仅作定位，不解释成单一物理距离。','',
      '## 对整体重构的约束','',
      '1. 保留已有联合力目标带来的轻握、绝对波动与超限剂量收益。碰撞大峰没有随这两档细化消失，后续仍需研究接触保护与姿态/运动协同。触觉证据到达前的首峰不能靠事后反馈宣称已解决。','2. 统一持握、接触转换和释放的执行合同。现有持握使用八参考执行层，释放另走18指关节原生轨迹；下一阶段以唯一原生参考执行接口承接生命周期任务，释放仍即时撤销持握维持意图。','3. 明确区分执行接口统一与动作自由度扩张。先做命令等价的接口桥接，再分别检验holding软协同和release协同，最后整体比较；不把接口变化、取消准入或增加信息当成算法收益。','4. 放置支撑不能从握力降低直接认定。若加入place任务，必须有显式用户请求及公开的支撑/接近证据，并给所有对照同等信息；当前直接release仍按原任务立即执行。','5. 后续比较继续报告任务、原全程力限、峰/剂量、均力、漂移与反向波动、相对波动、跟踪、释放、计算成本。使用相同物理条件和共同可评价窗，保留失败与NULL，不事后选择有利权重。','',
      '## 证据入口','',
      '- [九条数值记录读数与27项步长配对](review/contact_peak_fixed_input_sensitivity_v1.json)',
      '- [三条全轨迹桥接独立验收](review/contact_peak_convergence_bridge_independent_v1.json)',
      '- [完整六条细化终态](output/contact_peak_convergence_refined_v2/execution_complete.json)',
      '- [首次启动失败及零物理证据](review/contact_peak_refine_launcher_incident_v1.json)',
      '- [既有完整48组算法比较](V12执行一致性修订与48组复验_20261105.md)','']
    path=ROOT/'接触峰数值验证与下一阶段方案_20261105.md';assert not path.exists();path.write_text('\n'.join(lines),encoding='utf-8')
    print('Numerical report, PNG, SVG and figure provenance saved')
if __name__=='__main__':main()

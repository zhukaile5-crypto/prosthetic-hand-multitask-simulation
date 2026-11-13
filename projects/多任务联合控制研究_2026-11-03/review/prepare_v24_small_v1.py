"""Register fixed nine canonical-task runs after final actual software gates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
VARIANTS=('v23_joint','canonical_backend','canonical_joint')
OLD_VARIANTS=('v21','backend','joint')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def prepare(args):
    from v24_canonical_task_fusion.runtime import source_identity
    from v22_joint_task_fusion.numerics import numerical_identity
    from v13_a1.run_bridge import environment
    identity=source_identity(['ratio_effort']);num=numerical_identity();env=environment()
    assert len(identity)==192,'Actual complete192 source closure required'
    parents={v:read(ROOT/f'review/{v}_small_plan_v1.json') for v in ('v22','v23')}
    original=read(ROOT/'review/v21_complete16_plan_v1.json')
    source_refs={v:p['source_sha256'] for v,p in parents.items()};source_refs['v21']=original['source_sha256']
    assert [len(source_refs[v]) for v in ('v21','v22','v23')]==[170,179,186]
    for version,pins in source_refs.items():
        assert all(identity.get(k)==digest for k,digest in pins.items()),'Frozen '+version+' sources changed'
    assert env==original['environment']==parents['v22']['environment']==parents['v23']['environment'],'Original42 physical environment changed'
    assert num==parents['v23']['numerical_identity'],'Existing numerical identity changed'
    gates={}
    for label,relative in [('author',args.author_gate),('independent',args.math_gate),('integration',args.integration_gate),('public_execution',args.public_gate)]:
        path=ROOT/relative;gate=read(path)
        assert gate.get('passed') is True and gate.get('physics_steps')==0,(label,'Actual passed gate required')
        pins=gate.get('production_source_sha256')
        assert isinstance(pins,dict) and pins and all(identity.get(k)==v for k,v in pins.items()),(label,'Final production sources differ')
        complete_pins=gate.get('source_sha256',pins)
        if label!='author':
            assert len(complete_pins)==192 and complete_pins==identity,(label,'Complete actual192 gate required')
            assert all(complete_pins.get(k)==v for k,v in pins.items()),(label,'Production subset differs from actual full closure')
        assert gate.get('numerical_identity')==num,(label,'Gate numerical identity missing/different')
        gates[label]=dict(path=relative,sha256=sha(path),checked_sources=pins,complete_checked_sources=complete_pins)
    assert read(ROOT/args.public_gate).get('public_cases_assessed')==6,'Actual six public cases required'
    kernel=read(ROOT/'review/v23_finite81_author_v1.json')
    assert kernel.get('passed') is True
    for name,count in [('original65',65),('joint16',16),('all81',81)]:
        assert all(kernel.get('summaries',{}).get(name,{}).get(k)==count for k in ('assigned','assessed','available','independent_original_KKT','strict_box'))
    reader_gate=read(ROOT/'review/v24_small_readout_software_v1.json')
    assert reader_gate['passed'] and reader_gate['reader_sha256']==sha(ROOT/'review/read_v24_small_readout_v1.py')
    dependency_names=[
        'review/delivery_auxiliary_metrics_v3.py','review/delivery_auxiliary_contract_v3.json',
        'review/read_v18_small_auxiliary_v1.py','review/read_v21_release_variation_v1.py',
        'review/read_v24_small_readout_v1.py','review/v24_small_readout_software_v1.json',
        'review/verify_v24_v23_full_trace_bridges_v1.py','review/v18_passive_trace_comparison_v3.py',
        'review/V24规范参照多任务融合与同条件验证合同_v1.md',
        'review/v17b_delivery_auxiliary_readout_v1.json','review/v21_complete16_auxiliary_v1.json',
        'review/v22_small_plan_v1.json','review/v22_small_readout_v1.json',
        'review/v23_small_plan_v1.json','review/v23_small_readout_v1.json','review/v23_small_execution_v1.json',
        'review/v23_finite81_author_v1.json',
        'review/v23_joint_friction_first_divergence_v1.json','review/v23_canonical_normal_witness_probe_v1.json']
    dependencies={name:sha(ROOT/name) for name in dependency_names}
    assert dependencies['review/delivery_auxiliary_metrics_v3.py']=='0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55'
    assert dependencies['review/delivery_auxiliary_contract_v3.json']=='8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979'
    historical={}
    for version in ('v22','v23'):
        report=read(ROOT/f'review/{version}_small_readout_v1.json');parent=parents[version]
        assert report.get('complete') is True and report.get('issues')==[] and report.get('assigned')==9
        assert report['plan_sha256']==sha(ROOT/f'review/{version}_small_plan_v1.json')
        keys=[(r['variant'],*r['context_key']) for r in report['records']]
        assert len(keys)==len(set(keys))==9 and set(keys)=={(v,*c) for v in OLD_VARIANTS for c in CONTEXTS}
        refs=[]
        for row in report['records']:
            run=Path(row['run']);meta=read(run/'metadata.json');assert meta['research_source_sha256']==parent['source_sha256']
            for name,digest in row['input_sha256'].items():assert sha(run/name)==digest,'Historical input changed'
            refs.append(dict(variant=row['variant'],context_key=row['context_key'],run=str(run),input_sha256=row['input_sha256']))
        historical[version]=refs
    audit=read(ROOT/'review/v23_small_execution_v1.json')
    assert audit['passed'] and not audit['issues'] and audit['all_assigned_checked'] and len(audit['checks'])==127
    assert audit['production_source_sha256']==source_refs['v23']
    bridge_refs=[]
    for row in historical['v23']:
        if row['variant']!='joint':continue
        run=Path(row['run'])
        bridge_refs.append(dict(context_key=row['context_key'],run=str(run),input_sha256={n:sha(run/n) for n in ('trace.npz','metadata.json','result.json','release_decisions.jsonl.gz')}))
    assert len(bridge_refs)==3
    planpath=ROOT/'review/v24_small_plan_v1.json';assert not planpath.exists()
    for variant in VARIANTS:
        for scene,motion,seed in CONTEXTS:
            name=f'{variant}_{scene}_{motion}_{seed}'
            assert not (ROOT/f'output/v24_small_{name}_v1').exists()
            assert not (ROOT/f'review/v24_small_{name}_registration_v1.json').exists()
    stages=[]
    for variant in VARIANTS:
        for scene,motion,seed in CONTEXTS:
            name=f'{variant}_{scene}_{motion}_{seed}';output=f'output/v24_small_{name}_v1';registration=f'review/v24_small_{name}_registration_v1.json'
            assignment=dict(methods=['ratio_effort'],scenarios=[scene],seeds=[seed],motion_mode=motion,physical_randomization=True,task_policy='integrated',receiver_policy='coordinated',release_variant=variant)
            reg=dict(schema='v24_canonical_task_registration_v1',source_sha256=identity,environment=env,numerical_identity=num,
                assignment=assignment,software_gate_passed=True,software_gate_inputs=gates,dependencies=dependencies,
                phase='seen_development',no_parameter_sweep=True,performance_claim_allowed=False)
            write(ROOT/registration,reg)
            command=['-B','-m','v24_canonical_task_fusion.run_development','--out',output,'--registration',registration,
                '--release-variant',variant,'--task-policy','integrated','--receiver-policy','coordinated','--methods','ratio_effort',
                '--scenarios',scene,'--seeds',str(seed),'--motion-mode',motion,'--physical-randomization','--workers','1']
            stages.append(dict(name=name,variant=variant,context_key=[scene,motion,seed],planned=1,assignment=assignment,
                output=output,registration=registration,registration_sha256=sha(ROOT/registration),python_arguments=command))
    plan=dict(schema='v24_canonical_three_arm_small_plan_v1',planned=9,stages=stages,source_sha256=identity,
        environment=env,numerical_identity=num,software_gates=gates,dependencies=dependencies,
        reference_sources=source_refs,historical=historical,original_v23_joint_bridges=bridge_refs,
        preparer_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat(),variants=list(VARIANTS),contexts=[list(k) for k in CONTEXTS],
        dispatch_order=['v23_joint3 then exact fullV23 bridges','canonical_backend3','canonical_joint3'],
        results_all9_terminal_before_read=True,
        expansion_gate=dict(task=3,clearance=3,ever_floor=0,guard_at_least=2,final_low_motion_at_least=1),
        physics_dispatched=False,scope='Three seen contexts/two seed clusters; historical references clearly separate; unchangedv3/TV, no comprehensive superiority claim')
    write(planpath,plan)
    print(json.dumps(dict(plan=str(planpath),planned=9,source_count=len(identity),physics_dispatched=False),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true',required=True)
    for name in ('author-gate','math-gate','integration-gate','public-gate'):ap.add_argument('--'+name,required=True)
    prepare(ap.parse_args())

"""Register nine same-environment V22 runs only after actual software gates."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
VARIANTS=('v21','backend','joint')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def prepare(args):
    from v22_joint_task_fusion.runtime import source_identity
    from v22_joint_task_fusion.numerics import numerical_identity
    from v13_a1.run_bridge import environment
    identity=source_identity(['ratio_effort']);env=environment();num=numerical_identity()
    assert len(identity)==179,'Require the reviewed complete179 source closure.'
    old=read(ROOT/'review/v21_complete16_plan_v1.json')
    assert all(identity[k]==v for k,v in old['source_sha256'].items()),'Frozen original170 changed.'
    assert env==old['environment'],'Original42 physical environment differs.'
    gatepins={}
    for role,relative in [('kernel',args.kernel_gate),('math',args.math_gate),('integration',args.integration_gate),('public_execution',args.public_gate)]:
        p=ROOT/relative;g=read(p)
        assert g.get('passed') is True and g.get('physics_steps')==0,(role,'actual passed software gate required')
        pins=g.get('production_source_sha256')
        assert isinstance(pins,dict) and pins,(role,'actual production-source pins missing')
        assert all(k in identity and identity[k]==v for k,v in pins.items()),(role,'final production sources changed')
        if 'numerical_identity' in g:assert g['numerical_identity']==num,(role,'numerical identity changed')
        gatepins[role]={'path':relative,'sha256':sha(p),'checked_sources':pins}
    assert len(read(ROOT/args.integration_gate)['production_source_sha256'])==179,'Actual complete integration closure required.'
    kernel=read(ROOT/args.kernel_gate)
    assert kernel.get('corpus_assessed')==kernel.get('corpus_available')==kernel.get('corpus_original_KKT')==65,'Retain the complete65 numerical capability gate.'
    assert read(ROOT/args.public_gate).get('public_cases_assessed')==6,'Both new formulations across positive/P0/unknown require actual public execution.'
    reader_gate=read(ROOT/'review/v22_small_readout_software_v3.json')
    assert reader_gate['passed'] and reader_gate['reader_sha256']==sha(ROOT/'review/read_v22_small_readout_v1.py'),'Final all-nine reader protection gate required.'
    dependencies={n:sha(ROOT/n) for n in ['review/delivery_auxiliary_metrics_v3.py','review/delivery_auxiliary_contract_v3.json',
        'review/read_v18_small_auxiliary_v1.py','review/read_v21_release_variation_v1.py','review/read_v22_small_readout_v1.py',
        'review/v17b_delivery_auxiliary_readout_v1.json','review/v21_complete16_auxiliary_v1.json',
        'review/V22联合任务闭环实施与小规模对照合同_v1.md',
        'review/v18_passive_trace_comparison_v3.py','review/verify_v22_v21_full_trace_bridges_v1.py',
        'review/v22_small_readout_software_v3.json']}
    assert dependencies['review/delivery_auxiliary_metrics_v3.py']=='0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55'
    assert dependencies['review/delivery_auxiliary_contract_v3.json']=='8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979'
    planpath=ROOT/'review/v22_small_plan_v1.json';assert not planpath.exists()
    stages=[]
    for variant in VARIANTS:
        for scene,motion,seed in CONTEXTS:
            name=f'{variant}_{scene}_{motion}_{seed}'
            output=f'output/v22_small_{name}_v1';registration=f'review/v22_small_{name}_registration_v1.json'
            assert not (ROOT/output).exists() and not (ROOT/registration).exists()
            assignment=dict(methods=['ratio_effort'],scenarios=[scene],seeds=[seed],motion_mode=motion,
                physical_randomization=True,task_policy='integrated',receiver_policy='coordinated',release_variant=variant)
            reg=dict(schema='v22_joint_task_registration_v1',source_sha256=identity,environment=env,numerical_identity=num,
                assignment=assignment,software_gate_passed=True,software_gate_inputs=gatepins,dependencies=dependencies,
                phase='seen_development',no_parameter_sweep=True,performance_claim_allowed=False)
            write(ROOT/registration,reg)
            cmd=['-B','-m','v22_joint_task_fusion.run_development','--out',output,'--registration',registration,
                '--release-variant',variant,'--task-policy','integrated','--receiver-policy','coordinated',
                '--methods','ratio_effort','--scenarios',scene,'--seeds',str(seed),'--motion-mode',motion,
                '--physical-randomization','--workers','1']
            stages.append(dict(name=name,variant=variant,context_key=[scene,motion,seed],planned=1,assignment=assignment,
                output=output,registration=registration,registration_sha256=sha(ROOT/registration),python_arguments=cmd))
    original={tuple(r['context_key']):r for r in read(ROOT/'review/v21_complete16_auxiliary_v1.json')['records']}
    refs=[]
    for key in CONTEXTS:
        r=original[key];run=Path(r['run'])
        refs.append(dict(context_key=list(key),run=str(run),input_sha256={n:sha(run/n) for n in ['trace.npz','metadata.json','result.json','release_decisions.jsonl.gz']}))
    plan=dict(schema='v22_three_arm_small_plan_v1',planned=9,stages=stages,source_sha256=identity,environment=env,
        preparer_sha256=sha(__file__),
        numerical_identity=num,software_gates=gatepins,dependencies=dependencies,original_v21_bridges=refs,
        created_utc=datetime.now(timezone.utc).isoformat(),variants=list(VARIANTS),contexts=[list(k) for k in CONTEXTS],
        dispatch_order=['v21 complete3 exact bridges','backend3','joint3'],results_all9_terminal_before_read=True,
        expansion_gate=dict(task=3,clearance=3,ever_floor=0,guard_at_least=2,final_low_motion_at_least=1),
        physics_dispatched=False,scope='Three seen contexts / two seed clusters; all9 new, unchangedv3 and releaseTV, no comprehensive superiority claim.')
    write(planpath,plan)
    print(json.dumps({'plan':str(planpath),'planned':9,'source_count':len(identity),'physics_dispatched':False},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--kernel-gate',required=True);ap.add_argument('--math-gate',required=True)
    ap.add_argument('--integration-gate',required=True);ap.add_argument('--public-gate',required=True);prepare(ap.parse_args())

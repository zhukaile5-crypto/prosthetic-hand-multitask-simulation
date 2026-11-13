"""Freeze four finite V18 development assignments after software acceptance.

Preparation does not run physics. The three seen mechanism cases are not an
unseen confirmation set. Every stage gets its own fresh-output runner receipt.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v18_effort_transfer.runtime import source_identity
from v13_a1.run_bridge import environment


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration-gate',required=True)
    args=parser.parse_args()
    identity=source_identity(['ratio_effort']);env=environment()
    old=read(ROOT/'review/v17b_prospective80_plan_v1.json')
    if len(old['source_sha256'])!=136 or any(identity.get(k)!=v for k,v in old['source_sha256'].items()):
        raise ValueError('original executed136 must remain exact')
    if old['environment']!=env:raise ValueError('physical native environment changed')
    gate_path=(ROOT/args.integration_gate).resolve();gate=read(gate_path)
    if not gate.get('passed') or gate.get('source_sha256')!=identity:
        raise ValueError('integration PASS must bind the entire actual current source map')
    gates=[gate_path,ROOT/'review/v18_effort_task_independent_attempt1.json',ROOT/'review/v18_governor_independent_v1.json']
    if any(not read(p).get('passed') for p in gates):raise ValueError('required cross-check not passed')
    if read(gates[1])['source_sha256']!={'v18_effort_transfer/effort_task.py':identity['v18_effort_transfer/effort_task.py']}:
        raise ValueError('effort task changed after cross-check')
    if read(gates[2])['source_sha256']!=identity['v18_effort_transfer/governor.py']:
        raise ValueError('governor changed after cross-check')
    descriptors=[
        ('passive_ratio_transport','nominal','transport',619741,'legacy','blind','baselines_transport'),
        ('nominal_transport','nominal','transport',619741,'integrated','coordinated','full_V17_transport'),
        ('fragile_hold','fragile_conflict','hold',619741,'integrated','coordinated','full_V17_hold'),
        ('friction_transport','friction_drop','transport',619742,'integrated','coordinated','full_V17_transport'),
    ]
    stages=[]
    for name,scene,motion,seed,task,receiver,old_stage in descriptors:
        output='output/v18_small_'+name+'_v1'
        registration='review/v18_small_'+name+'_registration_v1.json'
        if (ROOT/output).exists() or (ROOT/registration).exists():
            raise FileExistsError('all attempts retained; refuse existing '+name)
        source_stage=next(x for x in old['stages'] if x['name']==old_stage)
        original=ROOT/source_stage['output']/'runs'/f'{scene}__ratio_effort__{seed}'
        if sha(original/'trace.npz')!=read(original/'metadata.json')['trace_sha256']:
            raise ValueError('same-condition original trace identity failed')
        assignment=dict(methods=['ratio_effort'],scenarios=[scene],seeds=[seed],motion_mode=motion,
            physical_randomization=True,task_policy=task,receiver_policy=receiver)
        reg=dict(schema='v18_seen_mechanism_small_registration_v1',created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=identity,environment=env,assignment=assignment,software_gate_passed=True,
            software_gates={str(p.relative_to(ROOT)):sha(p) for p in gates},
            scope='seen development integration and mechanism evidence, no global superiority claim',
            original_same_condition=str(original.relative_to(ROOT)),original_trace_sha256=sha(original/'trace.npz'))
        write(ROOT/registration,reg)
        stages.append(dict(name=name,output=output,registration=registration,assignment=assignment,
            original_same_condition=reg['original_same_condition'],original_trace_sha256=reg['original_trace_sha256'],
            planned=1))
    plan=dict(schema='v18_seen_mechanism_small_plan_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=identity,environment=env,parent136_unchanged=True,planned=4,stages=stages,
        integration_gate=str(gate_path.relative_to(ROOT)),software_gates={str(p.relative_to(ROOT)):sha(p) for p in gates},
        implementation_contract='review/V18统一努力接触转换实施合同_v1.md',
        implementation_contract_sha256=sha(ROOT/'review/V18统一努力接触转换实施合同_v1.md'),
        execution_order='passive16s exact bridge first; then three separate frozen assignments',
        evaluation='unchanged original metrics plus frozen v3 auxiliary; all failures and missing retained',
        no_parameter_sweep=True,performance_claim_allowed=False)
    write(ROOT/'review/v18_small_plan_v1.json',plan)
    print(json.dumps(dict(registered=4,sources=len(identity),physics_dispatched=False),ensure_ascii=False))


if __name__=='__main__':main()

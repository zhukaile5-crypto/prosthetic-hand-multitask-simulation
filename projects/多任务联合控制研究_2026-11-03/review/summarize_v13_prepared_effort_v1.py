"""Collect existing zero-physics ledgers; no evaluator or controller run."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    paths=['review/prepared_native_effort21_cache_author_attempt1.json','review/v13_prepared_native_cache_independent_v1.json',
        'review/v13_prepared_cache_only_hand_foundation_v1.json','review/prepared_value_effort21_author_v1.json',
        'review/v13_prepared_value_independent_v1.json','review/v13_prepared_value_hand_foundation_v1.json','review/v13_joint_public_packets_independent_v1.json']
    ledgers=[]
    for rel in paths:
        data=json.loads((ROOT/rel).read_text(encoding='utf-8'))
        if not data['passed']:raise ValueError('Evidence failed: '+rel)
        ledgers.append(dict(path=rel,sha256=sha(ROOT/rel),passed=True,checks=len(data['checks']),schema=data['schema']))
    data=json.loads((ROOT/paths[3]).read_text(encoding='utf-8'));profiles=data['profiles'];timing={}
    for k in ('build_ms','solve_ms','total_ms'):
        base=np.array([x['timings']['base'][k] for x in profiles]);value=np.array([x['timings']['prepared'][k] for x in profiles])
        timing[k]=dict(base_median=float(np.median(base)),value_median=float(np.median(value)),base_range=[float(min(base)),float(max(base))],
            value_range=[float(min(value)),float(max(value))],paired_reduction_median_pct=float(np.median((base-value)/base*100)))
    sources={p:sha(ROOT/p) for p in ('v13_joint/native_effort21.py','v13_joint/prepared_native_effort21.py','v13_joint/prepared_value_effort21.py','v11/robot_servo_effort.py','v13_a1/codec.py')}
    assert sources['v13_joint/native_effort21.py']=='6f097f88b19a80b8692be6d8a937c7a640d2ad6fe17a12d2f4d87b48c7e148f3'
    assert sources['v13_joint/prepared_native_effort21.py']=='3ae5efa293ad7f7987e07c51722c94943a3549bcd11cf639ea13366abd4d151b'
    assert sources['v13_joint/prepared_value_effort21.py']=='5b95ec9f1614415ec75b91ecb02629f4316d844960b6111932ffb8d1c76f81ac'
    delivery=dict(schema='v13_prepared_effort21_delivery_v1',software_checks_passed=True,source_sha256=sources,evidence=ledgers,
        export=dict(module='v13_joint.prepared_value_effort21',class_name='PreparedValueEffort21',constructor='PreparedValueEffort21(model)',
            build_domain='Same keyword API and EffortDomain21 as frozen NativeEffort21',solve='Same keyword API and NativeInverseResult as frozen NativeEffort21',
            new_diagnostic='diagnostics.prepared_value_final_full_recheck',optional_cache_diagnostics='cache_diagnostics()'),
        arithmetic_contract=['Original finite candidate family, scales multiply before squaring, fixed raw p0 and original requested d21; no target minus p0 substitution.',
            'First q/v snapshot and final combination use unchanged F067 full evaluation. Other candidates preserve g*u+b0+b1*length+b2*speed sequence.',
            'Intermediate J/K/D are NOT_COMPUTED_VALUE_ONLY, final derivative/clip diagnostic comes from original full evaluation.',
            'Final full evaluation independently rechecks native value, actual delta, residual, objective and transmission bytes; mismatch yields no command.',
            'No exact-zero shortcut, new PD, new candidate family, engine writer, contact-pressure prediction or physical-safety certificate.'],
        stronger_guard='Prepared fingerprints include actuator_group and F067 admission modes. Base6f missing-group guard counterexample is retained; standard frozen model numerical behavior agrees.',
        isolated_inverse_profile=dict(public_states=27,saved_run_archives=3,workloads_per_state=2,timing=timing,
            scope='One-host serial cold calls, base before prepared; all54 retained. Archive IO/imports/model compile/source hash excluded. Not independent physical trials or a hardware realtime bound.',
            meets_20ms_budget=False),
        controller_packet_scope='Nine off-policy real public packet calls (holding/transport/release x3methods), not closed-loop trajectories.7 QPs converged; nominal fixed/ratio2 nonconvergence kept honest direct-feedback fallback, native all9 verified. Private robot FK counted; integration/contact/writer0.',
        failed_prospective_optimization=dict(path='review/native_scalar_vector_clip_oracle_v1.json',sha256=sha(ROOT/'review/native_scalar_vector_clip_oracle_v1.json'),
            outcome='Direct vector np.clip changes signedzero bytes on12 boundary groups; NOT applied. Both released modules retain original scalar clip behavior.'),
        limitations=['Python source/method replacement is guarded by outer source freeze, not by the internal model fingerprint.',
            'Only private source-bound robot actuation at fixed public q/v is computed. No object state, actual mass/friction, future force, closed-loop advantage or global all-float optimum.',
            'Some controller packet QPs did not converge; fallback legality is software evidence, not successful physical execution.'],
        physics_steps=0,writer_calls=0,active_jobs_remaining=0,summary_script_sha256=sha(Path(__file__)))
    out=ROOT/'review/v13_prepared_effort21_delivery_v1.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(delivery,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(path=str(out),sha256=sha(out),timing=timing),ensure_ascii=False))
if __name__=='__main__':main()

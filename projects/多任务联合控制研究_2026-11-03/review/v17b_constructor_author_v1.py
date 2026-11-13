"""Zero-tick construction and runner-clone checks for logging-only version."""
from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from v17b_full_evidence import runtime as rt
    from v17_contact_transfer.runtime import source_identity as parent_identity
    before=rt.source_identity(rt.METHODS);checks=[]
    def ck(n,v,d=None):checks.append(dict(name=n,passed=bool(v),detail=d))
    ck('136_sources_parent133_exact',len(before)==136 and all(before[k]==v for k,v in parent_identity(rt.METHODS).items()))
    root=(ROOT/'v17_contact_transfer/run_development.py').read_text(encoding='utf-8')
    child=(ROOT/'v17b_full_evidence/run_development.py').read_text(encoding='utf-8')
    expected=root.replace('from v17_contact_transfer.runtime import','from v17b_full_evidence.runtime import')
    expected=expected.replace('V17ContactTransfer','V17bEvidenceOnly').replace('v17_contact_task','v17b_complete_motion_evidence')
    expected=expected.replace("entry_module='v17_contact_transfer.run_development'","entry_module='v17b_full_evidence.run_development'")
    ck('runner_literal_namespace_and_schema_only',child==expected)
    for method,task,receiver in [('ratio_effort','integrated','coordinated'),('ratio_effort','integrated','blind'),
        ('ratio_effort','legacy','blind'),('fixed_effort','legacy','blind'),('fusion_effort','legacy','blind')]:
        s=rt.ReleaseSession(method=method,scenario='nominal',seed=619732,split='robustness',motion_mode='transport',task_policy=task,receiver_policy=receiver)
        label=method+'/'+task+'/'+receiver
        ck(label+'/zero_tick_no_sensor_acquisition',s.time==0. and s.tick==0 and not s.trace and s.sensors.samples==0)
        ck(label+'/full_source_and_metadata',s.research_sources==before and s.metadata()['decision_algorithm']=='unchanged_V17_contact_transfer')
    ck('all136_sources_unchanged',before==rt.source_identity(rt.METHODS))
    result=dict(schema='v17b_constructor_author_v1',passed=all(c['passed'] for c in checks),checks=checks,
        source_sha256=before,scope=dict(new_physical_trials=0,zero_tick_constructions=5,
            initialization_forward_is_not_zero_forward_claim=True,algorithm_change=False))
    out=ROOT/'review/v17b_constructor_author_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),failed=[c['name'] for c in checks if not c['passed']])))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

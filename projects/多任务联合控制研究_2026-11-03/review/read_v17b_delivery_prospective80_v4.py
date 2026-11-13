"""Explicit receipt-chain entrypoint around frozen compatible auxiliary reader."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).parent))
import read_v17b_delivery_base_v3 as base
from v17b_completion_chain_reader_v1 import validate_chain

BASE_SHA='f9966f5c23fafc79031bae0572d353945ebdaa4495ff903476cfed3490f72120'

def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--completion-chain',required=True);p.add_argument('--out');args=p.parse_args()
    if base.sha(Path(base.__file__))!=BASE_SHA:raise ValueError('Reviewed scenario-normalized base changed')
    plan=base.pinned_plan();assigned=base.roster(plan)
    error=None
    try:state=validate_chain(plan,args.completion_chain,root=base.ROOT)
    except (ValueError,KeyError,OSError,TypeError) as exc:state=dict(status='invalid_completion_chain',ready=False);error=type(exc).__name__+': '+str(exc)
    report=dict(schema='v17b_delivery_auxiliary_prospective80_v1',reader_revision='v4_scenario_name_and_explicit_completion_receipt_chain',
        created_utc=datetime.now(timezone.utc).isoformat(),status=state['status'],complete=False,readiness=state,
        completion_chain_provenance=state,completion_chain_error=error,plan_sha256=base.PLAN_SHA,
        calculator_sha256=base.CALC_SHA,contract_sha256=base.CONTRACT_SHA,execution_source_sha256=plan['source_sha256'],
        reader_sha256=base.sha(Path(__file__)),chain_validator_sha256=base.sha(Path(__file__).with_name('v17b_completion_chain_reader_v1.py')),
        reused_reader_base_sha256=BASE_SHA,retained_prior_base_v2_sha256='4b624af2e43c708e74ab346c86f64a3669d0475fbcf635c23ac370b81865c693',planned=80,role_counts=dict(Counter(r['role'] for r in assigned)),metrics_read=False,
        physics_steps=0,production_source_modified=False,
        scope=dict(physical_seed_clusters=2,contexts=16,independent_n80_claim=False,prospective_definitions=True,
            prospective_development_not_global_confirmation=True,all_assigned=True,weighted_total_score=False,incomplete_subset_selection=False))
    if args.execute and state['ready']:
        records,issues=base.collect(plan,assigned)
        report.update(complete=True,status='computed_complete80',records=records,metrics_read=True,source_issues=issues,
            source_identity_passed=base.source_identity_ok(records,issues),record_status_counts=dict(Counter(r['status'] for r in records)),
            record_error_count=sum('error' in r for r in records),summaries=base.grouped(records),pairs=base.paired_records(records))
    if args.out:
        out=base.ROOT/args.out
        if out.exists():raise FileExistsError(out)
        out.write_text(json.dumps(base.aux._plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(path=str(out),status=report['status'],records=len(report.get('records',[])),metrics_read=report['metrics_read'],error=error,sha256=base.sha(out))))
    else:print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if error else 0 if args.inspect or report['complete'] else 3
if __name__=='__main__':raise SystemExit(main())

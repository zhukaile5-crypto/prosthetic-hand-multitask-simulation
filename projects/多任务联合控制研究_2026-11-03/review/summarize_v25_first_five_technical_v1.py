"""Post-readout public-clock/mask and actual-command summary; no fitting."""
from pathlib import Path
from collections import Counter
import json,gzip,hashlib,re,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from review.read_v25_first_five_response_v1 import observed_rank
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    out=ROOT/'review/v25_first_five_technical_summary_v1.json'
    if out.exists():raise FileExistsError('Fresh summary required')
    start=time.perf_counter();path=ROOT/'review/v25_first_five_response_v1.json';j=read(path)
    assert j['complete'] and j['issues']==[]
    planpath=ROOT/'review/v25_first_five_plan_v2.json';assert sha(planpath)==j['plan_sha256']
    plan=read(planpath);origin=j['origin_public']['time_s'];folder=ROOT/plan['output']
    summary={};supported_inputs=[]
    for arm,b in j['branches'].items():
        p=folder/'branches'/arm/'response_observations.jsonl.gz';assert sha(p)==b['input_sha256'][p.name]
        states=[];counts=Counter();last=None;n=0;delays=[]
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                m=re.search(r'"acquisition_time_s"\s*:\s*([0-9.eE+\-]+)',line)
                assert m is not None
                acq=float(m[1])
                if not origin-1e-9<=acq<=origin+.2+1e-9:continue
                x=json.loads(line);assert abs(x['acquisition_time_s']-acq)<1e-9
                n+=1;delays.append(x['available_at_s']-acq)
                signature=tuple(tuple(d.get(k) for k in ('sensor_valid','distributed_valid','contact_present','instrument_valid','status')) for d in x['digits'])
                for d in x['digits']:counts[(d['digit'],d['status'])]+=1
                if signature!=last:
                    states.append(dict(acquisition_time_s=acq,available_at_s=x['available_at_s'],
                        retained_instrument_digits=[d['digit'] for d in x['digits'] if d['instrument_valid'] is True],
                        observed_projection_digits=[d['digit'] for d in x['digits'] if d['external_projection2_m_s'] is not None],
                        digit_statuses=[d['status'] for d in x['digits']],
                        unknown_reasons={str(d['digit']):d['reasons'] for d in x['digits'] if d['status']=='UNKNOWN_PROJECTION'}))
                    last=signature
        phase=[];pp=folder/'branches'/arm/'release_decisions.jsonl.gz';assert sha(pp)==b['input_sha256'][pp.name]
        with gzip.open(pp,'rt',encoding='utf-8') as f:
            for line in f:
                m=re.search(r'"time_s"\s*:\s*([0-9.eE+\-]+)',line);assert m is not None
                t=float(m[1])
                if origin-1e-9<=t<=origin+.215+1e-9:
                    x=json.loads(line);phase.append((x['time_s'],x['phase']))
        hold=[x for x in b['program'] if origin-1e-9<=x['time_s']<origin+.2-1e-9]
        actual=np.asarray([x['actual_receipt']['actual_reference21'] for x in hold],float)
        carrier=np.asarray(b['program'][0]['carrier_at_origin21'],float)
        summaries=dict(terminal=b['assignment']['terminal_probe_status'],unsupported=b['unsupported'],
            first_actual_difference_from_zero21=b['first_actual_executed_difference_from_zero21'],
            max_abs_actual_difference_from_zero21=b['max_abs_actual_executed_difference_from_zero21'],
            hold_actual_reference_min21=actual.min(0).tolist(),hold_actual_reference_max21=actual.max(0).tolist(),
            hold_max_abs_actual_delta_from_origin_carrier21=np.abs(actual-carrier).max(0).tolist(),
            actual_issue_modes=sorted(set((x['mode'],x['mode_generation']) for x in b['actual_receipt_input_rows'])),
            acquisition_packets_0_to_200ms=n,actual_delay_min_s=min(delays),actual_delay_max_s=max(delays),
            per_digit_status_counts=[dict(digit=d,status=s,count=c) for (d,s),c in sorted(counts.items())],
            mask_change_count=max(0,len(states)-1),mask_states_including_initial=states,
            all_saved_probe_phase_counts=dict(Counter(x[1] for x in phase)),
            observed_digits_at_fixed_horizons={h:[d['digit'] for d in v.get('digits',[]) if d['external_projection2_m_s'] is not None] for h,v in b['labels'].items() if float(h)>0})
        summary[arm]=summaries
        if arm!='zero' and not b['unsupported']:
            supported_inputs.extend(x['actual_executed_difference_from_zero21'] for x in b['actual_receipt_input_rows'])
    paired=[d for r in j['paired_device_labels'] for d in r['digits']]
    result=dict(complete=True,issues=[],physics_steps=0,model_fits=0,
        readout_sha256=sha(path),plan_sha256=sha(planpath),script_sha256=sha(__file__),
        source_sha256=j['source_sha256'],origin_public=j['origin_public'],arms=summary,
        actual_program_all_assigned_rank=j['actual_program_support'],
        supported_direction_program_rank=observed_rank(supported_inputs),
        paired_digit_assignments=len(paired),paired_digit_available=sum(d['available'] for d in paired),
        equal_basis_and_world_projection=sum(d.get('acquisition_basis_equal',False) and d.get('world_projection_equal',False) for d in paired if d['available']),
        conditional_common_functions_available=sum(d.get('conditional_common_function',{}).get('available',False) for d in paired),
        wall_seconds=time.perf_counter()-start,
        scope='No model fit; public full-rate mask timeline and actual receipt footprint only. Unsupported adaptive arm is preserved separately; observed rank is not full21 causal support.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),supported_rank=result['supported_direction_program_rank']['rank'],
        mask_changes={a:b['mask_change_count'] for a,b in summary.items()},
        phase_counts={a:b['all_saved_probe_phase_counts'] for a,b in summary.items()},
        wall_seconds=result['wall_seconds']),ensure_ascii=False))
if __name__=='__main__':main()

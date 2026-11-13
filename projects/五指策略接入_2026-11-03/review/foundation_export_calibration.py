"""Export explicit public models from physical probes; no new dynamics."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',default='review/foundation_matrix_final')
    p.add_argument('--out',default='review/foundation_calibration_draft_v6.json')
    p.add_argument('--scope',nargs='+',default=['capsule','box','sphere'])
    p.add_argument('--certificate');a=p.parse_args()
    matrix=ROOT/a.matrix;destination=ROOT/a.out
    if destination.exists():raise SystemExit('Refusing to overwrite an exported model')
    rows=read(matrix/'manifest.json');barepath=ROOT/'review/foundation_bare_hand_final_v7/summary.json';bare=read(barepath)
    shapes={}
    for shape in a.scope:
        row=next(r for r in rows if r['config']['shape']==shape and r['config']['solref']==.008 and r['config']['timestep']==.0005)
        folder=Path(row['folder']);response_path=folder/'response.json';response=read(response_path)
        B=np.asarray(response['force_matrix'])
        holding=np.asarray(row['config']['holding_force'],float)
        passed=[h for h in response['holdout_by_radius'] if h['rmse']<=.35 and h['p95_absolute']<=.75]
        if not passed:raise SystemExit(f'No local execution domain passed for {shape}')
        selected=max(passed,key=lambda h:h['radius']);radius=selected['radius']
        raw_model_path=folder/'response_raw.npz'
        shape_data={**response,'config':row['config'],'force_gain':np.linalg.norm(B,axis=0).tolist(),
            'force_gain_interpretation':'positive normalization only; physical predictor uses the full signed force_matrix',
            'nominal_force_weights':(holding/holding.mean()).tolist(),
            'nominal_force_weights_source':'public common holding_force profile / its mean, shared by both policies before algorithm comparison',
            'shared_execution':{'close_rate':.8,'open_rate':.8,'release_rate':row['config']['release_open_rate'],
                'units':'normalized closure coordinates per second',
                'source':'HandPlant.execute_delta rate bounds; release fingers use FoundationConfig.release_open_rate',
                'thumb_release':'Shared t=0..0.8s clearance to [-1.03,1.21,.13,-.02,.985], hold to1.4s, then extend over0.8s; actual driver override is recorded.'},
            'model_reference_alpha':response['reference_alpha'],'trust_radius':radius,'validated_radius':radius,
            'validated_trust_radius':radius,'selected_trust_radius':radius,'selected_radius':radius,'execution_trust_radius':radius,
            'response_error_bound':response['response_error_bound_by_radius'][str(radius)],
            'selected_holdout':{'passed':True,'source_path':str(raw_model_path),'source_sha256':sha(raw_model_path),
                'summary_path':str(response_path),'summary_sha256':sha(response_path),'radius':radius,
                'force_rmse_N':selected['rmse'],'force_p95_N':selected['p95_absolute'],'maximum_absolute_N':selected['maximum']},
            'failed_radius_data':[h for h in response['holdout_by_radius'] if h not in passed],
            'validation_coverage':'Eight seeded multi-finger combinations per tested amplitude at 20/40/60 ms; empirical local evidence, not an exhaustive hypercube or real-force safety proof.',
            'effort_bias_N':bare['effort_bias_N'],'load_error_bound_N':bare['load_error_bound_N'],
            'source_response_path':str(response_path),'source_response_sha256':sha(response_path),
            'source_probe_path':str(folder/'raw_probe.npz'),'source_probe_sha256':sha(folder/'raw_probe.npz'),
            'source_sha256':row['summary']['source_sha256'],'model_sha256':row['summary']['model_sha256']}
        shapes[shape]=shape_data
    certificate=None
    if a.certificate:
        certificate_path=ROOT/a.certificate;certificate=read(certificate_path)
        if not certificate.get('passed',False):raise SystemExit('Physical certificate did not pass')
    local_model_pass=all(d['selected_holdout']['passed'] for d in shapes.values())
    result=dict(schema='shadow-public-holding-calibration-v1',benchmark_id='shadow_hand_strategy_integration_v1',
        calibration_model_passed=local_model_pass,
        physical_gate_status='external_physical_and_independent_certificates_required',
        scope=a.scope,shapes=shapes,hand_off_time=7.5,handoff_time=7.5,
        execution_trust_radius_by_shape={k:v['selected_radius'] for k,v in shapes.items()},sensor_dt=.005,decision_dt=.020,
        radius_selection_rule='Before any policy performance: largest tested radius among .02/.04 with independent 20/40/60ms force RMSE<=.35N and p95 absolute<=.75N; same domain for both methods.',
        units=dict(normal='N',force_matrix='N per normalized closure coordinate',force_tau='s',
                   reference_alpha='normalized actuator command coordinate',response_error_bound='N',wrench_moment='N m'),
        controller_permissions='Virtual local patches, joint encoders and own motor effort; no target identity, object mass, true friction or future schedule.',
        preparation_scope='Common reference preparation until 7.5s; no credit to either research policy. Model valid for local holding only.',
        bare_hand_source_path=str(barepath),bare_hand_source_sha256=sha(barepath),
        protocol_path=str(ROOT/'acceptance_protocol_local_domains_v8.json'),protocol_sha256=sha(ROOT/'acceptance_protocol_local_domains_v8.json'),
        foundation_protocol_path=rows[0]['summary'].get('protocol_path'),foundation_protocol_sha256=rows[0]['summary'].get('protocol_sha256'),
        source_export_sha256=sha(Path(__file__)))
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(destination),scope=a.scope,physical_status=result['physical_gate_status']),ensure_ascii=False))

if __name__=='__main__':main()

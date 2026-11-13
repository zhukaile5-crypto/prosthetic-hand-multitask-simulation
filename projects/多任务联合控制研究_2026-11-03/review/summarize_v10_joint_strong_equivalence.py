"""Complete-cohort retained physical equivalence and active-intent digest only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v10_audit_helpers',ROOT/'review/verify_v10_factorial_physical_independent.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize():
    audit=ROOT/'review/v10_all40_complete_physical_independent_v2.json';a=read(audit)
    if not a['passed'] or not a['full_registered40trial_cohort']:raise ValueError('Passing complete40trial audit required')
    rows=[];trace_cache={}
    for mode in ('hold','transport'):
        for scene in m.SCENES:
            base=ROOT/f'output/development_v10_{mode}_v1/runs/{scene}__ratio_preview__610212/trace.npz';joint=ROOT/f'output/development_v10_{mode}_v1/runs/{scene}__joint__610212/trace.npz';fields={}
            with np.load(base,allow_pickle=False) as b,np.load(joint,allow_pickle=False) as j:
                for key in m.PHYSICAL:fields[key]=m.bit_equal(j[key],b[key])
                if mode=='hold' and scene=='collision':
                    trace_cache={label:{key:file[key] for key in ('time','qpos','qvel','alpha_before','alpha','ctrl_before','executed_motion_reference','debug_json','observation_json')} for label,file in [('strong',b),('joint',j)]}
            rows.append(dict(motion_mode=mode,scenario=scene,all40physical_fields_bitexact=all(fields.values()),fields=fields))
    source=ROOT/'review/v10_all40_joint_active_servo_error_v1.json';error=read(source);active=[item for mode in error['modes'] for case in mode['records'] for item in case['intervals']]
    delta=np.asarray([item['declared_increment8'] for item in active]);known=[item for item in active if item['model_geometric_cost_before'] is not None];unknown=[item for item in active if item['model_geometric_cost_before'] is None]
    pressure=np.asarray([np.asarray(item['physical_skin_pressure_after_N'])-item['physical_skin_pressure_before_N'] for item in active]);detail=[]
    for item in active:
        i=round(item['start_s']/.005);d=json.loads(str(trace_cache['joint']['debug_json'][i]));b=trace_cache['strong'];j=trace_cache['joint']
        start_b=np.r_[b['alpha_before'][i],json.loads(str(b['observation_json'][i]))['executed_motion_reference']]
        actual_b=np.r_[b['alpha'][i+3],b['executed_motion_reference'][i+3]]-start_b
        detail.append(dict(start_s=item['start_s'],legal_hypothetical_nominal_intent8=d['nominal_increment8'],candidate_declared_increment8=item['declared_increment8'],actual_strong20ms_reference_increment8=actual_b.tolist(),
            nominal_component_scope=d['nominal_future_component_check'],actual_physical_state_before_equal=all(m.bit_equal(j[key][i],b[key][i]) for key in ('qpos','qvel','alpha_before','ctrl_before')),
            model_geometric_cost_before=item['model_geometric_cost_before'],model_geometric_cost_after=item['model_geometric_cost_after']))
    result=dict(schema='v10_joint_strong_equivalence_and_active_digest_v1',new_physics=False,audit_path=str(audit),audit_sha256=sha(audit),servo_error_path=str(source),servo_error_sha256=sha(source),
        comparisons=rows,all40physical_fields_bitexact_pairs=sum(item['all40physical_fields_bitexact'] for item in rows),assigned_pairs=8,active_details=detail,
        active_digest=dict(n=len(active),context_ids=sorted({item['context_id'] for item in active}),reference_outside_component_footprint_count=sum(item['reference_outside_component_footprint'] for item in active),
            planned_increment_min8=delta.min(0).tolist(),planned_increment_max8=delta.max(0).tolist(),planned_increment_sum8=delta.sum(0).tolist(),
            comparative_model_cost_available_intervals=len(known),unsupported_nominal_absolute_model_gate_intervals=len(unknown),
            candidate_model_cost_after_range=[min(item['model_geometric_cost_after'] for item in active),max(item['model_geometric_cost_after'] for item in active)],
            comparative_cost_improved_all=None if not known else all(item['model_geometric_cost_after']<item['model_geometric_cost_before']-1e-16 for item in known),
            robot_anchor_angular_max_distance_rad=max(item['robot_state_distance']['angular_max_abs_rad'] for item in active),robot_anchor_lift_max_distance_m=max(item['robot_state_distance']['lift_abs_m'] for item in active),
            thumb_raw_pressure_after20ms_minus_before_minmax_N=[float(pressure[:,0].min()),float(pressure[:,0].max())],thumb_raw_pressure_declined_intervals=int((pressure[:,0]<0).sum()),
            interpretation='Only accepted-plan increments summed. Unsupported nominal has no meaningful before-cost; its gate is absolute predicted geometry/tube. Pressure changes include evolving state/environment, not separate causal gains.'),source_sha256=sha(__file__))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();result=summarize()
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.out),sha256=sha(args.out),equivalence=[{key:item[key] for key in ('motion_mode','scenario','all40physical_fields_bitexact')} for item in result['comparisons']],active_digest=result['active_digest'],first_active=result['active_details'][0]),ensure_ascii=False,indent=2))

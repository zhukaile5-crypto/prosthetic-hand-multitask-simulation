"""Assemble independent physical admission from separately checked evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    protocol=ROOT/'acceptance_protocol_local_domains_v8.json'
    physical=ROOT/'review/foundation_physical_acceptance.json'
    calibration=ROOT/'review/foundation_calibration_final.json'
    paths={'raw':ROOT/'review/independent_foundation_final_raw.json',
           'packets':ROOT/'review/observation_physical_packets_final.json',
           'calibration':ROOT/'review/independent_calibration_final_v2.json',
           'capsule_structure':ROOT/'review/independent_model_structure_capsule_final.json',
           'box_structure':ROOT/'review/independent_model_structure_box_final.json',
           'sphere_structure':ROOT/'review/independent_model_structure_sphere_final.json',
           'adapter_synthetic':ROOT/'review/independent_adapter_20261104T020553343909Z.json',
           'copied_sources':ROOT/'review/independent_copied_sources.json'}
    reports={k:read(p) for k,p in paths.items()};p=read(physical);c=read(calibration);protocol_data=read(protocol)
    checks={'physical_self_check_passed':p.get('passed') is True,
            'physical_active_protocol':p['protocol_sha256']==sha(protocol),
            'calibration_active_protocol':c['protocol_sha256']==sha(protocol),
            'physical_calibration_hash_chain':p['calibration_sha256']==sha(calibration),
            'full_raw_matrix':reports['raw']['full_raw_matrix_passed'] is True,
            'raw_source_identity':reports['raw']['all_raw_sources_match_current'] is True,
            'raw_protocol_identity':reports['raw']['protocol_sha256']==sha(protocol),
            'physical_matrix_under_parent_v7':c['foundation_protocol_sha256']==protocol_data['trust_domain_revision']['parent_sha256'],
            'physical_gates_unchanged_in_domain_revision':protocol_data['trust_domain_revision']['physical_gates_and_model_design_unchanged'] is True}
    for name,r in reports.items():
        if name!='raw':checks[name+'_passed']=r.get('passed') is True
    for name in ('packets','calibration','capsule_structure','box_structure','sphere_structure','adapter_synthetic'):
        r=reports[name]
        sources=r.get('source_sha256',{})
        checks[name+'_sources_current']=all((ROOT/f).is_file() and sha(ROOT/f)==h for f,h in sources.items())
    for row in reports['raw']['probes']:
        checks['raw_'+row['config']['shape']+'_'+str(row['config']['solref'])+'_'+str(row['config']['timestep'])+'_identity']=sha(Path(row['path']))==row['raw_sha256']
    checks['independent_selected_calibration_identity']=reports['calibration']['calibration_sha256']==sha(calibration)
    checks['independent_selected_protocol_identity']=reports['calibration']['protocol_sha256']==sha(protocol)
    sources=('hand_plant.py','foundation_config.py','calibrate_foundation.py','contracts.py','sensors.py',
             'state_estimator.py','force_model.py','controller_adapter.py','policy_sources.py','vendor/frozen_controllers.py')
    result={'schema':'shadow-hand-independent-physical-admission-v1',
            'benchmark_id':'shadow_hand_strategy_integration_v1','checked_utc':datetime.now(timezone.utc).isoformat(),
            'passed':all(checks.values()),'checks':checks,
            'protocol_sha256':sha(protocol),'physical_certificate_sha256':sha(physical),
            'calibration_sha256':sha(calibration),'source_sha256':{f:sha(ROOT/f) for f in sources},
            'evidence':{k:{'path':str(v.relative_to(ROOT)),'sha256':sha(v)} for k,v in paths.items()},
            'scope':'Independent physical/model/observation readiness for the declared held-grasp strategy migration, not policy performance approval.',
            'counts':{'physical_trials':12,'timestep_pairs':6,'contact_packets':1752,'calibration_checks':30,'synthetic_adapter_checks':59},
            'limits':['Original mesh/6D-contact and sharp-box failed designs remain historical, not passed.',
                      'Accepted model uses 24 hand/wrist joints, original 20 hand actuators and explicit simplified contacts with rounded box.',
                      'Current policies only take over a commonly prepared held grasp at fixed 7.5 s.',
                      'Sample-validated local response regions are not mathematical worst-case force guarantees.',
                      'No real policy run or new dynamics was executed by this independent numeric audit.',
                      'Final 5168-file history rehash is deliberately deferred to delivery; copied sources only are checked here.']}
    output=ROOT/'review/physical_independent_acceptance.json'
    if output.exists():raise FileExistsError('Refusing to overwrite independent physical admission evidence.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'certificate':str(output),'sha256':sha(output)},ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()

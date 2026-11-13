"""Independent noise algebra/sampling only. No QP, simulator or plant imports."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import ast,runpy,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'五指策略接入_2026-11-03'
EXPECTED='2f3c3de3c081042116f026246e5e72e05bde77c660b8303d6701ef156e1a4bd6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def crossmat(v):
    return np.cross(v,np.eye(3)).T
def oracle(frame,spec):
    sf=spec['force_sigma_N'];st=spec['tangential_sigma_N'];sp=spec['position_sigma_m'];sa=spec['auxiliary_force_sigma_N'];sm=spec['auxiliary_moment_sigma_Nm'];cov=np.diag([sa*sa]*3+[sm*sm]*3)
    for i in range(5):
        g=frame.normal_wrench_map[:,i];B=frame.tangent_basis[i].T;r=frame.contact_pos[i];F=g[:3]*frame.normal[i]+frame.finger_non_normal_by_finger[i,:3]
        normals=np.outer(g,g)*sf*sf
        tangents=np.column_stack([np.r_[B[:,j],np.cross(r,B[:,j])] for j in range(2)])
        pos=np.column_stack([np.r_[np.zeros(3),np.cross(np.eye(3)[j],F)] for j in range(3)])
        pv=sp*sp if frame.contact_present[i] else 0.
        cov+=normals+st*st*(tangents@tangents.T)+pv*(pos@pos.T)
        C=sf*sf*np.outer(g[:3],g[:3])+st*st*(B@B.T)
        cov[3:,3:]+=pv*(np.eye(3)*np.trace(C)-C)
    return cov
def draw_sensor(frame,spec,count=100000):
    """Direct clean-geometry algebra in sensors.py, including quadratic noise."""
    rng=np.random.default_rng(2026110401);out=np.zeros((count,6));sf=spec['force_sigma_N'];st=spec['tangential_sigma_N'];sp=spec['position_sigma_m']
    for i in range(5):
        dn=rng.normal(0,sf,count);dt=rng.normal(0,st,(count,2));dp=rng.normal(0,sp,(count,3)) if frame.contact_present[i] else np.zeros((count,3))
        n=np.maximum(0,frame.normal[i]+dn);g=frame.normal_wrench_map[:3,i];m=frame.normal_wrench_map[3:,i];F=frame.finger_non_normal_by_finger[i,:3];M=frame.finger_non_normal_by_finger[i,3:];r=frame.contact_pos[i]
        extra=dt@frame.tangent_basis[i]
        out[:,:3]+=n[:,None]*g+F+extra
        out[:,3:]+=(m+np.cross(dp,g))*n[:,None]+M+np.cross(dp,F)+np.cross(r+dp,extra)
    out[:,:3]+=rng.normal(0,spec['auxiliary_force_sigma_N'],(count,3));out[:,3:]+=rng.normal(0,spec['auxiliary_moment_sigma_Nm'],(count,3))
    return np.cov(out,rowvar=False,ddof=0)
def main():
    checks={}
    def check(k,v):checks[k]=bool(v)
    check('final_source_identity_before',sha(ROOT/'coordinator_model.py')==EXPECTED)
    api=runpy.run_path(str(ROOT/'coordinator_model.py'));cls=api['ContactConditionedModel'];factory=api['synthetic_document']
    def model():return cls(dict(reference_alpha=np.zeros(5),response_error_bound=np.array([.508,.011,.022,.033,.044]),config={'shape':'capsule'}),document=factory())
    gen=np.random.default_rng(71);g=gen.normal(size=(5,3));g/=np.linalg.norm(g,axis=1)[:,None];g*=np.linspace(.7,1.,5)[:,None];r=gen.uniform(-.06,.06,(5,3));B=[]
    for v in g:
        unit=v/np.linalg.norm(v);a=np.cross(unit,[0.,0.,1.]);a/=np.linalg.norm(a);B.append([a,np.cross(unit,a)])
    M=np.vstack([g.T,np.cross(r,g).T]);T=gen.uniform(-.12,.12,(5,6));frame=SimpleNamespace(normal_wrench_map=M,tangent_basis=np.asarray(B),contact_pos=r,finger_non_normal_by_finger=T,normal=np.array([2.4,1.7,1.8,1.9,1.6]),contact_present=np.ones(5,bool))
    full=dict(force_sigma_N=.02,tangential_sigma_N=.008,position_sigma_m=.0004,auxiliary_force_sigma_N=.012,auxiliary_moment_sigma_Nm=.0007)
    base={k:0. for k in full};covariances={}
    for name,keys in [('normal',['force_sigma_N']),('tangent',['tangential_sigma_N']),('position',['position_sigma_m']),('aux',['auxiliary_force_sigma_N','auxiliary_moment_sigma_Nm']),('full',list(full))]:
        spec={**base,**{k:full[k] for k in keys}};m=model();before=m.error_by_horizon.copy();m.set_sensor_characteristics(spec);m._wrench_noise(frame);cov=np.asarray(m.noise_diagnostics['covariance']);expected=oracle(frame,spec);covariances[name]=cov
        check(name+'/full_covariance_matches_independent_algebra',np.allclose(cov,expected,atol=1e-16,rtol=1e-12))
        check(name+'/physical_units_psd_and_sigma',np.allclose(cov,cov.T,atol=1e-16) and np.linalg.eigvalsh(cov).min()>=-1e-15 and np.allclose(m.noise_floor[10:],np.maximum(1e-12,np.sqrt(np.diag(expected))),atol=1e-15))
        check(name+'/public_calibration_error_preserved',np.array_equal(before,m.error_by_horizon) and m.error_by_horizon[:,0].min()==.508)
    empirical=draw_sensor(frame,full);expected=covariances['full'];sigma=np.sqrt(np.diag(expected));standardized=(empirical-expected)/np.outer(sigma,sigma)
    check('100000_draw_direct_sensor_equations_covariance',np.max(abs(standardized))<.02)
    check('tangent_affects_both_force_and_moment',np.diag(covariances['tangent'])[:3].sum()>0 and np.diag(covariances['tangent'])[3:].sum()>0)
    check('position_affects_moment_not_force',np.diag(covariances['position'])[:3].sum()==0 and np.diag(covariances['position'])[3:].sum()>0)
    check('aux_moment_not_rescaled_by_characteristic_length',np.allclose(np.diag(covariances['aux']),[full['auxiliary_force_sigma_N']**2]*3+[full['auxiliary_moment_sigma_Nm']**2]*3,rtol=0,atol=1e-18))
    inactive=SimpleNamespace(**vars(frame));inactive.contact_present=np.zeros(5,bool);m=model();m.set_sensor_characteristics({**base,'position_sigma_m':.001});m._wrench_noise(inactive);check('inactive_contact_has_no_position_shift_noise',np.max(abs(np.asarray(m.noise_diagnostics['covariance'])))==0)
    m=model();m.set_sensor_characteristics(full);m.snapshot=SimpleNamespace(output_valid=np.ones(16,bool),fit_error=np.zeros(16));bad=SimpleNamespace(**vars(frame));bad.contact_pos=r.copy();bad.contact_pos[0,0]=np.nan;m._wrench_noise(bad);assessment=m.assess(np.zeros((4,16)))
    check('invalid_public_geometry_rejected_not_stale_noise_admitted',not assessment['usable'] and 'public_wrench_noise_geometry_unavailable' in assessment['reasons'])
    # Execute only the four top-level pure device-resolution assignments in the
    # production initializer, without constructing ResearchSession or HandPlant.
    tree=ast.parse((ROOT/'runtime.py').read_text(encoding='utf-8'));session=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ResearchSession');init=next(x for x in session.body if isinstance(x,ast.FunctionDef) and x.name=='__init__');assign=[]
    for node in init.body:
        if isinstance(node,ast.Assign):
            targets=[ast.unparse(x) for x in node.targets]
            if any(x in ['device','auxiliary_force_sigma','auxiliary_moment_sigma','self.sensor_characteristics'] for x in targets):assign.append(node)
    check('device_resolution_shared_top_level_not_method_branch',len(assign)==4)
    constants=runpy.run_path(str(BASE/'contracts.py'));length=constants['WRENCH_LENGTH'];resolver=compile(ast.fix_missing_locations(ast.Module(body=assign,type_ignores=[])),'production_device_resolution','exec')
    contract=runpy.run_path(str(ROOT/'task_contract.py'));resolved=[]
    for sa,sm in [(None,None),(.03,None),(.03,.002),(0.,None)]:
        device=SimpleNamespace(force_sigma=.01,tangential_sigma=.005,position_sigma=.0001,slip_sigma=.0002,effort_sigma=.02,delay=.01,auxiliary_force_sigma=sa,auxiliary_moment_sigma=sm);s=SimpleNamespace(sensors=SimpleNamespace(noise=device));exec(resolver,{'self':s,'WRENCH_LENGTH':length});spec=s.sensor_characteristics
        expected_aux=.01 if sa is None else sa;expected_M=expected_aux*length if sm is None else sm
        check('runtime_aux_default/'+str((sa,sm)),spec['auxiliary_force_sigma_N']==expected_aux and spec['auxiliary_moment_sigma_Nm']==expected_M)
        task=contract['build_task_request'](8.,'hold',[.1,0.,0.],[.1,0.,0.],12.,[5.]*5,18.,sensor_characteristics=spec);check('public_spec_roundtrip/'+str((sa,sm)),contract['validate_task_request'](task)==task);resolved.append(spec)
    extra={**resolved[0],'true_mass':.12}
    try:contract['build_task_request'](8.,'hold',[.1,0.,0.],[.1,0.,0.],12.,[5.]*5,18.,sensor_characteristics=extra);reject=False
    except ValueError:reject=True
    check('hidden_physical_field_rejected_in_public_spec',reject)
    versions=[]
    for version in [1,2]:
        path=ROOT/f'review/coordinator_v6_validation_v{version}.json';j=read(path);versions.append(dict(version=version,sha256=sha(path),decisions=sum(x['decisions'] for x in j['saved_public_streams']),prediction_available=sum(x['prediction_available_decisions'] for x in j['saved_public_streams']),source_identity=j['sources'],scope=j['scope']))
    check('prior_zero_coverage_evidence_retained',versions[0]['sha256']=='d3213d288238d024317fb637314d8f467b2ca97dde2f7e8201282746c677b856' and versions[0]['prediction_available']==0 and versions[0]['decisions']==1688)
    check('no_physics_module_imported','mujoco' not in sys.modules)
    check('final_source_identity_after',sha(ROOT/'coordinator_model.py')==EXPECTED)
    out=dict(schema='independent_v6_sensor_noise_propagation_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),source_sha256={k:sha(ROOT/k) for k in ['coordinator.py','coordinator_model.py','coordinator_contacts.py','runtime.py','task_contract.py']},sensors_sha256=sha(BASE/'sensors.py'),reviewer_sha256=sha(__file__),monte_carlo=dict(draws=100000,max_standardized_covariance_difference=float(np.max(abs(standardized))),analytic_sigma=sigma.tolist(),empirical_sigma=np.sqrt(np.diag(empirical)).tolist(),interpretation='Synthetic clean fixed coefficients and active contacts; not validation of noisy plug-in coefficients or real model/safety coverage.'),public_device_resolution=resolved,saved_stream_versions=versions,physics_steps_executed=0,qp_calls=0,limitations=['The propagated covariance is a public noisy-geometry/force plug-in engineering estimate. Clipping, contact switching, delayed feedback and temporal correlation prevent an exact Gaussian/high-probability or deterministic safety interpretation.','True sensor position perturbation uses noiseless normal>.02; public contact_present uses positive distributed contact. The latter is a conservative surrogate near the threshold, not an exact hidden mask.','3sigma residual floor and6sigma model-mode gate are engineering statistical compatibility rules, not a family-wise confidence limit over outputs/horizons/trials.','Calibration maxima are retained; runtime fit/noise and slope terms also enter trajectory uncertainty. No threshold was selected by our test using task-success or force-pass performance.','New saved-stream prediction coverage is an off-policy interface property and versions differ in multiple sources; it does not identify closed-loop benefit or isolate a causal performance effect of this revision.'])
    path=ROOT/'review'/('independent_v6_sensor_noise_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');path.write_text(json.dumps(serial(out),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=out['failed'],path=str(path),sha256=sha(path),monte_carlo=out['monte_carlo'],coverage=[(x['prediction_available'],x['decisions']) for x in versions]),ensure_ascii=False))
if __name__=='__main__':main()

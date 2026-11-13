"""Read-only AST-exact methods plus independent piecewise native-law oracle.

No model/data/servo construction, forward/FK/step or whole controller import.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, ast, hashlib, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
EXPECTED={
 'v15_support/controller.py':'3c1e1e3333644d9452c20035e8032a4498453a6680a2f262d74943808e5c8911',
 'v15_support/posture_tasks.py':'33a8ff95024c5e9751e72ef39f06eebf460e3b7520574eb4eb05881084d3272e',
 'v11/robot_servo_effort.py':'f0671211d2e15cc96eeb8c87207a43bc1839d55512ab18bdc2cebff4ed0a052c',
 'v14_coord/effort_domain.py':'254b4a95dbaa7557bd7e767b5db7692b26287c1dfcf898341df655dbf91d5060'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def method(path,cls,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return next(f for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for f in c.body if isinstance(f,ast.FunctionDef) and f.name==name)

def functions():
    scope=dict(np=np)
    fn=method(ROOT/'v15_support/controller.py','UnifiedTaskController','_native_task_inputs')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/'v15_support/controller.py'),'exec'),scope)
    # Actual frozen F067 native-value and clip body, no transmission or data.
    ev=method(ROOT/'v11/robot_servo_effort.py','RobotServoEffort','evaluate')
    block=[x for x in ev.body if 141<=x.lineno<=154]
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('self','u','length','speed')],kwonlyargs=[],kw_defaults=[],defaults=[])
    tail=ast.parse("return effective,unbounded,effort,g,ctrl_kink,force_kink").body
    native=ast.FunctionDef(name='frozen_native_value_body',args=args,body=block+tail,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[native],type_ignores=[])),str(ROOT/'v11/robot_servo_effort.py'),'exec'),scope)
    return scope['_native_task_inputs'],scope['frozen_native_value_body']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    checks=[];records=[];failures=[];before={k:sha(ROOT/k) for k in EXPECTED};old_mj=sys.modules.get('mujoco')
    # Values are documented MuJoCo enum tags. This fake module is used only
    # by extracted arithmetic, never by an engine or actual model constructor.
    fake=SimpleNamespace(mjtGain=SimpleNamespace(mjGAIN_FIXED=0,mjGAIN_AFFINE=1),mjtBias=SimpleNamespace(mjBIAS_NONE=0,mjBIAS_AFFINE=1),mjtDisableBit=SimpleNamespace(mjDSBL_CLAMPCTRL=128))
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:failures.append(name)
    def close(name,a,b):ck(name,np.allclose(a,b,atol=2e-14,rtol=2e-14),dict(actual=np.asarray(a).tolist(),expected=np.asarray(b).tolist()))
    try:
        ck('source_identity',before==EXPECTED)
        fn,native=functions();native.__globals__['mujoco']=fake;sys.modules['mujoco']=fake
        def case(name,*,gp=(2.,0.,0.),gaintype=0,bp=(-.5,0.,0.),length=0.,speed=0.,lo=.1,hi=.7,current=.3,nominal=.4,force_limited=True,expected=True):
            m=SimpleNamespace(actuator_gainprm=np.tile([1.,0.,0.],(21,1)),actuator_gaintype=np.zeros(21,int),
                actuator_biasprm=np.zeros((21,3)),actuator_biastype=np.ones(21,int),opt=SimpleNamespace(disableflags=0))
            m.actuator_gainprm[0]=gp;m.actuator_gaintype[0]=gaintype;m.actuator_biasprm[0]=bp
            servo=SimpleNamespace(model=m,aids=np.arange(21),ctrl_limited=np.ones(21,bool),ctrl_range=np.tile([-1.,1.],(21,1)),
                force_limited=np.zeros(21,bool),force_range=np.tile([-5.,5.],(21,1)))
            servo.ctrl_range[0]=[0.,1.];servo.force_limited[0]=force_limited;servo.force_range[0]=[-1.,1.]
            lens=np.zeros(21);vel=np.zeros(21);lens[0]=length;vel[0]=speed
            def snap(u):
                eff,unbounded,p,g,ckink,fkink=native(servo,u,lens,vel)
                return SimpleNamespace(actuator_length21=lens,actuator_velocity21=vel,predicted_effort21=p,
                    diagnostics=dict(control_clipped21=(u!=eff),force_clipped21=(p!=unbounded),control_at_clip_kink21=ckink,force_at_clip_kink21=fkink))
            cur=np.zeros(21);cur[0]=current;nom=np.zeros(21);nom[0]=nominal
            lower=np.ones(21)*(-.2);upper=np.ones(21)*.2;lower[0]=lo;upper[0]=hi
            a=snap(lower).predicted_effort21;b=snap(upper).predicted_effort21
            dom=SimpleNamespace(_context=dict(state=snap(cur),nominal_state=snap(nom)),effort_lower21=np.minimum(a,b),effort_upper21=np.maximum(a,b),reference_lower21=lower,reference_upper21=upper)
            owner=SimpleNamespace(native=SimpleNamespace(servo=servo))
            gain,known=fn(owner,dom)
            ck(name+'/admission',bool(known[0]) is expected,dict(gain=float(gain[0]),known=bool(known[0]),effort_interval=[float(dom.effort_lower21[0]),float(dom.effort_upper21[0])]))
            gg=gp[0] if gaintype==0 else gp[0]+gp[1]*length+gp[2]*speed
            close(name+'/same_gain_formula',gain[0],gg)
            if expected:
                maxerr=0.;any_clip=False
                for x in list(np.linspace(lo,hi,129))+[nominal]:
                    u=np.zeros(21);u[0]=x;ss=snap(u)
                    # Independent real piecewise model with scalar clamp;
                    # separate coefficients preserve F067 evaluation order.
                    effective=min(1.,max(0.,x));raw=gg*effective+bp[0]+bp[1]*length+bp[2]*speed
                    p=min(1.,max(-1.,raw)) if force_limited else raw
                    maxerr=max(maxerr,abs(ss.predicted_effort21[0]-p))
                    any_clip|=bool(ss.diagnostics['control_clipped21'][0] or ss.diagnostics['force_clipped21'][0])
                ck(name+'/whole_interval_and_raw_nominal_single_affine_branch',not any_clip and maxerr<=2e-14,dict(points=130,max_error=maxerr))
            records.append(dict(case=name,expected_known=expected,actual_known=bool(known[0]),g=float(gain[0]),allowed_reference=[lo,hi],raw_nominal=nominal))
            return gain,known
        case('positive_affine')
        case('negative_affine',gp=(-2.,0.,0.),bp=(.5,0.,0.))
        case('fixed_gain_ignores_length_velocity_coefficients',gp=(2.,100.,-100.),length=5.,speed=3.)
        case('affine_gain_uses_length_velocity',gp=(2.,3.,-4.),gaintype=1,length=5.,speed=4.,bp=(-.5,0.,0.))
        case('zero_gain_not_an_invertible_reference_branch',gp=(0.,0.,0.),bp=(.2,0.,0.),expected=False)
        case('force_crossing_in_future_interval_rejected',hi=.9,expected=False)
        case('force_kink_at_allowed_endpoint_rejected',hi=.75,expected=False)
        case('raw_nominal_force_clipped_even_outside_permission_rejected',nominal=.9,expected=False)
        case('raw_nominal_control_clipped_rejected',nominal=1.2,force_limited=False,expected=False)
        case('current_control_kink_conservatively_rejected',current=0.,expected=False)
        case('native_force_plateau_rejected',lo=.8,hi=.9,current=.8,nominal=.9,expected=False)
        case('force_limit_disabled_allows_unbounded_linear_law',gp=(3.,0.,0.),force_limited=False,hi=.9)
        case('closed_control_boundary_is_valid_one_sided_affine',gp=(.5,0.,0.),bp=(0.,0.,0.),lo=0.,hi=1.)
        # Actual posture producer with independent reference-coordinate oracle.
        from v15_support.posture_tasks import build_holding_native_tasks
        gain=np.ones(21)*(-2.);flags=np.ones(21,bool);ranges=np.tile([-1.,1.],(21,1));nom=np.ones(21)*.2;anchor=np.ones(21)*.3
        opened=np.zeros(18);closed=np.r_[np.arange(1.,6.),np.ones(13)]
        kwargs=dict(patch_count=0,time_s=1.,p0_21=np.ones(21),delta_scale21=np.ones(21),nominal_reference21=nom,grasp_anchor21=anchor,
            native_reference_gain21=gain,known_affine21=flags,control_range21=ranges,open18=opened,closed18=closed)
        tasks=build_holding_native_tasks(**kwargs);delta=np.linspace(-.1,.1,21);res=tasks.matrix@delta-tasks.target
        direction=closed[:5]/2.;projector=np.eye(5)-np.outer(direction,direction)/(direction@direction)
        close('negative_gain_posture_rows_match_reference_geometry',res[:5],projector@((nom+delta/gain-anchor)[:5]/2.))
        flags[0]=False;inactive=build_holding_native_tasks(**kwargs)
        ck('one_unknown_coordinate_disables_whole_finger_inverse_task',inactive.metadata['inactive_shape_groups'][0]['digit']==0 and not any(x.startswith('grasp_shape_digit0_') for x in inactive.labels))
        kwargs['known_affine21']=np.ones(21,bool);kwargs['native_reference_gain21']=gain.copy();kwargs['native_reference_gain21'][0]=0.
        inactive=build_holding_native_tasks(**kwargs);ck('zero_gain_never_divided_or_filled_as_zero_inverse',inactive.metadata['inactive_shape_groups'][0]['digit']==0 and np.isfinite(inactive.matrix).all())
        ck('all_sources_unchanged',before=={k:sha(ROOT/k) for k in EXPECTED})
    except Exception:failures.append('exception');records.append(dict(exception=traceback.format_exc()))
    finally:
        if old_mj is None:sys.modules.pop('mujoco',None)
        else:sys.modules['mujoco']=old_mj
    result=dict(schema='v15_native_affine_scope_independent_v1',passed=not failures,checks=checks,failures=failures,records=records,source_sha256=before,checker_sha256=sha(__file__),
        proof='At fixed q/v each SISO native force is monotone affine before clamping. FastDomain intersects candidate references with control bounds. Raw current/nominal snapshots reject clipping/kinks; strictly interior entire force-image interval excludes force saturation across both signs of g. Endpoint control boundaries remain affine on the closed permitted side, not a two-sided Jacobian certificate.',
        scope='AST-exact admission and F067 scalar value body on synthetic coherent domain fixtures; no actual model, transmission, MjData, FK, forward, physics or whole controller import. Parent owns actual9 public packets/history.',
        physical_safety_certified=False,physics_steps=0,model_constructions=0)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=failures,out=str(out)),ensure_ascii=False))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()

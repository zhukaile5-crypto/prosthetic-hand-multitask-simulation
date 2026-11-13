"""Author gate for process-only numerical helper; no engine/physical calls."""
from pathlib import Path
import sys,json,hashlib,os
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from v22_joint_task_fusion import numerics as n
rows=[];forbidden=[];helper_solver_calls=[]
def ck(name,value):rows.append(dict(name=name,passed=bool(value)));assert value,name
def deny(*a,**k):forbidden.append('engine_step_forward_contact');raise AssertionError(forbidden[-1])
def denysolver(*a,**k):helper_solver_calls.append('optimizer_called_by_identity');raise AssertionError(helper_solver_calls[-1])
def rejection(call):
 try:call()
 except RuntimeError:return True
 return False
oldnp=np;oldmj=mujoco;path_before=list(sys.path)
setup=n.ensure_numeric_environment();ck('legacy_module_objects_not_replaced',np is oldnp and mujoco is oldmj);ck('existing_SciPy181_from_declared_append',setup['package_origins']['scipy']['version']=='1.18.1' and setup['package_origins']['scipy']['file']==str(n.RESEARCH_SITE/'scipy/__init__.py'))
ck('append_only_sys_path_not_pth_execution',[p for p in sys.path if p!=str(n.RESEARCH_SITE)]==[p for p in path_before if p!=str(n.RESEARCH_SITE)] and sys.path.count(str(n.RESEARCH_SITE))==1)
second=n.ensure_numeric_environment();ck('configuration_idempotent',setup==second and sys.path.count(str(n.RESEARCH_SITE))==1 and os.environ['PATH'].split(os.pathsep)[0]==str(n.LIBRARY_BIN))
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_collision','mj_contactForce'):
 if hasattr(mujoco,name):setattr(mujoco,name,deny)
class NoModel:
 def __new__(cls,*a,**k):return deny()
 from_xml_path=staticmethod(deny);from_xml_string=staticmethod(deny);from_binary_path=staticmethod(deny)
mujoco.MjModel=NoModel;mujoco.MjData=deny
import scipy,scipy.optimize
with patch.object(scipy.optimize,'minimize',denysolver),patch.object(scipy.optimize,'linprog',denysolver),patch.object(scipy.optimize,'lsq_linear',denysolver):identity=n.numerical_identity()
old=json.loads((ROOT/'review/v21_small_plan_v1.json').read_text('utf8'))['environment']
from v13_a1.run_bridge import environment
ck('legacy42_actual_function_and_original_plan_exact',identity['legacy_pinned_environment']==environment()==old and len(old['native_binaries_sha256'])==42)
ck('new_dependencies_are_stable_explicit_pins',len(identity['scipy_native_binaries_sha256'])==99 and len(identity['scipy_python_implementation_sha256'])==11 and len(identity['necessary_runtime_dll_sha256'])>=18)
with patch.object(sys,'executable',str(n.RESEARCH_PREFIX/'python.exe')):ck('reject_wrong_interpreter_including_pure_math',rejection(n.ensure_numeric_environment))
with patch.object(np,'__version__','wrong'):ck('reject_wrong_loaded_numpy_version',rejection(n.ensure_numeric_environment))
with patch.object(np,'__file__',str(n.RESEARCH_SITE/'numpy/__init__.py')):ck('reject_wrong_loaded_numpy_origin',rejection(n.ensure_numeric_environment))
with patch.object(mujoco,'__version__','wrong'):ck('reject_wrong_loaded_mujoco_version',rejection(n.ensure_numeric_environment))
with patch.object(scipy,'__file__',str(n.LEGACY_SITE/'scipy/__init__.py')):ck('reject_wrong_loaded_scipy_origin',rejection(n.ensure_numeric_environment))
with patch.object(n,'RESEARCH_SITE',n.RESEARCH_PREFIX/'definitely_missing_existing_site'):ck('missing_directory_honest_not_install',rejection(n.ensure_numeric_environment))
originalsha=n._sha
with patch.object(n,'_sha',lambda p:'0'*64 if str(p).endswith('_multiarray_umath.cp312-win_amd64.pyd') else originalsha(p)):
 ck('reject_changed_legacy_binary_bytes_without_touching_files',rejection(n.numerical_identity))
x=np.linalg.lstsq(np.eye(2),np.array([1.,2.]),rcond=None)[0];ls=scipy.optimize.lsq_linear(np.eye(2),np.array([1.,2.]),bounds=(np.zeros(2),np.full(2,np.inf)),method='bvls',lsq_solver='exact',tol=1e-14,max_iter=400);opt=scipy.optimize.minimize(lambda x:float((x[0]-.5)**2),np.array([0.]),jac=lambda x:np.array([2*(x[0]-.5)]),method='SLSQP',options={'ftol':1e-12,'maxiter':400});lp=scipy.optimize.linprog([-1.],bounds=[(0.,1.)],method='highs')
ck('actual_numpy_LS_BVLS_SLSQP_HiGHS_coexist',np.array_equal(x,[1.,2.]) and np.array_equal(ls.x,[1.,2.]) and opt.success and abs(opt.x[0]-.5)<1e-12 and lp.success and lp.x[0]==1.)
after=n.numerical_identity();ck('identity_before_after_lazy_numeric_calls_exact',identity==after);ck('identity_has_no_solver_engine_calls',not helper_solver_calls and not forbidden)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out=ROOT/'review/v22_numerics_author_attempt1.json';r=dict(schema='v22_numerics_author_v1',passed=all(c['passed'] for c in rows),checks=rows,production_source_sha256={'v22_joint_task_fusion/numerics.py':sha(n.__file__)},numerical_identity=identity,forbidden_calls=forbidden,helper_solver_calls=helper_solver_calls,author_external_small_math_calls=4,physics_steps=0,checker_sha256=sha(__file__),global_environment_modified=False,packages_installed=False)
with out.open('x',encoding='utf8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=r['passed'],checks=len(rows),native_scipy=len(identity['scipy_native_binaries_sha256']),python_wrappers=len(identity['scipy_python_implementation_sha256']),runtime_dlls=len(identity['necessary_runtime_dll_sha256']),out=str(out),sha256=sha(out),source_sha256=sha(n.__file__)),ensure_ascii=False),flush=True)

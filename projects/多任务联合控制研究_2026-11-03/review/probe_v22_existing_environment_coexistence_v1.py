"""Import-and-small-math-only existing environment coexistence probe."""
from pathlib import Path
import sys,os,argparse,json,hashlib,traceback,ctypes
ROOT=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('research_extra_venv','venv_extra_research'),required=True);a=ap.parse_args()
VENV=ROOT.parent/'仿真调研_2026-10-08/local_probe/.venv/Lib/site-packages';RESEARCH=Path('D:/conda_envs/research312/Lib/site-packages');BIN=Path('D:/conda_envs/research312/Library/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['PATH']=str(BIN)+os.pathsep+os.environ.get('PATH','')
forbidden=[]
def deny(*args,**kwargs):forbidden.append('engine_constructor_step_forward_contact');raise AssertionError(forbidden[-1])
r=dict(schema='v22_environment_coexistence_probe_v1',mode=a.mode,physics_steps=0,forbidden_calls=forbidden,global_environment_modified=False,packages_installed=False)
try:
 import numpy as np
 if a.mode=='research_extra_venv':
  import scipy
  sys.path.append(str(VENV));extra=VENV
  import mujoco
 else:
  import mujoco
  sys.path.append(str(RESEARCH));extra=RESEARCH
  import scipy
 from scipy.optimize import minimize,linprog,lsq_linear
 l=np.linalg.lstsq(np.eye(2),np.array([1.,2.]),rcond=None)[0]
 ls=lsq_linear(np.eye(2),np.array([1.,2.]),bounds=(np.zeros(2),np.full(2,np.inf)),method='bvls',lsq_solver='exact',tol=1e-14,max_iter=400)
 opt=minimize(lambda x:float((x[0]-.5)**2),np.array([0.]),jac=lambda x:np.array([2*(x[0]-.5)]),method='SLSQP',options={'ftol':1e-12,'maxiter':400})
 lp=linprog([-1.],bounds=[(0.,1.)],method='highs')
 native_version=mujoco.mj_versionString()
 import_origins={name:dict(version=mod.__version__,file=mod.__file__,init_sha256=sha(mod.__file__)) for name,mod in (('numpy',np),('scipy',scipy),('mujoco',mujoco))}
 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_collision','mj_contactForce'):
  if hasattr(mujoco,name):setattr(mujoco,name,deny)
 class NoModel:
  def __new__(cls,*args,**kwargs):return deny()
  from_xml_path=staticmethod(deny);from_xml_string=staticmethod(deny);from_binary_path=staticmethod(deny)
 mujoco.MjModel=NoModel;mujoco.MjData=deny
 sys.path.insert(0,str(ROOT))
 from v13_a1.run_bridge import environment
 actual=environment();oldpath=ROOT/'review/v21_small_plan_v1.json';old=json.loads(oldpath.read_text('utf8'))['environment']
 def rolemap(packages):
  z={}
  for label,mod in packages:
   for pattern in ('*.pyd','*.dll'):
    for p in Path(mod.__file__).parent.rglob(pattern):z[label+'/'+p.relative_to(Path(mod.__file__).parent).as_posix()]=dict(path=str(p),sha256=sha(p))
  return z
 original_role={}
 for name,value in old['native_binaries_sha256'].items():
  path=Path(name);parts=path.parts
  for label in ('numpy','mujoco'):
   if label in parts:original_role[label+'/'+Path(*parts[parts.index(label)+1:]).as_posix()]=value;break
 current_role=rolemap((('numpy',np),('mujoco',mujoco)));diffs={k:dict(old=original_role.get(k),current=current_role.get(k)) for k in set(original_role)|set(current_role) if original_role.get(k)!=(current_role.get(k) or {}).get('sha256')}
 scipy_pins=rolemap((('scipy',scipy),))
 # Read the loaded-module paths in this process; no engine/data access.
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p;handle=kernel.GetCurrentProcess();buf=(ctypes.c_void_p*4096)();needed=ctypes.c_ulong();loaded={}
 psapi.EnumProcessModules.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong)];psapi.GetModuleFileNameExW.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_ulong]
 if psapi.EnumProcessModules(handle,buf,ctypes.sizeof(buf),ctypes.byref(needed)):
  for h in buf[:needed.value//ctypes.sizeof(ctypes.c_void_p)]:
   b=ctypes.create_unicode_buffer(32768)
   if psapi.GetModuleFileNameExW(handle,h,b,len(b)):
    p=Path(b.value)
    if any(p.is_relative_to(d) for d in (BIN,RESEARCH,VENV)) and p.is_file():loaded[str(p)]=sha(p)
 checks=dict(numpy_LS=np.array_equal(l,[1.,2.]),BVLS=np.array_equal(ls.x,[1.,2.]),SLSQP=opt.success and abs(opt.x[0]-.5)<1e-12,HiGHS=lp.success and lp.x[0]==1.,native_mujoco_version_matches_python=native_version==mujoco.__version__,no_forbidden_calls=not forbidden)
 r.update(complete=True,passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},extra_sys_path_append=str(extra),actual_imports=import_origins,mujoco_native_version=native_version,actual_environment=actual,old_environment_plan_sha256=sha(oldpath),old_environment_function_exact=actual==old,original42_role_file_count=len(original_role),current_role_file_count=len(current_role),all_original42_role_bytes_equal=not diffs,role_differences=diffs,scipy_additional_native_file_pins=scipy_pins,loaded_non_system_module_pins=loaded,source_environment_function_sha256=sha(ROOT/'v13_a1/run_bridge.py'),same_environment_is_not_a_physical_bridge=True)
except Exception as e:r.update(complete=False,passed=False,error=repr(e),traceback=traceback.format_exc())
out=ROOT/('review/v22_environment_coexistence_'+a.mode+'_v1.json')
with out.open('x',encoding='utf8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=r['passed'],mode=a.mode,checks=r.get('checks'),old_environment_exact=r.get('old_environment_function_exact'),original42_role_bytes_equal=r.get('all_original42_role_bytes_equal'),role_differences=len(r.get('role_differences',{})),extra_scipy_native_files=len(r.get('scipy_additional_native_file_pins',{})),loaded_non_system_modules=len(r.get('loaded_non_system_module_pins',{})),out=str(out),sha256=sha(out),error=r.get('error')),ensure_ascii=False),flush=True)
raise SystemExit(0 if r['passed'] else 1)

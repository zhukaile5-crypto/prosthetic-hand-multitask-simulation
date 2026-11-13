import os,sys,json,argparse
ap=argparse.ArgumentParser();ap.add_argument('--mode');ap.add_argument('--dllmode');a=ap.parse_args()
handles=[]
if a.dllmode=='directory':handles.append(os.add_dll_directory(r'D:\conda_envs\research312\Library\bin'))
elif a.dllmode=='path':os.environ['PATH']=r'D:\conda_envs\research312\Library\bin'+os.pathsep+os.environ.get('PATH','')
print(json.dumps({'stage':'before_import','mode':a.mode,'dllmode':a.dllmode,'executable':sys.executable}),flush=True)
import numpy as np
print('NUMPY_IMPORTED '+np.__version__,flush=True)
if a.mode=='numpy_lstsq':
 r=np.linalg.lstsq(np.array([[1.,0.],[0.,1.]]),np.array([1.,2.]),rcond=None);print('SUCCESS '+str(r[0].tolist()),flush=True)
elif a.mode=='lsq_linear':
 from scipy.optimize import lsq_linear
 print('SCIPY_IMPORTED',flush=True)
 r=lsq_linear(np.eye(2),np.array([1.,2.]),bounds=(np.zeros(2),np.full(2,np.inf)),method='bvls',lsq_solver='exact',tol=1e-14,max_iter=400);print('SUCCESS '+str(r.x.tolist()),flush=True)
else:
 from scipy.optimize import minimize
 print('SCIPY_IMPORTED',flush=True)
 r=minimize(lambda x:float((x[0]-1.)**2),np.array([0.]),jac=lambda x:np.array([2.*(x[0]-1.)]),method='SLSQP',options={'ftol':1e-12,'maxiter':400});print('SUCCESS '+str(r.x.tolist()),flush=True)

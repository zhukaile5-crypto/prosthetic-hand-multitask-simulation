"""Read-only AST identity and returned normal-dual re-evaluation, NumPy only."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'review/v13_joint_intent_qp_independent_final2df_v2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def tree_fn(path,name):
    return next(node for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8'))) if isinstance(node,ast.FunctionDef) and node.name==name)
def main():
    out=FOLDER/'final_certificate_v2.json'
    if out.exists():raise FileExistsError('Fresh final certificate required')
    old=ROOT/'review/v13_joint_intent_qp_independent_b115_v1/kernel_b115.py';new=FOLDER/'kernel_final2df.py'
    if sha(old)!='b115f01828a3d195e9846337d6c3435d797db0efde96e0ef364f17ab869dba86' or sha(new)!='2df172e57604b5c130b9cd7d274342994fbdfcd72a88e61e2a3fc615bd861418':raise ValueError('Audited kernel versions changed')
    checks=[]
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    for name in ('_array','_scalar','_project_z','_domain','project_normal_intents','projection','evaluate'):
        check(name+'_AST_unchanged',ast.dump(tree_fn(old,name),include_attributes=False)==ast.dump(tree_fn(new,name),include_attributes=False))
    old_simplex=tree_fn(old,'_simplex_cap');new_simplex=tree_fn(new,'_simplex_cap')
    branch=new_simplex.body.pop(2)
    expected_branch=ast.parse('if len(v) == 1:\n    return np.array([min(max(float(v[0]), 0.0), cap)], dtype=float)').body[0]
    check('singleton_shortcut_is_exact_declared_clip_formula',ast.dump(branch,include_attributes=False)==ast.dump(expected_branch,include_attributes=False))
    check('all_other_simplex_AST_unchanged',ast.dump(old_simplex,include_attributes=False)==ast.dump(new_simplex,include_attributes=False))
    kernel_spec=importlib.util.spec_from_file_location('independent_final_singleton',new);kernel=importlib.util.module_from_spec(kernel_spec);sys.modules[kernel_spec.name]=kernel;kernel_spec.loader.exec_module(kernel)
    for index,(value,cap) in enumerate(((-3.,2.),(-0.,2.),(0.,2.),(.5,2.),(3.,2.),(1e20,1e-12),(1.,1e-300),(1e-300,1.),(2.,0.))):
        actual=kernel._simplex_cap(np.array([value]),cap)
        expected=np.array([0. if value<=0 or cap==0 else cap if value>=cap else value])
        check('singleton_closed_form_'+str(index),np.array_equal(actual,expected))
    check('solver_signature_and_default_iterations_atol_rtol_unchanged',ast.dump(tree_fn(old,'solve_intent_qp').args,include_attributes=False)==ast.dump(tree_fn(new,'solve_intent_qp').args,include_attributes=False))
    helper=ROOT/'review/verify_v13_joint_intent_qp_independent_v1.py'
    spec=importlib.util.spec_from_file_location('independent_oracle_final_duals',helper);oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)
    math=read(FOLDER/'math268.json');direct=read(FOLDER/'direct_kkt29.json')
    check('math268_final_source_and_pass',math['passed'] and math['kernel_sha256']==sha(new) and len(math['checks'])==268)
    check('direct29_final_source_and_pass',direct['passed'] and direct['kernel_sha256']==sha(new) and len(direct['checks'])==29)
    details=[]
    for record in direct['records']:
        name=record['name'];P=len(record['zP']);k=oracle.fixture(P)
        if name=='large_L_roundoff':k['A21x25'][0,0]=1.;k['Jpx25'][0,1]=1.;k['scales']['residual25'][0]=2e8
        elif name=='narrow_z_cap_budget':k['z_intent_p']=np.array([5e-13]);k['caps5'][0]=1e-12;k['total_budget']=1e-12;k['support_coeff_p']=np.ones(1);k['load_proxy_N']=1.
        elif name=='narrow_d_box':
            k['A21x25'][0,0]=1.;k['Jpx25'][0,0]=1.;k['y_p']=np.ones(1);k['z_intent_p']=np.zeros(1);k['caps5'][0]=0.;k['total_budget']=0.;k['effort_lower21'][0]=-1e-12;k['effort_upper21'][0]=1e-12
        elif name=='multi_active_dual_complementarity':
            k['z_intent_p']=np.array([10.,9.,8.,7.,6.,0.]);k['patch_digit_p']=np.array([0,0,1,1,2,2]);k['caps5']=np.array([4.,4.,5.,10.,10.]);k['total_budget']=10.
            k['p0_21'][0]=5.;k['effort_lower21'][0]=3.;k['effort_upper21'][0]=4.;k['p0_21'][1]=-5.;k['effort_lower21'][1]=-4.;k['effort_upper21'][1]=-3.
        else:
            k['z_intent_p']=np.array([2.,1.]);k['patch_digit_p']=np.array([0,0])
            if name=='zero_group_cap':k['caps5'][0]=0.
            elif name=='zero_total_budget':k['total_budget']=0.
            elif name=='fixed_effort_box':k['effort_lower21'][0]=k['effort_upper21'][0]=1.
            else:raise ValueError('Unexpected checker record')
        d=np.asarray(record['d21']);z=np.asarray(record['zP']);x=np.r_[d,z];g=oracle.gradient(x,k)
        diag=record['diagnostics'];normal=diag['direct_normal_KKT'];eta=normal['global_multiplier'];lam=np.asarray(normal['group_multipliers'])
        shifted=g[21:]+eta+lam[np.asarray(k['patch_digit_p'],int)]
        lower_dual=np.where(z==0.,np.maximum(shifted,0.),0.)
        stationarity=float(np.max(abs(shifted-lower_dual),initial=0))
        sums=np.bincount(k['patch_digit_p'],weights=z,minlength=5)
        complementarity=float(max(abs(eta*(k['total_budget']-z.sum())),np.max(abs(lam*(k['caps5']-sums)),initial=0),np.max(abs(lower_dual*z),initial=0)))
        check(name+'_returned_dual_stationarity_recomputed',abs(stationarity-normal['stationarity_inf'])<1e-12)
        check(name+'_returned_dual_complementarity_recomputed',abs(complementarity-normal['complementarity_inf'])<1e-12)
        check(name+'_dual_nonnegative',eta>=0 and np.min(lam,initial=0)>=0 and np.min(lower_dual,initial=0)>=0)
        lo=k['effort_lower21']-k['p0_21'];hi=k['effort_upper21']-k['p0_21'];box=g[:21].copy()
        for i in range(21):
            if lo[i]==hi[i]:box[i]=0.
            elif d[i]==lo[i]:box[i]=min(g[i],0.)
            elif d[i]==hi[i]:box[i]=max(g[i],0.)
        check(name+'_box_stationarity_recomputed',abs(np.max(abs(box))-diag['direct_box_KKT']['stationarity_inf'])<1e-12)
        if record['available']:
            check(name+'_available_meets_direct_and_comp_tolerances',stationarity<=diag['numerical_tolerance_zP'] and np.max(abs(box))<=diag['numerical_tolerance_d21'] and complementarity<=diag['complementarity_tolerance'])
        details.append(dict(name=name,status=record['status'],iterations=record['iterations'],normal_stationarity=stationarity,box_stationarity=float(np.max(abs(box))),complementarity=complementarity))
    check('no_simulator_import','mujoco' not in sys.modules)
    result=dict(schema='v13_joint_final_independent_certificate_v2',passed=all(c['passed'] for c in checks),kernel_sha256=sha(new),
        checks=checks,failures=[c for c in checks if not c['passed']],returned_dual_rechecks=details,
        subaudit_sha256=dict(math268=sha(FOLDER/'math268.json'),direct29=sha(FOLDER/'direct_kkt29.json')),
        checker_sha256=sha(__file__),physics_steps=0,simulator_imported='mujoco' in sys.modules,
        reviewed_change='Direct KKT/stagnation plus exact singleton min/max projection shortcut; original objective/gradient/input/defaulttols and remaining projection AST unchanged',
        supersedes_only_assumption='v1 strict whole_simplex_AST equality failed because a mathematically equivalent singleton shortcut was added; v1 report and checker retained',
        scope='Pure numerical kernel certificate only; no native inverse, global model/pressure, physical performance, hard-real-time or safety admission')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=len(result['failures']),sha256=sha(out),out=str(out)),ensure_ascii=False))
if __name__=='__main__':main()

"""Read-only wrapper equivalence under an explicit namespace/label mapping.

No package import, model construction, execution, or tracker source evaluation.
"""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

class Normalize(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if isinstance(node.value,str):node.value=node.value.replace('v12_native','v12').replace('V12Native','V12')
        return node
    def visit_Name(self,node):node.id=node.id.replace('V12Native','V12');return node
    def visit_ClassDef(self,node):node.name=node.name.replace('V12Native','V12');return self.generic_visit(node)
    def visit_alias(self,node):
        node.name=node.name.replace('v12_native','v12').replace('V12Native','V12')
        return node
    def visit_ImportFrom(self,node):
        if node.module:node.module=node.module.replace('v12_native','v12')
        # New controller explicitly shares the frozen original observer/intents.
        if node.module in ('v12.observation','v12.intents'):
            node.module=node.module.split('.')[-1];node.level=1
        return self.generic_visit(node)

def main():
    records=[]
    for name in ('controller.py','runtime.py','run_development.py'):
        old=ROOT/'v12'/name;new=ROOT/'v12_native'/name;a=old.read_bytes();b=new.read_bytes()
        aa=ast.dump(Normalize().visit(ast.parse(a)),include_attributes=False)
        bb=ast.dump(Normalize().visit(ast.parse(b)),include_attributes=False)
        records.append(dict(file=name,old_sha256=hashlib.sha256(a).hexdigest(),new_sha256=hashlib.sha256(b).hexdigest(),
            normalized_AST_identical=aa==bb,stable_during_read=a==old.read_bytes() and b==new.read_bytes()))
    path=ROOT/'review/v12_native_wrapper_ast_independent_v1.json';assert not path.exists()
    value=dict(schema='v12_native_wrapper_ast_independent_v1',role='independent_read_only_peer',physics_steps=0,model_constructors=0,
        imports_executed=False,new_tracker_read=False,normalization='Only docstring expressions, v12_native/v12 and V12Native/V12 names/string labels, and explicit frozen observation/intents import binding',
        checks=records,passed=all(r['normalized_AST_identical'] and r['stable_during_read'] for r in records),
        limitation='Wrapper control logic equivalence only; no numerical tracker acceptance, source-closure freeze, physical-prefix or performance conclusion.')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passed=value['passed'],checks=len(records)),ensure_ascii=False))
    if not value['passed']:raise SystemExit(1)

if __name__=='__main__':main()

"""Read the draft. There is deliberately no --execute option or runner."""
from pathlib import Path
import argparse,json
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-draft',action='store_true');args=p.parse_args()
 plan=json.loads(Path(__file__).with_name('plan.json').read_text(encoding='utf-8'))
 missing=[k for k,v in plan['current_artifacts'].items() if v=='missing']
 print(json.dumps(dict(status=plan['status'],execution_allowed=False,missing=missing,measurement_output_artifacts_not_prerequisites=['response_kernel','mixed_input_error'],physics_steps=0),ensure_ascii=False))
 return 0 if args.check_draft else 2
if __name__=='__main__':raise SystemExit(main())

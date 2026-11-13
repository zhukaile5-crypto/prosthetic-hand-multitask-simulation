"""Small development-only probe of fusion's contact-preserving force floor."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_experiments import run_episode, code_hashes

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    out = Path(__file__).resolve().parent / 'floor_v1'
    out.mkdir(exist_ok=True)
    gains = json.loads((ROOT/'selected_gains.json').read_text(encoding='utf-8'))['fusion']
    gains['fusion_floor'] = 1.8
    results = []
    for scene in ('nominal','three_pulses','friction_drop','fragile_conflict','sensor_noise','local_limit'):
        row = run_episode(dict(scenario=scene,method='fusion',seed=1005,gains=gains,
                               output=str(out),save_trace=True))
        results.append(row)
        (out/'trials.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(scene, 'safe', row['safe_completed'], 'slip', row['max_slip_mm'],
              'excess',row['total_force_excess_Ns'],row['total_budget_excess_Ns'],flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(code_sha256=code_hashes(),gains=gains,
            seed=1005,development_only=True),indent=2),encoding='utf-8')

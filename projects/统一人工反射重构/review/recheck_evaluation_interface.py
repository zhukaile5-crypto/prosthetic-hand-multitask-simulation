"""Read existing development traces; no simulation or original-output mutation."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import evaluate_trial
from analyze import analyze_records, write_analysis
from plot_results import plot_results

source = ROOT / "output" / "controller_interface_v2"
if not source.exists():
    source = ROOT / "output" / "interface_dev_v1"
manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
records = []
for file in sorted((source / "traces").glob("*.npz")):
    with np.load(file, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        trace = {k: saved[k] for k in saved.files if k != "metadata"}
    records.append(evaluate_trial(trace, metadata, manifest["settings"].get("evaluation", {})))
result = analyze_records(records, bootstrap=5000)
output = ROOT / "review" / "updated_interface_analysis"
write_analysis(result, output)
figures = plot_results(result, output / "figures")
(output / "interface_scope.json").write_text(json.dumps(dict(
    source=str(source), source_manifest_unchanged=True, recomputation_only=True,
    simulations_run=0, trials=len(records), figures=figures,
    scope="old development input -> new independent evaluator; not confirmation efficacy"),
    ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(trials=len(records), out=str(output), simulations_run=0), ensure_ascii=False))

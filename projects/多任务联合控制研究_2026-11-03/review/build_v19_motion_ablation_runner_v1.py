"""One-time copy to a NEW namespace; never imports/executes the runner."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'v19_contact_reference/run_development.py'
target=ROOT/'v19_motion_ablation/run_development.py'
if target.exists():raise FileExistsError('retain all attempts; do not overwrite runner')
raw=source.read_bytes();text=raw.decode('utf-8')
if text.count('from v19_contact_reference.runtime import ')!=1:raise ValueError('unexpected parent runner import')
text=text.replace('v19_contact_reference','v19_motion_ablation').replace('V19ContactReference','V19MotionAblation')
text=text.replace('v19_release_runtime_or_evaluation_failure','v19_motion_ablation_runtime_or_evaluation_failure')
text=text.replace("preview_control='Frozen holding; new current5ms explicit-reference and native-branch contact-transfer task only for coordinated release. Original devices, plant, permissions, rates, metrics and16s schedule; no contact-force prediction'",
    "preview_control='Assigned FullV19 combined motion-effort-task ablation; same holding, progress, normal/reference QP, governor, branch, devices, plant, permissions, rates, metrics and16s schedule; actual observations retained and no force prediction'")
text=text.replace("source_namespace='V19MotionAblation full21 current-state task coordinator plus frozen A1/native parent closure and foundation assets/evidence'",
    "source_namespace='Assigned motion-task ablation with complete frozen FullV19 source parent plus four independent namespace files'")
with target.open('x',encoding='utf-8',newline='') as stream:stream.write(text)
if source.read_bytes()!=raw:raise RuntimeError('parent runner changed during copy')
print(json.dumps(dict(parent_sha256=hashlib.sha256(raw).hexdigest(),new_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    new_file=str(target),physics_dispatched=False)))

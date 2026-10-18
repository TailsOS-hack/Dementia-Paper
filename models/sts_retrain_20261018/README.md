# October 18 retraining inference states

Six validation-selected inference-state derivatives. Load with `torch.load(path, map_location="cpu", weights_only=True)`; construct the hash-pinned frozen MobileNetV3-Large helper with `network(pretrained=False)` and strictly load `model_state`. Load `aux_state` only when `aux_enabled` is true. Stage-only auxiliary heads were untrained and are omitted.

Run from the repository root with the research dependencies installed; construction does not download weights:

```python
import torch
from research.sts_retrain_20261018.train import load_legacy

path = "models/sts_retrain_20261018/stage_only_seed20261011_inference.pt"
state = torch.load(path, map_location="cpu", weights_only=True)
helper = load_legacy()
helper.imports()
model, aux = helper.network(pretrained=False)
model.load_state_dict(state["model_state"], strict=True)
model.eval()
if state["aux_enabled"]:
    aux.load_state_dict(state["aux_state"], strict=True)
    aux.eval()
```

The tensor names, shapes, dtypes and bytes are unchanged from the selected local originals. `provenance.json` records original and derivative file hashes, selected epochs and canonical tensor-content hashes. Optimizer, scheduler, RNG, config and histories are omitted; these files cannot resume training. No held-out performance or clinical-readiness claim follows from these weights.

"""CoreML model load and batched predict for Laya DecisionModel exports."""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np

COMPUTE_UNITS = {
    None: ct.ComputeUnit.ALL,
    "all": ct.ComputeUnit.ALL,
    "ane": ct.ComputeUnit.ALL,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "metal": ct.ComputeUnit.CPU_AND_GPU,
}


class CoreMLDecisionModel:
    """Thin wrapper around a converted DecisionModel.mlpackage.

    Exported graphs use fixed batch=1 for reliable ANE scheduling; this wrapper
    loops over the Python batch dimension.
    """

    def __init__(self, path, *, device=None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CoreML package missing: {path}")
        units = COMPUTE_UNITS.get(device if device is not None else "all")
        if units is None:
            raise ValueError(
                f"device must be one of None/'all'/'ane'/'cpu'/'gpu'/'metal'; got {device!r}"
            )
        self.model = ct.models.MLModel(str(path), compute_units=units)
        self.device = device if device is not None else "all"
        spec = self.model.get_spec()
        self.input_names = [i.name for i in spec.description.input]
        self.output_names = [o.name for o in spec.description.output]

    def predict(self, batch: dict) -> tuple[np.ndarray, np.ndarray]:
        """Run one collated batch; returns (logits, action) as float32 numpy arrays."""
        n = batch["input_ids"].shape[0]
        logits_rows, action_rows = [], []
        for i in range(n):
            feed = {}
            for name in self.input_names:
                value = batch[name][i : i + 1]
                arr = np.asarray(value)
                if name in ("input_ids", "marker_pos", "qtype"):
                    arr = arr.astype(np.int32)
                else:
                    arr = arr.astype(np.float32)
                feed[name] = arr
            out = self.model.predict(feed)
            logits_rows.append(np.asarray(out[self._pick_output("logits", 0)], dtype=np.float32))
            action_rows.append(np.asarray(out[self._pick_output("action", 1)], dtype=np.float32))
        return np.concatenate(logits_rows, axis=0), np.concatenate(action_rows, axis=0)

    def _pick_output(self, preferred: str, index: int) -> str:
        if preferred in self.output_names:
            return preferred
        if index < len(self.output_names):
            return self.output_names[index]
        raise KeyError(f"CoreML model is missing output {preferred!r}; have {self.output_names}")

"""Export the trained GAE to ONNX with fully **static** shapes (one file per bucket).

The architecture document forbids dynamic shapes for the low-latency / TensorRT
target: dynamic dims trigger CPU/GPU "ping-pong" and JIT kernel re-selection.
Instead we export one ONNX graph per static bucket (S/M/L). At runtime the Go
orchestrator pads the extracted ego-network up to the nearest bucket size (with
masked dummy nodes) and runs the matching engine.

For each bucket the exporter:
    1. builds an example input of the exact bucket shape,
    2. exports the eval-mode model (DropEdge off → deterministic graph),
    3. validates numerical parity against PyTorch via onnxruntime.

Usage::

    python -m graphagate.export_onnx [--atol 1e-4]
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from graphagate.config import DEFAULT_CONFIG, onnx_path
from graphagate.model.gae import DominantGAE
from graphagate.score import load_model

OUTPUT_NAMES = ["a_hat", "x_hat", "z"]
INPUT_NAMES = ["x", "adj"]


def _example_inputs(size: int, feature_dim: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a representative padded input of exact bucket shape ``size``.

    A random symmetric 0/1 adjacency (no self-loops) plus random [0,1] features —
    only the *shapes* are fixed by the export; values just drive the parity check.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(size, feature_dim, generator=g)
    a = (torch.rand(size, size, generator=g) < 0.05).float()
    a = torch.triu(a, diagonal=1)
    a = a + a.t()  # symmetric, zero diagonal
    return x, a


def export_bucket(
    model: DominantGAE, name: str, size: int, atol: float
) -> tuple[bool, float]:
    """Export one bucket to ONNX and verify parity. Returns (ok, max_abs_diff)."""
    feature_dim = model.cfg.feature_dim
    x, adj = _example_inputs(size, feature_dim)
    path = onnx_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    used = "dynamo"
    try:
        # Modern AOT exporter (torch.export based), as recommended by the document.
        torch.onnx.export(
            model,
            (x, adj),
            str(path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamo=True,
            opset_version=18,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        # Fallback to the legacy TorchScript exporter (static shapes, opset 17).
        used = f"legacy (dynamo failed: {type(exc).__name__})"
        torch.onnx.export(
            model,
            (x, adj),
            str(path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamo=False,
            opset_version=17,
        )

    # --- Parity check: PyTorch vs onnxruntime on identical input ---
    with torch.no_grad():
        torch_out = model(x, adj)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: x.numpy(), sess.get_inputs()[1].name: adj.numpy()}
    ort_out = sess.run(None, feed)

    max_diff = max(
        float(np.max(np.abs(t.numpy() - o)))
        for t, o in zip(torch_out, ort_out)
    )
    ok = max_diff < atol
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] bucket {name:>1s} (size {size:>3d}) "
          f"-> {path.name}  | exporter={used} | max|torch-ort|={max_diff:.2e}")
    return ok, max_diff


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the Graphagate GAE to ONNX per bucket.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Parity tolerance.")
    args = parser.parse_args(argv)

    model, _ = load_model()
    print("=" * 64)
    print("  Graphagate GAE — ONNX export (static buckets)")
    print("=" * 64)
    all_ok = True
    for name, size in sorted(DEFAULT_CONFIG.buckets.items(), key=lambda kv: kv[1]):
        ok, _ = export_bucket(model, name, size, args.atol)
        all_ok = all_ok and ok
    print("=" * 64)
    if all_ok:
        print("  All buckets exported and parity-checked successfully.")
    else:
        raise SystemExit("  ONNX parity check FAILED for at least one bucket.")


if __name__ == "__main__":
    main()

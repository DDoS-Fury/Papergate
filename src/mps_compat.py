"""Device selection and Apple-Silicon (MPS) compatibility shims.

Two responsibilities, both needed before any model is built:

1. :func:`resolve_device` — single place where the compute device is chosen, with a
   ``GRAPHAGATE_DEVICE`` environment override. Previously each entry point picked its own
   device inline and they had drifted apart (``train_tgn`` knew about MPS, ``serve_api`` and
   ``verify_tgn`` did not), and there was no way to force CPU without editing the source.

2. :func:`apply_mps_compat_patches` — works around a PyTorch Metal limitation that makes the
   TGN hot path unusable on Apple Silicon.

The MPS problem
---------------
PyTorch's Metal backend does not implement ``scatter_reduce_`` with ``reduce='amax'`` for
``torch.int64``::

    RuntimeError: not supported for torch.int64

``torch_geometric.nn.models.tgn`` hits it on every forward pass, on integer **timestamps**:

* ``TGNMemory._get_updated_memory`` → ``scatter(t, idx, 0, dim_size, reduce='max')``
* ``LastAggregator`` → ``scatter_argmax(t, index, ...)``

so ``python -m graphagate.train_tgn`` used to crash immediately on a Mac.

Why not simply cast to float32
------------------------------
The obvious workaround — run the reduction in ``float32`` and cast back — is **wrong for
timestamps**, and it is worth being explicit about it because that is the shape the fix took
while it lived in ``notebooks/train_and_eval.ipynb``.

``float32`` has a 24-bit mantissa, so it represents integers exactly only up to
``2**24 = 16_777_216``. Unix epoch timestamps are ~1.76e9 — two orders of magnitude past that,
where the spacing between representable values is 128 seconds. Measured::

    t        = [1758531600, 1758531603, 1758531607, 1758531611]   (two groups of two)
    exact    = [1758531603, 1758531611]
    float32  = [1758531584, 1758531584]     # both groups collapse onto the same value

The two groups become indistinguishable and every value is off by 19-27 s. Since ``last_update``
feeds every Δt in the model — pair recency, source activity, the time encoder — a silent error
of that size corrupts exactly the temporal signal the TGN exists to model. It would not crash;
it would just quietly make the Mac results differ from the CUDA ones.

The synthetic stream starts near zero and so would survive a while, but ``delta_t_cap`` is
604800 and a long-running server on real epoch timestamps would be wrong from the first event.

What we do instead
------------------
Run the unsupported reduction **on CPU** and move the result back. It is exact for any integer
magnitude, and the cost is negligible here: both call sites reduce a per-batch tensor of
timestamps (hundreds of elements), not the feature tensors. Float inputs, which Metal handles
natively, are never intercepted.
"""

from __future__ import annotations

import os

import torch

# Reductions Metal refuses for int64. 'min'/'max' are the PyG spellings, 'amin'/'amax' the
# torch.scatter_reduce_ ones; PyG forwards either depending on version.
_UNSUPPORTED_INT64_REDUCES = frozenset({"min", "max", "amin", "amax"})

_patched = False


def _needs_cpu_fallback(src: torch.Tensor) -> bool:
    return src.dtype == torch.int64 and src.device.type == "mps"


def _mps_available() -> bool:
    """True only on an Apple-Silicon build with a working Metal backend.

    Guarded with ``getattr`` rather than ``torch.backends.mps.is_available()`` directly: the
    attribute is present in every torch 2.x wheel we support, but a CPU-only or vendor build
    that omits it would otherwise raise ``AttributeError`` on Windows/Linux — platforms where
    the answer is simply "no".
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False


def apply_mps_compat_patches() -> bool:
    """Patch the PyG TGN scatter helpers for MPS. Idempotent; returns True if applied.

    Patches the names **bound inside** ``torch_geometric.nn.models.tgn``: that module does
    ``from torch_geometric.utils import scatter``, so rebinding ``torch_geometric.utils.scatter``
    would not affect it. Both wrappers are transparent whenever the fallback is not needed, so
    this is safe to call unconditionally.
    """
    global _patched
    if _patched:
        return False

    import torch_geometric.nn.models.tgn as pyg_tgn

    _orig_scatter = pyg_tgn.scatter
    _orig_scatter_argmax = pyg_tgn.scatter_argmax

    def _scatter(src, index, dim=0, dim_size=None, reduce="sum"):
        if reduce in _UNSUPPORTED_INT64_REDUCES and _needs_cpu_fallback(src):
            out = _orig_scatter(src.cpu(), index.cpu(), dim, dim_size, reduce)
            return out.to(src.device)
        return _orig_scatter(src, index, dim, dim_size, reduce)

    def _scatter_argmax(src, index, dim=0, dim_size=None):
        if _needs_cpu_fallback(src):
            out = _orig_scatter_argmax(src.cpu(), index.cpu(), dim=dim, dim_size=dim_size)
            return out.to(src.device)
        return _orig_scatter_argmax(src, index, dim=dim, dim_size=dim_size)

    pyg_tgn.scatter = _scatter
    pyg_tgn.scatter_argmax = _scatter_argmax
    _patched = True
    return True


def resolve_device(verbose: bool = True) -> torch.device:
    """Pick the compute device, honouring ``GRAPHAGATE_DEVICE``.

    ``GRAPHAGATE_DEVICE`` accepts any string ``torch.device`` understands (``cpu``, ``mps``,
    ``cuda``, ``cuda:1``); unset means auto-detect CUDA → MPS → CPU. An explicit request for a
    backend that is not available raises rather than silently falling back — a run that was
    meant to be on the GPU should not quietly become a CPU run.

    Applies :func:`apply_mps_compat_patches` whenever the chosen device is MPS.
    """
    requested = os.environ.get("GRAPHAGATE_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"GRAPHAGATE_DEVICE={requested!r} but CUDA is not available in this process."
            )
        if device.type == "mps" and not _mps_available():
            raise RuntimeError(
                f"GRAPHAGATE_DEVICE={requested!r} but the MPS backend is not available "
                "(needs macOS on Apple Silicon and a PyTorch build with Metal support)."
            )
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif _mps_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "mps":
        apply_mps_compat_patches()

    if verbose:
        origin = "GRAPHAGATE_DEVICE" if requested else "auto-detected"
        print(f"Using device: {device} ({origin})")
    return device

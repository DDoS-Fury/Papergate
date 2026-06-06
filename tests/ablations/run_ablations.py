"""Ablation study for the streaming TGN — isolates the two model-specific
components whose contribution is the actual novelty claim:

  * the **structural-compatibility head** (scaled cosine of projected embeddings),
    meant to carry the lateral-movement signal;
  * the **hashed-identity embedding** (deterministic per-entity identity).

It runs the full training+evaluation pipeline (``graphagate.train_tgn.train_tgn``)
three times on the same synthetic stream / seed — full model, no-struct-head,
no-hashed-identity — with ``save=False`` so the deployable checkpoint in ``public/``
is never touched, then prints a per-class comparison (lateral is the one that matters).

The temporal machinery itself (recurrent memory + temporal neighbourhood) is ablated
separately by the *Static GNN* baseline (``tests/baselines/simple_gnn``), which shares
the exact de-circularised curriculum.

Run inside the project's Docker image (torch required):

    docker compose --profile ablations up
"""

from graphagate.config import TGNConfig
from graphagate.train_tgn import train_tgn


def main():
    cfg = TGNConfig()
    runs = [
        ("full (struct+hash)", dict(use_struct_head=True, use_hash_identity=True)),
        ("no struct head", dict(use_struct_head=False, use_hash_identity=True)),
        ("no hashed identity", dict(use_struct_head=True, use_hash_identity=False)),
    ]

    results = []
    for name, flags in runs:
        print("\n" + "=" * 78)
        print(f"=== ABLATION: {name} ===")
        print("=" * 78)
        metrics = train_tgn(cfg, save=False, **flags)
        results.append((name, metrics))

    # --- summary table -------------------------------------------------------
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY (synthetic stream, FPR target {:.0%})".format(cfg.target_fpr))
    print("=" * 78)
    header = (
        f"{'variant':22s} | {'agg AUC':>7s} | {'agg AP':>6s} | "
        f"{'policy AUC':>10s} | {'ctx AUC':>7s} | {'lateral AUC':>11s} | {'lat Rec':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results:
        pt = m["per_type"]
        pol = pt.get("policy", {}).get("auc", float("nan"))
        ctx = pt.get("contextual", {}).get("auc", float("nan"))
        lat = pt.get("lateral", {}).get("auc", float("nan"))
        lat_rec = pt.get("lateral", {}).get("recall", float("nan"))
        print(
            f"{name:22s} | {m['agg_auc']:7.4f} | {m['agg_ap']:6.4f} | "
            f"{pol:10.4f} | {ctx:7.4f} | {lat:11.4f} | {lat_rec:7.4f}"
        )
    print(
        "\nReading: lateral AUC is the discriminating column — it is the only class that "
        "depends on the structural/temporal signal. Policy/contextual are near-saturated "
        "regardless (relational structure suffices)."
    )


if __name__ == "__main__":
    main()

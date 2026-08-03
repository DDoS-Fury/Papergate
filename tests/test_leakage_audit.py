"""Label-leakage audit of the synthetic generator.

    pytest tests/test_leakage_audit.py

The model's headline claim is that it detects lateral movement from the *structure and
timing* of the interaction graph. That claim is only meaningful if the synthetic task
cannot be solved without the graph. This module is the regression test for that
precondition, and it is deliberately strict: a dataset artifact that makes a class
trivially separable inflates every downstream number and is invisible in the metrics.

It caught three real defects when it was written:

  * ``node_feat[dst, 3]`` carried the raw resource index. Benign traffic concentrated on
    popular resources and popularity was the index, while attacks drew destinations
    uniformly — so that one column reached **AUC 0.92-0.94 on every anomaly class**,
    matching the model's own reported lateral AUC.
  * Per-class constants in the edge message (``bytes_in``/``bytes_out``/``http_status``)
    identified policy violations, credential theft, exfiltration and even benign service
    accounts with **100% precision and recall**.
  * Exfiltration was labelled as lateral movement, putting a sub-population separable by
    a single feature inside the class whose premise is that it has no feature tell.

Runs on the generator alone (no training), a few seconds per seed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data

SEEDS = [42, 7, 123]
N_EVENTS = 40_000

# Type id -> name. 0 is the benign reference class.
TYPE_NAMES = {
    1: "policy",
    2: "contextual",
    3: "lateral",
    4: "cred-theft",
    5: "exfil",
    6: "benign-denied",
}

# A single input column may not separate a class beyond this, unless allow-listed below.
MAX_SINGLE_FEATURE_AUC = 0.75

# Signals that ARE legitimately discriminative by design. Each entry is
# (type_id, source, column) -> rationale. These are not leaks: they are the observable
# evidence a deployed sensor/PDP genuinely has, and the paper reports them as the
# trivial-detector floor that the model must beat on the classes that matter.
ALLOWLIST: dict[tuple[int, str, int], str] = {
    # Contextual anomalies are the recon phase and are SUPPOSED to trip the IDS probes.
    # The rule baseline catches them; they are not the model's value-add.
    (2, "msg", 0): "ja3 validity bit — contextual recon uses an unknown TLS fingerprint",
    (2, "msg", 1): "Snort probe s1 — fires on 80% of recon events by design",
    (2, "msg", 2): "Snort probe s2",
    (2, "msg", 3): "Snort probe s3",
    # Exfiltration IS a massive transfer. It is a genuinely easy class, reported
    # separately precisely so it cannot flatter the lateral-movement numbers.
    (5, "msg", 7): "bytes_in — exfil moves data, that is what makes it exfil",
    (5, "msg", 8): "bytes_out — ditto",
    # Resource RISK is a real ZTA attribute known at decision time. Policy violations and
    # data theft target protected routes by definition, so the correlation is semantic.
    # It is bounded, reported as a floor, and identical for benign and attack traffic on
    # any given resource.
    (1, "nf_dst", 4): "resource risk — a policy violation is by definition on a protected route",
    (5, "nf_dst", 4): "resource risk — exfil targets the loot",
    (6, "nf_dst", 4): "resource risk — a benign OPA denial is also on a protected route",
    # The HTTP method is an input to the OPA decision itself: under Bell-LaPadula most
    # denials are write-downs, so writes are over-represented among denied requests. Both
    # denial classes are OPA-owned — the deterministic layer decides them and the model is
    # not claimed to add value there.
    (1, "msg", 4): "HTTP method — BLP denials are mostly writes; OPA decides this class",
    (6, "msg", 4): "HTTP method — ditto for benign denials",
}

# The classes the paper's contribution actually rests on. OPA cannot see them and the rule
# baseline is blind to them, so they get NO exemptions: every input column must stay under
# MAX_SINGLE_FEATURE_AUC. Enforced by test_critical_classes_have_no_allowlist_entries.
CRITICAL_TYPES = (3, 4)  # lateral movement, credential theft


def _stream(seed: int, n_events: int = N_EVENTS):
    cfg = TGNConfig(num_events=n_events, seed=seed)
    return generate_streaming_data(
        num_users=cfg.num_users,
        num_devices=cfg.num_devices,
        num_sources=cfg.num_sources,
        num_configs=cfg.num_configs,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        num_wipe_slots=cfg.num_wipe_slots,
        num_theft_slots=cfg.num_theft_slots,
        benign_explore_prob=cfg.benign_explore_prob,
        p_roam=cfg.p_roam,
        p_shared_device=cfg.p_shared_device,
        p_cookie_wipe=cfg.p_cookie_wipe,
        p_cred_theft=cfg.p_cred_theft,
        seed=cfg.seed,
        use_resource_risk=cfg.use_resource_risk,
        use_source_internal=cfg.use_source_internal,
        guest_device_fallback=cfg.guest_device_fallback,
    )


def _columns(s):
    """``{(source, column_index): value_per_event}`` for every scalar model input."""
    msg = s.msg.numpy()
    nf = s.node_features.numpy()
    dst = s.dst.numpy()
    user = s.user.numpy()
    cols = {("msg", j): msg[:, j] for j in range(msg.shape[1])}
    cols.update({("nf_dst", j): nf[dst, j] for j in range(nf.shape[1])})
    cols.update({("nf_user", j): nf[user, j] for j in range(nf.shape[1])})
    return cols


def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Two-sided AUC: a column that separates by being *low* leaks just as much."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    if np.ptp(values) == 0:
        return 0.5
    a = roc_auc_score(labels, values)
    return max(a, 1.0 - a)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_single_feature_shortcut(seed):
    """No individual input column may separate an anomaly class on its own."""
    s = _stream(seed)
    types = s.types.numpy()
    cols = _columns(s)
    benign = types == 0
    assert benign.sum() > 1000, "vacuous: too few benign events"

    violations = []
    for type_id, name in TYPE_NAMES.items():
        cls = types == type_id
        if cls.sum() < 30:  # too few to measure meaningfully
            continue
        sel = benign | cls
        labels = cls[sel].astype(int)
        for (src, j), values in cols.items():
            if (type_id, src, j) in ALLOWLIST:
                continue
            auc = _auc(labels, values[sel])
            if auc > MAX_SINGLE_FEATURE_AUC:
                violations.append(f"  {name:14s} {src}[{j}] AUC={auc:.4f} (n={int(cls.sum())})")

    assert not violations, (
        f"single-feature shortcut(s) at seed {seed} — a class is separable without the "
        f"graph, so any detection metric on it is a dataset artifact:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_no_exact_value_fingerprint(seed):
    """No exact value (or value pair) in the message may identify a class.

    Per-class constants let a model memorise a lookup table instead of learning
    behaviour, and they are invisible to the AUC check when the class is rare.
    """
    s = _stream(seed)
    types = s.types.numpy()
    msg = s.msg.numpy().round(6)

    # Single columns, and the (bytes_in, bytes_out) pair that used to be a class tell.
    candidates = [(j,) for j in range(msg.shape[1])] + [(7, 8)]

    violations = []
    for combo in candidates:
        keys = msg[:, combo]
        uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        for k in np.nonzero(counts >= 30)[0]:
            hit = inverse == k
            for type_id in list(TYPE_NAMES) + [0]:
                cls = types == type_id
                if cls.sum() == 0:
                    continue
                # A column that is allow-listed as a by-design signal for this class is not
                # a fingerprint: the Snort probes are *supposed* to fire on recon and only
                # on recon. Only flag combos with at least one non-allow-listed column.
                if all((type_id, "msg", j) in ALLOWLIST for j in combo):
                    continue
                precision = float((types[hit] == type_id).mean())
                recall = float(hit[cls].mean())
                if precision > 0.99 and recall > 0.5:
                    name = TYPE_NAMES.get(type_id, "benign")
                    violations.append(
                        f"  msg{list(combo)} == {uniq[k].tolist()} identifies {name}: "
                        f"precision={precision:.3f} recall={recall:.3f} n={int(hit.sum())}"
                    )

    assert not violations, (
        f"exact-value fingerprint(s) at seed {seed} — a constant identifies a class:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_destination_marginal_matches_benign(seed):
    """Benign and attack traffic must draw destinations from the same popularity law.

    If attacks land on unusual resources more often than benign traffic does, then
    "unusual destination" is a free label and lateral movement is detectable without ever
    consulting the interaction history — which is the entire premise of the model.
    Compared on the benign *popularity* of the destination, not on the resource id.
    """
    s = _stream(seed)
    types = s.types.numpy()
    dst = s.dst.numpy()
    benign = types == 0

    # Empirical popularity of each resource under benign traffic.
    counts = np.bincount(dst[benign] - s.res_lo, minlength=s.res_num).astype(float)
    popularity = counts / max(counts.sum(), 1.0)
    ben_pop = popularity[dst[benign] - s.res_lo]

    failures = []
    # Only lateral movement is tested here. Policy violations, exfiltration and credential
    # theft deliberately target protected routes — their destination marginal is shaped by
    # the policy model, not by the sampler, and that is semantic rather than an artifact.
    # Lateral movement has no such excuse: it draws from the same authorised action space
    # that benign exploration draws from.
    #
    # Judged on EFFECT SIZE, not on the p-value: with ~45k benign events against ~5k
    # lateral, a KS test rejects on differences far too small to be exploitable. A residual
    # gap is expected and legitimate — lateral targets the victim's *non-habitual* half of
    # the action space while benign traffic is mostly habitual, which is precisely the
    # phenomenon the model is meant to pick up. What matters is that it stays small enough
    # not to be readable off the global popularity of the destination, which is a static,
    # per-user-agnostic property available without any interaction history.
    max_ks = 0.15
    for type_id in (3,):
        cls = types == type_id
        if cls.sum() < 100:
            continue
        cls_pop = popularity[dst[cls] - s.res_lo]
        ks = stats.ks_2samp(ben_pop, cls_pop)
        if ks.statistic > max_ks:
            failures.append(
                f"  {TYPE_NAMES[type_id]}: KS={ks.statistic:.4f} (max {max_ks}) "
                f"p={ks.pvalue:.2e} (median popularity benign={np.median(ben_pop):.2e} "
                f"class={np.median(cls_pop):.2e})"
            )

    assert not failures, (
        f"destination marginal differs from benign at seed {seed} — 'unusual destination' "
        f"is a free label:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_route_method_pairs_are_served(seed):
    """Every event must use a method its route actually serves, attacks included.

    An unserved (route, method) pair is a region benign traffic never occupies, so it
    separates the class for free.
    """
    s = _stream(seed)
    msg = s.msg.numpy()
    dst = s.dst.numpy()
    types = s.types.numpy()

    from graphagate.data.stream_synthetic import build_resource_universe

    route_methods, _, resource_uris, _ = build_resource_universe(
        TGNConfig().num_resources - 19, seed
    )
    bad = []
    for type_id in list(TYPE_NAMES) + [0]:
        cls = types == type_id
        if cls.sum() == 0:
            continue
        idx = np.nonzero(cls)[0]
        served = np.array(
            [int(msg[i, 4]) in route_methods[resource_uris[dst[i] - s.res_lo]] for i in idx]
        )
        frac = 1.0 - served.mean()
        if frac > 0.001:
            bad.append(f"  {TYPE_NAMES.get(type_id, 'benign'):14s} {frac:.1%} unserved pairs")

    assert not bad, f"unserved (route, method) pairs at seed {seed}:\n" + "\n".join(bad)


def test_critical_classes_have_no_allowlist_entries():
    """Lateral movement and credential theft must never be granted an exemption.

    The allowlist is a legitimate escape hatch for signals that are discriminative by
    design, but it is also the obvious way to make this module pass without fixing
    anything. The two classes the contribution rests on are off-limits.
    """
    leaked = [k for k in ALLOWLIST if k[0] in CRITICAL_TYPES]
    assert not leaked, (
        f"allowlist entries exist for critical classes {CRITICAL_TYPES}: {leaked}. "
        f"These classes must be separable only through the interaction graph."
    )


def test_exfil_is_not_labelled_lateral():
    """Exfiltration must not be folded into the lateral-movement class.

    It carries a massive-transfer signal, so mixing the two puts a trivially separable
    sub-population inside the class the paper's central claim rests on.
    """
    s = _stream(SEEDS[0])
    types = s.types.numpy()
    msg = s.msg.numpy()
    lateral = types == 3
    assert lateral.sum() > 100, "vacuous: too few lateral events"

    # No lateral event may look like a bulk transfer.
    benign_out = msg[types == 0, 8]
    ceiling = float(benign_out.max())
    assert float(msg[lateral, 8].max()) <= ceiling, (
        "a lateral event carries a bulk-transfer volume above anything seen in benign "
        "traffic — exfil is leaking into the lateral class"
    )

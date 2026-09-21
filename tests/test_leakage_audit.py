"""Label-leakage audit of the synthetic generator.

    pytest tests/test_leakage_audit.py

The model's headline claim is that it detects lateral movement from the *structure and
timing* of the interaction graph. That claim is only meaningful if the synthetic task
cannot be solved without the graph. This module is the regression test for that
precondition, and it is deliberately strict: a dataset artifact that makes a class
trivially separable inflates every downstream number and is invisible in the metrics.

The three classes of defects it guards against:

  * A popularity-encoding column: a column carrying the raw resource index would reach
    **AUC 0.92-0.94 on every anomaly class**, matching the model's own reported lateral
    AUC, because benign traffic concentrates on popular resources (popularity is the
    index) while attacks draw destinations uniformly.
  * Per-class constants in the edge message (e.g. ``bytes_in``/``bytes_out``): they
    identify policy violations, credential theft, exfiltration and even benign service
    accounts with **100% precision and recall**.
  * Mislabelled volume events: labelling exfiltration as lateral movement puts a
    sub-population separable by a single feature inside the class whose premise is that
    it has no feature tell.
  * History shortcuts (v5): a single set-membership lookup — "this IP was never seen",
    "this role claim differs from the user's usual one" — must not solve a critical
    class either. The v4 generator passed every static check above while "IP never seen"
    reached AUC 1.000 on credential theft and a role-claim lookup caught 36% of lateral
    movement at zero false positives, because benign traffic lived in a closed world.

Runs on the generator alone (no training), at the training stream size: a vacuous pass
(too few events of a class to measure it) is itself a failure.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import lookup_flags
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

SEEDS = [42, 7, 123]
# The audit must see the stream the model is trained on: at 40k events the rarer classes
# fell below the per-class minimum and their checks were silently skipped.
N_EVENTS = TGNConfig().num_events
MIN_CLASS_EVENTS = 30          # whole stream, every class in TYPE_NAMES
MIN_TEST_EVENTS = 100          # test window, critical classes
# A single history lookup may not separate a critical class beyond this (per seed, paper
# protocol gating). A regression guard, not a target: over 11 seeds the worst v5 lookup
# is 0.71-0.80 (theft, src|usr_new / cfg|dev_new), while v4 reached 1.000. Their SUM is the
# stateful baseline the learned models must beat (scratch/generator_rule_audit.py).
MAX_SINGLE_LOOKUP_AUC = 0.85

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
# (type_id, source, column) -> rationale. These are observable signals available
# to a deployed PDP/sensor, representing the floor baseline detectors achieve.
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
    # denials are write-downs, so writes are over-represented among denied requests.
    (1, "msg", 4): "HTTP method — BLP denials are mostly writes; OPA decides this class",
    (6, "msg", 4): "HTTP method — ditto for benign denials",
}

# Critical target classes (lateral movement, credential theft) get NO exemptions:
# every input column must stay under MAX_SINGLE_FEATURE_AUC.
CRITICAL_TYPES = (3, 4)  # lateral movement, credential theft


@lru_cache(maxsize=len(SEEDS))
def _stream(seed: int, n_events: int = N_EVENTS):
    cfg = TGNConfig(num_events=n_events, seed=seed)
    return generate_streaming_data(**stream_kwargs_from_cfg(cfg))


def _columns(s):
    """``{(source, column_index): value_per_event}`` for every scalar model input:
    the message and the static features of all five endpoint nodes."""
    msg = s.msg.numpy()
    nf = s.node_features.numpy()
    cols = {("msg", j): msg[:, j] for j in range(msg.shape[1])}
    for role, ids in (("dst", s.dst), ("user", s.user), ("dev", s.device),
                      ("src", s.source), ("cfg", s.config)):
        ids = ids.numpy()
        cols.update({(f"nf_{role}", j): nf[ids, j] for j in range(nf.shape[1])})
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

    # Single columns, plus the (bytes_in, bytes_out) pair as a potential class tell.
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

    It carries a bulk-transfer volume signal, so mixing the two would introduce
    a trivial shortcut into the lateral-movement evaluation class.
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


@pytest.mark.parametrize("seed", SEEDS)
def test_every_class_is_measurable(seed):
    """Every class must occur often enough for the checks above to measure it.

    They skip classes below a minimum count, so a generator whose attack slots run out,
    or whose base rate is lowered, would pass them vacuously. The critical classes must
    also reach the test window — the v4 default config put zero credential-theft events
    there, so its per-class theft metric was computed on a different, hand-tuned stream.
    """
    s = _stream(seed)
    types = s.types.numpy()
    counts = {name: int((types == k).sum()) for k, name in TYPE_NAMES.items()}
    thin = {n: c for n, c in counts.items() if c < MIN_CLASS_EVENTS}
    assert not thin, f"classes too rare to audit at seed {seed}: {thin}"

    cfg = TGNConfig()
    test = types[int(len(types) * (cfg.train_frac + cfg.val_frac)):]
    thin = {TYPE_NAMES[k]: int((test == k).sum()) for k in CRITICAL_TYPES
            if (test == k).sum() < MIN_TEST_EVENTS}
    assert not thin, f"critical classes too rare in the test window at seed {seed}: {thin}"


def test_role_claim_matches_identity():
    """The role/clearance in the message is always the identity's real one.

    User roles never change, so any disagreement between the role claim and the user's
    history is a zero-false-positive rule. v4 spoofed it on half of lateral movement.
    A stolen identity is modelled as a stolen *user* (theft / credential pivot) instead.
    """
    s = _stream(SEEDS[0])
    user = s.user.numpy()
    role = s.msg.numpy()[:, 5]
    order = np.argsort(user, kind="stable")
    u, r = user[order], role[order]
    same_user = u[1:] == u[:-1]
    changed = same_user & (r[1:] != r[:-1])
    assert not changed.any(), (
        f"{int(changed.sum())} events carry a role claim that differs from the same "
        f"user's previous one — a zero-false-positive rule tell"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_no_single_lookup_shortcut(seed):
    """No single history lookup may separate a critical class on its own.

    Scored like the paper (benign-vs-class AUC in the test window) with the paper's
    memory protocol: rules remember labelled-benign history before the test window and
    predicted-benign events inside it (graphagate.data.lookup_rules, gate="proto-self").
    """
    s = _stream(seed)
    cfg = TGNConfig()
    types = s.types.numpy()
    test_start = int(len(types) * (cfg.train_frac + cfg.val_frac))
    flags = lookup_flags(s, "proto-self", test_start)
    te = np.arange(len(types)) >= test_start
    benign = te & (types == 0)

    violations = []
    for type_id in CRITICAL_TYPES:
        cls = te & (types == type_id)
        sel = benign | cls
        for rule in ("cfg_new", "src_new", "dev_new", "cfg|dev_new", "cfg|usr_new",
                     "dev|usr_new", "src|usr_new", "role_changed"):
            auc = _auc(cls[sel].astype(int), flags[rule][sel].astype(float))
            if auc > MAX_SINGLE_LOOKUP_AUC:
                violations.append(f"  {TYPE_NAMES[type_id]:10s} {rule:13s} AUC={auc:.4f} "
                                  f"(n={int(cls.sum())})")
    assert not violations, (
        f"single history-lookup shortcut(s) at seed {seed} — a dict lookup solves the "
        f"class, so it measures the generator's closed world, not the model:\n"
        + "\n".join(violations)
    )

"""Baseline GNN *non temporale* — ablation del TGN per la ZTA anomaly detection.

Perché esiste questo file
-------------------------
Ablation EQUA del TGN che isola la SOLA componente temporale. La baseline mantiene
lo stesso identico curriculum di negative sampling del TGN **de-circolarizzato** —
negativo strutturale a destinazione casuale + negativo contestuale a rumore
gaussiano, pesi uguali — e la stessa idea di link-prediction; rimuove unicamente la
macchina TEMPORALE (memoria ricorrente per nodo + vicinato temporale), sostituita da
un GNN su un grafo **statico** aggregato dalle interazioni benigne di train. Il delta
col TGN misura quindi quanto valga la dinamica temporale, a parità di curriculum.

NB (de-circolarizzazione): i negativi NON usano più l'abitualità/autorizzazione
(`adj`/`auth_mask`) né l'hard-negative ×10. Quella costruzione rispecchiava
esattamente la definizione di lateral movement del test (accesso autorizzato ma non
abituale), gonfiando la recall in modo circolare. La destinazione negativa è ora
pescata uniformemente sulle risorse, come nel TGN aggiornato.

Protocollo (identico al TGN per confrontabilità 1:1, vedi tests/baselines/README.md):
stessi dati/seed, stesso split cronologico 70/10/20, training solo su benigno,
soglia calibrata sul benigno di validazione al ``target_fpr``, metriche aggregate
+ breakdown per tipo sul segmento di test.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.eval_common import causal_hist_features, causal_precursor_factor


class StaticGNN(nn.Module):
    """2-layer GraphSAGE encoder sul grafo statico + link predictor MLP.

    Volutamente *semplice* rispetto al TGN: niente memoria ricorrente, niente
    vicinato temporale, niente testa strutturale a coseno, niente embedding di
    identità hashata. Gli embedding ``z`` dei nodi sono funzione delle sole
    feature statiche propagate sul grafo aggregato dal benigno di train.
    """

    def __init__(self, node_feat_dim, msg_dim, hidden=64, hist_dim=3):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

        # Link predictor: MLP su [z_src ‖ z_dst ‖ msg ‖ hist] -> 1 logit.
        # Analogo SEMPLICE del LinkPredictor del TGN, senza le feature statiche
        # concatenate né la testa a coseno: lo scopo è tenere il modello minimale per
        # misurare l'apporto della parte temporale — ma riceve le STESSE feature di
        # storia (conteggi causali) del TGN, così il confronto è equo.
        self.lin1 = nn.Linear(hidden * 2 + msg_dim + hist_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def encode(self, x, edge_index):
        """Embedding dei nodi via message passing sul grafo statico."""
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def link_pred(self, z_src, z_dst, msg, hist):
        """Logit di benignità per l'arco src->dst che trasporta ``msg`` (+ feature storia)."""
        h = torch.cat([z_src, z_dst, msg, hist], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h).squeeze(-1)


def _binary_metrics(scores, labels, threshold):
    """Precision / recall di ``score >= threshold`` contro ``labels``.

    Replica esatta di ``train_tgn._binary_metrics`` per confrontabilità.
    """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _score_events(model, z, src, dst, msg, hist, device, precursor_fac=None):
    """Score di anomalia per ogni evento (s, d, msg, hist).

    Orientamento "più alto = più anomalo", coerente con ``serve_tgn.infer_score``
    (= ``1 - P(benign)``): usiamo gli embedding ``z`` FISSI post-training
    (transduttivo sul grafo di train) e calcoliamo ``1 - sigmoid(link_pred)``, poi
    applichiamo lo stesso prior moltiplicativo del precursor kill-chain del TGN.
    """
    model.eval()
    with torch.no_grad():
        z_src = z[src.to(device)]
        z_dst = z[dst.to(device)]
        logits = model.link_pred(z_src, z_dst, msg.to(device), hist.to(device))
        scores = 1.0 - torch.sigmoid(logits)
    out = scores.cpu().numpy().astype(np.float64)
    if precursor_fac is not None:
        out = out * precursor_fac
    return out


def run(cfg: TGNConfig = TGNConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating synthetic streaming data (TGN params)...")
    stream = generate_streaming_data(
        num_users=cfg.num_users,
        num_devices=cfg.num_devices,
        num_sources=cfg.num_sources,
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
    )
    # Tabular actor = the DEVICE node (hardware id), the v2 analogue of the old
    # IP-keyed src; the access target stays the resource.
    src, dst, t, msg, y, types, node_features = (
        stream.device, stream.dst, stream.t, stream.msg, stream.y, stream.types,
        stream.node_features,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total_nodes = cfg.total_nodes

    # Causal interaction-history features (stessa info tabellare del TGN) + prior del
    # precursor kill-chain: dati alla baseline così il confronto isola la sola dinamica.
    hist_all = torch.tensor(
        causal_hist_features(src.numpy(), dst.numpy(), y.numpy()), dtype=torch.float
    )
    precursor_fac = causal_precursor_factor(
        src.numpy(), t.numpy(), msg.numpy(), cfg.precursor_half_life, cfg.precursor_max_boost
    )

    # Split cronologico identico al TGN (lo stream è già ordinato nel tempo).
    n = len(src)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_end, val_end = n_train, n_train + n_val

    # --- GRAFO STATICO dal solo benigno di train -----------------------------
    # Un unico grafo aggregato e immutabile: gli archi sono le interazioni benigne
    # del segmento di train (y==0), resi NON orientati duplicando le direzioni.
    # Questa aggregazione "appiattisce" la cronologia: è proprio l'informazione
    # che il TGN conserva e questa ablation no.
    tr_src = src[:train_end]
    tr_dst = dst[:train_end]
    tr_msg = msg[:train_end]
    tr_y = y[:train_end]

    benign = tr_y == 0
    b_src = tr_src[benign]
    b_dst = tr_dst[benign]
    b_msg = tr_msg[benign]
    b_hist = hist_all[:train_end][benign]  # causal history features dei positivi benigni

    # edge_index non orientato [2, 2*E]: entrambe le direzioni.
    edge_index = torch.stack(
        [
            torch.cat([b_src, b_dst]),
            torch.cat([b_dst, b_src]),
        ],
        dim=0,
    ).to(device)

    x = node_features.to(device)  # [total_nodes, 16] feature statiche

    model = StaticGNN(node_feat_dim=cfg.node_feat_dim, msg_dim=cfg.msg_dim, hidden=64).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Archi benigni di train su device per il training (positivi).
    p_src = b_src.to(device)
    p_dst = b_dst.to(device)
    p_msg = b_msg.to(device)
    p_hist = b_hist.to(device)
    num_pos = p_src.shape[0]
    bs = cfg.batch_size

    # Range id delle risorse per il campionamento dei negativi strutturali (la
    # destinazione negativa è pescata uniformemente su [res_lo, total_nodes)).
    num_res = cfg.num_resources
    res_lo = total_nodes - num_res                      # primo id globale di risorsa

    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED (GNN non temporale) ---")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # Permutazione degli archi benigni a ogni epoca (mini-batch SGD).
        perm = torch.randperm(num_pos, device=device)
        total_loss = 0.0
        num_batches = max(num_pos // bs, 1)

        for i in range(num_batches):
            optimizer.zero_grad()
            idx = perm[i * bs : i * bs + bs]
            if idx.numel() == 0:
                continue

            bp_src = p_src[idx]
            bp_dst = p_dst[idx]
            bp_msg = p_msg[idx]
            bp_hist = p_hist[idx]

            # Ricalcola gli embedding sul grafo statico fisso: il grafo non cambia,
            # ma z evolve man mano che i pesi del GNN si aggiornano.
            z = model.encode(x, edge_index)

            # POSITIVO: l'arco reale (src, dst, msg, hist) -> 1.
            pos_logit = model.link_pred(z[bp_src], z[bp_dst], bp_msg, bp_hist)

            # NEGATIVO strutturale: (src, risorsa casuale, msg) -> 0. Identico al TGN
            # de-circolarizzato: destinazione pescata UNIFORMEMENTE sulle risorse, SENZA
            # usare abitualità/autorizzazione (che rispecchierebbero la definizione di
            # lateral del test). Niente più hard-negative ×10.
            neg_dst = torch.randint(0, num_res, (idx.numel(),), device=device) + res_lo
            collide = neg_dst == bp_dst
            if collide.any():
                neg_dst[collide] = (
                    torch.randint(0, num_res, (int(collide.sum()),), device=device) + res_lo
                )
            # Per una destinazione casuale la coppia (src, dst) non è quasi mai stata
            # vista → le feature di storia del negativo sono ~zero (come per il TGN).
            neg_hist = torch.zeros_like(bp_hist)
            neg_logit = model.link_pred(z[bp_src], z[neg_dst], bp_msg, neg_hist)

            # NEGATIVO contestuale: rumore gaussiano additivo sul msg (meccanismo
            # DIVERSO dalla randomizzazione 0/1 delle anomalie contestuali di test).
            neg_msg = bp_msg + torch.randn_like(bp_msg) * 0.5
            ctx_logit = model.link_pred(z[bp_src], z[bp_dst], neg_msg, bp_hist)

            # Stessa loss del TGN de-circolarizzato: pos vs strutturale + contestuale,
            # pesi uguali (niente più ×10 sull'hard-negative).
            loss = (
                criterion(pos_logit, torch.ones_like(pos_logit))
                + criterion(neg_logit, torch.zeros_like(neg_logit))
                + criterion(ctx_logit, torch.zeros_like(ctx_logit))
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss / num_batches:.4f}")

    # Embedding FISSI post-training (transduttivi sul grafo di train); usati per
    # validazione, calibrazione soglia e test — come da spec.
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)

    # --- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) --------------
    print("\n--- CALIBRAZIONE SOGLIA (su flusso di validazione benigno) ---")
    val_scores = _score_events(
        model, z, src[train_end:val_end], dst[train_end:val_end], msg[train_end:val_end],
        hist_all[train_end:val_end], device, precursor_fac[train_end:val_end],
    )
    val_labels = y[train_end:val_end].numpy()
    benign_val_scores = val_scores[val_labels == 0]
    if benign_val_scores.size == 0:
        raise RuntimeError("No benign events in the validation slice for calibration.")
    threshold = float(np.quantile(benign_val_scores, 1.0 - cfg.target_fpr))
    print(
        f"Benign val score: mean={benign_val_scores.mean():.4f} "
        f"p95={np.quantile(benign_val_scores, 0.95):.4f} | "
        f"threshold@FPR={cfg.target_fpr}: {threshold:.4f}"
    )

    # --- INFERENZA / ANOMALY DETECTION sul test ------------------------------
    print("\n--- INIZIO FASE DI INFERENZA / ANOMALY DETECTION ---")
    test_scores = _score_events(
        model, z, src[val_end:], dst[val_end:], msg[val_end:],
        hist_all[val_end:], device, precursor_fac[val_end:],
    )
    test_labels = y[val_end:].numpy()
    test_types = types[val_end:].numpy()

    auc = roc_auc_score(test_labels, test_scores)
    ap = average_precision_score(test_labels, test_scores)
    precision, recall = _binary_metrics(test_scores, test_labels, threshold)
    print(f"Test Stream | AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"At threshold {threshold:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # --- BREAKDOWN PER TIPO DI ANOMALIA --------------------------------------
    # Calcolato benigno (tipo 0) vs quel-tipo, identico al TGN: un aggregato non
    # può nascondere una classe gestita male. Atteso: policy/contextual buoni,
    # lateral debole (manca il segnale temporale).
    print("\n--- METRICHE PER TIPO DI ANOMALIA ---")
    benign_mask = test_types == 0
    for type_id, name in ((1, "policy    "), (2, "contextual"), (3, "lateral   ")):
        sel = benign_mask | (test_types == type_id)
        s_sel, l_sel = test_scores[sel], (test_types[sel] == type_id).astype(int)
        if l_sel.sum() == 0:
            continue
        t_auc = roc_auc_score(l_sel, s_sel)
        t_ap = average_precision_score(l_sel, s_sel)
        _, t_recall = _binary_metrics(s_sel, l_sel, threshold)
        print(
            f"  {name} | n={int(l_sel.sum()):4d} | AUC: {t_auc:.4f} | "
            f"AP: {t_ap:.4f} | Recall@thr: {t_recall:.4f}"
        )


def main():
    run()


if __name__ == "__main__":
    main()

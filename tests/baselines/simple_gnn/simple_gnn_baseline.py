"""Baseline GNN *non temporale* — ablation del TGN per la ZTA anomaly detection.

Perché esiste questo file
-------------------------
Ablation EQUA del TGN che isola la SOLA componente temporale. La baseline mantiene
lo stesso identico curriculum di negative sampling del TGN — negativo strutturale,
hard-negative ×10 (risorsa NON abituale) e negativo contestuale (rumore 20%) — e
la stessa idea di link-prediction; rimuove unicamente la macchina TEMPORALE
(memoria ricorrente per nodo + vicinato temporale), sostituita da un GNN su un
grafo **statico** aggregato dalle interazioni benigne di train. Il delta col TGN
misura quindi quanto valga la dinamica temporale, a parità di curriculum.

L'"abitualità" qui è definita sul grafo statico: una coppia (IP, risorsa) è
abituale se esiste un arco benigno di train fra le due. L'hard-negative accoppia
ogni IP con una risorsa a cui NON è connesso — lo stesso segnale che a test
identifica il lateral movement (accesso autorizzato ma non abituale). È quindi una
verifica forte: se persino con questo curriculum la GNN statica non aggancia il
lateral, l'informazione mancante è davvero la cronologia temporale e non il
semplice fatto di non essere mai stata addestrata a cercarla.

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


class StaticGNN(nn.Module):
    """2-layer GraphSAGE encoder sul grafo statico + link predictor MLP.

    Volutamente *semplice* rispetto al TGN: niente memoria ricorrente, niente
    vicinato temporale, niente testa strutturale a coseno, niente embedding di
    identità hashata. Gli embedding ``z`` dei nodi sono funzione delle sole
    feature statiche propagate sul grafo aggregato dal benigno di train.
    """

    def __init__(self, node_feat_dim, msg_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(0.1)

        # Link predictor: MLP su [z_src ‖ z_dst ‖ msg] -> 1 logit.
        # Analogo SEMPLICE del LinkPredictor del TGN, senza le feature statiche
        # concatenate né la testa a coseno: lo scopo è proprio tenere il modello
        # minimale per misurare l'apporto della parte temporale.
        self.lin1 = nn.Linear(hidden * 2 + msg_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def encode(self, x, edge_index):
        """Embedding dei nodi via message passing sul grafo statico."""
        h = self.conv1(x, edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def link_pred(self, z_src, z_dst, msg):
        """Logit di benignità per l'arco src->dst che trasporta ``msg``."""
        h = torch.cat([z_src, z_dst, msg], dim=-1)
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


def _score_events(model, z, src, dst, msg, device):
    """Score di anomalia per ogni evento (s, d, msg).

    Orientamento "più alto = più anomalo", coerente con ``serve_tgn.infer_score``
    (= ``1 - P(benign)``): usiamo gli embedding ``z`` FISSI post-training
    (transduttivo sul grafo di train) e calcoliamo ``1 - sigmoid(link_pred)``.
    """
    model.eval()
    with torch.no_grad():
        z_src = z[src.to(device)]
        z_dst = z[dst.to(device)]
        logits = model.link_pred(z_src, z_dst, msg.to(device))
        scores = 1.0 - torch.sigmoid(logits)
    return scores.cpu().numpy().astype(np.float64)


def run(cfg: TGNConfig = TGNConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print("Generating streaming data...")
    src, dst, t, msg, y, types, node_features, resource_uris = generate_streaming_data(
        num_users=cfg.num_users,
        num_ips=cfg.num_ips,
        num_resources=cfg.num_resources,
        num_events=cfg.num_events,
        seed=cfg.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total_nodes = cfg.total_nodes

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
    num_pos = p_src.shape[0]
    bs = cfg.batch_size

    # Adiacenza IP->risorsa ABITUALE, definita sul grafo statico: ``adj[ip, r]`` è
    # True se esiste un arco benigno di train fra l'IP e la risorsa locale ``r``.
    # È l'analogo statico della "storia" che il neighbour loader dà al TGN, e serve
    # a costruire l'hard-negative (risorsa NON abituale) con lo stesso significato.
    num_res = cfg.num_resources
    res_lo = total_nodes - num_res                      # primo id globale di risorsa
    adj = torch.zeros(total_nodes, num_res, dtype=torch.bool, device=device)
    adj[p_src, p_dst - res_lo] = True                   # p_dst sono risorse (>= res_lo)

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

            # Ricalcola gli embedding sul grafo statico fisso: il grafo non cambia,
            # ma z evolve man mano che i pesi del GNN si aggiornano.
            z = model.encode(x, edge_index)

            # POSITIVO: l'arco reale (src, dst, msg) -> 1.
            pos_logit = model.link_pred(z[bp_src], z[bp_dst], bp_msg)

            # NEGATIVO strutturale: (src, nodo casuale, msg) -> 0, pescato in
            # [num_users, total_nodes) (IP+risorse), come il TGN.
            neg_dst = torch.randint(cfg.num_users, total_nodes, (idx.numel(),), device=device)
            neg_logit = model.link_pred(z[bp_src], z[neg_dst], bp_msg)

            # HARD-NEGATIVE ×10: (src, risorsa NON abituale, msg) -> 0. È il segnale
            # del lateral movement, identico nello spirito a quello del TGN ma con
            # l'abitualità letta dal grafo statico. Si pesca, per ogni src, una
            # risorsa con adj==False (rumore casuale, posizioni abituali messe a -1).
            occ = adj[bp_src]                                  # [B, num_res] abituali
            pick = torch.rand(idx.numel(), num_res, device=device)
            pick[occ] = -1.0
            hard_dst = pick.argmax(dim=1) + res_lo
            hard_logit = model.link_pred(z[bp_src], z[hard_dst], bp_msg)

            # NEGATIVO contestuale: stessa dst, 20% dei bit del msg invertiti (come il TGN).
            neg_msg = bp_msg.clone()
            noise_mask = torch.rand_like(neg_msg) < 0.20
            neg_msg[noise_mask] = 1.0 - neg_msg[noise_mask]
            ctx_logit = model.link_pred(z[bp_src], z[bp_dst], neg_msg)

            # Stessa loss del TGN: pos->1, negativi->0, con hard-negative pesato ×10.
            loss = (
                criterion(pos_logit, torch.ones_like(pos_logit))
                + criterion(neg_logit, torch.zeros_like(neg_logit))
                + 10.0 * criterion(hard_logit, torch.zeros_like(hard_logit))
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
        model, z, src[train_end:val_end], dst[train_end:val_end], msg[train_end:val_end], device
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
        model, z, src[val_end:], dst[val_end:], msg[val_end:], device
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

import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.model.tgn import ZTATemporalGraphNetwork

def train_tgn():
    print("Generating streaming data...")
    num_users = 50
    num_ips = 100
    num_resources = 20
    total_nodes = num_users + num_ips + num_resources
    
    # Generiamo un grande stream di dati
    src, dst, t, msg, y, node_features = generate_streaming_data(
        num_users=num_users, num_ips=num_ips, num_resources=num_resources, num_events=50000
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    msg_dim = 6
    
    model = ZTATemporalGraphNetwork(
        num_nodes=total_nodes,
        node_feat_dim=16,
        msg_dim=msg_dim,
        memory_dim=64,
        time_dim=32
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    batch_size = 200
    
    # Splitto i dati: l'80% per addestrare i comportamenti sani (Unsupervised), il 20% per testare se le anomalie vengono riconosciute
    train_size = int(len(src) * 0.8)
    
    print("--- INIZIO ADDESTRAMENTO UNSUPERVISED ---")
    for epoch in range(1, 4):
        model.memory.reset_state() # Reset memoria all'inizio dell'epoca
        model.train()
        
        total_loss = 0
        num_train_batches = train_size // batch_size
        
        for i in range(num_train_batches):
            optimizer.zero_grad()
            
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            b_src = src[start_idx:end_idx].to(device)
            b_dst = dst[start_idx:end_idx].to(device)
            b_t = t[start_idx:end_idx].to(device)
            b_msg = msg[start_idx:end_idx].to(device)
            b_y = y[start_idx:end_idx].to(device)
            
            # 1. Filtro i dati: la rete in addestramento vede SOLO traffico benigno per creare la sua "baseline"
            benign_mask = (b_y == 0)
            if not benign_mask.any(): continue
            
            p_src = b_src[benign_mask]
            p_dst = b_dst[benign_mask]
            p_t = b_t[benign_mask]
            p_msg = b_msg[benign_mask]
            
            # --- POSITIVE EDGES (Comportamento Sano) ---
            n_id, inverse_indices = torch.unique(torch.cat([p_src, p_dst]), return_inverse=True)
            local_src = inverse_indices[:len(p_src)]
            local_dst = inverse_indices[len(p_src):]
            pos_edge_index = torch.stack([local_src, local_dst], dim=0)
            
            pos_out = model(n_id, pos_edge_index, p_t, p_msg).squeeze(-1)
            
            # --- NEGATIVE EDGES (Comportamento Falso/Anomalo generato al volo) ---
            # Randomizziamo le risorse di destinazione per simulare connessioni che l'utente non ha richiesto
            neg_dst = torch.randint(0, total_nodes, (len(p_src),), device=device)
            
            n_id_neg, inverse_indices_neg = torch.unique(torch.cat([p_src, neg_dst]), return_inverse=True)
            local_src_neg = inverse_indices_neg[:len(p_src)]
            local_dst_neg = inverse_indices_neg[len(p_src):]
            neg_edge_index = torch.stack([local_src_neg, local_dst_neg], dim=0)
            
            # Valuto la probabilità degli archi finti (Anomalie Strutturali)
            neg_out_struct = model(n_id_neg, neg_edge_index, p_t, p_msg).squeeze(-1) 
            
            # --- NEGATIVE EDGES CONTESTUALI (Anomalie di Feature) ---
            # La rete deve imparare che se vede un Alert Snort (1.0) o un JA3 cattivo (0.0), è un'anomalia,
            # anche se la struttura (sorgente-destinazione) è formalmente corretta.
            neg_msg = p_msg.clone()
            neg_msg[:, 0] = 0.0 # JA3 malevolo
            neg_msg[:, 1] = 1.0 # Snort alert
            neg_out_ctx = model(n_id, pos_edge_index, p_t, neg_msg).squeeze(-1)
            
            # --- LOSS UNSUPERVISED ---
            # Positivo -> 1, Negativi -> 0
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss_struct = criterion(neg_out_struct, torch.zeros_like(neg_out_struct))
            neg_loss_ctx = criterion(neg_out_ctx, torch.zeros_like(neg_out_ctx))
            
            loss = pos_loss + neg_loss_struct + neg_loss_ctx
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Aggiorno la memoria dell'IP solo con il traffico buono accaduto
            model.memory.update_state(p_src, p_dst, p_t, p_msg)
            model.memory.detach()
            
        print(f"Epoch {epoch:02d} | Train Loss: {total_loss/num_train_batches:.4f}")
        
    print("\n--- INIZIO FASE DI INFERENZA / ANOMALY DETECTION (Su flusso ininterrotto) ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    num_test_batches = (len(src) - train_size) // batch_size
    with torch.no_grad():
        for i in range(num_test_batches):
            start_idx = train_size + i * batch_size
            end_idx = start_idx + batch_size
            
            b_src = src[start_idx:end_idx].to(device)
            b_dst = dst[start_idx:end_idx].to(device)
            b_t = t[start_idx:end_idx].to(device)
            b_msg = msg[start_idx:end_idx].to(device)
            b_y = y[start_idx:end_idx] # La label vera (che la rete non vede)
            
            n_id, inverse_indices = torch.unique(torch.cat([b_src, b_dst]), return_inverse=True)
            local_src = inverse_indices[:len(b_src)]
            local_dst = inverse_indices[len(b_src):]
            local_edge_index = torch.stack([local_src, local_dst], dim=0)
            
            # Valutiamo il traffico corrente: quanto è probabile che questo sia un comportamento sano?
            pos_out = model(n_id, local_edge_index, b_t, b_msg).squeeze(-1)
            prob_benign = torch.sigmoid(pos_out)
            
            # L'Anomaly Score è l'inverso: bassa probabilità di traffico sano = alto errore di ricostruzione = ANOMALIA
            anomaly_score = 1.0 - prob_benign
            
            all_preds.append(anomaly_score.cpu())
            all_labels.append(b_y.cpu())
            
            # Aggiorniamo la memoria con quello che è appena accaduto 
            model.memory.update_state(b_src, b_dst, b_t, b_msg)
            model.memory.detach()
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    
    print(f"Test Stream | Anomaly Detection AUC: {auc:.4f} | AP: {ap:.4f}")

if __name__ == "__main__":
    train_tgn()

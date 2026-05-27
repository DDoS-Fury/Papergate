import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

from graphagate.data.stream_synthetic import generate_streaming_data
from graphagate.model.tgn import ZTATemporalGraphNetwork

def train_tgn():
    # 1. Generate streaming data
    print("Generating streaming data...")
    num_users = 50
    num_ips = 100
    num_resources = 20
    total_nodes = num_users + num_ips + num_resources
    
    src, dst, t, msg, y, node_features = generate_streaming_data(
        num_users=num_users, num_ips=num_ips, num_resources=num_resources, num_events=50000
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Message dim: [JA3, Snort, Sonda1, Sonda2, Sonda3, Action] = 6
    msg_dim = 6
    
    # 2. Model initialization
    model = ZTATemporalGraphNetwork(
        num_nodes=total_nodes,
        node_feat_dim=16,
        msg_dim=msg_dim,
        memory_dim=64,
        time_dim=32
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    
    # Simple batching simulation (since TGN needs sequential processing)
    batch_size = 200
    num_batches = len(src) // batch_size
    
    for epoch in range(1, 6):
        model.memory.reset_state() # Reset memory at the start of epoch
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for i in range(num_batches):
            optimizer.zero_grad()
            
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            b_src = src[start_idx:end_idx].to(device)
            b_dst = dst[start_idx:end_idx].to(device)
            b_t = t[start_idx:end_idx].to(device)
            b_msg = msg[start_idx:end_idx].to(device)
            b_y = y[start_idx:end_idx].to(device).float()
            
            # For inference in TGN, we typically predict links BEFORE updating memory
            # to prevent information leakage (predicting the edge that we just added to memory)
            n_id, inverse_indices = torch.unique(torch.cat([b_src, b_dst]), return_inverse=True)
            local_src = inverse_indices[:len(b_src)]
            local_dst = inverse_indices[len(b_src):]
            local_edge_index = torch.stack([local_src, local_dst], dim=0)
            
            # Predict trust score / anomaly probability
            out = model(n_id, local_edge_index, b_t, b_msg)
            out = out.squeeze()
            
            # Calculate Loss (out is logits, label is 1 for anomaly, 0 for normal)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Now UPDATE memory with the new interactions
            # TGNMemory requires update_state. We simulate the message passing to memory
            model.memory.update_state(b_src, b_dst, b_t, b_msg)
            model.memory.detach() # Detach to prevent backprop through time indefinitely
            
            all_preds.append(torch.sigmoid(out).detach().cpu())
            all_labels.append(b_y.cpu())
            
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        
        print(f"Epoch {epoch:02d} | Loss: {total_loss/num_batches:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    train_tgn()

import torch
import numpy as np

def generate_streaming_data(num_users=50, num_ips=100, num_resources=20, num_events=5000):
    """
    Generates mock streaming data for TGN.
    We map all entities to a single node index space:
    [0, num_users-1] -> Users
    [num_users, num_users + num_ips - 1] -> IPs
    [num_users + num_ips, num_users + num_ips + num_resources - 1] -> Resources
    """
    total_nodes = num_users + num_ips + num_resources
    
    # Node features (static) - can be role, username hash, TPM status
    node_features = torch.randn(total_nodes, 16)
    
    # Events
    src_nodes = []
    dst_nodes = []
    timestamps = []
    edge_features = []
    labels = []
    
    current_time = 0
    for i in range(num_events):
        current_time += np.random.exponential(scale=1.0)
        
        # Random interaction: IP -> Resource (simplification, real case might be User -> Resource via IP)
        # Here we simulate IP -> Resource
        src_val = np.random.randint(num_users, num_users + num_ips)
        
        # Pattern strutturale: ogni IP ha una "risorsa preferita" (es. il suo server di dipartimento)
        pref_resource = (src_val % num_resources) + num_users + num_ips
        
        is_anomalous = np.random.rand() < 0.05
        
        if is_anomalous:
            # L'anomalia rompe il pattern (va su una risorsa a caso non sua)
            dst_val = np.random.randint(num_users + num_ips, total_nodes)
            if dst_val == pref_resource:
                dst_val = num_users + num_ips + ((dst_val - num_users - num_ips + 1) % num_resources)
            
            ja3 = 0.0
            snort = 1.0
            s1, s2, s3 = np.random.rand(3) > 0.5
            action = np.random.randint(0, 5)
            label = 1
        else:
            # Comportamento sano: 90% delle volte va sulla sua risorsa, 10% su risorse comuni
            if np.random.rand() < 0.90:
                dst_val = pref_resource
            else:
                dst_val = np.random.randint(num_users + num_ips, total_nodes)
                
            ja3 = 1.0
            snort = 0.0
            s1, s2, s3 = 0.0, 0.0, 0.0
            action = np.random.randint(0, 5)
            label = 0
            
        edge_feat = [ja3, snort, float(s1), float(s2), float(s3), float(action)]
        
        src_nodes.append(src_val)
        dst_nodes.append(dst_val)
        timestamps.append(current_time)
        edge_features.append(edge_feat)
        labels.append(label)
        
    src_tensor = torch.tensor(src_nodes, dtype=torch.long)
    dst_tensor = torch.tensor(dst_nodes, dtype=torch.long)
    t_tensor = torch.tensor(timestamps, dtype=torch.long) # TGN uses integer timestamps typically
    msg_tensor = torch.tensor(edge_features, dtype=torch.float)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    return src_tensor, dst_tensor, t_tensor, msg_tensor, y_tensor, node_features

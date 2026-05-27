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
        src = np.random.randint(num_users, num_users + num_ips)
        dst = np.random.randint(num_users + num_ips, total_nodes)
        
        # Edge features: [JA3_trust(0/1), snort_alert(0/1), sonda1, sonda2, sonda3, action_type]
        is_anomalous = np.random.rand() < 0.05
        
        if is_anomalous:
            ja3 = 0.0
            snort = 1.0
            s1, s2, s3 = np.random.rand(3) > 0.5
            action = np.random.randint(0, 5)
            label = 1
        else:
            ja3 = 1.0
            snort = 0.0
            s1, s2, s3 = 0.0, 0.0, 0.0
            action = np.random.randint(0, 5)
            label = 0
            
        edge_feat = [ja3, snort, float(s1), float(s2), float(s3), float(action)]
        
        src_nodes.append(src)
        dst_nodes.append(dst)
        timestamps.append(current_time)
        edge_features.append(edge_feat)
        labels.append(label)
        
    src_tensor = torch.tensor(src_nodes, dtype=torch.long)
    dst_tensor = torch.tensor(dst_nodes, dtype=torch.long)
    t_tensor = torch.tensor(timestamps, dtype=torch.long) # TGN uses integer timestamps typically
    msg_tensor = torch.tensor(edge_features, dtype=torch.float)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    return src_tensor, dst_tensor, t_tensor, msg_tensor, y_tensor, node_features

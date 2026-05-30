import torch
import numpy as np
import random

def generate_streaming_data(num_users=50, num_ips=100, num_resources=20, num_events=5000,
                            seed=None):
    """
    Generates mock streaming data for TGN based on ZTA policies.
    We map all entities to a single node index space:
    [0, num_users-1] -> Users
    [num_users, num_users + num_ips - 1] -> IPs (Devices)
    [num_users + num_ips, num_users + num_ips + num_resources - 1] -> Resources

    ``seed`` (optional) seeds both ``numpy`` and the stdlib ``random`` module so the
    stream is fully reproducible — ``random.choice`` (route/method selection) is used
    alongside ``np.random``, so seeding only numpy would leave the data partly random.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    total_nodes = num_users + num_ips + num_resources
    
    # ZTA Policies definition
    ROLES = ["plant_manager", "operator", "maintenance_technician", "radiation_protection_officer", "security_officer", "inspector"]
    CLEARANCES = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
    
    # Assign attributes to users
    user_roles = [np.random.choice(ROLES) for _ in range(num_users)]
    user_clearances = [np.random.randint(0, 5) for _ in range(num_users)] # 0 to 4
    
    # Assign attributes to IPs (Tier: 0=no cert/tpm, 1=cert, 2=cert+tpm)
    ip_tiers = [np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3]) for _ in range(num_ips)]
    
    # Assign an IP to a default User (Device association)
    ip_to_user = [i % num_users for i in range(num_ips)]
    
    # Define route templates based on ZTA policies (method: allowed_roles, min_tier, min_clearance)
    # methods: 0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH
    route_templates = [
        # public (like /health, /login, /materials)
        {0: (set(ROLES), 0, 0), 1: (set(ROLES), 0, 0)},
        # /api/v1/auth/register/begin
        {1: ({"plant_manager", "operator", "maintenance_technician", "radiation_protection_officer", "security_officer", "inspector"}, 1, 0)},
        # /api/v1/personnel
        {0: ({"security_officer", "plant_manager", "inspector"}, 1, 1),
         1: ({"plant_manager"}, 2, 3), 2: ({"plant_manager"}, 2, 3), 3: ({"plant_manager"}, 2, 3)},
        # /api/v1/zones
        {0: ({"operator", "plant_manager", "inspector", "maintenance_technician", "radiation_protection_officer", "security_officer"}, 0, 0),
         1: ({"plant_manager"}, 2, 3), 2: ({"plant_manager"}, 2, 3), 3: ({"plant_manager"}, 2, 3)},
        # /api/v1/badges
        {0: ({"security_officer", "plant_manager", "inspector"}, 1, 1),
         1: ({"plant_manager"}, 2, 2), 2: ({"plant_manager"}, 2, 2), 3: ({"plant_manager"}, 2, 2)},
        # /api/v1/reactor-parameters
        {0: ({"operator", "plant_manager", "inspector"}, 1, 2),
         1: ({"plant_manager"}, 2, 3), 2: ({"plant_manager"}, 2, 3), 3: ({"plant_manager"}, 2, 3)},
        # /api/v1/maintenance-orders
        {0: ({"maintenance_technician", "plant_manager"}, 1, 1),
         1: ({"maintenance_technician", "plant_manager"}, 1, 1), 2: ({"maintenance_technician", "plant_manager"}, 1, 1),
         3: ({"maintenance_technician", "plant_manager"}, 2, 2)},
        # /api/v1/documents
        {0: ({"operator", "plant_manager", "inspector", "maintenance_technician", "radiation_protection_officer", "security_officer"}, 0, 0),
         1: ({"plant_manager"}, 2, 2), 2: ({"plant_manager"}, 2, 2), 3: ({"plant_manager"}, 2, 2)},
        # /api/v1/nuclear-materials
        {0: ({"plant_manager", "inspector", "radiation_protection_officer"}, 2, 3),
         1: ({"plant_manager"}, 2, 4), 2: ({"plant_manager"}, 2, 4), 3: ({"plant_manager"}, 2, 4)}
    ]
    
    resource_rules = []
    for i in range(num_resources):
        resource_rules.append(route_templates[i % len(route_templates)])
        
    # Node features (static): 16-dim
    # Encode roles, clearance, tier into the features.
    node_features = torch.zeros(total_nodes, 16)
    for i in range(num_users):
        node_features[i, 0] = ROLES.index(user_roles[i]) / float(len(ROLES))
        node_features[i, 1] = user_clearances[i] / 4.0
    for i in range(num_ips):
        node_features[num_users + i, 2] = ip_tiers[i] / 2.0
        # encode the user this IP belongs to
        u_idx = ip_to_user[i]
        node_features[num_users + i, 0] = ROLES.index(user_roles[u_idx]) / float(len(ROLES))
        node_features[num_users + i, 1] = user_clearances[u_idx] / 4.0
    
    src_nodes = []
    dst_nodes = []
    timestamps = []
    edge_features = []
    labels = []
    # Anomaly type for per-class evaluation: 0=benign, 1=policy violation, 2=contextual.
    # The binary ``labels`` above are unchanged (training/serving consume only those).
    types = []
    
    current_time = 0
    for i in range(num_events):
        current_time += int(np.random.exponential(scale=300.0))
        hour_of_day = (current_time // 3600) % 24
        
        # We simulate IP -> Resource interactions
        src_ip_idx = np.random.randint(0, num_ips)
        src_val = num_users + src_ip_idx
        
        user_idx = ip_to_user[src_ip_idx]
        u_role = user_roles[user_idx]
        u_clearance = user_clearances[user_idx]
        u_tier = ip_tiers[src_ip_idx]
        
        is_anomalous = np.random.rand() < 0.05
        
        if not is_anomalous:
            valid_actions = []
            for res_idx, rules in enumerate(resource_rules):
                for method, (req_roles, min_tier, min_clearance) in rules.items():
                    if u_role in req_roles and u_tier >= min_tier and u_clearance >= min_clearance:
                        valid_actions.append((res_idx, method))
            
            if not valid_actions:
                # Fallback to the first route template (public path)
                res_idx, method = 0, 0
                for idx, r in enumerate(resource_rules):
                    if 0 in r and r[0][1] == 0 and r[0][2] == 0:
                        res_idx = idx
                        break
            else:
                res_idx, method = random.choice(valid_actions)
                
            dst_val = num_users + num_ips + res_idx
            
            ja3 = 1.0
            snort = 0.0
            s1, s2, s3 = 0.0, 0.0, 0.0
            label = 0
            etype = 0

        else:
            anomaly_type = np.random.choice(["policy", "context"])

            if anomaly_type == "policy":
                invalid_actions = []
                for res_idx, rules in enumerate(resource_rules):
                    for method, (req_roles, min_tier, min_clearance) in rules.items():
                        if u_role not in req_roles or u_tier < min_tier or u_clearance < min_clearance:
                            invalid_actions.append((res_idx, method))
                
                if invalid_actions:
                    res_idx, method = random.choice(invalid_actions)
                else:
                    res_idx = np.random.randint(0, num_resources)
                    method = np.random.randint(0, 4)
                    
                dst_val = num_users + num_ips + res_idx
                ja3 = 1.0
                snort = 0.0
                s1, s2, s3 = 0.0, 0.0, 0.0
                etype = 1

            else:
                res_idx = np.random.randint(0, num_resources)
                method = np.random.randint(0, 4)
                dst_val = num_users + num_ips + res_idx

                ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
                snort = 1.0 if np.random.rand() > 0.5 else 0.0
                s1, s2, s3 = np.random.rand(3) > 0.5
                etype = 2

            label = 1
            
        action = float(method)
        edge_feat = [ja3, snort, float(s1), float(s2), float(s3), action]
        
        src_nodes.append(src_val)
        dst_nodes.append(dst_val)
        timestamps.append(current_time)
        edge_features.append(edge_feat)
        labels.append(label)
        types.append(etype)

    src_tensor = torch.tensor(src_nodes, dtype=torch.long)
    dst_tensor = torch.tensor(dst_nodes, dtype=torch.long)
    t_tensor = torch.tensor(timestamps, dtype=torch.long)
    msg_tensor = torch.tensor(edge_features, dtype=torch.float)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    types_tensor = torch.tensor(types, dtype=torch.long)

    return src_tensor, dst_tensor, t_tensor, msg_tensor, y_tensor, types_tensor, node_features

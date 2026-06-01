import asyncio
import random
import numpy as np

async def event_generator(num_users=50, num_ips=100, num_resources=20, seed=None):
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
    
    RESOURCE_URIS = [
        "/public",
        "/api/v1/auth/register/begin",
        "/api/v1/personnel",
        "/api/v1/zones",
        "/api/v1/badges",
        "/api/v1/reactor-parameters",
        "/api/v1/maintenance-orders",
        "/api/v1/documents",
        "/api/v1/nuclear-materials"
    ]
    
    resource_rules = []
    resource_uris = []
    for i in range(num_resources):
        base_uri = RESOURCE_URIS[i % len(RESOURCE_URIS)]
        suffix = "" if num_resources <= len(RESOURCE_URIS) else f"/{i // len(RESOURCE_URIS)}"
        resource_rules.append(route_templates[i % len(route_templates)])
        resource_uris.append(base_uri + suffix)
        
    # Node features (static): 16-dim
    node_features = np.zeros((total_nodes, 16))
    for i in range(num_users):
        node_features[i, 0] = ROLES.index(user_roles[i]) / float(len(ROLES))
        node_features[i, 1] = user_clearances[i] / 4.0
    for i in range(num_ips):
        node_features[num_users + i, 2] = ip_tiers[i] / 2.0
        u_idx = ip_to_user[i]
        node_features[num_users + i, 0] = ROLES.index(user_roles[u_idx]) / float(len(ROLES))
        node_features[num_users + i, 1] = user_clearances[u_idx] / 4.0
        
    ip_valid_actions = []
    ip_habitual = []
    for ip_idx in range(num_ips):
        u_idx = ip_to_user[ip_idx]
        role, clr, tier = user_roles[u_idx], user_clearances[u_idx], ip_tiers[ip_idx]
        valid = []
        for res_idx, rules in enumerate(resource_rules):
            for method, (req_roles, min_tier, min_clearance) in rules.items():
                if role in req_roles and tier >= min_tier and clr >= min_clearance:
                    valid.append((res_idx, method))
        ip_valid_actions.append(valid)
        if valid:
            k = max(1, len(valid) // 2)
            hab_idx = np.random.choice(len(valid), size=k, replace=False)
            ip_habitual.append({valid[j] for j in hab_idx})
        else:
            ip_habitual.append(set())

    # Per allineare PERFETTAMENTE il tempo con il training, riproduciamo
    # la sequenza casuale originale del seed 42 per 50000 eventi.
    np.random.seed(42)
    current_time = 0
    for _ in range(50000):
        current_time += int(np.random.exponential(scale=300.0))
        
    print(f"[Generator] Starting seamlessly at t={current_time}")
    while True:
        current_time += int(np.random.exponential(scale=300.0))
        
        src_ip_idx = np.random.randint(0, num_ips)
        src_val = num_users + src_ip_idx
        
        user_idx = ip_to_user[src_ip_idx]
        u_role = user_roles[user_idx]
        u_clearance = user_clearances[user_idx]
        u_tier = ip_tiers[src_ip_idx]
        
        is_anomalous = np.random.rand() < 0.05
        
        if not is_anomalous:
            habit = list(ip_habitual[src_ip_idx])
            if habit:
                res_idx, method = random.choice(habit)
            elif ip_valid_actions[src_ip_idx]:
                res_idx, method = random.choice(ip_valid_actions[src_ip_idx])
            else:
                res_idx, method = 0, 0
                
            dst_val = num_users + num_ips + res_idx
            ja3 = 1.0
            snort = 0.0
            s1, s2, s3 = 0.0, 0.0, 0.0
            label = 0
            etype = 0
        else:
            anomaly_type = np.random.choice(["policy", "context", "lateral"])
            
            if anomaly_type == "lateral":
                non_habit = [a for a in ip_valid_actions[src_ip_idx] if a not in ip_habitual[src_ip_idx]]
                if non_habit:
                    res_idx, method = random.choice(non_habit)
                    dst_val = num_users + num_ips + res_idx
                    ja3 = 1.0
                    snort = 0.0
                    s1, s2, s3 = 0.0, 0.0, 0.0
                    etype = 3
                else:
                    anomaly_type = "policy"
                    
            if anomaly_type == "policy":
                invalid_actions = []
                for r_idx, rules in enumerate(resource_rules):
                    for m, (req_roles, min_tier, min_clearance) in rules.items():
                        if u_role not in req_roles or u_tier < min_tier or u_clearance < min_clearance:
                            invalid_actions.append((r_idx, m))
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
                
            elif anomaly_type == "context":
                res_idx = np.random.randint(0, num_resources)
                method = np.random.randint(0, 4)
                dst_val = num_users + num_ips + res_idx
                ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
                snort = 1.0 if np.random.rand() > 0.5 else 0.0
                s1 = 1.0 if np.random.rand() > 0.5 else 0.0
                s2 = 1.0 if np.random.rand() > 0.5 else 0.0
                s3 = 1.0 if np.random.rand() > 0.5 else 0.0
                etype = 2
                
            label = 1
            
        action = float(method)
        edge_feat = [float(ja3), float(snort), float(s1), float(s2), float(s3), float(action)]
        
        src_feat = node_features[src_val].tolist()
        dst_feat = [0.0] * 16
        
        yield {
            "key_src": f"user_{src_val}",
            "key_dst": resource_uris[res_idx],
            "timestamp": int(current_time),
            "features": edge_feat,
            "src_feat": src_feat,
            "dst_feat": dst_feat,
            "label": label,
            "type": etype
        }
        await asyncio.sleep(0)

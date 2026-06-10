import asyncio
import random
import numpy as np

async def event_generator(num_users=50, num_ips=100, num_resources=19, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    total_nodes = num_users + num_ips + num_resources

    # ZTA Policies definition — mirrors src/data/stream_synthetic.py (policy.rego roles)
    ROLES = ["guest", "operator", "manager", "admin"]
    CLEARANCES = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]

    # Assign attributes to users
    user_roles = [np.random.choice(ROLES) for _ in range(num_users)]
    user_clearances = [np.random.randint(0, 5) for _ in range(num_users)] # 0 to 4

    # Assign attributes to IPs (Tier: 0=no cert/tpm, 1=cert, 2=cert+tpm)
    ip_tiers = [np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3]) for _ in range(num_ips)]

    # Assign an IP to a default User (Device association)
    ip_to_user = [i % num_users for i in range(num_ips)]

    # Route templates keyed by URI: {method: (allowed_roles, min_tier, min_clearance)}.
    # methods: 0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH
    ALL_ROLES = set(ROLES)
    AUTH = {"operator", "manager", "admin"}
    MGR = {"manager", "admin"}
    ADM = {"admin"}
    PUBLIC_GET = {0: (ALL_ROLES, 0, 0)}
    PUBLIC_POST = {1: (ALL_ROLES, 0, 0)}

    route_templates = {
        "/": PUBLIC_GET,
        "/materials": PUBLIC_GET,
        "/reserved": PUBLIC_GET,
        "/login": PUBLIC_GET,
        "/register": PUBLIC_GET,
        "/static": PUBLIC_GET,
        "/favicon.ico": PUBLIC_GET,
        "/api/v1/auth/register": PUBLIC_POST,
        "/api/v1/auth/login": PUBLIC_POST,
        "/api/v1/auth/verify-otp": PUBLIC_POST,
        "/api/v1/auth/register/begin": {1: (AUTH, 1, 0)},
        "/api/v1/auth/register/finish": {1: (AUTH, 1, 0)},
        "/api/v1/auth/login/begin": PUBLIC_POST,
        "/api/v1/auth/login/finish": PUBLIC_POST,
        "/api/v1/personnel": {0: (AUTH, 1, 1), 1: (AUTH, 1, 1)},
        "/api/v1/documents": {0: (MGR, 1, 2), 1: (MGR, 1, 2), 3: (MGR, 1, 2)},
        "/api/v1/nuclear-materials": {0: (MGR, 1, 2), 1: (MGR, 1, 2), 3: (MGR, 1, 2)},
        "/api/v1/reactor-parameters": {0: (ADM, 1, 2), 1: (ADM, 1, 2), 3: (ADM, 1, 2)},
        "/api/v1/trusted-guard/sanitized-delete-personnel": {1: (ADM, 1, 2)},
    }
    RESOURCE_URIS = list(route_templates)

    assert num_resources == len(RESOURCE_URIS), (
        f"num_resources ({num_resources}) must equal len(RESOURCE_URIS) ({len(RESOURCE_URIS)})"
    )
    resource_uris = list(RESOURCE_URIS)
    resource_rules = [route_templates[uri] for uri in RESOURCE_URIS]

    # Node features (static): 16-dim
    node_features = np.zeros((total_nodes, 16))
    for i in range(num_users):
        pass
    for i in range(num_ips):
        node_features[num_users + i, 2] = ip_tiers[i] / 2.0

    for i in range(num_resources):
        node_features[num_users + num_ips + i, 3] = i / float(num_resources - 1)
        
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
    compromised_state = {}
    while True:
        current_time += int(np.random.exponential(scale=300.0))
        
        src_ip_idx = np.random.randint(0, num_ips)
        src_val = num_users + src_ip_idx
        
        user_idx = ip_to_user[src_ip_idx]
        u_role = user_roles[user_idx]
        u_clearance = user_clearances[user_idx]
        u_tier = ip_tiers[src_ip_idx]
        
        # APT Kill Chain Simulation
        if np.random.rand() < 0.005 and src_ip_idx not in compromised_state:
            compromised_state[src_ip_idx] = 1  # Phase 1: Recon
            
        is_anomalous = False
        if src_ip_idx in compromised_state:
            # Compromised IPs blend in 70% of the time, attack 30% of the time
            if np.random.rand() < 0.3:
                is_anomalous = True
        
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
            s1, s2, s3 = 0.0, 0.0, 0.0
            label = 0
            etype = 0
        else:
            state = compromised_state[src_ip_idx]
            if state == 1:
                anomaly_type = "context"  # Recon phase (often triggers Snort)
                compromised_state[src_ip_idx] = 2  # Advance to Lateral
            elif state == 2:
                anomaly_type = "lateral"  # Lateral Movement phase
                compromised_state[src_ip_idx] = 3  # Advance to Exfil
            else:
                anomaly_type = np.random.choice(["policy", "context", "lateral"])
            
            if anomaly_type == "lateral":
                non_habit = [a for a in ip_valid_actions[src_ip_idx] if a not in ip_habitual[src_ip_idx]]
                if non_habit:
                    res_idx, method = random.choice(non_habit)
                    dst_val = num_users + num_ips + res_idx
                    ja3 = 1.0
                    # Movimento laterale: è estremamente furtivo (stealth). Usa credenziali e protocolli legittimi.
                    # Rade volte fa scattare l'IDS, forzando la rete neurale a studiare il grafo.
                    s1 = 0.0
                    s2 = 1.0 if np.random.rand() > 0.98 else 0.0  # 2%
                    s3 = 1.0 if np.random.rand() > 0.90 else 0.0  # 10%
                    etype = 3
                    
                    if random.random() < 0.5:
                        stolen_role = random.choice([r for r in ROLES if r != u_role])
                        stolen_clearance = np.random.randint(0, 5)
                        u_role = stolen_role
                        u_clearance = stolen_clearance
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
                s1, s2, s3 = 0.0, 0.0, 0.0
                etype = 1
                
            elif anomaly_type == "context":
                res_idx = np.random.randint(0, num_resources)
                method = np.random.randint(0, 4)
                dst_val = num_users + num_ips + res_idx
                ja3 = 0.0 if np.random.rand() > 0.5 else 1.0
                # Recon/Context: attacco esterno, alta probabilità su Edge (80%), media su Mid (50%), bassa su Internal (20%)
                s1 = 1.0 if np.random.rand() > 0.2 else 0.0
                s2 = 1.0 if np.random.rand() > 0.5 else 0.0
                s3 = 1.0 if np.random.rand() > 0.8 else 0.0
                etype = 2
                
            label = 1
            
        action = float(method)
        u_role_val = ROLES.index(u_role) / float(len(ROLES) - 1)
        u_clearance_val = u_clearance / 4.0
        edge_feat = [float(ja3), float(s1), float(s2), float(s3), float(action), u_role_val, u_clearance_val]
        
        src_feat = node_features[src_val].tolist()
        dst_feat = node_features[dst_val].tolist()
        
        yield {
            "key_src": int(src_val),
            "key_dst": resource_uris[res_idx],
            "timestamp": int(current_time),
            "features": edge_feat,
            "src_feat": src_feat,
            "dst_feat": dst_feat,
            "label": label,
            "type": etype
        }
        await asyncio.sleep(0)

import torch
import numpy as np
import random

def generate_streaming_data(num_users=50, num_ips=100, num_resources=20, num_events=5000,
                            benign_explore_prob=0.15, seed=None):
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
        {0: (set(ROLES), 0, 0), 1: (set(ROLES), 0, 0), 2: (set(ROLES), 0, 0), 3: (set(ROLES), 0, 0)},
        # /api/v1/auth/register/begin
        {1: ({"plant_manager", "operator", "maintenance_technician", "radiation_protection_officer", "security_officer", "inspector"}, 1, 0)},
        # /api/v1/auth/register/finish
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
         1: ({"plant_manager"}, 2, 4), 2: ({"plant_manager"}, 2, 4), 3: ({"plant_manager"}, 2, 4)},
        # /api/v1/trusted-guard/sanitized-delete-personnel
        {1: ({"plant_manager"}, 2, 3)}
    ]
    
    RESOURCE_URIS = [
        "/public",
        "/api/v1/auth/register/begin",
        "/api/v1/auth/register/finish",
        "/api/v1/personnel",
        "/api/v1/zones",
        "/api/v1/badges",
        "/api/v1/reactor-parameters",
        "/api/v1/maintenance-orders",
        "/api/v1/documents",
        "/api/v1/nuclear-materials",
        "/api/v1/trusted-guard/sanitized-delete-personnel"
    ]
    
    resource_rules = []
    resource_uris = []
    for i in range(num_resources):
        base_uri = RESOURCE_URIS[i % len(RESOURCE_URIS)]
        suffix = "" if num_resources <= len(RESOURCE_URIS) else f"/{i // len(RESOURCE_URIS)}"
        resource_rules.append(route_templates[i % len(route_templates)])
        resource_uris.append(base_uri + suffix)
        
    # Node features (static): 16-dim
    # Encode roles, clearance, tier into the features.
    node_features = torch.zeros(total_nodes, 16)
    node_features[:, 14] = 1.0  # Trust Score
    for i in range(num_users):
        node_features[i, 0] = ROLES.index(user_roles[i]) / float(len(ROLES))
        node_features[i, 1] = user_clearances[i] / 4.0
    for i in range(num_ips):
        node_features[num_users + i, 2] = ip_tiers[i] / 2.0
        # encode the user this IP belongs to
        u_idx = ip_to_user[i]
        node_features[num_users + i, 0] = ROLES.index(user_roles[u_idx]) / float(len(ROLES))
        node_features[num_users + i, 1] = user_clearances[u_idx] / 4.0
        
    for i in range(num_resources):
        uri_idx = i % len(RESOURCE_URIS)
        node_features[num_users + num_ips + i, 3 + uri_idx] = 1.0
    
    # Per-IP behaviour model. For each IP we precompute the (resource, method) actions
    # its associated user is *authorised* to perform, then carve out a "habitual" subset
    # — the routes that device actually uses day-to-day. Benign traffic is drawn from the
    # habitual subset, EXCEPT a ``benign_explore_prob`` fraction that legitimately
    # exercises an authorised-but-non-habitual route (benign exploration). A
    # *lateral-movement* anomaly is *also* an authorised-but-non-habitual action
    # (policy-clean, signal-clean) — so novelty alone no longer separates the two: the
    # model must use the temporal/relational *pattern* (lateral is clustered on a
    # compromised IP, inside a recon→lateral→exfil kill chain) that the neighbour loader
    # and memory expose. Setting benign_explore_prob=0 recovers the old degenerate task.
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

    src_nodes = []
    dst_nodes = []
    timestamps = []
    edge_features = []
    labels = []
    # Anomaly type for per-class evaluation: 0=benign, 1=policy violation, 2=contextual,
    # 3=lateral movement. The binary ``labels`` above are unchanged (training/serving
    # consume only those). NOTE: policy (etype=1) is the OPA orchestrator's job — it is
    # blocked deterministically upstream and is NOT a value-add of this model; it is kept
    # only as a sanity column. The model's genuine target is lateral (etype=3).
    types = []
    
    current_time = 0
    compromised_state = {}  # ip_idx -> state (1: Recon, 2: Lateral, 3: Exfil)
    
    for i in range(num_events):
        current_time += int(np.random.exponential(scale=300.0))
        hour_of_day = (current_time // 3600) % 24
        
        # We simulate IP -> Resource interactions
        max_ip = max(1, int((i / num_events) * num_ips) + 1)
        if max_ip > num_ips:
            max_ip = num_ips
        src_ip_idx = np.random.randint(0, max_ip)
        src_val = num_users + src_ip_idx
        
        user_idx = ip_to_user[src_ip_idx]
        u_role = user_roles[user_idx]
        u_clearance = user_clearances[user_idx]
        u_tier = ip_tiers[src_ip_idx]
        
        # APT Kill Chain Simulation
        if np.random.rand() < 0.005 and src_ip_idx not in compromised_state:
            compromised_state[src_ip_idx] = 1  # IP becomes compromised -> Phase 1: Recon
            
        is_anomalous = False
        if src_ip_idx in compromised_state:
            # Compromised IPs blend in 70% of the time, attack 30% of the time
            if np.random.rand() < 0.3:
                is_anomalous = True
        
        if not is_anomalous:
            habit = list(ip_habitual[src_ip_idx])
            non_habit = [a for a in ip_valid_actions[src_ip_idx]
                         if a not in ip_habitual[src_ip_idx]]
            # Benign exploration: occasionally a legitimate device performs an
            # authorised-but-non-habitual action. This is policy-clean and signal-clean
            # and label=0 — so "non-habitual" is no longer a sufficient anomaly cue, and
            # the model cannot trivially equate novelty with lateral movement.
            if non_habit and random.random() < benign_explore_prob:
                res_idx, method = random.choice(non_habit)
            elif habit:
                res_idx, method = random.choice(habit)
            elif ip_valid_actions[src_ip_idx]:
                res_idx, method = random.choice(ip_valid_actions[src_ip_idx])
            else:
                res_idx, method = 0, 0  # public-path fallback (no authorised action)

            dst_val = num_users + num_ips + res_idx

            ja3 = 1.0
            snort = 0.0
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
                # Authorised but non-habitual access: policy-clean and signal-clean,
                # detectable only from the IP's interaction history.
                non_habit = [a for a in ip_valid_actions[src_ip_idx]
                             if a not in ip_habitual[src_ip_idx]]
                if non_habit:
                    res_idx, method = random.choice(non_habit)
                    dst_val = num_users + num_ips + res_idx
                    ja3 = 1.0
                    snort = 0.0
                    s1, s2, s3 = 0.0, 0.0, 0.0
                    etype = 3
                else:
                    # No non-habitual authorised action exists -> fall back to a policy
                    # violation so the event is still a genuine anomaly.
                    anomaly_type = "policy"

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

            elif anomaly_type == "context":
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

    auth_mask = torch.zeros((num_ips, num_resources), dtype=torch.bool)
    for i in range(num_ips):
        for res_idx, _ in ip_valid_actions[i]:
            auth_mask[i, res_idx] = True

    return src_tensor, dst_tensor, t_tensor, msg_tensor, y_tensor, types_tensor, node_features, resource_uris, auth_mask


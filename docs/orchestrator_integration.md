# ZTA Orchestrator and TGN Model Integration

This document clarifies the integration architecture between the Security Orchestrator (which talks to the Policy Decision Point, e.g. OPA) and the TGN (Temporal Graph Network) AI microservice.

## No Vector Database Needed

A common question when integrating AI models for structural Anomaly Detection (such as graphs) is whether an external vector database (e.g. Milvus, Pinecone) is needed to store past embeddings or request tuples.

**The answer for the TGN is no.**

The TGN model was designed specifically to be **stateful** and to autonomously manage its own temporal memory in RAM via PyTorch tensors. The structural history (the last `K` temporal neighbours of each entity) is also kept in RAM by a **bounded neighbour loader** (`MessageNeighborLoader`, a fixed-size ring buffer `O(num_nodes·K)`): this is what enables *lateral movement* detection **without** any external graph database.

### Execution Flow (Serving)

1. **Request forwarding (tuple, 5-node v4 schema)**
   The ZTA orchestrator must not pre-process vectors nor query historical databases. It simply forwards the single raw transaction (or event) to the model serving API (`src/serve_tgn.py -> score_event`). Every request is modelled as a **5-edge** causal chain: `source → config`, `config → device`, `config → user`, `device → user` and the access `user → resource`. **The keys are namespaced by TYPE** so an IP can never alias a device slot in the shared `NodeRegistry` (which is a single keyspace for all node types). The requested tuple includes:
   - `key_user`: identity (e.g. user id from the JWT; `anonymous` for guests).
   - `key_device`: hardware context — `tpm:<id>` if the TPM is attested, otherwise `ck:<cookie>` (signed persistent cookie/UUID; a new cookie = a never-seen machine). With no hardware id at all, fallback `ipdev:<ip>` (the IP as a weak device) **without** `key_source`. Never use a bare IP as the device.
   - `key_source` (optional): network context — `src:<ip>` (the client IP, namespaced). If absent, the source→config edge is simply skipped. The model derives from this IP the **internal/external** bit (RFC1918) written to `node_feat[*,5]` of the source node — it is a *feature*, not an authorization gate (the network grants no privileges: ZTA).
   - `key_config` (optional): client configuration — the TLS/JA3 fingerprint, `conf:<ja3>`. If omitted the server substitutes `conf:guest`, so **the config node is always present**.
   - `key_dst`: resource URI.
   - Timestamp (e.g. Unix epoch).
   - `features`: edge message of **`msg_dim` floats** (currently **10**, see
     `TGNConfig.msg_dim`; `/infer` rejects a different length with 422):
     `[ja3, s1, s2, s3, method, role, clearance, bytes_in, bytes_out, log1p(Δt user)/10]`.
     Only fields available **at decision time**: no response field (the HTTP status
     is deliberately not part of the message — see the docstring of
     `stream_synthetic`).
   - **Entity static attributes** (`user_feat` / `device_feat` / `dst_feat`, len ==
     `node_feat_dim` = 16): role, clearance, device tier. The orchestrator/OPA already
     knows them for every request, so they are passed per-event (no extra datastore).
     They are the signal that lets the model detect **policy violations** — anomalies
     whose edge features are identical to benign traffic. Because training happens on
     synthetic data, production users will all be "new": it is **mandatory** to pass these
     features so the model knows the privileges of the real user just met. *There is no
     `src_feat` field*: earlier revisions of this document mentioned it, but the API never
     had it.

2. **`NodeRegistry` handling**
   On arrival of a tuple, the TGN uses its `NodeRegistry` to map alphanumeric keys (e.g. a
   never-seen IP address) to integer indices in real time. The system supports the entry of
   nodes unseen during training (dynamic and unbounded node space).

3. **TGN memory and neighbourhood integration**
   The model accesses the historical state of the involved nodes by reading its internal
   tensors: the recurrent memory (`model.memory`) **and** the recent temporal neighbourhood
   (`model.neighbor_loader`). It concatenates the memory with the hashed node identity
   (**Hashed Identity**), runs the multi-hop GNN over the real neighbours and combines the
   *feature head* (policy/contextual) with the *structural head* (lateral movement). The
   anomaly score (`1 − P(benign)`, from `0.0` to `1.0`) is computed and returned to the
   orchestrator, which will pass it to OPA.

4. **"Anti-Poisoning" update (OPA gatekeeper)**
   For OPA to be the true final decider, the Orchestrator drives the model primitives in two
   steps (instead of `score_event`'s internal gate):
   - It calls **`infer_score`** to obtain the anomaly score (read-only operation with respect
     to the baseline: it mutates neither memory nor neighbourhood). The only side effect is
     that the `/infer` endpoint admits never-seen keys in the `NodeRegistry` (slot allocation,
     bookkeeping to be able to score): it is not learning, and the baseline (memory +
     neighbourhood) stays intact even for events later DENYed. The slot admission is declared
     in the endpoint table below (`no (admission only)`).
   - It sends the request and the score to OPA.
   - If **OPA answers ALLOW** (the event is fully legitimate and not anomalous), the
     Orchestrator calls **`update_memory`**, which advances the TGN memory **and** inserts
     the edge into the neighbour loader.
   - If **OPA answers DENY**, the `update_memory` call is omitted. This absolutely prevents
     attackers from "poisoning" the model, guaranteeing that the TGN learns only from what
     OPA explicitly approved — both in memory and in the neighbour history.

   > Note: `infer_score` / `update_memory` work on slot indices already mapped by the
   > `NodeRegistry`; the per-event static attributes must be written into the slot before
   > scoring (that is what `score_event` does internally).

### Persistence

The only storage required for this AI layer is the filesystem. The model save command
(`save_model`) serializes the entire state to disk:
- The trained network weights (including the node identity and the two scoring heads).
- The in-memory tensors with the access histories (TGN Memory) + the raw-message store.
- The neighbour loader buffers (the last `K` temporal neighbours per node).
- The NodeRegistry dictionary.

This file (`public/tgn_checkpoint.pt`) together with the metadata (`public/tgn_stats.json`)
lets the AI microservice restart exactly from where it was interrupted without losing the
users' historical context.

## HTTP API (inference service)

The primitives described above are exposed as a **REST/JSON microservice** by
`src/serve_api.py` (FastAPI + uvicorn), started with `python -m graphagate.serve_api`
(Docker Compose profile `serve-tgn`; container port `8088`, exposed on the host as
`8888`). The Go orchestrator talks to it with
`net/http` + `encoding/json` — no `.proto`/gRPC to maintain.

### Starting the service

**Prerequisite**: the service loads the artifacts `public/tgn_checkpoint.pt` and
`public/tgn_stats.json`. They must be produced **first**, once, by the training
(`docker compose --profile training-tgn up`; the artifacts are **generated** by the
training, gitignored). Without them the service does not start.

Start as a service (long-running):

```bash
# Via Docker Compose (dedicated profile, exposes :8888 on the host and the healthcheck on /health)
docker compose --profile serve-tgn up

# Or standalone, reusing the same image
docker run --rm --gpus all -p 8888:8088 \
  -v "$PWD/public:/app/public" graphagate graphagate.serve_api
```

The service is ready when `GET /health` answers `200` with
`{"status":"ok","model_loaded":true,...}`. **While the checkpoint is loading it answers
`503`** (body `{"status":"loading",...}`): the readiness gate — and the Compose
healthcheck — treat the 503 as "not ready yet", so depending on
`condition: service_healthy` from the orchestrator side guarantees the startup order
without a race on the loading.

Configuration via environment variables (all optional):

| Variable | Default | Role |
|---|---|---|
| `GRAPHAGATE_CHECKPOINT` | `public/tgn_checkpoint.pt` | path of the checkpoint (weights + memory + neighbourhood) |
| `GRAPHAGATE_STATS` | `public/tgn_stats.json` | path of the calibrated threshold + `NodeRegistry` |
| `GRAPHAGATE_HOST` | `0.0.0.0` | bind address |
| `GRAPHAGATE_PORT` | `8088` | bind port |

### Endpoints

| Method · path | Role | Mutates state? |
|---|---|---|
| `GET /health` | Readiness + loaded parameters (device, threshold, dimensions, registry slots) | no |
| `POST /infer` | Computes the anomaly score **without** advancing memory/neighbourhood (only admits the entity in the registry) — *step 1* of the anti-poisoning flow | no (admission only) |
| `POST /update` | Commits an **already approved** event (post-OPA-ALLOW): advances memory + neighbour history | yes |
| `POST /score` | Score + internal gate + conditional update (OPA-less use / tests) | yes if benign |
| `POST /persist` | Rewrites the evolved state to `public/` (also automatic at shutdown) | writes to disk |

### Request schema (events)

`/infer`, `/update`, `/score` accept the same JSON body:

```json
{
  "key_user": "alice",             // user key (string or int); "anonymous" for guests
  "key_device": "tpm:a1b2c3",      // device key: "tpm:<id>" | "ck:<cookie>" | "ipdev:<ip>"
  "key_source": "src:10.0.0.7",    // opt.: the client "src:<ip>" (if absent, no source→config edge)
  "key_config": "conf:771,4865-...", // opt.: TLS/JA3 fingerprint; default "conf:guest"
  "key_dst": "/api/v1/documents",  // resource key (normalized URI)
  "timestamp": 1717000000,         // integer (e.g. Unix epoch)
  "features": [1.0, 0.0, 0.0, 0.0, 0.0, 0.67, 0.5, 0.12, 0.08, 0.31],
                                              // edge message: len == msg_dim (10)
                                              // [0] JA3: 1.0 (ok), 0.0 (anomaly)
                                              // [1-3] Snort probes s1, s2, s3 (0.0 or 1.0)
                                              // [4] HTTP method (0=GET, 1=POST, 2=PUT, 3=DELETE, 4=PATCH)
                                              // [5] Normalized role (idx/(len-1))
                                              // [6] Normalized clearance (idx/4)
                                              // [7] Normalized bytes_in
                                              // [8] Normalized bytes_out
                                              // [9] log1p(Δt since the user's last request)/10
  "user_feat": [/* ... */],        // opt., static attributes, len == node_feat_dim (16)
  "device_feat": [/* ... */],      // opt., same (tier in node_feat[2])
  "dst_feat": [/* ... */],         // opt.; for preregistered resources the RISK
                                   // (node_feat[*,4]) is already baked in the checkpoint, so
                                   // dst_feat is NOT required in production.
  "flagged": false                 // only /update: the is_anomaly returned by the
                                   // previous /infer. OPA can ALLOW an event the model
                                   // flagged: sending it back is what arms the kill-chain
                                   // precursor and lowers the trust.
}
```

> ⚠️ There is no `src_feat` field (earlier revisions of this document showed it): the
> correct names are `user_feat` / `device_feat` / `dst_feat`. The lengths of `features`
> and of the `*_feat` are validated: a wrong length receives a **422**.

The edge message travels on the access edge `user → resource`; the four binding edges
(`source → config`, `config → device`, `config → user`, `device → user`)
carry null messages and capture the relational pattern breaks (e.g. credential theft:
never-seen IP, config and device attaching to a known user). The returned score is the
maximum over the edges present.

Response of `/infer` and `/score`:

```json
{ "anomaly_score": 0.83, "is_anomaly": true, "threshold": 0.6264 }
```

### Anti-poisoning flow mapping (OPA gatekeeper)

The two-step schema of the previous section is realized as follows:

1. Orchestrator → `POST /infer` → obtains `anomaly_score` (read-only with respect to the
   baseline; only admits new keys in the registry).
2. Orchestrator → OPA with the request + score.
3. If **ALLOW** → `POST /update` (commits into the model). If **DENY** → no call to
   `/update`: the hostile event never enters the baseline (memory + neighbourhood). The
   only trace of a DENYed event is the registry slot allocated at admission, which does
   not modify the model's learned state.

### Identity handling (new users and guests, Hashed Identity)

Being trained on synthetic data, in production the model will only see entities
(users/IPs) never seen before. Thanks to the dynamic memory handling and the use of the
**Hashed Identity**, the model allocates a new RAM slot in real time for every unknown
identity (cold-start) by computing the scalable URI hashing on the fly
(`hash(URI) % buckets`). This provides at once a coherent and inductive embedding base
even for the just-discovered nodes.

For this reason, the Orchestrator must inject the privileges at runtime via `user_feat`:

- **Authenticated users (new nodes)**: the Orchestrator must compute role and clearance
  (e.g. extracted from the JWT) as float values and pass them in `user_feat`. The model
  will write them into the just-allocated slot, and from that moment it will know how to
  apply the correct policies for that user.
- **Guest users (unauthenticated)**: when the request (e.g. to `/login` or public
  endpoints) comes from an IP without a session, `key_user` will be `anonymous`, the source
  key `src:<ip>`, and `user_feat` must be an array of zeros (`[0.0, 0.0, ...]`). This
  corresponds to the minimum privilege level (Clearance=0, Tier=0). The model will allow
  calls to the public routes, but will block as anomalous any attempt towards protected
  endpoints. As soon as the user logs in, the Orchestrator starts passing their real
  features, effectively "promoting" their privileges.

### Operational constraints

- **Single process/replica.** The model is a mutable in-RAM state (memory,
  neighbourhood, registry); multiple workers/replicas would diverge and overwrite each
  other in `/persist`. Start with a single uvicorn worker (already configured) and do
  **not** scale this service horizontally.
- **Key and Resource continuity.** The registry serialized by the training preregisters
  the exact strings of the endpoint URIs (e.g. `/api/v1/personnel`,
  `/api/v1/reactor-parameters` — the canonical list is `RESOURCE_URIS` in
  `src/data/stream_synthetic.py`). The Orchestrator MUST use these exact strings as
  `key_dst`: sub-routes with path-parameters (e.g. `/api/v1/personnel/123`) must be
  normalized to the base route before the call (see `normalizeAIPath` in
  services/security-orchestrator). If a different string is used, the model will interpret
  it as a never-seen backend (falsifying the detections).
- **Grace Period (Cold-Start break-in for users).** Because in production the
  Orchestrator will meet completely new user/device keys (`key_user`/`key_device`), the
  model will assign these identities a high initial anomaly score, for the lack of history
  (cold-start). The Orchestrator **must** apply a "Grace Period" on these new entities:
  for the very first events (e.g. the first 5-10), it must trust only OPA's static
  validation and force the call to `/update`, ignoring the AI score. This lets the model
  quickly build a "safe" baseline for the new user.

### Example: direct call (curl)

```bash
# Read-only score of an event
curl -s -X POST http://localhost:8888/infer \
  -H 'Content-Type: application/json' \
  -d '{"key_user":"alice","key_device":"tpm:a1b2c3","key_source":"src:10.0.0.7","key_config":"conf:guest","key_dst":"/api/v1/documents","timestamp":1717000000,"features":[1.0,0.0,0.0,0.0,0.0,0.67,0.5,0.12,0.08,0.31]}'
# -> {"anomaly_score":0.83,"is_anomaly":true,"threshold":0.6264}
```

### Example: integration from the orchestrator (Go)

The three-step anti-poisoning flow (`/infer` → OPA → `/update`) is written with the
standard library only:

```go
type Event struct {
    KeyUser   string    `json:"key_user"`
    KeyDevice string    `json:"key_device"`           // "tpm:<id>" | "ck:<cookie>" | "ipdev:<ip>"
    KeySource string    `json:"key_source,omitempty"` // the client "src:<ip>" (optional)
    KeyConfig string    `json:"key_config,omitempty"` // "conf:<ja3>" (default "conf:guest")
    KeyDst    string    `json:"key_dst"`
    Timestamp int64     `json:"timestamp"`
    Features  []float64 `json:"features"`             // len == msg_dim (10)
    UserFeat  []float64 `json:"user_feat,omitempty"`  // len == node_feat_dim (16)
    DeviceFeat []float64 `json:"device_feat,omitempty"`
    DstFeat   []float64 `json:"dst_feat,omitempty"`
    Flagged   bool      `json:"flagged,omitempty"`    // only /update: is_anomaly of /infer
}
type ScoreResp struct {
    AnomalyScore float64 `json:"anomaly_score"`
    IsAnomaly    bool    `json:"is_anomaly"`
    Threshold    float64 `json:"threshold"`
}

func post(base, path string, in, out any) error {
    b, _ := json.Marshal(in)
    resp, err := http.Post(base+path, "application/json", bytes.NewReader(b))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("graphagate %s: status %d", path, resp.StatusCode)
    }
    if out != nil {
        return json.NewDecoder(resp.Body).Decode(out)
    }
    return nil
}

// For every access event:
ev := Event{KeyUser: userID, KeyDevice: deviceID, KeySource: clientIP, KeyConfig: ja3,
    KeyDst: resURI, Timestamp: time.Now().Unix(),
    Features: edgeSignals, UserFeat: userAttrs, DeviceFeat: devAttrs, DstFeat: dstAttrs}

var s ScoreResp
if err := post(base, "/infer", ev, &s); err != nil { /* fail-closed */ }

allow := opa.Decide(req, s.AnomalyScore)   // OPA is the final decider
if allow {
    ev.Flagged = s.IsAnomaly               // reports the verdict: arms precursor + trust
    _ = post(base, "/update", ev, nil)     // commits ONLY if approved
}
```

> **Fail-closed**: if `/infer` does not answer (timeout, service not ready), treat
> the event as suspicious at the policy level instead of letting it pass.

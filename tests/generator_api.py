import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import ZTAStreamSimulator

app = FastAPI(title="ZTA Synthetic Generator API")

# Global simulator instance
cfg = TGNConfig()
simulator = ZTAStreamSimulator(
    num_users=cfg.num_users,
    num_devices=cfg.num_devices,
    num_sources=cfg.num_sources,
    # Same reason as tests/generator.py: these two must track the trained config, or the
    # generated stream lives in a different entity space than the checkpoint.
    num_configs=cfg.num_configs,
    guest_device_fallback=cfg.guest_device_fallback,
    num_resources=cfg.num_resources,
    num_wipe_slots=cfg.num_wipe_slots,
    num_theft_slots=cfg.num_theft_slots,
    benign_explore_prob=cfg.benign_explore_prob,
    p_roam=cfg.p_roam,
    p_shared_device=cfg.p_shared_device,
    p_cookie_wipe=cfg.p_cookie_wipe,
    p_cred_theft=cfg.p_cred_theft,
    admission_horizon=None,
    seed=cfg.seed,
    use_resource_risk=cfg.use_resource_risk,
    use_source_internal=cfg.use_source_internal,
)

class AddResourceRequest(BaseModel):
    uri: str
    methods: list[int]
    classification: Optional[str] = None
    categories: Optional[list[str]] = None
    risk: float = 0.5

@app.post("/add_resource")
def add_resource(req: AddResourceRequest):
    """Add a new resource to the synthetic generator."""
    if req.uri in simulator.resource_uris:
        raise HTTPException(status_code=400, detail="Resource already exists")
    
    cat_set = set(req.categories) if req.categories else None
    simulator.add_resource(
        uri=req.uri,
        methods=set(req.methods),
        classification=req.classification,
        categories=cat_set,
        risk=req.risk
    )
    return {"ok": True, "num_resources": simulator.num_resources}

@app.get("/next_event")
def next_event():
    """Generate and return the next synthetic event."""
    ev = simulator.step()
    nf = simulator.node_features
    event_dict = {
        "key_user": ev["key_user"],
        "key_device": ev["key_device"],
        "key_source": ev["key_source"],
        "key_config": ev["key_config"],
        "key_dst": ev["key_dst"],
        "timestamp": int(ev["t"]),
        "features": [float(v) for v in ev["features"]],
        "user_feat": nf[ev["user"]].tolist(),
        "device_feat": nf[ev["device"]].tolist(),
        "dst_feat": nf[ev["dst"]].tolist(),
        "label": ev["label"],
        "type": ev["etype"],
    }
    return event_dict

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("GENERATOR_HOST", "0.0.0.0")
    port = int(os.environ.get("GENERATOR_PORT", "8889"))
    uvicorn.run(app, host=host, port=port)

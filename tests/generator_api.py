import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import ZTAStreamSimulator, stream_kwargs_from_cfg

app = FastAPI(title="ZTA Synthetic Generator API")

# Global simulator instance (same entity space as the trained checkpoint).
cfg = TGNConfig()
simulator = ZTAStreamSimulator(**stream_kwargs_from_cfg(cfg), admission_horizon=None)

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

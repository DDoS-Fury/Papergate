"""HTTP inference service for the streaming TGN.

This is the **deployable inference microservice**. It wraps the serving primitives
in :mod:`graphagate.serve_tgn` behind a small REST/JSON API so a ZTA orchestrator
(here, written in Go) can score access events over the network without any Python
or `.proto`/gRPC toolchain — plain ``net/http`` + ``encoding/json``.

Run it (mirrors the ``python -m`` entrypoint convention of the Docker image)::

    python -m graphagate.serve_api

Endpoints
---------
- ``GET  /health``  — readiness + loaded parameters.
- ``POST /infer``   — score an event **without** mutating state (step 1 of the
  anti-poisoning flow: orchestrator scores, asks OPA, then commits only on ALLOW).
- ``POST /update``  — commit an event the caller already judged benign (post-ALLOW).
- ``POST /score``   — score + internal gate + conditional update (OPA-less / testing).
- ``POST /persist`` — write the evolved in-RAM state back to ``public/``.

Statefulness (important)
------------------------
The model holds **mutable** state in RAM (memory, neighbour history, registry). The
service therefore **must run with a single worker** (one process); multiple workers
would each own a divergent copy of the model and clobber each other on save-back. A
module-level lock serialises every model-touching request — the model is sequential
by design (event-by-event), so this is the correct semantics, not a bottleneck hack.
"""

from __future__ import annotations

import os
import threading
import time
import socket
import platform
from contextlib import asynccontextmanager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_sys_stats():
    if HAS_PSUTIL:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_gb": psutil.virtual_memory().used / (1024**3)
        }
    return {"cpu_percent": 0.0, "ram_gb": 0.0}
from typing import Optional, Union

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from graphagate.config import TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.serve_tgn import (
    commit_event,
    load_model,
    save_model,
    score_event,
)

# External entity keys arrive as JSON scalars; accept str or int (the orchestrator
# owns its key space — see the integration doc's note on key continuity).
EntityKey = Union[str, int]


class _State:
    """Process-wide serving state, populated at startup."""

    def __init__(self) -> None:
        self.model = None
        self.registry = None
        self.threshold: float = 0.0
        self.threshold_dirty: float = 0.0
        self.hp: dict = {}
        self.device: Optional[torch.device] = None
        self.checkpoint_path = os.environ.get("GRAPHAGATE_CHECKPOINT", str(TGN_CHECKPOINT_PATH))
        self.stats_path = os.environ.get("GRAPHAGATE_STATS", str(TGN_STATS_PATH))
        # Serialises all access to the (mutable, sequential) model state.
        self.lock = threading.Lock()
        self.active_websockets: list[WebSocket] = []

    @property
    def loaded(self) -> bool:
        return self.model is not None


STATE = _State()

async def _broadcast_task(event_data: dict):
    if not STATE.active_websockets:
        return
    dead = []
    for ws in STATE.active_websockets:
        try:
            await ws.send_json(event_data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in STATE.active_websockets:
            STATE.active_websockets.remove(ws)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint once on startup; save the evolved state on shutdown."""
    STATE.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, registry, threshold, threshold_dirty, hp = load_model(
        STATE.checkpoint_path, STATE.stats_path, STATE.device
    )
    STATE.model, STATE.registry, STATE.threshold, STATE.hp = model, registry, threshold, hp
    STATE.threshold_dirty = threshold_dirty
    print(
        f"[serve_api] model_loaded device={STATE.device} "
        f"registry={len(registry)}/{int(hp['capacity'])} "
        f"threshold={threshold:.6f} threshold_dirty={threshold_dirty:.6f}",
        flush=True,
    )
    yield
    # SIGTERM (container stop) → persist so approved events survive the restart.
    with STATE.lock:
        save_model(
            STATE.model, STATE.registry, STATE.threshold, STATE.hp,
            STATE.checkpoint_path, STATE.stats_path,
            threshold_dirty=STATE.threshold_dirty,
        )
    print("[serve_api] state persisted on shutdown", flush=True)


app = FastAPI(title="Graphagate TGN inference", version="0.1.0", lifespan=lifespan)


# --- request / response schemas ------------------------------------------------
class EventIn(BaseModel):
    key_user: EntityKey = Field(..., description="User entity key.")
    key_device: Optional[EntityKey] = Field(
        None, description="Device entity key (hardware id: 'tpm:<id>' or 'ck:<cookie>'). Optional."
    )
    key_dst: EntityKey = Field(..., description="Destination entity key (resource URI).")
    key_source: Optional[EntityKey] = Field(
        None,
        description=(
            "Source entity key (client IP / network context). Optional: when omitted "
            "the source→config edge is skipped (never alias the IP onto key_device)."
        ),
    )
    key_config: Optional[EntityKey] = Field(
        None,
        description=(
            "Client configuration entity key — the TLS/JA3 fingerprint ('conf:<ja3>'). "
            "Optional: defaults to the generic 'conf:guest' when the collector cannot "
            "resolve a fingerprint, so the config node is always present."
        ),
    )
    timestamp: int = Field(..., description="Event time (e.g. Unix epoch, integer).")
    features: list[float] = Field(..., description="Edge message (Zero-Trust signals).")
    user_feat: Optional[list[float]] = Field(
        None, description="User node static attributes (training contract: all zeros — role/clearance ride the edge message)."
    )
    device_feat: Optional[list[float]] = Field(
        None, description="Device node static attributes (tier in node_feat[2])."
    )
    dst_feat: Optional[list[float]] = Field(
        None, description="Destination static attributes."
    )


class ScoreOut(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float


class OkOut(BaseModel):
    ok: bool = True


class PersistOut(BaseModel):
    ok: bool = True
    checkpoint: str
    stats: str


def _validate_dims(ev: EventIn) -> None:
    """Reject payloads whose vectors don't match the trained dimensions."""
    msg_dim = int(STATE.hp["msg_dim"])
    feat_dim = int(STATE.hp["node_feat_dim"])
    if len(ev.features) != msg_dim:
        raise HTTPException(422, f"features must have length {msg_dim}, got {len(ev.features)}")
    for name, vec in (("user_feat", ev.user_feat), ("device_feat", ev.device_feat), ("dst_feat", ev.dst_feat)):
        if vec is not None and len(vec) != feat_dim:
            raise HTTPException(422, f"{name} must have length {feat_dim}, got {len(vec)}")


# --- endpoints -----------------------------------------------------------------

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    STATE.active_websockets.append(websocket)
    
    host_name = socket.gethostname()
    cpu_count = os.cpu_count() or 1
    arch_name = platform.machine()
    device_name = "CPU"
    if STATE.device and STATE.device.type == "cuda":
        device_name = torch.cuda.get_device_name(STATE.device)
    elif STATE.device:
        device_name = str(STATE.device)

    try:
        await websocket.send_json({
            "action": "init_sys_info",
            "host": host_name,
            "arch": f"{cpu_count}-Core {arch_name}",
            "device": device_name
        })
    except Exception:
        pass
        
    async def _periodic_stats():
        try:
            while True:
                await asyncio.sleep(2.0)
                stats = get_sys_stats()
                try:
                    await websocket.send_json({
                        "action": "periodic_stats",
                        "cpu_percent": stats["cpu_percent"],
                        "ram_gb": stats["ram_gb"]
                    })
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    stats_task = asyncio.create_task(_periodic_stats())

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        stats_task.cancel()
        if websocket in STATE.active_websockets:
            STATE.active_websockets.remove(websocket)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE.loaded else "loading",
        "model_loaded": STATE.loaded,
        "device": str(STATE.device),
        "registry_size": len(STATE.registry) if STATE.loaded else 0,
        "capacity": int(STATE.hp.get("capacity", 0)),
        "threshold": STATE.threshold,
        "threshold_dirty": STATE.threshold_dirty,
        "msg_dim": int(STATE.hp.get("msg_dim", 0)),
        "node_feat_dim": int(STATE.hp.get("node_feat_dim", 0)),
        "schema_version": int(STATE.hp.get("schema_version", 1)),
    }


@app.post("/infer", response_model=ScoreOut)
def infer(ev: EventIn, background_tasks: BackgroundTasks) -> ScoreOut:
    """Score an event read-only (admits the entities, never advances memory)."""
    _validate_dims(ev)
    t0 = time.perf_counter()
    with STATE.lock:
        score, is_anomaly = score_event(
            STATE.model, STATE.registry, STATE.threshold,
            ev.key_user, ev.key_device, ev.key_dst, ev.timestamp, ev.features, STATE.device,
            key_source=ev.key_source, key_config=ev.key_config,
            threshold_dirty=STATE.threshold_dirty,
            user_feat=ev.user_feat, device_feat=ev.device_feat, dst_feat=ev.dst_feat, update=False,
            guest_device_fallback=bool(STATE.hp.get("guest_device_fallback", False)),
        )
    t1 = time.perf_counter()
    sys_stats = get_sys_stats()

    background_tasks.add_task(_broadcast_task, {
        "action": "infer",
        "key_user": ev.key_user,
        "key_device": ev.key_device,
        "key_source": ev.key_source,
        "key_config": ev.key_config,
        "key_dst": ev.key_dst,
        "score": float(score),
        "is_anomaly": bool(is_anomaly),
        "inf_time_ms": (t1 - t0) * 1000,
        "cpu_percent": sys_stats["cpu_percent"],
        "ram_gb": sys_stats["ram_gb"]
    })
    return ScoreOut(anomaly_score=score, is_anomaly=is_anomaly, threshold=STATE.threshold)


@app.post("/update", response_model=OkOut)
def update(ev: EventIn) -> OkOut:
    """Commit a benign-approved event into memory + neighbour history."""
    _validate_dims(ev)
    with STATE.lock:
        commit_event(
            STATE.model, STATE.registry,
            ev.key_user, ev.key_device, ev.key_dst, ev.timestamp, ev.features, STATE.device,
            key_source=ev.key_source, key_config=ev.key_config,
            user_feat=ev.user_feat, device_feat=ev.device_feat, dst_feat=ev.dst_feat,
            guest_device_fallback=bool(STATE.hp.get("guest_device_fallback", False)),
        )
    return OkOut()


@app.post("/score", response_model=ScoreOut)
def score(ev: EventIn, background_tasks: BackgroundTasks) -> ScoreOut:
    """Score + internal anti-poisoning gate + conditional update (OPA-less path)."""
    _validate_dims(ev)
    t0 = time.perf_counter()
    with STATE.lock:
        s, is_anomaly = score_event(
            STATE.model, STATE.registry, STATE.threshold,
            ev.key_user, ev.key_device, ev.key_dst, ev.timestamp, ev.features, STATE.device,
            key_source=ev.key_source, key_config=ev.key_config,
            threshold_dirty=STATE.threshold_dirty,
            user_feat=ev.user_feat, device_feat=ev.device_feat, dst_feat=ev.dst_feat, update=True,
            guest_device_fallback=bool(STATE.hp.get("guest_device_fallback", False)),
        )
    t1 = time.perf_counter()
    sys_stats = get_sys_stats()

    background_tasks.add_task(_broadcast_task, {
        "action": "score",
        "key_user": ev.key_user,
        "key_device": ev.key_device,
        "key_source": ev.key_source,
        "key_config": ev.key_config,
        "key_dst": ev.key_dst,
        "score": float(s),
        "is_anomaly": bool(is_anomaly),
        "inf_time_ms": (t1 - t0) * 1000,
        "cpu_percent": sys_stats["cpu_percent"],
        "ram_gb": sys_stats["ram_gb"]
    })
    return ScoreOut(anomaly_score=s, is_anomaly=is_anomaly, threshold=STATE.threshold)


@app.post("/persist", response_model=PersistOut)
def persist() -> PersistOut:
    """Write the evolved state back to the configured artifact paths."""
    with STATE.lock:
        save_model(
            STATE.model, STATE.registry, STATE.threshold, STATE.hp,
            STATE.checkpoint_path, STATE.stats_path,
            threshold_dirty=STATE.threshold_dirty,
        )
    return PersistOut(checkpoint=STATE.checkpoint_path, stats=STATE.stats_path)


def main() -> None:
    import uvicorn

    host = os.environ.get("GRAPHAGATE_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAPHAGATE_PORT", "8088"))
    # workers=1 is mandatory: the model is a single mutable in-RAM object.
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()

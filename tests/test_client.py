import asyncio
import aiohttp
import time
import argparse
from generator import event_generator
from metrics import MetricsTracker

async def test_client(host="localhost", port=8888, duration_seconds=120, no_device=False):
    tracker = MetricsTracker()
    tracker.start()
    
    gen = event_generator(seed=42, omit_device=no_device)
    
    base_url = f"http://{host}:{port}"
    print(f"Starting test client for {duration_seconds} seconds against {base_url} (no_device={no_device})...")
    user_counts = {}
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # 1. Generate event
            event = await anext(gen)
            
            label = event.pop("label")
            etype = event.pop("type")
            # SIMULATE NEW ENTITIES (Cold-Start in Production)
            # Actors are prefixed so the API does not find them in memory and treats
            # them as virgin users/devices that just entered the ZTA. Resource nodes
            # (e.g. the API routes) are left unmodified.
            event["key_user"] = f"prod_{event['key_user']}"
            if event.get("key_device") is not None:
                event["key_device"] = f"prod_{event['key_device']}"
            if event.get("key_source") is not None:
                event["key_source"] = f"prod_{event['key_source']}"
            if event.get("key_config") is not None:
                event["key_config"] = f"prod_{event['key_config']}"
                
            key_actor = event["key_user"]
            
            # 2. Call /infer
            req_start = time.time()
            try:
                async with session.post(f"{base_url}/infer", json=event) as resp:
                    resp_data = await resp.json()
            except Exception as e:
                print(f"Error during /infer: {e}")
                await asyncio.sleep(1)
                continue
                
            latency = (time.time() - req_start) * 1000
            tracker.record_latency(latency)
            
            is_anomaly = resp_data.get("is_anomaly", False)
            tracker.record_prediction(is_anomaly, label == 1, etype)
            
            # 3. Simulate External Policy / Orchestrator decision (anti-poisoning gate)
            # Per protocol, memory commits advance on events admitted by external policy/sensor
            # validation, never on the model's own anomaly verdict.
            user_counts[key_actor] = user_counts.get(key_actor, 0) + 1
            # OPA would DENY both genuine policy violations (etype 1) and benign
            # human-error denials (etype 6): neither may commit into the baseline.
            is_policy_violation = etype in (1, 6)
            feats = event.get("features", [])
            is_signal_dirty = bool(feats and (feats[0] <= 0.5 or any(s > 0.5 for s in feats[1:4])))
            allow = (not is_policy_violation) and (not is_signal_dirty)
                
            if allow:
                try:
                    async with session.post(f"{base_url}/update", json=event) as resp:
                        if resp.status != 200:
                            print(f"Warning: /update returned {resp.status}")
                except Exception as e:
                    print(f"Error during /update: {e}")
                    
    print("Test finished. Generating report...")
    tracker.stop_and_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGN Streaming Test Client")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8888, help="API port")
    parser.add_argument("--duration", type=int, default=120, help="Test duration in seconds")
    parser.add_argument("--no-device", action="store_true", help="Omit key_device from events")
    args = parser.parse_args()
    
    asyncio.run(test_client(host=args.host, port=args.port, duration_seconds=args.duration, no_device=args.no_device))

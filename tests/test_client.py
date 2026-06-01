import asyncio
import aiohttp
import time
import argparse
from generator import event_generator
from metrics import MetricsTracker

async def test_client(duration_seconds=120):
    tracker = MetricsTracker()
    tracker.start()
    
    gen = event_generator()
    
    print(f"Starting test client for {duration_seconds} seconds...")
    user_counts = {}
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # 1. Generate event
            event = await anext(gen)
            
            label = event.pop("label")
            etype = event.pop("type")
            key_src = event["key_src"]
            
            # 2. Call /infer
            req_start = time.time()
            try:
                async with session.post("http://localhost:8088/infer", json=event) as resp:
                    resp_data = await resp.json()
            except Exception as e:
                print(f"Error during /infer: {e}")
                await asyncio.sleep(1)
                continue
                
            latency = (time.time() - req_start) * 1000
            tracker.record_latency(latency)
            
            is_anomaly = resp_data.get("is_anomaly", False)
            anomaly_score = resp_data.get("anomaly_score", 1.0)
            tracker.record_prediction(is_anomaly, label == 1, etype)
            
            # 3. Simulate Orchestrator/OPA decision
            user_counts[key_src] = user_counts.get(key_src, 0) + 1
            is_policy_violation = (etype == 1)
            
            # GRACE PERIOD: Per i primissimi eventi di un nuovo utente (cold-start),
            # l'Orchestrator si fida delle policy statiche (JWT/OPA) per fargli creare una baseline.
            # Se la policy è OK, ignora l'AI score. Dopo 5 eventi, applica lo sbarramento AI (>= 0.7).
            if user_counts[key_src] <= 5:
                allow = not is_policy_violation
            else:
                allow = (not is_policy_violation) and (anomaly_score < 0.70)
                
            if allow:
                try:
                    async with session.post("http://localhost:8088/update", json=event) as resp:
                        if resp.status != 200:
                            print(f"Warning: /update returned {resp.status}")
                except Exception as e:
                    print(f"Error during /update: {e}")
                    
    print("Test finished. Generating report...")
    tracker.stop_and_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGN Streaming Test Client")
    parser.add_argument("--duration", type=int, default=120, help="Test duration in seconds")
    args = parser.parse_args()
    
    asyncio.run(test_client(args.duration))

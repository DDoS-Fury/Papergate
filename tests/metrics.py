import time
import psutil
try:
    import pynvml
    HAS_NVML = True
    pynvml.nvmlInit()
except ImportError:
    HAS_NVML = False

from codecarbon import OfflineEmissionsTracker
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.latencies = []
        self.predictions = [] # tuples of (is_anomaly, expected_label, expected_type)
        self.tracker = OfflineEmissionsTracker(country_iso_code="ITA", log_level="error")
        
    def start(self):
        self.tracker.start()
        
    def record_latency(self, latency_ms):
        self.latencies.append(latency_ms)
        
    def record_prediction(self, is_anomaly, expected_label, expected_type):
        self.predictions.append((is_anomaly, expected_label, expected_type))
        
    def stop_and_report(self):
        emissions = self.tracker.stop()
        energy_kwh = self.tracker._total_energy.kWh
        
        if self.latencies:
            p50 = np.percentile(self.latencies, 50)
            p90 = np.percentile(self.latencies, 90)
            p99 = np.percentile(self.latencies, 99)
        else:
            p50 = p90 = p99 = 0
            
        ram_percent = psutil.virtual_memory().percent
        vram_info = "N/A"
        if HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_info = f"{meminfo.used / 1024**2:.1f} MB / {meminfo.total / 1024**2:.1f} MB"
            except Exception:
                pass
                
        # Calculate metrics for anomalies (positive class = True)
        tp = sum(1 for p in self.predictions if p[0] and p[1])
        fp = sum(1 for p in self.predictions if p[0] and not p[1])
        fn = sum(1 for p in self.predictions if not p[0] and p[1])
        tn = sum(1 for p in self.predictions if not p[0] and not p[1])
        
        total = len(self.predictions)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("="*50)
        print("TEST CLIENT REPORT")
        print("="*50)
        print(f"Total Events Processed: {total}")
        print(f"Power Consumed: {energy_kwh:.6f} kWh ({emissions:.6f} kg CO2)")
        print(f"RAM Usage: {ram_percent}%")
        print(f"VRAM Usage: {vram_info}")
        print(f"Latency (ms) - P50: {p50:.2f} | P90: {p90:.2f} | P99: {p99:.2f}")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print(f"Anomalies - Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        
        # Breakdown by type
        # types: 0=benign, 1=policy, 2=contextual, 3=lateral
        type_names = {0: "Benign", 1: "Policy", 2: "Contextual", 3: "Lateral"}
        print("\nAccuracy by Event Type (Recall for anomalies, Specificity for benign):")
        for t in [0, 1, 2, 3]:
            t_preds = [p for p in self.predictions if p[2] == t]
            if t_preds:
                t_correct = sum(1 for p in t_preds if p[0] == p[1])
                print(f" - {type_names[t]}: {t_correct}/{len(t_preds)} ({t_correct/len(t_preds)*100:.2f}%)")
        print("="*50)

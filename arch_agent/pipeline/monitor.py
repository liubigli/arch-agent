# arch_agent/pipeline/monitor.py
import json, time
from dataclasses import dataclass, asdict

@dataclass
class RunReport:
    timestamp: str
    laz_file: str
    params: dict          # eps, min_samples, distance_threshold, sample_n
    objects_per_class: dict
    total_objects: int
    total_relationships: dict  # {"L1": n, "L2": n, "L3": n}
    elapsed_pipeline_s: float
    llm_model: str

def save_report(report: RunReport, out_path="run_report.json"):
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

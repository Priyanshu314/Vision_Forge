import requests
import sys
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_inference_test(run_id: str):
    print(f"🚀 Starting Inference & Evaluation for Run: {run_id}")
    
    # 1. Trigger Inference
    print(f"\n[1/2] Triggering POST /infer/{run_id}...")
    try:
        resp = requests.post(f"{BASE_URL}/infer/{run_id}", timeout=300) # Inference can take time
        resp.raise_for_status()
        data = resp.json()
        print("✅ Inference Task Completed.")
    except Exception as e:
        print(f"❌ Failed to trigger inference: {e}")
        return

    # 2. Display Metrics
    print("\n[2/2] Final Metrics & Results:")
    results = data.get("results", {})
    metrics = results.get("metrics", {})
    
    if "error" in metrics:
        print(f"⚠️  Metric Calculation Warning: {metrics['error']}")
    else:
        print("-" * 30)
        print(f"📊 mAP (0.5:0.95): {metrics.get('mAP_50_95', 0):.4f}")
        print(f"📊 mAP (0.5):      {metrics.get('mAP_50', 0):.4f}")
        print(f"📊 mAP (0.75):     {metrics.get('mAP_75', 0):.4f}")
        print(f"📊 Recall (AR100): {metrics.get('AR_100', 0):.4f}")
        print("-" * 30)

    print(f"\n📁 Predictions saved at: {results.get('predictions_path')}")
    print(f"📝 Full results available via: GET {BASE_URL}/results/{run_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_inference.py <run_id>")
        sys.exit(1)
        
    target_run_id = sys.argv[1]
    run_inference_test(target_run_id)

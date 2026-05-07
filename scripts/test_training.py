import requests
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_IMAGE_DIR = Path("test_images")
TEST_IMAGE_DIR.mkdir(exist_ok=True)

def create_dummy_images(count=3):
    from PIL import Image
    import numpy as np
    paths = []
    for i in range(count):
        path = TEST_IMAGE_DIR / f"train_test_{i}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
        img.save(path)
        paths.append(path)
    return paths

def run_training_test():
    print("🚀 Starting End-to-End Training Test")
    
    # 1. Upload
    print("\n[1/4] Uploading images...")
    image_paths = create_dummy_images(5)
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in image_paths]
    resp = requests.post(f"{BASE_URL}/upload", files=files)
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    print(f"✅ Uploaded. Run ID: {run_id}")

    # 2. Annotate (Required for training)
    print("\n[2/4] Adding COCO annotations...")
    coco_data = {
        "images": [{"id": i, "file_name": f"train_test_{i}.jpg"} for i in range(5)],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "bbox": [100, 100, 50, 50],
                "category_id": 1
            } for i in range(5)
        ],
        "categories": [{"id": 1, "name": "defect"}]
    }
    resp = requests.post(f"{BASE_URL}/annotate/{run_id}", json=coco_data)
    resp.raise_for_status()
    print("✅ Annotations saved.")

    # 3. Trigger Training
    print("\n[3/4] Triggering training pipeline...")
    resp = requests.post(f"{BASE_URL}/train/{run_id}")
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"✅ Training started. Task ID: {task_id}")

    # 4. Poll Status
    print("\n[4/4] Polling training status (Ctrl+C to stop)...")
    while True:
        status_resp = requests.get(f"{BASE_URL}/train/status/{task_id}")
        status_resp.raise_for_status()
        state = status_resp.json()["state"]
        meta = status_resp.json().get("meta")
        
        if state == "SUCCESS":
            print(f"\n🎉 Training COMPLETED!")
            print(f"Model saved at: {meta.get('model_path')}")
            break
        elif state == "FAILURE":
            print(f"\n❌ Training FAILED!")
            print(f"Error: {meta}")
            break
        elif state == "PROGRESS":
            print(f"⏳ Epoch: {meta.get('epoch')} | Batch: {meta.get('batch')}/{meta.get('total_batches')} | Loss: {meta.get('loss'):.4f}", end="\r")
        else:
            print(f"📡 Current State: {state}", end="\r")
            
        time.sleep(2)

if __name__ == "__main__":
    try:
        run_training_test()
    except KeyboardInterrupt:
        print("\nStopped polling.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

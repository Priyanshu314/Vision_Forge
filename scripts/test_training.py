import requests
import time
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_test():
    print("🚀 Starting End-to-End Training Test\n")

    # 1. Upload Images
    print("[1/4] Uploading images...")
    img_dir = Path("backend/data/sample_images") # Using the sample images we created
    if not img_dir.exists():
        # Fallback to creating a dummy image if sample_images doesn't exist
        img_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        import numpy as np
        for i in range(3):
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(img_dir / f"test_{i}.jpg")

    files = [("files", open(f, "rb")) for f in img_dir.glob("*.jpg")]
    resp = requests.post(f"{BASE_URL}/upload", files=files)
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    print(f"✅ Uploaded. Run ID: {run_id}\n")

    # 2. Add Annotations
    print("[2/4] Adding COCO annotations...")
    # Mocking some COCO annotations for the uploaded images
    images = []
    annotations = []
    for i, f in enumerate(img_dir.glob("*.jpg")):
        images.append({"id": i, "file_name": f.name})
        annotations.append({
            "id": i,
            "image_id": i,
            "category_id": 1,
            "bbox": [10, 10, 50, 50],
            "area": 2500,
            "iscrowd": 0,
            "segmentation": []
        })
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "defect"}]
    }
    
    ann_resp = requests.post(f"{BASE_URL}/annotate/{run_id}", json=coco_data)
    ann_resp.raise_for_status()
    print("✅ Annotations saved.\n")

    # 3. Start Training
    print("[3/4] Triggering training pipeline...")
    train_resp = requests.post(f"{BASE_URL}/train/{run_id}")
    train_resp.raise_for_status()
    task_id = train_resp.json()["task_id"]
    print(f"✅ Training started. Task ID: {task_id}\n")

    # 4. Poll Status
    print("[4/4] Polling training status (Ctrl+C to stop)...")
    while True:
        status_resp = requests.get(f"{BASE_URL}/train/status/{task_id}")
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data["status"]
        result = data.get("result")
        
        if status == "SUCCESS":
            print(f"\n🎉 Training COMPLETED!")
            print(f"Model saved at: {result.get('model_path')}")
            break
        elif status == "FAILURE":
            print(f"\n❌ Training FAILED!")
            print(f"Error: {result}")
            break
        elif status == "PROGRESS":
            meta = result # In Progress, result contains the meta info
            print(f"⏳ Epoch: {meta.get('epoch')} | Batch: {meta.get('batch')}/{meta.get('total_batches')} | Loss: {meta.get('loss'):.4f}", end="\r")
        else:
            print(f"📡 Current State: {status}", end="\r")
            
        time.sleep(2)

if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")

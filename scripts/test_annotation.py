import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_annotation():
    # 1. Create a run first (using Phase 1 of our previous test)
    print("--- Setting up: Creating a run ---")
    import os
    from pathlib import Path
    import numpy as np
    from PIL import Image
    
    TEST_IMAGE_DIR = Path("test_images")
    TEST_IMAGE_DIR.mkdir(exist_ok=True)
    path = TEST_IMAGE_DIR / "ann_test.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(path)
    
    files = [("files", (os.path.basename(path), open(path, "rb"), "image/jpeg"))]
    resp = requests.post(f"{BASE_URL}/upload", files=files)
    run_id = resp.json()["run_id"]
    print(f"Run created: {run_id}")

    # 2. Test valid annotation
    print("\n--- Testing Valid Annotation ---")
    coco_data = {
        "images": [{"id": 1, "file_name": "ann_test.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [10.5, 20.0, 50.0, 30.0],
                "category_id": 1
            }
        ],
        "categories": [{"id": 1, "name": "defect"}]
    }
    
    resp = requests.post(f"{BASE_URL}/annotate/{run_id}", json=coco_data)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

    # 3. Test invalid category_id
    print("\n--- Testing Invalid Category ID ---")
    coco_data_invalid_cat = {
        "images": [{"id": 1, "file_name": "ann_test.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 10, 10], "category_id": 99}],
        "categories": [{"id": 1, "name": "defect"}]
    }
    resp = requests.post(f"{BASE_URL}/annotate/{run_id}", json=coco_data_invalid_cat)
    print(f"Status: {resp.status_code}")
    print(f"Detail: {resp.json().get('detail')}")

    # 4. Test invalid bbox (negative width)
    print("\n--- Testing Invalid Bbox (Negative Width) ---")
    coco_data_invalid_bbox = {
        "images": [{"id": 1, "file_name": "ann_test.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, -10, 10], "category_id": 1}],
        "categories": [{"id": 1, "name": "defect"}]
    }
    resp = requests.post(f"{BASE_URL}/annotate/{run_id}", json=coco_data_invalid_bbox)
    print(f"Status: {resp.status_code}")
    print(f"Detail: {resp.json().get('detail')}")

if __name__ == "__main__":
    test_annotation()

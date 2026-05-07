import requests
import os
from pathlib import Path
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_IMAGE_DIR = Path("test_images")
TEST_IMAGE_DIR.mkdir(exist_ok=True)

def create_dummy_images(count=5):
    from PIL import Image
    import numpy as np
    paths = []
    for i in range(count):
        path = TEST_IMAGE_DIR / f"test_{i}.jpg"
        # Create a random noise image with some variation so they cluster differently
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(path)
        paths.append(path)
    return paths

def test_pipeline():
    print("--- Phase 1: Uploading Images ---")
    image_paths = create_dummy_images(5)
    
    files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in image_paths]
    
    try:
        response = requests.post(f"{BASE_URL}/upload", files=files)
        response.raise_for_status()
    except Exception as e:
        print(f"Upload failed: {e}")
        return

    data = response.json()
    run_id = data["run_id"]
    print(f"Upload successful! Run ID: {run_id}")
    print(f"Images saved: {data['num_images']}")

    print("\n--- Phase 2: Sampling Representative Images ---")
    # Request 2 representative samples
    try:
        sample_response = requests.get(f"{BASE_URL}/sample/{run_id}?k=2")
        sample_response.raise_for_status()
        sample_data = sample_response.json()
        print(f"Sampling complete!")
        print(f"Sampled paths: {len(sample_data['sampled_paths'])} images selected")
        for p in sample_data['sampled_paths']:
            print(f" - {p}")
    except Exception as e:
        print(f"Sampling failed: {e}")

if __name__ == "__main__":
    # Wait a moment for server to be ready
    time.sleep(2)
    test_pipeline()

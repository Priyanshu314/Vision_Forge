import os
import shutil
from pathlib import Path
from ..core.celery_app import celery_app
from ..ml.model import ModelWrapper
from ..core.config import get_config, load_config

@celery_app.task(bind=True)
def train_model_task(self, run_id: str):
    """Celery task for training the model using rfdetr's native trainer."""
    
    # 1. Setup Config
    load_config("backend/config.yaml")
    config = get_config()
    train_cfg = config.training
    model_cfg = config.model
    
    # 2. Path Setup
    run_dir = Path("data/runs") / run_id
    images_dir = run_dir / "images"
    ann_file = run_dir / "annotations" / "train.json"
    dataset_dir = run_dir / "dataset"
    
    # We save official weights in a dedicated outputs folder identifiable by run_id
    output_base_dir = Path("outputs") / run_id
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Create Roboflow-style Dataset Structure
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"
    
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 4. Perform an actual dataset split (80% train, 10% valid, 10% test)
    import json
    import random
    
    if not ann_file.exists():
        raise Exception("Annotation file missing.")
        
    with open(ann_file, "r") as f:
        coco = json.load(f)
        
    images = coco.get("images", [])
    random.shuffle(images)
    
    n = len(images)
    
    train_split = train_cfg.train_split
    valid_split = train_cfg.valid_split
    
    train_end = max(1, int(n * train_split))
    valid_end = max(train_end + 1, int(n * (train_split + valid_split))) if n > 2 else train_end + 1
    
    # Fallback for very tiny datasets: ensure at least 1 image per set if possible
    if n < 3:
        splits = {"train": images, "valid": images, "test": images}
    else:
        splits = {
            "train": images[:train_end],
            "valid": images[train_end:valid_end],
            "test": images[valid_end:]
        }
        
    for split_name, split_images in splits.items():
        if split_name == "train":
            d = train_dir
        elif split_name == "valid":
            d = valid_dir
        else:
            d = test_dir
            
        # Filter annotations for this split
        split_img_ids = {img["id"] for img in split_images}
        split_anns = [ann for ann in coco.get("annotations", []) if ann["image_id"] in split_img_ids]
        
        # Save split COCO JSON
        split_coco = {
            "images": split_images,
            "categories": coco.get("categories", []),
            "annotations": split_anns
        }
        with open(d / "_annotations.coco.json", "w") as f:
            json.dump(split_coco, f)
            
        # Copy corresponding images
        for img in split_images:
            img_path = images_dir / img["file_name"]
            if img_path.exists():
                shutil.copy(img_path, d / img["file_name"])
            
    # 5. Initialize Official Model
    num_classes = model_cfg.num_classes
    wrapper = ModelWrapper(num_classes=num_classes)
    wrapper.load_model({
        "size": model_cfg.size,
        "pretrained": model_cfg.pretrained
    })
    
    # 6. Trigger Native Training
    print(f"Starting native rfdetr training for run: {run_id}...")
    dataset_path_abs = str(dataset_dir.resolve())
    output_dir_abs = str(output_base_dir.resolve())
    
    wrapper.train(
        dataset_path=dataset_path_abs,
        epochs=train_cfg.epochs,
        lr=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        output_dir=output_dir_abs
    )

    # 7. Identify the saved weight file
    # rfdetr typically saves as 'last.pth' or 'checkpoint.pth' in the output_dir
    model_path = output_base_dir / "last.pth"
    if not model_path.exists():
        # Fallback to checking for any .pth file in the output dir
        pth_files = list(output_base_dir.glob("*.pth"))
        if pth_files:
            model_path = pth_files[0]

    return {
        "status": "completed", 
        "run_id": run_id, 
        "model_path": str(model_path.resolve()),
        "output_directory": output_dir_abs
    }

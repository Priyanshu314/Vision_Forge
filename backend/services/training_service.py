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
    
    # 3. Create Roboflow-style Dataset Structure
    dataset_dir = run_dir / "dataset"
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"
    
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 4. Copy images and annotations to 'train', 'valid', and 'test' folders
    # We populate all splits to ensure the rfdetr pipeline has data for all phases
    for d in [train_dir, valid_dir, test_dir]:
        if images_dir.exists():
            for img_path in images_dir.glob("*"):
                if img_path.is_file():
                    shutil.copy(img_path, d / img_path.name)
        
        if ann_file.exists():
            shutil.copy(ann_file, d / "_annotations.coco.json")
            
    # 5. Initialize Official Model
    num_classes = model_cfg.num_classes
    wrapper = ModelWrapper(num_classes=num_classes)
    wrapper.load_model({
        "size": model_cfg.size,
        "pretrained": model_cfg.pretrained
    })
    
    # 6. Trigger Native Training
    print(f"Starting native rfdetr training for {run_id}...")
    dataset_path_abs = str(dataset_dir.resolve())
    
    wrapper.train(
        dataset_path=dataset_path_abs,
        epochs=train_cfg.epochs,
        lr=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size
    )

    # 7. Save Final Artifact
    save_dir = run_dir / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.pt"
    wrapper.save(str(save_path))
    
    return {"status": "completed", "run_id": run_id, "model_path": str(save_path)}

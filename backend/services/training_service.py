import os
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
    # rfdetr needs the directory containing annotations/train.json and the images
    run_dir = Path("data/runs") / run_id
    # Ensure full path for the library
    run_dir_abs = str(run_dir.resolve())
    
    # 3. Initialize Official Model
    num_classes = model_cfg.num_classes
    wrapper = ModelWrapper(num_classes=num_classes)
    wrapper.load_model({
        "size": model_cfg.size,
        "pretrained": model_cfg.pretrained
    })
    
    # 4. Trigger Native Training
    # rfdetr handles the mixed precision and logging internally
    print(f"Starting native rfdetr training for {run_id}...")
    
    # We pass the root run directory. 
    # The library will look for annotations/train.json and images/
    wrapper.train(
        dataset_path=run_dir_abs,
        epochs=train_cfg.epochs,
        lr=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size
    )

    # 5. Save Final Artifact
    save_dir = run_dir / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.pt"
    wrapper.save(str(save_path))
    
    return {"status": "completed", "run_id": run_id, "model_path": str(save_path)}

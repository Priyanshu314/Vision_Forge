import torch
from torch.utils.data import DataLoader
from pathlib import Path
from ..core.celery_app import celery_app
from ..ml.model import ModelWrapper
from ..ml.dataset import CocoDataset, collate_fn
from ..core.config import get_config

@celery_app.task(bind=True)
def train_model_task(self, run_id: str):
    """Celery task for training the model."""
    config = get_config()
    train_cfg = config.training
    model_cfg = config.model
    
    # 1. Setup Data
    dataset = CocoDataset(run_id)
    loader = DataLoader(
        dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 2. Setup Model
    # Determine num_classes from dataset or config
    num_classes = model_cfg.num_classes + 1 # +1 for background in some DETR implementations
    wrapper = ModelWrapper(num_classes=num_classes)
    wrapper.load_model({
        "pretrained": model_cfg.pretrained,
        "freeze_backbone": True # Mandatory requirement
    })
    
    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, wrapper.model.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. Training Loop
    epochs = train_cfg.epochs
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, targets) in enumerate(loader):
            loss = wrapper.train_step(images, targets, optimizer, scaler)
            total_loss += loss
            
            # Update Celery status
            self.update_state(
                state='PROGRESS',
                meta={
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'total_batches': len(loader),
                    'loss': loss
                }
            )
            
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(loader)}")

    # 5. Save Model
    save_dir = Path("data/runs") / run_id / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.pt"
    wrapper.save(str(save_path))
    
    return {"status": "completed", "run_id": run_id, "model_path": str(save_path)}

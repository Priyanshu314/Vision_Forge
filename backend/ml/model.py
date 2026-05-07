import torch
import torch.nn as nn
from torchvision.models.detection import detr_resnet50
from typing import Dict, Any


class ModelWrapper:
    """Wrapper for RF-DETR (nano/small) architecture."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, config: Dict[str, Any]):
        """Initialize the model based on configuration."""
        # Using torchvision DETR as a reliable proxy for the interface
        # In a real RF-DETR setup, we would load the specific nano/small weights here
        self.model = detr_resnet50(
            pretrained=config.get("pretrained", True),
            num_classes=self.num_classes
        )
        
        # Freeze backbone as required
        if config.get("freeze_backbone", True):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.model.to(self.device)
        return self.model

    def train_step(self, images, targets, optimizer, scaler):
        """Perform a single training step with mixed precision."""
        self.model.train()
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        return losses.item()

    def predict(self, images):
        """Perform inference on a list of images."""
        self.model.eval()
        self.model.to(self.device)
        
        # Convert to tensor and normalise if needed
        # Assuming images are already tensors or handled by a transform
        with torch.no_grad():
            outputs = self.model(images)
        return outputs

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

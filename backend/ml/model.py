import torch
import torch.nn as nn
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
from typing import Dict, Any

class ModelWrapper:
    """
    Official Wrapper for Roboflow RF-DETR architectures.
    Supports Nano, Small, and Medium variants.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_wrapper = None
        self.model = None

    def load_model(self, config: Dict[str, Any]):
        """Initialize the official RF-DETR model based on size."""
        size = config.get("size", "small").lower()
        
        if size == "nano":
            self.model_wrapper = RFDETRNano(num_classes=self.num_classes)
        elif size == "medium":
            self.model_wrapper = RFDETRMedium(num_classes=self.num_classes)
        else:
            self.model_wrapper = RFDETRSmall(num_classes=self.num_classes)
            
        # Get the underlying torch model for the training loop
        self.model = self.model_wrapper.model
        
        # Freeze backbone (DINOv2) for low-shot training
        if config.get("freeze_backbone", True):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.model.to(self.device)
        return self.model

    def train_step(self, images, targets, optimizer, scaler):
        """Perform a single training step."""
        self.model.train()
        
        # RF-DETR expectations: images as list of tensors, targets as list of dicts
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            # Standard DETR loss output
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        return losses.item()

    def predict(self, images):
        """Perform inference."""
        self.model.eval()
        images = [img.to(self.device) for img in images]
        with torch.no_grad():
            outputs = self.model(images)
        return outputs

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

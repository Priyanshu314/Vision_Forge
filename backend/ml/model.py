import torch
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
from typing import Dict, Any

class ModelWrapper:
    """
    Official Wrapper for Roboflow RF-DETR architectures.
    Interfaces with the high-level rfdetr library for training and inference.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, config: Dict[str, Any]):
        """Initialize the official RF-DETR model based on size."""
        size = config.get("size", "small").lower()
        
        # Instantiate the official model class
        if size == "nano":
            self.model = RFDETRNano(num_classes=self.num_classes)
        elif size == "medium":
            self.model = RFDETRMedium(num_classes=self.num_classes)
        else:
            self.model = RFDETRSmall(num_classes=self.num_classes)
            
        return self.model

    def train(self, dataset_path: str, epochs: int, lr: float, batch_size: int, output_dir: str):
        """
        Use the native rfdetr training method with automatic saving.
        """
        self.model.train(
            dataset_dir=dataset_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=str(self.device),
            output_dir=output_dir
        )

    def predict(self, image):
        """Perform inference using the library's high-level API."""
        return self.model.infer(image)

    def save(self, path: str):
        """
        Manual save fallback using the underlying torch module.
        Useful for exporting models outside the native output_dir.
        """
        target = self.model
        for _ in range(3):
            if hasattr(target, "state_dict"):
                torch.save(target.state_dict(), path)
                return
            if hasattr(target, "model"):
                target = target.model
            else:
                break
        torch.save(target, path)

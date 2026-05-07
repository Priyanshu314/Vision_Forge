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

    def train(self, dataset_path: str, epochs: int, lr: float, batch_size: int):
        """
        Use the native rfdetr training method.
        This handles mixed precision, backbone freezing, and optimizers internally.
        """
        # The rfdetr library typically expects a path to a COCO directory or similar
        # We will point it to the annotations we saved
        self.model.train(
            dataset_dir=dataset_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=str(self.device)
        )

    def predict(self, image):
        """Perform inference using the library's high-level API."""
        # Note: model.infer() handles PIL images or paths
        return self.model.infer(image)

    def save(self, path: str):
        """Save the model checkpoint."""
        # rfdetr usually saves its own checkpoints, but we can export/save specifically
        self.model.save(path)

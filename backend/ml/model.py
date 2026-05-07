import torch
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
from typing import Dict, Any, List, Union

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
        """
        Initialize the official RF-DETR model.
        Supports loading weights directly via 'pretrain_weights'.
        """
        size = config.get("size", "small").lower()
        weights = config.get("pretrain_weights")
        
        # Instantiate the official model class with weights if provided
        kwargs = {"num_classes": self.num_classes}
        if weights:
            kwargs["pretrain_weights"] = weights

        if size == "nano":
            self.model = RFDETRNano(**kwargs)
        elif size == "medium":
            self.model = RFDETRMedium(**kwargs)
        else:
            self.model = RFDETRSmall(**kwargs)
            
        return self.model

    def train(self, dataset_path: str, epochs: int, lr: float, batch_size: int, output_dir: str):
        """Use the native rfdetr training method."""
        self.model.train(
            dataset_dir=dataset_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=str(self.device),
            output_dir=output_dir
        )

    def predict(self, images: Union[Any, List[Any]], threshold: float = 0.5):
        """
        Perform batch inference using the high-level API.
        Returns a list of detections for each image.
        """
        # model.predict handles single images or lists of images
        return self.model.predict(images, threshold=threshold)

    def save(self, path: str):
        """Manual save fallback."""
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

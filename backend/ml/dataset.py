import os
import json
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pathlib import Path


class CocoDataset(Dataset):
    """COCO format dataset loader with Albumentations."""

    def __init__(self, run_id: str, data_root: str = "data/runs"):
        self.run_id = run_id
        self.image_dir = Path(data_root) / run_id / "images"
        self.ann_file = Path(data_root) / run_id / "annotations" / "train.json"
        
        with open(self.ann_file, 'r') as f:
            self.coco = json.load(f)
            
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        
        # Mapping image_id to annotations
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.transform = A.Compose([
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomCrop(height=224, width=224, p=0.5), # Configurable
            A.GridDropout(p=0.2),
            A.Resize(height=480, width=480), # Standard input size
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = str(self.image_dir / file_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]
        
        # Apply transforms
        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
        image = transformed['image'] / 255.0  # Normalise
        
        # Prepare targets for DETR
        # DETR expects [x_center, y_center, w, h] normalized for bboxes
        # And boxes in [0, 1] range
        target = {
            "boxes": torch.as_tensor(transformed['bboxes'], dtype=torch.float32),
            "labels": torch.as_tensor(transformed['category_ids'], dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

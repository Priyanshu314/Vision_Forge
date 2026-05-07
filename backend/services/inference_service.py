import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ..ml.model import ModelWrapper
from ..core.config import get_config, load_config
from .upload_service import DATA_DIR

def run_inference_and_eval(run_id: str):
    """
    1. Load trained model using native weights loading.
    2. Run high-level batch inference on all images.
    3. Save COCO-style predictions.
    4. Calculate industry-standard metrics.
    """
    # 1. Setup
    load_config("backend/config.yaml")
    config = get_config()
    run_dir = DATA_DIR / run_id
    images_dir = run_dir / "images"
    ann_path = run_dir / "annotations" / "train.json"
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate model weights
    model_weights = Path("outputs") / run_id / "last.pth"
    if not model_weights.exists():
        pth_files = list((Path("outputs") / run_id).glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No trained model found for run {run_id}")
        model_weights = pth_files[0]

    # 2. Initialize Model with Native Weights
    wrapper = ModelWrapper(num_classes=config.model.num_classes)
    wrapper.load_model({
        "size": config.model.size,
        "pretrain_weights": str(model_weights.resolve())
    })
    
    # 3. Batch Inference using High-Level API
    image_paths = list(images_dir.glob("*"))
    predictions = []
    
    print(f"Running native batch inference on {len(image_paths)} images...")
    
    # Process in batches of 8
    for i in range(0, len(image_paths), 8):
        batch_paths = image_paths[i:i+8]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        
        # Using the native model.predict() via our wrapper
        # This returns a list of supervision.Detections objects
        batch_results = wrapper.predict(batch_images, threshold=0.5)
        
        for img_path, detections in zip(batch_paths, batch_results):
            # detections is a supervision.Detections object
            # Convert to COCO format
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                predictions.append({
                    "image_id": img_path.name,
                    "category_id": int(detections.class_id[i]) + 1, # 1-indexed for COCO
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(detections.confidence[i])
                })

    # 4. Save Predictions
    pred_path = pred_dir / "preds.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # 5. Calculate Metrics
    metrics = {}
    if ann_path.exists():
        metrics = calculate_coco_metrics(str(ann_path), str(pred_path))
        
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return {"predictions_path": str(pred_path), "metrics": metrics}

def calculate_coco_metrics(gt_path: str, pred_path: str):
    """Use pycocotools to calculate mAP metrics."""
    try:
        coco_gt = COCO(gt_path)
        with open(pred_path, 'r') as f:
            preds = json.load(f)
            
        name_to_id = {img['file_name']: img['id'] for img in coco_gt.imgs.values()}
        
        coco_preds = []
        for p in preds:
            img_id = name_to_id.get(p['image_id'])
            if img_id is not None:
                p_copy = p.copy()
                p_copy['image_id'] = img_id
                coco_preds.append(p_copy)
        
        if not coco_preds:
            return {"error": "No matching images found"}

        coco_dt = coco_gt.loadRes(coco_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            "mAP_50_95": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "AR_100": coco_eval.stats[8]
        }
    except Exception as e:
        return {"error": str(e)}

import json
from pathlib import Path
from .upload_service import DATA_DIR
from ..schemas.annotation import CocoFormat


def save_annotations(run_id: str, coco_data: CocoFormat) -> str:
    """Validate and save annotations to data/runs/{run_id}/annotations/train.json."""
    
    # 1. Validation for category_id consistency
    category_ids = {cat.id for cat in coco_data.categories}
    for ann in coco_data.annotations:
        if ann.category_id not in category_ids:
            raise ValueError(f"Annotation category_id {ann.category_id} not found in categories list.")

    # 2. Validation and Enrichment
    image_ids = {img.id for img in coco_data.images}
    for ann in coco_data.annotations:
        # Auto-calculate area if not set or zero
        if ann.area <= 0:
            ann.area = ann.bbox[2] * ann.bbox[3]
        
        # Ensure segmentation and iscrowd are initialized
        if ann.segmentation is None:
            ann.segmentation = []

    # 3. Create directory if not exists
    ann_dir = DATA_DIR / run_id / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = ann_dir / "train.json"
    
    # 4. Save to JSON
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(coco_data.model_dump_json(indent=2))
        
    return str(file_path.resolve())

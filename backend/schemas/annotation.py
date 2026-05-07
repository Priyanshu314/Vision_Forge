from pydantic import BaseModel, Field, field_validator
from typing import List


class CocoImage(BaseModel):
    id: int
    file_name: str


class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[x, y, width, height]")
    area: float = Field(default=0.0)
    iscrowd: int = Field(default=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if any(x < 0 for x in v):
            raise ValueError("Bbox coordinates and dimensions must be non-negative")
        # width and height should be positive
        if v[2] <= 0 or v[3] <= 0:
            raise ValueError("Bbox width and height must be greater than zero")
        return v


class CocoCategory(BaseModel):
    id: int
    name: str


class CocoFormat(BaseModel):
    images: List[CocoImage]
    annotations: List[CocoAnnotation]
    categories: List[CocoCategory]

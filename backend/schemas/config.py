from pydantic import BaseModel, Field
from typing import List, Optional

class AugmentationConfig(BaseModel):
    """Configuration for data augmentation steps."""
    flip_horizontal: bool = Field(default=False, description="Whether to flip images horizontally")
    random_crop: bool = Field(default=False, description="Apply random cropping")
    noise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Amount of Gaussian noise to add")

class ModelConfig(BaseModel):
    """Model hyper‑parameters and architecture details."""
    size: str = Field(default="small", description="RF-DETR size: nano, small, or medium")
    pretrained: bool = Field(default=True, description="Use pretrained weights if available")
    num_classes: int = Field(default=1, gt=0, description="Number of output classes for detection")

class TrainingConfig(BaseModel):
    """Training loop configuration."""
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: str = Field(default="adam", description="Optimizer name")
    weight_decay: float = Field(default=0.0, ge=0.0)

class SamplingConfig(BaseModel):
    """Dataset sampling strategy."""
    strategy: str = Field(default="random", description="Sampling strategy, e.g., random, balanced")
    seed: int = Field(default=42, description="Random seed for reproducibility")

class RunConfig(BaseModel):
    """Top‑level run configuration aggregating all sections."""
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    experiment_name: str = Field(default="default_experiment")
    output_dir: str = Field(default="./outputs")

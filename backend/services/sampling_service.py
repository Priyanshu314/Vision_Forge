"""Sampling service – embedding extraction, clustering, and representative selection."""

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
# We lazily load the DINOv2 model so the import is fast and the heavy weight
# download only happens once on first call.

_dino_model = None
_dino_transform = None

EMBED_SIZE = 224  # DINOv2 expects 224×224 input


def _load_dino():
    """Load the DINOv2-small model via torch hub (cached after first call)."""
    global _dino_model, _dino_transform
    if _dino_model is not None:
        return

    import torch
    from torchvision import transforms

    _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    _dino_model.eval()

    _dino_transform = transforms.Compose([
        transforms.Resize((EMBED_SIZE, EMBED_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_embedding(image_path: str) -> np.ndarray:
    """Return a normalised 1-D embedding vector for a single image."""
    import torch

    _load_dino()

    img = Image.open(image_path).convert("RGB")
    tensor = _dino_transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        embedding = _dino_model(tensor)  # (1, D)

    vec = embedding.squeeze(0).numpy().astype(np.float64)
    # L2-normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Clustering + selection
# ---------------------------------------------------------------------------

def sample_representative_images(
    image_paths: List[str],
    k: int = 20,
    seed: int = 42,
) -> List[str]:
    """Select *k* representative images via KMeans on DINOv2 embeddings.

    Parameters
    ----------
    image_paths:
        Absolute or relative paths to the images to cluster.
    k:
        Number of clusters (and therefore number of returned images).
    seed:
        Random seed for KMeans reproducibility.

    Returns
    -------
    list[str]
        Paths to the *k* images closest to each cluster centroid.
    """
    if not image_paths:
        return []

    # Clamp k to the number of available images
    k = min(k, len(image_paths))

    # 1. Extract embeddings
    embeddings = np.array([_extract_embedding(p) for p in image_paths])

    # 2. KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(embeddings)

    # 3. For each centroid, pick the nearest image
    selected_paths: List[str] = []
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        nearest_idx = int(np.argmin(distances))
        selected_paths.append(image_paths[nearest_idx])

    # Deduplicate while preserving order
    seen = set()
    unique_paths: List[str] = []
    for p in selected_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    return unique_paths

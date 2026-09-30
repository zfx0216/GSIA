"""Per-image L2 constraints in ImageNet-normalized RGB space."""

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BUDGET = 8.0


def project_normalized(images, original, budget):
    """Project NCHW tensors onto each image's normalized L2 ball and valid RGB range."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if images.shape != original.shape or images.ndim != 4:
        raise ValueError("images and original must have matching NCHW shapes")
    delta = images - original
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    # EADEN's Attack wrapper temporarily works in raw [0, 1] space.
    if float(original.min()) >= -1e-5 and float(original.max()) <= 1 + 1e-5:
        delta = images - original
        norms = (delta / std).flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        return torch.clamp(original + delta * (budget * (1 - 1e-6) / norms.clamp_min(1e-12)).clamp(max=1), 0, 1)
    norms = delta.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
    projected = original + delta * (budget * (1 - 1e-6) / norms.clamp_min(1e-12)).clamp(max=1)
    return projected


def project_rgb_uint8(candidate, original, budget):
    """Return a PNG-ready RGB image whose normalized L2 distance is <= budget."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    candidate = np.asarray(candidate)
    original = np.asarray(original)
    if candidate.shape != original.shape or candidate.ndim != 3 or candidate.shape[-1] != 3:
        raise ValueError("candidate and original must be matching HWC RGB images")
    base = np.clip(original, 0, 255).astype(np.uint8)
    target = np.clip(candidate, 0, 255).astype(np.uint8)
    difference = target.astype(np.float64) - base.astype(np.float64)
    std = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(1, 1, 3)
    norm = np.linalg.norm((difference / 255.0 / std).ravel())
    if norm <= budget:
        return target
    # Truncation moves each integer channel toward the original integer value.
    scaled = np.trunc(difference * (budget / norm)).astype(np.int16)
    result = (base.astype(np.int16) + scaled).astype(np.uint8)
    assert np.linalg.norm(((result.astype(np.float64) - base) / 255.0 / std).ravel()) <= budget + 1e-10
    return result


def project_rgb_float(candidate, original, budget):
    """Project HWC RGB values in [0, 255] without losing subpixel updates."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    candidate = np.asarray(candidate, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    if candidate.shape != original.shape or candidate.ndim != 3 or candidate.shape[-1] != 3:
        raise ValueError("candidate and original must be matching HWC RGB images")
    delta = candidate - original
    std = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(1, 1, 3)
    norm = np.linalg.norm((delta / 255.0 / std).ravel())
    return np.clip(original + delta * min(1.0, budget * (1 - 1e-12) / max(norm, 1e-12)), 0, 255)

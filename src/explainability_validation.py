"""Perturbation and map-comparison utilities for dementia attribution."""
import numpy as np
from scipy.stats import spearmanr
from src.research_validation import require

def top_mask(cam, fraction):
    require(cam.ndim == 2 and np.isfinite(cam).all(), 'Invalid map')
    count = int(fraction * cam.size)
    require(0 < count < cam.size, 'Invalid masking fraction')
    mask = np.zeros(cam.size, dtype=bool)
    mask[np.argsort(-cam.ravel(), kind='stable')[:count]] = True
    return mask.reshape(cam.shape)

def shape_controls(mask, seed, limit=5):
    variants = []
    seen = {mask.tobytes()}
    for flipped in [False, True]:
        for rotation in range(4):
            result = np.ascontiguousarray(np.rot90(np.fliplr(mask) if flipped else mask, rotation))
            key = result.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append((f'flip{int(flipped)}_rot{rotation}', result))
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(variants))[:limit]
    return [variants[int(i)] for i in indexes]

def random_controls(mask, seed, count=5):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        m = np.zeros(mask.size, dtype=bool)
        m[rng.choice(mask.size, int(mask.sum()), replace=False)] = True
        result.append(m.reshape(mask.shape))
    return result

def map_similarity(original, randomized):
    require(original.shape == randomized.shape, 'Map shape mismatch')
    if np.ptp(randomized) < 1e-12 or np.ptp(original) < 1e-12:
        return (None, None)
    rho = float(spearmanr(original.ravel(), randomized.ravel()).statistic)
    a = top_mask(original, 0.2)
    b = top_mask(randomized, 0.2)
    return (rho, float((a & b).sum() / (a | b).sum()))

"""
Z-transform the similarity vectors.

Input : (N, 4096) array
Output: same shape, mean 0 / std 1 per window
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config


SIM_VEC_IN   = config.SIMILARITY_VECTOR_NPY
Z_VEC_OUT    = config.ZSCORE_VECTOR_NPY
N_WINDOWS    = config.SIMILARITY_WINDOW_COUNT   # 4096

print("Loading similarity vectors…")
sim_vectors = np.load(SIM_VEC_IN)
print("Loaded shape:", sim_vectors.shape)

if sim_vectors.ndim != 2 or sim_vectors.shape[1] != N_WINDOWS:
    raise ValueError(f"Expected shape (N, {N_WINDOWS}), got {sim_vectors.shape}")

mean = sim_vectors.mean(axis=0)
std  = sim_vectors.std(axis=0)
std[std == 0] = 1   # avoid division by zero

z_vectors = (sim_vectors - mean) / std
np.save(Z_VEC_OUT, z_vectors)
print(f"Saved z-scored vectors to {Z_VEC_OUT}   shape = {z_vectors.shape}")
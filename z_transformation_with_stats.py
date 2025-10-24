# Filename: z_transformation_with_stats.py
"""
Z-transform the (audio) vectors and save the statistics.

Input : (N, 4096) array (from prep_audio_spectrograms.py)
Output: 
    1. Z-transformed array (N, 4096) -> Z_VEC_OUT
    2. Mean array (4096,)           -> MEAN_NPY
    3. Std array (4096,)            -> STD_NPY
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config


SIM_VEC_IN   = config.SIMILARITY_VECTOR_NPY  # Input (from prep_audio...)
Z_VEC_OUT    = config.ZSCORE_VECTOR_NPY      # Output (for prep_ldm)
N_WINDOWS    = config.SIMILARITY_WINDOW_COUNT   # 4096

# New paths for saving the statistics
MEAN_OUT = config.MEAN_NPY
STD_OUT  = config.STD_NPY
MEAN_OUT.parent.mkdir(parents=True, exist_ok=True)


print("Loading (audio) vectors…")
sim_vectors = np.load(SIM_VEC_IN)
print("Loaded shape:", sim_vectors.shape)

if sim_vectors.ndim != 2 or sim_vectors.shape[1] != N_WINDOWS:
    raise ValueError(f"Expected shape (N, {N_WINDOWS}), got {sim_vectors.shape}")

# Calculate mean and standard deviation across the entire dataset
mean = sim_vectors.mean(axis=0)
std  = sim_vectors.std(axis=0)
std[std == 0] = 1   # avoid division by zero

# Z-transformation
z_vectors = (sim_vectors - mean) / std

# --- Saving ---
np.save(Z_VEC_OUT, z_vectors)
print(f"Saved z-scored vectors to {Z_VEC_OUT}   shape = {z_vectors.shape}")

# Save the statistics for live inference
np.save(MEAN_OUT, mean)
np.save(STD_OUT, std)
print(f"Statistics (Mean/Std) saved to: {MEAN_OUT.parent}")

print("\nNext step: Run 'prep_ldm.py'.")

"""Convert flat similarity vectors to multi-subject diffusion-model latents.

Input  : (N, 4096) array (from Z-transform)
Output : (N, 4, 32, 32) tensor saved as *.pt
"""

from pathlib import Path
import numpy as np
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root dir to sys.path
import config


# Paths
flat_path: Path = config.ZSCORE_VECTOR_NPY        # (N, 4096)
latent_out: Path = config.LATENT_TENSOR_PT        # e.g. latents_batch_similarity.pt

# Constants
C, H, W = (4, 32, 32)
EXPECTED = C * H * W                              # 4096

# Load flat vectors
print("Loading SNP similarity (z-score) vectors …")
flat = np.load(flat_path)
print("Loaded array shape:", flat.shape)

if flat.ndim != 2 or flat.shape[1] != EXPECTED:
    raise ValueError(f"Expected shape (N, {EXPECTED}), got {flat.shape}")

# Reshape all subjects
reshaped = flat.reshape(-1, C, H, W)              # shape: (N, 4, 32, 32)
tensor = torch.from_numpy(reshaped).float()

# Save
latent_out.parent.mkdir(parents=True, exist_ok=True)
torch.save(tensor, latent_out)
print(f"Saved full latent tensor → {latent_out}   shape = {tensor.shape}")

"""
Configuration file for the Audio-to-Image LDM pipeline.
Defines all shared constants and file paths.
"""

from pathlib import Path

# --- Base Directories ---
# The root directory of your project
BASE_DIR = Path(__file__).resolve().parent

# A central directory to store all data (audio, vectors, stats)
DATA_DIR = BASE_DIR / "data"

# --- Phase 1: Input Data ---
# Directory where you store your batch of audio files for Phase 1
AUDIO_INPUT_DIR = DATA_DIR / "audio_input_batch"

# --- Phase 1: Intermediate Vectors ---
# Directory to store the intermediate .npy vector files
VECTOR_DIR = DATA_DIR / "vectors"

# Path to the raw (N, 4096) vectors from 'prep_audio_spectrograms.py'
# This file is the input for 'z_transformation_with_stats.py'
# [cite: reshape_snp_vectors_similarity.py, z_transformation.py]
SIMILARITY_VECTOR_NPY = VECTOR_DIR / "audio_spectrogram_vectors.npy"

# Path to the normalized (N, 4096) vectors from 'z_transformation_with_stats.py'
# This file is the input for 'prep_ldm.py'
# [cite: z_transformation.py, prep_ldm.py]
ZSCORE_VECTOR_NPY = VECTOR_DIR / "audio_zscore_vectors.npy"

# --- Phase 1: Normalization Statistics ---
# Directory to store the mean/std statistics calculated in Phase 1
# These are REQUIRED for Phase 2 (live inference)
STATS_DIR = DATA_DIR / "stats"
MEAN_NPY = STATS_DIR / "mean.npy"
STD_NPY  = STATS_DIR / "std.npy"

# --- Phase 1: Final Latent Tensor ---
# Directory to store the final batch tensor
LATENT_DIR = DATA_DIR / "latents"

# Path to the final (N, 4, 32, 32) tensor from 'prep_ldm.py'
# [cite: prep_ldm.py]
LATENT_TENSOR_PT = LATENT_DIR / "audio_latents_batch.pt"

# --- Global Constants ---
# The target dimension for the flattened vector (64*64 = 4096)
# [cite: reshape_snp_vectors_similarity.py, z_transformation.py]
SIMILARITY_WINDOW_COUNT = 4096

# --- Ensure all directories exist ---
def create_dirs():
    for d in [DATA_DIR, AUDIO_INPUT_DIR, VECTOR_DIR, STATS_DIR, LATENT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # If you run this file directly, it will create all the needed folders
    create_dirs()
    print("All necessary data directories have been created:")
    print(f"- Audio Input: {AUDIO_INPUT_DIR}")
    print(f"- Vector Output: {VECTOR_DIR}")
    print(f"- Stats Output: {STATS_DIR}")
    print(f"- Latent Output: {LATENT_DIR}")
else:
    # Create directories when any script imports this config
    create_dirs()

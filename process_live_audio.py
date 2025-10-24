# Filename: process_live_audio.py
"""
Processes a single live audio file into a (1, 4, 32, 32) latent tensor.

This script:
1.  Takes a single audio file path as input.
2.  Converts it to a (64, 64) spectrogram and flattens it to (4096,).
3.  Loads the pre-calculated 'mean.npy' and 'std.npy' statistics.
4.  Applies Z-score normalization to the vector.
5.  Reshapes the vector into a (1, 4, 32, 32) torch tensor.
6.  Saves the tensor as a .pt file, ready for txt2img.py.
"""

import numpy as np
import librosa
import cv2  # Requires opencv-python
import torch
import sys
import argparse
from pathlib import Path

# Add root dir to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# --- Constants from the preparation pipeline ---

# Audio processing constants (from prep_audio_spectrograms.py)
SAMPLE_RATE = 22050
N_MELS = 64
TARGET_H = 64
TARGET_W = 64

# Latent shape constants (from prep_ldm.py)
C, H, W = (4, 32, 32)
EXPECTED_FLAT_DIM = C * H * W  # 4096

def process_audio_file(audio_path, mean_vec, std_vec):
    """Converts one audio file to a (4096,) normalized vector."""
    try:
        # 1. Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # 2. Create Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # 3. Resize to target (64x64)
        S_resized = cv2.resize(S_db, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        
        # 4. Flatten to (4096,)
        vector = S_resized.flatten()
        
        if vector.shape[0] != EXPECTED_FLAT_DIM:
            raise ValueError(f"Flattened vector has wrong shape: {vector.shape}")
            
        # 5. Apply Z-score normalization (using pre-calculated stats)
        z_vector = (vector - mean_vec) / std_vec
        
        return z_vector.astype(np.float32)

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a single audio file into an LDM-ready latent tensor."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input audio file (e.g., my_song.wav)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="live_latent.pt",
        help="Path to save the output .pt tensor."
    )
    args = parser.parse_args()

    # --- Paths from config ---
    MEAN_PATH = config.MEAN_NPY
    STD_PATH = config.STD_NPY
    OUTPUT_PATH = Path(args.output_file)

    # 1. Load normalization statistics
    print(f"Loading normalization stats from {MEAN_PATH.parent}...")
    try:
        mean_vec = np.load(MEAN_PATH)
        std_vec = np.load(STD_PATH)
    except FileNotFoundError:
        print(f"Error: Statistics files not found.")
        print(f"Please run 'z_transformation_with_stats.py' first.")
        sys.exit(1)

    # 2. Process the single audio file
    print(f"Processing audio file: {args.input_file}")
    z_vector = process_audio_file(args.input_file, mean_vec, std_vec)

    if z_vector is not None:
        # 3. Reshape to (1, C, H, W) tensor
        # Reshape to (C, H, W)
        reshaped_vec = z_vector.reshape(C, H, W)
        
        # Convert to tensor
        tensor = torch.from_numpy(reshaped_vec).float()
        
        # Add batch dimension -> (1, 4, 32, 32)
        tensor = tensor.unsqueeze(0) 

        # 4. Save the final tensor
        torch.save(tensor, OUTPUT_PATH)
        print(f"\nSuccess! Latent tensor saved to: {OUTPUT_PATH}")
        print(f"  Shape: {tensor.shape}")
        print("\nYou can now use this file with txt2img.py:")
        print(f"  python txt2img.py --prompt \"Your prompt\" --init_latents {OUTPUT_PATH} --subject_idx 0")

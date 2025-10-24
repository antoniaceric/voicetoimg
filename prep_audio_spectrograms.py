# prep_audio_spectrograms.py
""" Turns audio files into fixed-length vectors (length 4096)

Each audio file (variying length):
1. Loaded as mel-spectrogram
2. Scaled to fixed size (64x64)
3. Transformed into 1D-vector (length 4096)

Input : Audio files (z.B. .wav, .mp3)
Output: (N, 4096) array, saved as config.SIMILARITY_VECTOR_NPY
"""

import numpy as np
import librosa
import cv2  # meed OpenCV: pip install opencv-python
import sys
from pathlib import Path
from tqdm import tqdm

# Fügen Sie das Stammverzeichnis zum sys.path hinzu, um 'config' zu importieren
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# --- Configuration ---
AUDIO_DIR    = config.AUDIO_INPUT_DIR
OUTPUT_NPY   = config.SIMILARITY_VECTOR_NPY

# Audio parameter
SAMPLE_RATE = 22050  # Standard sample rate for Librosa
N_MELS      = 64     # Number of mel bands (later height)

# Goal dimensions
TARGET_H = 64
TARGET_W = 64
EXPECTED_FLAT_DIM = TARGET_H * TARGET_W  # 4096

# Make sure output folder exists
OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)

# --- Processing ---
all_vectors = []
print(f"Look for audio files in: {AUDIO_DIR}")

# Look for common audio formats
audio_files = list(AUDIO_DIR.glob('*.wav')) + \
              list(AUDIO_DIR.glob('*.mp3')) + \
              list(AUDIO_DIR.glob('*.flac'))

if not audio_files:
    raise FileNotFoundError(f"No audio files found in {AUDIO_DIR}.")

print(f"Process {len(audio_files)} audio files...")

for audio_file in tqdm(audio_files):
    try:
        # 1. Load audio (and mono + defined SR resampling)
        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        
        # 2. Mel-spectrogram
        # n_fft and hop_length automatically selected by librosa
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        
        # 3. Transform into decibel (logarithmic scale, better for features)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # 4. Rescaling (64x64)
        # cv2.resize needed (Breite, Höhe)
        S_resized = cv2.resize(S_db, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        
        # 5. Flatten
        S_flat = S_resized.flatten()
        
        # 6. Dimension check
        if S_flat.shape[0] != EXPECTED_FLAT_DIM:
            print(f"Warning: Wrong format {audio_file}. Skipped.")
            continue
            
        all_vectors.append(S_flat)

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

if not all_vectors:
    raise ValueError("No vectors could be created.")

# Aggregate all vectors to one NumPy array
final_array = np.stack(all_vectors, axis=0).astype(np.float32)

# Save
np.save(OUTPUT_NPY, final_array)
print(f"\nSaved: {OUTPUT_NPY}   Form = {final_array.shape}")
print(f"Array ready for 'z_transformation_with_stats.py'.")

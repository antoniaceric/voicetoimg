# Filename: run_live_workflow.py
"""
This master script controls the entire live workflow:
1. Waits for a key press to start recording.
2. Records audio until the key is pressed again.
3. Saves the audio to a temporary file.
4. Calls 'process_live_audio.py' to create a latent tensor.
5. Calls 'txt2img.py' to generate an image from that latent.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess  # The key to calling other scripts
import sys
import argparse
from pathlib import Path

# --- Configuration ---

# This MUST match the sample rate used in your processing scripts
SAMPLE_RATE = 22050 

# Names of the scripts to call
PROCESS_SCRIPT = "process_live_audio.py"
GENERATE_SCRIPT = "txt2img.py"

# Temporary file paths
TEMP_AUDIO_FILE = "temp_live_audio.wav"
LATENT_OUTPUT_FILE = "live_latent.pt" # The output of process_live_audio.py

# --- 1. Audio Recording Function ---

def record_audio(filename):
    """Records audio until Enter is pressed."""
    
    # Get the default input device's channel count
    try:
        channels = sd.query_devices(kind='input')['max_input_channels']
    except Exception as e:
        print(f"Warning: Could not detect input channels, defaulting to 1 (mono). Error: {e}")
        channels = 1
        
    print("\n--- STEP 1: RECORD AUDIO ---")
    input("Press ENTER to start recording...")
    
    # Start recording in a non-blocking way
    print("Recording... Press ENTER to stop.")
    recording = sd.rec(
        int(60 * SAMPLE_RATE), # Max 60 seconds (can be anything large)
        samplerate=SAMPLE_RATE, 
        channels=channels,
        dtype='float32'
    )

    # Wait for the user to press Enter again
    input()
    
    # Stop the recording
    sd.stop()
    
    # Trim silence (or empty parts) from the end
    recording = recording[:sd.get_stream().read_available]
    
    # Save the recording to a file
    sf.write(filename, recording, SAMPLE_RATE)
    print(f"Audio saved to {filename}")
    return filename

# --- 2. Main Workflow ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full text-to-image workflow from live audio.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="An abstract image of a human voice",
        help="The text prompt to guide the image generation."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps. Lower is faster."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Guidance scale for the prompt."
    )
    opt = parser.parse_args()

    # --- Step 1: Record Audio ---
    record_audio(TEMP_AUDIO_FILE)

    # --- Step 2: Process Audio to Latent ---
    print("\n--- STEP 2: PROCESSING AUDIO TO LATENT ---")
    process_command = [
        sys.executable,  # This is 'python'
        PROCESS_SCRIPT,
        "--input_file", TEMP_AUDIO_FILE,
        "--output_file", LATENT_OUTPUT_FILE
    ]
    
    # This runs the command: 
    # python process_live_audio.py --input_file temp_live_audio.wav --output_file live_latent.pt
    subprocess.run(process_command, check=True)
    print(f"Latent file saved to {LATENT_OUTPUT_FILE}")

    # --- Step 3: Generate Image from Latent ---
    print("\n--- STEP 3: GENERATING IMAGE ---")
    generate_command = [
        sys.executable,
        GENERATE_SCRIPT,
        "--prompt", opt.prompt,
        "--init_latents", LATENT_OUTPUT_FILE,
        "--subject_idx", "0",                  # Your latent file has only 1 subject at index 0
        "--ddim_steps", str(opt.steps),       
        "--scale", str(opt.scale),            
        "--n_samples", "1",                    # Only generate 1 sample
        "--n_iter", "1"                       
    ]
    
    # This runs the command:
    # python txt2img.py --prompt "..." --init_latents live_latent.pt --subject_idx 0 ...
    subprocess.run(generate_command, check=True)

    print("\n--- WORKFLOW COMPLETE ---")

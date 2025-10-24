# voicetoimg #workinprogress

# Audio-to-Image LDM Pipeline

This project converts audio recordings (e.g., a person's voice) into latent vectors, which are then used as the starting point (initial latents, or $x_T$) for a Latent Diffusion Model (LDM) to generate an image.

The core idea is to transform any audio clip into a fixed-size `(4096,)` vector by:
1.  Creating a Mel spectrogram from the audio.
2.  Resizing that spectrogram to a fixed `64x64` image.
3.  Flattening the `64x64` image into a `4096`-dimensional vector.

This vector is then Z-score normalized and reshaped into the `(1, 4, 32, 32)` format required by the LDM.

The workflow is divided into two phases:
* **Phase 1 (Offline Preparation):** A one-time process to analyze a large batch of audio files. This "calibrates" the system by calculating the mean and standard deviation of all audio features, which is essential for normalization.
* **Phase 2 (Live Inference):** The live-demo script that records a new voice, processes it using the statistics from Phase 1, and immediately generates an image.

---

## ⚙️ 1. Setup and Installation

### 1.1 Python Libraries
Install the required Python packages.

```bash
pip install numpy torch
pip install librosa
pip install opencv-python
pip install sounddevice soundfile
pip install omegaconf einops tqdm Pillow
```

### Phase 1: Offline Batch Preparation (Do This First)

The goal of this phase is to generate the `mean.npy` and `std.npy` files. These files are **required** for Phase 2 to work.

**Step 1: Collect Audio Data**
* Gather a large set of audio files (e.g., 100+ `.wav`, `.mp3` files) that are representative of what you'll use live.
* Place them all inside the directory you defined as `AUDIO_INPUT_DIR` in your `config.py`.

**Step 2: Create Spectrogram Vectors**
* Run the first script to process all audio files into a single `.npy` file.
    ```bash
    python prep_audio_spectrograms.py
    ```
* **Output:** This creates the `(N, 4096)` vector file at `config.SIMILARITY_VECTOR_NPY`.

**Step 3: Calculate Normalization Stats**
* Run the modified Z-transformation script. This will normalize the vectors from Step 2 and save the statistics.
    ```bash
    python z_transformation_with_stats.py
    ```
* **Output:**
    1.  The normalized `(N, 4096)` vectors at `config.ZSCORE_VECTOR_NPY`.
    2.  The crucial `mean.npy` and `std.npy` files.

**Step 4: (Optional) Create Batch Latent Tensor**
* If you want to create a single large `.pt` file containing all your "training" latents (e.g., for testing `txt2img.py` with the `--subject_idx` flag), run `prep_ldm.py`.
    ```bash
    python prep_ldm.py
    ```
* **Output:** The final `(N, 4, 32, 32)` tensor at `config.LATENT_TENSOR_PT`.

**You have now completed the one-time setup and are ready for live inference.**

---

### Phase 2: Live Inference Workflow (The Live Demo)

This is the main script to run your live demo. It automatically handles recording, processing, and generation in sequence.

**Prerequisite:** You *must* have completed Phase 1, as this script depends on the `mean.npy` and `std.npy` files.

**Step 1: Run the Master Script**
* From your terminal, run `run_live_workflow.py`. You can pass arguments like `--prompt` to control the image generation.
    ```bash
    python run_live_workflow.py --prompt "A psychedelic visualization of a voice" --steps 75
    ```

**Step 2: Follow the Prompts**
* The script will first ask you to **"Press ENTER to start recording..."**.
* When you are ready, press Enter.
* It will say **"Recording... Press ENTER to stop."**. Sing or speak into your microphone.
* When you are finished, press Enter again.

**Step 3: Wait for Generation**
* The script will now run the full pipeline automatically:
    1.  Your recording is saved as `temp_live_audio.wav`.
    2.  `process_live_audio.py` is called to convert the audio into a `(1, 4, 32, 32)` tensor named `live_latent.pt`, using the `mean.npy` and `std.npy` stats.
    3.  `txt2img.py` is called, using `live_latent.pt` as its `--init_latents`.

**Step 4: View Your Image**
* Once the `txt2img.py` script finishes, your generated image will be saved in the `outputs/txt2img-samples/samples/` directory (or wherever `txt2img.py` is configured to save samples).

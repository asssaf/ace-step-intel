# ACE-Step 1.5 Docker (Intel XPU Optimized)

A Dockerized environment for running [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) music foundation models, built on top of **Intel® Extension for PyTorch** (`intel/pytorch:latest`) for hardware-accelerated inference on Intel GPUs (Arc™ GPUs, Data Center GPU Flex/Max, and integrated Intel graphics).

This container includes a dedicated patch (`01-audio_utils.patch`) that replaces `torchaudio.save` with `soundfile.write` to avoid common `torchaudio` backend and codec incompatibilities when saving audio on Intel XPU / SYCL platforms.

---

## Features

- **Intel PyTorch Optimization**: Built on `intel/pytorch:latest` with native Intel XPU (SYCL / Level Zero) acceleration for fast DiT generation and CPU offloading.
- **Robust Audio Export via SoundFile**: Replaces `torchaudio.save` calls in ACE-Step's `AudioSaver` pipeline with `soundfile.write`, ensuring reliable WAV export and seamless conversion via FFmpeg without runtime backend errors.
- **Preconfigured Performance Flags**:
  - `SYCL_CACHE_PERSISTENT=1` (cached kernels for faster subsequent runs)
  - `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` (lower dispatch latency)
  - `PYTORCH_DEVICE=xpu`
  - `TORCH_COMPILE_BACKEND=eager`
- **Containerized Gradio Web UI**: Exposes ACE-Step's interactive Gradio interface (port 7860) and API endpoints (port 8001).
- **Persistent Storage**: Mount checkpoints and custom `.env` settings directly from the host.

---

## Repository Structure

```text
.
├── 01-audio_utils.patch    # Patch replacing torchaudio.save with soundfile
├── docker/
│   └── Dockerfile          # Container build definition based on intel/pytorch:latest
├── scripts/
│   └── run.sh              # Helper script to launch the container with GPU device passthrough
├── .env                    # Hardware, device, and network configuration
└── README.md
```

---

## Why `soundfile` instead of `torchaudio`?

In many containerized environments running Intel-optimized PyTorch on Linux, `torchaudio.save()` can encounter issues with underlying C++ dynamic library bindings (such as `libsox` or TorchAudio FFmpeg I/O backends), leading to failed audio export or missing backend warnings.

The included patch [`01-audio_utils.patch`](./01-audio_utils.patch) modifies `acestep/audio_utils.py` so that generated audio tensors are transposed and converted to NumPy arrays before being written directly with `soundfile`:

```python
import soundfile as sf
audio_np = tensor_to_save.transpose(0, 1).numpy()
sf.write(str(temp_wav_path), audio_np, int(target_sample_rate), format="WAV")
```

This guarantees fast, reliable audio disk writes that downstream FFmpeg commands can encode into requested formats without external torchaudio C++ backend dependencies.

---

## Hardware & System Requirements

- **Operating System**: Linux with Intel Compute Runtime installed (`intel-opencl-icd`, `intel-level-zero-gpu`).
- **GPU**:
  - Intel Arc™ Discrete GPUs (A770, A750, A580, A380, etc.)
  - Intel Data Center GPU Flex / Max Series
  - Intel Core / Core Ultra Integrated Graphics (iGPU)
- **Host Permissions**: Access to `/dev/dri` (user in `render` and `video` groups).
- **Docker**: Docker Engine with device passthrough support.

---

## Configuration (`.env`)

Hardware and server settings can be adjusted in `.env`:

```env
# --- Intel XPU & Hardware Optimizations ---
DEVICE=xpu
CPU_OFFLOAD=cpu
TORCH_DTYPE=bfloat16
SYCL_DEVICE_FILTER=level_zero

# --- Network & Server ---
PORT=7860
SERVER_NAME=127.0.0.1
```

Additional ACE-Step variables (such as `ACESTEP_CONFIG_PATH` or `ACESTEP_LM_MODEL_PATH`) can also be added here and are mounted into the container.

---

## Building the Docker Image

Build the Docker image from the repository root:

```bash
docker build -f docker/Dockerfile -t asssaf/ace-step-intel-xpu:latest .
```

The build process will:
1. Pull the official `intel/pytorch:latest` base image.
2. Install `ffmpeg` and `patch`.
3. Clone ACE-Step 1.5 (`v0.1.8`).
4. Install `requirements-xpu.txt`.
5. Apply `01-audio_utils.patch` to use `soundfile`.
6. Configure default SYCL / XPU environment variables.

---

## Running the Container

### Option 1: Using the Provided Launch Script

The repository includes a ready-to-use launch script in [`scripts/run.sh`](./scripts/run.sh):

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

You can override the image tag by setting the `IMAGE` environment variable:

```bash
IMAGE="my-custom-tag:latest" ./scripts/run.sh
```

### Option 2: Running with `docker run` Directly

```bash
docker run --rm -it --name ace-step \
    --device /dev/dri:/dev/dri \
    --group-add $(getent group render | cut -d: -f3) \
    --group-add $(getent group video | cut -d: -f3) \
    -v /var/cache/models/acestep/checkpoints:/ACE-Step-1.5/checkpoints \
    -v "${PWD}/.env:/ACE-Step-1.5/.env:ro" \
    -p 7860:7860 \
    --net host \
    --ipc=host \
    asssaf/ace-step-intel-xpu:latest
```

#### Key Flags Explained:
- `--device /dev/dri:/dev/dri`: Passes the Intel GPU Direct Rendering Infrastructure devices to the container.
- `--group-add`: Grants container access to the host GPU render and video groups (typically GIDs `27` and `28` or as configured on your host).
- `-v /var/cache/models/acestep/checkpoints:/ACE-Step-1.5/checkpoints`: Persists downloaded model weights on host storage.
- `-v "${PWD}/.env:/ACE-Step-1.5/.env:ro"`: Injects local environment variable configurations.
- `--net host` / `-p 7860:7860`: Exposes the Gradio web UI.
- `--ipc=host`: Prevents PyTorch shared memory issues during tensor operations.

---

## Usage

Once started, the container will launch the ACE-Step pipeline:

1. Open your browser and navigate to:
   ```text
   http://localhost:7860
   ```
2. In the Gradio UI:
   - Select your desired DiT checkpoint (e.g., `acestep-v15-turbo`) and Language Model (e.g., `acestep-5Hz-lm-4B` or smaller variants if VRAM is constrained).
   - Enter your prompt, desired style, lyrics, and audio duration.
   - Click **Generate** to create music on your Intel GPU.

---

## License & Credits

- **ACE-Step 1.5**: Developed by the ACE-Step team ([GitHub Repository](https://github.com/ace-step/ACE-Step-1.5)).
- **Base PyTorch Image**: Provided by Intel Corporation ([Intel PyTorch Container](https://hub.docker.com/r/intel/pytorch)).

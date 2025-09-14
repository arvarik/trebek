# Jeopardy! Transcription Service

An automated pipeline to monitor a directory for new video files, extract audio, and generate high-accuracy transcriptions using a local speech-to-text model.

## Features

-   **Automated Monitoring**: Watches a specified folder for new video recordings.
-   **High-Accuracy Transcription**: Uses OpenAI's Whisper model (via `whisper.cpp`).
-   **Local First**: All processing is done locally. No cloud services or APIs are needed.
-   **Apple Silicon + Metal**: Uses `whisper.cpp` with Apple's Metal for GPU acceleration on M-series Macs.
-   **Persistent Service**: Runs as a background `launchd` service on macOS, ensuring it's always on.

## Run a single file (standalone test)

Use the dedicated script with logging and progress:

```bash
source venv/bin/activate
python src/single_run.py
```

Customize paths at the top of `src/single_run.py`:
- `video_path` – your input video (e.g., the Jeopardy episode)
- `WhisperCppConfig.binary_path` – your whisper.cpp CLI binary
- `WhisperCppConfig.model_path` – the model to use (default: medium.en)

Notes:
- Transcripts are saved under `./transcripts` with a timestamp including microseconds, so repeated runs will not overwrite existing files.

---

## Deployment Guide (for Mac mini)

Follow these steps to deploy the service on a target macOS machine.

### 1. Prerequisites

Ensure you have [Homebrew](https://brew.sh/) installed. Then, install FFmpeg:

```bash
brew install ffmpeg
```

### 2. Clone the Repository

Clone this project from your GitHub repository to a suitable location (e.g., ~/Projects).

```bash
git clone <your-repository-url>
cd jeopardy-transcriber
```

### 3. Setup the Python Environment

Create a virtual environment and install the required dependencies.

```bash
# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

### 4. Install whisper.cpp (Metal on macOS)

Build `whisper.cpp` with Metal support and download a model.

```bash
# Prereqs
brew install cmake

# Get the source (choose a permanent location)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp

# Build with Metal support (CMake)
cmake -B build -DGGML_METAL=1
cmake --build build -j --config Release

# Optional: the build may produce a binary named `main` (make) or `bin/whisper-cli` (cmake)
# Adjust paths in config accordingly.

# Download a model
cd models
# Recommended default for Mac mini: medium.en
bash ./download-ggml-model.sh medium.en
# This will create: /Users/arvarik/Documents/github/whisper.cpp/models/ggml-medium.en.bin

# Optional: quantized large for lower VRAM (accuracy closer to large)
# Examples (availability may vary): ggml-large-v3-q5_0.bin, ggml-large-v3-q4_0.bin
# You can download quantized variants from community sources or quantize yourself; update model_path accordingly.
```

### 5. Configure Paths

This is the most important step. Open the `src/config.py` file and edit the paths to match your Mac mini's setup. You must use absolute paths.

For example:

```python
# src/config.py

# ⬇️ CHANGE THIS to the absolute path of your recordings folder on the external SSD
RECORDINGS_DIR = "/Volumes/MediaSSD/JeopardyRecordings"

# ⬇️ CHANGE THIS to the absolute path where you want transcripts saved
TRANSCRIPTS_DIR = "/Users/your_username/Projects/jeopardy-transcriber/transcripts"

# ⬇️ CHANGE THIS to the absolute path for the logs folder
LOGS_DIR = "/Users/your_username/Projects/jeopardy-transcriber/logs"

# ⬇️ Configure whisper.cpp locations and tuning
from config import WhisperCppConfig, WHISPER_CPP_MODEL_DIR, WHISPER_CPP_MODEL_NAME

# Example (you can also pass this at runtime as shown in the single-run snippet):
WHISPER_CPP = WhisperCppConfig(
    binary_path="/Users/arvarik/Documents/github/whisper.cpp/build/bin/whisper-cli",
    model_path=os.path.join(WHISPER_CPP_MODEL_DIR, WHISPER_CPP_MODEL_NAME),
    threads=8,
    language="en",
)
```

### 6. Configure and Install the Background Service

The `com.user.jeopardytranscriber.plist` file tells macOS how to run our script. You must edit it to include the absolute paths for your machine.

A. Edit the .plist file:
Open `com.user.jeopardytranscriber.plist` and replace the placeholder paths with your absolute paths.

- Replace `/Users/your_username/Projects/jeopardy-transcriber/venv/bin/python` with the path to the python executable in your venv.
- Replace `/Users/your_username/Projects/jeopardy-transcriber/src/main.py` with the path to your main script.

B. Install and run the service:
Once saved, copy the file to the macOS LaunchAgents directory and load it.

```bash
# Copy the file
cp com.user.jeopardytranscriber.plist ~/Library/LaunchAgents/

# Load and start the service
launchctl load ~/Library/LaunchAgents/com.user.jeopardytranscriber.plist
launchctl start com.user.jeopardytranscriber
```

### 6. Verify the Service

You can check if the service is running and view its logs.

```bash
# Check if the service is loaded (it may show a PID or '-')
launchctl list | grep jeopardytranscriber

# Tail the log file to see live activity
tail -f <path-to-your-logs-dir>/transcriber.log
```

The service is now running. It will start automatically on boot and will continuously monitor your recordings directory.

---


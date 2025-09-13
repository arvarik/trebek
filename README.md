# Jeopardy! Transcription Service

An automated pipeline to monitor a directory for new video files, extract audio, and generate high-accuracy transcriptions using a local speech-to-text model optimized for Apple Silicon.

## Features

-   **Automated Monitoring**: Watches a specified folder for new video recordings.
-   **High-Accuracy Transcription**: Uses OpenAI's Whisper model (`large-v3`) for best-in-class accuracy.
-   **Local First**: All processing is done locally. No cloud services or APIs are needed.
-   **Apple Silicon Optimized**: Leverages the M-series Neural Engine via Metal Performance Shaders (MPS) for fast transcription.
-   **Persistent Service**: Runs as a background `launchd` service on macOS, ensuring it's always on.

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

# Install PyTorch with Metal (MPS) support
pip3 install torch torchvision torchaudio

# Install all other dependencies
pip3 install -r requirements.txt
```

### 4. Configure Paths

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
```

### 5. Configure and Install the Background Service

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


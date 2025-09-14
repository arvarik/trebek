from __future__ import annotations

from typing import Dict

import json
import os
import subprocess
import tempfile
from typing import Any
import wave

from config import WhisperCppConfig


class Transcriber:
    """Whisper.cpp-based transcriber (Metal-accelerated on Apple Silicon).

    This wrapper calls the whisper.cpp CLI and parses its JSON output to a
    schema compatible with the rest of the application.
    """

    def __init__(self, cpp_config: WhisperCppConfig | None = None) -> None:
        self._cfg = cpp_config or WhisperCppConfig()
        if not os.path.exists(self._cfg.binary_path):
            raise FileNotFoundError(
                f"whisper.cpp binary not found: {self._cfg.binary_path}. See README for build steps."
            )
        if not os.path.exists(self._cfg.model_path):
            raise FileNotFoundError(
                f"whisper.cpp model not found: {self._cfg.model_path}. See README to download models."
            )

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe the given WAV file via whisper.cpp CLI.

        Returns a JSON-serializable dictionary with segments and text compatible
        with the existing file manager output.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # whisper.cpp writes output files when JSON is enabled. We capture to a
        # temp directory to avoid clutter and then parse the JSON.
        with tempfile.TemporaryDirectory(prefix="whcpp_") as tmpdir:
            # Build command
            json_flag = "-ojf"  # full JSON contains richer timing/metadata
            cmd = [
                self._cfg.binary_path,
                "-m",
                self._cfg.model_path,
                "-l",
                self._cfg.language,
                json_flag,
                "-otxt",
                "-of",
                os.path.join(tmpdir, "out"),
                "-t",
                str(self._cfg.threads),
                # Input file as positional arg (more compatible across builds)
                audio_path,
            ]

            # Stream progress if requested, otherwise capture
            if self._cfg.print_progress:
                completed = subprocess.run(cmd, check=False)
                captured_stdout = ""
                captured_stderr = ""
            else:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                captured_stdout = completed.stdout
                captured_stderr = completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(
                    f"whisper.cpp failed (code {completed.returncode})"
                )

            # whisper.cpp writes out files like out.json; parse it
            json_path = os.path.join(tmpdir, "out.json")
            if not os.path.exists(json_path):
                # Inspect produced files for diagnostics
                produced = ", ".join(sorted(os.listdir(tmpdir)))
                raise RuntimeError(
                    "whisper.cpp did not produce JSON output as expected. "
                    f"Flags used: {' '.join(cmd[1:])}. "
                    f"Produced files in temp dir: [{produced}]"
                )

            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Optionally capture plain text output for fallback
            txt_text = ""
            txt_path_tmp = os.path.join(tmpdir, "out.txt")
            if os.path.exists(txt_path_tmp):
                try:
                    with open(txt_path_tmp, "r", encoding="utf-8") as tf:
                        txt_text = tf.read().strip()
                except Exception:
                    txt_text = ""

            # No SRT parsing: we rely on JSON or plain text output

        # Prefer top-level text from JSON; otherwise use plain text output
        result_text = ""
        if isinstance(raw, dict) and isinstance(raw.get("text"), str):
            result_text = raw.get("text", "").strip()
        if not result_text:
            try:
                result_text = locals().get("txt_text", "").strip()
            except Exception:
                result_text = ""
        # Compute duration fallback using WAV metadata if still missing
        duration_value: float = (raw.get("duration") if isinstance(raw, dict) and isinstance(raw.get("duration"), (int, float)) else 0.0)
        if not duration_value:
            try:
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        duration_value = float(frames) / float(rate)
            except Exception:
                pass

        result: Dict[str, Any] = {
            "language": raw.get("language", self._cfg.language) if isinstance(raw, dict) else self._cfg.language,
            "duration": duration_value,
            "text": result_text,
            "model": "whisper.cpp",
            "model_size": os.path.basename(self._cfg.model_path),
        }
        return result



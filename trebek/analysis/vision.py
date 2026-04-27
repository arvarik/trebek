"""
Gemini Vision API client and podium illumination timestamp extraction.

Uses the Gemini Files API to upload video segments and analyze them
for precise visual cues like contestant podium indicator lights.
"""

import structlog
from typing import Optional

logger = structlog.get_logger()

_vision_client: Optional["VisionClient"] = None


def _get_vision_client() -> "VisionClient":
    """Lazy-initializes the Vision client on first use, not at import time."""
    global _vision_client
    if _vision_client is None:
        _vision_client = VisionClient()
    return _vision_client


class VisionClient:
    """Interface for Gemini 3.1 Pro via the Files API."""

    def __init__(self) -> None:
        import os
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required but not set. "
                "Set it in your .env file or export it before running the pipeline."
            )
        self.client = genai.Client(api_key=api_key)

    async def analyze_video(self, video_path: str, prompt: str) -> float:
        """
        Uploads the video via the Gemini Files API, then analyzes it with the
        specified prompt. Returns a float timestamp extracted from the response.

        NOTE: The Gemini Files API requires uploading the video first, then passing
        the file reference to the model. For very large files, consider chunking or
        pre-extracting the relevant video segment with FFmpeg before upload.
        """
        from google.genai import types

        # Upload the video file via the Files API
        # This is a synchronous upload; for production, consider async wrapper
        uploaded_file = self.client.files.upload(file=video_path)
        logger.info("Uploaded video for Vision analysis", video_path=video_path, file_name=uploaded_file.name)

        config = types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.0,
        )
        response = await self.client.aio.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=uploaded_file,
            config=config,
        )
        return float(str(response.text).strip())


async def extract_podium_illumination_timestamp(video_path: str, host_finish_timestamp: float) -> float:
    """
    Stage 6: Analyzes video frames immediately following the host's completion
    to determine the exact frame the podium lockout system disengages.
    """
    system_prompt = (
        f"Analyze the video starting precisely at {host_finish_timestamp} seconds. "
        "Find the exact timestamp (in seconds) where the contestant podium indicator "
        "lights illuminate, signaling the lockout system has disengaged. Return ONLY a float."
    )
    try:
        client = _get_vision_client()
        response = await client.analyze_video(video_path, prompt=system_prompt)
        return float(response)
    except Exception as e:
        logger.error("Vision model failed", error=str(e))
        return host_finish_timestamp + 0.1  # Fallback estimation

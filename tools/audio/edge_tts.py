"""EdgeTTS free text-to-speech provider tool using Microsoft Edge TTS service.

High-quality neural speech synthesis powered by Microsoft Edge's online TTS
service (via the ``edge-tts`` library). Requires no API key, zero cost, and
provides rich multilingual neural voices (including natural Vietnamese voices
such as ``vi-VN-HoaiMyNeural`` and ``vi-VN-NamMinhNeural``, and 40+ other languages).

Preferred free online TTS provider in OpenMontage, prioritizing over local piper_tts.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


def _srt_to_vtt(srt_text: str) -> str:
    """Convert SubMaker SRT string to WebVTT format."""
    lines = srt_text.strip().splitlines()
    if not lines:
        return "WEBVTT\n"

    vtt_lines = ["WEBVTT", ""]
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}),(\d{3})")
    for line in lines:
        vtt_lines.append(time_pattern.sub(r"\1.\2", line))
    return "\n".join(vtt_lines) + "\n"


class EdgeTTS(BaseTool):
    name = "edge_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "edge_tts"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = ["python:edge-tts"]
    install_instructions = (
        "Install edge-tts:\n"
        "  pip install edge-tts\n"
        "No API key is required. Free Microsoft Edge Neural voices."
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "rate_control",
        "pitch_control",
        "volume_control",
        "subtitle_generation",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "free high-quality neural narration",
        "multilingual spoken delivery without API key",
        "Vietnamese and global neural voice synthesis",
        "default free online TTS fallback",
    ]
    not_good_for = [
        "fully offline production (use piper_tts)",
        "voice cloning or custom cloned voices (use elevenlabs_tts)",
    ]

    RECOMMENDED_VOICES: dict[str, str] = {
        # Vietnamese (Natural Neural)
        "hoaimy": "vi-VN-HoaiMyNeural",
        "namminh": "vi-VN-NamMinhNeural",
        # English
        "jenny": "en-US-JennyNeural",
        "guy": "en-US-GuyNeural",
        "aria": "en-US-AriaNeural",
        "andrew": "en-US-AndrewMultilingualNeural",
        "ava": "en-US-AvaMultilingualNeural",
        "brian": "en-US-BrianMultilingualNeural",
        "christopher": "en-US-ChristopherNeural",
        "eric": "en-US-EricNeural",
        "emma": "en-US-EmmaMultilingualNeural",
        # Chinese
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",
        "yunxi": "zh-CN-YunxiNeural",
        "yunjian": "zh-CN-YunjianNeural",
        # Japanese
        "nanami": "ja-JP-NanamiNeural",
        "keita": "ja-JP-KeitaNeural",
        # French / German / Spanish
        "denise": "fr-FR-DeniseNeural",
        "henri": "fr-FR-HenriNeural",
        "katja": "de-DE-KatjaNeural",
        "conrad": "de-DE-ConradNeural",
        "elena": "es-ES-ElenaNeural",
        "alvaro": "es-ES-AlvaroNeural",
    }

    DEFAULT_VOICE = "vi-VN-HoaiMyNeural"

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to convert to speech"},
            "voice": {
                "type": "string",
                "description": "Edge TTS voice short name (e.g. vi-VN-HoaiMyNeural, en-US-JennyNeural) or alias.",
            },
            "voice_id": {
                "type": "string",
                "description": "Alias for voice parameter.",
            },
            "rate": {
                "type": "string",
                "description": "Speech rate adjustment, e.g. '+10%', '-20%', or '+0%'.",
            },
            "speed": {
                "type": "number",
                "description": "Numeric speaking speed multiplier (1.0 = normal, 1.2 = +20%).",
            },
            "speaking_rate": {
                "type": "number",
                "description": "Google-style speaking rate multiplier.",
            },
            "pitch": {
                "type": "string",
                "description": "Pitch adjustment, e.g. '+5Hz', '-5Hz', or '+0Hz'.",
            },
            "volume": {
                "type": "string",
                "description": "Volume adjustment, e.g. '+0%', '-10%', or '+10%'.",
            },
            "output_path": {"type": "string", "description": "Destination file path for audio."},
            "subtitle_path": {
                "type": "string",
                "description": "Optional destination file path for generated subtitles (.vtt or .srt).",
            },
            "output_format": {
                "type": "string",
                "default": "mp3",
                "enum": ["mp3", "wav"],
                "description": "Audio container format.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["timeout", "connection_error"])
    idempotency_key_fields = [
        "text",
        "voice",
        "voice_id",
        "rate",
        "speed",
        "speaking_rate",
        "pitch",
        "volume",
    ]
    side_effects = ["writes audio file to output_path", "calls Edge TTS WebSocket API"]
    user_visible_verification = ["Listen to generated audio for clear neural speech quality"]

    def get_status(self) -> ToolStatus:
        try:
            import edge_tts  # noqa: F401
            return ToolStatus.AVAILABLE
        except ImportError:
            if shutil.which("edge-tts"):
                return ToolStatus.AVAILABLE
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    @classmethod
    def resolve_voice(cls, voice_input: str | None) -> str:
        """Resolve alias or short name to full Microsoft Neural voice name."""
        if not voice_input:
            return cls.DEFAULT_VOICE

        cleaned = str(voice_input).strip()
        lower = cleaned.lower()
        if lower in cls.RECOMMENDED_VOICES:
            return cls.RECOMMENDED_VOICES[lower]

        # Case-insensitive match on recommended voice full names
        for v in cls.RECOMMENDED_VOICES.values():
            if lower == v.lower():
                return v

        return cleaned

    @classmethod
    def format_rate(cls, inputs: dict[str, Any]) -> str:
        """Format rate string from rate, speed, or speaking_rate."""
        raw_rate = inputs.get("rate")
        if raw_rate is not None:
            if isinstance(raw_rate, (int, float)):
                val = round(float(raw_rate))
                return f"{val:+d}%"
            rate_val = str(raw_rate).strip()
            if rate_val:
                if not rate_val.startswith(("+", "-")):
                    rate_val = f"+{rate_val}"
                if not rate_val.endswith("%"):
                    rate_val = f"{rate_val}%"
                return rate_val

        speed = inputs.get("speed", inputs.get("speaking_rate"))
        if speed is not None:
            try:
                percent = round((float(speed) - 1.0) * 100)
                return f"{percent:+d}%"
            except (ValueError, TypeError):
                pass
        return "+0%"

    @classmethod
    def format_pitch(cls, inputs: dict[str, Any]) -> str:
        """Format pitch string, e.g. '+0Hz' or '+5Hz'."""
        pitch_val = inputs.get("pitch")
        if pitch_val is None:
            return "+0Hz"

        if isinstance(pitch_val, (int, float)):
            val = round(float(pitch_val))
            return f"{val:+d}Hz"

        pitch_str = str(pitch_val).strip()
        if pitch_str.endswith("%"):
            pitch_str = pitch_str[:-1].strip()
        elif pitch_str.lower().endswith("hz"):
            pitch_str = pitch_str[:-2].strip()

        if not pitch_str or pitch_str in ("+", "-"):
            return "+0Hz"

        if not pitch_str.startswith(("+", "-")):
            pitch_str = f"+{pitch_str}"
        return f"{pitch_str}Hz"

    @classmethod
    def format_volume(cls, inputs: dict[str, Any]) -> str:
        """Format volume string, e.g. '+0%'."""
        vol_val = inputs.get("volume")
        if vol_val is None:
            return "+0%"

        if isinstance(vol_val, (int, float)):
            val = round(float(vol_val))
            return f"{val:+d}%"

        vol_str = str(vol_val).strip()
        if not vol_str or vol_str in ("+", "-"):
            return "+0%"

        if not vol_str.startswith(("+", "-")):
            vol_str = f"+{vol_str}"
        if not vol_str.endswith("%"):
            vol_str = f"{vol_str}%"
        return vol_str

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error="Edge TTS not available. " + self.install_instructions,
            )

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"EdgeTTS generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = 0.0
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        text = str(inputs.get("text", "")).strip()
        if not text:
            return ToolResult(success=False, error="Input 'text' cannot be empty.")

        voice_input = inputs.get("voice") or inputs.get("voice_id")
        voice = self.resolve_voice(voice_input)
        rate = self.format_rate(inputs)
        pitch = self.format_pitch(inputs)
        volume = self.format_volume(inputs)

        output_format = str(inputs.get("output_format", "mp3")).lower()
        ext = "mp3" if "mp3" in output_format else "wav"
        output_path = Path(inputs.get("output_path", f"edge_tts_output.{ext}")).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sub_path_input = inputs.get("subtitle_path")
        sub_path = Path(sub_path_input).resolve() if sub_path_input else None

        target_is_wav = ext == "wav"
        intermediate_audio_path = (
            output_path.with_suffix(".tmp.mp3") if target_is_wav else output_path
        )

        # 1. Try python edge_tts module first
        success = False
        last_error = ""
        try:
            import edge_tts

            async def _run_edge_tts() -> None:
                boundary = "WordBoundary" if sub_path else "SentenceBoundary"
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                    boundary=boundary,
                )
                if sub_path:
                    sub_path.parent.mkdir(parents=True, exist_ok=True)
                    submaker = edge_tts.SubMaker()
                    with open(intermediate_audio_path, "wb") as audio_f:
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                audio_f.write(chunk["data"])
                            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                                submaker.feed(chunk)
                    srt_content = submaker.get_srt()
                    ext_sub = sub_path.suffix.lower()
                    content = srt_content if ext_sub == ".srt" else _srt_to_vtt(srt_content)
                    sub_path.write_text(content, encoding="utf-8")
                else:
                    await communicate.save(str(intermediate_audio_path))

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(_run_edge_tts()))
                    future.result(timeout=120)
            else:
                asyncio.run(_run_edge_tts())

            success = (
                intermediate_audio_path.exists()
                and intermediate_audio_path.stat().st_size > 0
            )
        except Exception as py_err:
            last_error = str(py_err)

        # 2. Fallback to CLI if Python module call failed
        if not success and shutil.which("edge-tts"):
            temp_sub = (
                sub_path.with_suffix(".tmp.srt")
                if (sub_path and sub_path.suffix.lower() != ".srt")
                else sub_path
            )
            cmd = [
                "edge-tts",
                "--voice", voice,
                "--text", text,
                "--write-media", str(intermediate_audio_path),
            ]
            if temp_sub:
                cmd.extend(["--write-subtitles", str(temp_sub)])
            if rate != "+0%":
                cmd.extend(["--rate", rate])
            if pitch != "+0Hz":
                cmd.extend(["--pitch", pitch])
            if volume != "+0%":
                cmd.extend(["--volume", volume])

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if (
                proc.returncode == 0
                and intermediate_audio_path.exists()
                and intermediate_audio_path.stat().st_size > 0
            ):
                success = True
                if sub_path and sub_path.suffix.lower() == ".vtt" and temp_sub and temp_sub.exists():
                    srt_content = temp_sub.read_text(encoding="utf-8")
                    sub_path.write_text(_srt_to_vtt(srt_content), encoding="utf-8")
                    if temp_sub != sub_path and temp_sub.exists():
                        try:
                            temp_sub.unlink()
                        except OSError:
                            pass
            else:
                last_error = proc.stderr or last_error or "CLI generation failed"

        if not success:
            return ToolResult(
                success=False,
                error=f"EdgeTTS failed to generate audio: {last_error}",
            )

        # 3. Transcode to PCM WAV if output format is wav
        if target_is_wav and intermediate_audio_path.exists():
            ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
            transcode_cmd = [
                ffmpeg_cmd,
                "-y",
                "-i", str(intermediate_audio_path),
                "-vn",
                "-acodec", "pcm_s16le",
                str(output_path),
            ]
            transcode_res = subprocess.run(
                transcode_cmd, capture_output=True, text=True, timeout=60
            )
            if (
                transcode_res.returncode == 0
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                if intermediate_audio_path.exists() and intermediate_audio_path != output_path:
                    try:
                        intermediate_audio_path.unlink()
                    except OSError:
                        pass
            else:
                # Fallback: rename intermediate MP3 to destination path if transcoding fails
                if intermediate_audio_path.exists() and intermediate_audio_path != output_path:
                    intermediate_audio_path.replace(output_path)

        artifacts = [str(output_path)]
        data: dict[str, Any] = {
            "provider": self.provider,
            "voice": voice,
            "voice_requested": voice_input,
            "rate": rate,
            "pitch": pitch,
            "volume": volume,
            "text_length": len(text),
            "output": str(output_path),
            "format": ext,
        }
        if sub_path and sub_path.is_file() and sub_path.stat().st_size > 0:
            artifacts.append(str(sub_path))
            data["subtitle_path"] = str(sub_path)

        return ToolResult(
            success=True,
            data=data,
            artifacts=artifacts,
            model=voice,
        )

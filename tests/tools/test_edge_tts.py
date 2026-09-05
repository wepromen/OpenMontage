"""Tests for the EdgeTTS provider tool using Microsoft Edge TTS service."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, ToolRuntime, ToolStatus, ToolTier
from tools.tool_registry import ToolRegistry
from tools.audio.edge_tts import EdgeTTS
from tools.audio.tts_selector import TTSSelector


class TestEdgeTTSContract:
    def test_inherits_base_tool(self):
        assert issubclass(EdgeTTS, BaseTool)

    def test_identity(self):
        t = EdgeTTS()
        assert t.name == "edge_tts"
        assert t.capability == "tts"
        assert t.provider == "edge_tts"
        assert t.runtime == ToolRuntime.API
        assert t.tier == ToolTier.VOICE
        assert t.fallback == "piper_tts"
        assert "piper_tts" in t.fallback_tools
        assert "text-to-speech" in t.agent_skills
        assert "speaking_rate" in t.idempotency_key_fields
        assert len(t.capabilities) > 0

    def test_get_info_valid(self):
        info = EdgeTTS().get_info()
        assert info["name"] == "edge_tts"
        assert info["capability"] == "tts"
        assert "text" in info["input_schema"]["properties"]
        assert "voice" in info["input_schema"]["properties"]
        assert "rate" in info["input_schema"]["properties"]
        assert "pitch" in info["input_schema"]["properties"]

    def test_estimate_cost_is_zero(self):
        t = EdgeTTS()
        assert t.estimate_cost({"text": "Hello world, this is free TTS."}) == 0.0

    def test_status_available(self):
        t = EdgeTTS()
        # edge-tts is installed in the test environment
        assert t.get_status() == ToolStatus.AVAILABLE


class TestEdgeTTSParameters:
    def test_voice_resolution(self):
        t = EdgeTTS()
        # Default voice
        assert t.resolve_voice(None) == "vi-VN-HoaiMyNeural"
        assert t.resolve_voice("") == "vi-VN-HoaiMyNeural"

        # Alias resolution
        assert t.resolve_voice("hoaimy") == "vi-VN-HoaiMyNeural"
        assert t.resolve_voice("namminh") == "vi-VN-NamMinhNeural"
        assert t.resolve_voice("jenny") == "en-US-JennyNeural"
        assert t.resolve_voice("guy") == "en-US-GuyNeural"
        assert t.resolve_voice("aria") == "en-US-AriaNeural"
        assert t.resolve_voice("xiaoxiao") == "zh-CN-XiaoxiaoNeural"
        assert t.resolve_voice("nanami") == "ja-JP-NanamiNeural"

        # Direct full name pass-through
        assert t.resolve_voice("vi-VN-NamMinhNeural") == "vi-VN-NamMinhNeural"
        assert t.resolve_voice("custom-voice-name") == "custom-voice-name"

    def test_rate_formatting(self):
        t = EdgeTTS()
        assert t.format_rate({}) == "+0%"
        assert t.format_rate({"rate": "+15%"}) == "+15%"
        assert t.format_rate({"rate": "20%"}) == "+20%"
        assert t.format_rate({"rate": "-10%"}) == "-10%"
        assert t.format_rate({"speed": 1.2}) == "+20%"
        assert t.format_rate({"speed": 0.85}) == "-15%"
        assert t.format_rate({"speaking_rate": 1.1}) == "+10%"

    def test_pitch_formatting(self):
        t = EdgeTTS()
        assert t.format_pitch({}) == "+0Hz"
        assert t.format_pitch({"pitch": "+5Hz"}) == "+5Hz"
        assert t.format_pitch({"pitch": "-2Hz"}) == "-2Hz"
        assert t.format_pitch({"pitch": 5}) == "+5Hz"
        assert t.format_pitch({"pitch": -3}) == "-3Hz"
        assert t.format_pitch({"pitch": "+5%"}) == "+5Hz"
        assert t.format_pitch({"pitch": "-5%"}) == "-5Hz"
        assert t.format_pitch({"pitch": "10%"}) == "+10Hz"
        assert t.format_pitch({"pitch": "5hz"}) == "+5Hz"

    def test_volume_formatting(self):
        t = EdgeTTS()
        assert t.format_volume({}) == "+0%"
        assert t.format_volume({"volume": "+10%"}) == "+10%"
        assert t.format_volume({"volume": "-15%"}) == "-15%"

    def test_parameter_edge_cases(self):
        t = EdgeTTS()
        assert t.format_pitch({"pitch": ""}) == "+0Hz"
        assert t.format_pitch({"pitch": "  "}) == "+0Hz"
        assert t.format_pitch({"pitch": "%"}) == "+0Hz"
        assert t.format_pitch({"pitch": "Hz"}) == "+0Hz"
        assert t.format_volume({"volume": ""}) == "+0%"
        assert t.format_volume({"volume": "  "}) == "+0%"
        assert t.format_rate({"rate": ""}) == "+0%"
        assert t.format_rate({"rate": 15}) == "+15%"
        assert t.format_rate({"rate": -10}) == "-10%"


class TestEdgeTTSExecution:
    def test_empty_text_error(self):
        t = EdgeTTS()
        res = t.execute({"text": ""})
        assert not res.success
        assert "empty" in res.error.lower()

    def test_mock_generation(self, tmp_path):
        t = EdgeTTS()
        out_file = tmp_path / "test.mp3"

        with patch("edge_tts.Communicate") as mock_comm_cls:
            mock_comm = MagicMock()

            async def fake_save(path_str):
                Path(path_str).write_bytes(b"FAKE_AUDIO_DATA")

            mock_comm.save = fake_save
            mock_comm_cls.return_value = mock_comm

            res = t.execute({
                "text": "Kiểm tra sinh giọng nói",
                "voice": "hoaimy",
                "output_path": str(out_file),
            })

            assert res.success
            assert res.data["voice"] == "vi-VN-HoaiMyNeural"
            assert res.data["provider"] == "edge_tts"
            assert res.cost_usd == 0.0
            assert out_file.exists()

    def test_mock_generation_srt_subtitles(self, tmp_path):
        t = EdgeTTS()
        out_file = tmp_path / "test.mp3"
        sub_file = tmp_path / "test.srt"

        with patch("edge_tts.Communicate") as mock_comm_cls:
            mock_comm = MagicMock()

            async def fake_stream():
                yield {"type": "audio", "data": b"FAKE_MP3"}
                yield {
                    "type": "WordBoundary",
                    "offset": 1000000,
                    "duration": 2000000,
                    "text": "Xin",
                }
                yield {
                    "type": "WordBoundary",
                    "offset": 3000000,
                    "duration": 2000000,
                    "text": "chào",
                }

            mock_comm.stream = fake_stream
            mock_comm_cls.return_value = mock_comm

            res = t.execute({
                "text": "Xin chào",
                "output_path": str(out_file),
                "subtitle_path": str(sub_file),
            })

            assert res.success
            mock_comm_cls.assert_called_once()
            _, kwargs = mock_comm_cls.call_args
            assert kwargs.get("boundary") == "WordBoundary"

            assert sub_file.exists()
            content = sub_file.read_text(encoding="utf-8")
            assert "Xin" in content
            assert "chào" in content
            assert res.data["subtitle_path"] == str(sub_file)

    def test_mock_generation_vtt_subtitles(self, tmp_path):
        t = EdgeTTS()
        out_file = tmp_path / "test.mp3"
        sub_file = tmp_path / "test.vtt"

        with patch("edge_tts.Communicate") as mock_comm_cls:
            mock_comm = MagicMock()

            async def fake_stream():
                yield {"type": "audio", "data": b"FAKE_MP3"}
                yield {
                    "type": "WordBoundary",
                    "offset": 1250000,
                    "duration": 2250000,
                    "text": "Hello",
                }

            mock_comm.stream = fake_stream
            mock_comm_cls.return_value = mock_comm

            res = t.execute({
                "text": "Hello",
                "output_path": str(out_file),
                "subtitle_path": str(sub_file),
            })

            assert res.success
            assert sub_file.exists()
            content = sub_file.read_text(encoding="utf-8")
            assert content.startswith("WEBVTT")
            assert "00:00:00.125 --> 00:00:00.350" in content
            assert "Hello" in content

    def test_wav_output_transcoding(self, tmp_path):
        t = EdgeTTS()
        out_wav = tmp_path / "output.wav"

        with patch("edge_tts.Communicate") as mock_comm_cls, \
             patch("subprocess.run") as mock_subproc:
            mock_comm = MagicMock()

            async def fake_save(path_str):
                Path(path_str).write_bytes(b"FAKE_MP3_DATA")

            mock_comm.save = fake_save
            mock_comm_cls.return_value = mock_comm

            def fake_run(cmd, *args, **kwargs):
                if "ffmpeg" in cmd[0] or cmd[0] == "ffmpeg":
                    # simulate writing a real WAV with RIFF header
                    Path(cmd[-1]).write_bytes(b"RIFF....WAVEfmt ....data....")
                    return MagicMock(returncode=0, stdout="", stderr="")
                return MagicMock(returncode=0, stdout="", stderr="")

            mock_subproc.side_effect = fake_run

            res = t.execute({
                "text": "Test WAV output",
                "output_path": str(out_wav),
                "output_format": "wav",
            })

            assert res.success
            assert out_wav.exists()
            header = out_wav.read_bytes()[:4]
            assert header == b"RIFF"


class TestTTSSelectorIntegration:
    def test_adapt_inputs_for_edge_tts(self):
        selector = TTSSelector()
        adapted = selector._adapt_inputs(
            EdgeTTS(),
            {
                "text": "Hello",
                "voice_id": "vi-VN-HoaiMyNeural",
                "speed": 1.25,
                "pitch": 5,
                "output_format": "mp3",
            },
        )
        assert adapted["voice"] == "vi-VN-HoaiMyNeural"
        assert adapted["rate"] == "+25%"
        assert adapted["pitch"] == "+5Hz"
        assert adapted["output_format"] == "mp3"

    def test_selector_can_choose_edge_tts(self, tmp_path):
        reg = ToolRegistry()
        edge = EdgeTTS()
        selector = TTSSelector()
        reg.register(edge)
        reg.register(selector)

        out_file = tmp_path / "selector_test.mp3"

        with patch.object(edge, "_generate") as mock_gen:
            mock_gen.return_value = MagicMock(
                success=True,
                data={"output": str(out_file), "voice": "vi-VN-HoaiMyNeural"},
                artifacts=[str(out_file)],
                cost_usd=0.0,
            )

            res = selector.execute({
                "text": "Thử nghiệm qua selector",
                "preferred_provider": "edge_tts",
                "output_path": str(out_file),
            })

            assert res.success
            assert res.data["selected_provider"] == "edge_tts"
            assert res.data["selected_tool"] == "edge_tts"

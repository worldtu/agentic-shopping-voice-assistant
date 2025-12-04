"""
Voice module for Speech Processing

This package contains:
- tts.py: Text-to-Speech using OpenAI TTS
- asr.py: Speech-to-Text using local Whisper (faster-whisper)
"""

from .tts import synthesize_speech
from .asr import get_asr_instance, WhisperASR

__all__ = ["synthesize_speech", "get_asr_instance", "WhisperASR"]

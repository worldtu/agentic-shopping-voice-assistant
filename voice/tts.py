"""
Text-to-Speech (TTS) Module using ElevenLabs

Based on pipecat/services/elevenlabs/tts.py
Implements high-quality speech synthesis with voice customization.
"""

import base64
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import aiohttp
from loguru import logger
import os

class ElevenLabsTTS:
    """ElevenLabs Text-to-Speech Service
    
    Provides high-quality speech synthesis with multiple voices and languages.
    Uses ElevenLabs HTTP API for fragment-based audio generation.
    """
    
    # Default ElevenLabs voices
    DEFAULT_VOICES = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",  # English female
        "adam": "pNInz6obpgDQGcFmaJgB",     # English male
        "bella": "EXAVITQu4vr4xnSDxMaL",    # English female
        "antoni": "ErXwobaYiN019PkySvjV",   # English male
        "elli": "MF3mGyEYCl7XYWbV9V6O",     # English female
        "josh": "TxGEqnHWrfWFTfGW9XjX",     # English male
        "arnold": "VR6AewLTigWG4xSOukaG",   # English male
        "domi": "AZnzlk1XvdvUeBnXmlld",     # English female
        "sam": "yoZ06aMxZJJ28mfd3POQ",      # English male
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.elevenlabs.io",
        default_voice_id: Optional[str] = None,
        default_model: str = "eleven_turbo_v2_5",
        sample_rate: int = 44100
    ):
        """Initialize the ElevenLabs TTS service.
        
        Args:
            api_key: ElevenLabs API key (or reads from ELEVENLABS_API_KEY env var)
            base_url: Base URL for ElevenLabs API
            default_voice_id: Default voice ID to use
            default_model: TTS model to use
            sample_rate: Output audio sample rate
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY environment variable not set. "
                "Please set it with: export ELEVENLABS_API_KEY='your-key-here'"
            )
        
        self.base_url = base_url
        self.default_voice_id = default_voice_id or self.DEFAULT_VOICES["rachel"]
        self.model = default_model
        self.sample_rate = sample_rate
        
        # Voice settings
        self.voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    
    def _get_output_format(self) -> str:
        """Get the appropriate output format for the sample rate.
        
        Returns MP3 format for browser compatibility.
        PCM formats are not playable in browsers without conversion.
        """
        # Use MP3 format for browser compatibility
        format_map = {
            8000: "mp3_22050_32",
            16000: "mp3_22050_32",
            22050: "mp3_22050_32",
            24000: "mp3_44100_64",
            44100: "mp3_44100_128"
        }
        return format_map.get(self.sample_rate, "mp3_44100_128")
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        **voice_settings
    ) -> Dict[str, Any]:
        """Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (overrides default)
            language: Language code (e.g., 'en', 'zh', 'es')
            **voice_settings: Optional voice settings (stability, similarity_boost, etc.)
            
        Returns:
            Dictionary with 'audio_data', 'duration', and 'format' keys
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Use provided voice or default
        vid = voice_id or self.default_voice_id
        
        # Build the URL
        url = f"{self.base_url}/v1/text-to-speech/{vid}"
        
        # Merge voice settings
        settings = {**self.voice_settings, **voice_settings}
        
        # Build payload
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": settings
        }
        
        # Add language if supported by model
        if language and self.model in ["eleven_turbo_v2_5", "eleven_flash_v2_5"]:
            payload["language_code"] = language
        
        # Build headers
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        
        # Build query parameters
        params = {
            "output_format": self._get_output_format()
        }
        
        logger.info(f"Synthesizing with ElevenLabs: {len(text)} characters, voice={vid}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    params=params
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs API error: {error_text}")
                        raise Exception(f"ElevenLabs API error ({response.status}): {error_text}")
                    
                    # Read audio data
                    audio_data = await response.read()
                    
                    logger.info(f"ElevenLabs synthesis complete: {len(audio_data)} bytes")
                    
                    # Estimate duration (rough estimate)
                    # For MP3, roughly 1 second per 16KB at 128kbps
                    duration = len(audio_data) / 16000.0
                    
                    return {
                        "audio_data": audio_data,
                        "duration": duration,
                        "format": "mp3",
                        "sample_rate": self.sample_rate
                    }
        
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    async def get_voices(self) -> list:
        """Get available voices from ElevenLabs.
        
        Returns:
            List of voice dictionaries with 'id', 'name', and 'language' keys
        """
        url = f"{self.base_url}/v1/voices"
        headers = {"xi-api-key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch voices: {response.status}")
                        # Return default voices as fallback
                        return [
                            {"id": vid, "name": name.capitalize(), "language": "en"}
                            for name, vid in self.DEFAULT_VOICES.items()
                        ]
                    
                    data = await response.json()
                    voices = []
                    
                    for voice in data.get("voices", []):
                        voices.append({
                            "id": voice.get("voice_id"),
                            "name": voice.get("name"),
                            "language": voice.get("labels", {}).get("language", "en"),
                            "gender": voice.get("labels", {}).get("gender", ""),
                            "accent": voice.get("labels", {}).get("accent", "")
                        })
                    
                    return voices
        
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            # Return default voices as fallback
            return [
                {"id": vid, "name": name.capitalize(), "language": "en"}
                for name, vid in self.DEFAULT_VOICES.items()
            ]
    
    def set_voice_settings(
        self,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
        use_speaker_boost: Optional[bool] = None
    ):
        """Update default voice settings.
        
        Args:
            stability: Voice stability (0.0-1.0)
            similarity_boost: Similarity boost (0.0-1.0)
            style: Style control (0.0-1.0)
            use_speaker_boost: Whether to use speaker boost
        """
        if stability is not None:
            self.voice_settings["stability"] = stability
        if similarity_boost is not None:
            self.voice_settings["similarity_boost"] = similarity_boost
        if style is not None:
            self.voice_settings["style"] = style
        if use_speaker_boost is not None:
            self.voice_settings["use_speaker_boost"] = use_speaker_boost


# Global TTS instance (singleton pattern)
_tts_instance: Optional[ElevenLabsTTS] = None


def get_tts_instance(
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None
) -> ElevenLabsTTS:
    """Get or create the global TTS instance.
    
    Args:
        api_key: ElevenLabs API key (only used on first call)
        voice_id: Default voice ID (only used on first call)
        
    Returns:
        ElevenLabsTTS instance
    """
    global _tts_instance
    
    if _tts_instance is None:
        logger.info("Initializing ElevenLabs TTS instance for the first time...")
        _tts_instance = ElevenLabsTTS(
            api_key=api_key,
            default_voice_id=voice_id
        )
    
    return _tts_instance


def reset_tts_instance():
    """Reset the global TTS instance (useful for testing or changing configuration)."""
    global _tts_instance
    _tts_instance = None
    logger.info("TTS instance reset")


# ============================================================================
# Compatibility wrapper for existing code
# ============================================================================

async def synthesize_speech_async(
    text: str,
    output_path: str,
    voice: str = "rachel",
    model: str = "eleven_turbo_v2_5"
) -> str:
    """
    Convert text to speech using ElevenLabs TTS (async version)
    
    Args:
        text: Answer text to speak
        output_path: Where to save audio file (e.g., "output.mp3")
        voice: Voice selection - options:
            - "rachel": English female (default)
            - "adam": English male
            - "bella": English female
            - "antoni": English male
            - "elli": English female
            - "josh": English male
            - Or any ElevenLabs voice ID
        model: TTS model to use (default: "eleven_turbo_v2_5")
    
    Returns:
        Path to generated audio file (same as output_path)
    
    Raises:
        ValueError: If ElevenLabs API key is not set
        Exception: If TTS generation fails
    
    Examples:
        >>> await synthesize_speech_async("Hello world", "greeting.mp3")
        'greeting.mp3'
    """
    # Get TTS instance
    tts = get_tts_instance()
    
    # Map voice name to voice ID
    voice_id = ElevenLabsTTS.DEFAULT_VOICES.get(voice, voice)
    
    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate speech
        result = await tts.synthesize(text, voice_id=voice_id)
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(result["audio_data"])
        
        logger.info(f"Audio saved to: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise Exception(f"TTS generation failed: {str(e)}")


def synthesize_speech(
    text: str,
    output_path: str,
    voice: str = "rachel",
    model: str = "eleven_turbo_v2_5"
) -> str:
    """
    Convert text to speech using ElevenLabs TTS (synchronous wrapper)
    
    This is a compatibility wrapper that runs the async function synchronously.
    
    Args:
        text: Answer text to speak
        output_path: Where to save audio file (e.g., "output.mp3")
        voice: Voice selection (see synthesize_speech_async for options)
        model: TTS model to use
    
    Returns:
        Path to generated audio file (same as output_path)
    
    Examples:
        >>> synthesize_speech("Hello world", "greeting.mp3")
        'greeting.mp3'
    """
    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        synthesize_speech_async(text, output_path, voice, model)
    )


def estimate_audio_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate audio duration in seconds
    
    Args:
        text: Text to estimate
        words_per_minute: Speaking rate (default 150 wpm)
    
    Returns:
        Estimated duration in seconds
    
    Examples:
        >>> estimate_audio_duration("Hello world")
        0.8
    """
    words = len(text.split())
    return (words / words_per_minute) * 60


# ============================================================================
# Voice mapping for easy migration from OpenAI
# ============================================================================

VOICE_MAPPING = {
    # OpenAI voices -> ElevenLabs equivalents
    "alloy": "rachel",    # Neutral, balanced
    "echo": "adam",       # Male, clear
    "fable": "bella",     # Female, expressive
    "onyx": "arnold",     # Deep male
    "nova": "elli",       # Female, energetic
    "shimmer": "bella",   # Female, warm
}


def map_voice(openai_voice: str) -> str:
    """Map OpenAI voice names to ElevenLabs equivalents."""
    return VOICE_MAPPING.get(openai_voice, openai_voice)


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("ElevenLabs TTS Module Demo")
    print("=" * 70)
    
    # Check API key
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("\n❌ ELEVENLABS_API_KEY not set")
        print("Set it with: export ELEVENLABS_API_KEY='your-key-here'")
        print("\nGet your API key at: https://elevenlabs.io/app/settings/api-keys")
    else:
        print("\n✅ ELEVENLABS_API_KEY is set")
        
        # Test synthesis
        test_text = "Welcome to our voice shopping assistant. How can I help you today?"
        output_path = "test_output_elevenlabs.mp3"
        
        try:
            print(f"\n🎤 Generating speech...")
            result = synthesize_speech(test_text, output_path, voice="rachel")
            duration = estimate_audio_duration(test_text)
            print(f"✅ Generated: {result}")
            print(f"📊 Estimated duration: {duration:.1f} seconds")
            print(f"\n🎵 You can play the audio file: {output_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)

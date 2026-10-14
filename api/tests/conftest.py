import os
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
import torch
from hypothesis import settings

from api.src.inference.model_manager import ModelManager
from api.src.inference.voice_manager import VoiceManager
from api.src.services.tts_service import TTSService

settings.register_profile("dev", max_examples=300, deadline=None)
settings.register_profile("ci", max_examples=500, deadline=None, derandomize=True)
settings.load_profile("ci" if os.environ.get("CI") else "dev")


@pytest.fixture(autouse=True)
def mock_chunk_g2p(monkeypatch, request):
    if request.path.name == "test_real_model.py" or "integration" in request.path.parts:
        return
    from api.src.services.text_processing import text_processor

    def phonemize(text, language):
        return "h" * (len(text) * (3 if language == "z" else 1))

    monkeypatch.setattr(text_processor, "phonemize", phonemize)


@pytest.fixture
def mock_voice_tensor():
    """Load a real voice tensor for testing."""
    voice_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src/voices/af_bella.pt"
    )
    return torch.load(voice_path, map_location="cpu", weights_only=False)


@pytest.fixture
def mock_audio_output():
    """Load pre-generated test audio for consistent testing."""
    test_audio_path = os.path.join(
        os.path.dirname(__file__), "test_data/test_audio.npy"
    )
    return np.load(test_audio_path)  # Return as numpy array instead of bytes


@pytest_asyncio.fixture
async def mock_model_manager(mock_audio_output):
    """Mock model manager for testing."""
    manager = AsyncMock(spec=ModelManager)
    manager.get_backend = MagicMock()

    async def mock_generate(*args, **kwargs):
        # Simulate successful audio generation
        return np.random.rand(24000).astype(np.float32)  # 1 second of random audio data

    manager.generate = AsyncMock(side_effect=mock_generate)
    return manager


@pytest_asyncio.fixture
async def mock_voice_manager(mock_voice_tensor):
    """Mock voice manager for testing."""
    manager = AsyncMock(spec=VoiceManager)
    manager.get_voice_path = MagicMock(return_value="/mock/path/voice.pt")
    manager.load_voice = AsyncMock(return_value=mock_voice_tensor)
    manager.list_voices = AsyncMock(return_value=["voice1", "voice2"])
    return manager


@pytest_asyncio.fixture
async def tts_service(mock_model_manager, mock_voice_manager):
    """Get mocked TTS service instance."""
    service = TTSService()
    service.model_manager = mock_model_manager
    service._voice_manager = mock_voice_manager
    return service


@pytest.fixture
def test_voice():
    """Return a test voice name."""
    return "voice1"

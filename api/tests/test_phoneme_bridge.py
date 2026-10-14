import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.src.core import espeak
from api.src.core.config import settings
from api.src.core.languages import canonical_language
from api.src.inference.kokoro_v1 import KokoroV1
from api.src.main import app
from api.src.routers import openai_compatible
from api.src.services.audio import AudioService
from api.src.services.text_processing import phoneme_bridge as bridge
from api.src.services.text_processing.language_routing import (
    LanguageSpan,
    SpeechSegment,
    language_capabilities,
    prepare_speech_segments,
)
from api.src.services.tts_service import TTSService
from api.src.structures.schemas import NormalizationOptions, OpenAISpeechRequest


@pytest.fixture
def vocab():
    return json.loads(Path("api/src/models/v1_0/config.json").read_text())["vocab"]


@pytest.mark.parametrize("language", ["ru", "uk", "de", "ar", "vi"])
def test_explicit_routes(language):
    assert (
        OpenAISpeechRequest(input="test", lang_code=language).lang_code
        == "espeak:" + language
    )


def test_inventory_and_native_priority():
    inventory = espeak.installed_languages()
    assert inventory
    advertised = language_capabilities()["experimental"]
    assert {route["language"] for route in advertised} == set(inventory)
    for route in advertised:
        assert route["experimental"] and route["manual"]
        assert canonical_language(route["lang_code"]) == route["lang_code"]
    assert canonical_language("fr_FR") == "f"
    assert canonical_language("DE_de") == "espeak:de"
    assert canonical_language("VI_VN_X_CENTRAL") == "espeak:vi-vn-x-central"


@pytest.mark.parametrize(
    "language,text",
    [
        ("ru", "Привет, мир. Это проверка русской речи."),
        ("ru", "Это полное предложение на русском языке."),
        ("uk", "Привіт, світе. Це перевірка української мови."),
        ("de", "Guten Tag. Dies ist eine deutsche Sprachprobe."),
        ("ar", "مرحبا بالعالم. هذه تجربة باللغة العربية."),
        ("vi", "Xin chào thế giới. Đây là thử nghiệm tiếng Việt."),
    ],
)
def test_real_espeak_fixtures(language, text, vocab):
    ipa = bridge.phonemize(text, "espeak:" + language)
    mapped = bridge.map_ipa(ipa, vocab)
    assert mapped.phonemes.strip()
    assert set(mapped.phonemes) <= vocab.keys()
    assert mapped.phonemes.count("ˈ") == ipa.count("ˈ")
    assert mapped.phonemes.count("ː") == ipa.count("ː")
    assert "." in mapped.phonemes


@pytest.mark.parametrize("text", ["Привет!", "Привет, people!"])
def test_multi_uses_russian_word_phonemes(text, vocab):
    segment = prepare_speech_segments(text, "af_bella", "multi")[0]
    assert segment.span.lang_code == "espeak:ru"
    chunks = bridge.prepare_chunks(
        segment.span.text, segment.span.lang_code, NormalizationOptions(), vocab, 450
    )
    assert "prʲivʲˈet" in chunks[0].phonemes
    assert chunks == bridge.prepare_chunks(
        segment.span.text, "espeak:ru", NormalizationOptions(), vocab, 450
    )


def test_stress_softness_length_combining_marks_and_substitutions(vocab):
    assert bridge.map_ipa("ç ø œ ã", vocab).phonemes == "ç ø œ ã"
    result = bridge.map_ipa("ˌprʲˈivʲeːt n̩ ɫ ã t͡ʃ", vocab)
    assert result.phonemes == "ˌprʲˈivʲeːt ən l ã ʧ"
    assert bridge.map_ipa("ɭ", vocab).substitutions == (("ɭ", "l"),)
    assert ("ɫ", "l") in result.substitutions
    assert ("̩", "ən") in result.substitutions
    reduced = dict(vocab)
    reduced.pop("ʲ")
    assert bridge.map_ipa("pʲ", reduced).phonemes == "pj"
    assert bridge.map_ipa("t̪", vocab).substitutions == (("̪", ""),)


def test_unknown_symbol_rejection_does_not_echo_input(vocab):
    with pytest.raises(ValueError, match="U\\+2603") as error:
        bridge.map_ipa("secret ☃", vocab)
    assert "secret" not in str(error.value)
    assert bridge.MAPPING_VERSION in str(error.value)
    # Generic text_processing VOCAB includes characters the model does not.
    with pytest.raises(ValueError, match="U\\+0042"):
        bridge.map_ipa("B", vocab)
    with pytest.raises(ValueError, match="U\\+02D0"):
        bridge.map_ipa("aː", {"a": 1})


def test_missing_espeak_and_unknown_language(monkeypatch):
    monkeypatch.setattr(espeak, "installed_languages", lambda: {})
    assert language_capabilities()["experimental"] == []
    assert not language_capabilities()["phoneme_bridge"]["available"]
    with pytest.raises(ValueError, match="unavailable eSpeak"):
        canonical_language("ru")
    assert canonical_language("en") == "a"


def test_initialization_is_explicit(monkeypatch):
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    calls = []
    monkeypatch.setattr(espeakng_loader, "get_library_path", lambda: "library")
    monkeypatch.setattr(espeak, "_data_path", lambda: Path("safe-data"))
    monkeypatch.setattr(EspeakWrapper, "set_library", lambda path: calls.append(path))
    monkeypatch.setattr(EspeakWrapper, "set_data_path", lambda path: calls.append(path))
    monkeypatch.setattr(
        EspeakBackend,
        "supported_languages",
        lambda: calls.append("enumerate") or {"ru": "Russian"},
    )
    assert espeak.installed_languages.__wrapped__() == {"ru": "Russian"}
    assert calls == ["library", "safe-data", "enumerate"]


@pytest.mark.parametrize(
    "global_enabled,request_enabled", [(False, True), (True, False)]
)
def test_disabled_explicit_bridge(monkeypatch, global_enabled, request_enabled):
    monkeypatch.setattr(settings, "phoneme_bridge", global_enabled)
    with pytest.raises(ValueError, match="disabled"):
        prepare_speech_segments(
            "Привет", "af_bella", "ru", phoneme_bridge=request_enabled
        )


def test_long_chunks_keep_all_mapped_content(vocab, monkeypatch):
    ipa = "ˈaːñ " * 1000
    monkeypatch.setattr(bridge, "phonemize", lambda *_: ipa)
    chunks = bridge.prepare_chunks(
        "длинный текст", "espeak:ru", NormalizationOptions(), vocab, 120
    )
    assert len(chunks) > 1
    assert all(len(chunk.phonemes) <= 120 for chunk in chunks)
    assert "".join(chunk.phonemes for chunk in chunks) == ipa
    assert all(set(chunk.phonemes) <= vocab.keys() for chunk in chunks)


def test_bridge_retains_input_policies(vocab, monkeypatch):
    seen = []
    monkeypatch.setattr(bridge, "phonemize", lambda text, _: seen.append(text) or "ˈa")
    chunks = bridge.prepare_chunks(
        "12 cafe\u0301 😊 [pause:0.2s] [word](/pʲˈaː/)",
        "espeak:ru",
        NormalizationOptions(remove_emoji=True),
        vocab,
        100,
    )
    assert seen == ["12 café"]
    assert chunks[1].pause == 0.2
    assert chunks[2].phonemes == "pʲˈaː"
    assert chunks[2].text == "[word](/pʲˈaː/)"


@pytest.mark.parametrize("stream", [True, False])
def test_mapping_error_before_response_writer(stream, monkeypatch):
    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_bella"]
    service.prepare_bridge_segments.side_effect = ValueError(
        "Unmapped eSpeak phoneme (U+2603); choose a native language."
    )
    monkeypatch.setattr(
        openai_compatible, "get_tts_service", AsyncMock(return_value=service)
    )
    writer = MagicMock()
    monkeypatch.setattr(openai_compatible, "StreamingAudioWriter", writer)
    response = TestClient(app).post(
        "/v1/audio/speech",
        json={
            "input": "Привет",
            "voice": "af_bella",
            "lang_code": "ru",
            "stream": stream,
        },
    )
    assert response.status_code == 400
    assert "U+2603" in response.json()["detail"]["message"]
    writer.assert_not_called()
    service.generate_audio.assert_not_called()
    service.generate_audio_stream.assert_not_called()


@asynccontextmanager
async def hold():
    yield


async def test_service_uses_direct_tokens_keeps_order_voice_speed_and_cancel(
    vocab, monkeypatch
):
    backend = KokoroV1()
    backend._model = SimpleNamespace(vocab=vocab, context_length=512)
    service = TTSService()
    service.model_manager = MagicMock()
    service.model_manager.ensure_backend = AsyncMock()
    service.model_manager.get_backend.return_value = backend
    service.model_manager.hold = hold
    service.get_voices_path = AsyncMock(return_value=("af_bella", "voices/af_bella.pt"))
    calls = []

    async def generate(tokens, voice, speed, lang_code):
        calls.append((tokens, voice, speed, lang_code))
        yield np.ones(240, dtype=np.float32)

    backend.generate_from_tokens = generate
    monkeypatch.setattr(AudioService, "trim_audio", lambda chunk, *args: chunk)
    prepared = [
        SpeechSegment(
            "af_bella",
            1.2,
            LanguageSpan("текст", "espeak:ru"),
            (bridge.BridgeChunk("текст", "ˈa"), bridge.BridgeChunk("", "b")),
        )
    ]
    chunks = [
        chunk
        async for chunk in service.generate_audio_stream(
            "текст",
            "af_bella",
            MagicMock(),
            speed=1.25,
            output_format=None,
            prepared_segments=prepared,
        )
    ]
    assert len(chunks) == 3
    assert [call[0] for call in calls] == ["ˈa", "b"]
    assert all(
        call[1:] == (("af_bella", "voices/af_bella.pt"), 1.5, "espeak:ru")
        for call in calls
    )
    service.model_manager.generate.assert_not_called()
    service.get_voices_path.assert_awaited_once()
    calls.clear()
    stream = service.generate_audio_stream(
        "текст", "af_bella", MagicMock(), output_format=None, prepared_segments=prepared
    )
    await anext(stream)
    await stream.aclose()
    assert len(calls) == 1


async def test_loaded_vocab_direct_inference_never_calls_g2p(vocab, monkeypatch):
    backend = KokoroV1()
    backend._model = SimpleNamespace(vocab=vocab, context_length=512)
    backend._get_voice_tensor = AsyncMock(return_value=torch.ones(1))
    monkeypatch.setattr("api.src.core.paths.save_voice_tensor", AsyncMock())
    pipeline = MagicMock()
    pipeline.generate_from_tokens.return_value = [SimpleNamespace(audio=torch.ones(8))]
    backend._get_pipeline = MagicMock(return_value=pipeline)
    result = [
        audio
        async for audio in backend.generate_from_tokens(
            "pʲˈaː", "af_bella", speed=1.3, lang_code="espeak:ru"
        )
    ]
    assert len(result) == 1
    pipeline.assert_not_called()
    pipeline.g2p.assert_not_called()
    assert pipeline.generate_from_tokens.call_args.kwargs["tokens"] == "pʲˈaː"
    assert pipeline.generate_from_tokens.call_args.kwargs["speed"] == 1.3
    for invalid in ["☃", "a" * 511]:
        with pytest.raises(ValueError, match="vocabulary/limit"):
            async for _ in backend.generate_from_tokens(
                invalid, "af_bella", lang_code="espeak:ru"
            ):
                pass


async def test_prepare_validates_whole_request_using_loaded_vocab(vocab, monkeypatch):
    service = TTSService()
    backend = KokoroV1()
    backend._model = SimpleNamespace(vocab={"a": 43}, context_length=8)
    service.model_manager = MagicMock()
    service.model_manager.hold = hold
    service.model_manager.ensure_backend = AsyncMock()
    service.model_manager.get_backend.return_value = backend
    monkeypatch.setattr(
        bridge, "phonemize", lambda text, _: "a" if text == "first" else "b"
    )
    segments = [
        SpeechSegment("af_bella", 1, LanguageSpan(text, "espeak:ru"))
        for text in ["first", "second"]
    ]
    with pytest.raises(ValueError, match="U\\+0062"):
        await service.prepare_bridge_segments(segments)
    assert all(segment.bridge_chunks is None for segment in segments)


def test_mac_data_copy_is_real_owned_and_cleaned(monkeypatch, tmp_path):
    import shutil

    source = tmp_path / ("deep" * 50)
    source.mkdir()
    (source / "phontab").write_bytes(b"data")
    monkeypatch.setenv("ESPEAK_DATA_PATH", str(source))
    monkeypatch.setattr(espeak.sys, "platform", "darwin")
    original = espeak._DATA_COPY
    try:
        copied = espeak._data_path.__wrapped__()
        assert copied != source
        assert not copied.is_symlink()
        assert (copied / "phontab").read_bytes() == b"data"
        assert len(str(copied).encode()) < len(str(source).encode())
        shutil.rmtree(source)
        assert (copied / "phontab").is_file()
        espeak._DATA_COPY.cleanup()
        assert not copied.exists()
    finally:
        espeak._DATA_COPY = original


def test_reinitialization_restores_paths_after_misaki_override(monkeypatch):
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    path = espeak._data_path()
    monkeypatch.setattr(EspeakWrapper, "_ESPEAK_DATA_PATH", "/invalid/long-path")
    espeak.initialize_espeak()
    assert Path(EspeakWrapper().data_path) == path.resolve()


@pytest.mark.parametrize("cluster", ["и\u0301", "क्\u200dष"])
def test_source_chunks_preserve_graphemes_at_boundary(vocab, monkeypatch, cluster):
    import regex

    text = "б" * 255 + cluster + "в" * 260
    seen = []
    monkeypatch.setattr(bridge, "phonemize", lambda text, _: seen.append(text) or "a")
    bridge.prepare_chunks(text, "espeak:ru", NormalizationOptions(), vocab, 100)
    assert "".join(seen) == text
    assert seen[0].endswith(cluster)
    assert [
        unit for part in seen for unit in regex.findall(r"\X", part)
    ] == regex.findall(r"\X", text)


def test_source_chunks_prefer_word_boundaries():
    text = "слово " * 100
    pieces = list(bridge.split_text(text))
    assert "".join(pieces) == text
    assert all(piece.endswith(" ") for piece in pieces)


def test_unknown_dialect_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        canonical_language("ru-vn-x-made-up")

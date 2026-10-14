from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from lingua import Language

from api.src.core.config import settings
from api.src.core.languages import NATIVE_LANGUAGES
from api.src.inference.base import AudioChunk
from api.src.main import app
from api.src.routers import development, openai_compatible
from api.src.services.text_processing import (
    language_routing,
    phonemizer,
    text_processor,
)
from api.src.services.text_processing.language_routing import prepare_language_spans
from api.src.services.tts_service import TTSService
from api.src.structures.schemas import NormalizationOptions, OpenAISpeechRequest

RAW = NormalizationOptions(normalize=False)


def confidence(language, value=1.0):
    return SimpleNamespace(language=language, value=value)


def mixed_scores(text):
    if text == "日本語です":
        return [confidence(Language.JAPANESE)]
    if text in {"Bonjour", "à", "tous", "tout", "le", "monde"}:
        return [confidence(Language.FRENCH)]
    return [confidence(Language.ENGLISH)]


@pytest.fixture
def detector(monkeypatch):
    mock = MagicMock()
    mock.compute_language_confidence_values.return_value = [
        confidence(Language.ENGLISH, 0.9),
        confidence(Language.FRENCH, 0.1),
    ]
    monkeypatch.setattr(language_routing, "get_detector", lambda: mock)
    return mock


@pytest.mark.parametrize(
    "alias,code",
    [
        ("auto", "auto"),
        ("multi", "auto"),
        ("EN_us", "a"),
        ("en-GB", "b"),
        ("es", "e"),
        ("fr-FR", "f"),
        ("hi", "h"),
        ("it", "i"),
        ("pt-BR", "p"),
        ("ja", "j"),
        ("zh-Hant", "z"),
        ("es-ES", "e"),
        ("es-419", "e"),
        ("hi-IN", "h"),
        ("it-IT", "i"),
        ("ja-JP", "j"),
        ("zh-Hans-CN", "z"),
    ],
)
def test_native_aliases(alias, code):
    assert OpenAISpeechRequest(input="test", lang_code=alias).lang_code == code


@pytest.mark.parametrize("code", NATIVE_LANGUAGES)
def test_native_g2p_has_no_speech_model(code, monkeypatch):
    pipeline = MagicMock()
    pipeline.g2p.return_value = ("h", [])
    constructor = MagicMock(return_value=pipeline)
    monkeypatch.setattr("kokoro.KPipeline", constructor)
    backend = phonemizer.create_phonemizer(code)
    assert backend.phonemize("12 café हिन्दी 日本語") == "h"
    constructor.assert_called_once_with(
        lang_code=code, model=False, repo_id=settings.model_repo_id
    )
    pipeline.g2p.assert_called_once_with("12 café हिन्दी 日本語")


def test_mixed_spans_keep_every_character(detector):
    text = "Hello everyone! Bonjour à tous! 日本語です。"
    detector.compute_language_confidence_values.side_effect = mixed_scores
    spans = prepare_language_spans(text, "auto", "a", RAW)
    assert [span.lang_code for span in spans] == ["a", "f", "j"]
    assert "".join(span.text for span in spans) == text


def test_offline_lingua_refines_neighboring_sentences():
    text = "This is a complete English sentence. Bonjour tout le monde, comment allez-vous?"
    spans = prepare_language_spans(text, "auto", "a", RAW)
    assert [span.lang_code for span in spans] == ["a", "f"]
    assert "".join(span.text for span in spans) == text


def test_short_ambiguity_inherits_phrase_then_voice(detector):
    ambiguous = [confidence(Language.ENGLISH, 0.51), confidence(Language.FRENCH, 0.49)]
    detector.compute_language_confidence_values.side_effect = lambda word: (
        ambiguous if word == "OK" else mixed_scores(word)
    )
    text = "OK Bonjour tout le monde."
    assert prepare_language_spans(text, "auto", "a", RAW)[0].lang_code == "f"
    assert prepare_language_spans("OK.", "auto", "b", RAW)[0].lang_code == "b"


@pytest.mark.parametrize("language", [Language.RUSSIAN, Language.UKRAINIAN])
@pytest.mark.parametrize("bridge", [True, False])
def test_cyrillic_is_detected_as_a_language_not_a_script_route(
    detector, language, bridge
):
    detector.compute_language_confidence_values.return_value = [confidence(language)]
    if bridge:
        spans = prepare_language_spans(
            "Это длинное предложение.", "auto", "a", RAW, bridge
        )
        assert spans[0].lang_code == "espeak:" + language.iso_code_639_1.name.lower()
    else:
        with pytest.raises(ValueError, match="disabled"):
            prepare_language_spans("Это длинное предложение.", "auto", "a", RAW, bridge)


def test_unknown_script_is_explicit_error(detector):
    with pytest.raises(ValueError, match="explicit lang_code"):
        prepare_language_spans("សួស្តី", "auto", "a", RAW)
    detector.compute_language_confidence_values.assert_not_called()


def test_unidentifiable_text_suggests_override(detector):
    detector.compute_language_confidence_values.return_value = []
    with pytest.raises(ValueError, match="explicit lang_code"):
        prepare_language_spans("unidentifiable text", "auto", "a", RAW)


def test_auto_normalizes_unicode_without_removing_accents(detector):
    detector.compute_language_confidence_values.return_value = [
        confidence(Language.FRENCH)
    ]
    spans = prepare_language_spans("12 cafe\u0301s français", "auto", "a", RAW)
    assert spans[0].text == "12 cafés français"
    assert spans[0].lang_code == "f"


def test_global_bridge_switch_is_enforced(detector, monkeypatch):
    monkeypatch.setattr(settings, "phoneme_bridge", False)
    detector.compute_language_confidence_values.return_value = [
        confidence(Language.RUSSIAN)
    ]
    with pytest.raises(ValueError, match="bridge is disabled"):
        prepare_language_spans("Это длинное предложение.", "auto", "a", RAW, True)


def test_literal_policies_do_not_detect_control_or_ipa_text(detector):
    text = "Bonjour [pause:1s] [mot](/həlˈoʊ/) 😊"
    detector.compute_language_confidence_values.return_value = [
        confidence(Language.FRENCH)
    ]
    spans = prepare_language_spans(
        text, "auto", "a", NormalizationOptions(remove_emoji=True)
    )
    assert spans[0].text == text
    probes = [
        call.args[0]
        for call in detector.compute_language_confidence_values.call_args_list
    ]
    assert probes == ["Bonjour", "mot"]


async def test_language_survives_sentence_clause_and_word_sizing(monkeypatch):
    languages = []

    def count(text, language):
        languages.append(language)
        return list(range(len(text)))

    monkeypatch.setattr(text_processor, "process_text_chunk", count)
    chunks = [
        chunk
        async for chunk in text_processor.smart_split(
            "日本語日本語日本語、bonjour tout le monde.",
            max_tokens=5,
            lang_code="j",
            normalization_options=RAW,
        )
    ]
    assert chunks and all(language == "j" for language in languages)
    assert all(len(tokens) <= 5 for _, tokens, _ in chunks)


async def test_mixed_stream_preserves_voice_speed_and_order(detector, monkeypatch):
    text = "Hello everyone! Bonjour tout le monde!"
    boundary = text.index("Bonjour")
    detector.compute_language_confidence_values.side_effect = mixed_scores
    monkeypatch.setattr(
        text_processor, "phonemize", lambda text, language: "h" * len(text)
    )
    service = TTSService()
    service.model_manager = AsyncMock()
    service.get_voices_path = AsyncMock(
        return_value=("af_bella", "/voices/af_bella.pt")
    )
    calls = []

    async def process(text, tokens, name, path, speed, *args, **kwargs):
        if text:
            calls.append((text, name, path, speed, kwargs["lang_code"]))
            yield AudioChunk(np.zeros(4, dtype=np.int16), output=b"pcm")

    service._process_chunk = process
    chunks = [
        chunk
        async for chunk in service.generate_audio_stream(
            text,
            "af_bella",
            MagicMock(),
            speed=1.3,
            output_format=None,
            lang_code="auto",
            normalization_options=RAW,
        )
    ]
    assert chunks
    assert [call[4] for call in calls] == ["a", "f"]
    assert [call[0] for call in calls] == [text[:boundary].strip(), text[boundary:]]
    assert all(call[1:4] == ("af_bella", "/voices/af_bella.pt", 1.3) for call in calls)
    service.get_voices_path.assert_awaited_once_with("af_bella")


@pytest.fixture
def speech_service(monkeypatch):
    service = AsyncMock(spec=TTSService)
    service.list_voices.return_value = ["af_bella"]
    monkeypatch.setattr(
        openai_compatible, "get_tts_service", AsyncMock(return_value=service)
    )
    return service


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("bridge", [False])
@pytest.mark.parametrize("language", ["auto", "multi"])
def test_unsupported_language_rejected_before_writer_or_dispatch(
    detector, speech_service, monkeypatch, stream, bridge, language
):
    detector.compute_language_confidence_values.return_value = [
        confidence(Language.RUSSIAN)
    ]
    writer = MagicMock()
    monkeypatch.setattr(openai_compatible, "StreamingAudioWriter", writer)
    response = TestClient(app).post(
        "/v1/audio/speech",
        json={
            "input": "Это длинное предложение.",
            "voice": "af_bella",
            "lang_code": language,
            "stream": stream,
            "phoneme_bridge": bridge,
        },
    )
    assert response.status_code == 400
    assert "phoneme bridge" in response.json()["detail"]["message"]
    writer.assert_not_called()
    speech_service.generate_audio.assert_not_called()
    speech_service.generate_audio_stream.assert_not_called()


def test_native_route_survives_disabled_bridge(speech_service, monkeypatch):
    monkeypatch.setattr(settings, "phoneme_bridge", False)
    speech_service.generate_audio.return_value = AudioChunk(
        np.zeros(4, dtype=np.int16), output=b"audio"
    )
    response = TestClient(app).post(
        "/v1/audio/speech",
        json={
            "input": "Bonjour, 12 cafés!",
            "voice": "af_bella",
            "lang_code": "fr-FR",
            "stream": False,
            "speed": 1.25,
            "phoneme_bridge": False,
        },
    )
    assert response.status_code == 200
    args = speech_service.generate_audio.call_args.kwargs
    assert args["voice"] == "af_bella" and args["speed"] == 1.25
    assert args["prepared_segments"][0].span.lang_code == "f"


def test_captioned_route_rejects_unsupported_language_before_stream(
    detector, speech_service, monkeypatch
):
    speech_service.prepare_bridge_segments.side_effect = ValueError(
        "Unmapped eSpeak phoneme (U+2603)"
    )
    dependency = development.get_tts_service
    monkeypatch.setattr(
        development, "get_tts_service", AsyncMock(return_value=speech_service)
    )
    app.dependency_overrides[dependency] = lambda: speech_service
    detector.compute_language_confidence_values.return_value = [
        confidence(Language.RUSSIAN)
    ]
    try:
        response = TestClient(app).post(
            "/dev/captioned_speech",
            json={
                "input": "Это длинное предложение.",
                "voice": "af_bella",
                "lang_code": "auto",
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400
    speech_service.generate_audio_stream.assert_not_called()


def test_capabilities_separate_detection_from_native_speech():
    response = TestClient(app).get("/v1/audio/languages")
    assert response.status_code == 200
    data = response.json()
    assert data["protocol_version"] == 1
    assert {route["lang_code"] for route in data["native"]} == set(NATIVE_LANGUAGES)
    assert "ru" in data["auto_detection"]["languages"]
    assert len(data["auto_detection"]["languages"]) == len(Language.all())
    assert data["experimental"] and data["phoneme_bridge"]["available"] is True
    assert "ru" in data["auto_detection"]["speech_languages"]
    assert data["phoneme_bridge"]["mapping"]["version"] == "espeak-kokoro-ipa-v1"


def test_non_english_numbers_and_unicode_remain_native(monkeypatch):
    seen = []
    monkeypatch.setattr(
        text_processor,
        "phonemize",
        lambda text, language: seen.append((text, language)) or "h",
    )

    async def collect():
        return [
            chunk
            async for chunk in text_processor.smart_split(
                "12 cafés, हिन्दी!", lang_code="f"
            )
        ]

    import asyncio

    asyncio.run(collect())
    assert seen == [("12 cafés, हिन्दी!", "f")]


def test_russian_english_code_switching():
    text = "Привет, друзья. This is an English sentence."
    spans = prepare_language_spans(text, "auto", "a", RAW)
    assert [span.lang_code for span in spans] == ["espeak:ru", "a"]
    assert "".join(span.text for span in spans) == text


@pytest.mark.parametrize("fallback", ["a", "b"])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Привет!", ["espeak:ru"]),
        ("Привет, people!", ["espeak:ru", "en"]),
        ("people, Привет!", ["en", "espeak:ru"]),
        ("Hello people!", ["en"]),
        ("HelloПривет!", ["en", "espeak:ru"]),
        ("Привіт, people!", ["espeak:uk", "en"]),
        ("Bonjour people!", ["f", "en"]),
        ("日本語です。中文。", ["j", "z"]),
    ],
)
def test_real_word_routing_and_auto_alias(text, expected, fallback):
    expected = [fallback if code == "en" else code for code in expected]
    spans = prepare_language_spans(text, "multi", fallback, RAW)
    assert [span.lang_code for span in spans] == expected
    assert "".join(span.text for span in spans) == text
    assert spans == prepare_language_spans(text, "auto", fallback, RAW)


def test_ambiguous_words_use_only_compatible_context(detector):
    def scores(word):
        if word == "Привет":
            return [
                confidence(Language.RUSSIAN, 0.52),
                confidence(Language.BULGARIAN, 0.48),
            ]
        return [confidence(Language.ENGLISH)]

    detector.compute_language_confidence_values.side_effect = scores
    spans = prepare_language_spans("Привет people", "multi", "a", RAW)
    assert [span.lang_code for span in spans] == ["espeak:ru", "a"]


def test_ambiguous_context_ties_prefer_earlier_word(detector):
    detector.compute_language_confidence_values.side_effect = lambda word: {
        "Bonjour": [confidence(Language.FRENCH)],
        "chat": [confidence(Language.ENGLISH, 0.51), confidence(Language.FRENCH, 0.49)],
        "people": [confidence(Language.ENGLISH)],
    }[word]
    spans = prepare_language_spans("Bonjour chat people", "multi", "a", RAW)
    assert [(span.text, span.lang_code) for span in spans] == [
        ("Bonjour chat ", "f"),
        ("people", "a"),
    ]


def test_context_does_not_cross_sentence_boundaries(detector):
    detector.compute_language_confidence_values.side_effect = lambda word: (
        [confidence(Language.FRENCH)]
        if word == "Bonjour"
        else [confidence(Language.ENGLISH, 0.51), confidence(Language.FRENCH, 0.49)]
    )
    spans = prepare_language_spans("Bonjour! chat", "multi", "a", RAW)
    assert [span.lang_code for span in spans] == ["f", "a"]


def test_phrase_context_resolves_ambiguous_words(detector):
    detector.compute_language_confidence_values.side_effect = lambda word: (
        [confidence(Language.FRENCH)]
        if word == "tout le monde"
        else [confidence(Language.ENGLISH, 0.51), confidence(Language.FRENCH, 0.49)]
    )
    assert (
        prepare_language_spans("tout le monde", "multi", "a", RAW)[0].lang_code == "f"
    )


def test_repeated_words_are_scored_once_per_request(detector):
    prepare_language_spans("people people. people!", "multi", "a", RAW)
    detector.compute_language_confidence_values.assert_called_once_with("people")


def test_multi_rejects_missing_russian_route_before_dispatch(
    speech_service, monkeypatch
):
    monkeypatch.setattr(language_routing.espeak, "installed_languages", lambda: {})
    response = TestClient(app).post(
        "/v1/audio/speech",
        json={"input": "Привет, people!", "voice": "af_bella", "lang_code": "multi"},
    )
    assert response.status_code == 400
    assert "no installed speech route" in response.json()["detail"]["message"]
    speech_service.prepare_bridge_segments.assert_not_called()
    speech_service.generate_audio_stream.assert_not_called()


def test_multi_preserves_custom_ipa_numbers_and_normalized_text():
    text = "12 Привет, [two people](/həlˈoʊ/) [pause:0.2s] cafe\u0301s français 😊!"
    spans = prepare_language_spans(
        text, "multi", "a", NormalizationOptions(remove_emoji=True)
    )
    assert "".join(span.text for span in spans) == text.replace("cafe\u0301s", "cafés")
    assert any("[two people](/həlˈoʊ/)" in span.text for span in spans)
    assert spans[0].lang_code == "espeak:ru"

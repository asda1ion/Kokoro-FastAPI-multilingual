"""Prepare language spans without loading speech models or generating audio."""

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import regex
from lingua import Language, LanguageDetectorBuilder
from unicode_segmentation_rs import unicode_sentences

from ...core import espeak
from ...core.config import settings
from ...core.languages import NATIVE_LANGUAGES, canonical_language
from ...structures.schemas import NormalizationOptions
from .phoneme_bridge import BridgeChunk, mapping_metadata
from .text_processor import CUSTOM_PHONEMES, PAUSE_TAG_PATTERN, split_by_voice

KNOWN_LETTERS = regex.compile(
    r"[\p{Latin}\p{Cyrillic}\p{Arabic}\p{Armenian}\p{Bengali}"
    r"\p{Devanagari}\p{Georgian}\p{Greek}\p{Gujarati}\p{Gurmukhi}"
    r"\p{Han}\p{Hangul}\p{Hebrew}\p{Hiragana}\p{Katakana}"
    r"\p{Tamil}\p{Telugu}\p{Thai}]"
)


@dataclass(frozen=True)
class LanguageSpan:
    text: str
    lang_code: str


@dataclass(frozen=True)
class SpeechSegment:
    voice: str
    rate: float
    span: LanguageSpan
    bridge_chunks: tuple[BridgeChunk, ...] | None = None


@dataclass(frozen=True)
class DetectedSpan:
    language: Language
    start_index: int
    end_index: int


def detect_spans(text: str, detector) -> list[DetectedSpan]:
    detected = detector.detect_multiple_languages_of(text)
    refined = []
    for index, result in enumerate(detected):
        end = (
            detected[index + 1].start_index if index + 1 < len(detected) else len(text)
        )
        fragment = text[result.start_index : end]
        sentences = list(unicode_sentences(fragment))
        if len(sentences) <= 1:
            refined.append(result)
            continue
        # Lingua can merge neighboring sentences that use different languages.
        cursor = 0
        for sentence in sentences:
            offset = fragment.index(sentence, cursor)
            cursor = offset + len(sentence)
            items = detector.detect_multiple_languages_of(sentence)
            if not items and regex.search(r"\p{L}", sentence):
                raise ValueError(
                    "Cannot identify the language; specify an explicit lang_code."
                )
            for item in items:
                refined.append(
                    DetectedSpan(
                        item.language,
                        result.start_index + offset + item.start_index,
                        result.start_index + offset + item.end_index,
                    )
                )
    return refined


@lru_cache(maxsize=1)
def get_detector():
    return LanguageDetectorBuilder.from_all_languages().build()


def language_capabilities() -> dict:
    inventory = espeak.installed_languages()
    detector_languages = sorted(
        language.iso_code_639_1.name.lower() for language in Language.all()
    )
    supported = []
    for language in detector_languages:
        try:
            canonical_language(language, allow_auto=False)
            supported.append(language)
        except ValueError:
            pass
    return {
        "protocol_version": 1,
        "native": [
            {"lang_code": code, "language": locale, "name": name}
            for code, (locale, name) in NATIVE_LANGUAGES.items()
        ],
        "auto_detection": {
            "engine": "lingua",
            "offline": True,
            "languages": detector_languages,
            "speech_languages": supported,
        },
        "experimental": [
            {
                "lang_code": "espeak:" + code,
                "language": code,
                "name": name,
                "experimental": True,
                "manual": True,
            }
            for code, name in sorted(inventory.items())
        ],
        "phoneme_bridge": {
            "enabled": settings.phoneme_bridge,
            "available": bool(inventory),
            "mapping": mapping_metadata(),
        },
    }


def _detection_text(text: str, options: NormalizationOptions) -> str:
    text = PAUSE_TAG_PATTERN.sub(lambda match: " " * len(match[0]), text)
    text = CUSTOM_PHONEMES.sub(
        lambda match: match[0][1 : match[0].index("]")].ljust(len(match[0])), text
    )
    if options.remove_emoji:
        # Mask rather than delete so Lingua's indices still address the original text.
        from .text_processor import EMOJI_PATTERN

        text = EMOJI_PATTERN.sub(lambda match: " " * len(match[0]), text)
    return text


def _native_route(language: str, fallback: str, bridge: bool) -> str:
    if language == "en" and fallback == "b":
        return "b"
    try:
        route = canonical_language(language, allow_auto=False)
        if route.startswith("espeak:") and not bridge:
            raise ValueError(
                "phoneme bridge is disabled; choose an explicit native lang_code."
            )
        return route
    except ValueError as error:
        if "disabled" in str(error):
            raise
        raise ValueError(
            f"Detected language {language!r} has no installed speech route; "
            "check /v1/audio/languages or choose an explicit native lang_code."
        ) from None


def prepare_language_spans(
    text: str,
    lang_code: str,
    fallback: str,
    options: NormalizationOptions,
    phoneme_bridge: bool = True,
) -> list[LanguageSpan]:
    code = canonical_language(lang_code)
    if code != "auto":
        if code.startswith("espeak:") and not (
            phoneme_bridge and settings.phoneme_bridge
        ):
            raise ValueError("phoneme bridge is disabled; choose a native lang_code.")
        return [LanguageSpan(text, code)]
    fallback = canonical_language(fallback, allow_auto=False)
    text = unicodedata.normalize("NFC", text)
    probe = _detection_text(text, options)
    letters = regex.findall(r"\p{L}", probe)
    if any(not KNOWN_LETTERS.fullmatch(letter) for letter in letters):
        raise ValueError("Cannot identify this script; specify an explicit lang_code.")
    if not letters:
        return [LanguageSpan(text, fallback)]

    detector = get_detector()
    detections = detect_spans(probe, detector)
    if not detections:
        raise ValueError("Cannot identify the language; specify an explicit lang_code.")
    candidates = []
    for detection in detections:
        fragment = probe[detection.start_index : detection.end_index]
        language = detection.language.iso_code_639_1.name.lower()
        confidence = detector.compute_language_confidence_values(fragment)
        ambiguous = (
            len(regex.findall(r"\p{L}", fragment)) <= 12
            and len(confidence) > 1
            and confidence[0].value - confidence[1].value < 0.15
        )
        candidates.append(None if ambiguous else language)

    spans = []
    start = 0
    for index, language in enumerate(candidates):
        if language is None:
            neighbors = sorted(
                (abs(other - index), other, candidate)
                for other, candidate in enumerate(candidates)
                if candidate is not None
            )
            language = (
                neighbors[0][2]
                if neighbors
                else NATIVE_LANGUAGES[fallback][0].split("-")[0]
            )
        route = _native_route(
            language, fallback, phoneme_bridge and settings.phoneme_bridge
        )
        end = (
            detections[index + 1].start_index
            if index + 1 < len(detections)
            else len(text)
        )
        fragment = text[start:end]
        if spans and spans[-1].lang_code == route:
            spans[-1] = LanguageSpan(spans[-1].text + fragment, route)
        else:
            spans.append(LanguageSpan(fragment, route))
        start = end
    return spans


def prepare_speech_segments(
    text: str,
    voice: str,
    lang_code: str | None,
    normalization_options: NormalizationOptions | None = None,
    allow_voice_tags: bool = False,
    phoneme_bridge: bool = True,
) -> list[SpeechSegment]:
    options = normalization_options or NormalizationOptions()
    segments = split_by_voice(text, voice) if allow_voice_tags else [(voice, 1.0, text)]
    prepared = []
    for speaker, rate, fragment in segments:
        language = lang_code or settings.default_voice_code or speaker[:1].lower()
        fallback = speaker[:1].lower()
        if fallback not in NATIVE_LANGUAGES:
            fallback = settings.default_voice_code or "a"
            if fallback == "auto":
                fallback = "a"
        for span in prepare_language_spans(
            fragment, language, fallback, options, phoneme_bridge
        ):
            prepared.append(SpeechSegment(speaker, rate, span))
    return prepared

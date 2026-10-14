"""Prepare language spans without loading speech models or generating audio."""

import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

import regex
from lingua import Language, LanguageDetectorBuilder
from unicode_segmentation_rs import split_word_bounds, unicode_sentences

from ...core import espeak
from ...core.config import settings
from ...core.languages import NATIVE_LANGUAGES, canonical_language
from ...structures.schemas import NormalizationOptions
from .phoneme_bridge import BridgeChunk, mapping_metadata
from .text_processor import CUSTOM_PHONEMES, PAUSE_TAG_PATTERN, split_by_voice

KNOWN_LETTERS = regex.compile(
    "|".join(
        rf"(?P<{script}>\p{{{script}}})"
        for script in (
            "Latin",
            "Cyrillic",
            "Arabic",
            "Armenian",
            "Bengali",
            "Devanagari",
            "Georgian",
            "Greek",
            "Gujarati",
            "Gurmukhi",
            "Han",
            "Hangul",
            "Hebrew",
            "Hiragana",
            "Katakana",
            "Tamil",
            "Telugu",
            "Thai",
        )
    )
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


def _script(text: str) -> str | None:
    match = KNOWN_LETTERS.search(text)
    if match is None:
        return None
    script = match.lastgroup
    return "CJK" if script in {"Han", "Hiragana", "Katakana"} else script


def _words(text: str, protected=()):
    words = []

    def append(start, end, word):
        script = _script(word)
        if script is None:
            return
        if (
            words
            and script == "CJK"
            and words[-1][3] == script
            and words[-1][1] == start
        ):
            previous = words.pop()
            words.append((previous[0], end, previous[2] + word, script))
        else:
            words.append((start, end, word, script))

    cursor = 0
    for start, end in [*protected, (len(text), len(text))]:
        for word in split_word_bounds(text[cursor:start]):
            begin, script = cursor, None
            for cluster in regex.findall(r"\X", word):
                next_script = _script(cluster)
                if (
                    script is not None
                    and next_script is not None
                    and script != next_script
                ):
                    append(begin, cursor, text[begin:cursor])
                    begin = cursor
                script = next_script or script
                cursor += len(cluster)
            append(begin, cursor, text[begin:cursor])
        if end > start:
            # A custom IPA unit must never be split between pronunciation routes.
            append(start, end, text[start:end].strip())
        cursor = end
    return words


def _confident(scores) -> bool:
    return len(scores) == 1 or scores[0].value - scores[1].value >= 0.15


def detect_spans(
    text: str, detector, fallback: str, protected=()
) -> list[DetectedSpan]:
    words = _words(text, protected)
    cache = {}

    def scores(fragment):
        if fragment not in cache:
            cache[fragment] = [
                item
                for item in detector.compute_language_confidence_values(fragment)
                if item.value > 0
            ]
        if not cache[fragment]:
            raise ValueError(
                "Cannot identify the language; specify an explicit lang_code."
            )
        return cache[fragment]

    fallback_language = NATIVE_LANGUAGES[fallback][0].split("-")[0]
    detected = []
    cursor = word_index = 0
    for sentence in unicode_sentences(text):
        start = text.index(sentence, cursor)
        cursor = start + len(sentence)
        begin = word_index
        while word_index < len(words) and words[word_index][0] < cursor:
            word_index += 1
        sentence_words = words[begin:word_index]
        candidates = [scores(word[2]) for word in sentence_words]
        anchors = {}
        for index, values in enumerate(candidates):
            if _confident(values):
                anchors.setdefault(values[0].language, []).append(index)
        contexts = {}
        for _, group in groupby(enumerate(sentence_words), key=lambda item: item[1][3]):
            run = list(group)
            bounds = (run[0][1][0], run[-1][1][1])
            contexts.update((index, bounds) for index, _ in run)
        for index, (word, values) in enumerate(zip(sentence_words, candidates)):
            language = values[0].language
            if not _confident(values):
                compatible = {item.language for item in values}
                neighbors = []
                for candidate in compatible:
                    positions = anchors.get(candidate, [])
                    insertion = bisect_left(positions, index)
                    for position in positions[max(0, insertion - 1) : insertion + 1]:
                        neighbors.append((abs(position - index), position, candidate))
                if neighbors:
                    language = min(neighbors, key=lambda item: item[:2])[2]
                else:
                    left, right = contexts[index]
                    context = scores(text[left:right].strip())
                    if _confident(context) and context[0].language in compatible:
                        language = context[0].language
                    else:
                        language = next(
                            (
                                item.language
                                for item in values
                                if item.language.iso_code_639_1.name.lower()
                                == fallback_language
                            ),
                            language,
                        )
            detected.append(DetectedSpan(language, word[0], word[1]))
    return detected


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
    detections = detect_spans(
        probe,
        detector,
        fallback,
        [(match.start(), match.end()) for match in CUSTOM_PHONEMES.finditer(text)],
    )
    if not detections:
        raise ValueError("Cannot identify the language; specify an explicit lang_code.")
    spans = []
    start = 0
    for index, detection in enumerate(detections):
        language = detection.language.iso_code_639_1.name.lower()
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
            if fallback in {"auto", "multi"}:
                fallback = "a"
        for span in prepare_language_spans(
            fragment, language, fallback, options, phoneme_bridge
        ):
            prepared.append(SpeechSegment(speaker, rate, span))
    return prepared

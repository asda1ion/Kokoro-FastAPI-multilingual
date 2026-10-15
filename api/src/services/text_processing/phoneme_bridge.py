"""Versioned eSpeak IPA approximations constrained by the loaded model."""

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

import regex

from ...core.espeak import espeak_language, initialize_espeak
from ...structures.schemas import NormalizationOptions
from .text_processor import CUSTOM_PHONEMES, PAUSE_TAG_PATTERN, join_lines, strip_emoji

_PHONEMIZE_LOCK = Lock()

MAPPING_VERSION = "espeak-kokoro-ipa-v2"
SUBSTITUTIONS = {
    "ɪ^": "ʲ",
    "d͡ʒ": "ʤ",
    "t͡ʃ": "ʧ",
    "d͡z": "ʣ",
    "t͡s": "ʦ",
    "d͜ʒ": "ʤ",
    "t͜ʃ": "ʧ",
    "d͜z": "ʣ",
    "t͜s": "ʦ",
    "g": "ɡ",
    "ɫ": "l",
    "ɭ": "l",
    "ɬ": "l",
    "ɮ": "z",
    "ʐ": "ʒ",
    "ʑ": "ʒ",
    "ɦ": "h",
    "ħ": "h",
    "ʕ": "ʔ",
    "ʜ": "h",
    "ʢ": "ʔ",
    "ɢ": "ɡ",
    "ɠ": "ɡ",
    "ɓ": "b",
    "ɗ": "d",
    "ʄ": "ɟ",
    "ʛ": "ɡ",
    "ɺ": "ɾ",
    "ɱ": "m",
    "ɰ": "w",
    "ɧ": "x",
    "ɞ": "ə",
    "ɘ": "ə",
    "ɵ": "ø",
    "ɶ": "œ",
    "ɒ": "ɔ",
    "ʉ": "u",
    "ɿ": "ɨ",
    "ʅ": "ɨ",
    "ʲ": "j",
    "ʷ": "w",
    "ˠ": "ɣ",
    "ˤ": "ʔ",
    "˞": "ɹ",
    "ˑ": "ː",
    "˥": "↗",
    "˦": "↗",
    "˧": "→",
    "˨": "↘",
    "˩": "↓",
    "1": "↓",
    "2": "↘",
    "3": "→",
    "4": "↗",
    "5": "↗",
    "6": "↓",
    "7": "↗",
    "8": "↘",
    "9": "→",
    "-": " ",
    "\n": " ",
    "\t": " ",
    "'": '"',
    "’": '"',
    "‘": '"',
    "«": "“",
    "»": "”",
    "，": ",",
    "。": ".",
    "؟": "?",
    "،": ",",
    "؛": ";",
    "͡": "",
    "͜": "",
}
# Articulation marks retain the segment; syllabicity and rhoticity add a vowel/r.
ARTICULATION_MARKS = frozenset("̪̺̻̟̠̞̝̘̙̯̥̬̆̚")


def mapping_metadata() -> dict:
    payload = {
        "substitutions": SUBSTITUTIONS,
        "articulation_marks": sorted(ARTICULATION_MARKS),
    }
    return {
        "version": MAPPING_VERSION,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "target": "loaded KModel.vocab from model config",
    }


@dataclass(frozen=True)
class MappedPhonemes:
    phonemes: str
    substitutions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BridgeChunk:
    text: str
    phonemes: str = ""
    pause: float | None = None
    substitutions: tuple[tuple[str, str], ...] = ()


def map_ipa(ipa: str, vocab: dict[str, int]) -> MappedPhonemes:
    ipa = "".join(
        cluster if cluster in vocab else unicodedata.normalize("NFD", cluster)
        for cluster in regex.findall(r"\X", ipa)
    )
    rules = sorted(SUBSTITUTIONS, key=len, reverse=True)
    output, substitutions = [], []
    index = 0
    while index < len(ipa):
        matched = next(
            (key for key in rules if len(key) > 1 and ipa.startswith(key, index)), None
        )
        symbol = matched or ipa[index]
        if matched:
            replacement = SUBSTITUTIONS[symbol]
        elif symbol in vocab:
            replacement = symbol
        elif symbol in ARTICULATION_MARKS and output:
            replacement = ""
        elif symbol == "̩" and output:
            previous = output.pop()
            replacement = "ə" + previous
        elif symbol == "̰" and output:
            replacement = "ʔ"
        else:
            replacement = SUBSTITUTIONS.get(symbol)
        if replacement is None or any(char not in vocab for char in replacement):
            points = " ".join(f"U+{ord(char):04X}" for char in symbol)
            raise ValueError(
                f"Unmapped eSpeak phoneme ({points}) for {MAPPING_VERSION}; "
                "choose a native language or report these code points for a mapping update."
            )
        if replacement != symbol:
            substitutions.append((symbol, replacement))
        output.append(replacement)
        index += len(symbol)
    return MappedPhonemes("".join(output), tuple(substitutions))


@lru_cache(maxsize=160)
def get_backend(language: str):
    language = espeak_language(language)
    try:
        from phonemizer.backend import EspeakBackend

        initialize_espeak()
        return EspeakBackend(
            language=language,
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            words_mismatch="ignore",
        )
    except (ImportError, OSError, RuntimeError):
        raise ValueError(
            "eSpeak is unavailable; install its bundled library/data in the server environment."
        ) from None


def phonemize(text: str, language: str) -> str:
    try:
        with _PHONEMIZE_LOCK:
            result = get_backend(language).phonemize([text], strip=True)
    except (OSError, RuntimeError):
        raise ValueError(
            "eSpeak pronunciation failed; verify the installed language data."
        ) from None
    if not result or (regex.search(r"\p{L}|\p{N}", text) and not result[0].strip()):
        raise ValueError(
            "eSpeak produced no phonemes; choose a native language or check the language data."
        )
    return result[0]


def split_phonemes(phonemes: str, limit: int):
    while len(phonemes) > limit:
        end = phonemes.rfind(" ", 1, limit + 1)
        if end < 1:
            end = limit
            while end > 1 and (
                (unicodedata.combining(phonemes[end]) or phonemes[end] in "ːʲʰʷ")
                or phonemes[end - 1] in "ˈˌ"
            ):
                end -= 1
        yield phonemes[:end]
        phonemes = phonemes[end:]
    if phonemes:
        yield phonemes


def split_text(text: str, limit: int = 256):
    clusters = regex.findall(r"\X", text)
    start = 0
    while start < len(clusters):
        end = min(start + limit, len(clusters))
        if end < len(clusters):
            boundary = next(
                (
                    index + 1
                    for index in range(end - 1, start, -1)
                    if clusters[index].isspace()
                ),
                None,
            )
            if boundary is not None:
                end = boundary
        yield "".join(clusters[start:end])
        start = end


def prepare_chunks(
    text: str,
    language: str,
    options: NormalizationOptions,
    vocab: dict[str, int],
    limit: int,
) -> tuple[BridgeChunk, ...]:
    if limit < 2:
        raise ValueError(
            "Loaded model token limit is too small for the phoneme bridge."
        )
    chunks = []
    for index, part in enumerate(PAUSE_TAG_PATTERN.split(text)):
        if index % 2:
            chunks.append(BridgeChunk("", pause=float(part)))
            continue
        if options.remove_emoji:
            part = strip_emoji(part)
        part = unicodedata.normalize("NFC", join_lines(part))
        # Bound eSpeak calls while retaining graphemes and custom IPA units.
        units = []
        for fragment in CUSTOM_PHONEMES.split(part):
            if CUSTOM_PHONEMES.fullmatch(fragment):
                units.append((fragment, fragment[fragment.index("(/") + 2 : -2]))
            else:
                units.extend((piece, None) for piece in split_text(fragment))
        for fragment, custom in units:
            if not fragment.strip():
                continue
            mapped = map_ipa(
                custom if custom is not None else phonemize(fragment, language), vocab
            )
            if not mapped.phonemes.strip():
                continue
            pieces = list(split_phonemes(mapped.phonemes, limit))
            for piece_index, piece in enumerate(pieces):
                chunks.append(
                    BridgeChunk(
                        fragment if piece_index == 0 else "",
                        piece,
                        substitutions=mapped.substitutions,
                    )
                )
    return tuple(chunks)

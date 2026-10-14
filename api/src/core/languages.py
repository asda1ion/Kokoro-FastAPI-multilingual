"""Kokoro native aliases take priority over installed eSpeak routes."""

import re

NATIVE_LANGUAGES = {
    "a": ("en-US", "American English"),
    "b": ("en-GB", "British English"),
    "e": ("es", "Spanish"),
    "f": ("fr-FR", "French"),
    "h": ("hi", "Hindi"),
    "i": ("it", "Italian"),
    "p": ("pt-BR", "Brazilian Portuguese"),
    "j": ("ja", "Japanese"),
    "z": ("zh", "Mandarin Chinese"),
}

ALIASES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
    "zh-cn": "z",
    "zh-tw": "z",
    "zh-hans": "z",
    "zh-hant": "z",
}


def canonical_language(value: str, *, allow_auto: bool = True) -> str:
    value = value.strip().lower().replace("_", "-")
    code = ALIASES.get(value, value)
    if code == value and re.fullmatch(
        r"[a-z]{2,3}(?:-[a-z]{4})?(?:-[a-z]{2}|-\d{3})", value
    ):
        code = ALIASES.get(value.split("-")[0], value)
    if code in NATIVE_LANGUAGES or (allow_auto and code == "auto"):
        return code
    from .espeak import espeak_language

    return "espeak:" + espeak_language(value)

"""Native Kokoro pronunciation without speech model weights."""

from abc import ABC, abstractmethod

from ...core.config import settings
from ...core.languages import canonical_language

phonemizers = {}


class PhonemizerBackend(ABC):
    @abstractmethod
    def phonemize(self, text: str) -> str:
        pass


class NativeBackend(PhonemizerBackend):
    def __init__(self, language: str):
        from kokoro import KPipeline

        from ...core.espeak import initialize_espeak

        initialize_espeak()
        self.pipeline = KPipeline(
            lang_code=language, model=False, repo_id=settings.model_repo_id
        )

    def phonemize(self, text: str) -> str:
        phonemes, _ = self.pipeline.g2p(text)
        return phonemes or ""


def create_phonemizer(language: str = "a") -> PhonemizerBackend:
    return NativeBackend(canonical_language(language, allow_auto=False))


def phonemize(text: str, language: str = "a") -> str:
    language = canonical_language(language, allow_auto=False)
    if language not in phonemizers:
        phonemizers[language] = create_phonemizer(language)
    return phonemizers[language].phonemize(text.strip()).strip()

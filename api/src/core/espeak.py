"""Installed eSpeak routes, initialized independently of native G2P warmup."""

import os
import re
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import RLock

_LOCK = RLock()
_DATA_COPY = None


@lru_cache(maxsize=1)
def _data_path() -> Path:
    import espeakng_loader

    source = Path(
        os.environ.get("ESPEAK_DATA_PATH") or espeakng_loader.get_data_path()
    ).resolve()
    if not (source / "phontab").is_file():
        raise OSError("eSpeak data is missing")
    if sys.platform == "darwin":
        # Bundled macOS eSpeak can exit on deep checkout paths. Keep real data nearby.
        global _DATA_COPY
        _DATA_COPY = tempfile.TemporaryDirectory(prefix="kokoro-espeak-")
        destination = Path(_DATA_COPY.name) / "data"
        shutil.copytree(source, destination)
        return destination
    return source


def initialize_espeak() -> None:
    import espeakng_loader

    # Misaki sets wrapper paths at import time; apply our paths after that import.
    from misaki import espeak  # noqa: F401
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    with _LOCK:
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(str(_data_path()))


@lru_cache(maxsize=1)
def installed_languages() -> dict[str, str]:
    try:
        from phonemizer.backend import EspeakBackend

        with _LOCK:
            initialize_espeak()
            return dict(EspeakBackend.supported_languages())
    except (ImportError, OSError, RuntimeError):
        return {}


def espeak_language(value: str) -> str:
    value = value.removeprefix("espeak:").lower().replace("_", "-")
    inventory = {code.lower(): code for code in installed_languages()}
    if value in inventory:
        return inventory[value]
    primary = value.split("-")[0]
    if (
        re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{4})?(?:-[a-z]{2}|-\d{3})", value)
        and primary in inventory
    ):
        return inventory[primary]
    raise ValueError(
        "Unknown or unavailable eSpeak language; check /v1/audio/languages "
        "and verify bundled eSpeak library/data in the server environment."
    )

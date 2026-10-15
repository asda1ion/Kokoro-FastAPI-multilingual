"""Provision native Japanese dictionary data before starting the Mac server."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REQUIRED = ("mecabrc", "dicrc", "sys.dic", "unk.dic", "matrix.bin", "char.bin")


def verify_dictionary(directory: Path) -> None:
    from fugashi import Tagger

    for name in _REQUIRED:
        path = directory / name
        if not path.is_file() or (
            path.suffix in {".dic", ".bin"} and not path.stat().st_size
        ):
            raise RuntimeError(f"Missing or empty UniDic file: {path}")
    tagger = Tagger(f'-r "{directory / "mecabrc"}" -d "{directory}"')
    words = tagger("日本語")
    if not words or not all(word.feature.pron for word in words):
        raise RuntimeError(f"UniDic Japanese pronunciation check failed: {directory}")


def ensure_dictionary() -> Path:
    import unidic

    directory = Path(unidic.DICDIR).resolve()
    try:
        verify_dictionary(directory)
    except RuntimeError as error:
        default = Path(unidic.__file__).resolve().parent / "dicdir"
        if directory != default or all(
            (directory / name).exists() for name in _REQUIRED
        ):
            raise RuntimeError(
                f"Cannot use UniDic at {directory}; existing dictionary preserved. "
                "Repair its full UniDic data before restarting."
            ) from error

        print(f"Provisioning full UniDic at {directory}", flush=True)
        # The upstream downloader replaces dicdir. Stage it to preserve existing data.
        with tempfile.TemporaryDirectory(prefix="kokoro-unidic-") as temporary:
            package = Path(temporary) / "unidic"
            package.mkdir()
            for source in Path(unidic.__file__).resolve().parent.glob("*.py"):
                shutil.copy2(source, package / source.name)
            subprocess.run(
                [sys.executable, "-m", "unidic", "download"],
                cwd=temporary,
                env=dict(os.environ, PYTHONPATH=temporary),
                check=True,
            )
            downloaded = package / "dicdir"
            verify_dictionary(downloaded)
            directory.mkdir(parents=True, exist_ok=True)
            for source in downloaded.iterdir():
                destination = directory / source.name
                if source.is_file() and not destination.exists():
                    shutil.copy2(source, destination)
        verify_dictionary(directory)
    print(f"Verified full UniDic at {directory}", flush=True)
    return directory


if __name__ == "__main__":
    ensure_dictionary()

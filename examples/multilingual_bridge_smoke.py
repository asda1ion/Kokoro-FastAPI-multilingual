"""Bounded synthetic validation through the native macOS launcher."""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests
import soundfile as sf
import torch

from api.src.inference.kokoro_v1 import KokoroV1
from api.src.services.text_processing.language_routing import prepare_speech_segments
from api.src.services.text_processing.phoneme_bridge import (
    mapping_metadata,
    prepare_chunks,
)
from api.src.structures.schemas import NormalizationOptions

CORPUS = [
    ("native_japanese", "j", "こんにちは。これは日本語の音声テストです。"),
    (
        "mixed_japanese_english_russian",
        "multi",
        "日本語の音声テストです。 This is an English speech test. Привет, это проверка русской речи.",
    ),
    ("russian", "ru", "Привет, мир. Это проверка русской речи."),
    (
        "mixed",
        "auto",
        "Это полное предложение на русском языке. This is a complete sentence in English.",
    ),
    ("ukrainian", "uk", "Привіт, світе. Це перевірка української мови."),
    ("german", "de", "Guten Tag. Dies ist eine deutsche Sprachprobe."),
    ("arabic", "ar", "مرحبا بالعالم. هذه تجربة باللغة العربية."),
    ("vietnamese", "vi", "Xin chào thế giới. Đây là thử nghiệm tiếng Việt."),
    ("native_english", "a", "This is a bounded native English speech sample."),
    ("native_french", "f", "Bonjour tout le monde. Ceci est une phrase française."),
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(root):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    files = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in sorted(set(tracked + untracked))
        if (root / path).is_file()
    }
    return {
        "tested_head": head,
        "candidate_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "candidate_files_sha256": files,
    }


def run(args):
    root = Path.cwd().resolve()
    if platform.system() != "Darwin" or not torch.backends.mps.is_available():
        raise RuntimeError("This validation requires available native Apple MPS.")
    output = Path(args.output).resolve()
    output.relative_to(root / ".orbit" / "tmp")
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        HOST="127.0.0.1",
        PORT=str(port),
        API_LOG_LEVEL="DEBUG",
        MODEL_AUTO_UNLOAD_TIMEOUT_SECONDS="0",
    )
    config = json.loads((root / "api/src/models/v1_0/config.json").read_text())
    vocab = config["vocab"]
    # This config mirror must match the live loaded-vocabulary hash in server.log.
    backend = KokoroV1()
    backend._model = SimpleNamespace(
        vocab=vocab, context_length=config["plbert"]["max_position_embeddings"]
    )
    vocab_hash = hashlib.sha256(
        json.dumps(vocab, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    from api.src.core.espeak import initialize_espeak
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    initialize_espeak()
    espeak_version = EspeakWrapper().version
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        **source_provenance(root),
        "launch": {
            "command": "./start-gpu_mac.sh",
            "cwd": str(root),
            "host": "127.0.0.1",
            "port": port,
        },
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "espeak": espeak_version,
            "mps_available": True,
        },
        "mapping": mapping_metadata(),
        "model_vocab_sha256": vocab_hash,
        "model_sha256": hashlib.sha256(
            (root / "api/src/models/v1_0/kokoro-v1_0.pth").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (root / "api/src/models/v1_0/config.json").read_bytes()
        ).hexdigest(),
        "voice": "af_bella",
        "voice_sha256": hashlib.sha256(
            (root / "api/src/voices/v1_0/af_bella.pt").read_bytes()
        ).hexdigest(),
        "speed": 1.1,
        "samples": [],
        "limitations": [
            "No intelligibility or WER claim; no listening/physical-device check.",
            "Experimental IPA approximations lose articulation and tone contrasts.",
            "Timings include local HTTP/encoding overhead; first sample includes bridge setup.",
            "Kokoro MPS uses the launcher's CPU fallback for unsupported operations.",
        ],
    }
    server = None
    try:
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(
                ["./start-gpu_mac.sh"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 3600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        "Native launcher exited before readiness; inspect server.log."
                    )
                try:
                    response = requests.get(base + "/health", timeout=2)
                    if response.ok:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Native startup exceeded 3600 seconds.")
                time.sleep(1)
            capabilities = requests.get(base + "/v1/audio/languages", timeout=30).json()
            import unidic
            from fugashi import Tagger

            dictionary = Path(unidic.DICDIR).resolve()
            report["dictionary"] = {
                "path": str(dictionary),
                "version": (dictionary / "version").read_text().strip(),
                "fugashi": Tagger().dictionary_info,
                "files_sha256": {
                    path.name: file_sha256(path)
                    for path in sorted(dictionary.iterdir())
                    if path.is_file()
                },
            }
            (output / "capabilities.json").write_text(
                json.dumps(capabilities, indent=2, ensure_ascii=False)
            )
            if capabilities["phoneme_bridge"]["mapping"] != report["mapping"]:
                raise RuntimeError(
                    "Live mapping provenance differs from the candidate."
                )
            report["manual_routes"] = len(capabilities["experimental"])
            options = NormalizationOptions(normalize=False)
            for name, language, text in CORPUS:
                log_start = (output / "server.log").stat().st_size
                segments = prepare_speech_segments(text, "af_bella", language, options)
                routing = []
                for segment in segments:
                    route = {
                        "language": segment.span.lang_code,
                        "text": segment.span.text,
                    }
                    if segment.span.lang_code.startswith("espeak:"):
                        prepared = prepare_chunks(
                            segment.span.text,
                            segment.span.lang_code,
                            options,
                            backend.phoneme_vocabulary,
                            backend.phoneme_limit,
                        )
                        route["chunks"] = [
                            {
                                "phonemes": chunk.phonemes,
                                "substitutions": chunk.substitutions,
                                "tokens": len(chunk.phonemes),
                            }
                            for chunk in prepared
                        ]
                    routing.append(route)
                started = time.perf_counter()
                payload = {
                    "input": text,
                    "voice": "af_bella",
                    "speed": 1.1,
                    "lang_code": language,
                    "stream": True,
                    "response_format": "wav",
                    "normalization_options": {"normalize": False},
                }
                response = requests.post(
                    base + "/v1/audio/speech",
                    json=payload,
                    stream=True,
                    timeout=180,
                )
                response.raise_for_status()
                pieces, first_byte = [], None
                for piece in response.iter_content(4096):
                    if first_byte is None:
                        first_byte = time.perf_counter() - started
                    pieces.append(piece)
                    if sum(map(len, pieces)) > 2_000_000:
                        raise RuntimeError("Bounded sample exceeded 2 MB.")
                elapsed = time.perf_counter() - started
                data = b"".join(pieces)
                audio, rate = sf.read(io.BytesIO(data), dtype="float32")
                if (
                    rate != 24000
                    or audio.ndim != 1
                    or len(audio) < 240
                    or not np.isfinite(audio).all()
                    or np.max(np.abs(audio)) == 0
                ):
                    raise RuntimeError(f"Invalid or empty speech sample: {name}")
                (output / f"{name}.wav").write_bytes(data)
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                pcm_source = "decoded_wav"
                if name in ("native_japanese", "mixed_japanese_english_russian"):
                    response = requests.post(
                        base + "/v1/audio/speech",
                        json=dict(payload, response_format="pcm"),
                        stream=True,
                        timeout=180,
                    )
                    response.raise_for_status()
                    pieces = []
                    for piece in response.iter_content(4096):
                        pieces.append(piece)
                        if sum(map(len, pieces)) > 2_000_000:
                            raise RuntimeError("Bounded PCM sample exceeded 2 MB.")
                    pcm = b"".join(pieces)
                    if len(pcm) % 2 or len(pcm) < 480:
                        raise RuntimeError(f"Invalid or empty PCM stream: {name}")
                    if not np.any(np.frombuffer(pcm, dtype="<i2")):
                        raise RuntimeError(f"Silent PCM stream: {name}")
                    pcm_source = "http_pcm_stream"
                (output / f"{name}.pcm").write_bytes(pcm)
                duration = len(audio) / rate
                live_sample_log = (
                    (output / "server.log").read_bytes()[log_start:].decode()
                )
                voice_paths = re.findall(
                    r"Using single voice path: (.+)", live_sample_log
                )
                voice_preserved = bool(voice_paths) and all(
                    Path(path.strip()).name == "af_bella.pt" for path in voice_paths
                )
                if not voice_preserved:
                    raise RuntimeError(f"Selected voice evidence absent: {name}")
                report["samples"].append(
                    {
                        "name": name,
                        "lang_code": language,
                        "selected_voice": "af_bella",
                        "selected_voice_verified_in_server_log": voice_preserved,
                        "live_voice_paths": voice_paths,
                        "input": text,
                        "routing": routing,
                        "wav": f"{name}.wav",
                        "pcm": f"{name}.pcm",
                        "pcm_source": pcm_source,
                        "pcm_bytes": len(pcm),
                        "pcm_frames": len(pcm) // 2,
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                        "bytes": len(data),
                        "frames": len(audio),
                        "sample_rate": rate,
                        "peak": float(np.max(np.abs(audio))),
                        "rms": float(np.sqrt(np.mean(audio**2))),
                        "clipping_fraction": float(np.mean(np.abs(audio) >= 0.999)),
                        "duration_seconds": duration,
                        "first_byte_seconds": first_byte,
                        "wall_seconds": elapsed,
                        "real_time_factor": elapsed / duration,
                        "wav_sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
            log.flush()
            server_log = (output / "server.log").read_text()
            if (
                "Loaded Kokoro model device=mps" not in server_log
                or f"vocab_sha256={vocab_hash}" not in server_log
            ):
                raise RuntimeError(
                    "Live loaded MPS/vocabulary provenance is absent or mismatched."
                )
            report["runtime"]["actual_model_device"] = re.search(
                r"Loaded Kokoro model device=(mps(?::\d+)?)", server_log
            )[1]
            report["model_vocab_verified_against_live_log"] = True
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=30)
        report["server_exit_code"] = server.returncode if server else None
        with socket.socket() as probe:
            report["owned_port_closed"] = probe.connect_ex(("127.0.0.1", port)) != 0
        report["clean_owned_shutdown"] = (
            server is not None
            and server.returncode == 0
            and report["owned_port_closed"]
            and "Application shutdown complete." in (output / "server.log").read_text()
        )
        if report.get("status") == "passed" and not report["clean_owned_shutdown"]:
            report["status"] = "failed"
            report["error"] = "Owned server did not shut down cleanly."
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
    if report["status"] != "passed":
        raise RuntimeError(report["error"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": len(report["samples"]),
                "report": str(output / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", required=True)
    parser.add_argument("--output", default=".orbit/tmp/mps-smoke")
    parser.add_argument("--run-id", default=os.environ.get("ORBIT_RUN_ID", "manual"))
    run(parser.parse_args())

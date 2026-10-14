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
        API_LOG_LEVEL="INFO",
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
            )
            deadline = time.monotonic() + 900
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
                    raise TimeoutError("Native startup exceeded 900 seconds.")
                time.sleep(1)
            capabilities = requests.get(base + "/v1/audio/languages", timeout=30).json()
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
                response = requests.post(
                    base + "/v1/audio/speech",
                    json={
                        "input": text,
                        "voice": "af_bella",
                        "speed": 1.1,
                        "lang_code": language,
                        "stream": True,
                        "response_format": "wav",
                        "normalization_options": {"normalize": False},
                    },
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
                (output / f"{name}.pcm").write_bytes(pcm)
                duration = len(audio) / rate
                report["samples"].append(
                    {
                        "name": name,
                        "lang_code": language,
                        "input": text,
                        "routing": routing,
                        "wav": f"{name}.wav",
                        "pcm": f"{name}.pcm",
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
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=30)
        report["server_exit_code"] = server.returncode if server else None
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
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

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.src.core import japanese


@pytest.mark.parametrize("existing", [False, True])
def test_launcher_installs_core_dependencies_preserves_extras(tmp_path, existing):
    shutil.copy2("start-gpu_mac.sh", tmp_path / "start-gpu_mac.sh")
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        "#!/bin/bash\n"
        'echo "$*" >> "$PWD/commands"\n'
        'if [ "$1" = sync ]; then mkdir -p .venv/bin; touch .venv/bin/python; chmod +x .venv/bin/python; fi\n'
        'if [ "$1 $2" = "pip install" ]; then touch .venv/new-core-dependency; fi\n'
    )
    uv.chmod(0o755)
    if existing:
        python = tmp_path / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch(mode=0o755)
        (tmp_path / ".venv/installed-extra").write_text("preserve")
    subprocess.run(
        ["bash", str(tmp_path / "start-gpu_mac.sh")],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PATH=str(tools) + os.pathsep + os.environ["PATH"],
            HOST="127.0.0.1",
            PORT="8911",
        ),
        check=True,
        timeout=30,
    )
    commands = (tmp_path / "commands").read_text().splitlines()
    assert any(
        command.startswith("pip install --python ") and command.endswith(" -e .")
        for command in commands
    )
    assert not any("--no-deps" in command for command in commands)
    assert (tmp_path / ".venv/new-core-dependency").exists()
    if existing:
        assert (tmp_path / ".venv/installed-extra").read_text() == "preserve"
        assert not any(command.startswith("sync") for command in commands)
    else:
        assert commands[0] == "sync --extra cpu"
    assert (
        commands[-1]
        == "run --no-sync uvicorn api.src.main:app --host 127.0.0.1 --port 8911"
    )
    dictionary = "run --no-sync python -m api.src.core.japanese"
    assert commands.index(dictionary) < commands.index(
        "run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0"
    )


def test_launcher_dictionary_failure_exits_before_serving(tmp_path):
    shutil.copy2("start-gpu_mac.sh", tmp_path / "start-gpu_mac.sh")
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        "#!/bin/bash\n"
        'echo "$*" >> "$PWD/commands"\n'
        'if [ "$1" = sync ]; then mkdir -p .venv/bin; touch .venv/bin/python; chmod +x .venv/bin/python; fi\n'
        'if [ "$*" = "run --no-sync python -m api.src.core.japanese" ]; then exit 7; fi\n'
        "exit 0\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(tmp_path / "start-gpu_mac.sh")],
        cwd=tmp_path,
        env=dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"]),
        timeout=30,
    )
    assert result.returncode == 7
    commands = (tmp_path / "commands").read_text()
    assert "uvicorn" not in commands
    assert "download_model.py" not in commands


@pytest.fixture
def dictionary_environment(tmp_path, monkeypatch):
    package = tmp_path / "unidic"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture package\n")
    directory = package / "dicdir"
    module = SimpleNamespace(
        __file__=str(package / "__init__.py"), DICDIR=str(directory)
    )
    monkeypatch.setitem(sys.modules, "unidic", module)
    calls = []

    def download(command, *, cwd, env, check):
        assert command == [sys.executable, "-m", "unidic", "download"]
        assert env["PYTHONPATH"] == cwd
        assert check
        assert (Path(cwd) / "unidic/__init__.py").is_file()
        calls.append(command)
        populate(Path(cwd) / "unidic/dicdir")

    def tagger(arguments):
        assert '-r "' in arguments and '-d "' in arguments
        return lambda text: [SimpleNamespace(feature=SimpleNamespace(pron="ニホンゴ"))]

    monkeypatch.setattr(japanese.subprocess, "run", download)
    monkeypatch.setitem(sys.modules, "fugashi", SimpleNamespace(Tagger=tagger))
    return module, directory, calls


def populate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in japanese._REQUIRED:
        (directory / name).write_bytes(b"dictionary fixture")


@pytest.mark.parametrize("partial", [False, True])
def test_unidic_missing_data_provisioned_and_reused(dictionary_environment, partial):
    _, directory, calls = dictionary_environment
    if partial:
        directory.mkdir()
        (directory / "mecabrc").write_text("custom configuration")
        (directory / "user.dic").write_bytes(b"preserve")
    assert japanese.ensure_dictionary() == directory
    assert japanese.ensure_dictionary() == directory
    assert len(calls) == 1
    if partial:
        assert (directory / "mecabrc").read_text() == "custom configuration"
        assert (directory / "user.dic").read_bytes() == b"preserve"


@pytest.mark.parametrize("custom", [False, True])
def test_unidic_valid_data_reused(dictionary_environment, tmp_path, custom):
    module, directory, calls = dictionary_environment
    if custom:
        directory = tmp_path / "explicit dictionary"
        module.DICDIR = str(directory)
    populate(directory)
    (directory / "mecabrc").write_text("")
    assert japanese.ensure_dictionary() == directory
    assert not calls


def test_unidic_invalid_custom_path_preserved(dictionary_environment, tmp_path):
    module, _, calls = dictionary_environment
    directory = tmp_path / "custom"
    directory.mkdir()
    (directory / "user.dic").write_bytes(b"preserve")
    module.DICDIR = str(directory)
    with pytest.raises(RuntimeError, match="existing dictionary preserved"):
        japanese.ensure_dictionary()
    assert not calls
    assert (directory / "user.dic").read_bytes() == b"preserve"


@pytest.mark.parametrize("failure", ["download", "incomplete", "pronunciation"])
def test_unidic_failed_provisioning_not_installed(
    dictionary_environment, monkeypatch, failure
):
    _, directory, _ = dictionary_environment
    if failure == "download":

        def download(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr(japanese.subprocess, "run", download)
        expected = subprocess.CalledProcessError
    elif failure == "incomplete":
        monkeypatch.setattr(japanese.subprocess, "run", lambda *args, **kwargs: None)
        expected = RuntimeError
    else:
        monkeypatch.setitem(
            sys.modules, "fugashi", SimpleNamespace(Tagger=lambda args: lambda text: [])
        )
        expected = RuntimeError
    with pytest.raises(expected):
        japanese.ensure_dictionary()
    assert not directory.exists()


def test_unidic_existing_corruption_not_overwritten(
    dictionary_environment, monkeypatch
):
    _, directory, calls = dictionary_environment
    populate(directory)

    def tagger(args):
        raise RuntimeError("corrupt dictionary")

    monkeypatch.setitem(sys.modules, "fugashi", SimpleNamespace(Tagger=tagger))
    with pytest.raises(RuntimeError, match="existing dictionary preserved"):
        japanese.ensure_dictionary()
    assert not calls
    assert (directory / "sys.dic").read_bytes() == b"dictionary fixture"

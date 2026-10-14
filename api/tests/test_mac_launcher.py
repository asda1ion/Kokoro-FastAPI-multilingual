import os
import shutil
import subprocess

import pytest


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

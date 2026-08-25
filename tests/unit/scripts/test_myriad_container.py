"""Behaviour tests for the Myriad container scripts (DSE-053).

These scripts only ever run on the cluster, where a mistake costs a queue wait and an A100
allocation rather than a red test. Three of them have already shipped a bug that only surfaced
there - a `--nv` on a login node, an `icc` inherited from the host module stack, a `serve_env.json`
field split across two lines - so each test below guards a defect that actually happened.

The cluster is faked with a stub `apptainer` on PATH that records how it was invoked and then runs
the payload in-process with ``APPTAINER_CONTAINER`` set, which is exactly what the real thing does
from the scripts' point of view. Nothing here needs Apptainer, a GPU, or a network.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts" / "myriad"
COMMON = SCRIPTS / "_common.sh"

STUB_APPTAINER = """#!/bin/bash
# Records the invocation, then runs the payload as the container would.
{
  echo "CALL $*"
} >>"$STUB_LOG"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    exec) shift ;;
    --nv) echo "NV" >>"$STUB_LOG"; shift ;;
    --bind) echo "BIND $2" >>"$STUB_LOG"; shift 2 ;;
    --pwd) echo "PWD $2" >>"$STUB_LOG"; shift 2 ;;
    *.sif) shift; args=("$@"); break ;;
    *) shift ;;
  esac
done
export APPTAINER_CONTAINER=1
exec "${args[@]}"
"""


class Cluster:
    """A fake Myriad: stub binaries on PATH, a Scratch tree, and an image that exists."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.bin = tmp_path / "bin"
        self.venv = tmp_path / "venv"
        self.log = tmp_path / "stub.log"
        self.sif = self.home / "Scratch" / "containers" / "precept-python311.sif"

        for d in (self.bin, self.sif.parent, self.venv / "bin"):
            d.mkdir(parents=True, exist_ok=True)
        self.sif.write_text("not really a squashfs")
        (self.venv / "bin" / "activate").write_text("export VENV_ACTIVE=yes\n")
        self._stub("apptainer", STUB_APPTAINER)
        # Myriad login nodes have coreutils the scripts call; macOS lacks sha256sum, so stub it
        # rather than making the suite depend on the developer's brew state.
        self._stub("sha256sum", '#!/bin/bash\necho "0000000000000000  $1"\n')
        # Multi-line, like the real thing: this is what makes `head -1` close the pipe early and
        # the producer die of SIGPIPE. A single-line stub would let the DSE-052 bug pass unseen,
        # and macOS has no `ldd` at all, so without this the guard only works on Linux.
        self._stub(
            "ldd",
            "#!/bin/bash\n"
            'echo "ldd (Debian GLIBC 2.41-12+deb13u3) 2.41"\n'
            'for i in $(seq 1 20000); do echo "padding line $i"; done\n',
        )
        # One line per GPU, like the real --query-gpu output: `gpu` joins them with paste.
        self._stub(
            "nvidia-smi",
            "#!/bin/bash\n"
            'case "$*" in\n'
            '  *driver_version*) echo "550.127.05" ;;\n'
            '  *) echo "NVIDIA A100-PCIE-40GB" ;;\n'
            "esac\n",
        )
        self.log.write_text("")

    def _stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)

    def env(self, **overrides: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "PATH": f"{self.bin}{os.pathsep}{env['PATH']}",
                "HOME": str(self.home),
                "VENV": str(self.venv),
                "REPO_ROOT": str(REPO_ROOT),
                "STUB_LOG": str(self.log),
                "APPTAINER_CACHEDIR": str(self.root / "apptainer"),
                # Absent by default, so enter_container takes the login-node branch.
                "NV_SENTINEL": str(self.root / "no-such-device"),
            }
        )
        env.update(overrides)
        return env

    def run(self, script: str, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *args],
            capture_output=True,
            text=True,
            env=self.env(**overrides),
            cwd=REPO_ROOT,
            timeout=60,
        )

    @property
    def calls(self) -> list[str]:
        return [ln for ln in self.log.read_text().splitlines() if ln.startswith("CALL ")]

    @property
    def trace(self) -> str:
        return self.log.read_text()


@pytest.fixture
def cluster(tmp_path: Path) -> Cluster:
    return Cluster(tmp_path)


def test_the_script_enters_the_container_exactly_once(cluster: Cluster) -> None:
    """Re-exec, not per-command wrapping.

    pilot.sh backgrounds serve.sh and traps ``$!`` so the server dies on the scheduler's wallclock
    SIGTERM. A second `apptainer exec` would put them in different process namespaces and the trap
    would name the wrapper rather than vLLM.
    """
    result = cluster.run("shell.sh", "-c", "echo inside")
    assert result.returncode == 0, result.stderr
    assert "inside" in result.stdout
    assert len(cluster.calls) == 1, cluster.trace


def test_no_nv_flag_without_a_gpu(cluster: Cluster) -> None:
    """`--nv` on a login node fails looking for driver libraries that are not there."""
    cluster.run("shell.sh", "-c", "true")
    assert "NV" not in cluster.trace.splitlines()


def test_nv_flag_when_a_gpu_is_present(cluster: Cluster) -> None:
    sentinel = cluster.root / "fake-nvidiactl"
    sentinel.write_text("")
    cluster.run("shell.sh", "-c", "true", NV_SENTINEL=str(sentinel))
    assert "NV" in cluster.trace.splitlines(), cluster.trace


def test_home_is_bound_resolved_and_the_working_directory_survives(cluster: Cluster) -> None:
    """$HOME is a symlink into /myriadfs on Myriad; a dropped cwd sends artefacts to /runs."""
    cluster.run("shell.sh", "-c", "true")
    resolved = cluster.home.resolve()
    assert f"BIND {resolved}:{cluster.home}" in cluster.trace, cluster.trace
    assert f"PWD {REPO_ROOT}" in cluster.trace, cluster.trace


def test_leaked_intel_compiler_is_overridden_inside_the_container(cluster: Cluster) -> None:
    """DSE-053: Myriad's default-modules stack exports CC=icc, and Apptainer passes it through.

    torch/Triton JIT-compile a CUDA support module at engine start reading CC verbatim, so a leaked
    `icc` killed vLLM minutes into startup with the A100 already allocated.
    """
    result = cluster.run(
        "shell.sh",
        "-c",
        'printf "CC=%s CXX=%s PYTHONPATH=%s\\n" "$CC" "$CXX" "${PYTHONPATH-unset}"',
        CC="icc",
        CXX="icpc",
        PYTHONPATH="/shared/ucl/apps/site-packages",
    )
    assert result.returncode == 0, result.stderr
    assert "CC=gcc CXX=g++ PYTHONPATH=unset" in result.stdout


def test_the_compiler_preflight_fails_loud_when_the_image_has_none(tmp_path: Path) -> None:
    """A missing compiler must be one line on a login node, not a Triton traceback on the GPU."""
    only_gcc = tmp_path / "bin"
    only_gcc.mkdir()
    (only_gcc / "gcc").write_text("#!/bin/sh\n")
    (only_gcc / "gcc").chmod(0o755)
    # container_toolchain uses only shell builtins, so PATH can be this bare.
    result = subprocess.run(
        ["bash", "-c", f'source "{COMMON}"; PATH="{only_gcc}"; container_toolchain'],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path)},
        timeout=30,
    )
    assert result.returncode == 1
    assert "no g++" in result.stderr
    assert "-slim" in result.stderr


def test_a_missing_image_names_the_fix(cluster: Cluster) -> None:
    result = cluster.run("shell.sh", "-c", "true", SIF=str(cluster.root / "absent.sif"))
    assert result.returncode == 1
    assert "prefetch.sh" in result.stderr


def test_serve_env_is_valid_json_with_single_line_fields(cluster: Cluster) -> None:
    """DSE-052: `ldd --version | head -1 | awk …` under `set -o pipefail`.

    head closes the pipe, the producer dies of SIGPIPE, the pipeline reports 141 even though awk
    already printed correctly, and `|| echo unknown` appends a second line - which is how the live
    run recorded `"glibc": "2.41\\nunknown"` and broke the JSON value.
    """
    out = cluster.root / "serve_env.json"
    # Passed through the environment, not inlined: the checkout path may contain spaces.
    env = cluster.env(
        TIER="qwen14b",
        MODEL="Qwen/Qwen3-14B",
        REVISION="abc",
        CC="gcc",
        GUIDED_BACKEND="xgrammar",
        SIF=str(cluster.sif),
        CONTAINER_SOURCE="docker://python@sha256:test",
    )
    result = subprocess.run(
        # `set -euo pipefail` is not decoration: pipefail is the precondition for the SIGPIPE bug
        # this test guards, so without it the assertions below pass against the broken code.
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; write_serve_env "$2"',
            "bash",
            str(COMMON),
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    for key, value in payload.items():
        assert "\n" not in value, f"{key} spans multiple lines: {value!r}"
    assert payload["structured_outputs_backend"] == "xgrammar"
    assert payload["glibc"] == "2.41"
    assert payload["gpu"] == "NVIDIA A100-PCIE-40GB"
    assert payload["driver"] == "550.127.05"

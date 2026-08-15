"""Docker-based executor — runs task commands inside containers.

This is the sandboxed counterpart to
:class:`~computecloud_node.local_executor.LocalProcessExecutor`: instead
of running the payload command directly on the host shell, the command is
executed inside a Docker container built from the image named in the task
payload.  This is what makes "template" jobs (PyTorch, TensorFlow,
Python 3.x, ...) possible — and it keeps untrusted renter code off the
contributor's host system.

Expected task payload::

    {
        "image": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        "command": "python -c 'import torch; print(torch.__version__)'",
        "timeout_seconds": 300,          # optional
        "gpu": true,                     # optional — pass --gpus all
        "memory_mb": 4096,               # optional — container memory cap
        "cpu_cores": 2.0                 # optional — container CPU cap
    }

If *image* is missing, the executor raises so the pool records a clear
failure (use LocalProcessExecutor for plain shell tasks instead).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from computecloud_node.executor import TaskExecutor
from computecloud_node.local_executor import CommandResult

logger = logging.getLogger(__name__)


def _find_docker() -> str:
    """Resolve the full path to the Docker CLI executable.

    On Windows, Docker Desktop installs docker.exe in a per-user
    directory that may not be on the PATH when Python spawns
    subprocesses.  We search common install locations as a fallback.
    """
    if sys.platform != "win32":
        return "docker"
    path = shutil.which("docker")
    if path:
        return path
    candidates = [
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Programs", "DockerDesktop", "resources", "bin", "docker.exe",
        ),
        os.path.join(
            os.environ.get("ProgramFiles", ""),
            "Docker", "Docker", "resources", "bin", "docker.exe",
        ),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return "docker"


def _find_docker_desktop_app() -> str | None:
    """Resolve the Docker Desktop application executable (Windows only)."""
    if sys.platform != "win32":
        return None
    candidates = [
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Programs", "DockerDesktop", "Docker Desktop.exe",
        ),
        os.path.join(
            os.environ.get("ProgramFiles", ""),
            "Docker", "Docker", "Docker Desktop.exe",
        ),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _launch_docker_desktop() -> bool:
    """Best-effort launch of Docker Desktop.  True = a launch was attempted.

    Linux is intentionally excluded: the daemon is a system service there
    (``sudo systemctl start docker``) and needs root — we just report.
    """
    try:
        if sys.platform == "win32":
            app = _find_docker_desktop_app()
            if not app:
                return False
            subprocess.Popen(
                [app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        if sys.platform == "darwin":
            if not os.path.isdir("/Applications/Docker.app"):
                return False
            subprocess.Popen(
                ["open", "-a", "Docker"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
    except OSError:
        return False
    return False


class DockerExecutor(TaskExecutor):
    """Execute tasks inside Docker containers (template jobs).

    Parameters
    ----------
    default_timeout_seconds:
        Timeout when the payload doesn't specify one.  Note that image
        pulls happen within this window on first use of a template.
    allowed_image_prefixes:
        Optional allow-list of image-name prefixes (e.g. ``["pytorch/",
        "python:", "tensorflow/"]``).  Empty/None = any image allowed.
    extra_docker_args:
        Additional ``docker run`` arguments applied to every container
        (e.g. ``["--network", "none"]`` to disable networking).
    """

    def __init__(
        self,
        default_timeout_seconds: float = 600.0,
        allowed_image_prefixes: list[str] | None = None,
        extra_docker_args: list[str] | None = None,
    ) -> None:
        self.default_timeout_seconds = default_timeout_seconds
        self.allowed_image_prefixes = allowed_image_prefixes or []
        self.extra_docker_args = extra_docker_args or []

    # ── TaskExecutor API ─────────────────────────────────────────────

    def execute(
        self,
        task_id: str,
        job_id: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Run the payload command inside the payload's Docker image."""
        image = self._extract_image(payload)
        command = self._extract_command(payload)
        timeout = self._extract_timeout(payload)

        docker_cmd = self._build_docker_command(image, command, payload)
        result = self._run(docker_cmd, timeout)
        return {
            "image": image,
            "command": command,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # ── Availability check ───────────────────────────────────────────

    @staticmethod
    def is_docker_available() -> bool:
        """Return True if a working ``docker`` CLI is on PATH."""
        docker_bin = _find_docker()
        if docker_bin == "docker" and shutil.which("docker") is None:
            return False
        try:
            completed = subprocess.run(
                [docker_bin, "info"],
                capture_output=True,
                timeout=10,
            )
            return completed.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    # ── Internal helpers ─────────────────────────────────────────────

    def _extract_image(self, payload: dict[str, Any] | None) -> str:
        if not payload or not payload.get("image"):
            raise ValueError(
                "DockerExecutor requires an 'image' key in the task payload "
                "(e.g. 'python:3.12-slim'). Use LocalProcessExecutor for "
                "plain shell tasks."
            )
        image = str(payload["image"])
        if self.allowed_image_prefixes and not any(
            image.startswith(p) for p in self.allowed_image_prefixes
        ):
            raise ValueError(
                f"Image '{image}' is not in this node's allow-list "
                f"({self.allowed_image_prefixes})"
            )
        return image

    @staticmethod
    def _extract_command(payload: dict[str, Any] | None) -> str:
        if payload is None:
            raise ValueError("DockerExecutor requires a task payload")
        for key in ("command", "cmd", "shell"):
            value = payload.get(key)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    return " ".join(str(v) for v in value)
                return str(value)
        raise ValueError("DockerExecutor: payload must contain a 'command' key")

    def _extract_timeout(self, payload: dict[str, Any] | None) -> float:
        if payload and "timeout_seconds" in payload:
            try:
                return float(payload["timeout_seconds"])
            except (TypeError, ValueError):
                pass
        return self.default_timeout_seconds

    def _build_docker_command(
        self,
        image: str,
        command: str,
        payload: dict[str, Any] | None,
    ) -> list[str]:
        """Assemble the full ``docker run`` argument list with security hardening."""
        args: list[str] = [
            _find_docker(), "run", "--rm",
            # ── Security hardening (Tier 1) ──
            "--security-opt", "no-new-privileges",  # prevent privilege escalation
            "--cap-drop", "ALL",                     # drop all Linux capabilities
            "--pids-limit", "256",                    # prevent fork bombs
            "--ulimit", "nofile=1024:4096",           # file descriptor limit
            "--read-only",                           # read-only root filesystem
            "--tmpfs", "/tmp:rw,size=64m",           # writable /tmp (needed since root is RO)
        ]

        payload = payload or {}
        if payload.get("gpu"):
            args += ["--gpus", "all"]
        memory_mb = payload.get("memory_mb")
        if memory_mb:
            args += ["--memory", f"{int(memory_mb)}m"]
        cpu_cores = payload.get("cpu_cores")
        if cpu_cores:
            args += ["--cpus", str(float(cpu_cores))]

        args += self.extra_docker_args
        # Run the command through the image's shell so pipes/&& work.
        args += [image, "/bin/sh", "-c", command]
        return args

    def _run(self, docker_cmd: list[str], timeout: float) -> CommandResult:
        """Execute the docker command and capture its output."""
        try:
            completed = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout
            if isinstance(stdout, bytes):
                stdout = stdout.decode()
            return CommandResult(
                return_code=-1,
                stdout=stdout or "",
                stderr=f"Container timed out after {timeout}s",
            )
        except FileNotFoundError:
            return CommandResult(
                return_code=-1,
                stdout="",
                stderr="docker CLI not found — install Docker to run template jobs",
            )

        return CommandResult(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )



def ensure_docker_running(
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 2.0,
) -> bool:
    """Ensure the Docker daemon is up, starting Docker Desktop if needed.

    1. Daemon already responding → ``True`` immediately.
    2. Otherwise try to launch Docker Desktop (Windows/macOS) and poll
       ``docker info`` until the daemon answers or *timeout_seconds* elapses
       (cold starts commonly take 30–60s).
    3. Returns ``False`` when there is nothing to launch (Linux, or Docker
       not installed) or the daemon never came up.
    """
    if DockerExecutor.is_docker_available():
        return True
    logger.info("Docker daemon not responding — attempting to start Docker Desktop")
    if not _launch_docker_desktop():
        logger.info("No Docker Desktop app to launch (not installed or unsupported OS)")
        return False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if DockerExecutor.is_docker_available():
            logger.info("Docker daemon is up")
            return True
        time.sleep(poll_interval_seconds)
    logger.warning("Timed out waiting for the Docker daemon (%.0fs)", timeout_seconds)
    return DockerExecutor.is_docker_available()

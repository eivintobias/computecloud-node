"""Pluggable workbench executor.

Phase 17d extracts session-launching logic into a pluggable protocol
with two implementations: Docker (the jail) and Native (no Docker).
The relay/tunnel plumbing in node.py is executor-agnostic.

Guardrail #2 (Docker = sandbox, not product): Docker is ONE option.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class SessionHandle:
    """Opaque handle to a running workbench session."""
    host_port: int
    auth_token: str
    container_id: str | None = None
    pid: int | None = None
    executor_type: str = "docker"


@runtime_checkable
class WorkbenchExecutor(Protocol):
    """Protocol for launching and managing workbench sessions."""
    def start_session(self, session_data: dict[str, Any]) -> SessionHandle: ...
    def stop_session(self, handle: SessionHandle) -> None: ...
    def is_session_alive(self, handle: SessionHandle) -> bool: ...


class WorkbenchExecutorError(RuntimeError):
    """Raised when a workbench session cannot be started."""


def _allocate_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _wait_for_port(host_port: int, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(("127.0.0.1", host_port))
                return True
        except OSError:
            time.sleep(0.5)
    return False


def probe_session_service(
    host_port: int, session_type: str, timeout: float = 10.0
) -> str | None:
    """Deep-check that the session's service actually speaks on the port.

    A bare TCP connect is NOT enough: Docker's port proxy accepts connections
    before the in-container service is serving, so a broken container would
    otherwise become a "ready but silent" session.  For SSH the sshd banner
    (``SSH-...``) must arrive within *timeout*; other types only need the
    connect (HTTP services vary too much to probe generically).

    Returns ``None`` when healthy, else a short problem description.
    """
    try:
        with socket.create_connection(("127.0.0.1", host_port), timeout=timeout) as s:
            if session_type != "ssh":
                return None
            s.settimeout(timeout)
            banner = s.recv(64)
            if banner.startswith(b"SSH-"):
                return None
            if not banner:
                return "connection closed without banner"
            return f"unexpected banner: {banner[:32]!r}"
    except OSError as exc:
        return f"connect/read failed: {exc}"


def _find_docker() -> str:
    if sys.platform != "win32":
        return "docker"
    path = shutil.which("docker")
    if path:
        return path
    for c in [
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "DockerDesktop",
                     "resources", "bin", "docker.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""),
                     "Docker", "Docker",
                     "resources", "bin", "docker.exe"),
    ]:
        if c and os.path.isfile(c):
            return c
    return "docker"


def _docker_available() -> bool:
    docker_path = shutil.which("docker")
    if not docker_path:
        if os.name == "nt":
            for c in [
                os.path.join(os.environ.get("LOCALAPPDATA", ""),
                             "Programs", "DockerDesktop",
                             "resources", "bin", "docker.exe"),
                os.path.join(os.environ.get("ProgramFiles", ""),
                             "Docker", "Docker",
                             "resources", "bin", "docker.exe"),
            ]:
                if c and os.path.isfile(c):
                    docker_path = c
                    break
        if not docker_path:
            return False
    try:
        result = subprocess.run(
            [docker_path, "info", "--format",
             "{{.ServerVersion}}"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=10.0,
        )
        return result.returncode == 0
    except Exception:
        return False


# -- Docker executor --


class DockerWorkbenchExecutor:
    """Launch workbench sessions inside hardened Docker containers."""

    def start_session(self, session_data: dict[str, Any]) -> SessionHandle:
        import secrets as _secrets
        docker_image = session_data["docker_image"]
        command = session_data["command"]
        container_port = session_data["container_port"]
        auth_token = _secrets.token_urlsafe(12)
        host_port = _allocate_host_port()
        docker_cmd: list[str] = [
            _find_docker(), "run", "-d",
            "-p", f"{host_port}:{container_port}",
            "--memory",
            f"{int(session_data.get('memory_mb', 4096))}m",
            "--cpus", str(session_data.get("cpu_cores", 2.0)),
            "--pids-limit", "512",
            "--ulimit", "nofile=4096:8192",
            "--tmpfs", "/tmp:rw,size=256m",
        ]
        session_type = session_data.get("session_type", "")
        if session_type == "ssh":
            docker_cmd += self._ssh_security_args()
        else:
            docker_cmd += ["--security-opt", "no-new-privileges", "--cap-drop", "ALL"]
        try:
            from computecloud_node.security_monitor import (
                KNOWN_MINING_POOL_DOMAINS,
            )
            for domain in KNOWN_MINING_POOL_DOMAINS:
                docker_cmd += ["--add-host", f"{domain}:127.0.0.1"]
        except ImportError:
            pass

        # Inject Pool SDK env vars when workspace/pool_env is present.
        pool_env = session_data.get("pool_env")
        if pool_env:
            base_url = str(
                session_data.get("_http_base_url", "")
            ).rstrip("/")
            pool_token = pool_env.get("POOL_TOKEN", "")
            pool_ws = pool_env.get("POOL_WORKSPACE_ID", "")
            docker_cmd += [
                "-e", f"POOL_API_URL={base_url}",
                "-e", f"POOL_TOKEN={pool_token}",
                "-e", f"POOL_WORKSPACE_ID={pool_ws}",
            ]

        startup_parts = self._build_startup_parts(session_data)
        run_parts_post_start = False
        if session_type == "ssh" and not command:
            # Phase 18e: the SSH image must run its OWN entrypoint — s6's
            # /init starts sshd, and wrapping it in /bin/sh -c breaks the
            # init (the container idles on `tail -f /dev/null` or dies in
            # s6-overlay-suexec).  Startup parts (key injection, workspace
            # sync, SDK install) run post-start via docker exec below.
            docker_cmd.append(docker_image)
            run_parts_post_start = bool(startup_parts)
        else:
            if startup_parts and not command:
                # Phase 18d: command="" means "use the image's own
                # entrypoint"; resolve it and exec it after the startup
                # script instead of replacing it with tail -f.
                entrypoint = self._image_entrypoint(docker_image)
                if entrypoint:
                    import shlex as _shlex

                    command = "exec " + _shlex.join(entrypoint)
            docker_cmd += self._build_final_command(
                docker_image, command, startup_parts,
            )
        container_id = self._run_docker(docker_cmd)
        if not _wait_for_port(host_port, timeout=180.0):
            self._stop_container(container_id)
            raise WorkbenchExecutorError("Container port never opened")
        if run_parts_post_start:
            self._run_startup_parts_docker(container_id, startup_parts)
        return SessionHandle(
            host_port=host_port, auth_token=auth_token,
            container_id=container_id, executor_type="docker",
        )

    def stop_session(self, handle: SessionHandle) -> None:
        if handle.container_id:
            self._stop_container(handle.container_id)

    def is_session_alive(self, handle: SessionHandle) -> bool:
        if not handle.container_id:
            return False
        try:
            result = subprocess.run(
                [_find_docker(), "inspect", "-f",
                 "{{.State.Running}}", handle.container_id],
                capture_output=True, encoding="utf-8", errors="replace", timeout=5.0,
            )
            return (result.returncode == 0
                    and result.stdout.strip() == "true")
        except Exception:
            return False

    def repull_image(self, session_data: dict[str, Any]) -> None:
        """``docker pull`` the session's image (fixes stale/corrupt caches)."""
        image = session_data["docker_image"]
        logger.info("Re-pulling image %s for session self-heal", image)
        subprocess.run(
            [_find_docker(), "pull", image],
            capture_output=True, encoding="utf-8", errors="replace", timeout=600.0,
        )

    def container_logs(self, handle: SessionHandle, tail: int = 40) -> str:
        """Return the last *tail* lines of the container's logs ('' on failure)."""
        if not handle.container_id:
            return ""
        try:
            result = subprocess.run(
                [_find_docker(), "logs", "--tail", str(tail), handle.container_id],
                capture_output=True, encoding="utf-8", errors="replace", timeout=15.0,
            )
            return (result.stdout + result.stderr).strip()[-2000:]
        except Exception:
            return ""

    def _run_startup_parts_docker(
        self, container_id: str, startup_parts: list[str]
    ) -> None:
        """Run startup script parts inside a booted container via docker exec.

        Used for the SSH image, which must keep its own entrypoint (Phase
        18e).  Retries a few times while s6 init settles; failures are logged
        but non-fatal (the password fallback still works).
        """
        import time as _time

        script = " && ".join(startup_parts)
        for attempt in range(3):
            try:
                result = subprocess.run(
                    [_find_docker(), "exec", container_id, "/bin/sh", "-c", script],
                    capture_output=True, encoding="utf-8", errors="replace", timeout=60.0,
                )
                if result.returncode == 0:
                    return
                logger.warning(
                    "Startup parts failed (rc=%d, attempt %d/3): %s",
                    result.returncode, attempt + 1, result.stderr.strip()[:200],
                )
            except Exception as exc:
                logger.warning("Startup parts attempt %d/3 error: %s", attempt + 1, exc)
            _time.sleep(2.0)
        logger.warning("Startup parts never completed — session may miss keys/workspace")

    @staticmethod
    def _ssh_security_args() -> list[str]:
        return [
            "--cap-drop", "ALL",
            "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--cap-add", "SYS_CHROOT",
            "--cap-add", "DAC_OVERRIDE",
            "--cap-add", "CHOWN", "--cap-add", "FOWNER",
            "--cap-add", "KILL",
            "--cap-add", "NET_BIND_SERVICE",
            # NOTE: lowercase "true" — the image checks [[ == "true" ]].
            "-e", "PASSWORD_ACCESS=true",
            "-e", "USER_PASSWORD=poolpass",
            # Phase 18e: USER_NAME=pool (NOT root — the current image halts
            # init with "USER_NAME cannot be set to an user that already
            # exists" for existing users, and sshd never starts).  With
            # PUID=0 the pool user is uid 0 inside the sandbox anyway.
            "-e", "USER_NAME=pool",
            "-e", "PUID=0", "-e", "PGID=0",
        ]

    @staticmethod
    def _build_startup_parts(
        session_data: dict[str, Any],
    ) -> list[str]:
        """Build shell startup script parts for inside the container."""
        parts: list[str] = []
        ws = session_data.get("workspace")
        pool_env = session_data.get("pool_env")
        ssh_keys = session_data.get("ssh_public_keys", [])
        stype = session_data.get("session_type", "")

        if not (ws or pool_env):
            if stype == "ssh" and ssh_keys:
                # printf '%s\n' prints each arg on its own line (with trailing
                # newline) — the old "\\\\n".join glued keys together.
                keys_blob = " ".join("'" + k.strip() + "'" for k in ssh_keys if k.strip())
                parts.append(
                    "mkdir -p /config/.ssh"
                    f" && printf '%s\\n' {keys_blob}"
                    " >> /config/.ssh/authorized_keys"
                    " && chmod 700 /config/.ssh"
                    " && chmod 600 /config/.ssh/authorized_keys"
                    " 2>/dev/null || true"
                )
            return parts

        parts.append(
            "pip install computecloud-sdk 2>/dev/null || true"
        )

        if ws:
            files = ws.get("files", [])
            dl_prefix = ws.get("download_path_prefix", "")
            pool_token = (
                pool_env.get("POOL_TOKEN", "")
                if pool_env else ""
            )
            base_url = str(
                session_data.get("_http_base_url", "")
            ).rstrip("/")
            parts.append("mkdir -p /workspace")
            for f_info in files:
                rel_path = f_info.get("relative_path", "")
                if not rel_path:
                    continue
                url = f"{base_url}{dl_prefix}/{rel_path}"
                parts.append(
                    f'mkdir -p /workspace/'
                    f'$(dirname "{rel_path}")'
                )
                parts.append(
                    f'curl -sf -H '
                    f'"Authorization: Bearer {pool_token}" '
                    f'-o "/workspace/{rel_path}" '
                    f'"{url}" || true'
                )

        # pool-push helper (best-effort, may fail silently).

        if stype == "jupyter" and ws:
            try:
                import base64 as _b64

                from computecloud_node.notebook_bootstrap import (
                    WELCOME_NOTEBOOK_JSON,
                )
                nb_b64 = _b64.b64encode(
                    WELCOME_NOTEBOOK_JSON.encode()
                ).decode()
                parts.append(
                    f"test -f /workspace/welcome.ipynb"
                    f" || (mkdir -p /workspace && "
                    f"echo '{nb_b64}'"
                    f" | base64 -d"
                    f" > /workspace/welcome.ipynb)"
                    f" 2>/dev/null || true"
                )
            except ImportError:
                pass

        if stype == "ssh" and ssh_keys:
            keys_blob = " ".join("'" + k.strip() + "'" for k in ssh_keys if k.strip())
            parts.append(
                "mkdir -p /config/.ssh"
                f" && printf '%s\\n' {keys_blob}"
                " >> /config/.ssh/authorized_keys"
                " && chmod 700 /config/.ssh"
                " && chmod 600 /config/.ssh/authorized_keys"
                " 2>/dev/null || true"
            )
        return parts

    @staticmethod
    def _image_entrypoint(image: str) -> list[str]:
        """Resolve an image's configured ENTRYPOINT+CMD via ``docker image
        inspect`` (the image is always local — it was just pulled/run).

        Returns [] on any failure; callers fall back to the previous
        behavior (``tail -f /dev/null`` idle) which the 18c self-heal ladder
        will catch and report.
        """
        try:
            result = subprocess.run(
                [_find_docker(), "image", "inspect", "--format",
                 "{{json .Config}}", image],
                capture_output=True, encoding="utf-8", errors="replace", timeout=30.0,
            )
            if result.returncode != 0:
                return []
            import json as _json

            cfg = _json.loads(result.stdout)
            ep = cfg.get("Entrypoint") or []
            cmd = cfg.get("Cmd") or []
            return [str(x) for x in (*ep, *cmd)]
        except Exception:
            return []

    @staticmethod
    def _build_final_command(
        docker_image: str,
        command: str,
        startup_parts: list[str],
    ) -> list[str]:
        """Assemble the final docker run image+command arguments."""
        if startup_parts:
            script = " && ".join(startup_parts)
            if command:
                full = f"{script} ; {command}"
            else:
                full = f"{script} ; tail -f /dev/null"
            return [docker_image, "/bin/sh", "-c", full]
        if command:
            return [docker_image, "/bin/sh", "-c", command]
        return [docker_image]

    @staticmethod
    def _run_docker(docker_cmd: list[str]) -> str:
        """Execute docker run and return the container ID."""
        try:
            result = subprocess.run(
                docker_cmd, capture_output=True,
                encoding="utf-8", errors="replace", timeout=120.0,
            )
            if result.returncode != 0:
                raise WorkbenchExecutorError(
                    f"docker run failed: {result.stderr}"
                )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise WorkbenchExecutorError(
                "docker run timed out"
            )
        except FileNotFoundError:
            raise WorkbenchExecutorError(
                "docker CLI not found"
            )

    @staticmethod
    def _stop_container(container_id: str) -> None:
        """Stop and remove a Docker container."""
        for args in (
            [_find_docker(), "stop", container_id],
            [_find_docker(), "rm", "-f", "-v", container_id],
        ):
            try:
                subprocess.run(
                    args, capture_output=True,
                    encoding="utf-8", errors="replace", timeout=10.0,
                )
            except Exception:
                pass


# ── Native executor (no Docker) ─────────────────────────────────────────────

_NATIVE_WARNING = (
    "Native workbench executor provides NO isolation. "
    "Only use on trusted/self-hosted nodes."
)


class NativeWorkbenchExecutor:
    """Launch workbench sessions as local subprocesses — NO Docker.

    .. warning:: **No isolation.** Only use on trusted/self-hosted nodes.
    """

    def __init__(self, http_client: Any | None = None) -> None:
        self._http_client = http_client
        logger.warning(_NATIVE_WARNING)

    def start_session(
        self, session_data: dict[str, Any]
    ) -> SessionHandle:
        import secrets as _secrets

        stype = session_data.get("session_type", "")
        command = session_data.get("command", "")
        auth_token = _secrets.token_urlsafe(12)
        host_port = _allocate_host_port()

        ws_block = session_data.get("workspace")
        pool_env = session_data.get("pool_env")
        ws_dir = (
            "/workspace"
            if sys.platform != "win32"
            else os.path.join(os.getcwd(), "workspace")
        )

        if ws_block or pool_env:
            os.makedirs(ws_dir, exist_ok=True)
            if ws_block and self._http_client:
                self._sync_ws(ws_block, pool_env, ws_dir)
            if stype == "jupyter" and ws_block:
                try:
                    from computecloud_node.notebook_bootstrap import (
                        generate_welcome_notebook,
                    )
                    generate_welcome_notebook(ws_dir)
                except Exception:
                    pass

        ssh_keys = session_data.get("ssh_public_keys", [])
        if stype == "ssh" and ssh_keys:
            self._inject_ssh_keys(ssh_keys)

        env = os.environ.copy()
        if pool_env:
            base_url = (
                str(self._http_client.base_url).rstrip("/")
                if self._http_client else ""
            )
            env["POOL_API_URL"] = base_url
            env["POOL_TOKEN"] = pool_env.get("POOL_TOKEN", "")
            env["POOL_WORKSPACE_ID"] = (
                pool_env.get("POOL_WORKSPACE_ID", "")
            )

        if stype == "jupyter":
            proc = self._start_jupyter(
                command, host_port, ws_dir, env
            )
        elif stype == "ssh":
            proc = self._start_sshd(host_port, env)
        elif stype == "custom":
            proc = self._start_custom(command, env)
        else:
            raise WorkbenchExecutorError(
                f"Unknown session type: {stype!r}"
            )

        if proc is None:
            raise WorkbenchExecutorError(
                "Failed to start session process"
            )
        if not _wait_for_port(host_port, timeout=60.0):
            self._kill_process(proc)
            raise WorkbenchExecutorError(
                "Session port never opened"
            )
        return SessionHandle(
            host_port=host_port, auth_token=auth_token,
            pid=proc.pid, executor_type="native",
        )

    def stop_session(self, handle: SessionHandle) -> None:
        if handle.pid:
            try:
                import signal
                os.kill(handle.pid, signal.SIGTERM)
                time.sleep(1.0)
                try:
                    os.kill(handle.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            except (ProcessLookupError, OSError):
                pass
            except Exception:
                pass

    def is_session_alive(self, handle: SessionHandle) -> bool:
        if not handle.pid:
            return False
        try:
            os.kill(handle.pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False
        except Exception:
            return False

    def _start_jupyter(self, command, host_port, ws_dir, env):
        if command:
            cmd = (
                command
                .replace("--port=8888", f"--port={host_port}")
                .replace("--port 8888", f"--port {host_port}")
            )
        else:
            cmd = (
                f"jupyter notebook --ip=0.0.0.0"
                f" --port={host_port}"
                " --NotebookApp.token=''"
                " --NotebookApp.password=''"
                " --no-browser"
                f" --ServerApp.root_dir={ws_dir}"
            )
        if not shutil.which("jupyter"):
            logger.info(
                "jupyter not found - attempting pip install"
            )
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip",
                     "install", "jupyter"],
                    capture_output=True, text=True,
                    timeout=120.0,
                )
            except Exception as exc:
                logger.error(
                    "Failed to install jupyter: %s", exc
                )
        try:
            return subprocess.Popen(
                cmd, shell=True, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, Exception):
            return None

    def _start_sshd(self, host_port, env):
        sshd_path = shutil.which("sshd")
        if not sshd_path:
            if sys.platform == "win32":
                raise WorkbenchExecutorError(
                    "Native SSH requires sshd"
                    " (not available on Windows)."
                    " Use Docker executor or Jupyter."
                )
            raise WorkbenchExecutorError(
                "sshd not found. Install openssh-server"
                " or use the Docker executor."
            )
        try:
            return subprocess.Popen(
                [sshd_path, "-D", "-p", str(host_port)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, Exception):
            return None

    def _start_custom(self, command, env):
        if not command:
            raise WorkbenchExecutorError(
                "Custom session requires a command"
            )
        try:
            return subprocess.Popen(
                command, shell=True, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, Exception):
            return None

    def _sync_ws(self, ws_block, pool_env, ws_dir):
        if not self._http_client:
            return
        files = ws_block.get("files", [])
        dl_prefix = ws_block.get("download_path_prefix", "")
        pool_token = (
            pool_env.get("POOL_TOKEN", "")
            if pool_env else ""
        )
        base_url = str(
            self._http_client.base_url
        ).rstrip("/")
        headers = (
            {"Authorization": f"Bearer {pool_token}"}
            if pool_token else {}
        )
        for f_info in files:
            rel_path = f_info.get("relative_path", "")
            if not rel_path:
                continue
            url = f"{base_url}{dl_prefix}/{rel_path}"
            local_path = os.path.join(
                ws_dir, rel_path.replace("/", os.sep)
            )
            os.makedirs(
                os.path.dirname(local_path), exist_ok=True
            )
            try:
                resp = self._http_client.get(
                    url, headers=headers, timeout=30.0,
                )
                if resp.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
            except Exception:
                pass

    def _inject_ssh_keys(self, ssh_keys):
        home = os.path.expanduser("~")
        ssh_dir = os.path.join(home, ".ssh")
        auth_keys = os.path.join(ssh_dir, "authorized_keys")
        try:
            os.makedirs(ssh_dir, exist_ok=True)
            with open(auth_keys, "a") as f:
                for key in ssh_keys:
                    f.write(key + "\n")
            os.chmod(ssh_dir, 0o700)
            os.chmod(auth_keys, 0o600)
        except Exception:
            pass

    @staticmethod
    def _kill_process(proc):
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# ── Factory ─────────────────────────────────────────────────────────────────


def create_workbench_executor(
    mode: str = "auto",
    http_client: Any | None = None,
) -> WorkbenchExecutor:
    """Create a workbench executor based on the configured mode.

    Parameters
    ----------
    mode:
        ``"docker"`` — always use Docker (fail if unavailable).
        ``"native"`` — always use native (no isolation).
        ``"auto"`` — probe Docker at startup; use Docker if
        available, else fall back to native.
    http_client:
        Optional httpx client for workspace file downloads.
    """
    if mode == "docker":
        if not _docker_available():
            raise WorkbenchExecutorError(
                "Docker executor requested but Docker is"
                " not available."
            )
        logger.info("Workbench executor: docker (configured)")
        return DockerWorkbenchExecutor()  # type: ignore

    if mode == "native":
        logger.info("Workbench executor: native (configured)")
        return NativeWorkbenchExecutor(
            http_client=http_client,
        )  # type: ignore

    # auto
    if _docker_available():
        logger.info("Workbench executor: docker (auto)")
        return DockerWorkbenchExecutor()  # type: ignore
    logger.warning(
        "Docker not available - falling back to"
        " native workbench executor."
    )
    return NativeWorkbenchExecutor(
        http_client=http_client,
    )  # type: ignore


__all__ = [
    "SessionHandle",
    "WorkbenchExecutor",
    "WorkbenchExecutorError",
    "DockerWorkbenchExecutor",
    "NativeWorkbenchExecutor",
    "create_workbench_executor",
]


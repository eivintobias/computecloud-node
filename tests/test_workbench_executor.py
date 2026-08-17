"""Tests for the workbench executor port (Phase 17d / node_client v0.6.2).

Covers the factory auto-detection, NativeWorkbenchExecutor launch/stop, and
the NodeConfig workbench_executor field.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from computecloud_node.workbench_executor import (
    DockerWorkbenchExecutor,
    NativeWorkbenchExecutor,
    SessionHandle,
    WorkbenchExecutor,
    create_workbench_executor,
    probe_session_service,
)


class TestFactory:
    def test_native_mode_returns_native(self):
        ex = create_workbench_executor("native")
        assert isinstance(ex, NativeWorkbenchExecutor)

    def test_docker_mode_returns_docker(self):
        with mock.patch(
            "computecloud_node.workbench_executor._docker_available",
            return_value=True,
        ):
            ex = create_workbench_executor("docker")
        assert isinstance(ex, DockerWorkbenchExecutor)

    def test_auto_docker_available(self):
        with mock.patch(
            "computecloud_node.workbench_executor._docker_available",
            return_value=True,
        ):
            ex = create_workbench_executor("auto")
        assert isinstance(ex, DockerWorkbenchExecutor)

    def test_auto_docker_unavailable(self):
        with mock.patch(
            "computecloud_node.workbench_executor._docker_available",
            return_value=False,
        ):
            ex = create_workbench_executor("auto")
        assert isinstance(ex, NativeWorkbenchExecutor)

    def test_interface_check(self):
        assert isinstance(DockerWorkbenchExecutor(), WorkbenchExecutor)
        assert isinstance(NativeWorkbenchExecutor(), WorkbenchExecutor)


class TestSessionHandle:
    def test_docker_handle_fields(self):
        h = SessionHandle(
            host_port=9000,
            auth_token="tok",
            container_id="abc",
            executor_type="docker",
        )
        assert h.host_port == 9000
        assert h.container_id == "abc"
        assert h.executor_type == "docker"

    def test_native_handle_fields(self):
        h = SessionHandle(
            host_port=9001, auth_token="tok2", pid=12345, executor_type="native"
        )
        assert h.pid == 12345
        assert h.executor_type == "native"


class TestNativeWorkbenchExecutor:
    def _session_data(self, **overrides):
        data = {
            "session_id": "sess-1",
            "session_type": "jupyter",
            "docker_image": "",
            "command": "jupyter notebook --port=8888",
            "container_port": 8888,
            "memory_mb": 4096,
            "cpu_cores": 2.0,
        }
        data.update(overrides)
        return data

    @mock.patch("computecloud_node.workbench_executor.subprocess.Popen")
    @mock.patch(
        "computecloud_node.workbench_executor._wait_for_port", return_value=True
    )
    @mock.patch(
        "computecloud_node.workbench_executor._allocate_host_port",
        return_value=9999,
    )
    def test_start_session_returns_handle(self, _p, _w, mock_popen):
        proc = mock.MagicMock()
        proc.pid = 4242
        mock_popen.return_value = proc
        ex = NativeWorkbenchExecutor()
        handle = ex.start_session(self._session_data())
        assert isinstance(handle, SessionHandle)
        assert handle.host_port == 9999
        assert handle.auth_token
        assert handle.pid == 4242
        assert handle.container_id is None
        assert handle.executor_type == "native"

    def test_stop_session_kills_pid(self):
        ex = NativeWorkbenchExecutor()
        handle = SessionHandle(
            host_port=9999, auth_token="t", pid=999, executor_type="native"
        )
        with mock.patch(
            "computecloud_node.workbench_executor.os.kill"
        ) as m_kill:
            ex.stop_session(handle)
        assert m_kill.called
        assert m_kill.call_args[0][0] == 999

    def test_is_session_alive_true(self):
        ex = NativeWorkbenchExecutor()
        handle = SessionHandle(
            host_port=1, auth_token="t", pid=1, executor_type="native"
        )
        with mock.patch(
            "computecloud_node.workbench_executor.os.kill", return_value=None
        ):
            assert ex.is_session_alive(handle) is True

    def test_is_session_alive_false(self):
        ex = NativeWorkbenchExecutor()
        handle = SessionHandle(
            host_port=1, auth_token="t", pid=1, executor_type="native"
        )
        with mock.patch(
            "computecloud_node.workbench_executor.os.kill",
            side_effect=ProcessLookupError,
        ):
            assert ex.is_session_alive(handle) is False

    def test_ssh_unavailable_raises_clear_error(self):
        with mock.patch(
            "computecloud_node.workbench_executor.shutil.which", return_value=None
        ):
            with mock.patch(
                "computecloud_node.workbench_executor.sys.platform", "win32"
            ):
                ex = NativeWorkbenchExecutor()
                with pytest.raises(Exception, match="sshd"):
                    ex._start_sshd(2222, {})


class TestNodeConfigField:
    def test_workbench_executor_default(self):
        from computecloud_node.config import NodeConfig

        assert NodeConfig().workbench_executor == "auto"


class _BannerServer:
    """Configurable TCP server: sends a banner, or accepts-then-closes."""

    def __init__(self, banner: bytes | None = b"SSH-2.0-OpenSSH_9.6\r\n") -> None:
        import socket
        import threading

        self._banner = banner
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            if self._banner is not None:
                try:
                    conn.sendall(self._banner)
                except OSError:
                    pass
            conn.close()


class TestProbeSessionService:
    """Phase 18c — deep service probe used by the self-heal ladder."""

    def test_ssh_banner_ok(self):
        srv = _BannerServer()
        assert probe_session_service(srv.port, "ssh", timeout=3.0) is None

    def test_ssh_banner_wrong(self):
        srv = _BannerServer(banner=b"HTTP/1.1 400\r\n")
        problem = probe_session_service(srv.port, "ssh", timeout=3.0)
        assert problem is not None and "unexpected banner" in problem

    def test_ssh_no_banner(self):
        srv = _BannerServer(banner=None)
        problem = probe_session_service(srv.port, "ssh", timeout=1.0)
        assert problem is not None

    def test_closed_port(self):
        import socket as _socket

        with _socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert probe_session_service(port, "ssh", timeout=1.0) is not None

    def test_non_ssh_needs_only_connect(self):
        srv = _BannerServer(banner=None)
        assert probe_session_service(srv.port, "jupyter", timeout=2.0) is None


class TestDockerDiagnostics:
    """Phase 18c — repull_image / container_logs on DockerWorkbenchExecutor."""

    def test_repull_image_pulls(self):
        ex = DockerWorkbenchExecutor()
        with mock.patch(
            "computecloud_node.workbench_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            ex.repull_image({"docker_image": "linuxserver/openssh-server:latest"})
        cmd = mock_run.call_args[0][0]
        assert "pull" in cmd and "linuxserver/openssh-server:latest" in cmd

    def test_container_logs_tail(self):
        ex = DockerWorkbenchExecutor()
        h = SessionHandle(host_port=1, auth_token="t", container_id="abc123")
        with mock.patch(
            "computecloud_node.workbench_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout="log-line\n", stderr=""
            )
            logs = ex.container_logs(h, tail=10)
        assert "log-line" in logs

    def test_container_logs_no_container(self):
        ex = DockerWorkbenchExecutor()
        h = SessionHandle(host_port=1, auth_token="t")
        assert ex.container_logs(h) == ""

    def test_container_logs_uses_utf8_encoding(self):
        """docker logs emits UTF-8 (the linuxserver s6 ASCII-art banner has
        box-drawing chars).  Decoding with the Windows locale codec (cp1252)
        raises UnicodeDecodeError, which was swallowed and surfaced as "No
        container logs available".  Must decode as UTF-8 with replacement."""
        ex = DockerWorkbenchExecutor()
        h = SessionHandle(host_port=1, auth_token="t", container_id="abc123")
        with mock.patch(
            "computecloud_node.workbench_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="", stderr="")
            ex.container_logs(h, tail=10)
        kwargs = mock_run.call_args.kwargs
        assert kwargs.get("encoding") == "utf-8"
        assert kwargs.get("errors") == "replace"


class TestSelfHealLadder:
    """Phase 18c — ComputeNode._verify_and_heal_session."""

    def _node(self):
        from computecloud_node.node import ComputeNode

        node = ComputeNode.__new__(ComputeNode)
        node._report_session_terminated = mock.MagicMock()
        return node

    def _payload(self) -> dict:
        return {
            "session_id": "sess-1234-5678",
            "session_type": "ssh",
            "docker_image": "linuxserver/openssh-server:latest",
        }

    def test_healthy_first_probe(self):
        node = self._node()
        handle = SessionHandle(
            host_port=1, auth_token="t", container_id="c1", executor_type="docker"
        )
        executor = mock.MagicMock()
        with mock.patch(
            "computecloud_node.workbench_executor.probe_session_service",
            return_value=None,
        ):
            assert node._verify_and_heal_session(executor, self._payload(), handle) is handle
        executor.stop_session.assert_not_called()
        node._report_session_terminated.assert_not_called()

    def test_exhausted_reports_failure_with_logs(self):
        node = self._node()
        h1 = SessionHandle(
            host_port=1, auth_token="t", container_id="c1", executor_type="docker"
        )
        h2 = SessionHandle(
            host_port=2, auth_token="t", container_id="c2", executor_type="docker"
        )
        h3 = SessionHandle(
            host_port=3, auth_token="t", container_id="c3", executor_type="docker"
        )
        executor = mock.MagicMock()
        executor.start_session.side_effect = [h2, h3]
        executor.container_logs.return_value = "sshd: boom"
        with (
            mock.patch(
                "computecloud_node.workbench_executor.probe_session_service",
                return_value="dead",
            ),
            mock.patch("time.sleep"),
        ):
            result = node._verify_and_heal_session(executor, self._payload(), h1)
        assert result is None
        executor.repull_image.assert_called_once()
        report = node._report_session_terminated.call_args[0]
        assert report[1] == "failed"
        assert "did not start" in report[2]
        assert "sshd: boom" in report[2]

    def test_exhausted_captures_logs_before_each_rung(self):
        node = self._node()
        h1 = SessionHandle(host_port=1, auth_token="t", container_id="c1", executor_type="docker")
        h2 = SessionHandle(host_port=2, auth_token="t", container_id="c2", executor_type="docker")
        h3 = SessionHandle(host_port=3, auth_token="t", container_id="c3", executor_type="docker")
        executor = mock.MagicMock()
        executor.start_session.side_effect = [h2, h3]
        executor.container_logs.side_effect = ["log-from-c1", "", "log-from-c3"]
        with (
            mock.patch(
                "computecloud_node.workbench_executor.probe_session_service",
                return_value="dead",
            ),
            mock.patch("time.sleep"),
        ):
            result = node._verify_and_heal_session(executor, self._payload(), h1)
        assert result is None
        captured = [c.args[0] for c in executor.container_logs.call_args_list]
        assert captured == [h1, h2, h3]
        report = node._report_session_terminated.call_args[0]
        assert "log-from-c3" in report[2]

    def test_exhausted_empty_logs_never_reports_no_logs_available(self):
        node = self._node()
        h1 = SessionHandle(host_port=1, auth_token="t", container_id="c1", executor_type="docker")
        h2 = SessionHandle(host_port=2, auth_token="t", container_id="c2", executor_type="docker")
        h3 = SessionHandle(host_port=3, auth_token="t", container_id="c3", executor_type="docker")
        executor = mock.MagicMock()
        executor.start_session.side_effect = [h2, h3]
        executor.container_logs.return_value = ""
        with (
            mock.patch(
                "computecloud_node.workbench_executor.probe_session_service",
                return_value="dead",
            ),
            mock.patch("time.sleep"),
        ):
            result = node._verify_and_heal_session(executor, self._payload(), h1)
        assert result is None
        report = node._report_session_terminated.call_args[0]
        assert "No container logs available" not in report[2]
        assert "Container logs were empty on every rung." in report[2]


class TestEntrypointWrapping:
    """Phase 18d — startup scripts must not replace the image's entrypoint.

    Regression: an SSH workbench with injected keys used to idle on
    ``tail -f /dev/null`` — the image's /init (sshd) never ran.
    """

    def _payload(self, keys: list[str]) -> dict:
        return {
            "session_id": "sess-entrypoint",
            "session_type": "ssh",
            "docker_image": "linuxserver/openssh-server:latest",
            "command": "",
            "container_port": 2222,
            "ssh_public_keys": keys,
        }

    def test_ssh_with_keys_bare_image_and_post_start_injection(self):
        ex = DockerWorkbenchExecutor()
        with (
            mock.patch.object(
                DockerWorkbenchExecutor, "_run_docker", return_value="cid123"
            ) as mock_run,
            mock.patch.object(
                DockerWorkbenchExecutor, "_run_startup_parts_docker"
            ) as mock_parts,
            mock.patch(
                "computecloud_node.workbench_executor._wait_for_port",
                return_value=True,
            ),
        ):
            handle = ex.start_session(self._payload(["ssh-ed25519 AAAA test"]))
        assert handle.container_id == "cid123"
        docker_cmd = mock_run.call_args[0][0]
        # SSH image must run its OWN entrypoint: bare image, no /bin/sh -c wrap.
        assert docker_cmd[-1] == "linuxserver/openssh-server:latest"
        assert "-c" not in docker_cmd
        mock_parts.assert_called_once()
        parts = mock_parts.call_args[0][1]
        assert any("authorized_keys" in p for p in parts)
        assert any("ssh-ed25519 AAAA test" in p for p in parts)

    def test_startup_parts_docker_exec(self):
        ex = DockerWorkbenchExecutor()
        with mock.patch(
            "computecloud_node.workbench_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            ex._run_startup_parts_docker("cid123", ["echo hi", "echo bye"])
        cmd = mock_run.call_args[0][0]
        assert "exec" in cmd and "cid123" in cmd
        assert cmd[-1] == "echo hi && echo bye"

    def test_startup_parts_docker_retries_then_warns(self):
        ex = DockerWorkbenchExecutor()
        with (
            mock.patch(
                "computecloud_node.workbench_executor.subprocess.run"
            ) as mock_run,
            mock.patch("time.sleep"),
        ):
            mock_run.return_value = mock.MagicMock(returncode=1, stderr="boom")
            ex._run_startup_parts_docker("cid", ["x"])
        assert mock_run.call_count == 3

    def test_ssh_security_args_pool_user_lowercase_true(self):
        args = DockerWorkbenchExecutor._ssh_security_args()
        assert "USER_NAME=pool" in args and "USER_NAME=root" not in args
        assert "PASSWORD_ACCESS=true" in args

    def test_ssh_without_keys_keeps_bare_image(self):
        ex = DockerWorkbenchExecutor()
        with (
            mock.patch.object(
                DockerWorkbenchExecutor, "_run_docker", return_value="cid123"
            ) as mock_run,
            mock.patch(
                "computecloud_node.workbench_executor._wait_for_port",
                return_value=True,
            ),
        ):
            ex.start_session(self._payload([]))
        docker_cmd = mock_run.call_args[0][0]
        assert docker_cmd[-1] == "linuxserver/openssh-server:latest"
        assert "-c" not in docker_cmd

    def test_image_entrypoint_parses_config(self):
        with mock.patch(
            "computecloud_node.workbench_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0,
                stdout='{"Entrypoint": ["/init"], "Cmd": ["run"]}\n',
            )
            assert DockerWorkbenchExecutor._image_entrypoint("img") == ["/init", "run"]

    def test_image_entrypoint_failure_returns_empty(self):
        with mock.patch(
            "computecloud_node.workbench_executor.subprocess.run"
        ) as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=1, stdout="", stderr="x")
            assert DockerWorkbenchExecutor._image_entrypoint("img") == []



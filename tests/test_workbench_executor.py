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

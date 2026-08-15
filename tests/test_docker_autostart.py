"""Unit tests for Docker auto-start (no real Docker needed — subprocess mocked)."""

from __future__ import annotations

import sys
from unittest.mock import patch

from computecloud_node.docker_executor import (
    DockerExecutor,
    _launch_docker_desktop,
    ensure_docker_running,
)


class TestEnsureDockerRunning:
    """ensure_docker_running() — fully mocked, no real Docker needed."""

    def test_already_running_short_circuits(self):
        with (
            patch.object(DockerExecutor, "is_docker_available", return_value=True),
            patch("computecloud_node.docker_executor._launch_docker_desktop") as launch,
        ):
            assert ensure_docker_running(timeout_seconds=0) is True
            launch.assert_not_called()

    def test_launch_then_daemon_comes_up(self):
        with (
            patch.object(DockerExecutor, "is_docker_available", side_effect=[False, True]),
            patch(
                "computecloud_node.docker_executor._launch_docker_desktop",
                return_value=True,
            ),
            patch("computecloud_node.docker_executor.time.sleep"),
        ):
            assert ensure_docker_running(timeout_seconds=10) is True

    def test_timeout_returns_false(self):
        with (
            patch.object(DockerExecutor, "is_docker_available", return_value=False),
            patch(
                "computecloud_node.docker_executor._launch_docker_desktop",
                return_value=True,
            ),
        ):
            assert ensure_docker_running(timeout_seconds=0) is False

    def test_nothing_to_launch_returns_false(self):
        with (
            patch.object(DockerExecutor, "is_docker_available", return_value=False),
            patch(
                "computecloud_node.docker_executor._launch_docker_desktop",
                return_value=False,
            ) as launch,
        ):
            assert ensure_docker_running(timeout_seconds=10) is False
            launch.assert_called_once()


class TestLaunchDockerDesktop:
    def test_linux_never_launches(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        with patch("computecloud_node.docker_executor.subprocess.Popen") as popen:
            assert _launch_docker_desktop() is False
            popen.assert_not_called()

    def test_windows_launches_when_app_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        app = tmp_path / "Docker Desktop.exe"
        app.write_text("x")
        with (
            patch(
                "computecloud_node.docker_executor._find_docker_desktop_app",
                return_value=str(app),
            ),
            patch("computecloud_node.docker_executor.subprocess.Popen") as popen,
        ):
            assert _launch_docker_desktop() is True
            popen.assert_called_once()

    def test_windows_no_app_returns_false(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        with patch(
            "computecloud_node.docker_executor._find_docker_desktop_app",
            return_value=None,
        ):
            assert _launch_docker_desktop() is False

    def test_macos_requires_docker_app(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        with patch(
            "computecloud_node.docker_executor.os.path.isdir",
            return_value=False,
        ):
            assert _launch_docker_desktop() is False

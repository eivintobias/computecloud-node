"""Built-in executor that runs shell commands for the Node Client.

This module provides :class:`LocalProcessExecutor`, a concrete
:class:`~computecloud_node.executor.TaskExecutor` that takes a task
payload describing a shell command and runs it locally via
:mod:`subprocess`.

A task payload expected to look like::

    {"command": "python -c 'print(42)'", "timeout_seconds": 30}

If *command* is absent, the executor falls back to *cmd* or *shell*.
If *timeout_seconds* is absent, :attr:`default_timeout_seconds` is used.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from computecloud_node.executor import TaskExecutor


@dataclass
class CommandResult:
    """Structured outcome of a subprocess invocation."""

    return_code: int
    stdout: str
    stderr: str


class LocalProcessExecutor(TaskExecutor):
    """Execute tasks by running shell commands locally.

    The task payload is expected to contain a ``command`` key (string).
    A ``timeout_seconds`` key may be supplied per-task; otherwise the
    executor's :attr:`default_timeout_seconds` is used.

    Parameters
    ----------
    default_timeout_seconds:
        Default per-command timeout when the payload does not specify one.
    shell:
        If *True* (default) the command is passed to the system shell.
        If *False*, the command is expected to be a list of arguments and
        :func:`shutil.which` is used to resolve the executable.
        working_directory:
        Optional cwd for spawned processes.
    """

    def __init__(
        self,
        default_timeout_seconds: float = 300.0,
        shell: bool = True,
        working_directory: str | None = None,
    ) -> None:
        self.default_timeout_seconds = default_timeout_seconds
        self.shell = shell
        self.working_directory = working_directory

    def execute(
        self,
        task_id: str,
        job_id: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Run the shell command described by *payload*.

        Returns a dict with ``return_code``, ``stdout``, ``stderr``, and
        ``command`` keys.  On success (return-code 0) the result is
        returned normally; on failure (non-zero return-code or timeout)
        the exception propagates so that :class:`ComputeNode` converts it
        into a failed :class:`~computecloud_node.executor.TaskResult`.
        """
        command = self._extract_command(payload)
        timeout = self._extract_timeout(payload)
        result = self._run(command, timeout)
        return {
            "command": command,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_command(payload: dict[str, Any] | None) -> str:
        """Pull the command string out of the task payload."""
        if payload is None:
            raise ValueError("LocalProcessExecutor requires a 'command' in the task payload")
        for key in ("command", "cmd", "shell"):
            value = payload.get(key)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    return " ".join(str(v) for v in value)
                return str(value)
        raise ValueError(
            "LocalProcessExecutor: payload must contain a 'command' key"
        )

    def _extract_timeout(self, payload: dict[str, Any] | None) -> float:
        """Determine the timeout for this task."""
        if payload and "timeout_seconds" in payload:
            try:
                return float(payload["timeout_seconds"])
            except (TypeError, ValueError):
                pass
        return self.default_timeout_seconds

    def _run(self, command: str, timeout: float) -> CommandResult:
        """Execute *command* and capture its output."""
        if self.shell:
            args: str | list[str] = command
            use_shell = True
        else:
            args = command.split()
            executable = args[0]
            resolved = shutil.which(executable)
            if resolved is None:
                raise FileNotFoundError(
                    f"Executable not found: {executable}"
                )
            args = [resolved] + args[1:]
            use_shell = False

        try:
            completed = subprocess.run(
                args,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout
            if isinstance(stdout, bytes):
                stdout = stdout.decode()
            return CommandResult(
                return_code=-1,
                stdout=stdout or "",
                stderr=(
                    f"Command timed out after {timeout}s\n{exc.stderr}"
                    if exc.stderr
                    else f"Command timed out after {timeout}s"
                ),
            )

        return CommandResult(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

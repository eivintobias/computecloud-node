"""ComputeNode - the main runtime loop for a worker node in the pool.

A ComputeNode connects to a pool coordinator via gRPC or HTTP, registers itself,
sends periodic heartbeats, polls for tasks, executes them via a
user-supplied TaskExecutor, and reports results back to the pool.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from computecloud_node.config import NodeConfig
from computecloud_node.executor import TaskExecutor, TaskResult

logger = logging.getLogger(__name__)


class ComputeNode:
    """Worker node runtime that participates in the ComputeCloud pool.

    Parameters
    ----------
    config:
        NodeConfig with server address, capabilities, etc.
    executor:
        Optional TaskExecutor implementation. Can also be set later.
    """

    def __init__(
        self,
        config: NodeConfig,
        executor: TaskExecutor | None = None,
        *,
        enable_pipeline: bool = True,
    ) -> None:
        self.config = config
        self._executor: TaskExecutor | None = executor
        self._base_executor: TaskExecutor | None = executor
        self._enable_pipeline = enable_pipeline
        self._shard_worker = None
        self._data_worker = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._tasks_in_flight = 0
        self._exec_pool: ThreadPoolExecutor | None = None
        self._channel = None
        self._stub = None
        self._http_client = None
        self._active_sessions: dict[str, str] = {}  # session_id -> container_id

    # -- Public API --

    @property
    def node_id(self) -> str:
        """The unique identifier for this node."""
        return self.config.node_id  # type: ignore[attr-defined]

    def set_executor(self, executor: TaskExecutor) -> None:
        """Set (or replace) the task executor implementation."""
        with self._lock:
            self._executor = executor
            self._base_executor = executor

    @staticmethod
    def _find_docker() -> str:
        """Resolve the full path to the Docker CLI executable.

        On Windows, Docker Desktop installs docker.exe in a per-user
        directory that may not be on the PATH when Python spawns
        subprocesses.  We search common install locations as a fallback.

        Returns the full path to docker.exe (Windows) or 'docker'
        (Linux/macOS, where it's always on PATH).
        """
        import os
        import shutil
        import sys

        # On non-Windows, just use 'docker' (always on PATH).
        if sys.platform != "win32":
            return "docker"

        # First try the standard PATH lookup.
        path = shutil.which("docker")
        if path:
            return path

        # Docker Desktop common install locations (per-user and system-wide).
        candidates = [
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Programs", "DockerDesktop", "resources", "bin", "docker.exe",
            ),
            os.path.join(
                os.environ.get("ProgramFiles", ""),
                "Docker", "Docker", "resources", "bin", "docker.exe",
            ),
            os.path.join(
                os.environ.get("ProgramFiles(x86)", ""),
                "Docker", "Docker", "resources", "bin", "docker.exe",
            ),
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c

        # Last resort: just use 'docker' and let it fail with a clear error.
        return "docker"

    def run(self) -> None:
        """Register with the pool, then poll-execute-report in a loop.

        Blocks until :meth:`stop` is called (or SIGINT/SIGTERM received).
        """
        # Signal handlers can only be installed from the main thread.
        # When run() is invoked from a background thread (embedded usage,
        # tests), skip them — the caller controls shutdown via stop().
        if threading.current_thread() is threading.main_thread():
            import signal

            signal.signal(signal.SIGINT, lambda *_: self.stop())
            signal.signal(signal.SIGTERM, lambda *_: self.stop())

        self._connect_and_register()
        self._running = True
        logger.info("ComputeNode %s started - polling for tasks",
                     self.config.node_id)

        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        heartbeat_thread.start()

        try:
            while not self._stop_event.is_set():
                self._poll_once()
                self._session_poll_once()
                self._stop_event.wait(self.config.poll_interval_seconds)
        finally:
            self._running = False
            self._stop()

    def stop(self) -> None:
        """Signal the node to stop polling and shut down cleanly."""
        self._stop_event.set()

    # -- gRPC setup --

    def _connect_and_register(self) -> None:
        """Connect to the pool and register the node.

        Dispatches to HTTP or gRPC transport based on ``config.use_http``.
        """
        if self.config.use_http:
            import httpx

            self._http_client = httpx.Client(
                base_url=self.config.http_base_url or "http://localhost:8000",
                timeout=30.0,
            )
        else:
            self._connect_and_register_grpc()

        # Wrap the user-supplied executor with data-shard-awareness so this node
        # can serve generalized data-run shards (Phase 14b / v0.4.0) AND legacy
        # distributed-LLM pipeline shard tasks (Phase 12 / v0.3.0).  The
        # DataShardAwareExecutor supersedes ShardAwareExecutor: it routes
        # ``data_shard`` / ``data_merge`` payloads to a DataShardWorker,
        # ``pipeline_shard`` payloads to a PipelineShardWorker, and delegates
        # everything else to the original fallback executor.  Both workers
        # poll/post via this node's HTTP client.  Only enabled in HTTP mode
        # (the standalone client's default) and only when a base executor is
        # configured.
        if self._enable_pipeline and self._http_client is not None:
            from computecloud_node.data_worker import (
                DataShardAwareExecutor,
                DataShardWorker,
            )
            from computecloud_node.pipeline_worker import PipelineShardWorker

            # Phase 16 — probe for the optional [llm] extra (torch).  When the
            # extra is installed, construct an LLMShardExecutor and wire it into
            # the DataShardAwareExecutor routing so this node can execute real
            # LLM layer shards (kind == "llm_shard").  When the extra is absent,
            # the probe returns False, no LLMShardExecutor is built, and the
            # routing behaves exactly as before (zero behavior change).  The
            # heavy import lives behind probe_llm_capable() so this code path is
            # never entered without torch — the package import stays torch-free.
            llm_executor = None
            try:
                from computecloud_node.llm.executor import (
                    LLMShardExecutor,
                    probe_llm_capable,
                )
            except ImportError:
                # The llm subpackage is part of the package but its module
                # import can only fail if the package itself is broken — keep
                # the try/except so a partial install never crashes startup.
                probe_llm_capable = None
            if probe_llm_capable is not None and probe_llm_capable():
                llm_executor = LLMShardExecutor()
                self.config.capabilities.llm_capable = True

            with self._lock:
                self._data_worker = DataShardWorker(self._http_client)
                self._shard_worker = PipelineShardWorker(self._http_client)
                self._executor = DataShardAwareExecutor(
                    self._data_worker,
                    pipeline_worker=self._shard_worker,
                    fallback=self._base_executor,
                    llm_executor=llm_executor,
                )

        response = self._register()
        if not response.accepted:
            raise RuntimeError(f"Node registration rejected: {response.message}")
        if response.node_id:
            self.config.node_id = response.node_id

    def _connect_and_register_grpc(self) -> None:
        """Open a gRPC channel for talking to the pool coordinator."""
        import grpc

        from computecloud_node.proto_stub import (
            computecloud_pb2_grpc as pb2_grpc,
        )

        server_addr = f"{self.config.server_host}:{self.config.server_port}"
        if self.config.use_tls:
            self._channel = grpc.secure_channel(
                server_addr,
                grpc.ssl_channel_credentials(),
                options=[
                    ("grpc.ssl_target_name_override", self.config.server_host),
                ],
            )
        else:
            self._channel = grpc.insecure_channel(server_addr)
        self._stub = pb2_grpc.NodeServiceStub(self._channel)

    def _register(self):
        """Register with the pool via HTTP or gRPC."""
        if self.config.use_http:
            return self._register_http()
        return self._register_grpc()

    def _register_http(self):
        """Register via HTTP POST /api/v1/node/register."""
        caps = self.config.capabilities
        body: dict = {
            "node_id": self.config.node_id or "",
            "endpoint": self.config.endpoint or "",
            "tags": list(self.config.tags),
            "cpu_cores": caps.cpu_cores,
            "memory_mb": caps.memory_mb,
            "gpu_count": caps.gpu_count,
            "disk_mb": caps.disk_mb,
            "gpu_model": caps.gpu_model or "",
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            # Phase 12 — report per-GPU VRAM so the server can register a
            # VRAMPool segment for distributed LLM pipeline participation.
            # Defaults to 0 (backward compatible with servers that ignore it).
            "vram_mb": caps.vram_mb,
            # Phase 16 — advertise LLM capability (additive, backward
            # compatible).  True when the [llm] extra probed successfully at
            # startup; the server can then assign real LLM layer shards to
            # this node.  Older servers ignore the field.
            "llm_capable": bool(getattr(caps, "llm_capable", False)),
        }
        # If username/password are provided, include them so the server
        # can associate this node with the user's account.
        if self.config.username and self.config.password:
            body["username"] = self.config.username
            body["password"] = self.config.password
        resp = self._http_client.post("/api/v1/node/register", json=body)
        resp.raise_for_status()
        data = resp.json()
        from computecloud_node.config import NodeConfig as _NC  # noqa
        from dataclasses import dataclass as _dc

        @_dc
        class _RegResp:
            accepted: bool
            node_id: str
            message: str

        return _RegResp(
            accepted=data["accepted"],
            node_id=data.get("node_id", ""),
            message=data.get("message", ""),
        )

    def _register_grpc(self):
        """Register via gRPC RegisterNode RPC."""
        from computecloud_node.proto_stub import (
            computecloud_pb2 as pb2,
        )

        caps = self.config.capabilities
        resources = pb2.ResourceSpec(
            cpu_cores=caps.cpu_cores,
            memory_mb=caps.memory_mb,
            gpu_count=caps.gpu_count,
            disk_mb=caps.disk_mb,
            gpu_model=caps.gpu_model or "",
        )
        node_info = pb2.NodeInfo(
            node_id=self.config.node_id or "",
            endpoint=self.config.endpoint or "",
            tags=list(self.config.tags),
            capabilities=resources,
            max_concurrent_tasks=self.config.max_concurrent_tasks,
        )
        request = pb2.RegisterNodeRequest(
            api_key=self.config.api_key or "",
            node=node_info,
        )
        return self._stub.RegisterNode(request, timeout=10.0)

    # -- Heartbeating --

    def _heartbeat_loop(self) -> None:
        """Background thread that sends periodic heartbeats.

        If the pool responds NOT_FOUND (e.g. the coordinator restarted and
        lost its in-memory registry), the node automatically re-registers
        instead of erroring forever.
        """
        if self.config.use_http:
            self._heartbeat_loop_http()
        else:
            self._heartbeat_loop_grpc()

    def _heartbeat_loop_http(self) -> None:
        """HTTP heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                resp = self._http_client.post(
                    f"/api/v1/node/{self.config.node_id}/heartbeat"
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("acknowledged"):
                    logger.warning("Heartbeat not acknowledged, re-registering...")
                    reg = self._register()
                    if reg.accepted:
                        logger.info("Re-registered as %s", self.config.node_id)
            except Exception:
                logger.exception("Heartbeat (HTTP) failed")
            self._stop_event.wait(self.config.heartbeat_interval_seconds)

    def _heartbeat_loop_grpc(self) -> None:
        """gRPC heartbeat loop (original implementation)."""
        import grpc

        while not self._stop_event.is_set():
            try:
                self._send_heartbeat()
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.NOT_FOUND:
                    logger.warning(
                        "Pool no longer knows node %s (coordinator restart?) "
                        "- re-registering",
                        self.config.node_id,
                    )
                    try:
                        response = self._register_grpc()
                        if response.accepted:
                            logger.info(
                                "Re-registered with pool as %s",
                                self.config.node_id,
                            )
                        else:
                            logger.error(
                                "Re-registration rejected: %s", response.message
                            )
                    except Exception:
                        logger.exception("Re-registration failed")
                else:
                    logger.exception("Heartbeat failed")
            except Exception:
                logger.exception("Heartbeat failed")
            self._stop_event.wait(self.config.heartbeat_interval_seconds)

    def _send_heartbeat(self) -> None:
        """Send a single Heartbeat RPC to the pool."""
        from google.protobuf.timestamp_pb2 import Timestamp

        from computecloud_node.proto_stub.computecloud_pb2 import (
            HeartbeatRequest,
        )

        ts = Timestamp()
        ts.GetCurrentTime()
        self._stub.Heartbeat(
            HeartbeatRequest(
                node_id=self.config.node_id,
                timestamp=ts,
            ),
            timeout=5.0,
        )

    # -- Task polling --

    def _poll_once(self) -> None:
        """Pull one task and dispatch it to the executor thread pool."""
        if self._tasks_in_flight >= self.config.max_concurrent_tasks:
            return

        if self.config.use_http:
            self._poll_once_http()
        else:
            self._poll_once_grpc()

    def _poll_once_http(self) -> None:
        """Pull a task via HTTP POST /api/v1/node/{node_id}/pull."""
        try:
            resp = self._http_client.post(
                f"/api/v1/node/{self.config.node_id}/pull"
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("PullTask (HTTP) failed")
            return

        if not data.get("has_task"):
            return

        with self._lock:
            self._tasks_in_flight += 1

        if self._exec_pool is None:
            self._exec_pool = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_tasks,
                thread_name_prefix="exec",
            )

        self._exec_pool.submit(self._execute_task, data)

    def _poll_once_grpc(self) -> None:
        """Pull a task via gRPC (original implementation)."""
        from computecloud_node.proto_stub.computecloud_pb2 import (
            PullTaskRequest,
        )

        try:
            response = self._stub.PullTask(
                PullTaskRequest(node_id=self.config.node_id),
                timeout=5.0,
            )
        except Exception:
            logger.exception("PullTask RPC failed")
            return

        if not response.has_task:
            return

        task = response.task
        with self._lock:
            self._tasks_in_flight += 1

        if self._exec_pool is None:
            self._exec_pool = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_tasks,
                thread_name_prefix="exec",
            )

        self._exec_pool.submit(self._execute_task, task)

    def _execute_task(self, task) -> None:
        """Execute a single pulled task and report the outcome.

        ``task`` is either a dict (HTTP mode) or a proto TaskAssignment (gRPC).
        """
        import time as _time

        start = _time.monotonic()
        executor = self._executor
        result: TaskResult

        # Extract fields from either a dict (HTTP) or proto (gRPC) task object.
        if isinstance(task, dict):
            task_id = task.get("task_id", "")
            job_id = task.get("job_id", "")
            payload = task.get("payload")
        else:
            task_id = task.task_id
            job_id = task.job_id
            payload = None
            if task.payload and task.payload.fields:
                from google.protobuf.json_format import MessageToDict

                payload = MessageToDict(task.payload)

        if executor is None:
            result = TaskResult(
                success=False,
                error="No executor set - call set_executor() before running.",
                execution_time_seconds=_time.monotonic() - start,
            )
        else:
            try:
                outcome = executor.execute(
                    task_id=task_id,
                    job_id=job_id,
                    payload=payload,
                )
                result = TaskResult(
                    success=True,
                    result=outcome,
                    execution_time_seconds=_time.monotonic() - start,
                )
            except Exception as exc:
                logger.exception("Task %s failed", task_id)
                result = TaskResult(
                    success=False,
                    error=str(exc),
                    execution_time_seconds=_time.monotonic() - start,
                )

        self._report_result(task_id, result)

        with self._lock:
            self._tasks_in_flight -= 1

    def _report_result(self, task_id: str, result: TaskResult) -> None:
        """Send a task result to the pool via HTTP or gRPC."""
        if self.config.use_http:
            self._report_result_http(task_id, result)
        else:
            self._report_result_grpc(task_id, result)

    def _report_result_http(self, task_id: str, result: TaskResult) -> None:
        """Send a task result via HTTP POST /api/v1/node/{node_id}/result."""
        body = {
            "node_id": self.config.node_id,
            "task_id": task_id,
            "success": result.success,
            "error": result.error,
            "result": result.result,
            "execution_time_seconds": result.execution_time_seconds,
        }
        resp = self._http_client.post(
            f"/api/v1/node/{self.config.node_id}/result", json=body
        )
        resp.raise_for_status()

    def _report_result_grpc(self, task_id: str, result: TaskResult) -> None:
        """Send a ReportResult RPC for a completed task."""
        from computecloud_node.proto_stub.computecloud_pb2 import (
            ReportResultRequest,
        )
        from computecloud_node.proto_stub.computecloud_pb2 import (
            TaskResult as PbTaskResult,
        )

        proto_result = PbTaskResult(
            task_id=task_id,
            success=result.success,
            error=result.error or "",
        )
        if result.result:
            proto_result.result.update(result.result)

        self._stub.ReportResult(
            ReportResultRequest(
                node_id=self.config.node_id,
                result=proto_result,
            ),
            timeout=10.0,
        )

    # -- Shutdown --

    def _stop(self) -> None:
        """Clean up resources - channel/client and executor pool."""
        if self._exec_pool is not None:
            self._exec_pool.shutdown(wait=True)
            self._exec_pool = None
        if self._channel is not None:
            self._channel.close()
        if self._http_client is not None:
            self._http_client.close()

    # -- Session lifecycle (HTTP only) --

    def _session_poll_once(self) -> None:
        """Pull a session assignment from the pool (HTTP only)."""
        if not self.config.use_http:
            return
        try:
            resp = self._http_client.post(
                f"/api/v1/node/{self.config.node_id}/session/pull"
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return

        if not data.get("has_session"):
            return

        session_id = data.get("session_id", "")
        if session_id in self._active_sessions:
            return

        thread = threading.Thread(
            target=self._run_session_container,
            args=(data,),
            daemon=True,
            name=f"session-{session_id[:8]}",
        )
        thread.start()

    def _run_session_container(self, session_data: dict) -> None:
        """Start a workbench session via the pluggable WorkbenchExecutor.

        Phase 17d: the Docker-specific launch logic now lives in a pluggable
        executor (DockerWorkbenchExecutor / NativeWorkbenchExecutor).  The
        relay/tunnel wiring below is executor-agnostic — it only needs the
        host port from the SessionHandle.
        """
        import time as _time

        session_id = session_data["session_id"]

        from computecloud_node.workbench_executor import create_workbench_executor

        try:
            executor = create_workbench_executor(
                getattr(self.config, "workbench_executor", "auto"),
                http_client=self._http_client,
            )
        except Exception as exc:
            logger.error("Workbench executor init failed: %s", exc)
            self._report_session_terminated(session_id, "failed", str(exc))
            return

        payload = dict(session_data)
        payload.setdefault(
            "_http_base_url", str(self._http_client.base_url).rstrip("/")
        )

        try:
            handle = executor.start_session(payload)
        except Exception as exc:
            logger.error("Session %s start failed: %s", session_id[:8], exc)
            self._report_session_terminated(session_id, "failed", str(exc))
            return

        host_port = handle.host_port
        self._active_sessions[session_id] = handle

        try:
            resp = self._http_client.post(
                f"/api/v1/node/{self.config.node_id}/session/ready",
                json={
                    "session_id": session_id,
                    "host_port": host_port,
                    "auth_token": handle.auth_token,
                },
            )
            resp.raise_for_status()
        except Exception:
            try:
                executor.stop_session(handle)
            except Exception:
                pass
            self._report_session_terminated(session_id, "failed", "report ready failed")
            return

        logger.info("Session %s ready on port %d", session_id[:8], host_port)

        # Start the WebSocket reverse-tunnel (NAT-proof relay).
        tunnel_thread = threading.Thread(
            target=self._run_tunnel,
            args=(session_id, host_port, session_data.get("tunnel_token", "")),
            daemon=True,
            name=f"tunnel-{session_id[:8]}",
        )
        tunnel_thread.start()

        security_monitor = None
        security_scan_interval = 60.0
        last_scan_time = _time.monotonic()
        if getattr(handle, "executor_type", "") == "docker":
            try:
                from computecloud_node.security_monitor import SecurityMonitor

                security_monitor = SecurityMonitor(
                    cpu_threshold_percent=95.0,
                    sustained_checks=10,
                )
            except Exception:
                security_monitor = None

        while not self._stop_event.is_set():
            try:
                resp = self._http_client.post(
                    f"/api/v1/node/{self.config.node_id}/session/heartbeat",
                    json={"session_id": session_id},
                )
                resp.raise_for_status()
                hb = resp.json()
                if hb.get("should_terminate"):
                    break
            except Exception:
                logger.exception("Session heartbeat failed for %s", session_id[:8])

            try:
                alive = executor.is_session_alive(handle)
            except Exception:
                alive = False
            if not alive:
                self._report_session_terminated(
                    session_id, "completed", "Session process exited"
                )
                self._active_sessions.pop(session_id, None)
                return

            now = _time.monotonic()
            if security_monitor is not None and getattr(handle, "container_id", None):
                if now - last_scan_time >= security_scan_interval:
                    last_scan_time = now
                    try:
                        alert = security_monitor.scan_container(handle.container_id)
                    except Exception:
                        alert = None
                    if alert is not None:
                        logger.warning(
                            "Security alert for session %s: %s — %s",
                            session_id[:8],
                            alert.alert_type,
                            alert.details,
                        )
                        self._report_security_alert(session_id, alert)
                        if alert.severity == "critical":
                            try:
                                executor.stop_session(handle)
                            except Exception:
                                pass
                            self._report_session_terminated(
                                session_id, alert.alert_type, alert.details
                            )
                            self._active_sessions.pop(session_id, None)
                            return

            self._stop_event.wait(10.0)

        try:
            executor.stop_session(handle)
        except Exception:
            pass
        self._report_session_terminated(session_id, "completed", "Terminated by renter")
        self._active_sessions.pop(session_id, None)


    def _run_tunnel(self, session_id: str, host_port: int, tunnel_token: str = "") -> None:
        """Run a WebSocket reverse-tunnel to the server (NAT-proof).

        The node opens a persistent WebSocket to the server at
        /api/v1/nodes/{node_id}/tunnel/{session_id}.  Data from the
        server WebSocket is forwarded to the local Docker container via
        TCP, and data from the container is sent back through the WS.
        """
        import asyncio

        # Build the WebSocket URL from the HTTP base URL.
        base_url = str(self._http_client.base_url).rstrip("/")
        if base_url.startswith("https://"):
            ws_url = "wss://" + base_url[len("https://"):]
        elif base_url.startswith("http://"):
            ws_url = "ws://" + base_url[len("http://"):]
        else:
            ws_url = "ws://" + base_url

        tunnel_path = f"/api/v1/node/{self.config.node_id}/tunnel/{session_id}"
        ws_full_url = ws_url + tunnel_path
        # Phase 18a: present the per-session tunnel token (hijack protection).
        if tunnel_token:
            ws_full_url += f"?token={tunnel_token}"

        async def _tunnel_async():
            import websockets

            logger.info("Starting tunnel for session %s → %s", session_id[:8], ws_full_url)

            # Reconnect loop (in case the connection drops).
            while not self._stop_event.is_set():
                try:
                    async with websockets.connect(
                        ws_full_url,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=5,
                        max_size=2**24,  # 16 MB max message
                    ) as ws:
                        logger.info("Tunnel connected for session %s", session_id[:8])

                        # Connect to the local container port.
                        try:
                            reader, writer = await asyncio.open_connection(
                                "127.0.0.1", host_port
                            )
                        except Exception:
                            logger.exception(
                                "Cannot connect to container port %d for session %s",
                                host_port, session_id[:8],
                            )
                            return

                        async def ws_to_tcp():
                            """Server → WebSocket → container TCP."""
                            try:
                                while True:
                                    data = await ws.recv()
                                    writer.write(
                                        data if isinstance(data, bytes) else data.encode()
                                    )
                                    await writer.drain()
                            except Exception:
                                pass
                            finally:
                                writer.close()

                        async def tcp_to_ws():
                            """Container TCP → WebSocket → server."""
                            try:
                                while True:
                                    data = await reader.read(4096)
                                    if not data:
                                        logger.info(
                                            "Container connection closed for session %s"
                                            " (service may still be starting)",
                                            session_id[:8],
                                        )
                                        break
                                    await ws.send(data)
                            except Exception:
                                pass

                        # Phase 18b: FIRST_COMPLETED — if the container
                        # connection EOFs while no renter data flows, gather()
                        # would hang forever on the other pump and the tunnel
                        # would never reconnect (half-dead tunnel, silent
                        # relay).  Ending either pump exits the context, and
                        # the outer loop reconnects with a fresh container
                        # connection (fresh service banner for the next renter).
                        tasks = [
                            asyncio.ensure_future(ws_to_tcp()),
                            asyncio.ensure_future(tcp_to_ws()),
                        ]
                        _, pending = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for t in pending:
                            t.cancel()

                except Exception as exc:
                    if not self._stop_event.is_set():
                        logger.warning(
                            "Tunnel for session %s disconnected: %s, reconnecting...",
                            session_id[:8], exc,
                        )
                        import time as _time
                        _time.sleep(2.0)  # backoff before reconnect

        try:
            asyncio.run(_tunnel_async())
        except Exception:
            logger.exception("Tunnel coroutine crashed for session %s", session_id[:8])

    @staticmethod
    def _stop_container(container_id: str) -> None:
        """Stop and remove a Docker container with secure data deletion."""
        import subprocess

        docker_bin = ComputeNode._find_docker()

        # Stop the container.
        try:
            subprocess.run(
                [docker_bin, "stop", container_id],
                capture_output=True, text=True, timeout=10.0,
            )
        except Exception:
            pass
        # Remove the container and its volumes (-v) for secure data deletion.
        try:
            subprocess.run(
                [docker_bin, "rm", "-f", "-v", container_id],
                capture_output=True, text=True, timeout=10.0,
            )
            logger.info("Container %s stopped and volumes removed (secure deletion)",
                        container_id[:12])
        except Exception:
            pass

    def _report_session_terminated(
        self, session_id: str, reason: str, message: str
    ) -> None:
        """Tell the server that a session container has stopped."""
        try:
            self._http_client.post(
                f"/api/v1/node/{self.config.node_id}/session/terminated",
                json={
                    "session_id": session_id,
                    "reason": reason,
                    "message": message,
                },
            )
        except Exception:
            logger.exception(
                "Failed to report session terminated for %s", session_id[:8]
            )

    def _report_security_alert(self, session_id: str, alert) -> None:
        """Report a security alert to the server (Tier 3)."""
        try:
            self._http_client.post(
                f"/api/v1/node/{self.config.node_id}/security/alert",
                json={
                    "session_id": session_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "details": alert.details,
                    "process_names": alert.process_names,
                    "cpu_percent": alert.cpu_percent,
                },
            )
        except Exception:
            logger.exception("Failed to report security alert for %s", session_id[:8])

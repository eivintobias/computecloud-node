"""CLI entry point for the ComputeCloud Node Client.

Run ``python -m computecloud.client`` on a worker node to start polling
the pool coordinator for tasks and executing them with the configured
executor.

Configuration is via environment variables:

    COMPUTECLOUD_SERVER_HOST    Coordinator host       (default: localhost)
    COMPUTECLOUD_SERVER_PORT    Coordinator port       (default: 50051)
    COMPUTECLOUD_API_KEY        Optional API key
    COMPUTECLOUD_MAX_TASKS      Max concurrent tasks   (default: 4)
    COMPUTECLOUD_EXECUTOR       Executor entry point  (default: echo)

The default ``echo`` executor returns the task payload back as the result
— useful for smoke-testing pool connectivity without writing custom logic.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def _resolve_executor(name: str):
    """Resolve an executor name to an instance.

    Built-in:
        ``echo``    Returns the payload as-is (for smoke tests).
        ``print``   Prints the payload and returns ``{"received": true}``.
        ``local``   Runs a shell command from the task payload (see
                    :class:`~computecloud_node.local_executor.LocalProcessExecutor`).

    Anything else: treated as an import path ``module.attr`` (e.g.
    ``mypkg.MyExecutor``) that is instantiated with no arguments.
    """
    if name == "echo":
        from computecloud_node.executor import TaskExecutor

        class _Echo(TaskExecutor):
            def execute(self, task_id, job_id, payload):  # type: ignore[override]
                return {"echo": payload}

        return _Echo()

    if name == "local":
        from computecloud_node.local_executor import LocalProcessExecutor

        return LocalProcessExecutor()

    if name == "docker":
        from computecloud_node.docker_executor import DockerExecutor

        if not DockerExecutor.is_docker_available():
            raise RuntimeError(
                "Docker is not available on this machine. Install Docker "
                "Desktop (and start it) to accept template jobs, or use "
                "--executor local for plain shell tasks."
            )
        return DockerExecutor()

    if name == "auto":
        # Auto mode: template jobs (payload has 'image') go to Docker when
        # available; plain shell jobs go to LocalProcessExecutor.
        from computecloud_node.executor import TaskExecutor
        from computecloud_node.local_executor import LocalProcessExecutor

        local = LocalProcessExecutor()
        docker = None
        from computecloud_node.docker_executor import DockerExecutor

        if DockerExecutor.is_docker_available():
            docker = DockerExecutor()
            print("Docker detected -- template jobs enabled")
        else:
            print("Docker not detected -- template jobs will fail; shell jobs OK")

        class _Auto(TaskExecutor):
            def execute(self, task_id, job_id, payload):  # type: ignore[override]
                if payload and payload.get("image"):
                    if docker is None:
                        raise RuntimeError(
                            "This node has no Docker -- cannot run template "
                            f"job with image {payload['image']!r}"
                        )
                    return docker.execute(task_id, job_id, payload)
                return local.execute(task_id, job_id, payload)

        return _Auto()

    if name == "print":
        from computecloud_node.executor import TaskExecutor

        class _Print(TaskExecutor):
            def execute(self, task_id, job_id, payload):  # type: ignore[override]
                print(f"[task {task_id}] payload: {payload!r}")
                return {"received": True}

        return _Print()

    # Treat as dotted import path
    if "." not in name:
        raise ValueError(
            f"Unknown executor '{name}'. "
            f"Use 'echo', 'local', 'docker', 'auto', 'print', "
            f"or a dotted path like 'mypkg.MyExecutor'."
        )
    module_path, _, attr = name.rpartition(".")
    __import__(module_path)
    mod = sys.modules[module_path]
    cls = getattr(mod, attr)
    return cls()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m computecloud_node",
        description="ComputeCloud worker node client",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("COMPUTECLOUD_SERVER_HOST", "localhost"),
        help="Pool coordinator host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COMPUTECLOUD_SERVER_PORT", "50051")),
        help="Pool coordinator port (default: 50051)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("COMPUTECLOUD_API_KEY"),
        help="Optional API key for the pool",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=int(os.environ.get("COMPUTECLOUD_MAX_TASKS", "4")),
        help="Max concurrent tasks (default: 4)",
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        default=False,
        help="Use TLS (gRPC over HTTPS) — for Cloudflare Tunnel connections",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        default=False,
        help="Use HTTP instead of gRPC — for cloud deployments (e.g. Render)",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("COMPUTECLOUD_API_URL", ""),
        help="Pool server HTTP API URL (e.g. https://api.computepool.cloud)",
    )
    parser.add_argument(
        "--executor",
        default=os.environ.get("COMPUTECLOUD_EXECUTOR", "echo"),
        help=(
            "Executor: 'echo', 'local', 'docker', 'auto' (docker for template "
            "jobs + local shell otherwise), 'print', or a dotted import path "
            "(default: echo)"
        ),
    )
    # ── Resource flags (user-friendly naming) ───────────────────────────
    # --cpu / --ram / --gpu / --disk are the preferred flags for new users.
    # --cpu-cores / --memory-mb are kept for backward compatibility.
    parser.add_argument(
        "--cpu",
        type=float,
        default=None,
        help="CPU cores to share (e.g. --cpu 4). Alias for --cpu-cores.",
    )
    parser.add_argument(
        "--cpu-cores",
        type=float,
        default=float(os.environ.get("COMPUTECLOUD_CPU_CORES", "4")),
        help="CPU cores to share (default: 4)",
    )
    parser.add_argument(
        "--ram",
        type=int,
        default=None,
        help="RAM to share in GB (e.g. --ram 16). Converted to MB internally.",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=int(os.environ.get("COMPUTECLOUD_MEMORY_MB", "4096")),
        help="RAM to share in MB (default: 4096)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=int(os.environ.get("COMPUTECLOUD_GPU_COUNT", "0")),
        help="Number of GPUs to share (e.g. --gpu 1, default: 0)",
    )
    parser.add_argument(
        "--gpu-model",
        default=os.environ.get("COMPUTECLOUD_GPU_MODEL"),
        help="GPU model to advertise (e.g. 'RTX 4090')",
    )
    parser.add_argument(
        "--disk",
        type=int,
        default=None,
        help="Disk space to share in GB (e.g. --disk 100). Converted to MB internally.",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("COMPUTECLOUD_USERNAME", ""),
        help="Username for associating this node with your account (enables "
        "per-user contribution tracking and payouts). Same as web login.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("COMPUTECLOUD_PASSWORD", ""),
        help="Password for the username above.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags to advertise to the pool (e.g. gpu spot)",
    )
    args = parser.parse_args(argv)

    # Resolve aliases: --cpu overrides --cpu-cores, --ram overrides --memory-mb
    cpu_cores = args.cpu if args.cpu is not None else args.cpu_cores
    memory_mb = args.ram * 1024 if args.ram is not None else args.memory_mb
    disk_mb = args.disk * 1024 if args.disk is not None else 8192

    # Import from the standalone package (not the private computecloud package)
    from computecloud_node import ComputeNode, NodeCapabilities, NodeConfig

    config = NodeConfig(
        server_host=args.host,
        server_port=args.port,
        api_key=args.api_key,
        tags=args.tags,
        capabilities=NodeCapabilities(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_count=args.gpu,
            disk_mb=disk_mb,
            gpu_model=args.gpu_model,
        ),
        max_concurrent_tasks=args.max_tasks,
        use_tls=args.tls,
        use_http=args.http,
        http_base_url=args.api_url,
        username=args.username,
        password=args.password,
    )

    executor = _resolve_executor(args.executor)
    node = ComputeNode(config, executor=executor)

    print(f"Starting ComputeNode {config.node_id} -> {config.server_host}:{config.server_port}")
    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        node.stop()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

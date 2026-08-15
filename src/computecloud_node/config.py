"""Node Client configuration for the ComputeCloud marketplace.

A :class:`NodeConfig` captures everything a worker node needs to know to
connect to a pool coordinator over gRPC.
"""

from __future__ import annotations

import secrets
import socket
from dataclasses import dataclass, field


@dataclass
class NodeCapabilities:
    """Static resource description for a compute node.

    Mirrors :class:`computecloud.workpool.resources.NodeCapabilities` but is
    kept independently for the client SDK so it can be populated *before*
    the core library is imported (useful for standalone node installations).
    """

    cpu_cores: float = 4.0
    memory_mb: int = 4096
    gpu_count: int = 0
    disk_mb: int = 8192
    gpu_model: str | None = None
    # Phase 11/12 — per-GPU VRAM in MB.  When > 0 (and gpu_count > 0) the
    # server registers a VRAMPool segment so this node can participate in
    # distributed LLM pipeline runs.  Defaults to 0 (no VRAM contribution).
    vram_mb: int = 0
    # Phase 16 — whether this node has the optional [llm] extra installed
    # (torch + safetensors + tokenizers) and can execute real LLM layer shards.
    # Defaults to False; set to True when the llm extra probes successfully.
    # Additive, backward compatible (older servers ignore the field).
    llm_capable: bool = False


@dataclass
class NodeConfig:
    """Configuration for a :class:`~computecloud_node.node.ComputeNode`.

    Attributes
    ----------
    server_host / server_port:
        gRPC server address of the pool coordinator.
    api_key:
        Optional API key for authenticating with the pool.
    node_id:
        Unique identifier for this node.  Auto-generated from the hostname
        + random suffix when ``None``.
    endpoint:
        Publicly-reachable endpoint (``host:port``) for this node.  Auto-
        derived from the local hostname when ``None``.
    tags:
        Free-form labels (e.g. ``["gpu", "spot"]``).
    capabilities:
        Resource specification advertised to the pool.
    max_concurrent_tasks:
        How many tasks this node can run in parallel.
    heartbeat_interval_seconds:
        Interval for sending Heartbeat RPCs.
    poll_interval_seconds:
        Interval between PullTask RPCs when the queue is empty.
    """

    server_host: str = "localhost"
    server_port: int = 50051
    api_key: str | None = None
    node_id: str | None = None
    endpoint: str | None = None
    tags: list[str] = field(default_factory=list)
    capabilities: NodeCapabilities = field(default_factory=NodeCapabilities)
    max_concurrent_tasks: int = 4
    heartbeat_interval_seconds: float = 10.0
    poll_interval_seconds: float = 1.0
    use_tls: bool = False
    # HTTP transport mode (for cloud deployments where gRPC port isn't exposed).
    # When True, the client talks to the pool server's HTTP API instead of gRPC.
    use_http: bool = False
    # Base URL of the pool server's HTTP API (e.g. "https://api.computepool.cloud").
    # Only used when use_http=True.
    http_base_url: str = ""
    # Username/password for authenticating the node with the pool.
    # When set, the node is associated with the user's account so that
    # contributions and payouts can be tracked per-user.
    username: str = ""
    password: str = ""

    # Phase 17d: workbench executor mode (auto / docker / native).
    workbench_executor: str = "auto"

    def __post_init__(self) -> None:
        if self.node_id is None:
            hostname = socket.gethostname()
            rand = secrets.token_hex(4)
            self.node_id = f"{hostname}-{rand}"
        if self.endpoint is None:
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except OSError:
                local_ip = "127.0.0.1"
            self.endpoint = f"{local_ip}:{self.server_port}"

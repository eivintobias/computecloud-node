# ComputeCloud Node Client — Share Your Compute

This is a lightweight, standalone package for contributor nodes to join the
ComputeCloud pool at **computepool.cloud** — without cloning the entire repo.

## What's New in v0.6.5

- **Fix: SSH workbenches with injected SSH keys never started sshd (Phase
  18d)**.  When a session carried `ssh_public_keys`, the key-injection startup
  script *replaced* the image's own entrypoint with `tail -f /dev/null` — the
  container stayed "running" but nothing ever listened.  The node now resolves
  the image's real ENTRYPOINT via `docker image inspect` and execs it after
  the startup script (`exec /init` on linuxserver images), so s6 + sshd start
  normally.  Workbenches without keys were unaffected.

## What's New in v0.6.4

- **Session self-healing + diagnostics (Phase 18c)**: before reporting a
  workbench "ready", the node now *deep-verifies* the service (for SSH: the
  `SSH-...` banner must actually arrive — a bare TCP accept from Docker's
  port proxy is no longer enough).  If the probe fails, the node walks an
  automatic ladder: retry (slow service boot) → restart the container →
  `docker pull` the image and retry (fixes stale/corrupt caches) → give up
  and report the failure **with the container log tail attached**, visible in
  the session's dashboard message.  "Ready but silent" sessions should no
  longer be possible.

## What's New in v0.6.3

- **Docker Desktop auto-start**: when the node launches with `--executor docker`
  or `--executor auto` and the Docker daemon is not responding, the client now
  tries to launch Docker Desktop (Windows/macOS) and waits up to 2 minutes for
  the engine to come up before giving up.  This fixes workbench failures (e.g.
  SSH doors) on nodes where Docker Desktop was installed but not running.
  Disable with `--no-docker-autostart` (or `COMPUTECLOUD_DOCKER_AUTOSTART=0`).
  On Linux the daemon is a system service — start it with
  `sudo systemctl start docker` instead.
- **Tunnel authentication (Phase 18a)**: the node now presents the per-session
  tunnel token (from the session/pull payload) when opening the reverse
  tunnel WebSocket, so only the assigned node can pipe a session's traffic.
  Older servers ignore it — fully backward compatible.

## What's New in v0.6.2

- **Notebook door support (Phase 17c)**: the node now drops a starter
  `welcome.ipynb` into `/workspace` when a Jupyter workbench session starts.
  The notebook contains a markdown intro + pre-filled cells demonstrating
  `pool.status()`, `pool.map()`, `pool.run()`, and `pool.generate()` — all
  using the pre-authenticated Pool SDK (env vars injected by Phase 17b).
- **Jupyter root dir = /workspace**: the Jupyter session templates now set
  `--ServerApp.root_dir=/workspace` so the workspace is the notebook's file
  root (files you create are saved to your workspace).

## What's New in v0.6.0

- **Workbench door support (Phase 17b)**: the node now supports
  workspace-linked sessions — SSH/Jupyter sessions that are interactive
  entry points into the user's workspace.  When a session has a
  `workspace` block in the pull payload, the node:
  - Injects `POOL_API_URL`, `POOL_TOKEN`, `POOL_WORKSPACE_ID` env vars.
  - Best-effort `pip install computecloud-sdk` (images without Python/pip
    skip it — SSH still works).
  - Syncs workspace files into `/workspace` via the existing workspace
    HTTP endpoints (NAT-proof — no new file-transfer protocol).
  - Installs a `/usr/local/bin/pool-push` helper script for pushing
    results back.
  - Injects the user's SSH public keys into `authorized_keys` for
    passwordless auth.

## What's New in v0.5.0

- **Serve real LLM layer shards**: nodes can now execute **real** LLM
  transformer layer shards (download only your layer range's weights from
  Hugging Face, or read local `.safetensors` files), run a forward pass over
  those layers, and exchange activations via the stdlib-only
  `TensorPayload` wire format.  This is the Phase 16 sync of the in-repo Phase
  15b/15c LLM capability into the standalone package.
- **Install the optional `[llm]` extra** to enable it::

      pip install -e ".[llm]"        # from a local clone
      # or, for end users:
      pip install --upgrade "computecloud-node[llm]"

  This pulls in `torch`, `safetensors`, and `tokenizers` (the heavy ML deps).
  Without the extra, the node behaves **exactly as v0.4.0** — `import
  computecloud_node` works without torch, and the routing never touches the LLM
  path.  The extra is **lazily** imported: no heavy module is loaded at package
  import time, only inside the shard executor's methods.
- **GPU auto-detected, fp16 on GPU**: the shard executor picks `fp16` when a
  CUDA GPU is available and `fp32` otherwise (overridable via the module's
  `force_dtype`).  Shard weights are loaded **partially** — only the node's
  assigned layer range — so a 7B model split across 4 nodes each downloads
  ~1/4 of the weights.
- **Autoregressive generation with distributed KV cache** (Phase 15c): the
  executor keeps a per-`(run_id, shard_index)` LRU KV cache so multi-pass
  generation reuses keys/values across passes (only the newest token is
  processed after the first pass).  RoPE positions are offset to the cached
  sequence length for correctness.
- **`llm_capable` advertised in registration**: at startup the node probes
  `probe_llm_capable()` and, when true, constructs an `LLMShardExecutor` and
  passes it into the `DataShardAwareExecutor` routing, and sets
  `llm_capable=True` in the HTTP registration so the coordinator knows it can
  hand this node real LLM shards.  Everything is **on by default when the extra
  is installed**; **zero behavior change when it isn't**.
- **`--vram` flag recommendation**: to participate in distributed LLM
  inference, advertise your GPU's VRAM with `--vram <MB>` (e.g.
  `--vram 24000` for a 24 GB GPU) alongside `--gpu 1`.  The coordinator uses
  VRAM to plan layer-shard assignments across nodes.

### Serving LLM shards

```bash
# 1. Install with the [llm] extra
pip install -e ".[llm]"

# 2. Start the node — it auto-probes torch and advertises llm_capable
python -m computecloud_node --http \
  --api-url https://api.computepool.cloud \
  --executor auto \
  --cpu 4 --ram 16 --gpu 1 --vram 24000 --disk 100 \
  --gpu-model "RTX 4090" \
  --username admin --password YOUR_PASSWORD
```

The coordinator can now assign `kind == "llm_shard"` tasks (with
`weights_uri = hf://org/model@revision` or `file:///path/to/weights`).  The
node's `DataShardAwareExecutor` routes them to the `LLMShardExecutor`, which
downloads the layer range, runs the forward pass, and returns a
`TensorPayload` of hidden states (or `{token, token_id}` for the last shard).

## What's New in v0.4.0

- **Universal data-fabric shard participation**: nodes now automatically
  handle generalized data-run shard tasks — not just LLM pipeline shards. When
  the pool coordinator assigns a `data_shard` or `data_merge` task (from any
  registered workload type — rendering, batch processing, command chunks, model
  sharding, etc.), the node polls the server's data endpoint for its input,
  executes the shard, and posts the output back — all through the existing
  HTTP pull/report loop (no new connections, NAT-proof by construction).
  Supports all three topologies: **parallel** (independent shards), **pipeline**
  (sequential shards, output feeds next input), and **reduce** (shards + merge).
- **Same routing as the in-repo client**: a single `DataShardAwareExecutor`
  routes data shards → `DataShardWorker`, pipeline shards →
  `PipelineShardWorker`, and plain tasks → your configured executor. Pipeline
  participation from v0.3.0 keeps working unchanged.
- Data-fabric participation is **enabled by default** in HTTP mode. No extra
  flags needed — just start the node and it will accept data shards, pipeline
  shards, and regular compute tasks alike.

## What's New in v0.3.0

- **Distributed pipeline participation**: nodes now automatically handle
  distributed-LLM pipeline shard tasks. When the pool coordinator assigns a
  `pipeline_shard` task, the node polls the server for its input activations,
  computes its shard, and posts the output back — all through the existing
  HTTP pull/report loop (no new connections, NAT-proof by construction).
- **VRAM contribution (`--vram`)**: advertise your GPU's VRAM to the pool so
  it can be used for distributed LLM inference. Use `--vram <MB>` with
  `--gpu >= 1` to register a VRAM segment.
- Pipeline participation is **enabled by default** in HTTP mode. No extra
  flags needed — just start the node and it will accept shard tasks
  alongside regular compute tasks.

## Quick Start

```bash
# 1. Install (requires Python 3.10+)
pip install -e .

# 2. Join the pool — share your compute and earn credits
python -m computecloud_node --http \
  --api-url https://api.computepool.cloud \
  --executor auto \
  --cpu 4 --ram 16 --gpu 0 --disk 100 \
  --username admin --password YOUR_PASSWORD
```

That's it! Your machine is now a contributor node, sharing CPU/RAM with the pool.
Use the same username/password as the website login so your contributions and
payouts are tracked per-user.

### Contributing GPU VRAM for Distributed LLM

If you have a GPU and want to participate in distributed LLM pipeline runs:

```bash
python -m computecloud_node --http \
  --api-url https://api.computepool.cloud \
  --executor auto \
  --cpu 4 --ram 16 --gpu 1 --vram 24000 --disk 100 \
  --gpu-model "RTX 4090" \
  --username admin --password YOUR_PASSWORD
```

This advertises 24 GB of VRAM to the pool. The coordinator can then assign
model-shard tasks to your node for distributed inference.

## Requirements

- **Python 3.10+**
- **httpx** (installed automatically by `pip install`)
- **Docker** (optional — needed for template/instance jobs and SSH workbenches with `--executor docker` or `--executor auto`). The daemon must be **running**; v0.6.3+ tries to auto-start Docker Desktop at launch (disable with `--no-docker-autostart`)

## Options

```bash
python -m computecloud_node --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--http` | off | Use HTTP transport (required for cloud pool) |
| `--api-url` | `""` | Pool API URL (e.g. `https://api.computepool.cloud`) |
| `--executor` | `echo` | `echo`, `local` (shell), `docker` (sandboxed), `auto`, or `print` |
| `--cpu` | `4` | CPU cores to share |
| `--ram` | `4096` | RAM to share in GB (e.g. `--ram 16`) |
| `--gpu` | `0` | Number of GPUs to share (0 = no GPU) |
| `--disk` | `8192` | Disk space to share in GB (e.g. `--disk 100`) |
| `--gpu-model` | `""` | GPU model (e.g. `--gpu-model "RTX 4090"`, blank = any) |
| `--vram` | `0` | GPU VRAM to contribute in MB (e.g. `--vram 24000` for 24 GB) |
| `--username` | `""` | Your website login (for contribution tracking) |
| `--password` | `""` | Your website password |
| `--max-tasks` | `4` | Max concurrent tasks |
| `--tags` | `[]` | Tags (e.g. `gpu spot`) |
| `--no-docker-autostart` | off | Don't auto-start Docker Desktop when the daemon is down |

## How It Works

```
Your machine                    Render (computepool.cloud)
┌──────────────┐               ┌──────────────────┐
│  Node Client │──HTTP POST──→ │  FastAPI Server  │
│  (poll loop) │←──JSON──────  │  (port 8000)     │
└──────────────┘               └──────────────────┘
     ↑                              ↑
  executes                     assigns tasks
  shell/docker                  from the queue
  commands                     from renters
```

The client polls the pool server every second for tasks, executes them
locally (shell commands or Docker containers), and reports results back.
You earn credits for each completed task.

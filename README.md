# ComputeCloud Node Client — Share Your Compute

This is a lightweight, standalone package for contributor nodes to join the
ComputeCloud pool at **computepool.cloud** — without cloning the entire repo.

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
  --executor local \
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
  --executor local \
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
  --executor local \
  --cpu 4 --ram 16 --gpu 1 --vram 24000 --disk 100 \
  --gpu-model "RTX 4090" \
  --username admin --password YOUR_PASSWORD
```

This advertises 24 GB of VRAM to the pool. The coordinator can then assign
model-shard tasks to your node for distributed inference.

## Requirements

- **Python 3.10+**
- **httpx** (installed automatically by `pip install`)
- **Docker** (optional — only needed for template/instance jobs with `--executor docker` or `--executor auto`)

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

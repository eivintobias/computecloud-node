# ComputeCloud Node Client — Share Your Compute

This is a lightweight, standalone package for contributor nodes to join the
ComputeCloud pool at **computepool.cloud** — without cloning the entire repo.

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

"""Phase 17c — welcome-notebook bootstrap for Jupyter workbench sessions.

When a Jupyter workbench session starts on a node, the node drops a starter
notebook ``welcome.ipynb`` into ``/workspace`` so the user lands in a
ready-to-go environment with the Pool SDK pre-authenticated (env vars already
injected by the Phase 17b workbench machinery).

The notebook is generated from a template string (valid ``.ipynb`` JSON) and
written to disk only if it does not already exist — a user who customises
``welcome.ipynb`` keeps their changes across session restarts (the file lives
in their workspace, which is synced in fresh each time but not overwritten).

This module is stdlib-only (json + os) so it can be vendored into the
standalone ``node_client`` package without extra deps.
"""

from __future__ import annotations

import json
import os

__all__ = ["WELCOME_NOTEBOOK_JSON", "generate_welcome_notebook"]


def _md_cell(source: str) -> dict:
    """Build a markdown notebook cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _code_cell(source: str) -> dict:
    """Build a code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _build_notebook() -> dict:
    """Construct the welcome notebook dict (nbformat 4).

    The notebook contains a markdown intro, a bootstrap cell
    (``from computecloud_sdk import Pool; pool = Pool()`` — reads
    POOL_API_URL/POOL_TOKEN env vars injected by the node), and example cells
    for ``pool.status()``, ``pool.map()``, ``pool.run()``, and
    ``pool.generate()``.
    """
    cells = [
        _md_cell(
            "# Welcome to the ComputeCloud Pool\n"
            "\n"
            "This is a **workbench** — an interactive Jupyter notebook running\n"
            "on a pool node, with your workspace files synced into `/workspace`\n"
            "and the **Pool SDK** pre-installed and pre-authenticated.\n"
            "\n"
            "Your workspace is the notebook's file root — files you create here\n"
            "are saved to your workspace. Use `pool.map` / `pool.run` / "
            "`pool.generate`\n"
            "to fan work across the swarm. Interactive code runs on this one\n"
            "node; the pool magic enters through the SDK library.\n"
            "\n"
            "Run the cells below in order to get started."
        ),
        _code_cell(
            "# Bootstrap the Pool SDK — no args needed.\n"
            "# The node injected POOL_API_URL + POOL_TOKEN + POOL_WORKSPACE_ID\n"
            "# as environment variables, so Pool() reads them automatically.\n"
            "from computecloud_sdk import Pool\n"
            "\n"
            "pool = Pool()\n"
            "print('Connected to:', pool.api_url)"
        ),
        _code_cell(
            "# Check pool capacity — how many nodes / CPUs / GPUs are online.\n"
            "pool.status()"
        ),
        _md_cell(
            "## Fan work across the pool\n"
            "\n"
            "`pool.map(command_template, items)` runs one job per item, in\n"
            "parallel across the swarm. Each `{}` in the template is filled\n"
            "with an item. Results come back in order."
        ),
        _code_cell(
            "# Fan out: one echo job per item, parallel across the pool.\n"
            "items = ['alpha', 'beta', 'gamma']\n"
            "results = pool.map('echo {}', items)\n"
            "for r in results:\n"
            "    print(r.merged_result)"
        ),
        _md_cell(
            "## Run a single job\n"
            "\n"
            "`pool.run(command)` submits a command, splits it into chunks,\n"
            "executes them jointly across the pool, and merges the result."
        ),
        _code_cell(
            "# Submit a single job and wait for the merged result.\n"
            "result = pool.run('echo hello from the pool')\n"
            "print(result.merged_result)"
        ),
        _md_cell(
            "## Distributed LLM generation\n"
            "\n"
            "`pool.generate(model, prompt)` shards a model across pooled VRAM\n"
            "and generates text via the distributed pipeline. Requires GPU\n"
            "nodes with the `[llm]` extra installed."
        ),
        _code_cell(
            "# Generate text via the distributed LLM pipeline.\n"
            "# (Requires GPU nodes advertising VRAM — adjust the model name.)\n"
            "# text = pool.generate('TinyLlama/TinyLlama-1.1B-Chat-v1.0@main',\n"
            "#                      'The capital of France is',\n"
            "#                      max_new_tokens=16)\n"
            "# print(text)"
        ),
        _md_cell(
            "## Workspace I/O\n"
            "\n"
            "Files in `/workspace` are your workspace. You can also use the\n"
            "SDK to upload/download/list files programmatically:"
        ),
        _code_cell(
            "# List files in your workspace (this notebook lives here too).\n"
            "ws = pool.workspace()\n"
            "for f in ws.list():\n"
            "    print(f['relative_path'], f['size_bytes'], 'bytes')"
        ),
        _md_cell(
            "---\n"
            "\n"
            "That's it — happy pooling! See the docs for the full SDK API."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# Pre-built notebook JSON string (module-level so tests can inspect it
# without I/O).  Serialised once at import time.
WELCOME_NOTEBOOK_JSON: str = json.dumps(_build_notebook(), indent=1)


def generate_welcome_notebook(workspace_dir: str = "/workspace") -> str | None:
    """Write ``welcome.ipynb`` into *workspace_dir* if it doesn't exist.

    Args:
        workspace_dir: The notebook's file root (the node syncs workspace
            files here).  Defaults to ``"/workspace"``.

    Returns:
        The path the notebook was written to, or ``None`` if it already
        existed (the user's customisations are preserved).
    """
    path = os.path.join(workspace_dir, "welcome.ipynb")
    if os.path.exists(path):
        return None
    # Ensure the directory exists (the workspace sync usually does this, but
    # be defensive — the notebook root must exist before Jupyter starts).
    os.makedirs(workspace_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(WELCOME_NOTEBOOK_JSON)
    return path


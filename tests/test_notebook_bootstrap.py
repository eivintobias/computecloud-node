"""Tests for the Phase 17c welcome-notebook bootstrap (node_client side).

Covers: the generated welcome.ipynb is valid nbformat 4 JSON with expected
cells, the file is skipped if it already exists (user customisations
preserved), and the WELCOME_NOTEBOOK_JSON constant is valid.

This is the standalone node_client version — identical logic to the in-repo
``tests/test_notebook_bootstrap.py`` but imports from
``computecloud_node.notebook_bootstrap``.
"""

from __future__ import annotations

import json
import os

# The conftest.py at the node_client root ensures the local src/ directory is
# on sys.path ahead of any installed computecloud_node package.


class TestWelcomeNotebookJson:
    def test_welcome_notebook_json_is_valid(self):
        """WELCOME_NOTEBOOK_JSON parses as valid nbformat 4 JSON."""
        from computecloud_node.notebook_bootstrap import WELCOME_NOTEBOOK_JSON

        nb = json.loads(WELCOME_NOTEBOOK_JSON)
        assert nb["nbformat"] == 4
        assert isinstance(nb["cells"], list)
        assert len(nb["cells"]) > 0

    def test_notebook_has_bootstrap_cell(self):
        """The notebook contains a bootstrap cell with Pool() (no args)."""
        from computecloud_node.notebook_bootstrap import WELCOME_NOTEBOOK_JSON

        nb = json.loads(WELCOME_NOTEBOOK_JSON)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        bootstrap = [c for c in code_cells if "Pool()" in c["source"]]
        assert len(bootstrap) >= 1
        assert "from computecloud_sdk import Pool" in bootstrap[0]["source"]

    def test_notebook_has_example_cells(self):
        """The notebook has example cells for status, map, run, and generate."""
        from computecloud_node.notebook_bootstrap import WELCOME_NOTEBOOK_JSON

        nb = json.loads(WELCOME_NOTEBOOK_JSON)
        all_sources = " ".join(c["source"] for c in nb["cells"])
        assert "pool.status()" in all_sources
        assert "pool.map(" in all_sources
        assert "pool.run(" in all_sources
        assert "pool.generate(" in all_sources

    def test_notebook_has_markdown_intro(self):
        """The notebook starts with a markdown intro cell."""
        from computecloud_node.notebook_bootstrap import WELCOME_NOTEBOOK_JSON

        nb = json.loads(WELCOME_NOTEBOOK_JSON)
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert "ComputeCloud" in nb["cells"][0]["source"]


class TestGenerateWelcomeNotebook:
    def test_creates_notebook_if_absent(self, tmp_path):
        """generate_welcome_notebook writes welcome.ipynb into the dir."""
        from computecloud_node.notebook_bootstrap import generate_welcome_notebook

        path = generate_welcome_notebook(str(tmp_path))
        assert path is not None
        assert path.endswith("welcome.ipynb")
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        assert nb["nbformat"] == 4

    def test_skips_if_exists(self, tmp_path):
        """If welcome.ipynb already exists, it is NOT overwritten."""
        from computecloud_node.notebook_bootstrap import generate_welcome_notebook

        nb_path = tmp_path / "welcome.ipynb"
        nb_path.write_text('{"cells": [], "nbformat": 4, "nbformat_minor": 5}')
        path = generate_welcome_notebook(str(tmp_path))
        assert path is None
        content = nb_path.read_text()
        assert '"cells": []' in content

    def test_creates_dir_if_missing(self, tmp_path):
        """The target directory is created if it doesn't exist."""
        from computecloud_node.notebook_bootstrap import generate_welcome_notebook

        target = tmp_path / "deep" / "nested" / "dir"
        path = generate_welcome_notebook(str(target))
        assert path is not None
        assert os.path.isfile(path)

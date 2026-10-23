# Contributing

## Setting up

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"   # the extraction tools; ".[test,build]" adds open3d for helhest-odom-frame
uv venv --python 3.12 .venv-viewer && uv pip install --python .venv-viewer -e ".[viewer,test]"   # plus the viewer
```

Python 3.12 or newer (`requires-python` in `pyproject.toml`; CI runs 3.12).
Two venvs, as in the README: the viewer's extra pulls in SimView and with it
torch, which the extraction tools never need. `uv` takes torch from the
CPU-only index configured in `pyproject.toml`; plain `pip` ignores that and
fetches the CUDA build (~3 GB), so with pip run
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first.
The cloudini shim is committed prebuilt; `cloudini_shim/build_shim.sh` rebuilds
it, and its header says on which machine to do that.

## Tests

```bash
.venv/bin/python -m pytest          # extraction tools; the viewer tests skip
.venv-viewer/bin/python -m pytest   # everything
```

No bags are needed: every test runs on synthetic data written with `rosbags`
or on synthetic scene directories (`tests/helpers_viewer.py`); the one recorded
input is `tests/data/heightmap_parity.npz`, a few frames DRIFT built, which
holds `helhest-heightmap` to its maps. Keep it that way
-- a test that needs a recording cannot run in CI and will not be run. The
viewer's tests skip themselves when its extra is not installed.

## Style

The tree is formatted with `black` (line length 120) and linted with `ruff`;
CI pins both (`.github/workflows/ci.yml`), so run the same versions:

```bash
pip install "black==24.8.0" "ruff==0.6.9"
black dataset_tools scripts tests
ruff check dataset_tools scripts tests
```

The long lines that remain are deliberate and held by `# fmt: off`; see the
`[tool.black]` note in `pyproject.toml` before adding another.

## What goes where

- `dataset_tools/` is the installed package: the extractor, the validator, the
  ground-truth helpers and the viewer. Anything here needs a test.
- `scripts/` are tools run against a root or a bag from a checkout, not
  installed; `scripts/README.md` lists them.
- `slurm/` are the CTU RCI job files, kept as working examples; `slurm/README.md`.
- `docs/` holds the documentation site's pages (`mkdocs.yml`), the release
  specs and bag lists, and reference documents; `docs/README.md` lists them.

The version is declared in two places that must agree -- `pyproject.toml` and
`CITATION.cff`.

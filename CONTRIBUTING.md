# Contributing

## Setting up

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"            # the extraction tools, Python >= 3.10
.venv/bin/pip install -e ".[viewer,test]"     # plus the viewer, Python >= 3.12
```

The viewer's extra pulls in SimView and with it torch; on a machine without a
GPU, `--extra-index-url https://download.pytorch.org/whl/cpu` keeps that small.
The cloudini shim is committed prebuilt; `cloudini_shim/build_shim.sh` rebuilds
it, and its header says on which machine to do that.

## Tests

```bash
.venv/bin/python -m pytest
```

No bags are needed: every test runs on synthetic data written with `rosbags`
or on synthetic scene directories (`tests/helpers_viewer.py`). Keep it that way
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
- `docs/` holds the user documentation and the dated campaign records;
  `docs/README.md` says which is which.

The version is declared in three places that must agree -- `pyproject.toml`,
`setup.py` (for the ROS 2 / `colcon` build) and `package.xml` -- plus
`CITATION.cff`.

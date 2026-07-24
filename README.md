# kaval

A Python CLI for orchestrating computational experiments across HPC environments.
It reads YAML experiment suite files, expands parameter combinations (cartesian
product), and submits jobs to SLURM clusters or runs them locally.

## Install

```bash
uv sync
```

## Quick start

`run-experiments.py` takes a suite name, resolves it from `.suite.yaml` files in
your `--search-dirs`, and either runs it locally or generates/submits SLURM job
scripts, depending on `--machine`.

```bash
uv run python3 run-experiments.py <suite-name> --machine <target> [options]
```

To try it against the bundled example (a stand-in bash "binary" that just echoes
its args):

```bash
BUILD_DIR=examples uv run python3 run-experiments.py test \
  --machine shared --search-dirs examples/suites --output-dir /tmp/kaval_test
```

`--machine` selects the execution backend: `shared` | `horeka` | `supermuc` |
`lichtenberg` | `generic-job-file`.

## Learn more

See [`CLAUDE.md`](CLAUDE.md) for the full suite-YAML format, the reusable
instance-set / file-graph-registry mechanism, module architecture, and other
CLI flags (`--cores`, `--fresh`, `--test`, etc.).

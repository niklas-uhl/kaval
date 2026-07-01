# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**kaval** is a Python CLI tool for orchestrating computational experiments across HPC environments. It reads YAML experiment suite files, expands parameter combinations (cartesian product), and submits jobs to SLURM clusters or runs them locally.

## Setup

```bash
pipenv install       # install pyyaml dependency
pipenv shell         # activate environment
```

## Running experiments

```bash
pipenv run python3 run-experiments.py <suite-name> --machine <target> [options]
```

Key flags:
- `--machine`: `shared` | `horeka` | `supermuc` | `lichtenberg` | `generic-job-file`
- `--search-dirs`: directories to search for `.suite.yaml` files
- `--cores` / `--min-cores` / `--max-cores`: core count control; special tokens: `pow2` (1,2,4,…), `sqr` (1,4,9,16,…), `sqr-pow2` (powers of two that are also squares: 1,4,16,64,…), or `node-size-pow2` (tpn,2×tpn,… requires `--tasks-per-node`). The same tokens and `min_cores`/`max_cores` bounds may be set in the suite YAML (`ncores`, `min_cores`, `max_cores`); CLI flags override the suite values (defaults: min 1, max 8000).
- `--fresh`: wipe experiment directory before running
- `--test`: use test partition on SLURM clusters

No automated test suite exists. Use the example suite to verify behavior. `examples/test-app` is a stand-in bash "binary" (echoes its args and writes a minimal JSON result); point `BUILD_DIR` at its parent so the suite's `executable: test-app` resolves:
```bash
BUILD_DIR=examples pipenv run python3 run-experiments.py test --machine shared --search-dirs examples/suites --output-dir /tmp/kaval_test
```

## Architecture

### Entry point
`run-experiments.py` — parses CLI args, resolves suite files, instantiates the right runner, and drives execution.

### Core modules

**`expcore.py`** — experiment model:
- `ExperimentSuite` — loads and validates a `.suite.yaml` file
- `FileInputGraph`, `KaGenGraph`, `DummyInstance` — input graph abstractions (file reference, generated, or placeholder)
- `CLIArgumentType` — enum controlling how arguments are passed to the binary (`FLAGS`, `POSITIONAL`, `POSITIONAL_LIST`, `FLAG_LIST`)
- Parameter expansion: YAML config lists are exploded into individual runs via cartesian product

**`runners.py`** — execution backends:
- `BaseRunner` — shared logic (directory management, command building, deduplication via JSON state files)
- `SharedMemoryRunner` — executes directly via subprocess on the local machine
- `SBatchRunner` (abstract) → `SuperMUCRunner`, `HorekaRunner`, `GenericDistributedMemoryRunner` — generate and submit SLURM job scripts; each specialization knows its cluster's node size and scheduler quirks
- `get_runner()` — factory that selects the backend from `--machine`

**`slugify.py`** — converts arbitrary strings to filesystem-safe slugs used in output directory naming.

### Templates
- `command-templates/` — shell snippets for how the binary is invoked (MPI flavors, binding strategies)
- `sbatch-templates/` — SLURM job script skeletons per cluster

### Suite YAML format
See `examples/suites/test.suite.yaml` for a working example. Top-level keys include `executable`, `cores`, `seeds`, `graphs`, and `configurations`. Configurations support weak-scaling via `weak_scaling_graphs`.

### Reusable instance sets
To avoid repeating the same graphs across suites, define them once in a `*.instances.yaml` file (auto-discovered in `--search-dirs`, alongside `.suite.yaml` files). Each such file has a top-level `name` and a `graphs` list in the same format as a suite's `graphs`. A suite pulls a set in via an `import` entry in its own `graphs` list, and can mix imported sets with inline graphs:

```yaml
# common.instances.yaml
name: common-graphs
graphs:
  - generator: kagen
    type: rgg2d
    N: [20, 22]
```
```yaml
# my.suite.yaml
graphs:
  - import: common-graphs   # splices in every graph from the set
  - generator: kagen        # inline graphs may be mixed in
    type: rgg2d
    N: 24
```

See `examples/suites/common.instances.yaml` and `examples/suites/import.suite.yaml`. Parsing lives in `expcore.load_instance_sets()` (discovery) and `expcore.parse_graph_list()` (expansion, incl. imports); unknown or circular imports raise a `ValueError`.

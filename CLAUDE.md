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
- `KaGenGraph`, `GenericInstance` — input abstractions (KaGen-generated/file-based, or a generic pass-through of arbitrary CLI params for any other binary, e.g. a string generator)
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
See `examples/suites/test.suite.yaml` for a working example. Top-level keys include `executable`, `cores`, `seeds`, `graphs` (or `inputs` — an alias; kaval is used for non-graph experiments too, e.g. string sorting, where `inputs` reads more naturally), and `configurations`. Configurations support weak-scaling via `weak_scaling_graphs`.

Each `graphs`/`inputs` entry picks a generator via `generator: kagen | generic`. `generic` (`dummy` also still accepted, as a legacy alias) passes its remaining keys straight through as CLI params to `executable` — the mechanism for any generator that isn't KaGen (e.g. a colleague's string generator), not a placeholder despite the old name. See `examples/suites/generic-input.suite.yaml` for the `inputs:`/`generic` spelling.

### Reusable instance sets
To avoid repeating the same inputs across suites, define them once in a `*.instances.yaml` file (auto-discovered in `--search-dirs`, alongside `.suite.yaml` files). Each such file has a top-level `name` and a `graphs`/`inputs` list in the same format as a suite's. A suite pulls a set in via an `import` entry in its own `graphs`/`inputs` list, and can mix imported sets with inline graphs:

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

Imports are transitive: an instance set may itself `import` another set, resolved recursively. Unknown or circular imports raise a `ValueError`.

**Modifying an imported set (`with:`)** — an import entry may carry a `with:` mapping whose keys are merged into every resolved graph before it is built, so a set can be reused with a tweak instead of being duplicated. Keys override existing ones; values may be lists to explode; the modification applies through transitive imports, and on conflict the use-site (outer) `with:` wins:

```yaml
# reuse a whole set but permute every graph
graphs:
  - import: synthetic-large
    with:
      permute: True        # merged into each graph as a KaGen flag
```

A `with:` on a bare-string file input can't be applied — it is logged and the input passes through unmodified.

**Deduplication** — after all imports and modifications resolve, the input list is de-duplicated by graph name (first occurrence kept, order preserved), so diamond imports (two sets importing a common set) or an inline graph that duplicates an imported one don't produce repeated runs. Dropped names are logged.

**File-graph registry (`graph_name` / `root`)** — real, on-disk graphs (parhip/metis/etc.) need no dedicated abstraction: they're ordinary `generator: kagen, type: file` (or `type: partitioned_file`) entries, so the import/`with:`/dedup pipeline above applies to them unchanged. `KaGenGraph` accepts two extra sugar params for this case:

- `graph_name` — a friendly handle used as the file-identifying part of both `.name` (the dedup/time-limit key) and `.short_name` (output-directory naming), instead of the raw `filename`. Without it, every `type: file` graph's `short_name` collapses to the literal string `file`, colliding across any suite with multiple real-world graphs.
- `root` — an optional directory prefix. When `filename` is relative, it's joined onto `root` when the command is built, not when the YAML is parsed — so `root` remains a normal, overridable param through `with:`. `root` also expands `$VAR`/`${VAR}` references, so a checked-in registry works unmodified across clusters that each set the referenced variable (e.g. `GRAPH_ROOT`) via their existing environment setup, mirroring how `BUILD_DIR` locates the executable per-machine. An unset variable raises a `ValueError` when the suite loads, rather than embedding a literal `$GRAPH_ROOT` in a generated sbatch script.

A `*.instances.yaml` file may also set a top-level `root:`, applied as a default to every `generator: kagen` entry in that file that has a `filename` and doesn't already set its own `root` — so a registry of many file-based graphs doesn't need `root` repeated on each one. It's skipped for `import` entries and non-file graphs mixed into the same file, so those never inherit a root/env-var requirement they don't need; an entry's own `root`, or a `with: {root: ...}` override at the import site, still wins over the file-level default. Implemented in `expcore.apply_default_root()`, applied in `load_instance_sets()`.

```yaml
# real-world.instances.yaml
name: real-world-graphs
root: ${GRAPH_ROOT}   # default for every graph below that doesn't set its own root
graphs:
  - generator: kagen
    type: file
    graph_name: example-graph
    filename: example-graph.graph
```
```yaml
# my.suite.yaml
graphs:
  - import: real-world-graphs
    with:
      root: /scratch/other-graphs   # override for one run, beats the file-level default
```

See `examples/suites/common.instances.yaml`, `examples/suites/real-world.instances.yaml`, and `examples/suites/import.suite.yaml`. Parsing lives in `expcore.load_instance_sets()` (discovery) and `expcore.parse_graph_list()` (expansion, incl. imports and `with:` overrides); dedup is `expcore.dedup_inputs()`, applied in `load_suite_from_yaml()`.

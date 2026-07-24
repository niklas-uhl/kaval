# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**kaval** is a Python CLI tool for orchestrating computational experiments across HPC environments. It reads YAML experiment suite files, expands parameter combinations (cartesian product), and submits jobs to SLURM clusters or runs them locally.

## Setup

kaval is a proper installable package (`pyproject.toml` + hatchling), so `uv sync` builds it into `.venv` in editable mode and exposes a `kaval` console script:

```bash
uv sync               # install pyyaml + kaval itself (editable) into .venv
```

pipenv is no longer the primary tool, but `pyproject.toml` is still a usable source of truth for its dependencies if you'd rather stay on pipenv: `pipenv install` alone does *not* read `pyproject.toml` (it just creates an empty `Pipfile`) — use `pipenv install -e .` instead, which installs kaval editable via pip and pulls in `pyyaml` (and the `kaval` console script) transitively, same as `uv sync`.

## Running, submitting, and checking on experiments

`kaval` is the top-level entry point, with three subcommands. As an installed console script it can be invoked directly through `uv run` (no `python3 kaval.py` needed):

```bash
uv run kaval run <suite-name> --machine <target> [options]     # generate (and, for --machine shared, execute) a suite
uv run kaval submit <suite-name> [options]                     # sbatch the job files a prior `run` generated
uv run kaval status <suite-name> [options]                     # report submission/completion status
```

`uv run python3 kaval.py <subcommand> ...` and `uv run python3 run-experiments.py <suite-name> --machine <target> [options]` (the original entry point) both still work unchanged — `run-experiments.py` is a thin shim around `run_experiments.main()`, which `kaval run` also calls. `kaval.py`'s `main()` (the `kaval = "kaval:main"` target in `[project.scripts]`) just strips the subcommand off `sys.argv` and delegates; it holds no logic of its own.

### `kaval run` key flags
- `--machine`: `shared` | `horeka` | `supermuc` | `lichtenberg` | `generic-job-file`
- `--search-dirs`: directories to search for `.suite.yaml` files
- `--cores` / `--min-cores` / `--max-cores`: core count control; special tokens: `pow2` (1,2,4,…), `sqr` (1,4,9,16,…), `sqr-pow2` (powers of two that are also squares: 1,4,16,64,…), or `node-size-pow2` (tpn,2×tpn,… requires `--tasks-per-node`). The same tokens and `min_cores`/`max_cores` bounds may be set in the suite YAML (`ncores`, `min_cores`, `max_cores`); CLI flags override the suite values (defaults: min 1, max 8000).
- `--fresh`: wipe experiment directory before running
- `--test`: use test partition on SLURM clusters
- `--no-job-grouping`: SLURM machine types only. By default, every config/seed/thread-count combination for a given input and core count is packed into one sbatch job file as multiple sequential `mpiexec` invocations. This flag disables that grouping, producing one job file per `mpiexec` invocation instead (job/log/error-log file names and the SLURM time limit become per-invocation rather than aggregated).
- `--input-filter`: only generate jobs for inputs whose `.name`/`.short_name` (or, for bare file-path inputs, the string itself) contains one of the given substrings (case-insensitive, OR'd across multiple values)
- `--config-index`: only generate jobs for configs at these 0-based indices, matching the `idx` field written to the suite's `config.json`
- `--config-filter`: only generate jobs for configs matching all given `key=value` pairs (e.g. `--config-filter variant=bfs1`); values are compared as strings. Combines with `--config-index` (both must pass)

No automated test suite exists. Use the example suite to verify behavior. `examples/test-app` is a stand-in bash "binary" (echoes its args and writes a minimal JSON result); point `BUILD_DIR` at its parent so the suite's `executable: test-app` resolves:
```bash
BUILD_DIR=examples uv run kaval run test --machine shared --search-dirs examples/suites --output-dir /tmp/kaval_test
```

### `kaval submit` / `kaval status`
For SLURM machine types, `kaval run` only writes job files (it never calls `sbatch` itself) — it also writes a `manifest.json` into the jobfiles directory recording, per job file, its core count and the list of expected per-invocation output-file stems. `kaval submit`/`kaval status` read that manifest so they never need to reload the suite YAML:

- Suite resolution: `kaval submit <suite-name>` picks the most recently modified directory matching `<suite-name>` or `<suite-name>_*` under `--experiment-data-dir` (i.e. "the latest run of this suite"). Pass an explicit `--dir <path>` to bypass that and target a specific experiment directory directly.
- Filtering: `--cores`/`--min-cores`/`--max-cores` (same tokens as `kaval run`) restrict which job files get submitted/reported on, matched against the `cores` field recorded per job file in the manifest.
- `kaval submit` calls `sbatch` on each matching job file not already recorded as submitted (idempotent — re-running `kaval submit` after generating more job files only submits the new ones; `--resubmit` forces re-submission), and records `{job_id, submitted_at}` per job file in a sibling `submissions.json`, written incrementally so a crash mid-submit doesn't lose already-submitted state. `--dry-run` prints what would be submitted without calling `sbatch`.
- `kaval status` reports one line per job file. Completion is inferred, not queried directly: since `sacct` history isn't reliably available on every cluster (a job can vanish from accounting once finished), a job ID still in `squeue` reports that live state, while a job ID no longer in `squeue` is classified by checking whether its expected output files (from the manifest) exist — `COMPLETED`/`PARTIAL (n/total)`/`FAILED/UNKNOWN`. This means completion detection needs `--omit-output-path` to *not* be set; with it set, `status` can only report NOT IN QUEUE, not completion.

## Architecture

### Entry point
`kaval.py` — dispatches `run`/`submit`/`status` to the modules below by rewriting `sys.argv` and delegating; holds no logic itself. `run-experiments.py` is kept as a direct-invocation compat shim.

### Core modules

**`expcore.py`** — experiment model:
- `ExperimentSuite` — loads and validates a `.suite.yaml` file
- `KaGenGraph`, `GenericInstance` — input abstractions (KaGen-generated/file-based, or a generic pass-through of arbitrary CLI params for any other binary, e.g. a string generator)
- `CLIArgumentType` — enum controlling how arguments are passed to the binary (`FLAGS`, `POSITIONAL`, `POSITIONAL_LIST`, `FLAG_LIST`)
- Parameter expansion: YAML config lists are exploded into individual runs via cartesian product

**`run_experiments.py`** (`main()`) — parses CLI args, resolves suite files, instantiates the right runner, and drives execution. Backing module for both `run-experiments.py` and `kaval run`.

**`runners.py`** — execution backends:
- `BaseRunner` — shared logic (directory management, command building)
- `SharedMemoryRunner` — executes directly via subprocess on the local machine
- `SBatchRunner` (abstract) → `SuperMUCRunner`, `HorekaRunner`, `GenericDistributedMemoryRunner` — generate SLURM job scripts and a `manifest.json` describing them (see below); each specialization knows its cluster's node size and scheduler quirks. Job files are written, not submitted — `kaval submit` handles submission.
- `get_runner()` — factory that selects the backend from `--machine`

**`submit_experiments.py`** (`main_submit()`, `main_status()`) — backing module for `kaval submit`/`kaval status`. Resolves a suite name or `--dir` to an experiment directory, reads that directory's `jobfiles/manifest.json` (job file → cores/expected outputs, written by `SBatchRunner.execute()`) and `jobfiles/submissions.json` (job file → job ID/submitted-at, written incrementally as jobs get submitted), and implements the `squeue`-plus-output-file completion heuristic described above.

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

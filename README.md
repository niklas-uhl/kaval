# kaval

A Python CLI for orchestrating computational experiments across HPC environments.
It reads YAML experiment suite files, expands parameter combinations (cartesian
product), and either runs them locally or generates and submits jobs to SLURM
clusters.

## Install

```bash
uv sync
```

## Quick start

`kaval run` takes a suite name, resolves it from `.suite.yaml` files in your
`--search-dirs`, and either runs it locally or generates SLURM job scripts,
depending on `--machine`.

```bash
uv run kaval run <suite-name> --machine <target> [options]
```

To try it against the bundled example (a stand-in bash "binary" that just echoes
its args):

```bash
BUILD_DIR=examples uv run kaval run test \
  --machine shared --search-dirs examples/suites --output-dir /tmp/kaval_test
```

`--machine` selects the execution backend: `shared` runs locally right away;
`horeka` | `supermuc` | `lichtenberg` | `generic-job-file` write SLURM job
files for their respective cluster instead of running anything.

## Suite files

A suite is a `.suite.yaml` describing one executable and the parameter space to
sweep over it. The important pieces:

```yaml
name: my-suite
executable: my-binary          # resolved under $BUILD_DIR
ncores: [1, 8, 64]              # core counts to run at (also accepts pow2, sqr, ...)
seeds: [0, 1, 2]
graphs:                         # or `inputs:` - same thing, reads better for non-graph workloads
  - generator: kagen
    type: rgg2d
    N: [20, 22]
configurations:
  - variant: ["bfs1", "bfs2"]   # exploded into separate runs
```

Every list-valued field is a dimension in a cartesian product, so the example
above alone expands into `3 cores × 3 seeds × 2 N × 2 variants` runs. Inputs
that get reused across suites can be factored out once into a `*.instances.yaml`
file and pulled in with `import:`; see `CLAUDE.md` for that mechanism (including
reusing a whole set with a tweak via `with:`) and the full set of generators,
argument types, and flags.

## Submitting and checking on SLURM runs

`kaval run` against a SLURM machine type only *writes* job files — it doesn't
call `sbatch` for you. Use the other two subcommands for that:

```bash
uv run kaval submit <suite-name>   # sbatch every job file not already submitted
uv run kaval status <suite-name>   # one line per job file: queued/running/completed/failed
```

Both default to the most recently generated run of `<suite-name>`; pass
`--dir <path>` to target a specific experiment directory instead, and
`--cores`/`--min-cores`/`--max-cores` to act on a subset of job files.
Submission is idempotent and tracked in `submissions.json` next to the job
files, so re-running `kaval submit` after generating more jobs only submits
what's new.

## Learn more

See [`CLAUDE.md`](CLAUDE.md) for the full suite-YAML format, the reusable
instance-set / file-graph-registry mechanism, module architecture, and the
complete CLI flag reference.

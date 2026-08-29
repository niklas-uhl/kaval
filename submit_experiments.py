#!/bin/env python

# MIT License
#
# Copyright (c) 2020-2026 Tim Niklas Uhl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Submit generated sbatch job files and report on their progress.

Consumes the manifest.json written by SBatchRunner.execute() (see runners.py)
- it lists, per generated job file, the core count and the set of expected
per-invocation output name stems - so this module never needs to reload a
suite's YAML. Submission state (job IDs, submission time) is tracked in a
sibling submissions.json, written incrementally as jobs are submitted.

Completion is inferred rather than queried: many schedulers (e.g. plain
sacct on some clusters) drop a job's accounting record once it finishes, so
a job ID missing from `squeue` is ambiguous on its own. We disambiguate by
checking whether the expected output files for that job showed up on disk.
"""

import argparse, json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

from runners import _expand_cores_tokens, DEFAULT_MIN_CORES, DEFAULT_MAX_CORES

SUBMITTED_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def _common_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "suite",
        nargs="?",
        help="Name of the experiment suite to resolve. Picks the most "
        "recently modified experiment directory matching '<suite>' or "
        "'<suite>_*' under --experiment-data-dir. Ignored if --dir is given.",
    )
    parser.add_argument(
        "--dir",
        help="Explicit experiment directory (the one containing jobfiles/ "
        "and output/), bypassing suite-name resolution.",
    )
    default_experiment_data_dir = os.environ.get(
        "EXPERIMENT_DATA_DIR", Path(os.getcwd()) / "experiment_data"
    )
    parser.add_argument("--experiment-data-dir", default=default_experiment_data_dir)
    parser.add_argument(
        "--job-output-dir", help="Override for the jobfiles directory (default: <dir>/jobfiles)"
    )
    parser.add_argument(
        "--output-dir", help="Override for the output directory (default: <dir>/output)"
    )
    parser.add_argument(
        "--cores",
        nargs="*",
        help="Only act on job files with this core count (accepts explicit "
        "integers or the same tokens as `kaval run --cores`).",
    )
    parser.add_argument("--min-cores", type=int, default=None)
    parser.add_argument("--max-cores", type=int, default=None)
    return parser


def _resolve_dirs(args):
    if args.dir:
        scoped_dir = Path(args.dir)
        if not scoped_dir.is_dir():
            sys.exit(f"Not a directory: {scoped_dir}")
    else:
        if not args.suite:
            sys.exit("Provide a suite name or --dir.")
        scoped_dir = _resolve_experiment_dir(args.experiment_data_dir, args.suite)
        if scoped_dir is None:
            sys.exit(
                f"No experiment directory found for suite '{args.suite}' "
                f"under {args.experiment_data_dir}"
            )
    job_dir = Path(args.job_output_dir) if args.job_output_dir else scoped_dir / "jobfiles"
    output_dir = Path(args.output_dir) if args.output_dir else scoped_dir / "output"
    return job_dir, output_dir


def _resolve_experiment_dir(experiment_data_dir, suite_name):
    base = Path(experiment_data_dir)
    if not base.is_dir():
        return None
    candidates = [
        p
        for p in base.iterdir()
        if p.is_dir() and (p.name == suite_name or p.name.startswith(suite_name + "_"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_manifest(job_dir):
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(
            f"No manifest.json in {job_dir}. Was this suite generated with a "
            "SLURM machine type (--machine shared has nothing to submit)?"
        )
    with open(manifest_path) as f:
        return json.load(f)


def _load_submissions(job_dir):
    p = job_dir / "submissions.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_submissions(job_dir, submissions):
    with open(job_dir / "submissions.json", "w") as f:
        json.dump(submissions, f, indent=2)


def _resolve_cores_filter(args):
    if not args.cores:
        return None
    min_cores = args.min_cores if args.min_cores is not None else DEFAULT_MIN_CORES
    max_cores = args.max_cores if args.max_cores is not None else DEFAULT_MAX_CORES
    return set(_expand_cores_tokens(args.cores, min_cores, max_cores))


def _filter_jobs(jobs, args):
    cores_set = _resolve_cores_filter(args)
    result = {}
    for job_file, info in jobs.items():
        cores = info.get("cores")
        if cores_set is not None and cores not in cores_set:
            continue
        if args.min_cores is not None and cores is not None and cores < args.min_cores:
            continue
        if args.max_cores is not None and cores is not None and cores > args.max_cores:
            continue
        result[job_file] = info
    return result


def main_submit():
    parser = _common_parser("kaval submit")
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Re-submit job files that already have a recorded submission.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be submitted without calling sbatch."
    )
    args = parser.parse_args()

    job_dir, _ = _resolve_dirs(args)
    manifest = _load_manifest(job_dir)
    jobs = _filter_jobs(manifest.get("jobs", {}), args)
    submissions = _load_submissions(job_dir)

    submitted, skipped = 0, 0
    for job_file in sorted(jobs):
        if job_file in submissions and not args.resubmit:
            skipped += 1
            continue
        job_path = job_dir / job_file
        if not job_path.exists():
            print(f"Warning: {job_file} is in the manifest but missing on disk, skipping.")
            continue
        if args.dry_run:
            print(f"Would submit {job_file}")
            continue
        try:
            result = subprocess.run(
                ["sbatch", str(job_path)], capture_output=True, text=True
            )
        except FileNotFoundError:
            sys.exit("'sbatch' not found on PATH. Are you on a SLURM login node?")
        if result.returncode != 0:
            print(f"Failed to submit {job_file}: {result.stderr.strip()}")
            continue
        match = SUBMITTED_JOB_ID_RE.search(result.stdout)
        if not match:
            print(
                f"Submitted {job_file} but couldn't parse a job ID from: {result.stdout.strip()!r}"
            )
            continue
        job_id = match.group(1)
        submissions[job_file] = {
            "job_id": job_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_submissions(job_dir, submissions)
        print(f"Submitted {job_file} as job {job_id}")
        submitted += 1

    if not args.dry_run:
        print(f"Submitted {submitted} job(s), skipped {skipped} already-submitted job(s).")


def _query_squeue(job_ids):
    """Return {job_id: state} for job_ids currently known to squeue.

    Job IDs absent from the result may be finished, cancelled, or simply
    unknown to this scheduler - callers must disambiguate via output files.
    """
    if not job_ids:
        return {}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i %T"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("Warning: 'squeue' not found on PATH; falling back to output-file detection only.")
        return {}
    if result.returncode != 0:
        return {}
    states = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            states[parts[0]] = parts[1]
    return states


def main_status():
    parser = _common_parser("kaval status")
    args = parser.parse_args()

    job_dir, output_dir = _resolve_dirs(args)
    manifest = _load_manifest(job_dir)
    jobs = _filter_jobs(manifest.get("jobs", {}), args)
    submissions = _load_submissions(job_dir)
    omit_output_path = manifest.get("omit_output_path", False)

    job_ids = [info["job_id"] for info in submissions.values() if info.get("job_id")]
    live_states = _query_squeue(job_ids)

    for job_file in sorted(jobs):
        sub = submissions.get(job_file)
        if not sub:
            print(f"{job_file}: NOT SUBMITTED")
            continue
        job_id = sub["job_id"]
        state = live_states.get(job_id)
        if state:
            print(f"{job_file}: {state} (job {job_id})")
            continue
        outputs = jobs[job_file].get("outputs", [])
        if omit_output_path or not outputs:
            print(f"{job_file}: NOT IN QUEUE (job {job_id}, --omit-output-path was used so "
                  "completion can't be checked)")
            continue
        done = sum(1 for name in outputs if any(output_dir.glob(f"{name}*")))
        total = len(outputs)
        if done == total:
            print(f"{job_file}: COMPLETED ({done}/{total}) (job {job_id})")
        elif done == 0:
            print(f"{job_file}: FAILED/UNKNOWN (job {job_id}, no outputs found)")
        else:
            print(f"{job_file}: PARTIAL ({done}/{total}) (job {job_id})")


if __name__ == "__main__":
    sys.exit("Run this via `kaval submit` / `kaval status`, not directly.")

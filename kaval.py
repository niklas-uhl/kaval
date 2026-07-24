#!/bin/env python

"""Top-level kaval entry point: `kaval <run|submit|status> ...`.

Thin dispatcher over the existing modules - run_experiments.py generates
(and, for local runs, executes) experiments; submit_experiments.py submits
already-generated SLURM job files and reports on their progress. Kept as
separate importable modules rather than folded together so `python3
run-experiments.py ...` keeps working directly for existing scripts/muscle
memory.
"""

import sys

SUBCOMMANDS = {
    "run": "run_experiments.main",
    "submit": "submit_experiments.main_submit",
    "status": "submit_experiments.main_status",
}


def _usage():
    names = "|".join(SUBCOMMANDS)
    return (
        f"usage: kaval <{names}> [args...]\n\n"
        "  run     generate (and, for --machine shared, execute) experiments from a suite\n"
        "  submit  submit generated SLURM job files via sbatch, tracking job IDs\n"
        "  status  report submission/completion status for a suite's job files\n"
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_usage())
        sys.exit(1 if len(sys.argv) < 2 else 0)

    subcommand = sys.argv[1]
    if subcommand not in SUBCOMMANDS:
        sys.exit(f"Unknown subcommand '{subcommand}'.\n\n{_usage()}")

    # Drop 'kaval.py' and the subcommand so the delegated argparse only sees
    # its own args, with argv[0] naming the subcommand for --help output.
    sys.argv = [f"kaval {subcommand}"] + sys.argv[2:]

    module_name, func_name = SUBCOMMANDS[subcommand].rsplit(".", 1)
    module = __import__(module_name)
    getattr(module, func_name)()


if __name__ == "__main__":
    main()

#!/bin/env python

# MIT License
#
# Copyright (c) 2020-2024 Tim Niklas Uhl
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

from runners import *
import expcore
from expcore import parse_time_limit
import argparse, os, sys
from pathlib import Path


def load_suites(suite_files, search_paths):
    instance_sets = expcore.load_instance_sets(search_paths)
    suites = {}
    for path in suite_files:
        suite = expcore.load_suite_from_yaml(path, instance_sets)
        suites[suite.name] = suite
    for path in search_paths:
        if not path:
            continue
        for file in os.listdir(path):
            if file.endswith(".suite.yaml"):
                suite = expcore.load_suite_from_yaml(
                    os.path.join(path, file), instance_sets
                )
                suites[suite.name] = suite
    return suites


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "suite",
        nargs="*",
        help="Name(s) of the experiment suite for which job files will be generated.",
    )

    default_search_dirs = os.environ.get(
        "SUITE_SEARCH_PATH", default=os.getcwd()
    ).split(":")
    default_search_dirs.append(os.path.dirname(__file__))
    parser.add_argument("--search-dirs", nargs="*", default=default_search_dirs)
    parser.add_argument("--suite-files", nargs="*", default=[])

    parser.add_argument("--output-dir")
    default_experiment_data_dir = os.environ.get(
        "EXPERIMENT_DATA_DIR", Path(os.getcwd()) / "experiment_data"
    )
    parser.add_argument(
        "--experiment-data-dir",
        default=default_experiment_data_dir,
        help="Directory in which all relevant generated data (jobfiles and outputs) are stored.",
    )
    parser.add_argument(
        "--sbatch-template",
        help="The path to the sbatch template to be used. The template likely needs to be adapted for each execution platform.",
    )
    parser.add_argument(
        "--command-template",
        help="The path to the command template to be used. The template likely needs to be adapted for each mpi implementation/single-threaded vs. multithreaded execution.",
    )
    parser.add_argument(
        "--module-config",
        help="Name of module config which should be loaded before the job is executed using module load <config>.",
    )
    parser.add_argument(
        "--module-restore-cmd",
        default="module restore",
        help="Provide a command to restore the module config.",
    )

    parser.add_argument("--list", action="store_true")

    parser.add_argument("--job-output-dir")

    default_machine_type = os.environ.get("MACHINE", "generic-job-file")
    parser.add_argument(
        "--machine",
        choices=["shared", "horeka", "supermuc", "lichtenberg", "generic-job-file"],
        default=default_machine_type,
    )
    parser.add_argument(
        "--tasks-per-node", default=os.environ.get("TASKS_PER_NODE", None), type=int
    )
    parser.add_argument("--max-cores", default=None, type=int, help="Upper bound on core counts; overrides suite max_cores (default 8000)")
    parser.add_argument("--min-cores", default=None, type=int, help="Lower bound on core counts; overrides suite min_cores (default 1)")
    parser.add_argument("--cores", nargs="*", help="Override suite ncores with an explicit list of core counts or a special keyword: 'pow2' for powers of two (1,2,4,...), 'sqr' for square numbers (1,4,9,16,...), 'sqr-pow2' for powers of two that are also squares (1,4,16,64,...), 'node-size-pow2' for powers-of-two multiples of tasks_per_node")
    parser.add_argument(
        "-t", "--time-limit",
        default=parse_time_limit(os.environ["TIME_LIMIT"]) if "TIME_LIMIT" in os.environ else None,
        type=parse_time_limit,
        help="Time limit per job. Accepts minutes (number), 'H:MM', 'H:MM:SS', 'D-HH:MM:SS', or suffix style like '1h30m'.",
    )

    parser.add_argument("--test", action="store_true")

    parser.add_argument(
        "--omit-output-path",
        action="store_true",
        help="Do not pass specific json output path parameter to the executable",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Cleans the experiment data directory before starting the experiment"
    )
    parser.add_argument(
        "--copy-binary",
        action="store_true",
        help="Copy the suite's executable into <experiment-dir>/bin and point all "
        "generated commands/job files at the copy, so later rebuilds of the "
        "binary don't change what the generated jobs execute."
    )
    parser.add_argument(
        "--no-date-suffix",
        action="store_true",
        help="Do not append a date suffix to the experiment data and output directories"
    )
    parser.add_argument(
        "--name",
        help="Override the directory name for the suite. Errors if multiple suites would produce the same name.",
    )
    parser.add_argument(
        "--prefix",
        help="Prepend PREFIX_ to each suite's directory name.",
    )
    parser.add_argument(
        "--suffix",
        help="Append _SUFFIX to each suite's directory name (inserted before the date).",
    )

    args = parser.parse_args()
    suites = load_suites(args.suite_files, args.search_dirs)

    for suitename in args.suite:
        suite = suites.get(suitename)

    if args.list:
        for name in suites.keys():
            print(name)
        sys.exit(0)

    if not args.suite:
        args.suite = list(suites.keys())

    def effective_name(suite_name):
        if args.name:
            return args.name
        name = suite_name
        if args.prefix:
            name = args.prefix + "_" + name
        if args.suffix:
            name = name + "_" + args.suffix
        return name

    active_suites = [s for s in args.suite if suites.get(s)]
    names = [effective_name(s) for s in active_suites]
    seen, dupes = set(), set()
    for n in names:
        (dupes if n in seen else seen).add(n)
    if dupes:
        sys.exit(f"Error: --name/--prefix/--suffix would produce duplicate directory names: {sorted(dupes)}")

    for suitename in active_suites:
        suite = suites.get(suitename)
        runner = get_runner(args, suite, name_override=effective_name(suitename))
        runner.execute(suite)


if __name__ == "__main__":
    main()

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
import argparse, os, sys
from pathlib import Path


def load_suites(suite_files, search_paths):
    suites = {}
    for path in suite_files:
        suite = expcore.load_suite_from_yaml(path)
        suites[suite.name] = suite
    for path in search_paths:
        if not path:
            continue
        for file in os.listdir(path):
            if file.endswith(".suite.yaml"):
                suite = expcore.load_suite_from_yaml(os.path.join(path, file))
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
    parser.add_argument("--max-cores", default=sys.maxsize, type=int)
    parser.add_argument(
        "-t", "--time-limit", default=os.environ.get("TIME_LIMIT", 20), type=int
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

    args = parser.parse_args()
    suites = load_suites(args.suite_files, args.search_dirs)

    for suitename in args.suite:
        suite = suites.get(suitename)

    if args.list:
        for name in suites.keys():
            print(name)
        sys.exit(0)

    if not args.suite:
        args.suite = suites.keys()

    for suitename in args.suite:
        suite = suites.get(suitename)
        if suite:
            runner = get_runner(args, suite)
            runner.execute(suite)


if __name__ == "__main__":
    main()

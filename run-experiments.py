# MIT License
#
# Copyright (c) 2020-2023 Tim Niklas Uhl
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
            if file.endswith('.suite.yaml'):
                suite = expcore.load_suite_from_yaml(os.path.join(path, file))
                suites[suite.name] = suite
    return suites


def load_inputs(input_descriptions):
    inputs = {}
    partitions = {}
    for path in input_descriptions:
        sub_inputs, sub_partitions = expcore.load_inputs_from_yaml(path)
        inputs.update(sub_inputs)
        partitions.update(sub_partitions)
    return (inputs, partitions)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('suite', nargs='*')

    default_search_dirs = os.environ.get("SUITE_SEARCH_PATH",
                                         default=os.getcwd()).split(":")
    default_search_dirs.append(os.path.dirname(__file__))
    parser.add_argument('-d',
                        '--search-dirs',
                        nargs='*',
                        default=default_search_dirs)
    parser.add_argument('-s', '--suite-files', default=[])

    default_inputs = os.environ.get("INPUT_DESCRIPTIONS", default=[])
    if default_inputs:
        default_inputs = default_inputs.split(":")
    #default_inputs.append(
    #    Path(os.path.dirname(__file__)) / ".." / "examples" / 'examples.yaml')
    parser.add_argument('-i', '--input-descriptions', nargs='*', default=[])

    parser.add_argument('-o', '--output-dir')
    default_experiment_data_dir = os.environ.get(
        "EXPERIMENT_DATA_DIR",
        Path(os.getcwd()) / "experiment_data")
    parser.add_argument(
        '--experiment-data-dir',
        default=default_experiment_data_dir,
        help=
        "Directory in which all relevant generated data (jobfiles and outputs) are stored."
    )
    script_path = Path(os.path.dirname(__file__))
    parser.add_argument('--sbatch-template',
                        default=script_path / "sbatch-templates/sbatch_template.txt")
    parser.add_argument('--command-template',
                        default=script_path / "command-templates/command_template_intel.txt")
    parser.add_argument('--module-config')

    parser.add_argument('-l', '--list', action='store_true')
    parser.add_argument('-v', '--verify', action='store_true')

    parser.add_argument('-g', '--list-graphs', action='store_true')

    parser.add_argument('-j', '--job-output-dir')

    default_machine_type = os.environ.get("MACHINE", 'generic-job-file')
    parser.add_argument('-m',
                        '--machine',
                        choices=['shared', 'supermuc', 'lichtenberg', 'generic-job-file'],
                        default=default_machine_type)
    parser.add_argument('--tasks-per-node',
                        default=os.environ.get("TASKS_PER_NODE", 48),
                        type=int)
    parser.add_argument('-t',
                        '--time-limit',
                        default=os.environ.get("TIME_LIMIT", 20),
                        type=int)

    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    suites = load_suites(args.suite_files, args.search_dirs)
    inputs, partitions = load_inputs(args.input_descriptions + default_inputs)
    for suitename in args.suite:
        suite = suites.get(suitename)
        if suite:
            suite.load_inputs(inputs, partitions)

    if args.list:
        for name in suites.keys():
            print(name)
        sys.exit(0)
    if args.list_graphs:
        for name in inputs.keys():
            print(name)
        sys.exit(0)

    if not args.suite:
        args.suite = suites.keys()

    for suitename in args.suite:
        suite = suites.get(suitename)
        print(suite)
        if suite:
            runner = get_runner(args, suite)
            runner.execute(suite)

    if args.machine == 'shared':
        print(
            f"Summary: {runner.failed} jobs failed, {runner.incorrect} jobs returned an incorrect result."
        )


if __name__ == "__main__":
    main()

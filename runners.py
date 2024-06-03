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

from expcore import ExperimentSuite, explode, FileInputGraph, GenInputGraph
import expcore
from pathlib import Path
import subprocess, sys, json, os
import math as m
from string import Template
import time
import slugify
from datetime import date


class SharedMemoryRunner:

    def __init__(self, suite_name, experiment_data_directory, output_directory, verify_results=False):
        data_suffix = date.today().strftime("%y_%m_%d")
        self.experiment_data_directory = Path(experiment_data_directory) / (
            suite_name + "_" + data_suffix)
        self.experiment_data_directory.mkdir(exist_ok=True, parents=True)
        self.output_directory = Path(
            output_directory) if output_directory else (
                self.experiment_data_directory / "output")
        self.output_directory.mkdir(exist_ok=True, parents=True)
        self.verify_results = verify_results
        self.failed = 0
        self.incorrect = 0

    def execute(self, experiment_suite: ExperimentSuite):
        print(f"Running suite {experiment_suite.name} ...")
        with open(self.output_directory / "config.json", 'w') as file:
            json.dump(experiment_suite.configs, file, indent=4)
        for i, config in enumerate(experiment_suite.configs):
            for input in experiment_suite.inputs:
                for ncores in experiment_suite.cores:
                    # print(experiment_suite.threads_per_rank)
                    for threads in experiment_suite.threads_per_rank:
                        local_config = config.copy()
                        mpi_ranks = ncores // threads
                        if isinstance(input, expcore.InputGraph):
                            input_name = input.name
                        else:
                            input_name = str(input)
                        jobname = f"{input_name}-np{mpi_ranks}-t{threads}"
                        config_job_name = jobname + "-c" + str(i)
                        json_output_prefix_path = self.output_directory / f"{config_job_name}_timer.json"
                        local_config['json_output_path'] = str(json_output_prefix_path)
                        log_path = self.output_directory / f"{config_job_name}-log.txt"
                        err_path = self.output_directory / f"{config_job_name}-err.txt"
                        mpiexec = os.environ.get("MPI_EXEC", "mpiexec")
                        cmd = mpiexec.split(" ")
                        cmd += ["-np", str(mpi_ranks)]
                        cmd += expcore.command(experiment_suite.executable, ".", input, mpi_ranks, threads, escape=False,
                                                   **local_config)
                        print(
                            f"Running config {i} on {input_name} using {mpi_ranks} ranks and {threads} threads per rank ... ",
                            end='')
                        print(cmd)
                        sys.stdout.flush()
                        with open(log_path, 'w') as log_file:
                            with open(err_path, 'w') as err_file:
                                ret = subprocess.run(cmd,
                                                     stdout=log_file,
                                                     stderr=err_file)
                        if ret.returncode == 0:
                            #if self.verify_results and input.triangles:
                            #    print('finished.', end='')
                            #    with open(log_path) as output:
                            #        triangles = int(
                            #            json.load(output)["stats"][0]
                            #            ["counted_triangles"])
                            #    if triangles == input.triangles:
                            #        print(' correct.')
                            #    else:
                            #        self.incorrect += 1
                            #        print(' incorrect.')
                            #else:
                            print('finished.')
                        else:
                            self.failed += 1
                            print('failed.')
        print(f"Finished suite {experiment_suite.name}.")


def get_queue(cores, tasks_per_node):
    nodes = required_nodes(cores, tasks_per_node)
    if nodes <= 16:
        return "micro"
    elif nodes <= 768:
        return "general"
    else:
        return "large"


def required_nodes(cores, tasks_per_node):
    return int(max(int(m.ceil(float(cores) / tasks_per_node)), 1))


def required_islands(nodes):
    if nodes > 768:
        return 2
    else:
        return 1


class SBatchRunner:

    def __init__(self,
                 suite_name,
                 experiment_data_directory,
                 machine,
                 output_directory,
                 job_output_directory,
                 sbatch_template,
                 command_template,
                 module_config,
                 tasks_per_node,
                 time_limit,
                 use_test_partition=False):
        # append experiment_data_dir with current date
        data_suffix = date.today().strftime("%y_%m_%d")
        self.experiment_data_directory = Path(experiment_data_directory) / (
            suite_name + "_" + data_suffix)
        self.experiment_data_directory.mkdir(exist_ok=True, parents=True)
        self.machine = machine
        self.output_directory = Path(
            output_directory) if output_directory else (
                self.experiment_data_directory / "output")
        self.job_output_directory = Path(
            job_output_directory) if job_output_directory else (
                self.experiment_data_directory / "jobfiles")
        self.output_directory.mkdir(exist_ok=True, parents=True)
        self.job_output_directory.mkdir(exist_ok=True, parents=True)

        self.sbatch_template = Path(sbatch_template)
        self.command_template = Path(command_template)
        self.module_config = module_config
        self.tasks_per_node = tasks_per_node
        self.time_limit = time_limit
        self.use_test_partition = use_test_partition

    def make_cmd_for_config(self, suite: ExperimentSuite, input,
                            config_job_name, config_index, mpi_ranks,
                            threads_per_rank, config):
        raise NotImplementedError("Please implement this method.")

    def execute(self, experiment_suite: ExperimentSuite):
        project = os.environ.get("PROJECT", "PROJECT_NOT_SET")
        with open(self.output_directory / "config.json", 'w') as file:
            json.dump(experiment_suite.configs, file, indent=4)
        script_path = Path(os.path.dirname(__file__))
        with open(self.sbatch_template) as template_file:
            template = template_file.read()
        template = Template(template)
        with open(self.command_template) as template_file:
            command_template = template_file.read()
        command_template = Template(command_template)
        njobs = 0
        for input in experiment_suite.inputs:
            if isinstance(input, expcore.InputGraph):
                input_name = input.name
            else:
                input_name = str(input)
            for ncores in experiment_suite.cores:
                if experiment_suite.tasks_per_node:
                    tasks_per_node = experiment_suite.tasks_per_node
                else:
                    tasks_per_node = self.tasks_per_node
                aggregate_jobname = f"{experiment_suite.name}-{input_name}-cores{ncores}"
                log_path = self.output_directory / f"{input_name}-cores{ncores}-log.txt"
                err_log_path = self.output_directory / f"{input_name}-cores{ncores}-error-log.txt"
                subs = {}
                nodes = required_nodes(ncores, tasks_per_node)
                subs["nodes"] = nodes
                if self.machine == "lichtenberg":
                    subs["ntasks"] = ncores
                subs["output_log"] = str(log_path)
                subs["error_output_log"] = str(err_log_path)
                subs["job_name"] = aggregate_jobname
                if self.use_test_partition:
                    subs["job_queue"] = "test"
                else:
                    subs["job_queue"] = get_queue(ncores, tasks_per_node)
                subs["islands"] = required_islands(nodes)
                subs["account"] = project
                if self.module_config:
                    module_setup = f"module restore {self.module_config}"
                    subs["module_setup"] = module_setup
                else:
                    subs["module_setup"] = "# no specific module setup given"
                time_limit = 0
                commands = []
                for threads_per_rank in experiment_suite.threads_per_rank:
                    mpi_ranks = ncores // threads_per_rank
                    ranks_per_node = tasks_per_node // threads_per_rank
                    jobname = f"{input_name}-np{mpi_ranks}-t{threads_per_rank}"
                    for i, config in enumerate(experiment_suite.configs):
                        job_time_limit = experiment_suite.get_input_time_limit(
                            input.name)
                        if not job_time_limit:
                            job_time_limit = self.time_limit
                        time_limit += job_time_limit
                        config_jobname = jobname + "-c" + str(i)
                        cmd = self.make_cmd_for_config(experiment_suite, input,
                                                       config_jobname, i,
                                                       mpi_ranks,
                                                       threads_per_rank,
                                                       config)
                        cmd_string = command_template.substitute(
                            cmd=" ".join(cmd),
                            jobname=config_jobname,
                            mpi_ranks=mpi_ranks,
                            threads_per_rank=threads_per_rank,
                            ranks_per_node=ranks_per_node,
                            timeout=job_time_limit * 60)
                        commands.append(cmd_string)
                subs["commands"] = '\n'.join(commands)
                subs["time_string"] = time.strftime(
                    "%H:%M:%S", time.gmtime(time_limit * 60))
                job_script = template.substitute(subs)
                job_file = self.job_output_directory / aggregate_jobname
                with open(job_file, "w+") as job:
                    job.write(job_script)
                njobs += 1
        print(
            f"Created {njobs} job files in directory {self.job_output_directory}."
        )


class MQSBatchRunner(SBatchRunner):

    def __init__(self,
                 suite_name,
                 experiment_data_directory,
                 machine,
                 output_directory,
                 job_output_directory,
                 sbatch_template,
                 command_template,
                 module_config,
                 tasks_per_node,
                 time_limit,
                 use_test_partition=False):
        SBatchRunner.__init__(self, suite_name, experiment_data_directory, machine,
                              output_directory, job_output_directory,
                              sbatch_template, command_template, module_config,
                              tasks_per_node, time_limit, use_test_partition)

    def make_cmd_for_config(self, suite: ExperimentSuite, input,
                            config_job_name, config_index, mpi_ranks,
                            threads_per_rank, config):
        json_output_prefix_path = self.output_directory / f"{config_job_name}_timer.json"
        config = config.copy()
        config['json_output_path'] = str(json_output_prefix_path)
        cmd = expcore.command(suite.executable, ".", input, mpi_ranks, threads_per_rank, escape=True, **config)
        return cmd


def get_runner(args, suite):
    # print("type: ", suite.suite_type)
    if args.machine == 'shared':
        runner = SharedMemoryRunner(suite.name, args.experiment_data_dir, args.output_dir,
                                    verify_results=args.verify)
        return runner
    elif args.machine == 'supermuc' or args.machine == 'generic-job-file':
        return MQSBatchRunner(suite.name, args.experiment_data_dir,
                                       args.machine,
                                       args.output_dir, args.job_output_dir,
                                       args.sbatch_template,
                                       args.command_template,
                                       args.module_config, args.tasks_per_node,
                                       args.time_limit, args.test)
    else:
        exit("Unknown machine type: " + args.machine)

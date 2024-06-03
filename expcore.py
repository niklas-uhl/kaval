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

from genericpath import isfile
import subprocess
import logging
import os, re
from pathlib import Path
import yaml
import sys
import math
import slugify
import copy


class InputGraph:
    def __init__(self, name):
        self._name = name

    def args(self, mpi_rank, threads_per_rank, escape):
        raise NotImplementedError()


class FileInputGraph(InputGraph):
    def __init__(self, name, path, format='metis'):
        self._name = slugify.slugify(name)
        self.path = path
        self.format = format
        self.partitions = {}
        self.partitioned = False

    def args(self, mpi_ranks, threads_per_rank, escape):
        #file_args = [str(self.path), "--input-format", self.format]
        file_args = ["--graphtype", "BRAIN", "--infile_dir", str(self.path)]
        if self.partitioned and mpi_ranks > 1:
            partition_file = self.partitions.get(mpi_ranks, None)
            if not partition_file:
                logging.error(f"Could not load partitioning for p={mpi_ranks} for input {self.name}")
                sys.exit(1)
            file_args += ["--partitioning", partition_file]
        return file_args

    def add_partitions(self, partitions):
        self.partitions.update(partitions)

    @property
    def name(self):
        if self.partitioned:
            return self._name + "_partitioned"
        else:
            return self._name
    @name.setter
    def name(self, value):
        self._name = value

    def exists(self):
        if self.format == "metis":
            return self.path.exists()
        elif self.format == "binary":
            root = self.path.parent
            first_out = root / (self.path.stem + ".first_out")
            head = root / (self.path.stem + ".head")
            return first_out.exists() and head.exists()
        elif self.format == "brain_format":
            return True

    def __repr__(self):
        return f"FileInputGraph({self.name, self.path, self.format, self.partitioned, self.partitions})"


class GenInputGraph(InputGraph):

    parameter_list = {
        "rhg": ["m", "gamma"],
        "gnm": ["m"],
        "rgg2d": ["m"],
        "rgg3d": ["m"],
        "rmat": ["m"],
        "rdg2d": [],
        "rdg3d": [],
    }

    def __init__(self, generator, **kwargs):
        if generator not in GenInputGraph.parameter_list.keys():
            raise ValueError(f"Generator {generator} is not supported.")
        self.generator = generator
        self.params = kwargs
        self.scale_weak = self.params.get("scale_weak", False);
        required_parameters = set(
            GenInputGraph.parameter_list[self.generator] + ['n'])
        if not required_parameters.issubset(self.params.keys()):
            raise ValueError(
                f"Generator {self.generator} requires the following parameters: {required_parameters}"
            )

    def args(self, mpi_ranks, threads_per_rank, escape):
        p = mpi_ranks * threads_per_rank
        arg_list = ["--graphtype"]
        arg_list.append(self.generator);
        arg_list.append("--log_num_vertices")
        if self.scale_weak:
            if not math.log2(p).is_integer():
                sys.exit("Number of PEs must be a power of two")
            scaled_n = self.n(p)
            if "m" in self.params:
                scaled_m = self.m(p)
            else:
                scaled_m = None
            arg_list.append(str(scaled_n))
        else:
            arg_list.append(str(self.params["n"]))
        if self.generator == 'rgg2d':
            arg_list.append("--log_num_edges")
            arg_list.append(str(scaled_m))
        if self.generator == 'rgg3d':
            arg_list.append("--log_num_edges")
            arg_list.append(str(scaled_m))
        elif self.generator == 'rhg':
            arg_list.append("--gamma")
            arg_list.append(str(self.params["gamma"]))
            arg_list.append("--log_num_edges")
            arg_list.append(str(scaled_m))
        elif self.generator == 'gnm':
            arg_list.append("--log_num_edges")
            arg_list.append(str(scaled_m))
        elif self.generator == 'rmat':
            arg_list.append("--log_num_edges")
            arg_list.append(str(scaled_m))
            if 'a' in self.params:
                arg_list.append("--gen_a")
                arg_list.append(str(a))
            if 'b' in self.params:
                arg_list.append("--gen_b")
                arg_list.append(str(b))
            if 'c' in self.params:
                arg_list.append("--gen_c")
                arg_list.append(str(c))
        #arg_list.append("--gen_statistics")
        return arg_list

    def n(self, p):
        if self.scale_weak:
            return self.params["n"] + int(math.log2(p))
        else:
            return self.params["n"]

    def m(self, p):
        if self.scale_weak:
            return self.params["m"] + int(math.log2(p))
        else:
            return self.params["m"]

    @property
    def name(self):
        n = self.params["n"]
        name = f"{self.generator.upper()}({n}"
        for key in GenInputGraph.parameter_list[self.generator]:
            val = self.params[key]
            name += f"-{key}={val}"
        name += ")"
        if self.scale_weak:
            name += "_weak"
        return slugify.slugify(name)


def load_inputs_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    inputs = {}
    for rec in data["graphs"]:
        name = rec['name']
        path = Path(yaml_path).parent / rec['path']
        format = rec['format']
        graph = FileInputGraph(name, path, format)
        if not graph.exists():
            logging.warn(f"Could not load graph {graph.name}")
        else:
            inputs[graph.name] = graph
    partitions = {}
    if "includes" in data:
        for rec in data["includes"]:
            sub_inputs, sub_partitions = load_inputs_from_yaml(Path(yaml_path).parent / rec)
            inputs.update(sub_inputs)
            partitions.update(sub_partitions)
    if "partitions" in data:
        root = Path(yaml_path).parent / data["partitions"]
        for file in os.listdir(root):
            if os.path.isfile(os.path.join(root, file)):
                m = re.match(r"(.*)_k([0-9]+)", file)
                if not m:
                    logging.warn(f"Invalid partition name {file}")
                key = (m.group(1), int(m.group(2)))
                if not key[0] in partitions:
                    partitions[key[0]] = {}
                partitions[key[0]][key[1]] = os.path.join(root, file)
    #print(partitions)
    return (inputs, partitions)


class KaGenGraph(InputGraph):

    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        if not "type" in kwargs:
            raise ValueError("KaGen graph requires a type")
        try:
            self.n = kwargs.get("n", 1 << int(kwargs["N"]))
        except TypeError:
            self.n = None
        try:
            self.m = kwargs.get("m", 1 << int(kwargs["M"]))
        except TypeError:
            self.m = None
        kwargs.pop("n", None);
        kwargs.pop("N", None);
        kwargs.pop("m", None);
        kwargs.pop("M", None);
        self.scale_weak = kwargs.get("scale_weak", False);
        kwargs.pop("scale_weak", False)
        self.params = kwargs


    def get_n(self, p):
        if self.scale_weak:
            return self.n * p;
        else:
            return self.n

    def get_m(self, p):
        if self.scale_weak:
            return self.m * p;
        else:
            return self.m

    def args(self, mpi_ranks, threads_per_rank, escape):
        p = mpi_ranks * threads_per_rank
        params = self.stringify_params()
        if self.n:
            params.append(f"n={self.get_n(p)}")
        if self.m:
            params.append(f"m={self.get_m(p)}")
        kagen_option_string = ";".join(params)
        if escape:
            kagen_option_string = '"{}"'.format(kagen_option_string)
        return ["--kagen_option_string", kagen_option_string]

    def stringify_params(self):
        param_strings = []
        for (key, value) in  self.params.items():
            if isinstance(value, bool):
                param_strings.append(key)
            else:
                param_strings.append(f"{key}={value}")
        return param_strings

    @property
    def name(self):
        params = []
        if self.n:
            params.append(f"n={int(math.log2(self.n))}")
        if self.m:
            params.append(f"m={int(math.log2(self.m))}")
        params += self.stringify_params()
        if self.scale_weak:
            params.append("weak")
        name = f"KaGen_{'_'.join(params)}"
        return slugify.slugify(name)

class DummyInstance(InputGraph):
    def __init__(self, **kwargs):
        self.name_ = kwargs["name"]
        self.params = kwargs.copy()
        self.params.pop("name", None)

    def args(self, mpi_rank, treeads_per_rank, escape):
        params = []
        for (key, value) in self.params.items():
            params.append(f"--{key}");
            if not isinstance(value, bool):
                params.append(f"{value}")
        return params

    @property
    def name(self):
        param_strings = []
        for (key, value) in  self.params.items():
            if isinstance(value, bool):
                param_strings.append(key)
            else:
                param_strings.append(f"{key}={value}")
        name = self.name_ + "_" + "_".join(param_strings)
        return slugify.slugify(name)

class ExperimentSuite:
    def __init__(self,
                 name: str,
                 suite_type: str,
                 executable: None,
                 cores=[],
                 threads_per_rank=[1],
                 inputs=[],
                 configs=[],
                 tasks_per_node=None,
                 time_limit=None,
                 input_time_limit={}):
        self.name = name
        self.suite_type = suite_type
        self.executable = executable
        self.cores = cores
        self.threads_per_rank = threads_per_rank
        self.inputs = inputs
        self.configs = configs
        self.tasks_per_node = tasks_per_node
        self.time_limit = time_limit
        self.input_time_limit = input_time_limit

    def set_input_time_limit(self, input_name, time_limit):
        self.input_time_limit[input_name] = time_limit

    def get_input_time_limit(self, input_name):
        return self.input_time_limit.get(input_name, self.time_limit)

    def load_inputs(self, input_dict, partitions):
        inputs_new = []
        for graph in self.inputs:
            if isinstance(graph, str):
                graph = {"name": graph, "partitioned": False}
            elif isinstance(graph, tuple):
                graph_name, partitioned = graph
                graph = {"name": graph_name, "partitioned": partitioned}
            else:
                inputs_new.append(graph)
                continue
            input = copy.copy(input_dict.get(graph["name"]))
            if not input:
                logging.warn(f"Could not load input for {graph_name}")
                continue
            if graph["partitioned"]:
                input.add_partitions(partitions.get(graph["name"], {}))
                input.partitioned = graph["partitioned"]
            # print(input)
            inputs_new.append(input)

        self.inputs = inputs_new
        # print(self.inputs)

    def __repr__(self):
        return f"ExperimentSuite({self.name}, {self.cores}, {self.inputs}, {self.configs}, {self.time_limit}, {self.input_time_limit})"


def load_suite_from_yaml(path):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    configs = []
    if "config" not in data:
        configs = [dict()]
    elif type(data["config"]) == list:
        for config in data["config"]:
            configs = configs + explode(config)
    else:
        configs = explode(data["config"])
    inputs = []
    time_limits = {}
    for graph in data["graphs"]:
        if type(graph) == str:
            inputs.append(graph)
        else:
            if "generator" in graph:
                generator = graph.pop("generator")
                if (generator == "kagen"):
                    inputs.append(KaGenGraph(**graph))
                elif (generator == "dummy"):
                    inputs.append(DummyInstance(**graph))
                else:
                    inputs.append(GenInputGraph(generator, **graph))
            elif "name" in graph:
                partitioned = graph.get("partitioned", False)
                inputs.append((graph["name"], partitioned))
            time_limit = graph.get("time_limit")
            if time_limit:
                time_limits[graph["name"]] = time_limit
    if "type" in data:
        suite_type = data["type"]
    else:
        suite_type = "BFS"
    if "executable" in data:
        #if Path(path).is_absolute():
        #    executable = data["executable"]
        #else:
        executable = data["executable"]
    else:
        executable = None
    return ExperimentSuite(data["name"],
                           suite_type,
                           executable,
                           data["ncores"],
                           data.get("threads_per_rank", [1]),
                           inputs,
                           configs,
                           tasks_per_node=data.get("tasks_per_node"),
                           time_limit=data.get("time_limit"),
                           input_time_limit=time_limits)


def explode(config):
    configs = []
    for flag, value in config.items():
        if type(value) == list:
            for arg in value:
                exploded = config.copy()
                exploded[flag] = arg
                exp = explode(exploded)
                configs = configs + exp
            break
    if not configs:
        return [config]
    return configs

def params_to_flags(params):
    flags = []
    for flag, value in params.items():
        dash = "-"
        if (len(flag) > 1):
            dash += "-"
        if isinstance(value, bool):
            if value:
                flags.append(dash + flag)
        else:
            flags.append(dash + flag)
            flags.append(str(value))
    return flags


def command(binary_name, binary_path, input, mpi_ranks, threads_per_rank, escape, **kwargs):
    script_path = os.path.dirname(__file__)
    build_dir = Path(
        os.environ.get("BUILD_DIR", os.path.join(script_path, "../build/")))
    app = build_dir / binary_path / binary_name
    # print(build_dir, binary_path, binary_name)
    command = [str(app)]
    if input:
        if isinstance(input, InputGraph):
            command = command + input.args(mpi_ranks, threads_per_rank, escape)
        else:
            command.append(str(input))
    flags = []
    #flags = ["--num-threads", str(threads_per_rank)]
    flags = flags + params_to_flags(kwargs)
    command = command + flags
    return command

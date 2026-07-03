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

import copy
import logging
import math
import os
import re
import subprocess
import sys
from enum import Enum
from genericpath import isfile
from pathlib import Path

import yaml

import slugify


class CLIArgumentType(Enum):
    FLAG = ("flag",)
    POSITIONAL = "positional"
    POSITIONAL_LIST = "positional_list"
    FLAG_LIST = "positional_list"


def get_argument_type_from_str(arg_type: str) -> CLIArgumentType:
    if arg_type == "flag":
        return CLIArgumentType.FLAG
    elif arg_type == "positional":
        return CLIArgumentType.POSITIONAL
    elif arg_type == "positional_list":
        return CLIArgumentType.POSITIONAL_LIST
    elif arg_type == "flag_list":
        return CLIArgumentType.FLAG_LIST
    else:
        print("Found undefined argument type %s." % arg_type)
        sys.exit(1)


def parse_time_limit(value):
    """Parse a time limit to seconds.

    Accepts:
    - int or float, or a plain numeric string: interpreted as minutes
    - suffix notation: combinations of d/h/m/s, e.g. "1h30m", "90m", "2h"
    - colon-separated H:MM or H:MM:SS (hours:minutes[:seconds])
    - SLURM day-prefix D-HH:MM:SS or D-HH:MM
    """
    if isinstance(value, (int, float)):
        return int(value * 60)
    s = str(value).strip()
    try:
        return int(float(s) * 60)
    except ValueError:
        pass
    m = re.fullmatch(r'(\d+)-(\d+):(\d+)(?::(\d+))?', s)
    if m:
        d, h, mins, secs = m.groups()
        return int(d) * 86400 + int(h) * 3600 + int(mins) * 60 + int(secs or 0)
    m = re.fullmatch(r'(\d+):(\d+)(?::(\d+))?', s)
    if m:
        h, mins, secs = m.groups()
        return int(h) * 3600 + int(mins) * 60 + int(secs or 0)
    m = re.fullmatch(r'(?:(\d+(?:\.\d+)?)d)?(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?', s, re.IGNORECASE)
    if m and any(g is not None for g in m.groups()):
        d, h, mins, secs = (float(x) if x else 0.0 for x in m.groups())
        return int(d * 86400 + h * 3600 + mins * 60 + secs)
    raise ValueError(
        f"Cannot parse time limit {value!r}. Use minutes (number), "
        "'H:MM', 'H:MM:SS', 'D-HH:MM:SS', or suffix style like '1h30m'."
    )


def is_argument_flag_only(arg_type, arg_value):
    return arg_type == CLIArgumentType.FLAG and isinstance(arg_value, bool)


def is_argument_positional(arg_type):
    return arg_type in [CLIArgumentType.POSITIONAL, CLIArgumentType.POSITIONAL_LIST]


def for_each_argument(args, handler):
    for key, value in args.items():
        argument_type = CLIArgumentType.FLAG  # default

        if isinstance(value, dict):
            argument_type = get_argument_type_from_str(value["type"])
            value = value["value"]  # flatten param

        handler(argument_type, key, value)


class InputGraph:
    def __init__(self, name):
        self._name = name

    def args(self, mpi_rank, threads_per_rank, escape):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @property
    def short_name(self):
        return self.name


class FileInputGraph(InputGraph):
    def __init__(self, name, path, format="metis"):
        self._name = slugify.slugify(name)
        self.path = path
        self.format = format
        self.partitions = {}
        self.partitioned = False

    def args(self, mpi_ranks, threads_per_rank, escape):
        # file_args = [str(self.path), "--input-format", self.format]
        file_args = ["--graphtype", "BRAIN", "--infile_dir", str(self.path)]
        if self.partitioned and mpi_ranks > 1:
            partition_file = self.partitions.get(mpi_ranks, None)
            if not partition_file:
                logging.error(
                    f"Could not load partitioning for p={mpi_ranks} for input {self.name}"
                )
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


def stringify_params(params):
    param_strings = []
    for key, value in params.items():
        if isinstance(value, bool):
            # KaGen reads a bare flag as true and its absence as false, so only
            # emit the key when the flag is set; a False flag must be omitted.
            if value:
                param_strings.append(key)
        else:
            param_strings.append(f"{key}={value}")
    return param_strings


class KaGenGraph(InputGraph):

    def __init__(self, **kwargs):
        # deep copy so nested structures (e.g. extra_args) are not aliased
        # across variants produced by explode()'s shallow copies
        kwargs = copy.deepcopy(kwargs)
        try:
            self.type = kwargs.get("type")
        except KeyError:
            raise ValueError("KaGen graph requires a type")
        try:
            self.n = kwargs.get("n", 1 << int(kwargs["N"]))
        except (TypeError, KeyError):
            self.n = None
        try:
            self.m = kwargs.get("m", 1 << int(kwargs["M"]))
        except (TypeError, KeyError):
            self.m = None
        # handle extra_args similar to DummyInstance
        self.extra_args = kwargs.get("extra_args", {})
        self.parse_scale_weak_params({"scale_weak": kwargs.get("extra_args_scale_weak")})
        # cleanup consumed keys
        kwargs.pop("n", None)
        kwargs.pop("N", None)
        kwargs.pop("m", None)
        kwargs.pop("M", None)
        self.scale_weak = kwargs.get("scale_weak", False)
        kwargs.pop("scale_weak", False)
        kwargs.pop("extra_args", None)
        kwargs.pop("extra_args_scale_weak", None)
        self.params = kwargs

    def parse_scale_weak_params(self, kwargs):
        scale_weak_params = kwargs.get("scale_weak")
        if scale_weak_params is None:
            self.extra_scale_weak_params = []
        else:
            if not isinstance(scale_weak_params, list):
                scale_weak_params = [scale_weak_params]
            self.extra_scale_weak_params = scale_weak_params

    def do_scale_parameter(self, parameter):
        return parameter in getattr(self, "extra_scale_weak_params", [])

    def get_scaled_value(self, parameter, value, p):
        if self.do_scale_parameter(parameter):
            if not isinstance(value, int):
                raise ValueError(
                    f"Weak scaling parameter '{parameter}' has a non-integer value of '{value}'."
                )
            return int(value) * p
        else:
            return int(value)

    def args_for_extra(self, mpi_ranks, threads_per_rank):
        p = mpi_ranks * threads_per_rank
        params = []

        def parse_argument(arg_type, arg_key, arg_value):
            param = ""

            if not is_argument_positional(arg_type) and (
                not is_argument_flag_only(arg_type, arg_value) or arg_value
            ):
                param += "-"
                if len(arg_key) > 1:
                    param += "-"
                param += f"{arg_key}"

            if not is_argument_flag_only(arg_type, arg_value):
                if not isinstance(arg_value, list):
                    arg_value = [arg_value]  # pack arg_value

                if self.do_scale_parameter(arg_key):
                    param += " " + " ".join(
                        [
                            f'"{self.get_scaled_value(arg_key, val, p)}"'
                            for val in arg_value
                        ]
                    )
                else:
                    param += " " + " ".join([f'"{val}"' for val in arg_value])

            if param:
                params.append(param)

        for_each_argument(self.extra_args, parse_argument)
        return params

    def get_n(self, p):
        if self.scale_weak:
            return self.n * p
        else:
            return self.n

    def get_m(self, p):
        if self.scale_weak:
            return self.m * p
        else:
            return self.m

    def preprocess_file_based_graphs_params(self, mpi_ranks):
        params = self.params.copy()
        if params.get("type") not in ["partitioned_file", "partitioned-file"]:
            return params
        try:
            filename = params.get("filename")
        except KeyError:
            raise ValueError(
                "A partitioned file input graph requires a filename paramerter."
            )
        path, graph_format = filename.rsplit(".", 1)
        extended_filename = f"{path}_{mpi_ranks}.{graph_format}"
        params["type"] = (
            "file"  # to date KaGen does not possess a special type for pre-partitioned graphs
        )
        params["filename"] = extended_filename
        params["distribution"] = "explicit"
        params["explicit-distribution"] = extended_filename + ".partitions"
        return params

    def args(self, mpi_ranks, threads_per_rank, escape):
        p = mpi_ranks * threads_per_rank
        params = self.preprocess_file_based_graphs_params(mpi_ranks)
        params = stringify_params(params)
        if self.n:
            params.append(f"n={self.get_n(p)}")
        if self.m:
            params.append(f"m={self.get_m(p)}")
        kagen_option_string = ";".join(params)
        if escape:
            kagen_option_string = '"{}"'.format(kagen_option_string)
        # append extra_args after kagen option string
        extra = self.args_for_extra(mpi_ranks, threads_per_rank)
        return ["--kagen_option_string", kagen_option_string] + extra

    @property
    def name(self):
        params = []
        if self.n:
            params.append(f"n={int(math.log2(self.n))}")
        if self.m:
            params.append(f"m={int(math.log2(self.m))}")
        params += stringify_params(self.params)
        if self.scale_weak:
            params.append("weak")
        # include extra_args compactly
        extra_parts = []
        def add_extra(arg_type, arg_key, arg_value):
            if isinstance(arg_value, bool):
                extra_parts.append(arg_key)
            else:
                extra_parts.append(f"{arg_key}={arg_value}")
        for_each_argument(self.extra_args, add_extra)
        if extra_parts:
            params.append("extra=" + ",".join(extra_parts))
        name = f"KaGen_{'_'.join(params)}"
        return slugify.slugify(name)

    @property
    def short_name(self):
        name = f"{self.params['type']}"
        if self.n:
            name += f"_n={int(math.log2(self.n))}"
        return slugify.slugify(name)


class DummyInstance(InputGraph):
    def __init__(self, **kwargs):
        self.name_ = kwargs["name"]
        # deep copy so nested param structures are not aliased across variants
        # produced by explode()'s shallow copies
        self.params = copy.deepcopy(kwargs)
        self.params.pop("name", None)
        self.parse_scale_weak_params(self.params)
        self.params.pop("scale_weak", None)

    def parse_scale_weak_params(self, kwargs):
        scale_weak_params = kwargs.get("scale_weak")
        if scale_weak_params is None:
            self.scale_weak_params = []
        else:
            if not isinstance(scale_weak_params, list):
                scale_weak_params = [scale_weak_params]
            self.scale_weak_params = scale_weak_params

    def get_scaled_value(self, parameter, value, p):
        if self.do_scale_parameter(parameter):
            if not isinstance(value, int):
                raise ValueError(
                    f"Weak scaling parameter '{parameter}' has a non-integer value of '{value}'."
                )
            return int(value) * p
        else:
            return int(value)

    def do_scale_parameter(self, parameter):
        return parameter in self.scale_weak_params

    def args(self, mpi_ranks, threads_per_rank, escape):
        p = mpi_ranks * threads_per_rank
        params = []

        def parse_argument(arg_type, arg_key, arg_value):
            param = ""

            if not is_argument_positional(arg_type) and (
                not is_argument_flag_only(arg_type, arg_value) or arg_value
            ):
                param += "-"
                if len(arg_key) > 1:
                    param += "-"
                param += f"{arg_key}"

            if not is_argument_flag_only(arg_type, arg_value):
                if not isinstance(arg_value, list):
                    arg_value = [arg_value]  # pack arg_value

                if self.do_scale_parameter(arg_key):
                    param += " " + " ".join(
                        [
                            f'"{self.get_scaled_arg_value(arg_key, val, p)}"'
                            for val in arg_value
                        ]
                    )
                else:
                    param += " " + " ".join([f'"{val}"' for val in arg_value])

            if param:
                params.append(param)

        for_each_argument(self.params, parse_argument)
        return params

    @property
    def name(self):
        param_strings = []

        def parse_argument(arg_type, arg_key, arg_value):
            param = ""

            if not is_argument_positional(arg_type) and (
                not is_argument_flag_only(arg_type, arg_value) or arg_value
            ):
                param += f"{arg_key}"

            if arg_type != CLIArgumentType.FLAG or not isinstance(arg_value, bool):
                # pack arg_value
                if not isinstance(arg_value, list):
                    arg_value = [arg_value]  # pack arg_value

                # add arg_value to param
                param += "=" + ",".join([f'"{val}"' for val in arg_value])

            if param:
                param_strings.append(param)

        for_each_argument(self.params, parse_argument)
        name = self.name_ + "_" + "_".join(param_strings)
        return slugify.slugify(name)


class ExperimentSuite:
    def __init__(
        self,
        name: str,
        executable: None,
        output_path_option_name,
        cores=[],
        threads_per_rank=[1],
        inputs=[],
        configs=[],
        tasks_per_node=None,
        time_limit=None,
        seeds=[0],
        omit_seed=True,
        input_time_limit={},
        min_cores=None,
        max_cores=None,
    ):
        self.name = name
        self.executable = executable
        self.output_path_option_name = output_path_option_name
        self.cores = cores
        self.min_cores = min_cores
        self.max_cores = max_cores
        self.threads_per_rank = threads_per_rank
        self.inputs = inputs
        self.configs = configs
        self.tasks_per_node = tasks_per_node
        self.time_limit = time_limit
        self.seeds = seeds
        self.omit_seed = omit_seed
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


def load_instance_sets(search_paths):
    """Discover reusable instance sets from ``*.instances.yaml`` files.

    Each such file defines a named set of graphs (same format as a suite's
    ``graphs`` list) via a top-level ``name`` and ``graphs`` key. The returned
    dict maps set name to its raw graph list, ready to be spliced into a suite
    that imports it.
    """
    instance_sets = {}
    for path in search_paths:
        if not path:
            continue
        for file in os.listdir(path):
            if not file.endswith(".instances.yaml"):
                continue
            with open(os.path.join(path, file), "r") as f:
                data = yaml.safe_load(f)
            if not data or "graphs" not in data:
                continue
            name = data.get("name", file[: -len(".instances.yaml")])
            instance_sets[name] = data["graphs"]
    return instance_sets


def parse_graph_list(graph_list, instance_sets=None, _seen=None, overrides=None):
    """Expand a suite ``graphs`` list into InputGraph objects and time limits.

    Entries of the form ``{"import": <name>}`` are replaced by the graphs of the
    referenced instance set (see :func:`load_instance_sets`), recursively. An
    import entry may carry a ``with`` mapping whose keys are merged into every
    resolved graph (overriding existing keys) before it is built, e.g. to add
    ``permute: True`` to a whole set. ``with`` values may themselves be lists to
    explode. Returns ``(inputs, time_limits)``.
    """
    instance_sets = instance_sets or {}
    _seen = _seen if _seen is not None else set()
    inputs = []
    time_limits = {}
    for graph in graph_list:
        if type(graph) == str:
            if overrides:
                logging.warning(
                    "Cannot apply import modifications %s to file input '%s'; "
                    "including it unmodified.",
                    overrides,
                    graph,
                )
            inputs.append(graph)
            continue
        if "import" in graph:
            set_name = graph["import"]
            if set_name in _seen:
                raise ValueError(
                    f"Circular import of instance set '{set_name}'."
                )
            if set_name not in instance_sets:
                raise ValueError(
                    f"Unknown instance set '{set_name}'. "
                    f"Available: {sorted(instance_sets)}."
                )
            # merge this import's `with` block under any inherited overrides;
            # outer (use-site) overrides win on conflict
            merged_overrides = {**graph.get("with", {}), **(overrides or {})}
            _seen.add(set_name)
            sub_inputs, sub_limits = parse_graph_list(
                instance_sets[set_name], instance_sets, _seen, merged_overrides
            )
            _seen.discard(set_name)
            inputs.extend(sub_inputs)
            time_limits.update(sub_limits)
            continue
        # copy so that shared instance-set definitions are not mutated
        graph = copy.deepcopy(graph)
        if overrides:
            graph.update(copy.deepcopy(overrides))
        if "generator" in graph:
            generator = graph.pop("generator")
            time_limit_val = graph.pop("time_limit", None)
            if generator == "kagen":
                new_inputs = [KaGenGraph(**graph_variant) for graph_variant in explode(graph)]
            elif generator == "dummy":
                new_inputs = [DummyInstance(**graph_variant) for graph_variant in explode(graph)]
            else:
                raise ValueError(
                    f"'{generator}' is an unsupported argument for a graph generator. Use ['kagen', 'dummy'] instead."
                )
            inputs.extend(new_inputs)
            if time_limit_val is not None:
                for inp in new_inputs:
                    time_limits[inp.name] = parse_time_limit(time_limit_val)
        else:
            raise ValueError(f"No generator defined for graph: {graph}.")
    return inputs, time_limits


def _input_key(graph):
    """Identity of an input for deduplication.

    Two inputs are considered the same run if they produce the same name:
    ``KaGenGraph``/``DummyInstance`` names are derived from all their params,
    and bare file references are just their string.
    """
    if isinstance(graph, InputGraph):
        return graph.name
    return str(graph)


def dedup_inputs(inputs):
    """Drop duplicate inputs, keeping first occurrence to preserve order.

    Deep import graphs (e.g. two sets that both import a common set) can yield
    the same graph via multiple paths; running it more than once just repeats
    identical jobs. Returns the de-duplicated list; dropped names are logged.
    """
    seen = set()
    result = []
    dropped = []
    for graph in inputs:
        key = _input_key(graph)
        if key in seen:
            dropped.append(key)
            continue
        seen.add(key)
        result.append(graph)
    if dropped:
        logging.warning(
            "Dropped %d duplicate input(s): %s", len(dropped), ", ".join(dropped)
        )
    return result


def load_suite_from_yaml(path, instance_sets=None):
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
    inputs, time_limits = parse_graph_list(data["graphs"], instance_sets)
    inputs = dedup_inputs(inputs)
    if "executable" in data:
        executable = data["executable"]
    else:
        executable = None
    if "output_path_option_name" in data:
        output_path_option_name = data["output_path_option_name"]
    else:
        output_path_option_name = "json_output_path"

    return ExperimentSuite(
        data["name"],
        executable,
        output_path_option_name,
        data.get("ncores", []),
        data.get("threads_per_rank", [1]),
        inputs,
        configs,
        tasks_per_node=data.get("tasks_per_node"),
        time_limit=parse_time_limit(data["time_limit"]) if "time_limit" in data else None,
        seeds=data.get("seeds", [0]),
        omit_seed=data.get("omit_seed", True),
        input_time_limit=time_limits,
        min_cores=data.get("min_cores"),
        max_cores=data.get("max_cores"),
    )


def is_argument_explosive(arg_type, arg_val):
    if arg_type in [
        CLIArgumentType.FLAG_LIST,
        CLIArgumentType.POSITIONAL_LIST,
    ] and is_list_of_list(arg_val):
        # a suite.yaml snippet
        # BEGIN YAML
        # key:
        #   - type: "positional_list"
        #     value: [[1, 2], [2, 3], [3, 4]]
        # END YAML
        #
        # should result in three configurations:
        # ["\"1\" \"2\"", "\"2\" \"3\"", "\"3\" \"4\""]
        return True
    elif arg_type in [CLIArgumentType.FLAG, CLIArgumentType.POSITIONAL] and isinstance(
        arg_val, list
    ):
        # a suite.yaml snippet
        # BEGIN YAML
        # key:
        #   - type: "flag"
        #     value: [1, 2, 3]
        # END YAML
        #
        # should result in three configurations:
        # ["--key \"1\"", "--key \"2\"", "--key \"3\"]
        return True
    else:
        return False


def is_list_of_list(val):
    if not isinstance(val, list):
        return False
    for x in val:
        if not isinstance(x, list):
            return False
    return True


def explode(config):
    configs = []
    for flag, value in config.items():
        if isinstance(value, dict) and "value" in value and "type" in value:
            # assert "value" in value and "type" in value

            argument_type = get_argument_type_from_str(value["type"])
            argument_value = value["value"]
            if is_argument_explosive(argument_type, argument_value):
                for arg in argument_value:
                    if isinstance(arg, list):
                        exploded = copy.deepcopy(config)
                    else:
                        exploded = config.copy()
                    exploded[flag]["value"] = arg
                    exp = explode(exploded)
                    configs = configs + exp
                break

        elif isinstance(value, list):
            for arg in value:
                exploded = config.copy()
                exploded[flag] = arg
                exp = explode(exploded)
                configs = configs + exp
            break
    if not configs:
        return [config]

    return configs


def params_to_args(params):
    args = []

    def parse_argument(arg_type, arg_key, arg_value):
        arg = ""

        if not is_argument_positional(arg_type) and (
            not is_argument_flag_only(arg_type, arg_value) or arg_value
        ):
            arg += "-"
            if len(arg_key) > 1:
                arg += "-"
            arg += f"{arg_key}"

        if not is_argument_flag_only(arg_type, arg_value):
            if not isinstance(arg_value, list):
                arg_value = [arg_value]  # pack arg_value

            arg += " " + " ".join([f'"{val}"' for val in arg_value])

        if arg:
            args.append(arg)

    for_each_argument(params, parse_argument)
    return args


def command(
    binary_name, binary_path, input, mpi_ranks, threads_per_rank, escape, **kwargs
):
    script_path = os.path.dirname(__file__)
    build_dir = Path(
        os.environ.get("BUILD_DIR", os.path.join(script_path, "../build/"))
    )
    app = build_dir / binary_path / binary_name
    command = [str(app)]
    if input:
        if isinstance(input, InputGraph):
            command = command + input.args(mpi_ranks, threads_per_rank, escape)
        else:
            command.append(str(input))
    flags = []
    flags = flags + params_to_args(kwargs)
    command = command + flags
    return command

#!/bin/env python

# Backward-compat entry point. Prefer `kaval run` (see kaval.py). This file is
# kept so existing invocations of `python3 run-experiments.py ...` keep working.

from run_experiments import main

if __name__ == "__main__":
    main()

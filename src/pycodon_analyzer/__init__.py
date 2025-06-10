# src/pycodon_analyzer/__init__.py

# Copyright (c) 2025 Gabriel Falque
# Distributed under the terms of the MIT License.
# (See accompanying file LICENSE or copy at https://opensource.org/license/mit/)

# This file makes Python treat the directory as a package.
# Optionally, you can define package-level variables or import key functions.

import logging
import sys

# Only configure logging if not running under pytest
if 'pytest' not in sys.modules:
    # Configure root logger to use both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='[%H:%M:%S]',
        force=True  # Override any existing configuration
    )

    # Make sure all loggers propagate to the root logger
    for name in logging.root.manager.loggerDict:
        if name.startswith('pycodon_analyzer'):
            logging.getLogger(name).propagate = True

__version__ = "0.1.0" # Example version

# You could expose key functions here if desired, e.g.:
# from .analysis import calculate_rscu
# from .io import read_fasta
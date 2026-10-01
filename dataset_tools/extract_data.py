"""Compatibility entry point: ``python -m dataset_tools.extract_data`` runs :mod:`dataset_tools.extract.cli`."""

import sys

from dataset_tools.extract.cli import main

if __name__ == "__main__":
    sys.exit(main())

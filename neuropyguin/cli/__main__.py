"""Allow ``python -m neuropyguin.cli ...`` to run the command line interface."""

from __future__ import annotations

import sys

from .parser import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

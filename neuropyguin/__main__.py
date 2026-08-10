"""Allow ``python -m neuropyguin ...`` to run the CLI (or the GUI with no arguments)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""NeuroPyGuiN command line interface.

Exposes every workflow the GUI offers as a terminal command, grouped the same
way the window is:

===================  ====================================================
``preprocess``       Preprocessing tab: discovery, the CatGT/Kilosort
                     pipeline, concatenation, joint-sort splitting.
``curate``           Curation tab: phy, py_bombcell, unit labelling.
``postproc``         Post Processing tab: exports, PSTH, correlograms,
                     network analysis, cell-type classifiers.
``histology``        Histology tab: slice prep, atlas matching, alignment,
                     the IBL bridge.
``doctor``           Help > Run Diagnostics.
``tools``            The missing-tool installer.
``config``           Settings, including save/load and clear-history.
``gui`` / ``version``  Start the window, print versions.
===================  ====================================================

Entry points::

    python main.py <group> <command> ...
    python -m neuropyguin <group> <command> ...
    python -m neuropyguin.cli <group> <command> ...
"""

from __future__ import annotations

from .parser import build_parser, main

__all__ = ["build_parser", "main"]

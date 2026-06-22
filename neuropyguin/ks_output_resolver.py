"""Locate Kilosort output directories and related metrics files on disk.

The functions here take a user-supplied path (which may point at a file, a
parent folder, or the output directory itself) and search nearby directories
for a folder that contains a complete set of Kilosort result files. Candidate
directories are scored so the closest, best-matching folder is returned.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import re
from typing import Iterable


# Files that must all be present for a directory to count as Kilosort output.
KILOSORT_REQUIRED_FILES = (
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_templates.npy",
    "amplitudes.npy",
    "templates.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "whitening_mat_inv.npy",
)


# Relative locations of quality-metrics files, searched in priority order.
METRICS_RELATIVE_PATHS = ("metrics.csv", "bombcell/templates._bc_qMetrics.csv")


def has_kilosort_output(path: Path) -> bool:
    """Return True if ``path`` is a directory holding every required Kilosort file."""
    return path.exists() and path.is_dir() and all((path / name).exists() for name in KILOSORT_REQUIRED_FILES)


def _as_dir_candidate(path: str | Path) -> Path:
    """Coerce a path to its most likely directory form.

    Existing files map to their parent directory; non-existent paths that look
    like files (they have a suffix) also map to the parent, otherwise the path
    is returned unchanged.
    """
    p = Path(path).expanduser()
    if p.exists():
        return p if p.is_dir() else p.parent
    return p if not p.suffix else p.parent


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    """Drop case-insensitive duplicates while preserving first-seen order."""
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _iter_search_roots(requested: Path, extra_roots: Iterable[str | Path], parent_depth: int = 4) -> list[Path]:
    """Build the ordered, de-duplicated list of roots to search.

    Starts at the requested directory, walks up to ``parent_depth`` ancestors,
    then appends any caller-provided extra roots.
    """
    roots: list[Path] = [_as_dir_candidate(requested)]
    cur = roots[0]
    for _ in range(parent_depth):
        parent = cur.parent
        if parent == cur:
            break
        roots.append(parent)
        cur = parent
    for root in extra_roots:
        roots.append(_as_dir_candidate(root))
    return _dedupe_paths(roots)


def _walk_dirs(root: Path, max_depth: int) -> Iterable[tuple[Path, int]]:
    """Breadth-first walk yielding ``(directory, depth)`` pairs up to ``max_depth``.

    Children are visited in case-insensitive name order, already-seen paths are
    skipped, and unreadable directories are passed over silently.
    """
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    seen: set[str] = set()
    while queue:
        node, depth = queue.popleft()
        key = str(node).lower()
        if key in seen:
            continue
        seen.add(key)
        yield node, depth
        if depth >= max_depth or not node.exists() or not node.is_dir():
            continue
        try:
            children = sorted((child for child in node.iterdir() if child.is_dir()), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in children:
            queue.append((child, depth + 1))


def _tokenize_path(path: Path) -> set[str]:
    """Split a path into lowercase identifier tokens for overlap scoring.

    Tokens shorter than three characters and common, non-distinguishing words
    (such as "data" or "imec") are dropped so they do not inflate matches.
    """
    raw = str(path).replace("\\", "/").lower()
    return {
        token
        for token in re.split(r"[^a-z0-9]+", raw)
        if len(token) >= 3 and token not in {
            "spike", "sorting", "processeddata", "output", "neuropygui",
            "npx", "neuropixels", "data", "raw", "processed", "catgt",
            "imec", "test", "bin", "tmp",
        }
    }


def _candidate_bonus(candidate: Path, requested: Path, ks_tag: str | None, probe_string: str | None) -> int:
    """Return a sortable score for a candidate directory (more negative is better).

    Rewards name matches against the Kilosort tag and probe string, the presence
    of typical Kilosort sidecar files, and token overlap with the requested path.
    """
    bonus = 0
    name = candidate.name.lower()
    tag = str(ks_tag or "").strip().lower()
    probe = str(probe_string or "").strip().lower()
    if tag:
        if name == tag:
            bonus -= 50
        elif probe and name == f"imec{probe}_{tag}":
            bonus -= 48
        elif name.endswith(f"_{tag}"):
            bonus -= 44
        elif tag in name:
            bonus -= 18
    if probe and f"imec{probe}" in name:
        bonus -= 12
    if (candidate / "params.py").exists():
        bonus -= 4
    if (candidate / "metrics.csv").exists():
        bonus -= 3
    if (candidate / "cluster_group.tsv").exists():
        bonus -= 2
    bonus -= min(24, 3 * len(_tokenize_path(candidate) & _tokenize_path(requested)))
    return bonus


def find_kilosort_output_dir(
    path: str | Path,
    *,
    ks_tag: str | None = None,
    probe_string: str | None = None,
    extra_roots: Iterable[str | Path] = (),
    max_depth: int = 4,
) -> Path | None:
    """Search for the Kilosort output directory nearest to ``path``.

    Returns the requested directory directly if it already holds Kilosort
    output, otherwise scans the search roots and returns the best-scoring
    matching directory, or ``None`` if none is found.
    """
    raw = str(path).strip()
    if not raw:
        return None
    requested = Path(raw).expanduser()
    direct = _as_dir_candidate(requested)
    if has_kilosort_output(direct):
        return direct

    requested_tokens = _tokenize_path(requested)
    candidates: list[tuple[tuple[int, int, int, int, int], Path]] = []
    for root_index, root in enumerate(_iter_search_roots(requested, extra_roots)):
        if not root.exists() or not root.is_dir():
            continue
        for node, depth in _walk_dirs(root, max_depth):
            if not has_kilosort_output(node):
                continue
            overlap = _tokenize_path(node) & requested_tokens
            if not overlap:
                continue
            score = (
                root_index,
                depth,
                _candidate_bonus(node, requested, ks_tag, probe_string),
                len(node.parts),
                len(str(node)),
            )
            candidates.append((score, node))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_kilosort_output_dir(
    path: str | Path,
    *,
    ks_tag: str | None = None,
    probe_string: str | None = None,
    extra_roots: Iterable[str | Path] = (),
    max_depth: int = 4,
) -> Path:
    """Like :func:`find_kilosort_output_dir`, but always return a path.

    Falls back to the directory candidate derived from ``path`` when no
    Kilosort output directory can be found.
    """
    return find_kilosort_output_dir(
        path,
        ks_tag=ks_tag,
        probe_string=probe_string,
        extra_roots=extra_roots,
        max_depth=max_depth,
    ) or _as_dir_candidate(path)


def find_metrics_file(
    path: str | Path,
    *,
    ks_tag: str | None = None,
    probe_string: str | None = None,
    extra_roots: Iterable[str | Path] = (),
    max_depth: int = 4,
) -> Path | None:
    """Find a quality-metrics file for the Kilosort output near ``path``.

    Checks the direct directory candidate first, then the resolved Kilosort
    output directory, returning the first metrics file that exists or ``None``.
    """
    if not str(path).strip():
        return None
    direct = _as_dir_candidate(path)
    for rel in METRICS_RELATIVE_PATHS:
        candidate = direct / rel
        if candidate.exists():
            return candidate
    resolved = find_kilosort_output_dir(
        path,
        ks_tag=ks_tag,
        probe_string=probe_string,
        extra_roots=extra_roots,
        max_depth=max_depth,
    )
    if resolved is None:
        return None
    for rel in METRICS_RELATIVE_PATHS:
        candidate = resolved / rel
        if candidate.exists():
            return candidate
    return None


def next_archived_output_dir(path: str | Path, *, start_index: int = 0) -> Path:
    """Return the first unused ``<name>_<index>`` sibling for archiving ``path``."""
    target = _as_dir_candidate(path)
    index = int(start_index)
    while True:
        candidate = target.with_name(f"{target.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def archive_output_dir(path: str | Path) -> Path | None:
    """Rename ``path`` to its next archived name, returning the new path.

    Returns ``None`` without doing anything if the directory does not exist.
    """
    target = _as_dir_candidate(path)
    if not target.exists():
        return None
    archived = next_archived_output_dir(target)
    target.rename(archived)
    return archived

from __future__ import annotations

from collections import deque
from pathlib import Path
import re
from typing import Iterable


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


def has_kilosort_output(path: Path) -> bool:
    return path.exists() and path.is_dir() and all((path / name).exists() for name in KILOSORT_REQUIRED_FILES)


def _as_dir_candidate(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.exists():
        return p if p.is_dir() else p.parent
    return p if not p.suffix else p.parent


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
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
    raw = str(path).replace("\\", "/").lower()
    return {
        token
        for token in re.split(r"[^a-z0-9]+", raw)
        if len(token) >= 3 and token not in {"spike", "sorting", "processeddata", "output", "neuropygui"}
    }


def _candidate_bonus(candidate: Path, requested: Path, ks_tag: str | None, probe_string: str | None) -> int:
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
    raw = str(path).strip()
    if not raw:
        return None
    requested = Path(raw).expanduser()
    direct = _as_dir_candidate(requested)
    if has_kilosort_output(direct):
        return direct

    candidates: list[tuple[tuple[int, int, int, int, int], Path]] = []
    for root_index, root in enumerate(_iter_search_roots(requested, extra_roots)):
        if not root.exists() or not root.is_dir():
            continue
        for node, depth in _walk_dirs(root, max_depth):
            if not has_kilosort_output(node):
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
    if not str(path).strip():
        return None
    direct = _as_dir_candidate(path)
    for rel in ("metrics.csv", "bombcell/templates._bc_qMetrics.csv"):
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
    for rel in ("metrics.csv", "bombcell/templates._bc_qMetrics.csv"):
        candidate = resolved / rel
        if candidate.exists():
            return candidate
    return None


def next_archived_output_dir(path: str | Path, *, start_index: int = 0) -> Path:
    target = _as_dir_candidate(path)
    index = int(start_index)
    while True:
        candidate = target.with_name(f"{target.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def archive_output_dir(path: str | Path) -> Path | None:
    target = _as_dir_candidate(path)
    if not target.exists():
        return None
    archived = next_archived_output_dir(target)
    target.rename(archived)
    return archived

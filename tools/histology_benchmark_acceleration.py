"""Benchmark histology auto-match/auto-align acceleration on one slice."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuropyguin.histology import acceleration, alignment, atlas as hatlas, matching, slice_prep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--center-ap", type=int, default=540)
    parser.add_argument("--search-radius", type=int, default=280)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args(argv)

    at = hatlas.AllenCCFAtlas(str(args.atlas))
    img = slice_prep.load_image(args.image)
    px = slice_prep.pixel_size_um(args.image)
    scale = hatlas.CCF_VOXEL_UM / px if px and px > 0 else None
    print(f"hardware: {acceleration.hardware_summary()}")
    print(f"image: {args.image}")
    print(f"atlas_to_histology_scale: {scale if scale is not None else 'none'}")

    for workers in (1, None):
        label = "auto" if workers is None else str(workers)
        result = None
        times: list[float] = []
        for _ in range(max(1, int(args.repeat))):
            t0 = time.perf_counter()
            result = matching.automatch_coronal_ap(
                img,
                at,
                center_ap=args.center_ap,
                search_radius=args.search_radius,
                atlas_to_histology_scale=scale,
                workers=workers,
            )
            times.append(time.perf_counter() - t0)
        print(
            f"automatch workers={label}: "
            f"times={[round(t, 3) for t in times]} "
            f"ap={result['ap']} engine={result.get('engine')}"
        )

    result = matching.automatch_coronal_ap(
        img,
        at,
        center_ap=args.center_ap,
        search_radius=args.search_radius,
        atlas_to_histology_scale=scale,
    )
    sl = at.grab_atlas_slice(
        hatlas.coronal_slice_point(int(result["ap"]), at),
        hatlas.coronal_camera_vector(0, 0),
        spacing=1,
    )
    for _ in range(max(1, int(args.repeat))):
        t0 = time.perf_counter()
        T, status = alignment.auto_align_isolated(img, sl["tv_slices"], timeout=120)
        dt = time.perf_counter() - t0
        tx, ty = float(T[2, 0]), float(T[2, 1])
        print(f"autoalign: time={dt:.3f} status={status} tx={tx:.2f} ty={ty:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

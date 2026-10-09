"""Synthetic fusion benchmark: 1 camera vs 2 cameras across detector dropout rates.

Usage: python scripts/benchmark.py   (10 seeds x 200 frames per cell; seeds 100-109)
"""

from __future__ import annotations

import numpy as np

from multicam_fusion.sim import simulate


def main() -> None:
    print("| Cameras | Dropout | MOTA | MOTP (m) | ID switches / run | FN / run | FP / run |")
    print("|---|---|---|---|---|---|---|")
    for cams in (("A",), ("A", "B")):
        for dropout in (0.1, 0.3, 0.5):
            ms = [simulate(s, cams, dropout=dropout) for s in range(100, 110)]
            print(
                f"| {'+'.join(cams)} | {dropout:.1f} | {np.mean([m.mota for m in ms]):.3f} "
                f"| {np.nanmean([m.motp_m for m in ms]):.3f} | {np.mean([m.id_switches for m in ms]):.1f} "
                f"| {np.mean([m.false_negatives for m in ms]):.0f} | {np.mean([m.false_positives for m in ms]):.0f} |"
            )


if __name__ == "__main__":
    main()

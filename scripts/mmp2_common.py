"""Sequences and helpers shared by the detector study (docs/PROTOCOL_DETECTORS.md)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MMP2 = ROOT / "data" / "mmp2"
RESULTS = ROOT / "results"
STEP = 3  # every third labelled frame (about 5 frames per second)
DEV_STEPS = 600  # sampled frames used from each development sequence
SCENES = ("lobby", "office", "retail")

# tag -> (scene, sequence name inside the zips)
SEQUENCES: dict[str, tuple[str, str]] = {
    "ft_lobby": ("lobby", "lobby_1"),
    "ft_office": ("office", "office_0"),
    "ft_retail": ("retail", "retail_0"),
    "dev_lobby": ("lobby", "lobby_0"),
    "dev_office": ("office", "office_0"),
    "dev_retail": ("retail", "retail_1"),
    "test_lobby": ("lobby", "lobby_1"),
    "test_office": ("office", "office_1"),
    "test_retail": ("retail", "retail_1"),
}
DETECTORS = ("d0", "d1", "d2")

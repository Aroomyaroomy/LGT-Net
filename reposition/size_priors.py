"""Metric furniture priors from the headerless CSVs in ``scale_metrics``."""

from functools import lru_cache
from pathlib import Path
import re

import numpy as np


METRICS_DIR = Path(__file__).resolve().parents[1] / "scale_metrics"
ALIASES = {
    "sofa": "couch", "couch": "couch",
    "chair": "diningchair", "diningchair": "diningchair",
    "officechair": "officechair",
    "tv": "television", "television": "television",
    "monitor": "display", "display": "display",
    "fridge": "refrigerator", "refrigerator": "refrigerator",
    "washingmachine": "washer", "washer": "washer",
    "trashcan": "ashcan", "garbagecan": "ashcan", "ashcan": "ashcan",
    "pictureframe": "photoframe", "photoframe": "photoframe",
    "wallclock": "wallclock", "clock": "wallclock",
    "bookshelf": "bookcase", "arearug": "area_rugs", "rug": "area_rugs",
}
PROMPT_NAMES = {
    "couch": "sofa", "officechair": "office chair",
    "diningchair": "dining chair", "television": "television",
    "display": "computer monitor", "washer": "washing machine",
    "ashcan": "trash can", "photoframe": "picture frame",
    "wallclock": "wall clock", "alarmclock": "alarm clock",
    "remotecontrol": "remote control", "pendholder": "pen holder",
    "armchair": "arm chair", "loungechair": "lounge chair",
    "sectionalsofa": "sectional sofa", "coffeetable": "coffee table",
    "sidetable": "side table", "diningtable": "dining table",
    "mediaconsole": "media console", "entertainmentcenter": "entertainment center",
    "displaycabinet": "display cabinet", "filecabinet": "file cabinet",
    "floorlamp": "floor lamp", "tablelamp": "table lamp",
    "area_rugs": "area rug", "kitchenisland": "kitchen island",
    "barstool": "bar stool", "counterstool": "counter stool",
    "vanitytable": "vanity table", "wallmirror": "wall mirror",
    "floormirror": "floor mirror", "towelrack": "towel rack",
}
ROOM_CATEGORIES = {
    "living_room": (
        "sofa", "sectionalsofa", "loveseat", "armchair", "loungechair",
        "recliner", "coffeetable", "sidetable", "mediaconsole",
        "entertainmentcenter", "cabinet", "displaycabinet", "bookcase",
        "television", "floorlamp", "tablelamp", "area_rugs",
    ),
    "study_room": (
        "desk", "officechair", "bookcase", "filecabinet", "cabinet",
        "display", "printer", "tablelamp", "floorlamp", "armchair",
    ),
    "kitchen": (
        "cabinet", "kitchenisland", "diningtable", "diningchair",
        "barstool", "counterstool", "sideboard", "buffet",
        "refrigerator", "microwave",
    ),
    "bed_room": (
        "bed", "nightstand", "dresser", "cabinet", "bookcase",
        "vanitytable", "armchair", "television", "tablelamp",
        "floorlamp", "hamper",
    ),
    "bathroom": (
        "bathtub", "toilet", "cabinet", "wallmirror", "floormirror",
        "hamper", "towelrack", "washer",
    ),
}
_SPECIAL = {
    "person": ((None, None, 0), 1.0),
    "book": ((0, None, 1), 1.0),
    "laptop": ((0, 1, None), 1.0),
    "paper": ((0, None, 1), 1.0),
    "stop_sign": ((0, None, 1), 1.0),
    "diningchair": ((0, 1, 2), None),
}
_RELIABLE_AXES = {
    "bed": (0, 1), "television": (0, 2), "display": (0, 2),
    "piano": (0, 2), "door": (0, 2), "person": (2,),
}


def normalize_category(label: str | None) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "", str(label or "").lower())
    if not token:
        return None
    if (METRICS_DIR / f"{token}.csv").is_file():
        return token
    for alias in sorted(ALIASES, key=len, reverse=True):
        if alias in token:
            return ALIASES[alias]
    return None


@lru_cache(maxsize=None)
def size_prior(category: str | None) -> dict | None:
    """Return robust W,D,H medians and P10/P90 bounds in metres."""
    name = normalize_category(category)
    path = METRICS_DIR / f"{name}.csv" if name else None
    if path is None or not path.is_file():
        return None
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    indices, unit = _SPECIAL.get(name, ((0, 1, 2), 1.0))
    if unit is None:
        unit = 25.4 if np.nanmedian(raw) < 100 else 1.0
    values = np.full((len(raw), 3), np.nan)
    for dst, src in enumerate(indices):
        if src is not None:
            values[:, dst] = raw[:, src] * unit / 1000.0
    for col in range(3):
        valid = values[:, col] > 0
        logs = np.log(values[valid, col])
        if not len(logs):
            continue
        median = np.median(logs)
        mad = np.median(np.abs(logs - median))
        if mad > 0:
            rows = np.flatnonzero(valid)
            values[rows[np.abs(logs - median) > 4.0 * 1.4826 * mad], col] = np.nan
    return {
        "category": name,
        "median": np.nanmedian(values, axis=0).tolist(),
        "p10": np.nanpercentile(values, 10, axis=0).tolist(),
        "p90": np.nanpercentile(values, 90, axis=0).tolist(),
        "axes": _RELIABLE_AXES.get(name, tuple(np.flatnonzero(np.any(np.isfinite(values), axis=0)))),
        "samples": len(raw),
    }


def prompt_for_room(room_type: str) -> str:
    categories = ROOM_CATEGORIES.get(room_type)
    if categories is None:
        raise ValueError(f"Unknown room_type: {room_type}")
    return ". ".join(PROMPT_NAMES.get(name, name) for name in categories) + "."


def scale_from_prior(extents, prior: dict, limits=(0.2, 3.0)) -> tuple[float, list[float]]:
    """Uniform scale matching upright mesh X,Z,Y to prior W,D,H."""
    mesh = np.asarray(extents, dtype=float)[[0, 2, 1]]
    median = np.asarray(prior["median"], dtype=float)
    p10, p90 = np.asarray(prior["p10"]), np.asarray(prior["p90"])
    best = None
    for dims in (mesh, mesh[[1, 0, 2]]):
        valid = np.isfinite(median) & np.isfinite(dims) & (dims > 1e-6)
        valid &= np.isin(np.arange(3), prior.get("axes", (0, 1, 2)))
        factor = float(np.exp(np.median(np.log(median[valid] / dims[valid]))))
        lower = np.max(p10[valid] / dims[valid])
        upper = np.min(p90[valid] / dims[valid])
        if lower <= upper:
            factor = float(np.clip(factor, lower, upper))
        factor = float(np.clip(factor, *limits))
        score = float(np.median(np.abs(np.log(factor * dims[valid] / median[valid]))))
        candidate = score, factor, (factor * mesh).tolist()
        best = candidate if best is None or score < best[0] else best
    return best[1], best[2]

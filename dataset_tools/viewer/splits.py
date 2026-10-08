"""Scene-level train/val assignment, read from the root, never derived.

Source: ``<root>/splits.json`` (written by DRIFT's ``scripts/data/dump_split.py``;
short keys ``val``/``train``/``excluded``/``name`` also accepted), else the root's
own declaration (``manifest.held_out.scenes`` + ``keyframes_train.lst``). Scene ids
only mean something within one tree, hence `split_warning`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPLITS_FILE = "splits.json"

TRAIN = "train"
VAL = "val"
EXCLUDED = "excluded"

#: Drawing order, held out first.
SPLIT_ORDER: tuple[str, ...] = (VAL, TRAIN, EXCLUDED)

#: Tooltip per assignment, keyed by what `SceneSplits.of` returns.
SPLIT_TITLES: dict[str, str] = {
    TRAIN: "training scene: a model fits on this one",
    VAL: "validation scene: held out, a model never fits on this one",
    EXCLUDED: "in neither split: not trained on and not scored on",
}
#: Spelled-out names for the index's filter buttons.
SPLIT_NAMES: dict[str, str] = {
    TRAIN: "Training",
    VAL: "Validation",
    EXCLUDED: "Excluded",
}


@dataclass(frozen=True)
class SceneSplits:
    """A root's scene-level train/val assignment, as declared by its own file."""

    name: str | None = None
    note: str | None = None
    source: str | None = None
    #: The tree and version the ids were resolved against, for `split_warning`.
    root: str | None = None
    dataset_version: str | None = None
    val: frozenset[str] = field(default_factory=frozenset)
    train: frozenset[str] | None = None
    excluded: frozenset[str] = field(default_factory=frozenset)
    #: Per-scene counts as the export wrote them. ``{scene: count}``.
    labels: dict[str, int] = field(default_factory=dict)
    windows: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """False when nothing is declared, so a caller can skip the whole feature."""
        return bool(self.val or self.train or self.excluded)

    def of(self, scene_id: str) -> str | None:
        """``"train"``/``"val"``/``"excluded"`` for one scene, or ``None``.

        ``None`` means unassigned (an explicit train list that omits it), not
        training. Validation wins over every other list.
        """
        if not self:
            return None
        if scene_id in self.val:
            return VAL
        if scene_id in self.excluded:
            return EXCLUDED
        if self.train is None:
            return TRAIN  # a bare holdout list: everything it omits trains
        return TRAIN if scene_id in self.train else None

    def shares(self, frames: Mapping[str, int]) -> dict[str, float] | None:
        """``{side: fraction}`` of the frames in ``{scene: n_frames}``.

        Unassigned scenes count towards the total but no side.
        """
        total = sum(frames.values())
        if not total:
            return None
        sides: Counter[str] = Counter()
        for scene, n in frames.items():
            which = self.of(scene)
            if which is not None:
                sides[which] += n
        return {which: n / total for which, n in sides.items()}


#: A root that declares nothing (falsy).
NO_SPLITS = SceneSplits()


def _scene_set(data: dict, *keys: str) -> frozenset[str] | None:
    """Scene ids out of the first key that is present, or ``None`` if none is.

    Integer ids are normalised to four-digit directory names (``9`` -> ``0009``).
    """
    for key in keys:
        raw = data.get(key)
        if raw is None:
            continue
        if not isinstance(raw, list):
            logger.warning("%s: %r is not a list, ignoring it", SPLITS_FILE, key)
            continue
        out: set[str] = set()
        for item in raw:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.add(f"{item:04d}")
            elif isinstance(item, str) and item.strip():
                out.add(item.strip())
        return frozenset(out)
    return None


def _counts(raw: Any) -> dict[str, int]:
    """A ``{scene: count}`` mapping, keeping only the entries that are numbers."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for scene, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if isinstance(scene, str) and scene.strip():
            out[scene.strip()] = int(value)
    return out


def _str(data: dict, *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_splits(data: Any) -> SceneSplits:
    """Read the file's contents into a `SceneSplits`, or `NO_SPLITS`."""
    if not isinstance(data, dict):
        return NO_SPLITS
    val = _scene_set(data, "val_scenes", "val") or frozenset()
    train = _scene_set(data, "train_scenes", "train")
    excluded = _scene_set(data, "excluded_scenes", "excluded") or frozenset()
    overlap = val & (train or frozenset())
    if overlap:
        logger.warning(
            "%s: scene(s) %s are in both train and val; treating them as val",
            SPLITS_FILE,
            ", ".join(sorted(overlap)),
        )
    # Windows are exported per mode; sum the halves.
    per_mode = data.get("windows_per_scene")
    windows: Counter[str] = Counter()
    if isinstance(per_mode, dict):
        for half in per_mode.values():
            windows.update(_counts(half))
    return SceneSplits(
        name=_str(data, "split_mode", "name"),
        note=_str(data, "note", "n_folds_note"),
        source=_str(data, "source"),
        root=_str(data, "root"),
        dataset_version=_str(data, "dataset_version"),
        val=val,
        train=train,
        excluded=excluded,
        labels=_counts(data.get("labels_per_scene")),
        windows=dict(windows),
    )


TRAIN_LIST = "keyframes_train.lst"
HELDOUT_LIST = "keyframes_heldout.lst"
#: The files a declared split is read from, for a cache keyed on their state.
DECLARED_FILES: tuple[str, ...] = ("manifest.json", TRAIN_LIST, HELDOUT_LIST)


def _list_scenes(path: Path) -> frozenset[str] | None:
    """Scene ids (first path component) in a keyframe list, or None without the file."""
    try:
        text = path.read_text()
    except OSError:
        return None
    return frozenset(line.split("/", 1)[0] for line in text.splitlines() if line.strip())


def declared_splits(root_path: Path) -> SceneSplits:
    """``manifest.held_out.scenes`` as val, ``keyframes_train.lst`` as train if present.

    `NO_SPLITS` when the manifest holds no ``held_out``.
    """
    root_path = Path(root_path)
    try:
        manifest = json.loads((root_path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return NO_SPLITS
    if not isinstance(manifest, dict):
        return NO_SPLITS
    held = manifest.get("held_out")
    scenes = held.get("scenes") if isinstance(held, dict) else held if isinstance(held, list) else None
    val = _scene_set({"val": scenes}, "val") if scenes else None
    if not val:
        return NO_SPLITS
    train = _list_scenes(root_path / TRAIN_LIST)
    if train is not None:
        train = train - val
    parts = []
    if isinstance(held, dict):
        parts = [str(held[k]) for k in ("decision", "what_it_supports") if held.get(k)]
    source = "manifest.json held_out.scenes" + (f" + {TRAIN_LIST}" if train is not None else "")
    return SceneSplits(
        name="held_out",
        note=" ".join(parts) or None,
        source=source,
        root=str(root_path),
        dataset_version=str(manifest.get("dataset_version")) if manifest.get("dataset_version") else None,
        val=val,
        train=train,
        excluded=frozenset(),
    )


def read_splits(root_path: Path) -> SceneSplits:
    """``<root>/splits.json``, else `declared_splits`. A malformed file gives `NO_SPLITS`."""
    path = Path(root_path) / SPLITS_FILE
    try:
        text = path.read_text()
    except OSError:
        return declared_splits(root_path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("cannot parse %s: %s", path, exc)
        return NO_SPLITS
    return parse_splits(data)


def split_warning(
    splits: SceneSplits,
    root_path: Path,
    scene_ids: list[str] | None = None,
    manifest: dict | None = None,
) -> str | None:
    """Why this root's split should not be believed, if there is a reason.

    Scene ids only mean something inside one tree. Checks, most likely first: a
    different tree, unknown scenes, then a stale (usually harmless) version.
    """
    if not splits:
        return None
    declared = splits.root
    if declared and Path(declared) != Path(root_path):
        return (
            f"{SPLITS_FILE} was exported against {declared}, this root is {root_path} - "
            "scene ids only mean the same thing inside one tree, so the train/val marks "
            f"may be wrong; regenerate it for this root (DRIFT scripts/data/dump_split.py)"
        )
    if scene_ids is not None:
        unknown = [s for s in sorted(scene_ids) if splits.of(s) is None]
        if unknown:
            shown = ", ".join(unknown[:6]) + ("…" if len(unknown) > 6 else "")
            return (
                f"{SPLITS_FILE} says nothing about {len(unknown)} scene(s) in this root "
                f"({shown}) - they are shown unmarked rather than assumed to be training "
                "data; regenerate it for this root (DRIFT scripts/data/dump_split.py)"
            )
    theirs = splits.dataset_version
    ours = (manifest or {}).get("dataset_version")
    if theirs and ours and str(theirs) != str(ours):
        return (
            f"{SPLITS_FILE} was exported against dataset {theirs}, this root is {ours} - "
            "scene ids have survived every renumbering so far, so the marks are most "
            "likely still right, but the file is stale"
        )
    return None

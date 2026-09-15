"""Which scenes a model trains on and which are held out from it.

Scene-level only, and that is the whole point. The fold DRIFT trains under by
default is a *stride* over keyframes (``i % 4``), which cuts inside every scene:
under it no scene is a training scene and none is a validation scene, so there
is nothing a card could be coloured. Only a whole-scene split can be drawn, so
only a whole-scene split is read here. DRIFT's ``scene_strat`` is the one that
is, and its exporter refuses the other modes rather than inventing a list.

The assignment is **declared, not derived**. Which scenes to hold out is a
modelling decision -- stratified on outing and on how compressible the terrain
is -- so the viewer reads it out of the root rather than recomputing a hash, or
retyping a list, it would then have to keep in step with DRIFT. A hand-copied
duplicate is how DRIFT's own ``train.py`` came to reject the mode name it had
just gained; this module exists so that does not happen a third time. A root
with no ``splits.json`` simply has no split, and every page renders unmarked,
the same way a root with no ``atlas/`` renders without prose.

The file is written by ``scripts/data/dump_split.py`` in DRIFT, which serialises
the constant the loader itself reads::

    python3 scripts/data/dump_split.py --rootdir <root> --out <root>/splits.json

and carries, beside the two lists, the root and dataset version they were taken
from, plus per-scene label and window counts:

===================== ====================================================
``split_mode``        which fold this is (``scene_strat``)
``root``              the tree the ids were resolved against
``dataset_version``   that tree's manifest version
``val_scenes``        the held-out ids
``train_scenes``      the rest
``excluded_scenes``   in neither list (present but empty today)
``labels_per_scene``  proprio label files per scene
``windows_per_scene`` ``{"train": {...}, "val": {...}}`` after every filter
``source``            where the lists came from
===================== ====================================================

**Scene ids are only meaningful together with the root.** They are stable across
root *versions* -- v1.2 through v1.5 renumbered segments, never scenes -- but not
across *trees*: the ScaLR copy carries rear-camera scenes DRIFT's does not, and
the two disagree from ``0030`` on. A file exported against one tree therefore
mislabels the other from that scene on, silently and plausibly, which is why
`split_warning` checks the root it was written for before the colours are
believed.

Short key names (``val``, ``train``, ``excluded``, ``name``) are accepted too,
for a split written by hand.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPLITS_FILE = "splits.json"

TRAIN = "train"
VAL = "val"
EXCLUDED = "excluded"

#: Drawing order, held out first: it is the short list and the one a reader of
#: the page is looking for.
SPLIT_ORDER: tuple[str, ...] = (VAL, TRAIN, EXCLUDED)

#: What each assignment is called on the page, and the tooltip that says what it
#: means. Keyed by what `SceneSplits.of` returns; a scene it returns ``None`` for
#: gets no tag and no band at all.
SPLIT_LABELS: dict[str, str] = {
    TRAIN: "train",
    VAL: "val",
    EXCLUDED: "excluded",
}
SPLIT_TITLES: dict[str, str] = {
    TRAIN: "training scene: a model fits on this one",
    VAL: "validation scene: held out, a model never fits on this one",
    EXCLUDED: "in neither split: not trained on and not scored on",
}
#: What the index's filter buttons call each side, spelled out: the band on a
#: card has room for ``val``, the panel that explains the bands does not need
#: the abbreviation.
SPLIT_NAMES: dict[str, str] = {
    TRAIN: "Training",
    VAL: "Validation",
    EXCLUDED: "Excluded",
}

#: Which count a held-out share can be quoted in, and what to call it. The two
#: disagree substantially -- one scene is 30 % of the held-out labels but 56 % of
#: the held-out windows -- so a bare percentage would be meaningless and the
#: panel always names its unit.
SHARE_UNITS: tuple[tuple[str, str], ...] = (
    ("windows_per_scene", "windows"),
    ("labels_per_scene", "labels"),
)


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
    #: Per-scene weights, for the held-out share. ``{scene: count}``.
    labels: dict[str, int] = field(default_factory=dict)
    windows: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """False when nothing is declared, so a caller can skip the whole feature."""
        return bool(self.val or self.train or self.excluded)

    def of(self, scene_id: str) -> str | None:
        """``"train"``/``"val"``/``"excluded"`` for one scene, or ``None``.

        ``None`` means *this file does not say*, which is not the same as
        training: an explicit ``train_scenes`` list that omits a scene leaves it
        unassigned on purpose, and that is what keeps a scene the export never
        saw -- a root built later, or a different tree that numbers scenes
        differently past ``0030`` -- from being coloured as though it had been.
        Validation wins over every other list, because the one mistake that
        matters here is showing a held-out scene as trainable.
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

    def shares(self, scene_ids: Iterable[str] | None = None) -> tuple[dict[str, float], str] | None:
        """``({side: fraction}, unit)`` from the richest count the file carries.

        Windows are preferred over labels because windows are what a training
        run actually iterates; the unit travels with the numbers because the two
        answers differ by several points and by a lot for a single scene. Given
        `scene_ids`, only those scenes count, so the fractions describe the
        scenes a page shows; a scene the file does not assign still counts
        towards the total, and towards no side.
        """
        wanted = None if scene_ids is None else set(scene_ids)
        for key, unit in SHARE_UNITS:
            counts = self.windows if key == "windows_per_scene" else self.labels
            if wanted is not None:
                counts = {scene: n for scene, n in counts.items() if scene in wanted}
            total = sum(counts.values())
            if not total:
                continue
            sides: dict[str, int] = {}
            for scene, n in counts.items():
                which = self.of(scene)
                if which is not None:
                    sides[which] = sides.get(which, 0) + n
            return {which: n / total for which, n in sides.items()}, unit
        return None


#: A root that declares nothing. Shared so the common case allocates nothing and
#: ``if not splits`` is the same check everywhere.
NO_SPLITS = SceneSplits()


def _scene_set(data: dict, *keys: str) -> frozenset[str] | None:
    """Scene ids out of the first key that is present, or ``None`` if none is.

    Ids are normalised to the four-digit directory names the rest of the viewer
    uses, so a file written with integers (``[0, 9, 15]`` -- the obvious thing
    to do from a notebook) still matches ``0000``/``0009``/``0015`` instead of
    quietly selecting nothing.
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
        # Both lists claiming a scene is a contradiction in the source file, not
        # something to resolve silently: `of` holds it out, and the log says so.
        logger.warning(
            "%s: scene(s) %s are in both train and val; treating them as val",
            SPLITS_FILE,
            ", ".join(sorted(overlap)),
        )
    # The exporter splits windows by mode; only the val half is needed, and the
    # total has to be both halves added up, so flatten them back together.
    per_mode = data.get("windows_per_scene")
    windows: dict[str, int] = {}
    if isinstance(per_mode, dict):
        for half in per_mode.values():
            for scene, count in _counts(half).items():
                windows[scene] = windows.get(scene, 0) + count
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
        windows=windows,
    )


def read_splits(root_path: Path) -> SceneSplits:
    """``<root>/splits.json``, or `NO_SPLITS` when it is absent or unreadable.

    Unreadable is deliberately not fatal: a malformed split file must cost the
    index its two colours, not the whole page.
    """
    path = Path(root_path) / SPLITS_FILE
    try:
        text = path.read_text()
    except OSError:
        return NO_SPLITS
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

    Scene ids are the whole identity of the declaration and they only mean
    something inside one tree: the ScaLR copy of HELHEST carries rear-camera
    scenes DRIFT's does not and the two part company at ``0030``, so a file
    exported against one tree labels the other correctly up to there and wrongly
    after it -- a ``val`` badge on a scene a model trained on, with nothing on
    the page to suggest it. Checked in the order the mistake is likely: a
    different tree, then scenes the file has never heard of, then a stale
    version, which is the harmless one (ids survived every renumbering so far).
    """
    if not splits:
        return None
    declared = splits.root
    if declared and Path(declared) != Path(root_path):
        return (
            f"{SPLITS_FILE} was exported against {declared}, this root is {root_path} — "
            "scene ids only mean the same thing inside one tree, so the train/val marks "
            f"may be wrong; regenerate it for this root (DRIFT scripts/data/dump_split.py)"
        )
    if scene_ids is not None:
        unknown = [s for s in sorted(scene_ids) if splits.of(s) is None]
        if unknown:
            shown = ", ".join(unknown[:6]) + ("…" if len(unknown) > 6 else "")
            return (
                f"{SPLITS_FILE} says nothing about {len(unknown)} scene(s) in this root "
                f"({shown}) — they are shown unmarked rather than assumed to be training "
                "data; regenerate it for this root (DRIFT scripts/data/dump_split.py)"
            )
    theirs = splits.dataset_version
    ours = (manifest or {}).get("dataset_version")
    if theirs and ours and str(theirs) != str(ours):
        return (
            f"{SPLITS_FILE} was exported against dataset {theirs}, this root is {ours} — "
            "scene ids have survived every renumbering so far, so the marks are most "
            "likely still right, but the file is stale"
        )
    return None

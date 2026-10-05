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

**Without the file, the root's own declaration is read** (`declared_splits`).
From HELHEST v2.1 the manifest names the scenes held out of training
(``held_out.scenes``: a whole session, by decision), and from v2.3 the root
carries ``keyframes_train.lst`` and ``keyframes_heldout.lst`` beside
``keyframes.lst`` instead of one list per symlinked root. Those are the same
facts DRIFT's loader enforces (``datasets.helhest.held_out_scenes``), so the
viewer tags them the way it tags a ``scene_strat`` fold: held out as
``val``, the training list's scenes as ``train``, a scene in neither as
unassigned. ``splits.json`` still wins when present, because it names the fold.
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

#: Drawing order, held out first: it is the short list and the one a reader of
#: the page is looking for.
SPLIT_ORDER: tuple[str, ...] = (VAL, TRAIN, EXCLUDED)

#: The tooltip that says what each assignment means; the page calls it by its
#: own name. Keyed by what `SceneSplits.of` returns; a scene it returns ``None``
#: for gets no tag and no band at all.
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

    def shares(self, frames: Mapping[str, int]) -> dict[str, float] | None:
        """``{side: fraction}`` of the frames in ``{scene: n_frames}``.

        Frames, not the file's label or window counts: those exist only in an
        exported ``splits.json`` and disagree with each other by several points
        (one scene is 30 % of the held-out labels but 56 % of the held-out
        windows), while every root has frames and a declared split has nothing
        else. The scenes given are the ones a page shows; one the split does not
        assign still counts towards the total, and towards no side.
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
    """The scene ids a keyframe list names (its first path component), or None without the file."""
    try:
        text = path.read_text()
    except OSError:
        return None
    return frozenset(line.split("/", 1)[0] for line in text.splitlines() if line.strip())


def declared_splits(root_path: Path) -> SceneSplits:
    """The split a root declares itself: ``manifest.held_out.scenes`` as the held-out
    side, ``keyframes_train.lst`` as the training side when the root carries one.

    `NO_SPLITS` when the manifest holds no ``held_out``: a root that declares
    nothing is shown unmarked, as before.
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
    """``<root>/splits.json``, else what the root declares (`declared_splits`),
    else `NO_SPLITS`.

    Unreadable is deliberately not fatal: a malformed split file must cost the
    index its two colours, not the whole page.
    """
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

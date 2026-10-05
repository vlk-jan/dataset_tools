"""Reviewer notes: what a person found while looking at a scene.

A note is free text pinned to a **frame range** of one scene, with a topic
(``odometry``, ``camera``, ``event``, ``other``) so a stretch can be found again
by what is wrong with it rather than only by reading every note. The frame range
is the anchor because it is the most durable identifier a scene has: segment
indices are renumbered whenever ``filter_segments.py`` or another cutting pass
rewrites ``odometry_segments.txt``, and a note anchored to "segment 7" would
then silently point at a different stretch of driving.

Notes are stored next to the data, one file per scene::

    <root>/notes/0007.json

which is the same arrangement ``atlas/`` uses, and for the same reason: what is
in them is editorial, so it cannot be derived and has to be kept. One file per
scene keeps each file small enough to read, hand-edit and diff, and means two
scenes are never written at once.

The root is not always writable -- it is routinely a read-only or shared copy --
so the store resolves to one of three states at startup and *says* which:
notes written into the root, notes written to an explicit ``--notes-dir``, or
disabled. A read-only root with no override still **reads** its notes; only
composing is refused, with the reason shown rather than the UI silently
vanishing.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import secrets
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("dataset_tools.viewer.notes")

#: Topics offered in the UI. A file hand-edited to carry something else keeps
#: it: an unknown topic is rendered verbatim, exactly as an unknown segment
#: `reason` tag is (``windows.describe_reason``), rather than being rewritten.
TOPICS: tuple[str, ...] = ("odometry", "camera", "event", "other")
#: A reviewer's decision about the noted range, if they made one: keep it,
#: drop it from the dataset, or fix it (re-extract, re-segment). Empty when
#: the note only describes. Exported with the range by ``/api/notes.csv``.
VERDICTS: tuple[str, ...] = ("keep", "drop", "fix")

#: Caps, so one pasted log cannot make a scene's notes file unreadable.
MAX_TEXT = 4000
MAX_TOPIC = 32
MAX_AUTHOR = 64
#: Above this the file is not notes any more; refusing is better than truncating.
MAX_FILE_BYTES = 4_000_000


class NotesUnavailable(RuntimeError):
    """Notes cannot be written here (disabled, or nowhere writable to put them)."""


@dataclass
class Note:
    """One note: a frame range of one scene, a topic and what was seen there.

    ``frame_start`` and ``frame_end`` are both **inclusive** -- "frames 1204 to
    1388" means what a person writing that into the file means. This is
    deliberately not :class:`windows.Window`'s convention, whose ``frame_end``
    is exclusive, so the two never quietly stand in for one another.
    """

    id: str
    frame_start: int
    frame_end: int
    topic: str
    text: str
    author: str = ""
    created: str = ""
    updated: str | None = None
    verdict: str = ""

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start + 1

    def overlaps(self, start: int, end: int) -> bool:
        """Does this note touch the half-open frame range ``[start, end)``?

        Half-open because every caller is a :class:`windows.Window`, which is.
        """
        return self.frame_start < end and start <= self.frame_end

    @property
    def label(self) -> str:
        """How the range reads in the UI: one frame, or a span."""
        if self.frame_start == self.frame_end:
            return f"frame {self.frame_start}"
        return f"frames {self.frame_start}-{self.frame_end}"

    def to_json(self) -> dict:
        d = asdict(self)
        d["n_frames"] = self.n_frames
        d["label"] = self.label
        return d


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _clean_text(value: object) -> str:
    return str(value or "").strip()[:MAX_TEXT]


def _clean_topic(value: object) -> str:
    topic = str(value or "").strip().lower()[:MAX_TOPIC]
    return topic or "other"


def _clean_author(value: object) -> str:
    return str(value or "").strip()[:MAX_AUTHOR]


def _clean_verdict(value: object) -> str:
    verdict = str(value or "").strip().lower()
    if verdict and verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {', '.join(VERDICTS)} or empty, not {verdict!r}")
    return verdict


def _note_from_json(entry: dict) -> Note | None:
    """One stored entry, or None when it is not a note at all.

    Hand-editing this file is expected, so a single malformed entry drops out
    and the rest of the scene's notes still load.
    """
    try:
        frames = entry.get("frames")
        if isinstance(frames, list | tuple) and len(frames) == 2:
            start, end = int(frames[0]), int(frames[1])
        else:
            start = int(entry["frame_start"])
            end = int(entry["frame_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start > end:
        start, end = end, start
    text = _clean_text(entry.get("text"))
    if not text:
        return None
    return Note(
        id=str(entry.get("id") or secrets.token_hex(3)),
        frame_start=max(0, start),
        frame_end=max(0, end),
        topic=_clean_topic(entry.get("topic")),
        text=text,
        author=_clean_author(entry.get("author")),
        created=str(entry.get("created") or ""),
        updated=str(entry["updated"]) if entry.get("updated") else None,
        # Hand-edited files may say anything; an unknown verdict reads as none.
        verdict=v if (v := str(entry.get("verdict") or "").strip().lower()) in VERDICTS else "",
    )


def _note_to_json(note: Note) -> dict:
    """The stored form. ``frames`` as a pair reads as a range in the file."""
    return {
        "id": note.id,
        "frames": [note.frame_start, note.frame_end],
        "topic": note.topic,
        "text": note.text,
        "author": note.author,
        "created": note.created,
        "updated": note.updated,
        **({"verdict": note.verdict} if note.verdict else {}),
    }


def _probe_writable(directory: Path) -> bool:
    """Can this directory actually be written to?

    An actual create-and-remove, not ``os.access``: network and container
    mounts routinely report permissions they will not honour, and the point of
    this check is to tell the user *before* they type a note that it cannot be
    saved.
    """
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".dtv-probe-"):
            return True
    except OSError:
        return False


class NoteStore:
    """Where a root's notes live, and whether they can be added to.

    Resolved once, at startup: the state has to be reportable (the scene page
    says why composing is off) and probing the filesystem per request would put
    a write on the hot path of a page that is otherwise read-only.
    """

    def __init__(
        self,
        root_path: Path,
        notes_dir: Path | None = None,
        enabled: bool = True,
        author: str = "",
        dataset_version: str = "",
    ) -> None:
        self.root_path = Path(root_path)
        self.enabled = enabled
        self.author = _clean_author(author) or _default_author()
        self.dataset_version = str(dataset_version or "")
        #: True when ``--notes-dir`` named the location, i.e. the notes do NOT
        #: travel with the dataset. Worth saying out loud in the UI.
        self.external = notes_dir is not None
        self.dir = Path(notes_dir).expanduser().resolve() if notes_dir else self.root_path / "notes"
        self.writable = False
        self.reason = ""
        self._resolve()

    def _resolve(self) -> None:
        if not self.enabled:
            self.reason = "notes are disabled (--no-notes)"
            return
        # The notes directory itself is only created when a note is actually
        # written: opening a root to look at it should leave nothing behind.
        # So the probe goes to whichever of the two exists.
        target = self.dir if self.dir.is_dir() else self.dir.parent
        if self.external:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                target = self.dir
            except OSError as exc:
                self.reason = f"cannot create {self.dir}: {exc}"
                return
        if not target.is_dir():
            self.reason = f"{target} does not exist"
            return
        if _probe_writable(target):
            self.writable = True
            return
        self.reason = (
            f"{target} is not writable; pass --notes-dir PATH to keep notes "
            "somewhere else, or --no-notes to turn them off"
        )

    # -- locations ---------------------------------------------------------

    def path_for(self, scene_id: str) -> Path:
        if not scene_id.isdigit():
            raise ValueError(f"bad scene id {scene_id!r}")
        return self.dir / f"{scene_id}.json"

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "writable": self.writable,
            "dir": str(self.dir),
            "reason": self.reason,
            "author": self.author,
            "dataset_version": self.dataset_version,
            "topics": list(TOPICS),
            "verdicts": list(VERDICTS),
        }

    # -- reading -----------------------------------------------------------

    def list(self, scene_id: str) -> list[Note]:
        """Every note on a scene, in frame order. [] when there are none.

        Never raises on a bad file: a notes file someone hand-edited into
        invalid JSON degrades to "no notes" with a warning in the log, the same
        way a malformed ``atlas.json`` does.
        """
        if not self.enabled:
            return []
        try:
            path = self.path_for(scene_id)
        except ValueError:
            return []
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                logger.warning("notes file %s is implausibly large; ignoring it", path)
                return []
            with open(path, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read notes for scene %s: %s", scene_id, exc)
            return []
        entries = data.get("notes") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return []
        notes = [n for n in (_note_from_json(e) for e in entries if isinstance(e, dict)) if n]
        notes.sort(key=lambda n: (n.frame_start, n.frame_end))
        return notes

    def list_all(self) -> dict[str, list[Note]]:
        """Every scene's notes, keyed by scene id, in scene order.

        One listing of the notes directory, then :meth:`list` per file it
        names -- so a scene whose file is malformed drops out with the same
        warning it would give on its own page, and a file that is not a
        scene's (``NNNN.json`` is the only shape written) is left alone.
        ``{}`` when notes are off or nothing has been written yet.
        """
        if not self.enabled:
            return {}
        try:
            names = os.listdir(self.dir)
        except OSError:
            return {}
        out: dict[str, list[Note]] = {}
        for name in sorted(names):
            stem, ext = os.path.splitext(name)
            if ext != ".json" or not stem.isdigit():
                continue
            notes = self.list(stem)
            if notes:
                out[stem] = notes
        return out

    def stale_version(self, scene_id: str) -> str | None:
        """The dataset version a scene's notes were written against, when it is
        no longer the root's.

        Frame numbering only survives as an anchor while the extraction does.
        When it has not, the notes may point at a different stretch of driving
        than they were written about -- so the scene page says so instead of
        showing them as if nothing had moved. Same check, and the same reason,
        as the atlas's ``dataset_version`` (``index_page.atlas_warning``).
        """
        if not self.enabled or not self.dataset_version:
            return None
        try:
            path = self.path_for(scene_id)
            with open(path, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        written = str(data.get("dataset_version") or "")
        if written and written != self.dataset_version:
            return written
        return None

    # -- writing -----------------------------------------------------------

    def _require_writable(self) -> None:
        if not self.enabled or not self.writable:
            raise NotesUnavailable(self.reason or "notes cannot be written here")

    def _save(self, scene_id: str, notes: list[Note]) -> None:
        """Replace a scene's notes file atomically.

        tmp-then-``os.replace``: a note is typed once and a half-written file
        would lose every other note in the scene along with it.
        """
        path = self.path_for(scene_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene": scene_id,
            "dataset_version": self.dataset_version,
            "notes": [_note_to_json(n) for n in notes],
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def add(
        self,
        scene_id: str,
        frame_start: int,
        frame_end: int,
        text: str,
        topic: str = "other",
        author: str = "",
        n_frames: int | None = None,
        verdict: str = "",
    ) -> Note:
        """Add one note and return it as stored.

        ``n_frames`` is the scene's frame count, used to clamp the range: a
        note past the end of the scene would never be shown by anything.
        """
        self._require_writable()
        text = _clean_text(text)
        if not text:
            raise ValueError("a note needs some text")
        verdict = _clean_verdict(verdict)
        start, end = _clamp_range(frame_start, frame_end, n_frames)
        notes = self.list(scene_id)
        taken = {n.id for n in notes}
        note = Note(
            id=_fresh_id(taken),
            frame_start=start,
            frame_end=end,
            topic=_clean_topic(topic),
            text=text,
            author=_clean_author(author) or self.author,
            created=_now(),
            verdict=verdict,
        )
        notes.append(note)
        notes.sort(key=lambda n: (n.frame_start, n.frame_end))
        self._save(scene_id, notes)
        return note

    def update(
        self,
        scene_id: str,
        note_id: str,
        text: str | None = None,
        topic: str | None = None,
        frames: tuple[int, int] | None = None,
        n_frames: int | None = None,
        verdict: str | None = None,
    ) -> Note | None:
        """Edit a note in place. None when the scene has no such note.

        Only the fields passed change; ``created`` and ``author`` never do --
        an edited note is still the note that person wrote.
        """
        self._require_writable()
        notes = self.list(scene_id)
        for note in notes:
            if note.id != note_id:
                continue
            if text is not None:
                cleaned = _clean_text(text)
                if not cleaned:
                    raise ValueError("a note needs some text")
                note.text = cleaned
            if topic is not None:
                note.topic = _clean_topic(topic)
            if verdict is not None:
                note.verdict = _clean_verdict(verdict)
            if frames is not None:
                note.frame_start, note.frame_end = _clamp_range(*frames, n_frames)
            note.updated = _now()
            notes.sort(key=lambda n: (n.frame_start, n.frame_end))
            self._save(scene_id, notes)
            return note
        return None

    def delete(self, scene_id: str, note_id: str) -> bool:
        self._require_writable()
        notes = self.list(scene_id)
        kept = [n for n in notes if n.id != note_id]
        if len(kept) == len(notes):
            return False
        self._save(scene_id, kept)
        return True


def _fresh_id(taken: set[str]) -> str:
    while (candidate := secrets.token_hex(3)) in taken:
        pass
    return candidate


def _clamp_range(start: int, end: int, n_frames: int | None) -> tuple[int, int]:
    """A usable inclusive frame range: ordered, non-negative, inside the scene."""
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        raise ValueError("frame range must be two integers") from None
    if start > end:
        start, end = end, start
    start = max(0, start)
    end = max(0, end)
    if n_frames:
        start = min(start, n_frames - 1)
        end = min(end, n_frames - 1)
    return start, end


def _default_author() -> str:
    """Who a note is from when nothing said: the user running the viewer."""
    try:
        return _clean_author(getpass.getuser()) or "anonymous"
    except (OSError, KeyError):  # no login name anywhere
        return "anonymous"

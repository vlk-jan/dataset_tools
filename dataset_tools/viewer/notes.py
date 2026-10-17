"""Reviewer notes: free text pinned to a frame range of one scene, with a topic.

Frame ranges, not segment indices, because segments get renumbered. Stored one
file per scene at ``<root>/notes/NNNN.json`` (or ``--notes-dir``). A read-only
root still reads its notes; only writing is refused, with the reason shown.
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

from ..gt_io import _replace_atomically

logger = logging.getLogger("dataset_tools.viewer.notes")

#: Topics offered in the UI; an unknown topic in a file is kept verbatim.
TOPICS: tuple[str, ...] = ("odometry", "camera", "event", "other")
#: A reviewer's optional decision about the range; empty when the note only describes.
VERDICTS: tuple[str, ...] = ("keep", "drop", "fix")

#: Caps, so one pasted log cannot make a scene's notes file unreadable.
MAX_TEXT = 4000
MAX_TOPIC = 32
MAX_AUTHOR = 64
#: Above this the file is not notes any more; refusing is better than truncating.
MAX_FILE_BYTES = 4_000_000


class NotesUnavailable(RuntimeError):
    """Notes cannot be written here (disabled, or nowhere writable to put them)."""


class NotesUnreadable(OSError):
    """A notes file exists but cannot be read as notes, so a write must not replace it.

    An ``OSError`` so the app's "could not save" path reports it.
    """

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(f"notes file {path} {why}; fix it or move it aside before writing")


@dataclass
class Note:
    """One note: a frame range of one scene, a topic and what was seen there.

    ``frame_start`` and ``frame_end`` are both **inclusive**, unlike
    :class:`windows.Window`'s exclusive ``frame_end``.
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
        """Does this note touch the half-open frame range ``[start, end)``?"""
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
    """One stored entry, or None when malformed (the rest still load)."""
    try:
        start, end = _clamp_range(entry.get("frames"), None)
    except ValueError:
        return None
    text = _clean_text(entry.get("text"))
    if not text:
        return None
    return Note(
        id=str(entry.get("id") or secrets.token_hex(3)),
        frame_start=start,
        frame_end=end,
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

    A real create-and-remove: network mounts' ``os.access`` answers can lie.
    """
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".dtv-probe-"):
            return True
    except OSError:
        return False


class NoteStore:
    """Where a root's notes live, and whether they can be added to.

    Resolved once at startup, so the state is reportable and no probe write
    sits on the request path.
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
        #: True when ``--notes-dir`` is used: the notes do not travel with the dataset.
        self.external = notes_dir is not None
        self.dir = Path(notes_dir).expanduser().resolve() if notes_dir else self.root_path / "notes"
        self.writable = False
        self.reason = ""
        self._resolve()

    def _resolve(self) -> None:
        if not self.enabled:
            self.reason = "notes are disabled (--no-notes)"
            return
        # The notes dir is only created on the first write; probe whichever exists.
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

    def list(self, scene_id: str, strict: bool = False) -> list[Note]:
        """Every note on a scene, in frame order. [] when there are none.

        A malformed file logs a warning and reads as no notes. With ``strict``
        (every write path) it raises :class:`NotesUnreadable` instead, so a
        write never replaces notes it could not read; only a missing file is empty.
        """
        return self.read(scene_id, strict)[0]

    def read(self, scene_id: str, strict: bool = False) -> tuple[list[Note], str | None]:
        """:meth:`list`, plus the dataset version the notes were written against
        when it is not the root's (a re-extraction can move frame numbers), else None."""
        if not self.enabled:
            return [], None
        try:
            path = self.path_for(scene_id)
        except ValueError:
            return [], None
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise NotesUnreadable(path, "is implausibly large")
            with open(path, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return [], None
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise NotesUnreadable(path, f"is unreadable ({exc})") from exc
            logger.warning("cannot read notes for scene %s: %s", scene_id, exc)
            return [], None
        written = str(data.get("dataset_version") or "") if isinstance(data, dict) else ""
        stale = written if written and self.dataset_version and written != self.dataset_version else None
        entries = data.get("notes") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            if strict:
                raise NotesUnreadable(path, "holds no list of notes")
            return [], stale
        parsed = [_note_from_json(e) if isinstance(e, dict) else None for e in entries]
        if strict and None in parsed:
            raise NotesUnreadable(path, "has entries this viewer cannot read")
        notes = [n for n in parsed if n]
        notes.sort(key=lambda n: (n.frame_start, n.frame_end))
        return notes, stale

    def list_all(self) -> dict[str, list[Note]]:
        """Every scene's notes, keyed by scene id, in scene order.

        Only ``NNNN.json`` files are read. ``{}`` when off or empty.
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

    # -- writing -----------------------------------------------------------

    def _require_writable(self) -> None:
        if not self.enabled or not self.writable:
            raise NotesUnavailable(self.reason or "notes cannot be written here")

    def _save(self, scene_id: str, notes: list[Note]) -> None:
        """Replace a scene's notes file atomically, so a failed write never loses the other notes."""
        path = self.path_for(scene_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene": scene_id,
            "dataset_version": self.dataset_version,
            "notes": [_note_to_json(n) for n in notes],
        }
        # json.dump's default ensure_ascii: the bytes are the same in any locale encoding.
        _replace_atomically(path, lambda fh: fh.write(json.dumps(payload, indent=2) + "\n"))

    def add(
        self,
        scene_id: str,
        frames: object,
        text: str,
        topic: str = "other",
        author: str = "",
        n_frames: int | None = None,
        verdict: str = "",
    ) -> Note:
        """Add one note and return it as stored.

        ``frames`` is the inclusive ``[first, last]`` pair; ``n_frames`` (the
        scene's frame count) clamps it.
        """
        self._require_writable()
        text = _clean_text(text)
        if not text:
            raise ValueError("a note needs some text")
        verdict = _clean_verdict(verdict)
        start, end = _clamp_range(frames, n_frames)
        notes = self.list(scene_id, strict=True)
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
        frames: object = None,
        n_frames: int | None = None,
        verdict: str | None = None,
    ) -> Note | None:
        """Edit a note in place. None when the scene has no such note.

        Only the fields passed change; ``created`` and ``author`` never do.
        """
        self._require_writable()
        notes = self.list(scene_id, strict=True)
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
                note.frame_start, note.frame_end = _clamp_range(frames, n_frames)
            note.updated = _now()
            notes.sort(key=lambda n: (n.frame_start, n.frame_end))
            self._save(scene_id, notes)
            return note
        return None

    def delete(self, scene_id: str, note_id: str) -> bool:
        self._require_writable()
        notes = self.list(scene_id, strict=True)
        kept = [n for n in notes if n.id != note_id]
        if len(kept) == len(notes):
            return False
        self._save(scene_id, kept)
        return True


def _fresh_id(taken: set[str]) -> str:
    while (candidate := secrets.token_hex(3)) in taken:
        pass
    return candidate


def _clamp_range(frames: object, n_frames: int | None) -> tuple[int, int]:
    """A usable inclusive frame range from a ``[first, last]`` pair: ordered,
    non-negative, inside the scene. ValueError for anything else."""
    try:
        if not isinstance(frames, list | tuple) or len(frames) != 2:
            raise ValueError(frames)
        start, end = int(frames[0]), int(frames[1])
    except (TypeError, ValueError, OverflowError):  # json reads 1e999 as inf
        raise ValueError("a note needs a frame range: frames [first, last]") from None
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

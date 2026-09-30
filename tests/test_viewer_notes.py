"""Viewer: reviewer notes (dataset_tools.viewer.notes) and their routes.

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import json
import os

import pytest
from helpers_viewer import (
    make_root,
)

# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------


def _notes_client(tmp_path, **kwargs):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dataset_tools.viewer.app import create_app

    root_path = kwargs.pop("root_path", None) or make_root(tmp_path, frames=12)
    return root_path, TestClient(create_app(root_path, cache_dir=tmp_path / "cache", **kwargs))


def test_a_note_round_trips_through_the_api_and_the_file(tmp_path):
    root_path, client = _notes_client(tmp_path)

    created = client.post(
        "/api/scene/0000/notes",
        json={"frames": [3, 7], "topic": "odometry", "text": "GT walks off the terrain"},
    )
    assert created.status_code == 201
    note = created.json()
    assert (note["frame_start"], note["frame_end"]) == (3, 7)
    assert note["label"] == "frames 3–7"

    # Stored next to the data, one file per scene, with the range as a pair.
    stored = json.loads((root_path / "notes" / "0000.json").read_text())
    assert stored["scene"] == "0000"
    assert stored["notes"][0]["frames"] == [3, 7]
    assert stored["notes"][0]["text"] == "GT walks off the terrain"

    listed = client.get("/api/scene/0000/notes").json()
    assert [n["id"] for n in listed["notes"]] == [note["id"]]
    assert listed["status"]["writable"] is True

    edited = client.patch(f"/api/scene/0000/notes/{note['id']}", json={"text": "and never recovers"})
    assert edited.status_code == 200
    assert edited.json()["text"] == "and never recovers"
    assert edited.json()["updated"]  # an edit is dated; `created` is not moved
    assert edited.json()["created"] == note["created"]

    assert client.delete(f"/api/scene/0000/notes/{note['id']}").status_code == 200
    assert client.get("/api/scene/0000/notes").json()["notes"] == []
    assert client.delete(f"/api/scene/0000/notes/{note['id']}").status_code == 404


def test_a_notes_frame_range_is_inclusive_ordered_and_inside_the_scene(tmp_path):
    """Both ends are included, a range typed backwards still means a range, and
    nothing is stored past the end of a scene -- a note out there would never
    be shown by anything that reads it back."""
    _, client = _notes_client(tmp_path)

    one = client.post("/api/scene/0000/notes", json={"frame": 5, "text": "here"}).json()
    assert (one["frame_start"], one["frame_end"], one["n_frames"]) == (5, 5, 1)
    assert one["label"] == "frame 5"

    backwards = client.post("/api/scene/0000/notes", json={"frames": [9, 2], "text": "typed backwards"}).json()
    assert (backwards["frame_start"], backwards["frame_end"]) == (2, 9)

    past_the_end = client.post("/api/scene/0000/notes", json={"frames": [10, 900], "text": "off the end"}).json()
    assert past_the_end["frame_end"] == 11  # the scene has 12 frames

    assert client.post("/api/scene/0000/notes", json={"frames": [1, 2]}).status_code == 400
    assert client.post("/api/scene/0000/notes", json={"text": "nowhere"}).status_code == 400
    assert client.post("/api/scene/9999/notes", json={"frame": 0, "text": "x"}).status_code == 404


def test_an_edit_reaches_a_notes_range_and_topic_and_not_only_its_text(tmp_path):
    """Every field of a note is typed by hand and so is gettable wrong: a range
    pinned a segment too early, a topic picked before the window was read. All
    three move under one PATCH, and the note stays the one that person wrote --
    id, author and created date do not move with them."""
    _, client = _notes_client(tmp_path)
    note = client.post(
        "/api/scene/0000/notes",
        json={"frames": [3, 7], "topic": "camera", "text": "dropped out"},
    ).json()

    moved = client.patch(
        f"/api/scene/0000/notes/{note['id']}",
        json={"frames": [1, 4], "topic": "odometry", "text": "GT jumps sideways"},
    ).json()
    assert (moved["frame_start"], moved["frame_end"]) == (1, 4)
    assert moved["topic"] == "odometry"
    assert moved["label"] == "frames 1–4"
    assert (moved["id"], moved["author"], moved["created"]) == (
        note["id"],
        note["author"],
        note["created"],
    )

    # One end on its own moves; everything left out of the request stays put.
    one_end = client.patch(f"/api/scene/0000/notes/{note['id']}", json={"frame_end": 9}).json()
    assert (one_end["frame_start"], one_end["frame_end"]) == (1, 9)
    assert one_end["topic"] == "odometry"
    assert one_end["text"] == "GT jumps sideways"

    # An edited range is ordered and clamped exactly as a new note's is.
    fixed = client.patch(f"/api/scene/0000/notes/{note['id']}", json={"frames": [900, 2]}).json()
    assert (fixed["frame_start"], fixed["frame_end"]) == (2, 11)  # the scene has 12 frames


def test_a_note_is_marked_on_every_window_its_frames_touch():
    """A note is an inclusive frame range and a window a half-open one; a note
    spanning a boundary belongs to both windows, since opening either is a
    reason to be warned."""
    from dataset_tools.viewer.notes import Note

    note = Note(id="a", frame_start=4, frame_end=8, topic="odometry", text="bad")
    assert note.overlaps(0, 6)  # a window ending inside the note
    assert note.overlaps(8, 20)  # one starting on its last frame
    assert not note.overlaps(0, 4)  # ends exclusive at the note's first frame
    assert not note.overlaps(9, 20)
    assert note.covers(4) and note.covers(8) and not note.covers(9)


def test_the_scene_page_shows_its_notes_and_which_windows_carry_them(tmp_path):
    _, client = _notes_client(tmp_path)
    client.post(
        "/api/scene/0000/notes",
        json={"frames": [0, 2], "topic": "odometry", "text": "wrong odometry here"},
    )
    page = client.get("/scene/0000").text
    assert "wrong odometry here" in page
    assert "frames 0–2" in page
    assert "1 note" in page  # the windows table marks the window it falls in
    assert 'id="noteform"' in page


def test_the_scene_page_hands_its_editor_the_fields_as_stored(tmp_path):
    """The inline editor opens on a note's own range and topic, so both are on
    the element as stored: re-parsing them back out of the rendered label
    ("frames 3–7") would break the first time the label read differently. The
    topics the picker offers come from the card, so an editor can be opened on
    a note whether or not the compose form is on the page."""
    _, client = _notes_client(tmp_path)
    client.post(
        "/api/scene/0000/notes",
        json={"frames": [3, 7], "topic": "camera", "text": "dropped out"},
    )
    page = client.get("/scene/0000").text
    assert 'data-frame-start="3"' in page and 'data-frame-end="7"' in page
    assert 'data-topic="camera"' in page
    assert 'data-topics="odometry,camera,event,other"' in page


def test_the_view_page_hands_the_overlay_its_notes_endpoint_and_range(client):
    page = client.get("/view/0000/seg000").text
    assert '"notesUrl": "/api/scene/0000/notes"' in page
    # The window's own frames, as an inclusive pair: what the compose form
    # starts pinned to.
    assert '"frameStart": 0' in page
    assert '"frameEnd":' in page


def test_the_overlay_range_ends_are_typeable_fields(client):
    """Pinning an end to the playhead is the quick path, not the only one: a
    range read off a plot or a previous note has to be typeable."""
    page = client.get("/view/0000/seg000").text
    assert 'id="dt-note-from"' in page and 'id="dt-note-to"' in page
    assert page.count('type="number"') >= 2
    assert page.count('class="dt-nowbtn"') == 2


def test_a_read_only_root_still_reads_its_notes_but_refuses_new_ones(tmp_path):
    """The routine case: the dataset is a shared or mounted copy. Existing
    notes still show, composing is refused with the reason, and nothing about
    the rest of the viewer changes."""
    if os.geteuid() == 0:
        pytest.skip("root ignores the write bits this test relies on")
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps({"scene": "0000", "notes": [{"frames": [1, 3], "text": "from before"}]})
    )
    (root_path / "notes").chmod(0o555)
    root_path.chmod(0o555)
    try:
        _, client = _notes_client(tmp_path, root_path=root_path)
        body = client.get("/api/scene/0000/notes").json()
        assert [n["text"] for n in body["notes"]] == ["from before"]
        assert body["status"]["writable"] is False
        assert "not writable" in body["status"]["reason"]
        refused = client.post("/api/scene/0000/notes", json={"frame": 0, "text": "no"})
        assert refused.status_code == 403
        page = client.get("/scene/0000").text
        assert "from before" in page
        assert "read-only" in page.lower()
        assert 'id="noteform"' not in page
    finally:
        root_path.chmod(0o755)
        (root_path / "notes").chmod(0o755)


def test_notes_can_be_kept_outside_a_root_that_cannot_take_them(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores the write bits this test relies on")
    root_path = make_root(tmp_path, frames=12)
    root_path.chmod(0o555)
    elsewhere = tmp_path / "elsewhere"
    try:
        _, client = _notes_client(tmp_path, root_path=root_path, notes_dir=elsewhere)
        assert client.post("/api/scene/0000/notes", json={"frame": 4, "text": "kept aside"}).status_code == 201
    finally:
        root_path.chmod(0o755)
    assert json.loads((elsewhere / "0000.json").read_text())["notes"][0]["text"] == "kept aside"
    assert not (root_path / "notes").exists()


def test_notes_off_reads_writes_and_shows_nothing(tmp_path):
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps({"scene": "0000", "notes": [{"frames": [1, 3], "text": "invisible"}]})
    )
    _, client = _notes_client(tmp_path, root_path=root_path, notes=False)
    body = client.get("/api/scene/0000/notes").json()
    assert body["notes"] == [] and body["status"]["enabled"] is False
    assert client.post("/api/scene/0000/notes", json={"frame": 0, "text": "x"}).status_code == 403
    assert "invisible" not in client.get("/scene/0000").text
    assert client.get("/healthz").json()["notes"]["enabled"] is False


def test_opening_a_root_writes_no_notes_directory_into_it(tmp_path):
    """Looking at a dataset must leave it as it was found. The notes directory
    appears when a note is written, not when the viewer is pointed at a root."""
    root_path, client = _notes_client(tmp_path)
    assert client.get("/scene/0000").status_code == 200
    assert not (root_path / "notes").exists()


def test_a_malformed_notes_file_degrades_to_no_notes(tmp_path):
    """Hand-editing this file is expected; a broken one must not take a scene's
    page down with it, and one bad entry must not lose the others."""
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text("{not json at all")
    _, client = _notes_client(tmp_path, root_path=root_path)
    assert client.get("/api/scene/0000/notes").json()["notes"] == []
    assert client.get("/scene/0000").status_code == 200

    (root_path / "notes" / "0000.json").write_text(
        json.dumps(
            {
                "notes": [
                    {"frames": [1, 2], "text": "kept"},
                    {"text": "no range at all"},
                    {"frames": [3, 4]},
                    "not even an object",
                ]
            }
        )
    )
    kept = client.get("/api/scene/0000/notes").json()["notes"]
    assert [n["text"] for n in kept] == ["kept"]


def test_notes_written_against_another_dataset_version_are_flagged(tmp_path):
    """Frame numbers mean the same thing only while the extraction does. The
    same check, for the same reason, as the atlas's `dataset_version`."""
    root_path = make_root(tmp_path, frames=12)
    (root_path / "manifest.json").write_text(json.dumps({"dataset_version": "v1.3"}))
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps(
            {
                "scene": "0000",
                "dataset_version": "v1.2",
                "notes": [{"frames": [1, 3], "text": "older run"}],
            }
        )
    )
    _, client = _notes_client(tmp_path, root_path=root_path)
    assert client.get("/api/scene/0000/notes").json()["stale_version"] == "v1.2"
    assert "written against dataset v1.2" in client.get("/scene/0000").text

    # A note written now carries this root's version, and the warning goes.
    client.post("/api/scene/0000/notes", json={"frame": 5, "text": "current"})
    assert client.get("/api/scene/0000/notes").json()["stale_version"] is None


def test_an_unknown_topic_in_a_hand_edited_file_is_kept_as_written(tmp_path):
    """Unknown tokens pass through verbatim, exactly as an unrecognised segment
    `reason` does -- the file is a person's to edit."""
    root_path = make_root(tmp_path, frames=12)
    (root_path / "notes").mkdir()
    (root_path / "notes" / "0000.json").write_text(
        json.dumps({"notes": [{"frames": [1, 2], "topic": "Lidar-Dropout", "text": "x"}]})
    )
    _, client = _notes_client(tmp_path, root_path=root_path)
    assert client.get("/api/scene/0000/notes").json()["notes"][0]["topic"] == "lidar-dropout"


def test_the_notes_page_lists_every_note_in_the_root_by_topic(tmp_path):
    root_path, client = _notes_client(tmp_path, root_path=make_root(tmp_path, scenes=("0000", "0001")))
    client.post("/api/scene/0000/notes", json={"frames": [1, 3], "topic": "odometry", "text": "GT walks off"})
    client.post("/api/scene/0001/notes", json={"frames": [5, 5], "topic": "camera", "text": "dropped out"})
    # A file for a scene the root no longer has: listed and said so, not dropped.
    (root_path / "notes" / "0007.json").write_text(
        json.dumps({"notes": [{"frames": [0, 1], "topic": "event", "text": "outlived its scene"}]})
    )
    from dataset_tools.viewer.notes import NoteStore

    assert sorted(NoteStore(root_path).list_all()) == ["0000", "0001", "0007"]

    page = client.get("/notes").text
    assert "3 notes across 3 scenes" in page
    assert page.index("<h2>odometry") < page.index("<h2>camera") < page.index("<h2>event")
    assert 'href="/scene/0000/frame/1"' in page
    assert 'href="/view/0000/seg000?frame=1"' in page  # the window holding it, opened on the frame
    assert "no scene 0007 in this root" in page and 'href="/scene/0007"' not in page

    only = client.get("/notes?topic=camera")
    assert only.text.count("<li>") == 1 and "dropped out" in only.text
    data = client.get("/api/notes?topic=camera").json()
    assert data["n_notes"] == 1 and data["groups"][0]["notes"][0]["view_url"] == "/view/0001/seg000?frame=5"
    assert 'href="/notes"' in client.get("/").text
    assert 'href="/notes"' in client.get("/scene/0000").text


def test_the_notes_page_says_so_when_notes_are_off(tmp_path):
    _, client = _notes_client(tmp_path, notes=False)
    page = client.get("/notes")
    assert page.status_code == 200 and "Notes are off" in page.text
    assert client.get("/api/notes").json()["n_notes"] == 0
    assert 'href="/notes"' not in client.get("/").text

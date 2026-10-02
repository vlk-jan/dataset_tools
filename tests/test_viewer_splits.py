"""Viewer: the scene-level train/val split (dataset_tools.viewer.splits).

The split is *declared* by the root, never derived here, so most of what is
worth testing is how a declaration is read: which spellings are accepted, what a
scene the file does not mention means, and that a root with no declaration comes
out of the viewer exactly as it did before the feature existed.
"""

from __future__ import annotations

import json

import pytest
from helpers_viewer import make_root, viewer_client, write_atlas, write_splits

from dataset_tools.viewer import index_page, probe, splits

# --------------------------------------------------------------------------
# reading the declaration
# --------------------------------------------------------------------------


def test_a_root_without_a_splits_file_declares_no_split(tmp_path):
    root = probe.DatasetRoot(make_root(tmp_path), cache_dir=tmp_path / "cache")
    assert not root.splits()
    assert root.splits().of("0000") is None
    assert index_page.split_mark(root, "0000") is None


def test_a_bare_holdout_list_trains_on_every_scene_it_does_not_name(tmp_path):
    """The point of the `train`-less spelling: a scene added later is not val."""
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    write_splits(root_path, val=("0001",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    assert root.splits().of("0001") == "val"
    assert root.splits().of("0000") == "train"
    # Not in the root at all, and still not validation data.
    assert root.splits().of("9999") == "train"


def test_an_explicit_train_list_leaves_an_unlisted_scene_unassigned(tmp_path):
    """A split still being decided must read as unmarked, not as training."""
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    write_splits(root_path, val=("0001",), train=("0000",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    assert root.splits().of("0000") == "train"
    assert root.splits().of("0001") == "val"
    assert root.splits().of("0002") is None


def test_an_excluded_scene_is_neither_trained_on_nor_held_out(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    write_splits(root_path, val=("0001",), excluded=("0002",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    assert root.splits().of("0002") == "excluded"
    assert root.splits().of("0000") == "train"


def test_validation_wins_when_a_scene_is_in_two_lists(tmp_path, caplog):
    """A contradiction in the file must not be resolved towards 'trainable'."""
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_splits(root_path, val=("0000",), train=("0000", "0001"))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    with caplog.at_level("WARNING"):
        assert root.splits().of("0000") == "val"
    assert "both train and val" in caplog.text


def test_the_exporter_s_own_spelling_is_what_is_read():
    """The schema DRIFT's scripts/data/dump_split.py actually writes."""
    parsed = splits.parse_splits(
        {
            "split_mode": "scene_strat",
            "root": "/data/HELHEST_v1_5",
            "dataset_version": "1.5",
            "n_folds_note": "two-way train/val only; there is no test split",
            "val_scenes": ["0000", "0019"],
            "train_scenes": ["0001", "0002"],
            "excluded_scenes": [],
            "labels_per_scene": {"0000": 100, "0019": 300, "0001": 500, "0002": 100},
            "windows_per_scene": {
                "train": {"0001": 200, "0002": 200},
                "val": {"0000": 100, "0019": 500},
            },
            "source": "datasets/helhest.py SCENE_STRAT_VAL",
        }
    )
    assert parsed.name == "scene_strat"
    assert parsed.dataset_version == "1.5"
    assert parsed.source == "datasets/helhest.py SCENE_STRAT_VAL"
    assert parsed.note == "two-way train/val only; there is no test split"
    assert parsed.of("0019") == "val"
    assert parsed.of("0002") == "train"
    # An empty `excluded_scenes` is a present key, not a third state.
    assert parsed.excluded == frozenset()


def test_the_caveat_note_wins_over_the_fold_shape_note():
    """Every real export carries both keys; `note` is the one worth the space."""
    parsed = splits.parse_splits(
        {
            "val_scenes": ["0000"],
            "note": "Tests a new patch of a KNOWN SITE, not unseen terrain.",
            "n_folds_note": "two-way train/val only; there is no test split",
        }
    )
    assert parsed.note.startswith("Tests a new patch")

    # And a file exported before `note` existed still shows the other one.
    older = splits.parse_splits(
        {"val_scenes": ["0000"], "n_folds_note": "two-way train/val only; there is no test split"}
    )
    assert older.note == "two-way train/val only; there is no test split"


def test_a_hand_written_split_may_use_the_short_key_names():
    parsed = splits.parse_splits({"name": "by hand", "val": ["0001"], "train": ["0000"]})
    assert parsed.name == "by hand"
    assert parsed.of("0001") == "val"
    assert parsed.of("0000") == "train"


def test_scene_ids_written_as_integers_still_match_their_directories():
    """`[0, 9, 15]` is the obvious thing to write from a notebook."""
    parsed = splits.parse_splits({"val_scenes": [0, 9, 15]})
    assert parsed.val == frozenset({"0000", "0009", "0015"})
    assert parsed.of("0009") == "val"


def test_the_held_out_share_is_taken_over_frames():
    """Every root has frames; label and window counts exist only in an export."""
    parsed = splits.parse_splits(
        {
            "val_scenes": ["0000"],
            "train_scenes": ["0001"],
            "labels_per_scene": {"0000": 1, "0001": 9},
            "windows_per_scene": {"train": {"0001": 1}, "val": {"0000": 1}},
        }
    )
    # The file's own counts would say 10 % or 50 %; the frames say 25 %.
    assert parsed.shares({"0000": 100, "0001": 300}) == {"val": 0.25, "train": 0.75}
    # Limited to the scenes a page shows, the fractions are over those alone.
    assert parsed.shares({"0000": 100}) == {"val": 1.0}
    assert parsed.shares({}) is None
    assert parsed.shares({"0000": 0}) is None


def test_a_scene_the_file_does_not_assign_counts_towards_no_side():
    parsed = splits.parse_splits({"val_scenes": ["0000"], "train_scenes": ["0001"]})
    assert parsed.shares({"0000": 1, "0001": 2, "0002": 1}) == {"val": 0.25, "train": 0.5}


@pytest.mark.parametrize(
    "data",
    [
        {},  # no lists at all
        {"val_scenes": "0000"},  # a string where a list belongs
        {"val_scenes": None},
        [],  # not an object
    ],
)
def test_a_declaration_that_names_no_scene_is_no_split(data):
    assert not splits.parse_splits(data)


def test_a_malformed_splits_file_costs_the_colours_not_the_page(tmp_path, caplog):
    root_path = make_root(tmp_path)
    (root_path / "splits.json").write_text("{not json")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    with caplog.at_level("WARNING"):
        assert not root.splits()
    assert "cannot parse" in caplog.text
    # And the index still builds, unmarked.
    page = index_page.build_index(root, root.probe_all())
    assert page["split_panel"] is None
    assert page["rows"][0]["split"] is None


# --------------------------------------------------------------------------
# believing the declaration: scene ids only mean something inside one tree
# --------------------------------------------------------------------------


def test_a_split_exported_against_another_tree_is_not_believed_silently(tmp_path):
    """The ScaLR copy numbers scenes differently past 0030; a file from one tree
    labels the other right up to there and wrong after it, plausibly."""
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_splits(root_path, val=("0000",), train=("0001",), root="/somewhere/else/HELHEST")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    warning = index_page.build_index(root, root.probe_all())["split_warning"]
    assert "/somewhere/else/HELHEST" in warning
    assert "inside one tree" in warning
    # The marks are still drawn -- the banner says not to trust them, and hiding
    # them would leave nothing to explain the banner.
    assert index_page.split_mark(root, "0000")["which"] == "val"


def test_a_split_that_has_not_heard_of_a_scene_in_the_root_says_so(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    write_splits(root_path, val=("0000",), train=("0001",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    page = index_page.build_index(root, root.probe_all())
    assert "0002" in page["split_warning"]
    assert page["rows"][2]["split"] is None


def test_a_stale_version_is_flagged_as_probably_still_right(tmp_path):
    """Ids survived v1.2 through v1.5: worth saying, not worth alarming about."""
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    (root_path / "manifest.json").write_text(json.dumps({"dataset_version": "1.5"}))
    write_splits(root_path, val=("0000",), train=("0001",), dataset_version="1.3")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    warning = index_page.build_index(root, root.probe_all())["split_warning"]
    assert "1.3" in warning and "1.5" in warning
    assert "still right" in warning


def test_a_split_exported_against_this_very_root_raises_nothing(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    (root_path / "manifest.json").write_text(json.dumps({"dataset_version": "1.5"}))
    resolved = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache").path
    write_splits(root_path, val=("0000",), train=("0001",), root=str(resolved), dataset_version="1.5")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache2")

    assert index_page.build_index(root, root.probe_all())["split_warning"] is None


def test_an_unsplit_root_is_not_warned_about(tmp_path):
    root = probe.DatasetRoot(make_root(tmp_path), cache_dir=tmp_path / "cache")
    assert index_page.build_index(root, root.probe_all())["split_warning"] is None


# --------------------------------------------------------------------------
# what the pages do with it
# --------------------------------------------------------------------------


def test_an_index_row_carries_the_mark_the_rim_is_drawn_from(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_splits(root_path, val=("0001",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    rows = {r["probe"].scene_id: r for r in index_page.build_index(root, root.probe_all())["rows"]}
    assert rows["0001"]["split"]["which"] == "val"
    assert "held out" in rows["0001"]["split"]["title"]
    # The tooltip names the fold, because the badge is the only place a reader of
    # a scene page would learn which split put the scene where it is.
    assert "scene_strat" in rows["0001"]["split"]["title"]
    assert rows["0000"]["split"]["which"] == "train"


def test_the_panel_counts_the_scenes_this_root_has_not_the_file_s_lists(tmp_path):
    """A split file travels between root versions and may name absent scenes."""
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_splits(
        root_path,
        val=("0001", "7777", "8888"),
        name="scene_strat",
        note="why these.",
        # 7777 is not in this root, so it must not skew the shares.
        windows={"train": {"0000": 300}, "val": {"0001": 100, "7777": 600}},
    )
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    panel = index_page.build_index(root, root.probe_all())["split_panel"]
    assert panel["name"] == "scene_strat"
    assert panel["note"] == "why these."
    assert panel["total"] == 2
    assert {i["which"]: i["count"] for i in panel["parts"]} == {"val": 1, "train": 1}
    # Held out first: it is the shorter list and the one a reader is looking for.
    assert [i["which"] for i in panel["parts"]] == ["val", "train"]
    # Over this root's frames (12 per scene), not the file's window counts.
    assert panel["unit"] == "frames"
    assert {i["which"]: i["pct"] for i in panel["parts"]} == {"val": "50%", "train": "50%"}
    # The sizes table: one row per unit the root counts, a cell per side.
    measures = {m["label"]: m for m in panel["measures"]}
    assert measures["Scenes"]["total"] == "2"
    assert [c["value"] for c in measures["Scenes"]["cells"]] == ["1", "1"]
    assert [c["pct"] for c in measures["Windows"]["cells"]] == ["25%", "75%"]
    assert "Keyframes" not in measures  # no keyframes.lst: no row, not a row of zeros

    (root_path / "keyframes.lst").write_text(
        "0000/clouds/0.npy\n0000/clouds/1.npy\n0000/clouds/2.npy\n0001/clouds/0.npy\n"
    )
    panel = index_page.build_index(root, root.probe_all())["split_panel"]
    keyframes = next(m for m in panel["measures"] if m["label"] == "Keyframes")
    assert [c["value"] for c in keyframes["cells"]] == ["1", "3"] and keyframes["total"] == "4"


def test_an_unsplit_root_gets_no_panel_and_no_marks(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_atlas(root_path, scenes=("0000", "0001"))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    page = index_page.build_index(root, root.probe_all())
    assert page["split_panel"] is None
    assert all(row["split"] is None for row in page["rows"])


def test_the_scene_page_badge_agrees_with_the_index_rim(tmp_path):
    """One function feeds both, so a held-out scene cannot be badged 'train'."""
    build = pytest.importorskip("dataset_tools.viewer.build")
    from dataset_tools.viewer import scene_page

    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_splits(root_path, val=("0001",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    marks = {}
    for scene_id in ("0000", "0001"):
        page = scene_page.build_scene_page(root, build.SceneAccess(root, scene_id))
        row = index_page.build_row(root, root.probe(scene_id))
        assert page["split"] == row["split"]
        marks[scene_id] = page["split"]["which"]
    assert marks == {"0000": "train", "0001": "val"}


def test_the_declaration_is_read_again_only_when_it_changes(tmp_path):
    """Like the atlas: not re-parsed per view, but a re-decided split shows
    without a restart."""
    root_path = make_root(tmp_path)
    write_splits(root_path, val=("0000",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")

    first = root.splits()
    assert root.splits() is first
    (root_path / "splits.json").write_text(json.dumps({"val_scenes": ["9999"]}))
    assert root.splits().val == frozenset({"9999"})


# --------------------------------------------------------------------------
# rendered markup
# --------------------------------------------------------------------------


def _split_client(tmp_path, **kwargs):
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"), frames=8)
    write_splits(root_path, **kwargs)
    return viewer_client(root_path, tmp_path / "cache")


def test_the_rendered_index_carries_the_bands_the_panel_and_the_filter(tmp_path):
    """Rendered, not just assembled: the panel's list is reached through Jinja
    attribute lookup, where a key named `items` resolves to ``dict.items`` and
    the page dies with 'builtin_function_or_method is not iterable'."""
    client = _split_client(
        tmp_path,
        val=("0001",),
        train=("0000", "0002"),
        name="scene_strat",
        note="tests a new patch of a known site, not unseen terrain.",
        windows={"train": {"0000": 300, "0002": 500}, "val": {"0001": 200}},
    )
    html = client.get("/")
    assert html.status_code == 200
    html = html.text

    # The band on each card, and the attribute the filter reads.
    assert 'class="scene split" id="s0001" data-split="val"' in html
    assert 'class="scene split" id="s0000" data-split="train"' in html
    assert '<div class="splitband val"' in html and '<div class="splitband train"' in html
    # The panel: one button per side plus "All", each count in scenes, each
    # share with its unit spelled out.
    assert 'data-split="" aria-pressed="true">All scenes <b>3</b>' in html
    assert "Validation <b>1</b> <small>33% of frames</small>" in html
    assert "Training <b>2</b> <small>67% of frames</small>" in html
    assert 'class="splitbar"' in html
    assert "scene_strat &middot; dataset" not in html  # no version in this file
    assert "not unseen terrain" in html
    assert '<script type="module" src="/viewer-static/index.js">' in html
    assert client.get("/viewer-static/index.js").status_code == 200
    # Nothing is wrong with this root, so it must not be banner-ed.
    assert 'class="stale"' not in html


def test_the_rendered_scene_page_badges_the_scene(tmp_path):
    client = _split_client(tmp_path, val=("0001",), train=("0000", "0002"), name="scene_strat")
    for scene_id, want in (("0001", "val"), ("0000", "train")):
        page = client.get(f"/scene/{scene_id}")
        assert page.status_code == 200
        assert f'<span class="split-tag {want}"' in page.text
        assert "scene_strat" in page.text


def test_an_unsplit_root_renders_exactly_as_it_did_before(tmp_path):
    pytest.importorskip("simview")
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")

    root_path = make_root(tmp_path, scenes=("0000", "0001"), frames=8)
    html = viewer_client(root_path, tmp_path / "cache").get("/").text
    assert "splitband" not in html
    assert "data-split" not in html
    assert 'class="splitpanel"' not in html
    assert "index.js" not in html


# --------------------------------------------------------------------------
# the root's own declaration, when there is no splits.json
# --------------------------------------------------------------------------


def _declare(root_path, held=("0002",), train_list=True):
    manifest = json.loads((root_path / "manifest.json").read_text()) if (root_path / "manifest.json").exists() else {}
    manifest.update(
        {
            "dataset_version": "2.3",
            "held_out": {"scenes": list(held), "decision": "the last session is validation only"},
        }
    )
    (root_path / "manifest.json").write_text(json.dumps(manifest))
    if train_list:
        lines = [f"{s}/clouds/00000.npy {s}/images/00000.png" for s in ("0000", "0001", "0002") if s not in held]
        (root_path / "keyframes_train.lst").write_text("\n".join(lines) + "\n")
        (root_path / "keyframes_heldout.lst").write_text(
            "".join(f"{s}/clouds/00000.npy {s}/images/00000.png\n" for s in held)
        )


def test_a_manifest_hold_out_and_a_training_list_are_a_split_without_the_file(tmp_path):
    """v2.3: no splits.json, but the manifest holds a session out and the root
    carries keyframes_train.lst, which is what DRIFT's loader enforces too."""
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    _declare(root_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    s = root.splits()
    assert s and s.name == "held_out" and s.dataset_version == "2.3"
    assert s.of("0002") == "val" and s.of("0000") == "train" and s.of("0001") == "train"
    assert s.of("0009") is None  # the training list is explicit: an unlisted scene is unassigned
    assert "keyframes_train.lst" in s.source and "validation only" in s.note
    assert index_page.split_mark(root, "0002")["which"] == "val"
    assert splits.split_warning(s, root_path, ["0000", "0001", "0002"], {"dataset_version": "2.3"}) is None


def test_a_hold_out_without_a_training_list_trains_on_the_rest(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    _declare(root_path, train_list=False)
    s = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache").splits()
    assert s.of("0002") == "val" and s.of("0000") == "train" and s.of("0009") == "train"
    assert s.source == "manifest.json held_out.scenes"


def test_the_split_file_wins_over_the_declaration_and_both_refresh(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002"))
    _declare(root_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.splits().of("0002") == "val"
    write_splits(root_path, val=("0000",))
    assert root.splits().of("0000") == "val" and root.splits().of("0002") == "train"
    (root_path / "splits.json").unlink()
    assert root.splits().of("0002") == "val"  # back to the declaration
    # Re-deciding the hold-out in the manifest is picked up without a restart.
    # (The cache keys on mtime and size; a rewrite within the same clock tick
    # would not be seen, as for every root file, so the test moves the clock.)
    _declare(root_path, held=("0001",))
    import os  # noqa: E401
    import time

    later = time.time() + 5
    for name in ("manifest.json", "keyframes_train.lst", "keyframes_heldout.lst"):
        os.utime(root_path / name, (later, later))
    assert root.splits().of("0001") == "val" and root.splits().of("0002") == "train"


def test_a_root_declaring_no_hold_out_is_unmarked(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000",))
    (root_path / "keyframes_train.lst").write_text("0000/clouds/00000.npy 0000/images/00000.png\n")
    assert not probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache").splits()

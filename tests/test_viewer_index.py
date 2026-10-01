"""Viewer: the scene index and the curated atlas (dataset_tools.viewer.index_page).

Every test builds a synthetic dataset root (``helpers_viewer``), so nothing
here touches a real dataset or the network. The end-to-end HTTP tests skip
cleanly when the viewer extra (SimView + torch) is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers_viewer import (
    CURATED_THUMBS,
    make_root,
    write_atlas,
)

from dataset_tools.viewer import index_page, probe

# --------------------------------------------------------------------------
# curated atlas metadata (optional, root-level)
# --------------------------------------------------------------------------


def test_a_root_without_an_atlas_directory_reports_no_curated_metadata(tmp_path):
    root = probe.DatasetRoot(make_root(tmp_path), cache_dir=tmp_path / "cache")
    assert root.atlas() is None
    assert root.atlas_scene("0000") == {}
    assert root.atlas_thumb("0000", "a") is None


def test_atlas_thumbnails_are_looked_up_through_the_manifest_not_the_path(tmp_path):
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.atlas_thumb("0000", "a") is not None
    # A slot the atlas does not list is not a path to try -- it is simply absent.
    assert root.atlas_thumb("0000", "../../../etc/passwd") is None
    assert root.atlas_thumb("0000", "z") is None


def test_a_thumbnail_slot_resolves_to_the_frame_the_atlas_names(tmp_path):
    """The curated index wins over the fraction, and both spellings are read."""
    curated = {"thumbs": dict(CURATED_THUMBS)}
    assert index_page.thumb_slots(curated) == ["a", "b", "c"]
    assert [index_page.thumb_index(curated, s, 12) for s in ("a", "b", "c")] == [2, 7, 11]

    # The older spelling names no frame, so the fractions stand in -- which is
    # what those files were cut at anyway.
    legacy = {"thumbs": ["a", "b", "c"]}
    assert index_page.thumb_slots(legacy) == ["a", "b", "c"]
    assert [index_page.thumb_index(legacy, s, 12) for s in ("a", "b", "c")] == [1, 6, 10]

    # A slot the atlas skipped, and an index pointing past a re-cut scene.
    assert index_page.thumb_slots({"thumbs": {"b": 3}}) == ["b"]
    assert index_page.thumb_index({}, "a", 12) == 1
    assert index_page.thumb_index({"thumbs": {"a": 900}}, "a", 12) == 11
    assert index_page.thumb_index({"thumbs": {"a": 5}}, "a", 0) == 0


def test_an_atlas_from_another_release_says_so_on_the_index(tmp_path):
    """The v1.3 failure: atlas/ symlinked to v1.2's, so every curated number on
    the page described the previous release under the previous release's title."""
    root_path = make_root(tmp_path)
    (root_path / "manifest.json").write_text(json.dumps({"dataset_version": "1.3"}))
    write_atlas(root_path, dataset_version="1.2")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    assert "1.2" in page["atlas_warning"] and "1.3" in page["atlas_warning"]

    # Regenerated against this root: nothing to say.
    write_atlas(root_path, dataset_version="1.3")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache2")
    assert index_page.build_index(root, root.probe_all())["atlas_warning"] is None


def test_a_malformed_atlas_file_degrades_to_no_atlas(tmp_path):
    root_path = make_root(tmp_path)
    (root_path / "atlas").mkdir()
    (root_path / "atlas" / "atlas.json").write_text("{not json")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.atlas() is None


def test_probe_records_segment_spans_as_fractions_and_marks_filled_ones(tmp_path):
    root_path = make_root(tmp_path)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write("# id start_ts end_ts n_poses reason\n")
        fh.write(f"0 {ts[0]:.6f} {ts[3]:.6f} 4 filtered:pose_jump\n")
        fh.write(f"1 {ts[6]:.6f} {ts[9]:.6f} 4 v11_fill\n")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000", refresh=True)
    assert len(p.segment_spans) == 2
    assert p.fill_spans == [p.segment_spans[1]]
    lo, hi = p.segment_spans[0]
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(3 / 11, abs=1e-3)  # frames 0..3 of 0..11
    assert all(0.0 <= v <= 1.0 for span in p.segment_spans for v in span)


def test_a_stale_probe_cache_from_an_older_format_is_discarded(tmp_path):
    root_path = make_root(tmp_path)
    cache = tmp_path / "cache"
    root = probe.DatasetRoot(root_path, cache_dir=cache)
    root.probe("0000")
    stale = json.loads((cache / "probe_cache.json").read_text())
    stale["format"] = 0
    for entry in stale["scenes"].values():
        entry["segment_spans"] = []
    (cache / "probe_cache.json").write_text(json.dumps(stale))
    again = probe.DatasetRoot(root_path, cache_dir=cache)
    assert again.probe("0000").segment_spans, "an old cache must be re-probed, not trusted"


def test_index_rows_use_curated_content_when_it_exists(tmp_path):
    from dataset_tools.viewer import index_page

    root_path = make_root(tmp_path)
    write_atlas(root_path)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    assert [g["title"] for g in page["groups"]] == ["An outing"]
    row = page["rows"][0]
    assert row["description"].startswith("A synthetic place")
    labels = [s["label"] for s in row["stats"]]
    assert "path in GT" in labels and "time with GT" in labels
    # Regeneration diagnostics stay off the index (they live on the scene page).
    assert "2 s err med / p95" not in labels and "v1 coverage" not in labels
    assert [t["url"] for t in row["thumbs"]] == [f"/api/scene/0000/thumb/{s}.jpg" for s in ("a", "b", "c")]
    assert page["totals"][0]["label"] == "scenes"


def test_index_rows_fall_back_to_probed_content_without_an_atlas(tmp_path):
    from dataset_tools.viewer import index_page

    root = probe.DatasetRoot(make_root(tmp_path), cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    row = page["rows"][0]
    assert row["description"] is None
    labels = [s["label"] for s in row["stats"]]
    assert "duration" in labels and "frames" in labels
    # Nothing curated may leak into the fallback.
    assert "2 s err med / p95" not in labels
    # The timeline still exists: it comes from the segment file, not the atlas.
    assert row["spans"]
    # Thumbnails point at the same endpoint; it will render live frames.
    assert row["thumbs"][0]["url"] == "/api/scene/0000/thumb/a.jpg"
    assert page["groups"][0]["title"] == "Scenes"


def test_a_scene_missing_from_the_atlas_is_still_listed(tmp_path):
    from dataset_tools.viewer import index_page

    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    write_atlas(root_path, scenes=("0000",))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    listed = {r["probe"].scene_id for g in page["groups"] for r in g["rows"]}
    assert listed == {"0000", "0001"}
    assert page["groups"][-1]["title"] == "Not in the atlas"


def test_duration_formatting_matches_the_atlas_phrasing():
    from dataset_tools.viewer.index_page import coverage_class, fmt_duration

    assert fmt_duration(39.4) == "39 s"
    assert fmt_duration(653.6) == "10 min 53 s"
    assert fmt_duration(3780) == "1 h 3 min"
    assert fmt_duration(None) == "-"
    assert coverage_class(0.9) == "hi"
    assert coverage_class(0.6) == "mid"
    assert coverage_class(0.42) == "lo"
    assert coverage_class(None) == "none"


# --------------------------------------------------------------------------
# the live segment file, not a stale atlas, decides the timeline
# --------------------------------------------------------------------------


def test_the_atlas_cannot_add_filled_segments_the_live_file_does_not_have(tmp_path):
    """The segment file is the truth. "No filled segments" is an answer, not a
    hole to fall through to the atlas with -- a scene re-filtered since the
    atlas was written used to keep claiming amber v1.1 stretches, on the index
    card, on the scene timeline and in the card's prose."""
    root_path = make_root(tmp_path)
    write_atlas(root_path)
    atlas_file = root_path / "atlas" / "atlas.json"
    data = json.loads(atlas_file.read_text())
    data["scenes"]["0000"]["seg_spans"] = [[0.0, 0.4]]
    data["scenes"]["0000"]["fill_spans"] = [[0.5, 0.9]]
    atlas_file.write_text(json.dumps(data))

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000")
    assert p.segment_spans and not p.fill_spans  # the live file has no fills

    from dataset_tools.viewer.build import SceneAccess
    from dataset_tools.viewer.index_page import build_row
    from dataset_tools.viewer.scene_page import ground_truth_card

    row = build_row(root, p)
    assert row["fill_spans"] == []
    assert row["spans"] == p.segment_spans

    card = ground_truth_card(SceneAccess(root, "0000"), root.atlas_scene("0000"))
    assert card["timeline"]["fill"] == []
    assert "filled from v1.1" not in (card["note"] or "")


def test_a_filled_span_is_drawn_once_not_twice(tmp_path):
    """`fill_spans` is a subset of `segment_spans`, so the plain half has to
    have it removed or a filled stretch gets a green bar under its amber one."""
    root_path = make_root(tmp_path, segments=False)
    scene = root_path / "0000"
    ts = np.loadtxt(scene / "data_ts.txt")
    with open(scene / "odometry_segments.txt", "w") as fh:
        fh.write(f"0 {ts[0]:.6f} {ts[4]:.6f} 40 filtered:pose_jump\n")
        fh.write(f"1 {ts[6]:.6f} {ts[-1]:.6f} 50 v11_fill:end_of_run\n")

    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    p = root.probe("0000")
    assert len(p.segment_spans) == 2 and len(p.fill_spans) == 1

    from dataset_tools.viewer.index_page import build_row

    row = build_row(root, p)
    assert len(row["spans"]) == 1 and len(row["fill_spans"]) == 1
    assert row["spans"][0] not in row["fill_spans"]


# --------------------------------------------------------------------------
# disk usage (optional, root-level, from helhest-disk-usage)
# --------------------------------------------------------------------------


def test_disk_usage_is_summed_over_the_scenes_the_index_shows(tmp_path):
    root_path = make_root(tmp_path, scenes=("0000", "0001"))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert index_page.build_index(root, root.probe_all())["disk_panel"] is None

    # 0001 was never scanned, and 9999 is a scene this root does not have.
    mods = {"images": {"bytes": 900, "files": 3}, "lidar_raw": {"bytes": 60, "files": 1}, "other": {"bytes": 40}}
    scenes = {
        "0000": {"bytes": 1000, "span_s": 3.6, "modalities": mods},
        "9999": {"bytes": 10**12, "span_s": 1.0, "modalities": {"images": {"bytes": 10**12}}},
    }
    (root_path / "disk_usage.json").write_text(json.dumps({"scenes": scenes}))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())

    dp = page["disk_panel"]
    assert dp["missing"] == ["0001"] and dp["total"] == "1.0 kB"
    # An unknown directory folds into "other" instead of taking a new colour.
    assert [(p["key"], p["pct"]) for p in dp["parts"]] == [("images", "90%"), ("other", "10%")]
    assert page["totals"][-1]["label"] == "on disk"

    cards = {r["probe"].scene_id: r["stats"] for r in page["rows"]}
    assert cards["0000"][-1] == {"label": "on disk", "value": "1.0 kB", "sub": "0 GB/h"}
    assert all(s["label"] != "on disk" for s in cards["0001"])


VALIDATION_CSV = (
    "scene,status,frames,duration_s,points_mean,cloud_dtype,nan_frac,max_step_m,max_speed_mps,"
    "path_len_m,tilt_mean_deg,tilt_max_deg,max_sync_offset_s,errors,warnings\n"
    "0000,OK,12,1.100,64,float32,0.00e+00,0.301,3.0,3.3,0.0,0.0,0.005,0,0\n"
    "0001,WARN,12,1.100,64,float32,0.00e+00,1.842,18.4,3.3,0.0,0.0,0.005,0,2\n"
    "0002,FAIL,12,1.100,64,float32,0.00e+00,0.301,3.0,3.3,0.0,0.0,0.005,1,0\n"
    "0009,OK,12,1.100,64,float32,0.00e+00,0.301,3.0,3.3,0.0,0.0,0.005,0,0\n"
)


def test_the_validation_report_marks_each_scene_it_names(tmp_path):
    """validation.csv at the root gives every scene it names a chip and the
    scene page a card; a scene it does not name, and a root without the file,
    get nothing. The file carries counts, so the card says the messages are
    not in it."""
    from dataset_tools.viewer.build import SceneAccess
    from dataset_tools.viewer.scene_page import build_scene_page

    root_path = make_root(tmp_path, scenes=("0000", "0001", "0002", "0003"))
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    assert all(r["validation"] is None for r in page["rows"])
    assert not any(t["label"] == "validated" for t in page["totals"])
    assert build_scene_page(root, SceneAccess(root, "0000"))["cards"][-1]["title"] != "Validation"

    (root_path / "validation.csv").write_text(VALIDATION_CSV)
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    page = index_page.build_index(root, root.probe_all())
    marks = {r["probe"].scene_id: r["validation"] for r in page["rows"]}
    assert (marks["0000"]["cls"], marks["0000"]["label"]) == ("yes", "validated ok")
    assert (marks["0001"]["cls"], marks["0001"]["label"]) == ("partial", "2 warnings")
    assert (marks["0002"]["cls"], marks["0002"]["label"]) == ("no", "1 error")
    assert marks["0003"] is None
    assert "max_step_m 1.842" in marks["0001"]["title"]
    total = next(t for t in page["totals"] if t["label"] == "validated")
    # Counted over the scenes the index shows: 0009 is in the file, not the root.
    assert total["value"] == "1 ok" and total["sub"] == "1 warning, 1 error; 1 not in the report"

    card = build_scene_page(root, SceneAccess(root, "0001"))["cards"][-1]
    assert card["title"] == "Validation" and card["badge"] == "WARN"
    assert next(s["value"] for s in card["stats"] if s["label"] == "warnings") == "2"
    assert next(s["value"] for s in card["stats"] if s["label"] == "max step") == "1.842"
    assert "run helhest-validate 0001" in card["note"]
    assert build_scene_page(root, SceneAccess(root, "0003"))["cards"][-1]["title"] != "Validation"


def test_a_malformed_validation_report_degrades_to_none(tmp_path):
    root_path = make_root(tmp_path)
    (root_path / "validation.csv").write_text("not,a,report\n")
    root = probe.DatasetRoot(root_path, cache_dir=tmp_path / "cache")
    assert root.validation() == {}
    assert index_page.build_row(root, root.probe("0000"))["validation"] is None


def test_disk_sizes_read_in_decimal_units():
    assert index_page.fmt_bytes(80_154_887_377) == "80.2 GB"
    assert index_page.fmt_bytes(454_081_342_044_000) == "454.08 TB"
    assert index_page.fmt_bytes(3_625_000) == "3.6 MB" and index_page.fmt_bytes(326_000) == "326 kB"
    assert index_page.fmt_share(0.00025) == "<0.1%" and index_page.fmt_share(0.074) == "7.4%"

"""Viewer: a root served over HTTP, mirrored on demand (dataset_tools.viewer.remote)."""

from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from helpers_viewer import make_root, viewer_client

from dataset_tools.viewer import probe, remote

APACHE_ROWS = """
<tr><td valign="top"><img src="/icons/back.gif" alt="[PARENTDIR]"></td><td><a href="/">Parent Directory</a></td><td>&nbsp;</td><td align="right">  - </td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/icons/folder.gif" alt="[DIR]"></td><td><a href="clouds/">clouds/</a></td><td align="right">2026-10-19 12:34  </td><td align="right">  - </td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/icons/text.gif" alt="[TXT]"></td><td><a href="data_ts.txt">data_ts.txt</a></td><td align="right">2026-10-19 12:34  </td><td align="right"> 25K</td><td>&nbsp;</td></tr>
<tr><td><a href="../">..</a></td></tr><tr><td><a href="?C=N;O=D">Name</a></td></tr><tr><td><a href="..%2Fetc">x</a></td></tr>
"""


def test_an_apache_index_lists_entries_and_nothing_that_leaves_the_directory():
    assert remote.parse_listing(APACHE_ROWS) == {
        "clouds": [True, "2026-10-19 12:34 -"],
        "data_ts.txt": [False, "2026-10-19 12:34 25K"],
    }


@pytest.fixture
def served(tmp_path):
    """``make_root`` behind a local HTTP server; yields (root, url)."""
    root = make_root(tmp_path, scenes=("0000", "0001"))
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.func.log_message = lambda *a: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    remote._mount = None


def test_a_url_root_probes_and_builds_like_the_local_one(served, tmp_path):
    root, url = served
    mirror = remote.mount(url, tmp_path / "cache")
    # Small files come with the sync; per-frame ones wait for a read.
    assert (mirror / "0000").is_dir() and not (mirror / "0000" / "data_ts.txt").exists()
    remote_root = probe.DatasetRoot(mirror, cache_dir=tmp_path / "cache")
    local_root = probe.DatasetRoot(root, cache_dir=tmp_path / "local-cache")
    assert remote_root.scene_ids() == local_root.scene_ids() == ["0000", "0001"]
    got, want = remote_root.probe("0000"), local_root.probe("0000")
    assert (got.n_frames, got.counts, got.present) == (want.n_frames, want.counts, want.present)
    # Probed from the arrays' headers alone, by range reads.
    assert got.counts["transforms"] == 12 and (mirror / "0000" / "transforms.npy").stat().st_size == 0
    cloud = mirror / "0000" / "clouds" / "00000.npy"
    assert cloud.stat().st_size == 0

    fingerprint = probe.scene_fingerprint(mirror / "0000")
    client = viewer_client(mirror, tmp_path / "cache")
    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    assert client.get("/api/scene/0000/frame/0/image.jpg").status_code == 200
    assert cloud.read_bytes() == (root / "0000" / "clouds" / "00000.npy").read_bytes()
    # Fetching frames must not invalidate the scene's probe or built windows.
    assert probe.scene_fingerprint(mirror / "0000") == fingerprint


def test_a_mirror_refuses_a_foreign_directory_an_oversized_body_and_an_off_host_redirect(served, tmp_path, monkeypatch):
    root, url = served
    foreign = tmp_path / "theirs" / "mirror"
    foreign.mkdir(parents=True)
    (foreign / "keep.txt").write_text("not the viewer's")
    with pytest.raises(OSError, match="did not mirror"):
        remote.mount(url, tmp_path / "theirs", clear=True)
    assert (foreign / "keep.txt").exists()

    remote.mount(url, tmp_path / "cache")
    monkeypatch.setattr(remote, "MAX_BYTES", 100)
    with pytest.raises(OSError, match="over"):
        remote._get(url + "0000/data_ts.txt")

    class Away(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://localhost:9/elsewhere")
            self.end_headers()

    away = ThreadingHTTPServer(("127.0.0.1", 0), Away)
    threading.Thread(target=away.serve_forever, daemon=True).start()
    try:
        with pytest.raises(OSError, match="redirect off"):
            remote._get(f"http://127.0.0.1:{away.server_port}/x")
    finally:
        away.shutdown()

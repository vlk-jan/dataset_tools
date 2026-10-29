"""Viewer: a root served over HTTP, mirrored on demand (dataset_tools.viewer.remote)."""

from __future__ import annotations

import functools
import html
import io
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

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
    assert got.counts["transforms"] == 12 and not (mirror / "0000" / "transforms.npy").exists()
    # Listed, not written: no placeholder files, yet present to the viewer.
    cloud = mirror / "0000" / "clouds" / "00000.npy"
    assert not cloud.exists() and remote.is_file(cloud) and "00000.npy" in remote.listdir(cloud.parent)
    assert not remote.is_file(cloud.with_name("99999.npy"))

    fingerprint = probe.scene_fingerprint(mirror / "0000")
    client = viewer_client(mirror, tmp_path / "cache")
    opts = "40-160000-80-500000-0.1-height"
    meta = client.get(f"/api/data/0000/seg000/{opts}/x/meta.json")
    assert meta.status_code == 200, meta.text
    assert client.get("/api/scene/0000/frame/0/image.jpg").status_code == 200
    assert cloud.read_bytes() == (root / "0000" / "clouds" / "00000.npy").read_bytes()
    cloud.write_bytes(b"")  # the empty placeholder an older mirror left: still fetched
    assert remote.local(cloud).read_bytes() == (root / "0000" / "clouds" / "00000.npy").read_bytes()
    # Fetching frames must not invalidate the scene's probe or built windows.
    assert probe.scene_fingerprint(mirror / "0000") == fingerprint


def test_the_parsed_listings_kept_are_bounded(served, tmp_path, monkeypatch):
    root, url = served
    mirror = remote.mount(url, tmp_path / "cache")
    probe.DatasetRoot(mirror, cache_dir=tmp_path / "cache").probe_all()  # syncs the scenes
    monkeypatch.setattr(remote, "LISTINGS_KEPT", 1)
    for scene in ("0000", "0001", "0000"):
        assert sorted(remote.listdir(mirror / scene / "clouds")) == sorted(os.listdir(root / scene / "clouds"))
    assert list(remote._listings) == [mirror / "0000" / "clouds"]


def _apache(root, listed: list[str]):
    """A server over ``root`` whose directory pages carry an Apache index's date and
    size columns; the paths of the pages it serves go into ``listed``."""

    class Apache(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def list_directory(self, path):
            listed.append(self.path)
            rows = []
            for name in sorted(os.listdir(path)):
                st = os.stat(os.path.join(path, name))
                slash = "/" if os.path.isdir(os.path.join(path, name)) else ""
                date = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
                rows.append(
                    f'<tr><td><a href="{quote(name)}{slash}">{html.escape(name)}{slash}</a></td>'
                    f'<td align="right">{date}  </td><td align="right">{"-" if slash else st.st_size}</td></tr>'
                )
            body = ("<table>" + "\n".join(rows) + "</table>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return io.BytesIO(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Apache, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_published_listings_stand_in_for_the_per_directory_index_pages(tmp_path):
    root = make_root(tmp_path, scenes=("0000", "0001"))
    past = time.time() - 600
    for d in root.glob("0*/*/"):
        os.utime(d, (past, past))  # extracted well before the listings are written
    assert remote.write_listings(root) == sum(len(os.listdir(d)) for d in root.glob("0*/*/"))
    listed: list[str] = []
    server = _apache(root, listed)
    try:
        mirror = remote.mount(f"http://127.0.0.1:{server.server_port}/", tmp_path / "cache")
        got = probe.DatasetRoot(mirror, cache_dir=tmp_path / "cache").probe("0000")
        want = probe.DatasetRoot(root, cache_dir=tmp_path / "local-cache").probe("0000")
        assert (got.n_frames, got.counts, got.present) == (want.n_frames, want.counts, want.present)
        assert not [p for p in listed if p.startswith("/0000/") and p != "/0000/"]
        cloud = mirror / "0000" / "clouds" / "00000.npy"
        assert remote.local(cloud).read_bytes() == (root / "0000" / "clouds" / "00000.npy").read_bytes()

        # A directory written after the listings is listed by the server instead.
        future = time.time() + 600
        os.utime(root / "0001" / "clouds", (future, future))
        assert probe.DatasetRoot(mirror, cache_dir=tmp_path / "cache").probe("0001").counts["clouds"] == 12
        assert "/0001/clouds/" in listed and "/0001/images/" not in listed
    finally:
        server.shutdown()
        remote._mount = None


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

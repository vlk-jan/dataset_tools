import json

import numpy as np

from dataset_tools import keyframes
from dataset_tools.transforms import euler_matrix


def _scene(root, sc, n, tilted=(), no_map=()):
    d = root / sc
    for sub in ("clouds", "images", "height_maps", "height_masks"):
        (d / sub).mkdir(parents=True)
    for i in range(n):
        (d / "clouds" / f"{i:05d}.npy").touch()
        (d / "images" / f"{i:05d}.jpg").touch()
        if i not in no_map:
            (d / "height_maps" / f"{i:05d}.npz").touch()
            (d / "height_masks" / f"{i:05d}.npz").touch()
    g = np.stack([euler_matrix(np.radians(90) if i in tilted else 0, 0, 0)[:3, :3] for i in range(n)])
    np.save(d / "gravity_align.npy", g)
    np.savetxt(d / "data_ts.txt", np.arange(n) * 0.5)


def test_lists_split_by_the_manifest_and_gate_tilt_ahead(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"images": {"extension": "jpg"}, "held_out": {"scenes": ["0001"]}})
    )
    _scene(tmp_path, "0000", 20, tilted={15}, no_map={3})
    _scene(tmp_path, "0001", 5)
    r = keyframes.write_lists(tmp_path)
    assert r["lists"] == {"keyframes.lst": 15, "keyframes_train.lst": 10, "keyframes_heldout.lst": 5}
    # 15 is tilted, and the 4 s before it (frames 7-14 at 2 Hz) see it coming
    assert r["scenes"]["0000"] == {"listed": 10, "no_map_mask_or_image": 1, "tilted": 9}
    lines = (tmp_path / "keyframes_train.lst").read_text().split("\n")[:-1]
    stems = [int(ln[12:17]) for ln in lines]
    assert 3 not in stems and not set(range(7, 16)) & set(stems) and 16 in stems
    assert lines[0] == "0000/clouds/00000.npy 0000/images/00000.jpg"
    held = (tmp_path / "keyframes_heldout.lst").read_text()
    assert held.count("\n") == 5 and "0000/" not in held

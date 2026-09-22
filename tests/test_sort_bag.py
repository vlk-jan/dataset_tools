"""helhest-sort-bag: a sorted copy, and no output directory lost by accident."""

from pathlib import Path

from rosbags.highlevel import AnyReader
from test_extract_bag import write_bag

from dataset_tools import sort_bag


def _stamps(bag: Path) -> list[int]:
    with AnyReader([bag]) as reader:
        return [t for _, t, _ in reader.messages()]


def test_the_copy_is_in_timestamp_order_and_complete(tmp_path):
    src = write_bag(tmp_path / "src", frames=6)
    out = tmp_path / "sorted"
    n = sort_bag.sort_bag(src, out)
    stamps = _stamps(out)
    assert n == len(stamps) == len(_stamps(src))
    assert stamps == sorted(stamps)


def test_an_existing_output_is_refused_unless_asked_to_overwrite(tmp_path, capsys):
    src = write_bag(tmp_path / "src", frames=3)
    out = tmp_path / "sorted"
    out.mkdir()
    keep = out / "precious.txt"
    keep.write_text("mine")

    assert sort_bag.main([str(src), str(out)]) == 2
    assert keep.read_text() == "mine"
    assert "--overwrite" in capsys.readouterr().err

    assert sort_bag.main([str(src), str(out), "--overwrite"]) == 0
    assert not keep.exists()
    assert (out / "metadata.yaml").is_file()

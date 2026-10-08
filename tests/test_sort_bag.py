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


def test_header_order_unset_stamps_and_qos_survive(tmp_path):
    """Sorted by header stamp; an unset (boot-clock) header keeps its receive time
    rather than becoming a stamp of 0.12 s; /tf_static stays transient_local."""
    from rosbags.interfaces import Qos, QosDurability, QosHistory, QosLiveliness, QosReliability, QosTime
    from rosbags.rosbag2 import Writer
    from test_extract_bag import T0, TS, _imu_msg, _tf_static_msg

    latched = Qos(
        QosHistory.KEEP_LAST, 1, QosReliability.RELIABLE, QosDurability.TRANSIENT_LOCAL,
        QosTime(0, 0), QosTime(0, 0), QosLiveliness.AUTOMATIC, QosTime(0, 0), False,
    )  # fmt: skip
    src = tmp_path / "src"
    with Writer(src, version=8) as w:
        imu = w.add_connection("/imu", "sensor_msgs/msg/Imu", typestore=TS)
        tf = w.add_connection("/tf_static", "tf2_msgs/msg/TFMessage", typestore=TS, offered_qos_profiles=[latched])
        w.write(tf, T0 * 10**9, TS.serialize_cdr(_tf_static_msg(), tf.msgtype))
        # (receive s, header s): out of order, and one header left on the boot clock
        for recv, stamp in ((T0 + 0.0, T0 + 1.0), (T0 + 0.1, T0 + 0.5), (T0 + 0.2, 0.12)):
            w.write(imu, int(recv * 10**9), TS.serialize_cdr(_imu_msg(stamp, 0.0), imu.msgtype))

    out = tmp_path / "sorted"
    sort_bag.sort_bag(src, out)
    with AnyReader([out]) as reader:
        stamps = [t for c, t, _ in reader.messages() if c.topic == "/imu"]
        qos = next(c for c in reader.connections if c.topic == "/tf_static").ext.offered_qos_profiles
    assert stamps == [int((T0 + 0.2) * 10**9), int((T0 + 0.5) * 10**9), int((T0 + 1.0) * 10**9)]
    assert qos[0].durability == QosDurability.TRANSIENT_LOCAL

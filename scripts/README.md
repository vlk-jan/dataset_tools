# Scripts

Command-line tools that work on a dataset root or a bag but are not part of the
installed package. They import the installed `dataset_tools`, so run them with
the virtualenv the checkout is installed into (`pip install -e .`):

```bash
.venv/bin/python scripts/make_atlas.py /mnt/personal/$USER/HELHEST_v2_4 --bags docs/v2_4_bags.csv --check
```

## Tools for any root

| Script | What it does |
| --- | --- |
| `make_atlas.py` | Generate `<root>/atlas/atlas.json` and its thumbnails from the root itself, merging the hand-written prose in `atlas/curation.json`; `--check` reports a stale atlas without writing. `--bags` names the bag list the sessions and bag sizes come from (`docs/v2_<n>_bags.csv`, as `slurm/finalize_v2.sh` does); without it those two fields are left out. There is no default because every list numbers its scenes from 0000. The viewer's index page renders it. |
| `measure_time_offset.py` | Audit an extracted scene's frame clock against its poses: was `--time_align` applied, is the scene still off, and by how much over its course. |
| `probe_clock_skew.py` | Every publisher's clock against the recorder's, straight out of a bag (`recv - header` per message), to see *when* a sensor's clock moved. |

## Viewer launchers

| Script | What it does |
| --- | --- |
| `viewer_public.sh` | start the viewer on the public release (`http://subtdata.felk.cvut.cz/HELHEST/`, or `$VIEWER_URL`) and open a browser, using the checkout's `.venv-viewer` or `.venv` when there is one; extra flags go to `helhest-viewer` |
| `viewer_connect.sh` | open an ssh tunnel from a laptop to a viewer started by `slurm/viewer.batch` on an RCI compute node (`[local_port] [job_id]`; without a job id, your newest running `helhest_viewer` job) |

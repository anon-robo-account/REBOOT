# Episode Trimming Script

Trims frame windows out of LeRobot v3 datasets (e.g. cutting recovery episodes
down to their recovery-onset point) and writes the result as a new, fully
consistent dataset.

## Setup

Install REBOOT first:

```bash
git clone https://github.com/anon-robo-account/REBOOT
cd REBOOT
pip install -e .
```

Then drop `dataset_episode_trim.py` into the repo (or anywhere on your `PYTHONPATH`).

## Usage

Trim automatically from a dataset's `phase.json`, cutting at a named field and pushing the result to the Hub:

```bash
python dataset_episode_trim.py \
    --repo-id REBOOT26/sample_recovery-demonstration \
    --output-dir ~/datasets/sample_recovery_trimmed \
    --phase-json \
    --cut-key recovery_started_at_frame \
    --push-to-hub
```

Or trim using an inline or file-based cut spec:

```bash
python dataset_episode_trim.py \
    --repo-id REBOOT26/sample_recovery-demonstration \
    --output-dir ~/datasets/sample_recovery_trimmed \
    --truncations '{"3": [0, 120], "7": [50, null]}'
```

`--phase-json` with no path reads `meta/phase.json` from the dataset (local or
Hub, downloading it if needed). `--cut-key` names the field to head-trim at —
it's looked up in each episode's `failure` block first, then on the episode
entry itself; default is `t_r`.

## Notes

- Episodes not listed in the cut spec are copied unchanged.
- Depth features are dropped from the output.
- The script verifies episode counts, frame counts, and stats after building,
  and fails loudly if anything doesn't match.

Run `python dataset_episode_trim.py --help` for the full option list.

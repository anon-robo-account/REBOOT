#!/usr/bin/env python
"""Trim episode windows in a LeRobot v3 dataset.

Keeps a [start, stop) frame window per episode and writes the result as a new
dataset via LeRobotDataset.create/add_frame/save_episode, so frame indices,
videos, episode metadata, and statistics are all regenerated consistently.
Depth features are dropped from the output schema.

Cut points come from one of two sources:

  --truncations   inline JSON or a file: {"3": [0, 120], "7": [50, null]}
                  [start, stop) slice semantics; stop=null means to the end.
  --phase-json    a phase annotation file; each episode is head-trimmed at
                  failure.<cut-key> (default t_r), keeping [t_r, end). With no
                  path, reads meta/phase.json from the dataset, downloading it
                  from the Hub if it is not already local.

Episodes not listed are copied unchanged.

Examples:
    python dataset_episode_trim.py \
        --repo-id REBOOT26/sample_recovery-demonstration \
        --output-dir ~/datasets/1sample_recovery_trimmed \
        --phase-json \
        --cut-key recovery_started_at_frame
        --push_to_hub

    python dataset_episode_trim.py \
        --repo-id REBOOT26/dataset_name \
        --output-dir ~/datasets/dataset_name_trimmed \
        --truncations '{"3": [0, 120], "7": [50, null]}'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

AUTO = "auto"


def load_truncations(spec: str) -> dict[int, tuple[int, int | None]]:
    """Parse {episode: [start, stop]} from inline JSON or a file path."""
    text = Path(spec).read_text() if Path(spec).exists() else spec
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--truncations is neither valid JSON nor an existing file: {e}")
    if not isinstance(raw, dict):
        raise SystemExit('--truncations must be a JSON object, e.g. {"3": [0, 120]}')

    out = {}
    for k, v in raw.items():
        try:
            ep = int(k)
        except (TypeError, ValueError):
            raise SystemExit(f"--truncations keys must be episode indices, got {k!r}")
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise SystemExit(f"episode {k}: expected a [start, stop] pair, got {v!r}")
        start, stop = v
        try:
            out[ep] = (int(start), None if stop is None else int(stop))
        except (TypeError, ValueError):
            raise SystemExit(f"episode {k}: start/stop must be integers or null, got {v!r}")
    return out


def resolve_phase_json(spec: str, ds: LeRobotDataset, repo_id: str) -> Path:
    """Locate phase.json: an explicit path, else meta/phase.json in the dataset."""
    if spec != AUTO:
        p = Path(spec).expanduser()
        if not p.exists():
            raise SystemExit(f"--phase-json file not found: {spec}")
        return p

    local = Path(ds.meta.root) / "meta" / "phase.json"
    if local.exists():
        return local

    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename="meta/phase.json",
                                    repo_type="dataset"))
    except Exception as e:
        raise SystemExit(
            f"No phase.json found at {local} or in {repo_id} on the Hub ({e}). "
            "Pass an explicit path with --phase-json PATH."
        )


def load_truncations_from_phase_json(
    path: Path, cut_key: str, expect_repo: str | None = None
) -> dict[int, tuple[int, int | None]]:
    """Head-trim windows from a phase annotation file: keep [failure.<cut_key>, end)."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")

    annotated = data.get("dataset") or data.get("repo_id")
    if expect_repo and annotated and annotated != expect_repo:
        print(f"  warning: phase file annotates '{annotated}' but trimming '{expect_repo}'; "
              "frame indices may not correspond")

    episodes = data.get("episodes")
    if not episodes:
        raise SystemExit(f"{path} has no 'episodes' list")

    out = {}
    for entry in episodes:
        try:
            ep = int(entry["episode_index"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"bad episode_index in phase.json: {entry.get('episode_index')!r}")
        failure = entry.get("failure") or {}
        value = failure.get(cut_key, entry.get(cut_key))
        if value is None:
            available = sorted(set(failure) | set(entry))
            raise SystemExit(f"episode {ep}: no '{cut_key}' found (available: {available})")
        out[ep] = (int(value), None)
    return out


def validate_truncations(
    trunc: dict[int, tuple[int, int | None]],
    ep_lengths: dict[int, int],
    min_length: int,
) -> dict[int, tuple[int, int]]:
    """Resolve null stops, clamp overshoot, reject invalid windows, drop no-ops."""
    total = len(ep_lengths)
    effective = {}
    for ep, (start, stop) in sorted(trunc.items()):
        if ep not in ep_lengths:
            raise SystemExit(f"episode {ep} out of range (dataset has {total} episodes)")
        length = ep_lengths[ep]
        stop = length if stop is None else min(stop, length)
        if start < 0 or stop < 0:
            raise SystemExit(f"episode {ep}: negative window [{start}, {stop})")
        if start >= length:
            raise SystemExit(f"episode {ep}: start {start} >= episode length {length}")
        if stop <= start:
            raise SystemExit(f"episode {ep}: empty window [{start}, {stop})")
        if stop - start < min_length:
            raise SystemExit(
                f"episode {ep}: window keeps {stop - start} frames, below --min-length {min_length}"
            )
        if start == 0 and stop == length:
            continue  # whole episode, nothing to trim
        effective[ep] = (start, stop)
    return effective


def episode_row_map(ds: LeRobotDataset) -> dict[int, np.ndarray]:
    """episode_index -> ordered flat row indices, from a single column read."""
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    order = np.argsort(ep_col, kind="stable")
    sorted_eps = ep_col[order]
    boundaries = np.searchsorted(sorted_eps, np.unique(sorted_eps))
    unique_eps = sorted_eps[boundaries]
    splits = np.split(order, boundaries[1:])
    return {int(e): np.sort(rows) for e, rows in zip(unique_eps, splits, strict=True)}


def to_hwc_uint8(img) -> np.ndarray:
    """Decoded frames arrive CHW float [0,1]; add_frame expects HWC uint8."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def truncate_dataset(
    src: LeRobotDataset,
    new_repo_id: str,
    output_dir: Path,
    effective: dict[int, tuple[int, int]],
    push_to_hub: bool,
) -> LeRobotDataset:
    meta = src.meta
    features = {
        k: v for k, v in meta.features.items()
        if not (isinstance(v, dict) and (v.get("dtype") == "depth" or k.endswith("_depth")))
    }
    dropped = set(meta.features) - set(features)
    if dropped:
        print(f"Dropping depth features from output schema: {sorted(dropped)}")

    image_keys = [k for k, f in features.items()
                  if isinstance(f, dict) and f.get("dtype") in ("image", "video")]
    managed = {"index", "episode_index", "frame_index", "timestamp", "task_index"}
    passthrough_keys = [k for k in features if k not in image_keys and k not in managed]

    out = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=meta.fps,
        root=str(output_dir),
        robot_type=meta.robot_type,
        features=features,
        use_videos=bool(image_keys),
    )

    def episode_task(ep: int) -> str:
        try:
            tasks = meta.episodes[ep]["tasks"]
            return str(tasks[0]) if isinstance(tasks, (list, tuple)) else str(tasks)
        except Exception:
            return "task"

    rows_by_ep = episode_row_map(src)

    for ep in sorted(rows_by_ep):
        rows = rows_by_ep[ep]
        start, stop = effective.get(ep, (0, len(rows)))
        kept = rows[start:stop]
        task = episode_task(ep)

        label = f"[{start},{stop}) -> {len(kept)}" if ep in effective else f"{len(rows)} frames"
        print(f"  episode {ep:>4} ({label})")

        for row in kept:
            item = src[int(row)]
            frame = {}
            for k in image_keys:
                if k not in item:
                    raise KeyError(f"'{k}' in output schema but missing from item at row {row}; "
                                   f"item keys: {sorted(item.keys())}")
                frame[k] = to_hwc_uint8(item[k])
            for k in passthrough_keys:
                if k not in item:
                    raise KeyError(f"'{k}' in output schema but missing from item at row {row}; "
                                   f"item keys: {sorted(item.keys())}")
                frame[k] = item[k]
            frame["task"] = task
            out.add_frame(frame)

        out.save_episode()

    out.finalize()
    if push_to_hub:
        print("Pushing to hub...")
        out.push_to_hub()
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--new-repo-id", default=None)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--truncations",
                        help='inline JSON or file: {"3": [0, 120], "7": [50, null]}')
    source.add_argument("--phase-json", nargs="?", const=AUTO, metavar="PATH",
                        help="head-trim at failure.<cut-key>. With no PATH, reads "
                             "meta/phase.json from the dataset itself.")
    p.add_argument("--cut-key", default="t_r",
                   help="failure-block key to trim at with --phase-json (default t_r)")
    p.add_argument("--min-length", type=int, default=1,
                   help="refuse windows keeping fewer than this many frames")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--root", default=None)
    args = p.parse_args()

    new_repo_id = args.new_repo_id or f"{args.repo_id}_truncated"
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"--output-dir {output_dir} exists and is non-empty; pick a fresh path")

    print("Loading source dataset...")
    src = LeRobotDataset(args.repo_id, root=args.root)
    rows_by_ep = episode_row_map(src)
    ep_lengths = {e: len(r) for e, r in rows_by_ep.items()}
    print(f"  {args.repo_id}: {len(ep_lengths)} episodes, {src.meta.total_frames} frames")

    if args.phase_json:
        phase_path = resolve_phase_json(args.phase_json, src, args.repo_id)
        print(f"  phase annotations: {phase_path}")
        trunc = load_truncations_from_phase_json(phase_path, args.cut_key, args.repo_id)
    else:
        trunc = load_truncations(args.truncations)
    effective = validate_truncations(trunc, ep_lengths, args.min_length)
    if not effective:
        raise SystemExit("All requested windows cover whole episodes; nothing to do.")

    frames_before = src.meta.total_frames
    frames_removed = sum(ep_lengths[e] - (b - a) for e, (a, b) in effective.items())
    print(f"\nTrimming {len(effective)} episode(s), "
          f"{frames_before} -> {frames_before - frames_removed} frames")
    for e, (a, b) in sorted(effective.items()):
        print(f"    episode {e}: {ep_lengths[e]} -> keep [{a},{b}) = {b - a}")

    print(f"\nBuilding '{new_repo_id}' at {output_dir}")
    truncate_dataset(src, new_repo_id, output_dir, effective, args.push_to_hub)

    print("\nVerifying...")
    check = LeRobotDataset(new_repo_id, root=str(output_dir))
    new_lengths = {e: len(r) for e, r in episode_row_map(check).items()}
    ok = True
    for e, length in ep_lengths.items():
        expected = (effective[e][1] - effective[e][0]) if e in effective else length
        got = new_lengths.get(e, 0)
        if got != expected:
            print(f"  episode {e}: expected {expected}, got {got}")
            ok = False
    stats_ok = check.meta.stats is not None and "action" in check.meta.stats
    print(f"  episodes: {check.meta.total_episodes}, frames: {check.meta.total_frames}, "
          f"stats: {'ok' if stats_ok else 'MISSING'}")
    if not (ok and stats_ok and check.meta.total_episodes == len(ep_lengths)):
        raise SystemExit("Verification failed.")
    print(f"\nDone: {new_repo_id} at {output_dir}")


if __name__ == "__main__":
    main()
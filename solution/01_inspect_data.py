"""STEP 1 -- Inspect every data file and check the frame alignment.

    python solution/01_inspect_data.py            # text report
    python solution/01_inspect_data.py --plots    # + two PNGs in solution/figures/

Nothing here is a model. It tells you whether your download is complete and whether the files
mean what DATA.md says they mean. Run it once, read the output, then move on to step 2.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import common as C

sys.path.insert(0, os.path.join(C.ROOT, "starter"))
import load_atd         
import load_homography  
import load_smart_tagging 

from games import asset_paths, half_video_blocks


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def check_files():
    section("1. FILES")
    ok = True
    for source in ("tactical", "broadcast"):
        for name, path in asset_paths(source=source).items():
            exists = os.path.isfile(path)
            ok &= exists or source == "broadcast"
            size = "{:8.1f} MB".format(os.path.getsize(path) / 1e6) if exists else "  MISSING"
            print("  {:<10} {:<14} {}  {}".format(source, name, size, os.path.relpath(path, C.ROOT)))
    return ok


def check_tracking():
    section("2. TRACKING (ATD) - tactical")
    report = load_atd.check_alignment_raw()
    print("  records: {n_records} (video has {n_video_frames}), frame_id {0}..{1}, "
          "contiguous: {contiguous}".format(*report["frame_id_range"], **report))
    print("  alignment OK (video_frame = frame_id - 1): {}".format(report["agrees"]))

    players, ball, team = C.load_tracking()
    meta, _ = load_atd.atd_paths()
    print("  teams: H = {}, A = {}".format(*C.team_names(meta)))
    print("  track slots: {}  (home {}, away {})   <- slots, NOT players".format(
        players.shape[1], int((team == C.HOME).sum()), int((team == C.AWAY).sum())))

    graded = C.graded_mask()
    ball_ok = ~np.isnan(ball[:, 0])
    n_obj = (~np.isnan(players[:, :, 0])).sum(axis=1)
    print("  ball visible: {:.1f}% of all frames, {:.1f}% of graded frames  (DATA.md: 56.9 / 59.3)"
          .format(100 * ball_ok.mean(), 100 * ball_ok[graded].mean()))
    print("  tracked objects per graded frame: mean {:.1f}, min {}, max {}  (DATA.md: mean 20.5)"
          .format(n_obj[graded].mean(), n_obj[graded].min(), n_obj[graded].max()))
    print("  tracked objects at halftime (frame 100000): {}".format(n_obj[100000]))

    runs = C.runs(~ball_ok & graded)
    lengths = np.array([e - s for s, e in runs])
    print("  ball-missing runs inside graded play: {}, median {} frames, longest {} frames ({:.0f} s)"
          .format(len(runs), int(np.median(lengths)), lengths.max(), lengths.max() / C.FPS))
    print("  ball range (m): x {:.1f}..{:.1f}, y {:.1f}..{:.1f}  (pitch is {} x {})".format(
        np.nanmin(ball[:, 0]), np.nanmax(ball[:, 0]), np.nanmin(ball[:, 1]), np.nanmax(ball[:, 1]),
        C.PITCH_L, C.PITCH_W))
    return players, ball, team


def check_tagging():
    section("3. SMART TAGGING")
    events = C.load_tagging()
    print("  {} instances".format(len(events)))
    counts = pd.Series([e["code"] for e in events]).value_counts()
    print(counts.to_string())
    print("  Team tag values:", sorted({e["tags"].get("Team") for e in events} - {None}))
    report = load_smart_tagging.check_time_base(load_smart_tagging.load())
    print("  time base check (tagged kickoffs contain the real ones): {}".format(report["agrees"]))
    sp = [e for e in events if e["code"] == "SET PIECES"]
    print("  set pieces by Type:", pd.Series([e["tags"].get("Type") for e in sp]).value_counts().to_dict())
    na = [e for e in sp if C.tag_team_to_side(e["tags"].get("Team")) is None]
    print("  set pieces with Team = N/A: {} (all throw-ins, per DATA.md)".format(len(na)))
    print("  duration of a tagged interval, seconds: median {:.1f}, max {:.1f}".format(
        np.median([(e["end"] - e["start"]) / C.FPS for e in events]),
        max((e["end"] - e["start"]) / C.FPS for e in events)))


def check_homography():
    section("4. HOMOGRAPHY - tactical")
    H, err, valid = load_homography.load()
    report = load_homography.check_alignment()
    print("  rows: {} (video has {}), contiguous from 0: {}".format(
        report["n_rows"], report["n_video_frames"], report["contiguous"]))
    print("  valid: {:.1f}% of frames".format(100 * valid.mean()))
    graded = C.graded_mask()
    print("  reprojection error, graded frames: median {:.2f}, 95th pct {:.2f}".format(
        np.median(err[graded]), np.percentile(err[graded], 95)))
    print("  pitch corners in pixels at frame 11185:", [(round(x), round(y)) for x, y in
                                                       load_homography.pitch_corners(H[11185])])


def check_public_gt():
    section("5. PUBLIC GROUND TRUTH (first 5 minutes only)")
    gt = pd.read_csv(os.path.join(C.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    print("  rows: {}".format(len(gt)))
    print("  labels:", gt["label"].value_counts().to_dict())
    real = gt[gt["label"] != "X"]
    print("  covered frames: {}..{}".format(real["frame"].min(), real["frame"].max()))
    print("  halves (video frames):", half_video_blocks("BAR-ATH"))


def plots(players, ball, team):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(C.HERE, "figures")
    os.makedirs(out, exist_ok=True)

    # (a) ball availability per minute of video
    ok = (~np.isnan(ball[:, 0])).astype(float)
    per_min = ok[: len(ok) // 1500 * 1500].reshape(-1, 1500).mean(axis=1) * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(len(per_min)), per_min, width=1.0)
    ax.set_xlabel("minute of video"); ax.set_ylabel("% frames with ball"); ax.set_ylim(0, 100)
    ax.set_title("Ball availability (valleys = halftime / missing ball)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ball_availability.png"), dpi=120)

    # (b) a snapshot with a visible ball in the public window
    frame = next(f for f in range(11185 + 500, 18684) if not np.isnan(ball[f, 0]))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for t, c, n in ((C.HOME, "tab:red", "home"), (C.AWAY, "tab:blue", "away")):
        sel = team == t
        ax.scatter(players[frame, sel, 0], players[frame, sel, 1], c=c, s=70, label=n)
    ax.scatter(*ball[frame], c="k", s=90, marker="*", label="ball")
    ax.set_xlim(0, C.PITCH_L); ax.set_ylim(C.PITCH_W, 0); ax.set_aspect("equal"); ax.legend()
    ax.set_title("Frame {}".format(frame))
    fig.tight_layout(); fig.savefig(os.path.join(out, "snapshot.png"), dpi=120)
    print("\nplots written to", os.path.relpath(out, C.ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    #if not check_files():
    #   sys.exit("\nSome tactical files are missing - fix data/ first (see DATA.md).")
    players, ball, team = check_tracking()
    check_tagging()
    check_homography()
    check_public_gt()
    if args.plots:
        plots(players, ball, team)
    print("\nStep 1 done. If every check says True/OK, go to step 2 (baseline).")

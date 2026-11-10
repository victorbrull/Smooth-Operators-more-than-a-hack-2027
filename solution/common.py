"""Shared helpers: a fast ATD reader (with cache) and a Smart Tagging reader.

Everything is indexed by VIDEO frame (0 .. 199318), i.e. the same index as a submission row.
Run everything from the repository root, e.g.  python solution/01_inspect_data.py
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import (GAME, GAMES, DATA_DIR, require, source_frames,  # noqa: E402
                   half_video_blocks)

N_FRAMES = source_frames("tactical")            # 199319
FPS = GAMES[GAME]["fps"]                        # 25
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]         # 105.3 x 68.0 metres
HOME, AWAY = 0, 1

# Derived tracking is stored INSIDE data/ on purpose: data/ is gitignored, and the rules forbid
# publishing frame-level tracking exports.
CACHE_DIR = os.path.join(DATA_DIR, "cache")


# --------------------------------------------------------------------------------------------
# Tracking (ATD)
# --------------------------------------------------------------------------------------------
def team_of_each_slot(meta_path):
    """0 = home, 1 = away for every track slot, in the order the metadata lists them."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return tuple(t.findtext("Name") for t in root.iter("Team"))


def _parse_raw(raw_path, n_slots):
    """Raw line: '<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>'  ->  players (n,slots,2), ball (n,2).

    Coordinates are returned in METRES (the file stores 0-1). Missing values are NaN.
    """
    players = np.full((N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    scale = np.array([PITCH_L, PITCH_W], dtype=np.float64)
    bad = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                bad += 1
                continue
            frame = int(parts[0]) - 1                       # frame_id counts from 1
            if not 0 <= frame < N_FRAMES:
                bad += 1
                continue
            try:
                groups = np.array(parts[1].strip(";").replace(";", ",").split(","),
                                  dtype=np.float64).reshape(-1, 3)
            except ValueError:
                bad += 1
                continue
            k = min(len(groups), n_slots)
            players[frame, :k] = groups[:k, :2] * scale
            tokens = parts[2].strip().split(",")
            if len(tokens) >= 2:
                try:
                    ball[frame] = np.array(tokens[:2], dtype=np.float64) * scale
                except ValueError:
                    pass
    if bad:
        print("  warning: {} malformed lines skipped".format(bad))
    return players, ball


def load_tracking(use_cache=True):
    """(players (n,slots,2) m, ball (n,2) m, team_of_slot (slots,)). First call ~1 min, then cached."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    cache = os.path.join(CACHE_DIR, "tactical_tracking.npz")
    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        return z["players"], z["ball"], z["team"]

    team = team_of_each_slot(meta_path)
    print("parsing {} (one-off, about a minute) ...".format(os.path.basename(raw_path)))
    players, ball = _parse_raw(raw_path, n_slots=len(team))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players, ball=ball, team=team)
    return players, ball, team


# --------------------------------------------------------------------------------------------
# Smart Tagging
# --------------------------------------------------------------------------------------------
def load_tagging():
    """[{code, start, end, tags}] with start/end already in video frames (end is exclusive)."""
    path = require(["smart_tagging"])[0]
    out = []
    for inst in ET.parse(path).iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "tags": {l.findtext("group"): (l.findtext("text") or "").strip()
                     for l in inst.findall("label")},
        })
    return out


def tag_team_to_side(text):
    """'Barcelona' -> HOME, 'Atletic Club' (sic, one t) -> AWAY, 'N/A' or anything else -> None."""
    t = (text or "").lower()
    if "barc" in t or "bar\u00e7a" in t:
        return HOME
    if "ath" in t or "atl" in t:
        return AWAY
    return None


def graded_mask():
    """Boolean (n,) mask of graded frames: the two halves, nothing else."""
    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def runs(mask):
    """[(start, end_exclusive)] of every maximal True run in a boolean array."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))

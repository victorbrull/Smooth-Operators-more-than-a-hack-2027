"""STEP 3 -- A first real possession model (v1), built to beat the nearest-player baseline.

    python solution/03_possession_v1.py                    # writes evaluation/results/v1.csv
    python solution/03_possession_v1.py --name v1b --set switch_cost=60

IDEA (all rule-based, no training, no labels, so nothing here touches the ground truth):

  A. Ball cleaning       Fill short ball gaps (<= 0.5 s) by interpolation; longer gaps stay empty.
  B. Dead-ball detector  D when the ball is (1) stationary for ~2 s, (2) outside the pitch for
                         ~1 s, (3) lost for >= 12 s, or (4) in the known pre-kickoff stretch of the
                         2nd half (documented in FAQ.md). Short D blips are removed.
  C. Evidence            For every frame with a ball: which team has the nearest player, and how
                         clearly (distance margin). Smart Tagging adds a weak prior: a set piece's
                         team restarts with the ball; attacking phases -> that team has it,
                         defending phases -> the opponent has it.
  D. Possession          A 2-state Viterbi (H/A) inside each live stretch. Evidence disagreeing with
                         the current state costs its confidence; SWITCHING costs `switch_cost`
                         frames' worth. So possession only changes after sustained, confident
                         evidence ("controlled" possession), and frames without a ball are bridged
                         for free. This is what kills the flicker that ruins spellF1.
  E. Output              X outside the two graded halves, otherwise H / A / D.

The PARAMS below are reasoned starting points, NOT tuned. Do not tune them on the public 5
minutes (RULES.md) -- use them to sanity-check the pipeline and read the diagnostics instead.
"""

import argparse
import os

import numpy as np
import pandas as pd

import common as C

PARAMS = dict(
    gap_fill=12,           # frames. Interpolate ball gaps up to 0.5 s.
    ctrl_radius=4.0,       # m. Nearest player further than this gives no evidence.
    margin_full=2.0,       # m. |d_away - d_home| at which evidence is at full strength.
    switch_cost=40.0,      # frames of full-strength opposing evidence needed to change team.
    still_half=25,         # frames. Rolling window = 2*still_half+1 (about 2 s).
    still_min_obs=30,      # min ball observations inside that window.
    still_extent=1.5,      # m. Ball wanders less than this -> stationary.
    oob_margin=0.5,        # m beyond the touchline / goal line.
    oob_min=25,            # frames outside the pitch before it is a dead ball.
    lost_min=300,          # frames with no ball at all -> dead (12 s).
    min_dead=50,           # drop dead-ball runs shorter than 2 s ...
    merge_dead=25,         # ... and merge dead runs closer than 1 s.
    setpiece_w=0.5,        # prior weight of a set piece's Team tag (per frame, signed).
    phase_w=0.15,          # prior weight of attacking / defending phase tags.
    second_half_kickoff=113742,   # FAQ.md: frames 110808..113741 are graded and all D.
)

ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
          "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
          "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


# ---- A. ball cleaning ---------------------------------------------------------------------
def fill_short_gaps(ball, max_gap):
    filled = ball.copy()
    for k in range(2):
        s = pd.Series(ball[:, k]).interpolate(method="linear", limit_area="inside")
        filled[:, k] = s.values
    for start, end in C.runs(np.isnan(ball[:, 0])):
        if end - start > max_gap:
            filled[start:end] = np.nan
    return filled


# ---- B. dead ball -------------------------------------------------------------------------
def dead_ball_mask(ball_raw, ball_filled, graded, p):
    n = len(ball_raw)
    reasons = {}

    win = 2 * p["still_half"] + 1
    ext = []
    for k in range(2):
        s = pd.Series(ball_raw[:, k])
        roll = s.rolling(win, center=True, min_periods=p["still_min_obs"])
        ext.append((roll.max() - roll.min()).values)
    still = np.fmax(ext[0], ext[1]) < p["still_extent"]          # NaN -> False
    reasons["stationary ball"] = still & graded

    x, y = ball_filled[:, 0], ball_filled[:, 1]
    m = p["oob_margin"]
    outside = (x < -m) | (x > C.PITCH_L + m) | (y < -m) | (y > C.PITCH_W + m)   # NaN -> False
    oob = np.zeros(n, dtype=bool)
    for s, e in C.runs(outside):
        if e - s >= p["oob_min"]:
            oob[s:e] = True
    reasons["ball outside pitch"] = oob & graded

    lost = np.zeros(n, dtype=bool)
    for s, e in C.runs(np.isnan(ball_raw[:, 0]) & graded):
        if e - s >= p["lost_min"]:
            lost[s:e] = True
    reasons["ball lost >= 12 s"] = lost

    pre = np.zeros(n, dtype=bool)
    second_start = [f for h, f, _ in C.half_video_blocks(C.GAME) if h == 2][0]
    pre[second_start:p["second_half_kickoff"]] = True
    reasons["2nd-half pre-kickoff"] = pre

    dead = still | oob | lost
    # merge close runs, drop tiny ones
    for s, e in C.runs(~dead):
        if s > 0 and e < n and e - s < p["merge_dead"]:
            dead[s:e] = True
    for s, e in C.runs(dead):
        if e - s < p["min_dead"]:
            dead[s:e] = False
    dead |= pre
    return dead & graded, reasons


# ---- C. evidence --------------------------------------------------------------------------
def tracking_evidence(players, ball, team, p):
    """Signed evidence per frame in [-1, 1]: +1 home has the ball, -1 away. 0 = no information."""
    n = len(ball)
    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(axis=2))        # (n, slots), NaN-safe
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, team == C.HOME].min(axis=1)
    d_away = dist[:, team == C.AWAY].min(axis=1)
    ok = np.isfinite(d_home) & np.isfinite(d_away) & ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        margin = np.where(ok, d_away - d_home, 0.0)                          # >0: home closer
    near = np.minimum(d_home, d_away) <= p["ctrl_radius"]
    conf = np.clip(np.abs(margin) / p["margin_full"], 0.0, 1.0)
    return np.where(ok & near, np.sign(margin) * conf, 0.0)


def tagging_prior(events, n, p):
    prior = np.zeros(n)
    by_code = {}
    for e in events:
        side = C.tag_team_to_side(e["tags"].get("Team"))
        if side is None:
            continue
        sign = 1.0 if side == C.HOME else -1.0
        if e["code"] == "SET PIECES":
            w, s = p["setpiece_w"], sign
        elif e["code"] in ATTACK:
            w, s = p["phase_w"], sign
        elif e["code"] in DEFEND:
            w, s = p["phase_w"], -sign
        else:
            continue
        prior[e["start"]:e["end"]] += s * w
        by_code.setdefault(e["code"], []).append((e["start"], e["end"], s))
    return np.clip(prior, -1, 1), by_code


# ---- D. two-state Viterbi -----------------------------------------------------------------
def viterbi(evidence, switch_cost):
    """States 0 = home, 1 = away. Cost of a state at t = evidence against it; switching costs
    `switch_cost`. Returns the cheapest path as an int array."""
    n = len(evidence)
    c_home = np.maximum(-evidence, 0.0).tolist()
    c_away = np.maximum(evidence, 0.0).tolist()
    back = np.zeros((n, 2), dtype=bool)
    h, a = c_home[0], c_away[0]
    for t in range(1, n):
        from_away = a + switch_cost
        if from_away < h:
            nh, back[t, 0] = from_away + c_home[t], True
        else:
            nh = h + c_home[t]
        from_home = h + switch_cost
        if from_home < a:
            na, back[t, 1] = from_home + c_away[t], True
        else:
            na = a + c_away[t]
        h, a = nh, na
    path = np.zeros(n, dtype=np.int8)
    state = 0 if h <= a else 1
    for t in range(n - 1, -1, -1):
        path[t] = state
        if back[t, state]:
            state = 1 - state
    return path


# ---- pipeline -----------------------------------------------------------------------------
def run(p):
    players, ball_raw, team = C.load_tracking()
    graded = C.graded_mask()
    n = C.N_FRAMES

    ball = fill_short_gaps(ball_raw, p["gap_fill"])
    dead, reasons = dead_ball_mask(ball_raw, ball, graded, p)

    evidence = tracking_evidence(players, ball, team, p)
    events = C.load_tagging()
    prior, by_code = tagging_prior(events, n, p)
    total = np.clip(evidence + prior, -1.0, 1.0)

    labels = np.full(n, "X", dtype="<U1")
    labels[graded] = "D"
    side = np.full(n, -1, dtype=np.int8)
    for s, e in C.runs(graded & ~dead):
        side[s:e] = viterbi(total[s:e], p["switch_cost"])
    labels[side == 0] = "H"
    labels[side == 1] = "A"
    return labels, dict(evidence=evidence, by_code=by_code, reasons=reasons, graded=graded, dead=dead)


def report(labels, info):
    graded = info["graded"]
    print("\nDEAD-BALL RULES (frames flagged by each rule, before cleaning):")
    for name, mask in info["reasons"].items():
        print("  {:<24} {:>7} frames in {:>3} runs".format(name, int(mask.sum()), len(C.runs(mask))))
    print("  after cleaning: {} dead frames in {} runs".format(
        int(info["dead"].sum()), len(C.runs(info["dead"]))))

    counts = {k: int((labels == k).sum()) for k in "HADX"}
    spells = [(e - s) for lab in "HA" for s, e in C.runs(labels == lab)]
    print("\nOUTPUT: H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  possessions (maximal H or A runs): {}   mean {:.1f} s   median {:.1f} s".format(
        len(spells), np.mean(spells) / C.FPS, np.median(spells) / C.FPS))
    print("  home share of live play: {:.1f}%".format(100 * counts["H"] / (counts["H"] + counts["A"])))

    print("\nSANITY CHECK of the Smart-Tagging sign convention (no ground truth involved):")
    print("  mean agreement between tag-implied side and tracking evidence; should be > 0")
    ev = info["evidence"]
    for code, items in sorted(info["by_code"].items()):
        vals = [np.mean(s * ev[a:b][np.abs(ev[a:b]) > 0.5]) for a, b, s in items
                if (np.abs(ev[a:b]) > 0.5).any()]
        if vals:
            print("  {:<32} n={:>3}  agreement {:+.2f}".format(code, len(vals), np.mean(vals)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="v1", help="output file name (no .csv)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override a PARAM")
    args = ap.parse_args()

    params = dict(PARAMS)
    for item in args.set:
        key, value = item.split("=")
        if key not in params:
            raise SystemExit("unknown param {!r}; have {}".format(key, sorted(params)))
        params[key] = type(params[key])(float(value))

    labels, info = run(params)
    report(labels, info)

    out = os.path.join(C.ROOT, "evaluation", "results", args.name + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"frame": np.arange(C.N_FRAMES), "label": labels}).to_csv(out, index=False)
    print("\nwrote", os.path.relpath(out, C.ROOT))
    print("next:  python evaluation/validate_submission.py {}".format(os.path.relpath(out, C.ROOT)))
    print("       python evaluation/run_evaluation.py")

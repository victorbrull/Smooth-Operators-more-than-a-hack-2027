# Hackathon 2027 - working steps (put this `solution/` folder at the root of the cloned repo)

Run every command **from the repository root**, next to `games.py`.

## Step 0 - Environment (once)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-starter.txt        # numpy, pandas, kloppy, matplotlib, notebook
```
If your data is not in `./data`:  `export MTAH_DATA_DIR=/path/to/data`  (PowerShell: `$env:MTAH_DATA_DIR="D:\path"`).

## Step 1 - Inspect and verify the data
```bash
python solution/01_inspect_data.py --plots
```
Checks that all files exist, that frames align (`video_frame = frame_id - 1`), ball visibility
(~57 % of frames, ~59 % of graded), Smart Tagging time base, homography validity, public ground truth.
The first run parses the 150 MB tracking text and caches it in `data/cache/` (gitignored). Plots go to `solution/figures/`.
Expected: every check `True`/`OK`, numbers close to those quoted in DATA.md.

## Step 2 - Reproduce the official baselines and the scorer
```bash
python starter/baseline.py                      # rebuilds the 4 nearest-player baselines (~2 min)
python evaluation/run_evaluation.py             # scores them on the public 5 minutes
```
You should get exactly the table in the main README (smooth125 0.4242 ... raw 0.0378). If not, something is misaligned.

## Step 3 - First real model (v1) -> your first submission file
```bash
python solution/03_possession_v1.py             # writes evaluation/results/v1.csv
python evaluation/validate_submission.py evaluation/results/v1.csv
python evaluation/run_evaluation.py --bootstrap 1000
```
Read the printed diagnostics: dead-ball frames per rule, number/mean length of possessions
(a football match has far fewer than the raw baseline's ~131 per 5 min), and the Smart-Tagging
sign check (every line should be positive; if a code is negative, flip it in `ATTACK`/`DEFEND`).

## Step 4 - Improve (ideas, in order of expected value)
1. **Dead ball**: it is half of the label quality. Combine ball speed, out-of-pitch, player clustering
   at restarts and the set-piece windows (`SET PIECES`, tagged ~5 s around the restart).
2. **Ball recovery**: when the ball is missing (41 %), infer its position from player movement
   (e.g. the centroid of the players converging on a point) instead of relying on short interpolation.
3. **Control proxy**: replace "nearest player" with "nearest player AND ball speed low / ball
   decelerating" to ignore passes flying past opponents.
4. **Evidence per half** and `switch_cost` chosen by reasoning (e.g. typical possession length), not by the 5 public minutes.

## Rules reminders (RULES.md / FAQ.md)
- Never upload the data, or extracts of it, to a hosted LLM (ChatGPT, Claude, Gemini...).
- Do not tune or train on the public 5 minutes; use them only to check the pipeline.
- No hand labelling of this match. Your repo must regenerate the exact CSV and must NOT contain `data/`.
- Submission: `<team_name>.csv` (199,319 rows) + `<team_name>.pdf` + public repo URL, before **17 Nov 2026**.

## The data files in one line each
| file | what it is |
|---|---|
| `tactical/BAR-ATH_tactical.mp4` | The reference video: 199,319 frames, 25 fps. Row *i* of your submission = frame *i*. |
| `tactical/..._atd.txt` | Tracking, one line per frame: `frame_id:x,y,speed;...(46 slots)...:ball_x,ball_y,speed`. Coordinates 0-1 (x length, y width); `NaN` = not seen. |
| `tactical/..._atd.xml` | Metadata of the tracking: which team each of the 46 slots belongs to (first team = home, Barcelona). Slots are tracklets, not players. |
| `tactical/..._homography.csv` | One 3x3 matrix per frame mapping pitch (0-1) to image pixels; `valid` flag and `reprojection_error`. Used to draw tracking on the video. |
| `tactical/..._smart_tagging.xml` | 638 automatic tactical phases / set pieces: start, end (seconds), code, Team, Half, Type. Noisy prior, not truth. |
| `broadcast/*` | Same four things for the TV feed, on its own timeline (164,488 frames), NOT aligned to tactical. Optional. |
| `evaluation/gt/..._public.csv` | Truth for the first 5 minutes (frames 11185-18684) only. For checking, not for tuning. |

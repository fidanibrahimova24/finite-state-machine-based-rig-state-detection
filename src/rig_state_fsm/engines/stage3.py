# ============================================================
# MEMORY-SAFE FSM RIG STATE DETECTION PIPELINE
# Corrected tripping / reaming / backreaming / connection logic
# Stage-2 FSM-ready inputs. One-cell notebook version.
# ============================================================

import pandas as pd
import numpy as np
import os
import re
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from IPython.display import display
except ImportError:  # CLI/package execution does not require Jupyter/IPython.
    def display(obj):
        print(obj)

# ============================================================
# 0. USER SETTINGS
# ============================================================

OUTPUT_DIR = Path(os.environ.get("FSM_OUTPUT_DIR", "fsm_memory_safe_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Critical memory controls
FSM_RESAMPLE_RULE = "10s"          # Use "5s" for more detail, None for full resolution only on strong computer
PLOT_ACTIVE_MAX_POINTS = 12000
PLOT_OVERLAY_MAX_POINTS = 18000
SHOW_PLOTS_IN_NOTEBOOK = False     # Keep False to avoid browser/RAM crash in Jupyter
SAVE_HTML_PLOTS = True
SAVE_FULL_LABELED_CSV = False      # Full labeled CSV can be very large
SAVE_COMPACT_LABELED_PARQUET = False

# Shared visualization defaults used by all generated dashboards. They affect
# display only; FSM classification and calculated values are unchanged.
VISUALIZATION_LINE_WIDTH = 4.0
VISUALIZATION_STATE_LINE_WIDTH = 5.0
VISUALIZATION_AXIS_FONT_SIZE = 18
VISUALIZATION_TITLE_FONT_SIZE = 24
VISUALIZATION_LEGEND_FONT_SIZE = 16
VISUALIZATION_TICK_FONT_SIZE = 15
VISUALIZATION_MARKER_SIZE = 10
VISUALIZATION_EXPORT_SCALE = 3

# Logic thresholds. These are intentionally conservative.
RPM_ON = 5.0
FLOW_ON = 30.0
SPP_ON = 100.0
WOB_ON = 2.0
ROP_ON = 0.5

# Movement thresholds are ft/min after smoothing.
BIT_MOVE_FTPM = 1.5
BLOCK_MOVE_FTPM = 1.0
HOLE_ADVANCE_FTPM = 0.03

# On-bottom / connection controls
ON_BOTTOM_DEFAULT_FT = 15.0
ON_BOTTOM_MAX_FT = 40.0
ON_BOTTOM_MIN_FT = 5.0
NEAR_BOTTOM_FOR_CONNECTION_FT = 250.0

# Episode rules
MIN_CONNECTION_SEC = 20
MAX_CONNECTION_SEC = 25 * 60
MIN_MOVING_EPISODE_SEC = 20
MIN_STATE_GROUP_SEC = 20
STAND_LENGTH_FT = 90.0
STAND_TOLERANCE_FT = 18.0
MIN_VALID_STAND_FT = 60.0
MAX_VALID_STAND_FT = 120.0
MIN_STAND_DRILLING_FRACTION = 0.35

# Stand-pattern connection detection. This adds expected stand connections
# from drilled footage and parameter-drop patterns.
ENABLE_STAND_PATTERN_CONNECTIONS = True
CONNECTION_DEPTH_TOL_FT = 45.0
CONNECTION_SEARCH_PAD_MIN = 20.0
MAX_DRILL_SESSION_GAP_MIN = 240.0
MIN_DRILL_SESSION_FOOTAGE_FT = 120.0
MIN_CONNECTION_PATTERN_SCORE = 4
MIN_WEAK_CONNECTION_PATTERN_SCORE = 3

# Active-window plot length
ACTIVE_WINDOW_HOURS = 8

def _apply_visualization_style(fig):
    """Apply shared readable styling without changing the plotted data."""
    for trace in fig.data:
        mode = str(getattr(trace, "mode", "") or "")
        if "lines" in mode:
            current = getattr(getattr(trace, "line", None), "width", None)
            trace.line.width = max(float(current or 0), VISUALIZATION_LINE_WIDTH)
        if "markers" in mode:
            current = getattr(getattr(trace, "marker", None), "size", None)
            if current is None or np.isscalar(current):
                trace.marker.size = max(float(current or 0), VISUALIZATION_MARKER_SIZE)
    fig.update_layout(
        font=dict(size=VISUALIZATION_AXIS_FONT_SIZE, color="#172033"),
        title_font=dict(size=VISUALIZATION_TITLE_FONT_SIZE),
        legend_font=dict(size=VISUALIZATION_LEGEND_FONT_SIZE),
        hoverlabel=dict(font_size=16),
    )
    fig.update_xaxes(title_font=dict(size=VISUALIZATION_AXIS_FONT_SIZE), tickfont=dict(size=VISUALIZATION_TICK_FONT_SIZE), tickwidth=2, linewidth=2, gridwidth=1.2)
    fig.update_yaxes(title_font=dict(size=VISUALIZATION_AXIS_FONT_SIZE), tickfont=dict(size=VISUALIZATION_TICK_FONT_SIZE), tickwidth=2, linewidth=2, gridwidth=1.2)
    for annotation in fig.layout.annotations or ():
        size = getattr(getattr(annotation, "font", None), "size", None)
        annotation.font.size = max(float(size or 0), 16)
    return fig

# ============================================================
# 1. INPUT DATAFRAME WITHOUT FULL COPY
# ============================================================

input_name = None
for candidate in ["processed_df", "fsm_ready_df", "stage2_df", "df_clean", "working_df", "df"]:
    if candidate in globals() and isinstance(globals()[candidate], pd.DataFrame):
        input_name = candidate
        raw = globals()[candidate]       # IMPORTANT: no .copy() here
        break

if input_name is None:
    raise ValueError("No dataframe found. Run Stage 2 first; expected processed_df or fsm_ready_df.")

print("=== INPUT ===")
print(f"Using dataframe by reference: {input_name}")
print(f"Rows in source dataframe: {len(raw):,}")
print(f"Source columns: {len(raw.columns):,}")

# ============================================================
# 2. FLEXIBLE COLUMN MAPPING
# ============================================================

def _norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

def pick_col(df, candidates, required=True):
    norm_map = {_norm(c): c for c in df.columns}

    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]

    actual_cols = list(df.columns)
    actual_norms = {c: _norm(c) for c in actual_cols}
    for cand in candidates:
        key = _norm(cand)
        for c, cn in actual_norms.items():
            if key == cn or key in cn or cn in key:
                return c

    if required:
        raise ValueError(
            "Could not find required column. Tried:\n"
            + "\n".join(map(str, candidates))
            + "\n\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )
    return None

# Time columns
date_col = pick_col(raw, ["DATE", "Date", "date", "YYYY/MM/DD"], required=False)
time_col = pick_col(raw, ["TIME", "Time", "time", "HH:MM:SS"], required=False)
timestamp_col = pick_col(raw, ["TIMESTAMP", "Timestamp", "UtcTime", "UTC_TIME", "DateTime", "RigTime"], required=False)

colmap = {
    "BLOCK_POSITION": pick_col(raw, ["BLOCK_POSITION_FSM_FT", "BLOCK_POSITION_FSM_INPUT", "BLOCK_POSITION (ft)", "BLOCK_POSITION", "Block Position", "Block Height", "BPOS", "HKH"], required=False),
    "HOOK_LOAD": pick_col(raw, ["HOOK_LOAD_FSM_INPUT", "HOOK_LOAD (klbs)", "HOOK_LOAD", "Hook Load", "HKLD", "HKLA", "WOH"]),
    "BIT_DEPTH": pick_col(raw, ["BIT_DEPTH_FSM_FT", "BIT_DEPTH_FSM_INPUT", "BIT_DEPTH (ft)", "BIT_DEPTH", "Bit Depth", "DBTM", "BDEP", "BitDepth"]),
    "HOLE_DEPTH": pick_col(raw, ["HOLE_DEPTH_FSM_FT", "HOLE_DEPTH_FSM_INPUT", "HOLE_DEPTH (ft)", "HOLE_DEPTH", "Hole Depth", "DMEA", "DEPTH", "MD", "HoleDepth"]),
    "PUMP_OUTPUT": pick_col(raw, ["PUMP_OUTPUT_FSM_GPM", "PUMP_OUTPUT_FSM_INPUT", "PUMP_OUTPUT (gpm)", "PUMP_OUTPUT", "Pump Output", "Total Pump Output", "Flow In", "FLOW_IN", "MFIA"]),
    "STANDPIPE_PRESSURE": pick_col(raw, ["PUMP_PRESSURE_FSM_PSI", "PUMP_PRESSURE_FSM_INPUT", "STANDPIPE_PRESSURE (psi)", "STANDPIPE_PRESSURE", "Standpipe Pressure", "Pump Pressure", "PUMP_PRESSURE", "SPPA", "SPP"]),
    "DIFF_PRESSURE": pick_col(raw, ["DIFF_PRESSURE_FSM_PSI", "DIFF_PRESSURE_SELECTED_FSM_INPUT", "DIFF_PRESSURE (psi)", "DIFF_PRESSURE", "Differential Pressure", "DP"], required=False),
    "WOB": pick_col(raw, ["WOB_FSM_KLBF", "WOB_SELECTED_FSM_INPUT", "WOB (klbs)", "WOB", "Weight on Bit", "WOBA"]),
    "ROP": pick_col(raw, ["ROP_FSM_FT_HR", "ROP_SELECTED_FSM_INPUT", "ROP (ft/hr)", "ROP", "Rate Of Penetration", "ROPA"]),
    "ROTARY_RPM": pick_col(raw, ["ROTARY_RPM_FSM_INPUT", "ROTARY_RPM (rpm)", "ROTARY_RPM", "Rotary RPM", "RPMA", "RPM"]),
    "ROTARY_TORQUE": pick_col(raw, ["ROTARY_TORQUE_FSM_KFT_LBF", "ROTARY_TORQUE_FSM_INPUT", "ROTARY_TORQUE (kft-lbf)", "ROTARY_TORQUE", "Rotary Torque", "TQA", "Torque"]),
    "MSE_TEALE_PSI": pick_col(raw, ["MSE_TEALE_FSM_PSI"], required=False),
    "MSE_HYDRAULIC_PSI": pick_col(raw, ["MSE_HYDRAULIC_FSM_PSI"], required=False),
}

print("\n=== COLUMN MAP USED ===")
display(pd.DataFrame([{"Internal Name": k, "Original Column": v} for k, v in colmap.items()]))

# ============================================================
# 3. BUILD COMPACT DATAFRAME ONLY FROM NEEDED COLUMNS
# ============================================================

if date_col is not None and time_col is not None:
    ts = pd.to_datetime(
        raw[date_col].astype(str).str.strip() + " " + raw[time_col].astype(str).str.strip(),
        errors="coerce"
    )
    timestamp_source = "DATE + TIME"
elif timestamp_col is not None:
    ts = pd.to_datetime(raw[timestamp_col], errors="coerce")
    timestamp_source = str(timestamp_col)
else:
    raise ValueError("No timestamp found. Need DATE + TIME or TIMESTAMP / UtcTime / DateTime.")

# Honor the Stage-2 row-quality contract before resampling or interpolation.
if "FSM_INPUT_VALID" in raw.columns:
    stage2_valid = raw["FSM_INPUT_VALID"].fillna(False).astype(bool)
    stage2_valid_pct = float(stage2_valid.mean() * 100.0)
else:
    stage2_valid = pd.Series(True, index=raw.index)
    stage2_valid_pct = np.nan
    warnings.warn("FSM_INPUT_VALID is absent; compatibility mode is being used.")

# Build only required numeric columns, not a copy of the whole raw dataframe.
df = pd.DataFrame({"FSM_TIMESTAMP": ts, "_STAGE2_VALID": stage2_valid.to_numpy()})
for internal, source in colmap.items():
    if source is None:
        df[internal] = np.nan
    else:
        df[internal] = pd.to_numeric(raw[source], errors="coerce")

# Drop bad timestamps and duplicate timestamps before feature creation.
df = df[df["FSM_TIMESTAMP"].notna() & df["_STAGE2_VALID"]].drop(columns="_STAGE2_VALID")
df = df.sort_values("FSM_TIMESTAMP").reset_index(drop=True)

numeric_cols = [c for c in df.columns if c != "FSM_TIMESTAMP"]

if df["FSM_TIMESTAMP"].duplicated().any():
    dup_count = int(df["FSM_TIMESTAMP"].duplicated().sum())
    df = df.groupby("FSM_TIMESTAMP", as_index=False)[numeric_cols].median().sort_values("FSM_TIMESTAMP").reset_index(drop=True)
else:
    dup_count = 0

# Optional resampling. This is the main protection against memory/browser crashes.
if FSM_RESAMPLE_RULE is not None:
    before_rows = len(df)
    df = (
        df.set_index("FSM_TIMESTAMP")[numeric_cols]
          .resample(FSM_RESAMPLE_RULE)
          .median()
          .dropna(how="all")
          .reset_index()
    )
    print(f"\nResampled from {before_rows:,} rows to {len(df):,} rows using {FSM_RESAMPLE_RULE}.")

print("\n=== TIMESTAMP CHECK ===")
print("Timestamp source:", timestamp_source)
print("Start:", df["FSM_TIMESTAMP"].min())
print("End:  ", df["FSM_TIMESTAMP"].max())
print("Rows after compact build/resample:", f"{len(df):,}")
print("Duplicate timestamps aggregated:", dup_count)

# ============================================================
# 4. NUMERIC CLEANING
# ============================================================

sentinels = [-9999, -999.25, -999, -8888, -7777, -1e20, 1e20]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").replace(sentinels, np.nan)
    df.loc[df[col].abs() > 1e10, col] = np.nan

nonnegative_cols = [
    "HOOK_LOAD", "BIT_DEPTH", "HOLE_DEPTH", "PUMP_OUTPUT",
    "STANDPIPE_PRESSURE", "WOB", "ROP", "ROTARY_RPM", "ROTARY_TORQUE"
]
for col in nonnegative_cols:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

# Short interpolation only. Do not create long artificial runs across gaps.
for col in numeric_cols:
    df[col] = df[col].interpolate(limit=3, limit_direction="both")

# Stage 2 already distinguishes true zero from missing. Never turn missing
# operational data into "off", and never bridge long depth/load gaps here.

# Reduce memory after cleaning.
for col in numeric_cols:
    df[col] = df[col].astype("float32")

# MSE is used for engineering interpretation, not as a state-transition input.
# Plot fixed-unit Stage-2 MSE in ksi so the scale is readable.
df["MSE_TEALE_KSI"] = df["MSE_TEALE_PSI"] / 1000.0
df["MSE_HYDRAULIC_KSI"] = df["MSE_HYDRAULIC_PSI"] / 1000.0

# ============================================================
# 5. FEATURE EXTRACTION
# ============================================================

raw_dt_sec = df["FSM_TIMESTAMP"].diff().dt.total_seconds()
med_dt = float(raw_dt_sec.replace([np.inf, -np.inf], np.nan).median())
if not np.isfinite(med_dt) or med_dt <= 0:
    med_dt = 1.0

# Never bridge Stage-2 exclusions or acquisition outages into a false state
# episode. The first sample after a gap is explicitly labeled Data Gap, and
# missing time is not credited to the following operational state.
data_gap_limit_sec = max(60.0, med_dt * 3.0)
df["DATA_GAP"] = raw_dt_sec.gt(data_gap_limit_sec).fillna(False)
df["DT_SEC"] = raw_dt_sec.where(
    raw_dt_sec.gt(0) & raw_dt_sec.le(data_gap_limit_sec), med_dt
).astype("float32")

sample_dt = max(med_dt, 1.0)
vel_win = max(3, int(round(60.0 / sample_dt)))
stable_win = max(3, int(round(120.0 / sample_dt)))

# Smooth depths lightly before velocity. This prevents one-sample noise from becoming fake trips.
df["BIT_DEPTH_SM"] = df["BIT_DEPTH"].rolling(vel_win, center=True, min_periods=1).median().astype("float32")
df["HOLE_DEPTH_SM"] = df["HOLE_DEPTH"].rolling(vel_win, center=True, min_periods=1).median().astype("float32")
df["BLOCK_POSITION_SM"] = df["BLOCK_POSITION"].rolling(vel_win, center=True, min_periods=1).median().astype("float32")
df["HOOK_LOAD_SM"] = df["HOOK_LOAD"].rolling(vel_win, center=True, min_periods=1).median().astype("float32")

for col in ["PUMP_OUTPUT", "STANDPIPE_PRESSURE", "DIFF_PRESSURE", "WOB", "ROP",
            "ROTARY_RPM", "ROTARY_TORQUE", "MSE_TEALE_KSI", "MSE_HYDRAULIC_KSI"]:
    df[col + "_SM"] = df[col].rolling(max(3, int(round(30.0 / sample_dt))), center=True, min_periods=1).median().astype("float32")

# Velocity in ft/min.
df["BIT_VEL_FTPM"] = (df["BIT_DEPTH_SM"].diff() / df["DT_SEC"] * 60.0).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
df["HOLE_VEL_FTPM"] = (df["HOLE_DEPTH_SM"].diff() / df["DT_SEC"] * 60.0).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
df["BLOCK_VEL_FTPM"] = (df["BLOCK_POSITION_SM"].diff() / df["DT_SEC"] * 60.0).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

# Sustained trends are more stable than single-sample velocity.
trend_n = max(3, int(round(90.0 / sample_dt)))
df["BIT_TREND_FT"] = (df["BIT_DEPTH_SM"] - df["BIT_DEPTH_SM"].shift(trend_n)).fillna(0).astype("float32")
df["HOLE_TREND_FT"] = (df["HOLE_DEPTH_SM"] - df["HOLE_DEPTH_SM"].shift(trend_n)).fillna(0).astype("float32")
df["BLOCK_TREND_FT"] = (df["BLOCK_POSITION_SM"] - df["BLOCK_POSITION_SM"].shift(trend_n)).fillna(0).astype("float32")

# Stable range windows.
df["BIT_RANGE_STABLE_WIN"] = (df["BIT_DEPTH_SM"].rolling(stable_win, min_periods=1).max() - df["BIT_DEPTH_SM"].rolling(stable_win, min_periods=1).min()).astype("float32")
df["HOLE_RANGE_STABLE_WIN"] = (df["HOLE_DEPTH_SM"].rolling(stable_win, min_periods=1).max() - df["HOLE_DEPTH_SM"].rolling(stable_win, min_periods=1).min()).astype("float32")
df["BLOCK_RANGE_STABLE_WIN"] = (df["BLOCK_POSITION_SM"].rolling(stable_win, min_periods=1).max() - df["BLOCK_POSITION_SM"].rolling(stable_win, min_periods=1).min()).astype("float32")

# ============================================================
# 6. FLAGS AND DYNAMIC ON-BOTTOM DISTANCE
# ============================================================

df["PUMP_ON"] = (df["PUMP_OUTPUT_SM"] >= FLOW_ON) | (df["STANDPIPE_PRESSURE_SM"] >= SPP_ON)
df["RPM_ON"] = df["ROTARY_RPM_SM"] >= RPM_ON
df["WOB_ON"] = df["WOB_SM"] >= WOB_ON
df["ROP_ON"] = df["ROP_SM"] >= ROP_ON

df["BIT_TO_BOTTOM"] = (df["HOLE_DEPTH_SM"] - df["BIT_DEPTH_SM"]).astype("float32")

pump_like = df["PUMP_ON"]
drill_like = pump_like & (df["WOB_ON"] | df["ROP_ON"] | (df["HOLE_VEL_FTPM"] > HOLE_ADVANCE_FTPM))
btb = df.loc[drill_like, "BIT_TO_BOTTOM"].abs()
btb = btb[(btb >= 0) & (btb <= 300)]
if len(btb) > 100:
    ON_BOTTOM_DISTANCE = float(np.clip(btb.quantile(0.60), ON_BOTTOM_MIN_FT, ON_BOTTOM_MAX_FT))
else:
    ON_BOTTOM_DISTANCE = ON_BOTTOM_DEFAULT_FT

# Near bottom is looser than on bottom, mainly for connections.
df["ON_BOTTOM"] = df["BIT_TO_BOTTOM"].abs() <= ON_BOTTOM_DISTANCE
df["NEAR_BOTTOM_FOR_CONNECTION"] = df["BIT_TO_BOTTOM"].abs() <= NEAR_BOTTOM_FOR_CONNECTION_FT

# Movement flags use velocity OR sustained trend. This fixes missed trip-in/out.
df["BIT_MOVING_DOWN"] = (df["BIT_VEL_FTPM"] > BIT_MOVE_FTPM) | (df["BIT_TREND_FT"] > max(5.0, BIT_MOVE_FTPM * 2.0))
df["BIT_MOVING_UP"] = (df["BIT_VEL_FTPM"] < -BIT_MOVE_FTPM) | (df["BIT_TREND_FT"] < -max(5.0, BIT_MOVE_FTPM * 2.0))
df["BIT_MOVING"] = df["BIT_MOVING_DOWN"] | df["BIT_MOVING_UP"]
df["BLOCK_MOVING"] = (df["BLOCK_VEL_FTPM"].abs() > BLOCK_MOVE_FTPM) | (df["BLOCK_TREND_FT"].abs() > max(3.0, BLOCK_MOVE_FTPM * 2.0))
df["HOLE_ADVANCING"] = (df["HOLE_VEL_FTPM"] > HOLE_ADVANCE_FTPM) | (df["HOLE_TREND_FT"] > 0.5)

df["BIT_STABLE"] = df["BIT_RANGE_STABLE_WIN"] <= 15.0
df["HOLE_STABLE"] = df["HOLE_RANGE_STABLE_WIN"] <= 5.0
df["BLOCK_STABLE"] = df["BLOCK_RANGE_STABLE_WIN"] <= 5.0

threshold_report = pd.DataFrame([{
    "FSM_RESAMPLE_RULE": FSM_RESAMPLE_RULE,
    "median_dt_sec": med_dt,
    "velocity_window_samples": vel_win,
    "stable_window_samples": stable_win,
    "RPM_ON_rpm": RPM_ON,
    "FLOW_ON_gpm": FLOW_ON,
    "SPP_ON_psi": SPP_ON,
    "WOB_ON_klbs": WOB_ON,
    "ROP_ON_ft_hr": ROP_ON,
    "ON_BOTTOM_DISTANCE_ft": ON_BOTTOM_DISTANCE,
    "BIT_MOVE_ft_min": BIT_MOVE_FTPM,
    "BLOCK_MOVE_ft_min": BLOCK_MOVE_FTPM,
    "HOLE_ADVANCE_ft_min": HOLE_ADVANCE_FTPM,
}])

print("\n=== THRESHOLDS USED ===")
display(threshold_report)

# ============================================================
# 7. ROW-LEVEL FSM LOGIC
# ============================================================

off_bottom = ~df["ON_BOTTOM"]
low_wob_rop = (~df["WOB_ON"]) & (~df["ROP_ON"])

# Drilling first: on bottom, pump on, and cutting/advancing evidence.
drilling_rotate = df["ON_BOTTOM"] & df["PUMP_ON"] & df["RPM_ON"] & (df["WOB_ON"] | df["ROP_ON"] | df["HOLE_ADVANCING"])
drilling_slide = df["ON_BOTTOM"] & df["PUMP_ON"] & (~df["RPM_ON"]) & (df["WOB_ON"] | df["ROP_ON"] | df["HOLE_ADVANCING"])

# Off-bottom movement states.
reaming = off_bottom & df["PUMP_ON"] & df["RPM_ON"] & df["BIT_MOVING_DOWN"]
backreaming = off_bottom & df["PUMP_ON"] & df["RPM_ON"] & df["BIT_MOVING_UP"]
run_in_pump = off_bottom & df["PUMP_ON"] & (~df["RPM_ON"]) & df["BIT_MOVING_DOWN"]
pull_up_pump = off_bottom & df["PUMP_ON"] & (~df["RPM_ON"]) & df["BIT_MOVING_UP"]
run_in_rotate = off_bottom & (~df["PUMP_ON"]) & df["RPM_ON"] & df["BIT_MOVING_DOWN"]
pull_up_rotate = off_bottom & (~df["PUMP_ON"]) & df["RPM_ON"] & df["BIT_MOVING_UP"]
trip_in = off_bottom & (~df["PUMP_ON"]) & (~df["RPM_ON"]) & df["BIT_MOVING_DOWN"]
trip_out = off_bottom & (~df["PUMP_ON"]) & (~df["RPM_ON"]) & df["BIT_MOVING_UP"]

# Stationary active states.
circulating = df["PUMP_ON"] & (~df["RPM_ON"]) & (~df["BIT_MOVING"]) & (~drilling_slide)
rotate_circulate = df["PUMP_ON"] & df["RPM_ON"] & (~df["BIT_MOVING"]) & (~drilling_rotate)
rotate_only = (~df["PUMP_ON"]) & df["RPM_ON"] & (~df["BIT_MOVING"])

# Row-level connection evidence. Episode logic below is stronger.
connection_row = (
    df["NEAR_BOTTOM_FOR_CONNECTION"] &
    (~df["PUMP_ON"]) &
    (~df["RPM_ON"]) &
    (~df["BIT_MOVING"]) &
    low_wob_rop &
    (df["BIT_STABLE"] | df["HOLE_STABLE"]) &
    (df["BLOCK_MOVING"] | (~df["BLOCK_STABLE"]) | (df["HOOK_LOAD_SM"].diff().abs().rolling(stable_win, min_periods=1).max() > 2.0))
)

idle = (~df["PUMP_ON"]) & (~df["RPM_ON"]) & (~df["BIT_MOVING"]) & (~connection_row)

state = pd.Series("Unknown", index=df.index, dtype="object")

# Apply priority: drilling and exact off-bottom movement first.
state.loc[idle] = "Idle"
# `connection_row` is diagnostic evidence only. Final Connection labels are
# assigned below only after episode-level or stand-pattern validation.
state.loc[rotate_only] = "Rotate Only"
state.loc[circulating] = "Circulating"
state.loc[rotate_circulate] = "Rotate + Circulate"
state.loc[trip_in] = "Trip-In"
state.loc[trip_out] = "Trip-Out"
state.loc[run_in_pump] = "Run In + Pump"
state.loc[pull_up_pump] = "Pull Up + Pump"
state.loc[run_in_rotate] = "Run In + Rotate"
state.loc[pull_up_rotate] = "Pull Up + Rotate"
state.loc[reaming] = "Reaming"
state.loc[backreaming] = "Backreaming"
state.loc[drilling_slide] = "Drilling Slide"
state.loc[drilling_rotate] = "Drilling Rotate"

# Conservative fallback for remaining unknowns.
state.loc[state.eq("Unknown") & (~df["PUMP_ON"]) & (~df["RPM_ON"]) & (~df["BIT_MOVING"])] = "Idle"
state.loc[state.eq("Unknown") & df["PUMP_ON"] & (~df["RPM_ON"]) & (~df["BIT_MOVING"])] = "Circulating"
state.loc[state.eq("Unknown") & (~df["PUMP_ON"]) & df["RPM_ON"] & (~df["BIT_MOVING"])] = "Rotate Only"
state.loc[state.eq("Unknown") & df["PUMP_ON"] & df["RPM_ON"] & (~df["BIT_MOVING"])] = "Rotate + Circulate"
state.loc[state.eq("Unknown") & off_bottom & df["BIT_MOVING_DOWN"] & (~df["PUMP_ON"]) & (~df["RPM_ON"])] = "Trip-In"
state.loc[state.eq("Unknown") & off_bottom & df["BIT_MOVING_UP"] & (~df["PUMP_ON"]) & (~df["RPM_ON"])] = "Trip-Out"
state.loc[df["DATA_GAP"]] = "Data Gap"

df["RigState_raw"] = state

# ============================================================
# 8. EPISODE-LEVEL CONNECTION CORRECTION
# ============================================================

def _state_groups(state_series):
    g = (state_series != state_series.shift()).cumsum()
    return pd.DataFrame({"state": state_series, "group": g}).groupby("group").agg(
        State=("state", "first"),
        start_idx=("state", lambda x: int(x.index[0])),
        end_idx=("state", lambda x: int(x.index[-1])),
        samples=("state", "size")
    ).reset_index(drop=True)

def detect_connection_episodes(df_in):
    temp = df_in.reset_index(drop=True)
    drilling_mask = temp["RigState_raw"].isin(["Drilling Rotate", "Drilling Slide"])
    groups = _state_groups(drilling_mask.map({True: "Drilling", False: "NonDrilling"}))
    drill_groups = groups[groups["State"] == "Drilling"].reset_index(drop=True)

    accepted = []
    rows = []
    last_connection_depth = None

    for i in range(len(drill_groups) - 1):
        prev_e = int(drill_groups.loc[i, "end_idx"])
        next_s = int(drill_groups.loc[i + 1, "start_idx"])
        gap_s = prev_e + 1
        gap_e = next_s - 1
        if gap_e <= gap_s:
            continue

        gap = temp.loc[gap_s:gap_e].copy()
        if gap.empty:
            continue

        dur_sec = max((gap["FSM_TIMESTAMP"].iloc[-1] - gap["FSM_TIMESTAMP"].iloc[0]).total_seconds(), 0)
        hole_range = float(gap["HOLE_DEPTH_SM"].max() - gap["HOLE_DEPTH_SM"].min())
        bit_range = float(gap["BIT_DEPTH_SM"].max() - gap["BIT_DEPTH_SM"].min())
        block_range = float(gap["BLOCK_POSITION_SM"].max() - gap["BLOCK_POSITION_SM"].min())
        hook_range = float(gap["HOOK_LOAD_SM"].max() - gap["HOOK_LOAD_SM"].min())
        near_bottom_frac = float(gap["NEAR_BOTTOM_FOR_CONNECTION"].mean())
        pump_off_frac = float((~gap["PUMP_ON"]).mean())
        rpm_off_frac = float((~gap["RPM_ON"]).mean())
        low_wob_rop_frac = float(((~gap["WOB_ON"]) & (~gap["ROP_ON"])).mean())
        bit_stable_frac = float((~gap["BIT_MOVING"]).mean())

        prev_drill = temp.loc[int(drill_groups.loc[i, "start_idx"]):prev_e]
        drilled_since_last = float(prev_drill["HOLE_DEPTH_SM"].iloc[-1] - prev_drill["HOLE_DEPTH_SM"].iloc[0]) if len(prev_drill) else 0.0

        if last_connection_depth is None:
            stand_evidence = abs(drilled_since_last - STAND_LENGTH_FT) <= STAND_TOLERANCE_FT or drilled_since_last >= 45.0
        else:
            current_depth = float(gap["HOLE_DEPTH_SM"].median())
            stand_evidence = abs((current_depth - last_connection_depth) - STAND_LENGTH_FT) <= STAND_TOLERANCE_FT

        accepted_connection = (
            MIN_CONNECTION_SEC <= dur_sec <= MAX_CONNECTION_SEC and
            near_bottom_frac >= 0.50 and
            pump_off_frac >= 0.60 and
            rpm_off_frac >= 0.60 and
            low_wob_rop_frac >= 0.50 and
            bit_stable_frac >= 0.50 and
            hole_range <= 12.0 and
            bit_range <= 60.0 and
            (block_range >= 3.0 or hook_range >= 2.0 or stand_evidence)
        )

        if accepted_connection:
            accepted.append((gap_s, gap_e))
            last_connection_depth = float(gap["HOLE_DEPTH_SM"].median())

        rows.append({
            "start_idx": gap_s,
            "end_idx": gap_e,
            "start_time": gap["FSM_TIMESTAMP"].iloc[0],
            "end_time": gap["FSM_TIMESTAMP"].iloc[-1],
            "duration_sec": dur_sec,
            "duration_min": dur_sec / 60.0,
            "hole_range_ft": hole_range,
            "bit_range_ft": bit_range,
            "block_range_ft": block_range,
            "hookload_range_klbs": hook_range,
            "near_bottom_frac": near_bottom_frac,
            "pump_off_frac": pump_off_frac,
            "rpm_off_frac": rpm_off_frac,
            "low_wob_rop_frac": low_wob_rop_frac,
            "bit_stable_frac": bit_stable_frac,
            "drilled_since_previous_drilling_start_ft": drilled_since_last,
            "stand_evidence": stand_evidence,
            "accepted_connection": accepted_connection,
        })

    return accepted, pd.DataFrame(rows)

connection_intervals, connection_candidates_df = detect_connection_episodes(df)

# ============================================================
# 8B. EPISODE-LEVEL MOVING-PIPE CORRECTION
# ============================================================
# Purpose:
#   The row-level logic can miss parts of long trip/ream passes because a few rows
#   temporarily look stable, on-bottom, or noisy after resampling/smoothing.
#   This correction groups sustained bit-depth movement episodes and classifies the
#   whole episode by dominant pump/RPM status.
#
# Rules:
#   Trip-In       = bit depth moving down, off bottom, pumps off, rotary off
#   Trip-Out      = bit depth moving up,   off bottom, pumps off, rotary off
#   Reaming       = bit depth moving down, off bottom, pumps on,  rotary on
#   Backreaming   = bit depth moving up,   off bottom, pumps on,  rotary on

def detect_moving_pipe_episodes(df_in):
    temp = df_in.reset_index(drop=True).copy()

    drilling_raw = temp["RigState_raw"].isin(["Drilling Rotate", "Drilling Slide"])

    bridge_n = max(2, int(round(180.0 / max(med_dt, 1.0))))  # about 3 minutes
    move_core = (~drilling_raw) & temp["BIT_MOVING"]

    move_bridge = (
        move_core.astype("int8")
        .rolling(bridge_n, center=True, min_periods=1)
        .max()
        .astype(bool)
    )

    candidate = move_bridge & (~drilling_raw)
    run_id = (candidate != candidate.shift()).cumsum()

    accepted = []
    rows = []

    for _, idx in temp[candidate].groupby(run_id[candidate]).groups.items():
        idx = list(idx)
        if not idx:
            continue

        s = int(min(idx))
        e = int(max(idx))
        ep = temp.loc[s:e].copy()

        duration_sec = max((ep["FSM_TIMESTAMP"].iloc[-1] - ep["FSM_TIMESTAMP"].iloc[0]).total_seconds(), 0.0)
        bit_change = float(ep["BIT_DEPTH_SM"].iloc[-1] - ep["BIT_DEPTH_SM"].iloc[0])
        bit_range = float(ep["BIT_DEPTH_SM"].max() - ep["BIT_DEPTH_SM"].min())
        hole_range = float(ep["HOLE_DEPTH_SM"].max() - ep["HOLE_DEPTH_SM"].min())

        if duration_sec < MIN_MOVING_EPISODE_SEC or bit_range < 40.0:
            accepted_episode = False
            state_name = "Rejected"
        else:
            direction_down = bit_change >= 0
            off_bottom_frac = float((~ep["ON_BOTTOM"]).mean())
            hole_advancing_frac = float(ep["HOLE_ADVANCING"].mean())
            pump_frac = float(ep["PUMP_ON"].mean())
            rpm_frac = float(ep["RPM_ON"].mean())
            wob_rop_frac = float((ep["WOB_ON"] | ep["ROP_ON"]).mean())

            movement_valid = (
                bit_range >= 40.0
                and off_bottom_frac >= 0.35
                and hole_advancing_frac <= 0.40
                and wob_rop_frac <= 0.75
            )

            if pump_frac >= 0.35 and rpm_frac >= 0.35:
                state_name = "Reaming" if direction_down else "Backreaming"
            elif pump_frac >= 0.35 and rpm_frac < 0.35:
                state_name = "Run In + Pump" if direction_down else "Pull Up + Pump"
            elif pump_frac < 0.35 and rpm_frac >= 0.35:
                state_name = "Run In + Rotate" if direction_down else "Pull Up + Rotate"
            else:
                state_name = "Trip-In" if direction_down else "Trip-Out"

            accepted_episode = bool(movement_valid)

        if accepted_episode:
            accepted.append((s, e, state_name))

        rows.append({
            "start_idx": s,
            "end_idx": e,
            "start_time": ep["FSM_TIMESTAMP"].iloc[0],
            "end_time": ep["FSM_TIMESTAMP"].iloc[-1],
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "bit_change_ft": bit_change,
            "bit_range_ft": bit_range,
            "hole_range_ft": hole_range,
            "off_bottom_frac": float((~ep["ON_BOTTOM"]).mean()),
            "pump_on_frac": float(ep["PUMP_ON"].mean()),
            "rpm_on_frac": float(ep["RPM_ON"].mean()),
            "hole_advancing_frac": float(ep["HOLE_ADVANCING"].mean()),
            "assigned_state": state_name,
            "accepted": accepted_episode,
        })

    return accepted, pd.DataFrame(rows)

moving_pipe_intervals, moving_pipe_candidates_df = detect_moving_pipe_episodes(df)

df["RigState"] = df["RigState_raw"].copy()

for s, e, state_name in moving_pipe_intervals:
    local = df.loc[s:e].copy()
    not_drilling = ~local["RigState_raw"].isin(["Drilling Rotate", "Drilling Slide"])
    df.loc[local.index[not_drilling], "RigState"] = state_name

for s, e in connection_intervals:
    df.loc[s:e, "RigState"] = "Connection"

# ============================================================
# 8B. STAND-PATTERN CONNECTION CORRECTION
# ============================================================
# This pass is specifically for the situation where drilling footage clearly
# implies several stand connections, but the row-level classifier only caught
# one or two. It does not blindly force every 90 ft target to be a connection.
# It searches around each expected stand depth and requires a connection-like
# parameter pattern: pump/RPM/WOB drop, hole/bit stability, and block/hook motion.

def detect_stand_pattern_connections(df_in):
    temp = df_in.reset_index(drop=True).copy()

    if not ENABLE_STAND_PATTERN_CONNECTIONS:
        return [], pd.DataFrame()

    # Core drilling evidence. Use this rather than final state only, because
    # missed connection rows break drilling into many small pieces.
    drilling_core = (
        temp["NEAR_BOTTOM_FOR_CONNECTION"] &
        temp["PUMP_ON"] &
        (temp["HOLE_ADVANCING"] | temp["WOB_ON"] | temp["ROP_ON"] | temp["RigState"].isin(["Drilling Rotate", "Drilling Slide"]))
    )

    drill_groups = []
    groups = _state_groups(drilling_core.map({True: "Drilling", False: "NonDrilling"}))
    raw_drill_groups = groups[groups["State"].eq("Drilling")].reset_index(drop=True)

    # Merge drilling blocks separated by short non-trip gaps. This forms one
    # practical drilling interval where stand-by-stand connections are expected.
    current = None
    for _, row in raw_drill_groups.iterrows():
        s0, e0 = int(row["start_idx"]), int(row["end_idx"])
        if current is None:
            current = [s0, e0]
            continue

        gap = temp.loc[current[1] + 1:s0 - 1]
        gap_min = 0.0
        if not gap.empty:
            gap_min = (gap["FSM_TIMESTAMP"].iloc[-1] - gap["FSM_TIMESTAMP"].iloc[0]).total_seconds() / 60.0
        gap_has_trip = bool(gap["RigState"].isin(["Trip-In", "Trip-Out", "Run In + Pump", "Pull Up + Pump", "Run In + Rotate", "Pull Up + Rotate"]).mean() > 0.30) if not gap.empty else False

        if gap_min <= MAX_DRILL_SESSION_GAP_MIN and not gap_has_trip:
            current[1] = e0
        else:
            drill_groups.append(tuple(current))
            current = [s0, e0]

    if current is not None:
        drill_groups.append(tuple(current))

    # Rolling evidence windows for the connection signature.
    score_win = max(3, int(round(90.0 / max(med_dt, 1.0))))       # about 1.5 min
    expand_win = max(2, int(round(90.0 / max(med_dt, 1.0))))      # about 1.5 min

    block_range = temp["BLOCK_POSITION_SM"].rolling(score_win, center=True, min_periods=1).max() - temp["BLOCK_POSITION_SM"].rolling(score_win, center=True, min_periods=1).min()
    hook_range = temp["HOOK_LOAD_SM"].rolling(score_win, center=True, min_periods=1).max() - temp["HOOK_LOAD_SM"].rolling(score_win, center=True, min_periods=1).min()

    # IMPORTANT FIX:
    # Do not use Python max() with a pandas Series. That raises:
    # "ValueError: The truth value of a Series is ambiguous".
    # np.maximum() compares element-by-element and returns a Series-compatible array.
    flow_ref = temp["PUMP_OUTPUT_SM"].rolling(score_win, center=True, min_periods=1).median() * 0.45
    rpm_ref = temp["ROTARY_RPM_SM"].rolling(score_win, center=True, min_periods=1).median() * 0.45
    wob_ref = temp["WOB_SM"].rolling(score_win, center=True, min_periods=1).median() * 0.45

    flow_drop = temp["PUMP_OUTPUT_SM"] <= np.maximum(FLOW_ON, flow_ref)
    rpm_drop = temp["ROTARY_RPM_SM"] <= np.maximum(RPM_ON, rpm_ref)
    wob_drop = temp["WOB_SM"] <= np.maximum(WOB_ON, wob_ref)

    temp["CONNECTION_PATTERN_SCORE"] = (
        temp["NEAR_BOTTOM_FOR_CONNECTION"].astype(int) +
        (~temp["HOLE_ADVANCING"]).astype(int) +
        (~temp["BIT_MOVING"]).astype(int) +
        flow_drop.astype(int) +
        rpm_drop.astype(int) +
        wob_drop.astype(int) +
        ((block_range >= 3.0) | (hook_range >= 2.0) | temp["BLOCK_MOVING"]).astype(int)
    )

    accepted = []
    rows = []
    used = np.zeros(len(temp), dtype=bool)

    for session_id, (session_s, session_e) in enumerate(drill_groups, start=1):
        sess = temp.loc[session_s:session_e]
        if sess.empty:
            continue

        start_depth = float(sess["HOLE_DEPTH_SM"].min())
        end_depth = float(sess["HOLE_DEPTH_SM"].max())
        drilled_ft = end_depth - start_depth

        if drilled_ft < MIN_DRILL_SESSION_FOOTAGE_FT:
            continue

        expected_count = int(np.floor(drilled_ft / STAND_LENGTH_FT))
        for n in range(1, expected_count + 1):
            target_depth = start_depth + n * STAND_LENGTH_FT

            depth_near = temp["HOLE_DEPTH_SM"].between(target_depth - CONNECTION_DEPTH_TOL_FT, target_depth + CONNECTION_DEPTH_TOL_FT)
            in_session = (temp.index >= session_s) & (temp.index <= session_e)
            not_moving_pipe = ~temp["RigState"].isin(["Trip-In", "Trip-Out", "Reaming", "Backreaming", "Run In + Pump", "Pull Up + Pump", "Run In + Rotate", "Pull Up + Rotate"])
            win = temp.loc[in_session & depth_near & not_moving_pipe]

            if win.empty:
                rows.append({
                    "session_id": session_id, "expected_no": n, "target_depth_ft": target_depth,
                    "accepted_connection": False, "reason": "no rows near expected stand depth"
                })
                continue

            # Prefer rows with strong score. If none, accept weak score only when
            # there is a clear pump/RPM/WOB drop and block/hook activity.
            strong = win[win["CONNECTION_PATTERN_SCORE"] >= MIN_CONNECTION_PATTERN_SCORE]
            weak = win[
                (win["CONNECTION_PATTERN_SCORE"] >= MIN_WEAK_CONNECTION_PATTERN_SCORE) &
                ((~win["PUMP_ON"]) | (~win["RPM_ON"]) | (~win["WOB_ON"]))
            ]

            if not strong.empty:
                cand = strong
                reason = "strong parameter-drop pattern near expected stand depth"
            elif not weak.empty:
                cand = weak
                reason = "weak but plausible parameter-drop pattern near expected stand depth"
            else:
                rows.append({
                    "session_id": session_id, "expected_no": n, "target_depth_ft": target_depth,
                    "accepted_connection": False, "reason": "expected stand depth found but no connection pattern",
                    "max_score": int(win["CONNECTION_PATTERN_SCORE"].max()),
                    "window_start_time": win["FSM_TIMESTAMP"].iloc[0],
                    "window_end_time": win["FSM_TIMESTAMP"].iloc[-1],
                    "window_min_depth_ft": float(win["HOLE_DEPTH_SM"].min()),
                    "window_max_depth_ft": float(win["HOLE_DEPTH_SM"].max()),
                })
                continue

            peak_idx = int(cand["CONNECTION_PATTERN_SCORE"].idxmax())
            s_idx = max(int(win.index.min()), peak_idx - expand_win)
            e_idx = min(int(win.index.max()), peak_idx + expand_win)

            # Expand through adjacent high-score rows inside the same depth window.
            local_idx = win.index.to_numpy()
            high_idx = win.index[(win["CONNECTION_PATTERN_SCORE"] >= MIN_WEAK_CONNECTION_PATTERN_SCORE)].to_numpy()
            if len(high_idx):
                near_high = high_idx[(high_idx >= s_idx) & (high_idx <= e_idx)]
                if len(near_high):
                    s_idx = int(min(s_idx, near_high.min()))
                    e_idx = int(max(e_idx, near_high.max()))

            # Do not duplicate an existing accepted connection at the same rows.
            overlap = bool(used[s_idx:e_idx + 1].any())
            already_connection_frac = float(temp.loc[s_idx:e_idx, "RigState"].eq("Connection").mean())

            accepted_connection = not overlap
            if accepted_connection:
                used[s_idx:e_idx + 1] = True
                accepted.append((s_idx, e_idx))

            seg = temp.loc[s_idx:e_idx]
            rows.append({
                "session_id": session_id,
                "expected_no": n,
                "target_depth_ft": target_depth,
                "accepted_connection": bool(accepted_connection),
                "reason": reason if accepted_connection else "duplicate/overlap with another stand connection",
                "start_idx": s_idx,
                "end_idx": e_idx,
                "start_time": seg["FSM_TIMESTAMP"].iloc[0],
                "end_time": seg["FSM_TIMESTAMP"].iloc[-1],
                "duration_sec": float((seg["FSM_TIMESTAMP"].iloc[-1] - seg["FSM_TIMESTAMP"].iloc[0]).total_seconds()),
                "window_min_depth_ft": float(seg["HOLE_DEPTH_SM"].min()),
                "window_max_depth_ft": float(seg["HOLE_DEPTH_SM"].max()),
                "max_score": int(seg["CONNECTION_PATTERN_SCORE"].max()),
                "pump_off_frac": float((~seg["PUMP_ON"]).mean()),
                "rpm_off_frac": float((~seg["RPM_ON"]).mean()),
                "wob_off_frac": float((~seg["WOB_ON"]).mean()),
                "block_range_ft": float(seg["BLOCK_POSITION_SM"].max() - seg["BLOCK_POSITION_SM"].min()),
                "hookload_range_klbs": float(seg["HOOK_LOAD_SM"].max() - seg["HOOK_LOAD_SM"].min()),
                "already_connection_frac": already_connection_frac,
                "session_start_depth_ft": start_depth,
                "session_end_depth_ft": end_depth,
                "session_drilled_ft": drilled_ft,
                "session_expected_connections": expected_count,
            })

    return accepted, pd.DataFrame(rows)

stand_connection_intervals, stand_connection_candidates_df = detect_stand_pattern_connections(df)

for s, e in stand_connection_intervals:
    local = df.loc[s:e]
    # Do not overwrite true trips/reaming, but do overwrite missed idle/circulating/rotate-only gaps.
    overwrite = ~local["RigState"].isin(["Trip-In", "Trip-Out", "Reaming", "Backreaming", "Run In + Pump", "Pull Up + Pump", "Run In + Rotate", "Pull Up + Rotate"])
    df.loc[local.index[overwrite], "RigState"] = "Connection"

# ============================================================
# 8C. FINAL SEMANTIC RECONCILIATION OF OFF-BOTTOM MOVEMENT
# ============================================================
# Episode detection deliberately uses dominant fractions to stabilize long
# moving-pipe episodes. At row level, however, the final label must still agree
# with the contemporaneous pump/RPM combination. The scope includes both rows
# still flagged as moving and stationary transition rows that inherited an
# episode-level movement label. This prevents stale Trip/Reaming semantics at
# pump or rotary transitions.
MOVEMENT_STATES = {
    "Trip-In", "Trip-Out", "Reaming", "Backreaming",
    "Run In + Pump", "Pull Up + Pump",
    "Run In + Rotate", "Pull Up + Rotate",
}
DOWN_STATES = {"Trip-In", "Reaming", "Run In + Pump", "Run In + Rotate"}
UP_STATES = {"Trip-Out", "Backreaming", "Pull Up + Pump", "Pull Up + Rotate"}

def reconcile_movement_semantics(frame):
    before = frame["RigState"].copy()
    valid = ~frame["DATA_GAP"]
    actual_movement = valid & (~frame["ON_BOTTOM"]) & frame["BIT_MOVING"]
    # Inherited episode labels must be checked even when the instantaneous
    # on-bottom or movement flag has changed. Otherwise a few transition rows
    # can retain Trip while pumps/RPM are already on.
    inherited_movement = valid & frame["RigState"].isin(MOVEMENT_STATES)
    movement_family_scope = actual_movement | inherited_movement

    direction_down = actual_movement & frame["BIT_MOVING_DOWN"] & (~frame["BIT_MOVING_UP"])
    direction_up = actual_movement & frame["BIT_MOVING_UP"] & (~frame["BIT_MOVING_DOWN"])

    # Keep the episode direction for stationary transition rows. For rare
    # simultaneous/ambiguous movement flags, use the signed smoothed velocity.
    unresolved = movement_family_scope & ~(direction_down | direction_up)
    direction_down = direction_down | (
        unresolved &
        (
            frame["RigState"].isin(DOWN_STATES) |
            (~frame["RigState"].isin(UP_STATES) & frame["BIT_VEL_FTPM"].ge(0))
        )
    )
    direction_up = direction_up | (unresolved & ~direction_down)

    moving = actual_movement
    frame.loc[moving & direction_down & frame["PUMP_ON"] & frame["RPM_ON"], "RigState"] = "Reaming"
    frame.loc[moving & direction_up & frame["PUMP_ON"] & frame["RPM_ON"], "RigState"] = "Backreaming"
    frame.loc[moving & direction_down & frame["PUMP_ON"] & ~frame["RPM_ON"], "RigState"] = "Run In + Pump"
    frame.loc[moving & direction_up & frame["PUMP_ON"] & ~frame["RPM_ON"], "RigState"] = "Pull Up + Pump"
    frame.loc[moving & direction_down & ~frame["PUMP_ON"] & frame["RPM_ON"], "RigState"] = "Run In + Rotate"
    frame.loc[moving & direction_up & ~frame["PUMP_ON"] & frame["RPM_ON"], "RigState"] = "Pull Up + Rotate"
    frame.loc[moving & direction_down & ~frame["PUMP_ON"] & ~frame["RPM_ON"], "RigState"] = "Trip-In"
    frame.loc[moving & direction_up & ~frame["PUMP_ON"] & ~frame["RPM_ON"], "RigState"] = "Trip-Out"

    # A stationary transition row cannot retain a movement-state label whose
    # required systems are inconsistent with the current measurements.
    transition_inherited = inherited_movement & (~actual_movement)
    drilling_evidence = frame["ON_BOTTOM"] & (
        frame["WOB_ON"] | frame["ROP_ON"] | frame["HOLE_ADVANCING"]
    )
    frame.loc[
        transition_inherited & frame["PUMP_ON"] & frame["RPM_ON"] & drilling_evidence,
        "RigState"
    ] = "Drilling Rotate"
    frame.loc[
        transition_inherited & frame["PUMP_ON"] & ~frame["RPM_ON"] & drilling_evidence,
        "RigState"
    ] = "Drilling Slide"
    frame.loc[
        transition_inherited & frame["PUMP_ON"] & frame["RPM_ON"] & ~drilling_evidence,
        "RigState"
    ] = "Rotate + Circulate"
    frame.loc[
        transition_inherited & frame["PUMP_ON"] & ~frame["RPM_ON"] & ~drilling_evidence,
        "RigState"
    ] = "Circulating"
    frame.loc[transition_inherited & ~frame["PUMP_ON"] & frame["RPM_ON"], "RigState"] = "Rotate Only"
    frame.loc[
        transition_inherited &
        ~frame["PUMP_ON"] &
        ~frame["RPM_ON"] &
        frame["RigState"].isin(["Reaming", "Backreaming"]),
        "RigState"
    ] = "Idle"

    return int((before != frame["RigState"]).sum()), int(movement_family_scope.sum())

semantic_rows_reconciled, semantic_rows_checked = reconcile_movement_semantics(df)

# ============================================================
# 9. SMOOTHING WITHOUT DESTROYING TRIPS / REAMING / CONNECTIONS
# ============================================================

PROTECTED_STATES = {
    "Trip-In", "Trip-Out", "Reaming", "Backreaming", "Run In + Pump", "Pull Up + Pump",
    "Run In + Rotate", "Pull Up + Rotate", "Connection", "Drilling Rotate", "Drilling Slide",
    "Data Gap"
}

def smooth_only_unprotected_short_groups(states, sample_dt_sec, min_group_sec=20, passes=2):
    s = states.copy().reset_index(drop=True)
    min_size = max(2, int(round(min_group_sec / max(sample_dt_sec, 1.0))))

    for _ in range(passes):
        info = _state_groups(s)
        for i in range(1, len(info) - 1):
            curr = info.iloc[i]
            prev_state = info.iloc[i - 1]["State"]
            next_state = info.iloc[i + 1]["State"]
            curr_state = curr["State"]

            if curr_state in PROTECTED_STATES:
                continue
            if int(curr["samples"]) <= min_size and prev_state == next_state and prev_state != curr_state:
                s.iloc[int(curr["start_idx"]):int(curr["end_idx"]) + 1] = prev_state
    return s

df["RigState"] = smooth_only_unprotected_short_groups(df["RigState"], med_dt, MIN_STATE_GROUP_SEC, passes=2).values

# Smoothing may copy a neighboring state into a short transition group. Re-run
# the semantic finalizer so exported labels and QC describe the same final data.
rows_reconciled_after_smoothing, rows_checked_after_smoothing = reconcile_movement_semantics(df)
semantic_reconciliation_report = pd.DataFrame([{
    "rows_reconciled": int(semantic_rows_reconciled + rows_reconciled_after_smoothing),
    "movement_rows_checked": int(max(semantic_rows_checked, rows_checked_after_smoothing)),
    "rows_reconciled_after_smoothing": int(rows_reconciled_after_smoothing),
    "trip_rows_with_pump_on_after": int(
        (df["RigState"].isin(["Trip-In", "Trip-Out"]) & df["PUMP_ON"]).sum()
    ),
    "trip_rows_with_rpm_on_after": int(
        (df["RigState"].isin(["Trip-In", "Trip-Out"]) & df["RPM_ON"]).sum()
    ),
    "ream_rows_with_pump_off_after": int(
        (df["RigState"].isin(["Reaming", "Backreaming"]) & ~df["PUMP_ON"]).sum()
    ),
    "ream_rows_with_rpm_off_after": int(
        (df["RigState"].isin(["Reaming", "Backreaming"]) & ~df["RPM_ON"]).sum()
    ),
}])

df["PrevState"] = df["RigState"].shift().fillna("Start")
df["StateGroup"] = ((df["RigState"] != df["RigState"].shift()) | df["DATA_GAP"]).cumsum()

# ============================================================
# 10. OUTPUT TABLES
# ============================================================

state_counts = df["RigState"].value_counts().reset_index()
state_counts.columns = ["RigState", "Count"]
state_counts["Percentage"] = (state_counts["Count"] / len(df) * 100).round(2)
state_counts["Time_hr"] = (state_counts["Count"] * med_dt / 3600.0).round(2)

state_intervals = (
    df.groupby("StateGroup")
      .agg(
          Start_Time=("FSM_TIMESTAMP", "first"),
          End_Time=("FSM_TIMESTAMP", "last"),
          RigState=("RigState", "first"),
          Samples=("RigState", "size"),
          Start_Bit_Depth=("BIT_DEPTH_SM", "first"),
          End_Bit_Depth=("BIT_DEPTH_SM", "last"),
          Start_Hole_Depth=("HOLE_DEPTH_SM", "first"),
          End_Hole_Depth=("HOLE_DEPTH_SM", "last"),
          Mean_WOB=("WOB_SM", "mean"),
          Mean_ROP=("ROP_SM", "mean"),
          Mean_RPM=("ROTARY_RPM_SM", "mean"),
          Mean_Flow=("PUMP_OUTPUT_SM", "mean"),
          Mean_SPP=("STANDPIPE_PRESSURE_SM", "mean"),
          Mean_Torque=("ROTARY_TORQUE_SM", "mean")
      )
      .reset_index(drop=True)
)
state_intervals["Duration_sec"] = (state_intervals["End_Time"] - state_intervals["Start_Time"]).dt.total_seconds().fillna(0)
state_intervals["Duration_min"] = (state_intervals["Duration_sec"] / 60.0).round(2)
state_intervals["Bit_Depth_Change_ft"] = (state_intervals["End_Bit_Depth"] - state_intervals["Start_Bit_Depth"]).round(2)
state_intervals["Hole_Depth_Change_ft"] = (state_intervals["End_Hole_Depth"] - state_intervals["Start_Hole_Depth"]).round(2)

condition_counts = pd.DataFrame([
    ["Drilling Rotate raw mask", int(drilling_rotate.sum())],
    ["Drilling Slide raw mask", int(drilling_slide.sum())],
    ["Reaming raw mask", int(reaming.sum())],
    ["Backreaming raw mask", int(backreaming.sum())],
    ["Run In + Pump raw mask", int(run_in_pump.sum())],
    ["Pull Up + Pump raw mask", int(pull_up_pump.sum())],
    ["Run In + Rotate raw mask", int(run_in_rotate.sum())],
    ["Pull Up + Rotate raw mask", int(pull_up_rotate.sum())],
    ["Trip-In raw mask", int(trip_in.sum())],
    ["Trip-Out raw mask", int(trip_out.sum())],
    ["Circulating raw mask", int(circulating.sum())],
    ["Rotate + Circulate raw mask", int(rotate_circulate.sum())],
    ["Rotate Only raw mask", int(rotate_only.sum())],
    ["Connection row mask", int(connection_row.sum())],
    ["Episode-level accepted connections", int(len(connection_intervals))],
    ["Stand-pattern accepted connections", int(len(stand_connection_intervals))],
    ["Idle raw mask", int(idle.sum())],
], columns=["Condition", "Samples"])

unknown_pct = float(state_counts.loc[state_counts["RigState"] == "Unknown", "Percentage"].sum()) if "Unknown" in state_counts["RigState"].values else 0.0

fsm_qc = pd.DataFrame([
    ["Input dataframe", input_name, "", "Source dataframe used by reference, not copied."],
    ["Stage 2 valid-row coverage before FSM filtering", stage2_valid_pct, ">=95%", "Rows failing the Stage 2 quality gate are excluded before FSM resampling."],
    ["Rows after compact build/resample", len(df), "Memory-safe", "Lower row count prevents computer/browser crash."],
    ["FSM resample rule", str(FSM_RESAMPLE_RULE), "10s default", "Set to 5s for more detail; None only on strong machine."],
    ["Median sample interval sec", med_dt, "", "Used for time-normalized durations."],
    ["Acquisition/quality gaps split from state episodes", int(df["DATA_GAP"].sum()), "Reported, not bridged", "Prevents excluded or missing time from becoming a false connection/trip/state duration."],
    ["Unknown state percentage", unknown_pct, "<5% preferred", "High value means thresholds still need tuning."],
    ["Trip samples with pump on", int(df[df["RigState"].isin(["Trip-In", "Trip-Out"])]["PUMP_ON"].sum()), "0", "Pure trip should have pumps off."],
    ["Trip samples with RPM on", int(df[df["RigState"].isin(["Trip-In", "Trip-Out"])]["RPM_ON"].sum()), "0", "Pure trip should have rotation off."],
    ["Ream/backream samples pump off", int((~df[df["RigState"].isin(["Reaming", "Backreaming"])]["PUMP_ON"]).sum()), "0", "Reaming/backreaming should have flow."],
    ["Ream/backream samples RPM off", int((~df[df["RigState"].isin(["Reaming", "Backreaming"])]["RPM_ON"]).sum()), "0", "Reaming/backreaming should have rotation."],
    ["Accepted episode connections", len(connection_intervals), "Dataset-dependent", "Drilling-gap connection logic."],
    ["Accepted stand-pattern connections", len(stand_connection_intervals), "Dataset-dependent", "Expected stand-depth + parameter-drop pattern logic."],
    ["Classical Teale MSE coverage", float(df["MSE_TEALE_KSI_SM"].notna().mean() * 100.0), "Drilling-active rows", "MSE is intentionally sparse outside valid drilling conditions."],
    ["Hydraulic MSE coverage", float(df["MSE_HYDRAULIC_KSI_SM"].notna().mean() * 100.0), "Drilling-active rows", "Hydraulic MSE is intentionally sparse outside valid drilling conditions."],
], columns=["Check", "Value", "Target", "Interpretation"])

print("\n=== RIG STATE COUNTS ===")
display(state_counts)
print("\n=== FSM CONDITION COUNTS ===")
display(condition_counts)
print("\n=== STATE INTERVALS PREVIEW ===")
display(state_intervals.head(30))
print("\n=== FSM QC CHECKS ===")
display(fsm_qc)
print("\n=== FINAL MOVEMENT-STATE SEMANTIC RECONCILIATION ===")
display(semantic_reconciliation_report)
print("\n=== CONNECTION CANDIDATES PREVIEW ===")
display(connection_candidates_df.head(50))

print("\n=== STAND-PATTERN CONNECTION CANDIDATES PREVIEW ===")
display(stand_connection_candidates_df.head(80))

# ============================================================
# 11. ACTIVE WINDOWS FOR PLOTTING
# ============================================================

def downsample_for_plot(plot_df, max_points):
    if len(plot_df) <= max_points:
        return plot_df.copy()
    step = int(np.ceil(len(plot_df) / max_points))
    return plot_df.iloc[::step].copy()

def find_active_windows(df_in, hours=8, top_n=10):
    temp = df_in[["FSM_TIMESTAMP", "PUMP_ON", "RPM_ON", "WOB_ON", "ROP_ON", "BIT_MOVING", "HOLE_ADVANCING"]].copy()
    temp["activity"] = temp[["PUMP_ON", "RPM_ON", "WOB_ON", "ROP_ON", "BIT_MOVING", "HOLE_ADVANCING"]].astype(int).sum(axis=1)
    minute_score = temp.set_index("FSM_TIMESTAMP")["activity"].resample("1min").sum().fillna(0)
    rolling_score = minute_score.rolling(f"{hours}h", min_periods=30).sum()
    if rolling_score.dropna().empty:
        start = df_in["FSM_TIMESTAMP"].min()
        return pd.DataFrame([{"Start": start, "End": start + pd.Timedelta(hours=hours), "Activity Score": 0}])
    top = rolling_score.sort_values(ascending=False).head(top_n)
    return pd.DataFrame([{"Start": end_t - pd.Timedelta(hours=hours), "End": end_t, "Activity Score": score} for end_t, score in top.items()])

active_windows = find_active_windows(df, hours=ACTIVE_WINDOW_HOURS, top_n=10)
auto_start = active_windows.iloc[0]["Start"]
auto_end = active_windows.iloc[0]["End"]

print("\n=== SUGGESTED ACTIVE WINDOWS FOR PLOTTING ===")
display(active_windows)
print("Automatic active plotting window:", auto_start, "to", auto_end)

# ============================================================
# 12. VISUALIZATIONS
# ============================================================

STATE_COLORS = {
    "Idle": "#d62728",
    "Connection": "#1f77b4",
    "Drilling Rotate": "#2ca02c",
    "Drilling Slide": "#ff7f0e",
    "Reaming": "#e377c2",
    "Backreaming": "#8c564b",
    "Trip-In": "#9467bd",
    "Trip-Out": "#17becf",
    "Run In + Pump": "#7b3294",
    "Pull Up + Pump": "#c2a5cf",
    "Run In + Rotate": "#008837",
    "Pull Up + Rotate": "#a6dba0",
    "Circulating": "#7f3c8d",
    "Rotate + Circulate": "#4daf4a",
    "Rotate Only": "#bcbd22",
    "Data Gap": "#4b5563",
    "Unknown": "#8c8c8c",
}

def downsample_for_plot(plot_df, max_points):
    if len(plot_df) <= max_points:
        return plot_df.copy()
    step = int(np.ceil(len(plot_df) / max_points))
    return plot_df.iloc[::step].copy()

def find_active_windows(df_in, hours=8, top_n=10):
    temp = df_in[["FSM_TIMESTAMP", "PUMP_ON", "RPM_ON", "WOB_ON", "ROP_ON", "BIT_MOVING", "HOLE_ADVANCING"]].copy()
    temp["activity"] = temp[["PUMP_ON", "RPM_ON", "WOB_ON", "ROP_ON", "BIT_MOVING", "HOLE_ADVANCING"]].astype(int).sum(axis=1)
    minute_score = temp.set_index("FSM_TIMESTAMP")["activity"].resample("1min").sum().fillna(0)
    rolling_score = minute_score.rolling(f"{hours}h", min_periods=30).sum()
    if rolling_score.dropna().empty:
        start = df_in["FSM_TIMESTAMP"].min()
        return pd.DataFrame([{"Start": start, "End": start + pd.Timedelta(hours=hours), "Activity Score": 0}])
    top = rolling_score.sort_values(ascending=False).head(top_n)
    return pd.DataFrame([{"Start": end_t - pd.Timedelta(hours=hours), "End": end_t, "Activity Score": score} for end_t, score in top.items()])

active_windows = find_active_windows(df, hours=ACTIVE_WINDOW_HOURS, top_n=10)
auto_start = active_windows.iloc[0]["Start"]
auto_end = active_windows.iloc[0]["End"]

print("\n=== SUGGESTED ACTIVE WINDOWS FOR PLOTTING ===")
display(active_windows)
print("Automatic active plotting window:", auto_start, "to", auto_end)

def _make_state_segment_traces(plot_df, x_col, y_col="BIT_DEPTH_SM", name_prefix=""):
    traces = []
    temp = plot_df.copy().reset_index(drop=True)
    temp["_state_segment_id"] = (temp["RigState"] != temp["RigState"].shift()).cumsum()
    shown_legend = set()
    for _, seg in temp.groupby("_state_segment_id", sort=False):
        st = str(seg["RigState"].iloc[0])
        showleg = st not in shown_legend
        shown_legend.add(st)
        traces.append(go.Scattergl(
            x=seg[x_col], y=seg[y_col], mode="lines",
            name=name_prefix + st, legendgroup=st, showlegend=showleg,
            line=dict(color=STATE_COLORS.get(st, "#8c8c8c"), width=VISUALIZATION_STATE_LINE_WIDTH),
            hovertemplate=("State: " + st + "<br>Time: %{customdata[0]}" + "<br>Bit depth: %{y:.1f} ft" + "<br>Hole depth: %{customdata[1]:.1f} ft" + "<extra></extra>"),
            customdata=np.column_stack([seg["FSM_TIMESTAMP"].astype(str), seg["HOLE_DEPTH_SM"].to_numpy()])
        ))
    return traces

def make_interactive_fsm_html(df_in, max_overview_points=35000, max_detail_points=25000):
    full_df = df_in.sort_values("FSM_TIMESTAMP").reset_index(drop=True).copy()
    full_df["Days"] = (full_df["FSM_TIMESTAMP"] - full_df["FSM_TIMESTAMP"].min()).dt.total_seconds() / 86400.0
    overview_df = downsample_for_plot(full_df, max_overview_points)
    detail_df = downsample_for_plot(full_df, max_detail_points)
    fig = make_subplots(
        rows=9, cols=1, shared_xaxes=False, vertical_spacing=0.022,
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{}]],
        subplot_titles=[
            "Overview: Depth vs Days with State-Colored Bit Depth",
            "Depth", "WOB and Hook Load", "Rotary RPM and ROP", "Pump Output and Standpipe Pressure",
            "Rotary Torque and Differential Pressure", "Block Position and Bit Depth",
            "Mechanical Specific Energy — Classical Teale and Hydraulic", "Rig State"
        ],
        row_heights=[0.27, 0.12, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.07]
    )
    fig.add_trace(go.Scattergl(x=overview_df["Days"], y=overview_df["HOLE_DEPTH_SM"], mode="lines", name="Hole Depth", line=dict(width=2, color="#4169e1"), hovertemplate="Days: %{x:.2f}<br>Hole depth: %{y:.1f} ft<extra></extra>"), row=1, col=1)
    for tr in _make_state_segment_traces(overview_df, "Days", "BIT_DEPTH_SM"):
        fig.add_trace(tr, row=1, col=1)
    fig.update_yaxes(title_text="Depth (ft)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(
        title_text="Days. Zoom, box-select, or use the controls above; the parameter dashboard below follows this range.",
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=3, label="3d", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            x=0.0, y=1.12
        ),
        row=1, col=1
    )

    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["HOLE_DEPTH_SM"], mode="lines", name="Detail Hole Depth", line=dict(color="#4169e1", width=1.5), showlegend=False), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["BIT_DEPTH_SM"], mode="lines", name="Detail Bit Depth", line=dict(color="#ff5733", width=1.0), showlegend=False), row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Hole Depth", autorange="reversed", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Bit Depth", autorange="reversed", row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["WOB_SM"], mode="lines", name="WOB", showlegend=False), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["HOOK_LOAD_SM"], mode="lines", name="Hook Load", showlegend=False), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["ROTARY_RPM_SM"], mode="lines", name="Rotary RPM", showlegend=False), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["ROP_SM"], mode="lines", name="ROP", showlegend=False), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["PUMP_OUTPUT_SM"], mode="lines", name="Pump Output", showlegend=False), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["STANDPIPE_PRESSURE_SM"], mode="lines", name="SPP", showlegend=False), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["ROTARY_TORQUE_SM"], mode="lines", name="Torque", showlegend=False), row=6, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["DIFF_PRESSURE_SM"], mode="lines", name="Diff Pressure", showlegend=False), row=6, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["BLOCK_POSITION_SM"], mode="lines", name="Block Position", showlegend=False), row=7, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["BIT_DEPTH_SM"], mode="lines", name="Bit Depth Track", showlegend=False), row=7, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["MSE_TEALE_KSI_SM"], mode="lines", name="Classical Teale MSE", line=dict(color="#5b5bd6"), showlegend=False), row=8, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=detail_df["FSM_TIMESTAMP"], y=detail_df["MSE_HYDRAULIC_KSI_SM"], mode="lines", name="Hydraulic MSE", line=dict(color="#00a884"), showlegend=False), row=8, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Teale MSE (ksi)", row=8, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Hydraulic MSE (ksi)", row=8, col=1, secondary_y=True)
    fig.update_yaxes(type="log", row=8, col=1, secondary_y=False)
    fig.update_yaxes(type="log", row=8, col=1, secondary_y=True)
    for st in sorted(detail_df["RigState"].dropna().unique()):
        state_df = detail_df[detail_df["RigState"] == st]
        fig.add_trace(go.Scattergl(x=state_df["FSM_TIMESTAMP"], y=np.zeros(len(state_df)), mode="markers", marker=dict(size=7, color=STATE_COLORS.get(st, "#cccccc")), name="Detail " + st, legendgroup=st, showlegend=False, hovertemplate=f"State: {st}<br>Time: %{{x}}<extra></extra>"), row=9, col=1)
    fig.update_yaxes(range=[-1, 1], showticklabels=False, showgrid=False, zeroline=False, row=9, col=1)
    for r in range(2, 10):
        fig.update_xaxes(range=[auto_start, auto_end], row=r, col=1)
    fig.update_layout(title="FSM Rig State Detection — Linked Overview, Parameters and MSE", width=1900, height=2450, template="plotly_white", hovermode="closest", legend=dict(x=1.02, y=1), margin=dict(l=115, r=330, t=115, b=95))
    _apply_visualization_style(fig)
    base_ms = int(full_df["FSM_TIMESTAMP"].min().timestamp() * 1000)
    detail_xaxes = ["xaxis2", "xaxis3", "xaxis4", "xaxis5", "xaxis6", "xaxis7", "xaxis8", "xaxis9"]
    html = fig.to_html(include_plotlyjs=True, full_html=True, config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": VISUALIZATION_EXPORT_SCALE}})
    controls_html = """
<div id="fsm-control-panel" style="font-family:Arial, sans-serif; margin:12px 20px; padding:12px; border:1px solid #d0d7de; border-radius:8px; background:#f8fafc; width:1840px;">
  <div style="font-weight:700; margin-bottom:8px;">FSM inspection controls</div>
  <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center;">
    <label>Start time <input id="fsmStartTime" type="datetime-local" step="1"></label>
    <label>End time <input id="fsmEndTime" type="datetime-local" step="1"></label>
    <button id="fsmApplyTime">Apply time range</button>
    <label>Start day <input id="fsmStartDay" type="number" step="0.01" style="width:90px"></label>
    <label>End day <input id="fsmEndDay" type="number" step="0.01" style="width:90px"></label>
    <button id="fsmApplyDays">Apply day range</button>
    <label>Depth from <input id="fsmDepthFrom" type="number" step="10" style="width:90px"></label>
    <label>Depth to <input id="fsmDepthTo" type="number" step="10" style="width:90px"></label>
    <button id="fsmApplyDepth">Find time range by depth</button>
    <button id="fsmResetAll">Reset all</button>
  </div>
  <div id="fsmHoverReadout" style="margin-top:8px; font-size:17px; color:#334155;">Hover over the depth plot to read day/time/depth/state.</div>
</div>
"""
    sync_js = f"""
<script>
(function() {{
  const baseMs = {base_ms};
  const detailAxes = {detail_xaxes};
  const gd = document.querySelector('.plotly-graph-div');
  if (!gd) return;

  function daysToMs(d) {{ return baseMs + Number(d) * 86400000; }}
  function daysToIso(d) {{ return new Date(daysToMs(d)).toISOString(); }}
  function isoToDays(iso) {{ return (new Date(iso).getTime() - baseMs) / 86400000; }}
  function localInputToIso(v) {{ if (!v) return null; const d = new Date(v); return d.toISOString(); }}
  function isoToLocalInput(iso) {{
    const d = new Date(iso); const z = new Date(d.getTime() - d.getTimezoneOffset()*60000);
    return z.toISOString().slice(0,19);
  }}
  function setDetailRangeByDays(d0, d1) {{
    const t0 = daysToIso(d0); const t1 = daysToIso(d1);
    const update = {{'xaxis.range':[Number(d0), Number(d1)]}};
    detailAxes.forEach(function(ax) {{ update[ax + '.range'] = [t0, t1]; }});
    Plotly.relayout(gd, update);
    const st = document.getElementById('fsmStartTime'); const et = document.getElementById('fsmEndTime');
    const sd = document.getElementById('fsmStartDay'); const ed = document.getElementById('fsmEndDay');
    if (st) st.value = isoToLocalInput(t0); if (et) et.value = isoToLocalInput(t1);
    if (sd) sd.value = Number(d0).toFixed(3); if (ed) ed.value = Number(d1).toFixed(3);
  }}

  gd.on('plotly_relayout', function(ev) {{
    let r0 = ev['xaxis.range[0]']; let r1 = ev['xaxis.range[1]'];
    if (r0 === undefined || r1 === undefined) return;
    setDetailRangeByDays(r0, r1);
  }});

  gd.on('plotly_hover', function(ev) {{
    const p = ev.points && ev.points[0]; if (!p) return;
    const xday = Number(p.x); const time = daysToIso(xday).replace('T',' ').replace('Z',' UTC');
    const state = p.data && p.data.name ? p.data.name.replace(/^Detail /,'') : '';
    const depth = (typeof p.y === 'number') ? p.y.toFixed(1) : p.y;
    const el = document.getElementById('fsmHoverReadout');
    if (el) el.innerHTML = 'Day: <b>' + xday.toFixed(3) + '</b> | Time: <b>' + time + '</b> | Depth: <b>' + depth + ' ft</b> | Trace/state: <b>' + state + '</b>';
  }});

  const applyTime = document.getElementById('fsmApplyTime');
  if (applyTime) applyTime.onclick = function() {{
    const s = localInputToIso(document.getElementById('fsmStartTime').value);
    const e = localInputToIso(document.getElementById('fsmEndTime').value);
    if (!s || !e) return alert('Enter both start and end time.');
    setDetailRangeByDays(isoToDays(s), isoToDays(e));
  }};

  const applyDays = document.getElementById('fsmApplyDays');
  if (applyDays) applyDays.onclick = function() {{
    const d0 = Number(document.getElementById('fsmStartDay').value);
    const d1 = Number(document.getElementById('fsmEndDay').value);
    if (!isFinite(d0) || !isFinite(d1)) return alert('Enter both start and end day.');
    setDetailRangeByDays(d0, d1);
  }};

  const applyDepth = document.getElementById('fsmApplyDepth');
  if (applyDepth) applyDepth.onclick = function() {{
    const a = Number(document.getElementById('fsmDepthFrom').value);
    const b = Number(document.getElementById('fsmDepthTo').value);
    if (!isFinite(a) || !isFinite(b)) return alert('Enter depth from/to.');
    const lo = Math.min(a,b), hi = Math.max(a,b);
    const holeTrace = gd.data.find(tr => tr.name === 'Hole Depth');
    if (!holeTrace) return alert('Hole Depth trace not found.');
    const xs = holeTrace.x || []; const ys = holeTrace.y || [];
    let found = [];
    for (let i=0; i<xs.length; i++) {{ const y = Number(ys[i]); if (y >= lo && y <= hi) found.push(Number(xs[i])); }}
    if (!found.length) return alert('No overview points found inside that depth interval. Try a wider depth range.');
    setDetailRangeByDays(Math.min(...found), Math.max(...found));
  }};

  const reset = document.getElementById('fsmResetAll');
  if (reset) reset.onclick = function() {{ Plotly.relayout(gd, {{'xaxis.autorange': true}}); }};
}})();
</script>
"""
    html = html.replace("<body>", "<body>" + controls_html)
    html = html.replace("</body>", sync_js + "\n</body>")
    out_path = OUTPUT_DIR / "fsm_linked_overview_parameter_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    return fig, out_path

def plot_dashboard(df_in, start, end, max_points=PLOT_ACTIVE_MAX_POINTS):
    plot_df = df_in[(df_in["FSM_TIMESTAMP"] >= pd.to_datetime(start)) & (df_in["FSM_TIMESTAMP"] <= pd.to_datetime(end))].copy()
    plot_df = plot_df.sort_values("FSM_TIMESTAMP").reset_index(drop=True)
    plot_df = downsample_for_plot(plot_df, max_points)
    if plot_df.empty: raise ValueError("Selected active plotting window is empty.")
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.025, specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{}]], subplot_titles=["Depth", "WOB and Hook Load", "Rotary RPM and ROP", "Pump Output and Standpipe Pressure", "Rotary Torque and Differential Pressure", "Block Position and Bit Depth", "Classical Teale and Hydraulic MSE", "Rig State"], row_heights=[0.16, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.07])
    fig.add_trace(go.Scattergl(x=plot_df["FSM_TIMESTAMP"], y=plot_df["HOLE_DEPTH_SM"], mode="lines", name="Hole Depth"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=plot_df["FSM_TIMESTAMP"], y=plot_df["BIT_DEPTH_SM"], mode="lines", name="Bit Depth"), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Hole Depth (ft)", autorange="reversed", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Bit Depth (ft)", autorange="reversed", row=1, col=1, secondary_y=True)
    for y1, y2, n1, n2, r in [("WOB_SM","HOOK_LOAD_SM","WOB","Hook Load",2),("ROTARY_RPM_SM","ROP_SM","Rotary RPM","ROP",3),("PUMP_OUTPUT_SM","STANDPIPE_PRESSURE_SM","Pump Output","Standpipe Pressure",4),("ROTARY_TORQUE_SM","DIFF_PRESSURE_SM","Rotary Torque","Differential Pressure",5),("BLOCK_POSITION_SM","BIT_DEPTH_SM","Block Position","Bit Depth",6)]:
        fig.add_trace(go.Scattergl(x=plot_df["FSM_TIMESTAMP"], y=plot_df[y1], mode="lines", name=n1), row=r, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=plot_df["FSM_TIMESTAMP"], y=plot_df[y2], mode="lines", name=n2), row=r, col=1, secondary_y=True)
    fig.add_trace(go.Scattergl(x=plot_df["FSM_TIMESTAMP"], y=plot_df["MSE_TEALE_KSI_SM"], mode="lines", name="Classical Teale MSE", line=dict(color="#5b5bd6")), row=7, col=1, secondary_y=False)
    fig.add_trace(go.Scattergl(x=plot_df["FSM_TIMESTAMP"], y=plot_df["MSE_HYDRAULIC_KSI_SM"], mode="lines", name="Hydraulic MSE", line=dict(color="#00a884")), row=7, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Teale MSE (ksi)", row=7, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Hydraulic MSE (ksi)", row=7, col=1, secondary_y=True)
    fig.update_yaxes(type="log", row=7, col=1, secondary_y=False)
    fig.update_yaxes(type="log", row=7, col=1, secondary_y=True)
    for st in sorted(plot_df["RigState"].dropna().unique()):
        state_df = plot_df[plot_df["RigState"] == st]
        fig.add_trace(go.Scattergl(x=state_df["FSM_TIMESTAMP"], y=np.zeros(len(state_df)), mode="markers", marker=dict(size=8, color=STATE_COLORS.get(st, "#cccccc")), name=st, hovertemplate=f"State: {st}<br>Time: %{{x}}<extra></extra>"), row=8, col=1)
    fig.update_yaxes(range=[-1, 1], showticklabels=False, showgrid=False, zeroline=False, row=8, col=1)
    fig.update_layout(title="FSM Rig State Detection Dashboard — Active Interval", width=1800, height=1500, template="plotly_white", hovermode="x unified", legend=dict(x=1.02, y=1), margin=dict(l=110, r=300, t=105, b=90))
    _apply_visualization_style(fig)
    fig.update_xaxes(title_text="Time", row=8, col=1)
    if SHOW_PLOTS_IN_NOTEBOOK: fig.show()
    if SAVE_HTML_PLOTS: fig.write_html(OUTPUT_DIR / "fsm_dashboard_active_interval.html", include_plotlyjs="cdn", full_html=True)
    return fig

def plot_depth_state_overlay(df_in, max_points=PLOT_OVERLAY_MAX_POINTS):
    plot_df = df_in.sort_values("FSM_TIMESTAMP").reset_index(drop=True)
    plot_df = downsample_for_plot(plot_df, max_points)
    plot_df["Days"] = (plot_df["FSM_TIMESTAMP"] - plot_df["FSM_TIMESTAMP"].min()).dt.total_seconds() / 86400
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=plot_df["Days"], y=plot_df["HOLE_DEPTH_SM"], mode="lines", name="Hole Depth", line=dict(width=2, color="#4169e1")))
    for tr in _make_state_segment_traces(plot_df, "Days", "BIT_DEPTH_SM"):
        fig.add_trace(tr)
    fig.update_layout(title="Depth vs Days with FSM Rig State Overlay", xaxis_title="Days", yaxis_title="Depth (ft)", width=1800, height=900, template="plotly_white", hovermode="closest", legend=dict(x=1.02, y=1), margin=dict(l=110, r=300, t=105, b=90))
    _apply_visualization_style(fig)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(rangeslider=dict(visible=True))
    if SHOW_PLOTS_IN_NOTEBOOK: fig.show()
    if SAVE_HTML_PLOTS: fig.write_html(OUTPUT_DIR / "fsm_depth_state_overlay.html", include_plotlyjs="cdn", full_html=True)
    return fig

plot_dashboard(df, auto_start, auto_end, max_points=PLOT_ACTIVE_MAX_POINTS)
plot_depth_state_overlay(df, max_points=PLOT_OVERLAY_MAX_POINTS)
linked_fsm_fig, linked_fsm_html_path = make_interactive_fsm_html(df)

# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

state_counts.to_csv(OUTPUT_DIR / "fsm_state_counts_corrected.csv", index=False)
state_intervals.to_csv(OUTPUT_DIR / "fsm_state_intervals_corrected.csv", index=False)
# duplicate name for clarity: this is the step/localized state interval table
state_intervals.to_csv(OUTPUT_DIR / "fsm_step_state_intervals_corrected.csv", index=False)
condition_counts.to_csv(OUTPUT_DIR / "fsm_condition_counts_corrected.csv", index=False)
fsm_qc.to_csv(OUTPUT_DIR / "fsm_qc_corrected.csv", index=False)
threshold_report.to_csv(OUTPUT_DIR / "fsm_thresholds_corrected.csv", index=False)
active_windows.to_csv(OUTPUT_DIR / "fsm_active_plot_windows.csv", index=False)
connection_candidates_df.to_csv(OUTPUT_DIR / "fsm_connection_candidates_corrected.csv", index=False)
moving_pipe_candidates_df.to_csv(OUTPUT_DIR / "fsm_moving_pipe_candidates_corrected.csv", index=False)
stand_connection_candidates_df.to_csv(OUTPUT_DIR / "fsm_stand_connection_candidates_corrected.csv", index=False)
semantic_reconciliation_report.to_csv(
    OUTPUT_DIR / "fsm_semantic_reconciliation_report.csv", index=False
)

# Compact state labels. Always saved.
compact_cols = [
    "FSM_TIMESTAMP", "RigState", "RigState_raw", "PrevState",
    "BIT_DEPTH_SM", "HOLE_DEPTH_SM", "BLOCK_POSITION_SM", "HOOK_LOAD_SM",
    "PUMP_OUTPUT_SM", "STANDPIPE_PRESSURE_SM", "DIFF_PRESSURE_SM",
    "WOB_SM", "ROP_SM", "ROTARY_RPM_SM", "ROTARY_TORQUE_SM",
    "MSE_TEALE_KSI_SM", "MSE_HYDRAULIC_KSI_SM",
    "PUMP_ON", "RPM_ON", "WOB_ON", "ROP_ON", "ON_BOTTOM", "NEAR_BOTTOM_FOR_CONNECTION",
    "BIT_MOVING_DOWN", "BIT_MOVING_UP", "BIT_MOVING",
    "BIT_VEL_FTPM", "BLOCK_VEL_FTPM", "BIT_TO_BOTTOM",
    "HOLE_ADVANCING", "BIT_TREND_FT", "HOLE_TREND_FT"
]
compact_cols = [c for c in compact_cols if c in df.columns]
df[compact_cols].to_csv(OUTPUT_DIR / "fsm_labeled_compact_corrected.csv", index=False)

if SAVE_FULL_LABELED_CSV:
    df.to_csv(OUTPUT_DIR / "fsm_labeled_full_corrected.csv", index=False)

if SAVE_COMPACT_LABELED_PARQUET:
    try:
        df[compact_cols].to_parquet(OUTPUT_DIR / "fsm_labeled_compact_corrected.parquet", index=False)
    except Exception as exc:
        print("Parquet save skipped:", exc)

# Expose conventional variable names for notebook use.
df_states = df
fsm_state_counts = state_counts
fsm_state_intervals = state_intervals
fsm_threshold_report = threshold_report
fsm_connection_candidates = connection_candidates_df
fsm_active_windows = active_windows
fsm_moving_pipe_candidates = moving_pipe_candidates_df
fsm_stand_connection_candidates = stand_connection_candidates_df
fsm_semantic_reconciliation_report = semantic_reconciliation_report
fsm_linked_overview_parameter_dashboard = linked_fsm_html_path

print("\n=== SAVED FILES ===")
for p in sorted(OUTPUT_DIR.glob("*")):
    print("-", p.resolve())

print("\nDataFrame outputs now available in memory:")
print("- df_states")
print("- fsm_state_counts")
print("- fsm_state_intervals")
print("- fsm_threshold_report")
print("- fsm_connection_candidates")
print("- fsm_active_windows")
print("- fsm_moving_pipe_candidates")
print("- fsm_stand_connection_candidates")
print("- fsm_linked_overview_parameter_dashboard")


# ============================================================
# 14. UNIFIED LINKED FSM + OPERATIONAL + AUTOMATIC BIT-RUN HTML
# ============================================================
# This section creates ONE standalone HTML containing:
#   - the linked FSM depth/parameter overview
#   - overall, section, time-period, stand and bit-run state statistics
#   - editable hole/casing and lithology tables
#   - automatic bit-run detection using deep-in-hole / near-surface boundaries
#   - bit-run state and performance comparison
# It does not change the FSM classification above.

import json
from plotly.offline import get_plotlyjs

# Well identity is dashboard metadata only; it never changes the FSM logic.
# CLI users can set it with --well-name or the FSM_WELL_NAME environment variable.
WELL_NAME = os.environ.get("FSM_WELL_NAME", globals().get("WELL_NAME", "FSM_Well"))
UNIFIED_MAX_POINTS = 24000
BITRUN_BOUNDARY_FT = 150.0 * 3.280839895
BITRUN_NEAR_SURFACE_FT = 5.0 * 3.280839895
BITRUN_FALSE_SURFACE_GAP_MIN = 5.0
BITRUN_TRUE_SURFACE_GAP_MIN = 20.0
BITRUN_MAX_DATA_GAP_MIN = 360.0
BITRUN_MIN_DURATION_MIN = 10.0
BITRUN_PRIMARY_FOOTAGE_FT = 100.0
BITRUN_SECONDARY_FOOTAGE_FT = 2.0

def _metadata_rows(global_name, environment_name):
    if global_name in globals():
        return globals()[global_name]
    raw = os.environ.get(environment_name, "").strip()
    if not raw:
        return []
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError(f"{environment_name} must contain a JSON list")
    return rows


DEFAULT_HOLE_SECTION_ROWS = _metadata_rows(
    "DEFAULT_HOLE_SECTION_ROWS", "FSM_HOLE_SECTIONS_JSON"
)
DEFAULT_LITHOLOGY_ROWS = _metadata_rows(
    "DEFAULT_LITHOLOGY_ROWS", "FSM_LITHOLOGY_JSON"
)

# -------------------------
# 14A. Duration-safe rows
# -------------------------
ops = df[["FSM_TIMESTAMP","RigState","DT_SEC","HOLE_DEPTH_SM","BIT_DEPTH_SM",
          "WOB_SM","ROP_SM","ROTARY_RPM_SM","ROTARY_TORQUE_SM",
          "PUMP_OUTPUT_SM","STANDPIPE_PRESSURE_SM","DIFF_PRESSURE_SM","HOOK_LOAD_SM",
          "BLOCK_POSITION_SM","MSE_TEALE_KSI_SM","MSE_HYDRAULIC_KSI_SM"]].copy()
ops["Duration_sec"] = pd.to_numeric(ops["DT_SEC"], errors="coerce").fillna(med_dt)
ops.loc[(ops["Duration_sec"] <= 0) | (ops["Duration_sec"] > max(3600.0, med_dt * 30.0)), "Duration_sec"] = med_dt
ops["Duration_hr"] = ops["Duration_sec"] / 3600.0
# Positive measured-depth advance attributed to the state active at that sample.
ops["Drilled_ft"] = pd.to_numeric(ops["HOLE_DEPTH_SM"], errors="coerce").diff().clip(lower=0).fillna(0.0)
# Suppress isolated depth spikes while retaining genuine drilling advance.
_depth_cap = max(30.0, float(ops["Drilled_ft"].quantile(0.999)) if len(ops) else 30.0)
ops.loc[ops["Drilled_ft"] > _depth_cap, "Drilled_ft"] = 0.0


def _summary(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["RigState","Samples","Duration_sec","Duration_hr","Percentage"])
    z = frame.groupby("RigState", dropna=False).agg(
        Samples=("RigState","size"),
        Duration_sec=("Duration_sec","sum"),
        Drilled_ft=("Drilled_ft","sum"),
    ).reset_index()
    total = float(z["Duration_sec"].sum())
    z["Duration_hr"] = z["Duration_sec"] / 3600.0
    z["Percentage"] = np.where(total > 0, z["Duration_sec"] / total * 100.0, 0.0)
    return z.sort_values("Duration_sec", ascending=False).reset_index(drop=True)

unified_overall_summary = _summary(ops)

# -------------------------
# 14B. Automatic bit runs
# -------------------------
def detect_bit_runs_from_fsm(df_in):
    x = df_in[["FSM_TIMESTAMP","BIT_DEPTH_SM","HOLE_DEPTH_SM","RigState","Duration_sec","Drilled_ft"]].copy().reset_index(drop=True)
    x["deep_raw"] = x["BIT_DEPTH_SM"] > BITRUN_BOUNDARY_FT
    sid = x["deep_raw"].ne(x["deep_raw"].shift()).cumsum()
    seg = x.groupby(sid).agg(start_idx=("FSM_TIMESTAMP",lambda s:int(s.index.min())), end_idx=("FSM_TIMESTAMP",lambda s:int(s.index.max())), state=("deep_raw","first"), start_time=("FSM_TIMESTAMP","first"), end_time=("FSM_TIMESTAMP","last")).reset_index(drop=True)
    seg["duration_min"] = (seg["end_time"]-seg["start_time"]).dt.total_seconds()/60.0
    x["deep"] = x["deep_raw"]
    for i in range(1,len(seg)-1):
        if bool(seg.iloc[i-1].state) and not bool(seg.iloc[i].state) and bool(seg.iloc[i+1].state) and float(seg.iloc[i].duration_min) <= BITRUN_FALSE_SURFACE_GAP_MIN:
            x.loc[int(seg.iloc[i].start_idx):int(seg.iloc[i].end_idx),"deep"] = True
    gap_min = x["FSM_TIMESTAMP"].diff().dt.total_seconds().div(60).fillna(0)
    x["grp"] = x["deep"].ne(x["deep"].shift()) | (gap_min > BITRUN_MAX_DATA_GAP_MIN)
    x["grp"] = x["grp"].cumsum()
    cores = x[x["deep"]].groupby("grp").agg(core_start=("FSM_TIMESTAMP",lambda s:int(s.index.min())), core_end=("FSM_TIMESTAMP",lambda s:int(s.index.max()))).reset_index(drop=True)
    records=[]
    prev_end=-1
    for k,r in cores.iterrows():
        cs,ce=int(r.core_start),int(r.core_end)
        s=cs; shallow=0.0
        while s>prev_end+1:
            gap=(x.loc[s,"FSM_TIMESTAMP"]-x.loc[s-1,"FSM_TIMESTAMP"]).total_seconds()/60.0
            if gap>BITRUN_MAX_DATA_GAP_MIN: break
            if float(x.loc[s-1,"BIT_DEPTH_SM"])<=BITRUN_NEAR_SURFACE_FT:
                shallow += max(gap,0)
                if shallow>=BITRUN_TRUE_SURFACE_GAP_MIN: break
            else: shallow=0.0
            s-=1
        next_start=int(cores.iloc[k+1].core_start) if k<len(cores)-1 else len(x)
        e=ce; shallow=0.0
        while e<next_start-1:
            gap=(x.loc[e+1,"FSM_TIMESTAMP"]-x.loc[e,"FSM_TIMESTAMP"]).total_seconds()/60.0
            if gap>BITRUN_MAX_DATA_GAP_MIN: break
            if float(x.loc[e+1,"BIT_DEPTH_SM"])<=BITRUN_NEAR_SURFACE_FT:
                shallow += max(gap,0)
                if shallow>=BITRUN_TRUE_SURFACE_GAP_MIN: break
            else: shallow=0.0
            e+=1
        sub=x.loc[s:e]
        if sub.empty: continue
        dur=(sub["FSM_TIMESTAMP"].iloc[-1]-sub["FSM_TIMESTAMP"].iloc[0]).total_seconds()/3600.0
        if dur*60<BITRUN_MIN_DURATION_MIN: continue
        depth_in=float(sub["HOLE_DEPTH_SM"].iloc[0]); depth_out=float(sub["HOLE_DEPTH_SM"].max()); footage=depth_out-depth_in
        bit_exc=float(sub["BIT_DEPTH_SM"].max()-sub["BIT_DEPTH_SM"].min())
        drilling_hr = float(sub.loc[
            sub["RigState"].isin(["Drilling Rotate", "Drilling Slide"]), "Duration_sec"
        ].sum() / 3600.0)
        if footage>=BITRUN_PRIMARY_FOOTAGE_FT and drilling_hr > 0:
            run_class="primary_depth_extension_run"; role="Primary drilling bit run"; comparable=True
        elif footage>=BITRUN_SECONDARY_FOOTAGE_FT:
            run_class="secondary_depth_extension_run"; role="Secondary / short-footage review interval"; comparable=False
        else:
            run_class="zero_depth_service_run"; role="Zero-depth service / conditioning run"; comparable=False
        records.append({"run_id":len(records)+1,"run_key":f"Bit Run {len(records)+1}","start_idx":s,"end_idx":e,"start_time":sub["FSM_TIMESTAMP"].iloc[0],"end_time":sub["FSM_TIMESTAMP"].iloc[-1],"depth_in_ft":depth_in,"depth_out_ft":depth_out,"footage_ft":footage,"duration_hr":dur,"bit_excursion_ft":bit_exc,"drilling_hr":drilling_hr,"run_class":run_class,"role":role,"selectable":comparable})
        prev_end=e
    return pd.DataFrame(records)

fsm_detected_bit_runs = detect_bit_runs_from_fsm(ops)

# Bit-run state distribution and performance metrics.
bitrun_state_rows=[]; bitrun_perf_rows=[]
for _,r in fsm_detected_bit_runs.iterrows():
    sub=ops.loc[int(r.start_idx):int(r.end_idx)].copy()
    sm=_summary(sub)
    for _,q in sm.iterrows():
        bitrun_state_rows.append({"run_key":r.run_key,"RigState":q.RigState,"Duration_sec":float(q.Duration_sec),"Duration_hr":float(q.Duration_hr),"Percentage":float(q.Percentage),"Drilled_ft":float(q.Drilled_ft)})
    drilling=sub[sub["RigState"].isin(["Drilling Rotate","Drilling Slide"])]
    bitrun_perf_rows.append({
        "run_key":r.run_key,"run_class":r.run_class,"role":getattr(r,"role",r.run_class),"selectable":bool(r.selectable),"start_time":r.start_time,"end_time":r.end_time,
        "depth_in_ft":float(r.depth_in_ft),"depth_out_ft":float(r.depth_out_ft),"footage_ft":float(r.footage_ft),"duration_hr":float(r.duration_hr),
        "drilling_hr":float(drilling.Duration_hr.sum()),"avg_rop_ft_hr":float(drilling.ROP_SM.mean()) if len(drilling) else np.nan,
        "avg_wob":float(drilling.WOB_SM.mean()) if len(drilling) else np.nan,"avg_rpm":float(drilling.ROTARY_RPM_SM.mean()) if len(drilling) else np.nan,
        "avg_torque":float(drilling.ROTARY_TORQUE_SM.mean()) if len(drilling) else np.nan,"avg_flow":float(drilling.PUMP_OUTPUT_SM.mean()) if len(drilling) else np.nan,
        "avg_spp":float(drilling.STANDPIPE_PRESSURE_SM.mean()) if len(drilling) else np.nan,
        "idle_hr":float(sub.loc[sub.RigState.eq("Idle"),"Duration_hr"].sum()),"connection_hr":float(sub.loc[sub.RigState.eq("Connection"),"Duration_hr"].sum()),
        "trip_hr":float(sub.loc[sub.RigState.isin(["Trip-In","Trip-Out"]),"Duration_hr"].sum()),"ream_hr":float(sub.loc[sub.RigState.isin(["Reaming","Backreaming"]),"Duration_hr"].sum())
    })
fsm_bitrun_state_distribution=pd.DataFrame(bitrun_state_rows)
fsm_bitrun_performance=pd.DataFrame(bitrun_perf_rows)

# -------------------------
# 14C. Stand-by-stand — connection-validated and footage constrained
# -------------------------
def build_validated_stands(ops_df, intervals_df):
    """
    Build drilling stands from connection evidence and measured-depth progression.

    A normal Range-2 drill-pipe joint is roughly 27–32 ft; a triple is therefore
    normally near 80–96 ft. The algorithm does not blindly force 90 ft. It first
    estimates the well-specific nominal stand from accepted connection-to-connection
    footage, then rejects/merges fragments and splits long missed-connection gaps.
    """
    conn = intervals_df[intervals_df["RigState"].eq("Connection")].copy()
    if conn.empty:
        return pd.DataFrame(), pd.DataFrame()
    conn = conn.sort_values("Start_Time").reset_index(drop=True)

    # Connection measured depth: robust median hole depth inside each connection.
    conn_depths=[]
    for _,r in conn.iterrows():
        a,b=pd.to_datetime(r.Start_Time),pd.to_datetime(r.End_Time)
        z=ops_df[(ops_df.FSM_TIMESTAMP>=a)&(ops_df.FSM_TIMESTAMP<=b)]["HOLE_DEPTH_SM"]
        conn_depths.append(float(z.median()) if len(z) else float(r.End_Hole_Depth))
    conn["Connection_MD_ft"]=conn_depths

    raw_diff=conn["Connection_MD_ft"].diff()
    plausible=raw_diff[(raw_diff>=60.0)&(raw_diff<=120.0)]
    nominal=float(plausible.median()) if len(plausible)>=3 else STAND_LENGTH_FT
    nominal=float(np.clip(nominal,75.0,105.0))

    boundaries=[]
    last_depth=None
    for _,r in conn.iterrows():
        d=float(r.Connection_MD_ft)
        t=pd.to_datetime(r.End_Time)
        if not np.isfinite(d):
            continue
        if last_depth is None:
            boundaries.append((t,d,True))
            last_depth=d
            continue
        delta=d-last_depth
        if delta < MIN_VALID_STAND_FT:
            # Duplicate/false connection; do not create a short stand.
            continue
        # Find an integer number of stands that covers the entire depth change
        # while keeping every inferred stand inside the permitted footage
        # range. Using only round(delta / nominal) misses borderline cases such
        # as 131.6 ft: one stand is too long, but two 65.8-ft stands are valid.
        n_min=max(1,int(np.ceil(delta/MAX_VALID_STAND_FT)))
        n_max=int(np.floor(delta/MIN_VALID_STAND_FT))
        if n_min<=n_max:
            n=int(np.clip(round(delta/nominal),n_min,n_max))
        else:
            n=0

        if n>=1:
            # Observed normal stand (n=1) or one/more missing connections
            # represented by interpolated depth/time boundaries (n>1).
            t0=boundaries[-1][0]
            for j in range(1,n+1):
                frac=j/n
                tj=t0+(t-t0)*frac
                dj=last_depth+delta*frac
                boundaries.append((tj,dj,j==n))
        else:
            # The increment cannot be partitioned into plausible stand footage.
            # Keep the new observed anchor, but it will remain explicit in the
            # exported continuity QC rather than being presented as a stand.
            boundaries.append((t,d,True))
        last_depth=d

    stand_rows=[]; stand_state_rows=[]
    for i in range(len(boundaries)-1):
        s,md0,start_observed=boundaries[i]; e,md1,end_observed=boundaries[i+1]
        footage=md1-md0
        if footage<MIN_VALID_STAND_FT or footage>MAX_VALID_STAND_FT:
            continue
        sub=ops_df[(ops_df.FSM_TIMESTAMP>s)&(ops_df.FSM_TIMESTAMP<=e)].copy()
        drill=sub[sub.RigState.isin(["Drilling Rotate","Drilling Slide"])] if not sub.empty else sub
        duration=float(sub.Duration_hr.sum()) if not sub.empty else 0.0
        drilling_hr=float(drill.Duration_hr.sum()) if not drill.empty else 0.0
        drilling_fraction=(drilling_hr/duration) if duration>0 else 0.0
        boundary_source=(
            "Observed connection boundaries"
            if bool(start_observed) and bool(end_observed)
            else "Interpolated missing-connection boundary"
        )
        if drilling_fraction>=MIN_STAND_DRILLING_FRACTION:
            validation_status=(
                "Validated"
                if bool(start_observed) and bool(end_observed)
                else "Inferred stand — drilling supported"
            )
            selectable=True
        else:
            # Preserve the depth bin instead of silently renumbering later
            # stands. This makes missing/weak evidence visible and prevents
            # apparent jumps such as Stand 1 ending near 352 ft and Stand 2
            # starting near 850 ft.
            validation_status="Review — insufficient drilling evidence"
            selectable=False
        key=f"Stand {len(stand_rows)+1}"
        sm=_summary(sub) if not sub.empty else pd.DataFrame()
        dominant_state = str(sm.sort_values("Duration_sec", ascending=False).iloc[0]["RigState"]) if len(sm) else "Unknown"
        dominant_pct = float(sm.sort_values("Duration_sec", ascending=False).iloc[0]["Percentage"]) if len(sm) else 0.0
        stand_rows.append({
            "Stand":key,"Start_Time":s,"End_Time":e,
            "Start_MD_ft":float(md0),"End_MD_ft":float(md1),
            "Footage_ft":float(footage),"Nominal_Stand_ft":float(nominal),
            "Duration_hr":duration,"Duration_min":duration*60.0,
            "Drilling_hr":drilling_hr,"Drilling_min":drilling_hr*60.0,
            "Drilling_Fraction":float(drilling_fraction),
            "Avg_ROP_ft_hr":float(drill["ROP_SM"].mean()) if len(drill) else np.nan,
            "Dominant_State":dominant_state,"Dominant_State_pct":dominant_pct,
            "Connection_End_MD_ft":float(md1),
            "Boundary_Source":boundary_source,
            "Validation_Status":validation_status,
            "Selectable":bool(selectable),
        })
        for _,q in sm.iterrows():
            stand_state_rows.append({"Stand":key,"RigState":q.RigState,"Duration_sec":float(q.Duration_sec),"Duration_hr":float(q.Duration_hr),"Percentage":float(q.Percentage),"Drilled_ft":float(q.Drilled_ft)})
    return pd.DataFrame(stand_rows),pd.DataFrame(stand_state_rows)

fsm_stand_summary,fsm_stand_state_distribution=build_validated_stands(ops,state_intervals)
fsm_stand_gap_report = (
    fsm_stand_summary.loc[
        ~fsm_stand_summary["Validation_Status"].eq("Validated")
    ].copy()
    if not fsm_stand_summary.empty and "Validation_Status" in fsm_stand_summary.columns
    else pd.DataFrame()
)
if not fsm_stand_summary.empty:
    _stand_continuity = fsm_stand_summary[
        ["Stand", "Start_MD_ft", "End_MD_ft", "Validation_Status"]
    ].copy()
    _stand_continuity["Previous_End_MD_ft"] = _stand_continuity["End_MD_ft"].shift()
    _stand_continuity["Gap_From_Previous_ft"] = (
        _stand_continuity["Start_MD_ft"] - _stand_continuity["Previous_End_MD_ft"]
    )
    fsm_stand_continuity_report = _stand_continuity.loc[
        _stand_continuity["Gap_From_Previous_ft"].abs() > 0.5
    ].copy()
else:
    fsm_stand_continuity_report = pd.DataFrame(columns=[
        "Stand", "Start_MD_ft", "End_MD_ft", "Validation_Status",
        "Previous_End_MD_ft", "Gap_From_Previous_ft",
    ])

# -------------------------
# 14D. Period summaries
# -------------------------
def period_summary(freq):
    z=ops.copy(); z["Period"]=z.FSM_TIMESTAMP.dt.to_period(freq).dt.start_time
    return z.groupby(["Period","RigState"],as_index=False).agg(Duration_sec=("Duration_sec","sum"),Samples=("RigState","size"),Drilled_ft=("Drilled_ft","sum")).assign(Duration_hr=lambda a:a.Duration_sec/3600.0)
fsm_hourly_summary=period_summary("h"); fsm_daily_summary=period_summary("D"); fsm_weekly_summary=period_summary("W"); fsm_monthly_summary=period_summary("M")

# -------------------------
# 14E. JSON payload
# -------------------------
plot_df=ops.sort_values("FSM_TIMESTAMP").reset_index(drop=True)
if len(plot_df)>UNIFIED_MAX_POINTS:
    plot_df=plot_df.iloc[::int(np.ceil(len(plot_df)/UNIFIED_MAX_POINTS))].copy()
t0=plot_df.FSM_TIMESTAMP.min(); plot_df["Days"]=(plot_df.FSM_TIMESTAMP-t0).dt.total_seconds()/86400.0

# Parameter curves may be uniformly thinned, but categorical FSM states must
# preserve every transition. Build a separate compact state payload from:
#   1) a uniform background sample for long movement paths,
#   2) rows immediately around every state transition, and
#   3) the midpoint of every state episode.
# This is generic for every state and every well; no Connection-only detection
# thresholds are introduced here.
_ops_state = ops.sort_values("FSM_TIMESTAMP").reset_index(drop=True)
_state_keep = set(range(0, len(_ops_state), max(1, int(np.ceil(len(_ops_state)/UNIFIED_MAX_POINTS)))))
_change_idx = np.flatnonzero(
    _ops_state["RigState"].ne(_ops_state["RigState"].shift()).to_numpy()
)
for _i in _change_idx:
    for _j in (_i-1, _i, _i+1):
        if 0 <= _j < len(_ops_state):
            _state_keep.add(int(_j))
_state_group_id = _ops_state["RigState"].ne(_ops_state["RigState"].shift()).cumsum()
for _, _g in _ops_state.groupby(_state_group_id, sort=False):
    _state_keep.add(int(_g.index[len(_g)//2]))
state_plot_df = _ops_state.iloc[sorted(_state_keep)].copy()
state_plot_df["Days"] = (
    state_plot_df["FSM_TIMESTAMP"] - _ops_state["FSM_TIMESTAMP"].min()
).dt.total_seconds()/86400.0

# Preserve the sparse drilling-only MSE samples independently from the general
# dashboard downsampling. Uniformly thinning the full well can discard short
# MSE intervals and make valid data appear cut off.
mse_plot_df = ops.loc[
    ops["MSE_TEALE_KSI_SM"].notna() | ops["MSE_HYDRAULIC_KSI_SM"].notna(),
    ["FSM_TIMESTAMP", "MSE_TEALE_KSI_SM", "MSE_HYDRAULIC_KSI_SM"]
].sort_values("FSM_TIMESTAMP").reset_index(drop=True)
if len(mse_plot_df) > UNIFIED_MAX_POINTS:
    mse_plot_df = mse_plot_df.iloc[
        ::int(np.ceil(len(mse_plot_df) / UNIFIED_MAX_POINTS))
    ].copy()

def clean_json(v):
    if isinstance(v,dict): return {str(k):clean_json(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean_json(x) for x in v]
    if isinstance(v,pd.DataFrame): return clean_json(v.to_dict("records"))
    if isinstance(v,pd.Timestamp): return v.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    try:
        if pd.isna(v): return None
    except Exception: pass
    return v

def records(df0): return clean_json(df0) if isinstance(df0,pd.DataFrame) else []

main_payload={"time":plot_df.FSM_TIMESTAMP.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),"days":clean_json(plot_df.Days.tolist()),"state":plot_df.RigState.astype(str).tolist()}
for c in ["HOLE_DEPTH_SM","BIT_DEPTH_SM","WOB_SM","HOOK_LOAD_SM","ROTARY_RPM_SM","ROP_SM","PUMP_OUTPUT_SM","STANDPIPE_PRESSURE_SM","ROTARY_TORQUE_SM","DIFF_PRESSURE_SM","BLOCK_POSITION_SM","MSE_TEALE_KSI_SM","MSE_HYDRAULIC_KSI_SM"]:
    main_payload[c]=clean_json(plot_df[c].tolist())

state_track_payload = {
    "time": state_plot_df["FSM_TIMESTAMP"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
    "days": clean_json(state_plot_df["Days"].tolist()),
    "state": state_plot_df["RigState"].astype(str).tolist(),
    "BIT_DEPTH_SM": clean_json(state_plot_df["BIT_DEPTH_SM"].tolist()),
    "HOLE_DEPTH_SM": clean_json(state_plot_df["HOLE_DEPTH_SM"].tolist()),
}

mse_payload = {
    "time": mse_plot_df["FSM_TIMESTAMP"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
    "teale": clean_json(mse_plot_df["MSE_TEALE_KSI_SM"].tolist()),
    "hydraulic": clean_json(mse_plot_df["MSE_HYDRAULIC_KSI_SM"].tolist()),
    "coverage_teale_pct": float(ops["MSE_TEALE_KSI_SM"].notna().mean() * 100.0),
    "coverage_hydraulic_pct": float(ops["MSE_HYDRAULIC_KSI_SM"].notna().mean() * 100.0),
}

interval_payload=[]
for _,r in state_intervals.iterrows():
    interval_payload.append({"start":clean_json(r.Start_Time),"end":clean_json(r.End_Time),"state":str(r.RigState),"duration_sec":float(r.Duration_sec),"start_depth":float(r.Start_Hole_Depth),"end_depth":float(r.End_Hole_Depth),"drilled_ft":max(0.0,float(r.End_Hole_Depth)-float(r.Start_Hole_Depth))})

payload=clean_json({
    "well_name":WELL_NAME,"main":main_payload,"state_track":state_track_payload,"mse":mse_payload,"overall":unified_overall_summary,"intervals":interval_payload,
    "hourly":fsm_hourly_summary,"daily":fsm_daily_summary,"weekly":fsm_weekly_summary,"monthly":fsm_monthly_summary,
    "stands":fsm_stand_summary,"stand_states":fsm_stand_state_distribution,
    "bit_runs":fsm_bitrun_performance,"bitrun_states":fsm_bitrun_state_distribution,
    "hole_sections":DEFAULT_HOLE_SECTION_ROWS,"lithology_rows":DEFAULT_LITHOLOGY_ROWS,"state_colors":STATE_COLORS,
    "start_time":clean_json(ops.FSM_TIMESTAMP.min()),"end_time":clean_json(ops.FSM_TIMESTAMP.max())
})

# Save analytical CSVs.
unified_overall_summary.to_csv(OUTPUT_DIR/"fsm_operational_state_summary.csv",index=False)
fsm_detected_bit_runs.to_csv(OUTPUT_DIR/"fsm_detected_bit_runs.csv",index=False)
fsm_bitrun_state_distribution.to_csv(OUTPUT_DIR/"fsm_bitrun_state_distribution.csv",index=False)
fsm_bitrun_performance.to_csv(OUTPUT_DIR/"fsm_bitrun_performance.csv",index=False)
fsm_stand_summary.to_csv(OUTPUT_DIR/"fsm_stand_by_stand_summary.csv",index=False)
fsm_stand_state_distribution.to_csv(OUTPUT_DIR/"fsm_stand_by_stand_state_distribution.csv",index=False)
fsm_stand_gap_report.to_csv(OUTPUT_DIR/"fsm_stand_gap_report.csv",index=False)
fsm_stand_continuity_report.to_csv(OUTPUT_DIR/"fsm_stand_continuity_report.csv",index=False)

# -------------------------
# 14F. One self-contained HTML
# -------------------------
html_template=r'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Unified FSM Operational Dashboard</title><script>__PLOTLY_JS__</script>
<style>
body{font-family:Arial,sans-serif;margin:14px;background:#fff;color:#172033}.panel{border:1px solid #d0d7de;border-radius:10px;padding:12px;margin:12px 0;background:#fbfdff}.grid2{display:grid;grid-template-columns:1.05fr .95fr;gap:14px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.section-card-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.section-card{display:grid;grid-template-columns:minmax(260px,.8fr) minmax(320px,1.2fr);gap:10px;align-items:start}.section-card .plot{height:360px}.overall-compact{display:grid;grid-template-columns:minmax(360px,560px) minmax(520px,1fr);gap:18px;align-items:start}.overall-compact .plot{height:430px}.bitrun-pie-grid{display:grid;grid-template-columns:repeat(5,minmax(180px,1fr));gap:8px}.shared-state-legend{display:flex;flex-wrap:wrap;gap:8px 10px;padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;background:#fff;margin:10px 0;font-size:12px;position:relative;z-index:2}.shared-state-legend span{white-space:nowrap;border-radius:14px;padding:4px 8px;font-weight:700;border:1px solid rgba(0,0,0,.10)}.shared-state-legend i{display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px;border-radius:2px}.stand-pie-grid{display:grid;grid-template-columns:repeat(5,minmax(210px,1fr));gap:12px;align-items:start;margin-bottom:24px}.stand-pie-card{padding:10px;margin:0;min-height:390px;display:flex;flex-direction:column;overflow:hidden}.stand-pie-card .plot{height:270px;min-height:270px;flex:0 0 270px}.stand-pie-card .metric{margin-top:12px;min-height:82px;flex:0 0 auto}.stand-table-panel{clear:both;margin-top:28px!important;position:relative;z-index:1}.stand-table-panel .scroll{max-height:420px}.stand-check-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:6px 12px;max-height:300px;overflow:auto;border:1px solid #d0d7de;border-radius:8px;padding:10px;background:#fff}.stand-check-item{display:flex;align-items:flex-start;gap:7px;padding:5px;border-bottom:1px solid #eef2f6;font-size:12px}.stand-check-item input{margin-top:2px}.plot-title-spacer{height:8px}.plotly-compact-label{font-size:11px}.bitrun-pie-card{padding:7px;margin:0}.bitrun-pie-card .plot{height:285px}.bitrun-check-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:6px 12px;max-height:340px;overflow:auto;border:1px solid #d0d7de;border-radius:8px;padding:10px;background:#fff}.bitrun-check-item{display:flex;align-items:flex-start;gap:7px;padding:5px;border-bottom:1px solid #eef2f6;font-size:12px}.bitrun-check-item input{margin-top:2px}.stand-dominant{display:inline-block;padding:2px 7px;border-radius:10px;color:#fff;font-size:11px;font-weight:bold}button,select,input{padding:5px 7px;margin:3px;border:1px solid #b8c2cc;border-radius:5px;background:#fff}table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}th,td{border:1px solid #d0d7de;padding:5px;text-align:left}th{background:#eef4ff;position:sticky;top:0}.scroll{max-height:320px;overflow:auto}.empty-cell{color:#8a94a3}.table-note{font-size:11px;color:#5f6b7a;margin:4px 0}.plot{width:100%;height:620px}.bigplot{width:100%;height:2500px}.bitrunplot{width:100%;height:1180px;margin-top:0}.plot-heading{text-align:center;margin:10px 0 0;padding:0}.plot-heading h3{margin:0;font-size:24px;font-weight:600;color:#263238}.plot-heading p{margin:6px 0 0;font-size:13px;color:#5f6b7a}.fsm-state-chip-wrap{display:flex;flex-wrap:wrap;gap:7px 9px;margin:8px 0 12px;padding:10px;border:1px solid #d0d7de;border-radius:8px;background:#fff}.fsm-state-chip-bottom{margin-top:6px}.fsm-legend-caption{font-size:12px;font-weight:700;color:#475569;margin:10px 2px 0}.fsm-state-chip{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:14px;font-size:12px;font-weight:700;color:#fff;white-space:nowrap;box-shadow:0 1px 2px rgba(0,0,0,.08)}.status{padding:9px;border:1px solid #d0d7de;border-radius:7px;background:#f5f9ff}.tabs{display:none}.tab{display:block;margin-top:18px}.tab.active{display:block}.section-title{margin:22px 0 8px;border-bottom:2px solid #d0d7de;padding-bottom:6px}.small{font-size:12px;color:#5f6b7a}.editable input{width:95%;box-sizing:border-box}.metric{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:8px}.error{background:#fff1f1;border-color:#d93025;color:#8a1f11}
/* Shared readable styling for dashboard controls and exported plots. */
body{font-size:17px}.panel{padding:14px}.small,.table-note,.fsm-legend-caption{font-size:15px}.fsm-state-chip,.shared-state-legend,.stand-check-item,.bitrun-check-item{font-size:15px}button,select,input{font-size:16px;padding:7px 10px}table{font-size:15px}th,td{padding:7px}.plot-heading p{font-size:16px}
@media(max-width:1400px){.bitrun-pie-grid,.stand-pie-grid{grid-template-columns:repeat(4,minmax(180px,1fr))}.section-card-grid{grid-template-columns:1fr}}@media(max-width:900px){.bitrun-pie-grid,.stand-pie-grid{grid-template-columns:repeat(2,minmax(180px,1fr))}.grid2,.overall-compact,.section-card{grid-template-columns:1fr}}</style></head><body>
<h2 style="text-align:center">Unified FSM Rig-State, Operational and Bit-Run Dashboard</h2><div id="status" class="status">Loading dashboard...</div>
<div class="panel" id="inspectionControls">
<h3 style="margin:0 0 8px">FSM inspection controls</h3>
<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
<label>Start time <input id="fsmStartTime" type="datetime-local" step="1"></label>
<label>End time <input id="fsmEndTime" type="datetime-local" step="1"></label><button onclick="applyTimeRange()">Apply time range</button>
<label>Start day <input id="fsmStartDay" type="number" step="0.01" style="width:95px"></label>
<label>End day <input id="fsmEndDay" type="number" step="0.01" style="width:95px"></label><button onclick="applyDayRange()">Apply day range</button>
<label>Depth from <input id="fsmDepthFrom" type="number" step="10" style="width:95px"></label>
<label>Depth to <input id="fsmDepthTo" type="number" step="10" style="width:95px"></label><button onclick="applyDepthRange()">Find time range by depth</button>
<button onclick="resetFSMRange()">Reset all</button>
</div>
<div style="margin-top:8px"><b>Visible states:</b> <button onclick="setAllStateVisibility(true)">Select all</button><button onclick="setAllStateVisibility(false)">Clear</button><span id="stateChecks"></span></div>
<div id="fsmHoverReadout" class="small" style="margin-top:8px">Hover over the days-vs-depth plot to inspect day, time, measured depth and state.</div>
</div>
<div class="panel"><h3 class="section-title">FSM Rig-State Detection Overview</h3><div id="mainPlot" class="bigplot"></div><div class="fsm-legend-caption">Rig-state color key</div><div id="mainStateLegend" class="fsm-state-chip-wrap fsm-state-chip-bottom"></div></div>
<div class="panel"><h3 class="section-title">Automatic Bit-Run, Hole-Section and Lithology Overview</h3><div class="plot-heading"><h3>Final Detected Bit Runs — Primary, Secondary and Zero-Depth</h3><p>Hover over a marker for run type, time interval, measured-depth interval, footage and duration. Click a run marker to add or remove it from comparison.</p></div><div id="bitRunDepthPlot" class="bitrunplot"></div></div>
<div class="panel"><button onclick="saveEditedHtml()">Save edited HTML</button><span class="small">All sections are displayed below; no tabs or pop-ups are required.</span></div>
<div id="tab-overall" class="tab"><h2 class="section-title">Overall FSM Statistics</h2><div class="panel overall-compact"><div id="overallPie" class="plot"></div><div class="scroll"><table id="overallTable"></table></div></div></div>
<div id="tab-sections" class="tab"><h2 class="section-title">Hole-Section Statistics</h2><div class="panel"><h3>Editable Hole / Casing Sections</h3><button onclick="addSection()">Add section</button><button onclick="renderSections();updateMainOverlays();generateSectionCards()">Apply changes</button><button onclick="exportSections()">Export CSV</button><div class="scroll"><table id="sectionTable" class="editable"></table></div></div><div id="sectionStateLegend" class="shared-state-legend"></div><div id="sectionCards" class="section-card-grid"></div></div>
<div id="tab-time" class="tab"><h2 class="section-title">Hourly / Daily / Weekly / Monthly Statistics</h2><div class="panel"><select id="periodType"><option value="hourly">Hourly</option><option value="daily" selected>Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option></select><input id="timeStart" type="datetime-local"><input id="timeEnd" type="datetime-local"><button onclick="updateTime()">Apply</button></div><div class="panel grid2"><div id="timePie" class="plot"></div><div class="scroll"><table id="timeTable"></table></div></div><div class="panel"><div id="timeBars" class="plot"></div></div></div>
<div id="tab-stands" class="tab"><h2 class="section-title">Validated Stand-by-Stand Statistics</h2><div class="panel"><b>Select stands for comparison:</b><div><button onclick="selectAllStands(true)">Select all</button><button onclick="selectAllStands(false)">Clear</button><button onclick="selectFastestStands()">Fastest five</button><button onclick="selectSlowestStands()">Slowest five</button></div><div id="standChecks" class="stand-check-grid"></div></div><div id="standSharedLegend" class="shared-state-legend"></div><div id="standComparePies" class="stand-pie-grid"></div><div class="panel stand-table-panel"><h3 style="margin:0 0 10px">Selected-Stand State Statistics</h3><div class="scroll"><table id="standCompareTable"></table></div></div><div class="panel"><div id="standBars" class="plot"></div></div></div>
<div id="tab-bitruns" class="tab"><h2 class="section-title">Automatic Bit-Run Statistics</h2><div class="panel"><h3>Automatically Detected Bit Runs</h3><div class="small">Detection uses a 150 m deep-in-hole boundary, near-surface backtracking and sustained surface-gap separation. Edit names/bit size/notes; intervals remain automatically detected.</div><button onclick="exportBitRuns()">Export bit-run CSV</button><div class="scroll"><table id="bitRunTable" class="editable"></table></div></div><div class="panel"><b>Select bit runs for state-distribution comparison:</b><div style="margin:6px 0"><button onclick="selectAllBitRuns(true)">Select all comparable</button><button onclick="selectAllBitRuns(false)">Clear</button><button onclick="selectBitRunClass('primary_depth_extension_run')">Primary only</button><button onclick="selectBitRunClass('secondary_depth_extension_run')">Secondary only</button><button onclick="selectBitRunClass('zero_depth_service_run')">Zero-depth only</button></div><div id="bitRunChecks" class="bitrun-check-grid"></div></div><div id="bitRunSharedLegend" class="shared-state-legend"></div><div id="bitRunComparePies" class="bitrun-pie-grid"></div><div class="panel"><div class="scroll"><table id="bitRunStateTable"></table></div></div><div class="panel"><div id="bitRunPerformance" class="plot"></div></div></div>
<div id="tab-lithology" class="tab"><h2 class="section-title">Lithology / Formation Inputs</h2><div class="panel"><h3>Editable Lithology / Formation Intervals</h3><button onclick="addLithology()">Add interval</button><button onclick="renderLithology();updateMainOverlays()">Apply overlay</button><button onclick="exportLithology()">Export CSV</button><div class="scroll"><table id="lithologyTable" class="editable"></table></div></div></div>
<script>
const P=__PAYLOAD__;let sections=JSON.parse(JSON.stringify(P.hole_sections||[])),lithology=JSON.parse(JSON.stringify(P.lithology_rows||[])),bitRuns=JSON.parse(JSON.stringify(P.bit_runs||[]));const C=P.state_colors||{};
function err(m){const e=document.getElementById('status');e.className='status error';e.innerHTML=m}window.onerror=(m,s,l,c)=>{err('JavaScript error: '+m+' at line '+l+':'+c);return false};
function showTab(n){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+n).classList.add('active');setTimeout(()=>window.dispatchEvent(new Event('resize')),80)}
function esc(v){return String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}function num(v,d=2){v=Number(v);return isFinite(v)?v.toFixed(d):''}function color(s){return C[s]||'#8c8c8c'}
function stateSummary(rows){const m={};rows.forEach(r=>{const st=r.state||r.RigState,d=Number(r.duration_sec??r.Duration_sec??0),ft=Number(r.drilled_ft??r.Drilled_ft??0);if(!m[st])m[st]={duration_sec:0,drilled_ft:0};m[st].duration_sec+=d;m[st].drilled_ft+=ft});const t=Object.values(m).reduce((a,b)=>a+b.duration_sec,0);return Object.entries(m).map(([state,v])=>({state,duration_sec:v.duration_sec,duration_hr:v.duration_sec/3600,drilled_ft:v.drilled_ft,percentage:t?v.duration_sec/t*100:0})).sort((a,b)=>b.duration_sec-a.duration_sec)}
function pie(id,rows,title,opt={}){const isBit=String(id).startsWith('bitPie'),isOverall=id==='overallPie',isSection=String(id).startsWith('secPie');const h=isBit?285:(isSection?350:(isOverall?410:500));const showLegend=opt.showLegend!==undefined?opt.showLegend:(!isBit&&!isSection);const pct=rows.map(r=>Number(r.percentage||0));const txt=pct.map(v=>v>=3?num(v,1)+'%':'');const el=document.getElementById(id);if(el&&el.data)Plotly.purge(el);Plotly.react(id,[{type:'pie',labels:rows.map(r=>r.state),values:rows.map(r=>r.duration_sec),text:txt,textinfo:'text',textposition:'inside',sort:false,hole:0,marker:{colors:rows.map(r=>color(r.state)),line:{color:'#fff',width:1.5}},hovertemplate:'<b>%{label}</b><br>%{value:.0f} sec<br>%{percent}<extra></extra>',insidetextorientation:'auto',textfont:{size:15}}],{template:'plotly_white',font:{size:15},height:h,title:{text:title,x:.5,font:{size:isBit?17:20}},margin:{l:8,r:showLegend?150:8,t:55,b:22},showlegend:showLegend,legend:{orientation:'v',x:1.0,y:1,font:{size:15}},uniformtext:{minsize:13,mode:'hide'}},{responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'png',scale:3}})}
function statTable(id,rows){document.getElementById(id).innerHTML='<thead><tr><th>State</th><th>Hours</th><th>Minutes</th><th>Percentage</th><th>Drilled depth (ft)</th></tr></thead><tbody>'+rows.map(r=>{const v=Number(r.drilled_ft);const cell=isFinite(v)?num(v,1):'<span class="empty-cell">0.0</span>';return `<tr><td>${esc(r.state)}</td><td>${num(r.duration_hr,2)}</td><td>${num(r.duration_sec/60,1)}</td><td>${num(r.percentage,2)}%</td><td>${cell}</td></tr>`}).join('')+'</tbody>'}
function sharedStateLegend(id,states){const root=document.getElementById(id);if(!root)return;root.innerHTML=[...new Set(states.filter(Boolean))].map(st=>{const c=color(st);return `<span style="background:${c}22;color:${c};border-color:${c}66"><i style="background:${c}"></i>${esc(st)}</span>`}).join('')}
function mainPlot(){
 const M=P.main,S=P.state_track||M,states=[...new Set(S.state)],code={};states.forEach((st,i)=>code[st]=i);
 const tr=[];
 tr.push({x:M.days,y:M.HOLE_DEPTH_SM,type:'scattergl',mode:'lines',name:'Hole Depth',line:{color:'#4169e1',width:4.5},xaxis:'x',yaxis:'y',hovertemplate:'Day %{x:.3f}<br>Hole depth %{y:.1f} ft<extra></extra>'});
 // One WebGL trace per state, separated by nulls. The previous implementation
 // created thousands of traces (one per interval), which made a single state
 // checkbox issue thousands of Plotly updates and freeze the page.
 const stateSeries={};states.forEach(st=>stateSeries[st]={x:[],y:[],cd:[]});
 let last=0;
 for(let i=1;i<=S.state.length;i++){
   if(i===S.state.length||S.state[i]!==S.state[last]){
     const st=S.state[last],idx=[];
     if(last>0)idx.push(last-1);
     for(let j=last;j<i;j++)idx.push(j);
     if(i<S.state.length)idx.push(i);
     const z=stateSeries[st];
     idx.forEach(j=>{z.x.push(S.days[j]);z.y.push(S.BIT_DEPTH_SM[j]);z.cd.push([S.time[j],S.HOLE_DEPTH_SM[j],st])});
     z.x.push(null);z.y.push(null);z.cd.push(null);
     last=i;
   }
 }
 states.forEach(st=>{const z=stateSeries[st];tr.push({x:z.x,y:z.y,type:'scattergl',mode:'lines',name:st,legendgroup:st,meta:{state:st},showlegend:true,connectgaps:false,line:{color:color(st),width:(st==='Trip-In'||st==='Trip-Out'?6:5)},customdata:z.cd,hovertemplate:'State: %{customdata[2]}<br>Day: %{x:.3f}<br>Time: %{customdata[0]}<br>Bit depth: %{y:.1f} ft<br>Hole depth: %{customdata[1]:.1f} ft<extra></extra>'})});
 const baseMs=new Date(P.start_time).getTime(),connections=(P.intervals||[]).filter(r=>String(r.state)==='Connection');
 const connectionMidDays=connections.map(r=>((new Date(r.start).getTime()+new Date(r.end).getTime())/2-baseMs)/86400000);
 // P.start_time, interval boundaries, and state-track timestamps are exported
 // without a timezone suffix.  Converting their midpoint with toISOString()
 // changes the wall-clock time by the browser's UTC offset and visibly moves
 // the label away from its shaded connection window.  Format the midpoint as
 // a timezone-naive local timestamp so every connection layer uses the same
 // x-coordinate convention.
 const pad2=v=>String(v).padStart(2,'0');
 const plotTimestamp=ms=>{const d=new Date(ms);return d.getFullYear()+'-'+pad2(d.getMonth()+1)+'-'+pad2(d.getDate())+'T'+pad2(d.getHours())+':'+pad2(d.getMinutes())+':'+pad2(d.getSeconds())};
 const connectionMidTimes=connections.map(r=>plotTimestamp((new Date(r.start).getTime()+new Date(r.end).getTime())/2));
 // Keep the overview scientifically readable: connection events are markers
 // only here.  Their persistent text labels belong exclusively to the
 // categorical rig-state track at the bottom of the linked dashboard.
 tr.push({x:connectionMidDays,y:connections.map(r=>Number(r.end_depth)),type:'scattergl',mode:'markers',name:'Connection events',legendgroup:'Connection',meta:{state:'Connection'},showlegend:false,marker:{symbol:'diamond-open',size:15,color:color('Connection'),line:{width:3,color:color('Connection')}},customdata:connections.map(r=>[r.start,r.end,r.duration_sec,r.end_depth]),hovertemplate:'<b>Connection</b><br>Start: %{customdata[0]}<br>End: %{customdata[1]}<br>Duration: %{customdata[2]:.0f} sec<br>MD: %{customdata[3]:.1f} ft<extra></extra>'});
 const tracks=[
  ['HOLE_DEPTH_SM','BIT_DEPTH_SM','Hole Depth','Bit Depth','y2','y3'],
  ['WOB_SM','HOOK_LOAD_SM','WOB','Hook Load','y4','y5'],
  ['ROTARY_RPM_SM','ROP_SM','Rotary RPM','ROP','y6','y7'],
  ['PUMP_OUTPUT_SM','STANDPIPE_PRESSURE_SM','Pump Output','SPP','y8','y9'],
  ['ROTARY_TORQUE_SM','DIFF_PRESSURE_SM','Rotary Torque','Differential Pressure','y10','y11'],
  ['BLOCK_POSITION_SM','BIT_DEPTH_SM','Block Position','Bit Depth','y12','y13']];
 tracks.forEach((a,j)=>{const xa='x'+(j+2);tr.push({x:M.time,y:M[a[0]],type:'scattergl',mode:'lines',name:a[2],xaxis:xa,yaxis:a[4],showlegend:false,line:{width:4}});tr.push({x:M.time,y:M[a[1]],type:'scattergl',mode:'lines',name:a[3],xaxis:xa,yaxis:a[5],showlegend:false,line:{width:4}})});
 const MS=P.mse||{time:[],teale:[],hydraulic:[]};
 function gapAware(values){const x=[],y=[];let prev=null;MS.time.forEach((t,i)=>{const ms=new Date(t).getTime();if(prev!==null&&ms-prev>60000){x.push(t);y.push(null)}x.push(t);y.push(values[i]);prev=ms});return {x,y}}
 const mt=gapAware(MS.teale||[]),mh=gapAware(MS.hydraulic||[]);
 tr.push({x:mt.x,y:mt.y,type:'scattergl',mode:'lines',name:'Classical Teale MSE',xaxis:'x8',yaxis:'y14',showlegend:false,connectgaps:false,line:{color:'#5b5bd6',width:4},hovertemplate:'Classical Teale MSE: %{y:.2f} ksi<br>%{x}<extra></extra>'});
 tr.push({x:mh.x,y:mh.y,type:'scattergl',mode:'lines',name:'Hydraulic MSE',xaxis:'x8',yaxis:'y15',showlegend:false,connectgaps:false,line:{color:'#00a884',width:4},hovertemplate:'Hydraulic MSE: %{y:.2f} ksi<br>%{x}<extra></extra>'});
 const ycode=S.state.map(st=>code[st]);tr.push({x:S.time,y:ycode,type:'scattergl',mode:'lines',name:'Rig State Step',line:{color:'#444',width:3,shape:'hv'},xaxis:'x9',yaxis:'y16',showlegend:false,hoverinfo:'skip'});
 states.forEach(st=>{const x=[],y=[];S.state.forEach((v,i)=>{if(v===st){x.push(S.time[i]);y.push(code[st])}});tr.push({x,y,type:'scattergl',mode:'markers',name:'Track '+st,legendgroup:st,meta:{state:st},marker:{size:st==='Connection'?14:10,symbol:st==='Connection'?'diamond':'circle',color:color(st)},xaxis:'x9',yaxis:'y16',showlegend:false,hovertemplate:st+'<br>%{x}<extra></extra>'})});
 // The event rail keeps a clean diamond at each connection.  Filled label
 // boxes are added below as annotations so Connection matches the appearance
 // of every other labeled state in screenshots and exports.
 if(code.Connection!==undefined)tr.push({x:connectionMidTimes,y:connections.map(()=>code.Connection),type:'scatter',mode:'markers',name:'Connection event rail',legendgroup:'Connection',meta:{state:'Connection'},marker:{size:17,symbol:'diamond-open',color:color('Connection'),line:{width:3,color:color('Connection')}},cliponaxis:false,xaxis:'x9',yaxis:'y16',showlegend:false,customdata:connections.map(r=>[r.start,r.end,r.duration_sec]),hovertemplate:'<b>Connection</b><br>%{customdata[0]}–%{customdata[1]}<br>%{customdata[2]:.0f} sec<extra></extra>'});
 // Label only long, well-spaced state segments so labels remain readable.
 const stateAnn=[];let segStart=0,lastLabelMs=-Infinity;
 for(let i=1;i<=S.state.length;i++){
   if(i===S.state.length||S.state[i]!==S.state[segStart]){
     const n=i-segStart,mid=Math.floor((segStart+i-1)/2),midMs=new Date(S.time[mid]).getTime();
     if(n>=10 && midMs-lastLabelMs>=45*60*1000){
       const st=S.state[segStart];stateAnn.push({xref:'x9',yref:'y16',x:S.time[mid],y:code[st],text:esc(st),showarrow:false,bgcolor:color(st),bordercolor:'#ffffff',borderwidth:2,borderpad:4,font:{color:'#fff',size:16},opacity:.95});lastLabelMs=midMs;
     }
     segStart=i;
   }
 }
 // Connection labels use the same filled-box treatment as other state labels.
 // They are confined to the categorical state track and never appear over the
 // depth or parameter plots.
 if(code.Connection!==undefined)connections.forEach((r,i)=>stateAnn.push({
   xref:'x9',yref:'y16',x:connectionMidTimes[i],y:code.Connection,
   text:'<b>Con.</b>',showarrow:false,bgcolor:color('Connection'),
   bordercolor:'#ffffff',borderwidth:1.5,borderpad:2,
   font:{color:'#fff',size:12,family:'Arial Black, Arial, sans-serif'},opacity:.94
 }));
 const connectionShapes=[];
 connections.forEach(r=>{
   const d0=(new Date(r.start).getTime()-baseMs)/86400000,d1=(new Date(r.end).getTime()-baseMs)/86400000;
   connectionShapes.push({type:'rect',xref:'x',yref:'paper',x0:d0,x1:d1,y0:.76,y1:1,fillcolor:'rgba(31,119,180,.16)',line:{color:'rgba(31,119,180,.55)',width:.5},layer:'below',name:'FSM Connection window'});
   connectionShapes.push({type:'rect',xref:'x2',yref:'paper',x0:r.start,x1:r.end,y0:0,y1:.745,fillcolor:'rgba(31,119,180,.08)',line:{color:'rgba(31,119,180,.35)',width:.4},layer:'below',name:'FSM Connection window'});
 });
 stateAnn.push({xref:'paper',yref:'paper',x:.465,y:.162,xanchor:'center',yanchor:'middle',text:'MSE is drilling-only; blank intervals are unavailable by definition. Coverage: Teale '+num(MS.coverage_teale_pct,1)+'%, hydraulic '+num(MS.coverage_hydraulic_pct,1)+'%.',showarrow:false,font:{size:14,color:'#5f6b7a'},bgcolor:'rgba(255,255,255,.94)',borderpad:2});
 const axisTitle=(t,standoff=18)=>({text:'<b>'+t+'</b>',font:{size:22,color:'#172033',family:'Arial Black, Arial, sans-serif'},standoff});
 const layout={template:'plotly_white',height:2500,font:{family:'Arial, sans-serif',size:20,color:'#172033'},title:{text:'FSM Rig State Detection — Linked Overview, Parameters and MSE',x:.5,font:{size:28}},hoverlabel:{font:{size:18}},hovermode:'closest',margin:{l:170,r:220,t:105,b:90},legend:{x:1.005,y:1,font:{size:17}},annotations:stateAnn,shapes:connectionShapes,
 xaxis:{domain:[0,.93],anchor:'y',title:'Days from start',rangeselector:{buttons:[{count:6,label:'6h',step:'hour',stepmode:'backward'},{count:12,label:'12h',step:'hour',stepmode:'backward'},{count:1,label:'1d',step:'day',stepmode:'backward'},{count:3,label:'3d',step:'day',stepmode:'backward'},{step:'all',label:'All'}]}},yaxis:{domain:[.76,1],autorange:'reversed',title:axisTitle('Measured Depth<br>(ft)',18)},
 xaxis2:{domain:[0,.93],anchor:'y2'},yaxis2:{domain:[.645,.725],autorange:'reversed',title:axisTitle('Hole Depth<br>(ft)',18)},yaxis3:{overlaying:'y2',side:'right',autorange:'reversed',title:axisTitle('Bit Depth<br>(ft)',18)},
 xaxis3:{domain:[0,.93],anchor:'y4',matches:'x2'},yaxis4:{domain:[.55,.63],title:axisTitle('Weight on Bit<br>(WOB)',48)},yaxis5:{overlaying:'y4',side:'right',title:axisTitle('Hook Load',48)},
 xaxis4:{domain:[0,.93],anchor:'y6',matches:'x2'},yaxis6:{domain:[.455,.535],title:axisTitle('Rotary Speed<br>(RPM)',18)},yaxis7:{overlaying:'y6',side:'right',title:axisTitle('Rate of Penetration<br>(ROP)',18)},
 xaxis5:{domain:[0,.93],anchor:'y8',matches:'x2'},yaxis8:{domain:[.36,.44],title:axisTitle('Pump Output<br>/ Flow',48)},yaxis9:{overlaying:'y8',side:'right',title:axisTitle('Standpipe Pressure<br>(SPP)',48)},
 xaxis6:{domain:[0,.93],anchor:'y10',matches:'x2'},yaxis10:{domain:[.265,.345],title:axisTitle('Rotary Torque',18)},yaxis11:{overlaying:'y10',side:'right',title:axisTitle('Differential<br>Pressure',18)},
 xaxis7:{domain:[0,.93],anchor:'y12',matches:'x2'},yaxis12:{domain:[.17,.25],title:axisTitle('Block Position',48)},yaxis13:{overlaying:'y12',side:'right',autorange:'reversed',title:axisTitle('Bit Depth<br>(ft)',48)},
 xaxis8:{domain:[0,.93],anchor:'y14',matches:'x2'},yaxis14:{domain:[.075,.155],title:axisTitle('Classical Teale<br>MSE (ksi)',18),type:'log',rangemode:'normal'},yaxis15:{overlaying:'y14',side:'right',title:axisTitle('Hydraulic MSE<br>(ksi)',18),type:'log',matches:'y14',showticklabels:false,showgrid:false},
 xaxis9:{domain:[0,.93],anchor:'y16',matches:'x2',title:'Time'},yaxis16:{domain:[0,.06],title:axisTitle('Detected<br>Rig State',58),tickmode:'array',tickvals:states.map(st=>code[st]),ticktext:states,showticklabels:false,showgrid:true,gridcolor:'#eef2f6',zeroline:false,range:[-.5,states.length-.5]}};
 Plotly.newPlot('mainPlot',tr,layout,{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:3}}).then(()=>{renderStateChecks(states);renderMainStateLegend(states);installAxisSync();installHoverReadout()});
}
function bitRunDepthPlot(){
 const M=P.main,tr=[];const base=new Date(P.start_time).getTime();
 tr.push({x:M.days,y:M.BIT_DEPTH_SM,type:'scattergl',mode:'lines',name:'Bit Depth',line:{color:'royalblue',width:4},hovertemplate:'Day %{x:.3f}<br>Bit depth %{y:.1f} ft<extra></extra>'});
 tr.push({x:M.days,y:M.HOLE_DEPTH_SM,type:'scattergl',mode:'lines',name:'Hole Depth / TD Progress',line:{color:'red',width:4.5},hovertemplate:'Day %{x:.3f}<br>Hole depth %{y:.1f} ft<extra></extra>'});
 const styles={primary_depth_extension_run:['#1a9d35','circle-open','Primary drilling runs','rgba(26,157,53,.055)'],secondary_depth_extension_run:['#00a6d6','diamond-open','Secondary runs','rgba(0,166,214,.065)'],zero_depth_service_run:['#ff8c00','square-open','Zero-depth/service runs','rgba(255,140,0,.065)']};const shapes=[],ann=[];
 Object.keys(styles).forEach(cls=>{const rows=bitRuns.filter(r=>String(r.run_class)===cls),st=styles[cls];rows.forEach(r=>{const x0=(new Date(r.start_time).getTime()-base)/86400000,x1=(new Date(r.end_time).getTime()-base)/86400000;shapes.push({type:'rect',xref:'x',yref:'paper',x0,x1,y0:0,y1:1,fillcolor:st[3],line:{width:0},layer:'below'})});tr.push({x:rows.map(r=>((new Date(r.start_time).getTime()+new Date(r.end_time).getTime())/2-base)/86400000),y:rows.map(r=>cls==='zero_depth_service_run'?Math.max(0,Number(r.depth_out_ft||r.depth_in_ft)):(Number(r.depth_in_ft)+Number(r.depth_out_ft))/2),type:'scatter',mode:'markers',name:st[2],marker:{color:st[0],size:14,symbol:st[1],line:{width:2.5,color:st[0]}},customdata:rows.map(r=>[r.run_key,r.run_name||r.run_key,r.role||r.run_class,r.start_time,r.end_time,r.depth_in_ft,r.depth_out_ft,r.footage_ft,r.duration_hr]),hovertemplate:'<b>%{customdata[1]}</b> (%{customdata[0]})<br>Type: %{customdata[2]}<br>Start: %{customdata[3]}<br>End: %{customdata[4]}<br>Depth in: %{customdata[5]:.1f} ft<br>Depth out: %{customdata[6]:.1f} ft<br>Drilled: %{customdata[7]:.1f} ft<br>Duration: %{customdata[8]:.2f} hr<extra></extra>'})});
 sections.forEach((r,idx)=>{const top=Number(r.top_depth_ft),baseD=Number(r.base_depth_ft),cd=Number(r.casing_depth_ft);if(isFinite(top)){shapes.push({type:'line',xref:'x domain',yref:'y',x0:0,x1:1,y0:top,y1:top,line:{dash:'dash',color:'#555'}});ann.push({xref:'paper',yref:'y',x:0.005,y:top,text:'<b>'+esc(r.hole_size||r.section_name||'')+'</b><br>'+num(top,0)+'–'+num(baseD,0)+' ft',showarrow:false,xanchor:'left',bgcolor:'rgba(255,255,255,.78)',font:{size:10}})}if(isFinite(baseD))shapes.push({type:'line',xref:'x domain',yref:'y',x0:0,x1:1,y0:baseD,y1:baseD,line:{dash:'dot',color:'#999'}});if(isFinite(cd))ann.push({xref:'paper',yref:'y',x:.985,y:cd,text:esc(r.casing_size||'')+' Casing<br>'+num(cd,0)+' ft',showarrow:false,xanchor:'right',bgcolor:'rgba(255,255,255,.75)',font:{size:9}})});
 lithology.forEach(r=>{const top=Number(r.top_depth_ft??r.top_depth),baseD=Number(r.base_depth_ft??r.base_depth);if(isFinite(top)&&isFinite(baseD)){shapes.push({type:'rect',xref:'paper',yref:'y',x0:.925,x1:.955,y0:top,y1:baseD,fillcolor:r.color||'rgba(210,180,120,.65)',line:{color:'#555',width:.5}});ann.push({xref:'paper',yref:'y',x:.94,y:(top+baseD)/2,text:esc(r.formation||r.lithology||''),showarrow:false,font:{size:8}})}});
 const gd=document.getElementById('bitRunDepthPlot');if(gd&&gd.data)Plotly.purge(gd);
 Plotly.newPlot('bitRunDepthPlot',tr,{template:'plotly_white',font:{size:18},height:1080,title:null,xaxis:{title:{text:'Days from start',font:{size:19}},tickfont:{size:16},domain:[0,.94],rangeslider:{visible:true},rangeselector:{buttons:[{count:1,label:'1d',step:'day',stepmode:'backward'},{count:3,label:'3d',step:'day',stepmode:'backward'},{count:7,label:'7d',step:'day',stepmode:'backward'},{step:'all',label:'All'}]}},yaxis:{title:{text:'Measured Depth (ft)',font:{size:19}},tickfont:{size:16},autorange:'reversed'},hoverlabel:{font:{size:16}},margin:{l:115,r:130,t:120,b:100},legend:{orientation:'h',y:1.07,x:.5,xanchor:'center',yanchor:'bottom',font:{size:16},bgcolor:'rgba(255,255,255,.92)',bordercolor:'#d0d7de',borderwidth:1},shapes,annotations:ann,hovermode:'closest',clickmode:'event+select'},{responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'png',scale:3}}).then(()=>{const p=document.getElementById('bitRunDepthPlot');p.on('plotly_click',ev=>{const cd=ev.points&&ev.points[0]&&ev.points[0].customdata;if(!cd||!cd[0])return;const key=String(cd[0]);if(selectedBitRuns.has(key))selectedBitRuns.delete(key);else selectedBitRuns.add(key);syncBitRunChecks();scheduleBitRunComparison()})});
}
function installAxisSync(){const gd=document.getElementById('mainPlot');if(!gd||gd._fsmSyncInstalled)return;gd._fsmSyncInstalled=true;let busy=false;const base=new Date(P.start_time).getTime();const axes=['xaxis2','xaxis3','xaxis4','xaxis5','xaxis6','xaxis7','xaxis8','xaxis9'];gd.on('plotly_relayout',ev=>{if(busy)return;let upd={};if(ev['xaxis.range[0]']!==undefined&&ev['xaxis.range[1]']!==undefined){const d0=Number(ev['xaxis.range[0]']),d1=Number(ev['xaxis.range[1]']);if(isFinite(d0)&&isFinite(d1)){const t0=new Date(base+d0*86400000).toISOString(),t1=new Date(base+d1*86400000).toISOString();axes.forEach(a=>upd[a+'.range']=[t0,t1]);document.getElementById('fsmStartDay').value=d0.toFixed(3);document.getElementById('fsmEndDay').value=d1.toFixed(3);document.getElementById('fsmStartTime').value=t0.slice(0,19);document.getElementById('fsmEndTime').value=t1.slice(0,19)}}else{for(const ax of axes){if(ev[ax+'.range[0]']!==undefined&&ev[ax+'.range[1]']!==undefined){const t0=new Date(ev[ax+'.range[0]']).getTime(),t1=new Date(ev[ax+'.range[1]']).getTime();if(isFinite(t0)&&isFinite(t1)){upd['xaxis.range']=[(t0-base)/86400000,(t1-base)/86400000];axes.filter(a=>a!==ax).forEach(a=>upd[a+'.range']=[new Date(t0).toISOString(),new Date(t1).toISOString()])}break}}}if(Object.keys(upd).length){busy=true;Plotly.relayout(gd,upd).finally(()=>busy=false)}})}
function renderStateChecks(states){const root=document.getElementById('stateChecks');root.innerHTML=states.map(st=>{const c=color(st);return `<label style="display:inline-flex;align-items:center;gap:6px;margin:4px 6px;padding:6px 10px;border-radius:15px;background:${c};color:#fff;font-weight:700;font-size:16px"><input class="stateVis" type="checkbox" value="${esc(st)}" checked onchange="applyStateVisibility()" style="margin:0">${esc(st)}</label>`}).join('')}
function renderMainStateLegend(states){const root=document.getElementById('mainStateLegend');if(!root)return;root.innerHTML=states.map(st=>`<span class="fsm-state-chip" style="background:${color(st)}">${esc(st)}</span>`).join('')}
function applyStateVisibility(){const gd=document.getElementById('mainPlot'),wanted=new Set([...document.querySelectorAll('.stateVis:checked')].map(x=>x.value)),idx=[],vis=[];gd.data.forEach((t,i)=>{if(t.meta&&t.meta.state){idx.push(i);vis.push(wanted.has(t.meta.state)?true:'legendonly')}});const tasks=[];if(idx.length)tasks.push(Plotly.restyle(gd,'visible',vis,idx));const shapeUpdate={};(gd.layout.shapes||[]).forEach((s,i)=>{if(s.name==='FSM Connection window')shapeUpdate['shapes['+i+'].visible']=wanted.has('Connection')});if(Object.keys(shapeUpdate).length)tasks.push(Plotly.relayout(gd,shapeUpdate));return Promise.all(tasks)}
function setAllStateVisibility(v){document.querySelectorAll('.stateVis').forEach(x=>x.checked=v);applyStateVisibility()}
function installHoverReadout(){const gd=document.getElementById('mainPlot');gd.on('plotly_hover',ev=>{const p=ev.points&&ev.points[0];if(!p)return;const el=document.getElementById('fsmHoverReadout');const cd=p.customdata;el.innerHTML=cd&&cd.length>=3?`Day: <b>${num(p.x,3)}</b> | Time: <b>${esc(cd[0])}</b> | Bit depth: <b>${num(p.y,1)} ft</b> | Hole depth: <b>${num(cd[1],1)} ft</b> | State: <b>${esc(cd[2])}</b>`:`Trace: <b>${esc(p.data.name)}</b> | X: <b>${esc(p.x)}</b> | Y: <b>${num(p.y,1)}</b>`})}
function _setAllTimeAxes(t0,t1){const gd=document.getElementById('mainPlot'),base=new Date(P.start_time).getTime(),d0=(new Date(t0).getTime()-base)/86400000,d1=(new Date(t1).getTime()-base)/86400000,upd={'xaxis.range':[d0,d1]};['xaxis2','xaxis3','xaxis4','xaxis5','xaxis6','xaxis7','xaxis8','xaxis9'].forEach(a=>upd[a+'.range']=[new Date(t0).toISOString(),new Date(t1).toISOString()]);Plotly.relayout(gd,upd)}
function applyTimeRange(){const a=document.getElementById('fsmStartTime').value,b=document.getElementById('fsmEndTime').value;if(a&&b)_setAllTimeAxes(a,b)}
function applyDayRange(){const a=Number(document.getElementById('fsmStartDay').value),b=Number(document.getElementById('fsmEndDay').value),base=new Date(P.start_time).getTime();if(isFinite(a)&&isFinite(b))_setAllTimeAxes(new Date(base+a*86400000),new Date(base+b*86400000))}
function applyDepthRange(){const a=Number(document.getElementById('fsmDepthFrom').value),b=Number(document.getElementById('fsmDepthTo').value),lo=Math.min(a,b),hi=Math.max(a,b),idx=[];(P.main.HOLE_DEPTH_SM||[]).forEach((v,i)=>{if(Number(v)>=lo&&Number(v)<=hi)idx.push(i)});if(idx.length)_setAllTimeAxes(P.main.time[idx[0]],P.main.time[idx[idx.length-1]])}
function resetFSMRange(){const gd=document.getElementById('mainPlot');Plotly.relayout(gd,{'xaxis.autorange':true,'xaxis2.autorange':true,'xaxis3.autorange':true,'xaxis4.autorange':true,'xaxis5.autorange':true,'xaxis6.autorange':true,'xaxis7.autorange':true,'xaxis8.autorange':true});document.getElementById('fsmStartDay').value='';document.getElementById('fsmEndDay').value=''}
function updateMainOverlays(){bitRunDepthPlot()}
function renderSections(){const keys=['section_name','hole_size','casing_size','top_depth_ft','base_depth_ft','casing_depth_ft','notes'];document.getElementById('sectionTable').innerHTML='<thead><tr><th>Delete</th>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+sections.map((r,i)=>'<tr><td><button onclick="sections.splice('+i+',1);renderSections();updateMainOverlays();generateSectionCards()">Delete</button></td>'+keys.map(k=>`<td><input value="${esc(r[k]??'')}" onchange="sections[${i}]['${k}']=this.value"></td>`).join('')+'</tr>').join('')+'</tbody>'}
function addSection(){sections.push({section_name:'',hole_size:'',casing_size:'',top_depth_ft:'',base_depth_ft:'',casing_depth_ft:'',notes:''});renderSections()}
function generateSectionCards(){const root=document.getElementById('sectionCards');root.innerHTML='';const allStates=[];sections.forEach((r,i)=>{const top=Number(r.top_depth_ft),base=Number(r.base_depth_ft);if(!isFinite(top)||!isFinite(base))return;const rows=stateSummary((P.intervals||[]).filter(x=>((Number(x.start_depth)+Number(x.end_depth))/2)>=Math.min(top,base)&&((Number(x.start_depth)+Number(x.end_depth))/2)<Math.max(top,base)));rows.forEach(x=>allStates.push(x.state));const d=document.createElement('div');d.className='panel section-card';d.innerHTML=`<div id="secPie${i}" class="plot"></div><div><h3>${esc(r.section_name||r.hole_size)} | ${esc(r.hole_size)} | Casing ${esc(r.casing_size)}</h3><p><b>Measured-depth interval:</b> ${num(top,0)}–${num(base,0)} ft</p><div class="scroll"><table id="secTab${i}"></table></div></div>`;root.appendChild(d);pie('secPie'+i,rows,(r.section_name||r.hole_size)+' State Distribution',{showLegend:false});statTable('secTab'+i,rows)});sharedStateLegend('sectionStateLegend',allStates)}
function renderLithology(){const keys=['top_depth_ft','base_depth_ft','lithology','formation','color','notes'];document.getElementById('lithologyTable').innerHTML='<thead><tr><th>Delete</th>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+lithology.map((r,i)=>'<tr><td><button onclick="lithology.splice('+i+',1);renderLithology();updateMainOverlays()">Delete</button></td>'+keys.map(k=>`<td><input value="${esc(r[k]??'')}" onchange="lithology[${i}]['${k}']=this.value"></td>`).join('')+'</tr>').join('')+'</tbody>'}
function addLithology(){lithology.push({top_depth_ft:'',base_depth_ft:'',lithology:'',formation:'',color:'#d8c090',notes:''});renderLithology()}
function updateTime(){const typ=document.getElementById('periodType').value,a=new Date(document.getElementById('timeStart').value),b=new Date(document.getElementById('timeEnd').value),raw=P[typ]||[],f=raw.filter(r=>{const t=new Date(r.Period);return t>=a&&t<=b}),rows=stateSummary(f);pie('timePie',rows,typ+' selected range');statTable('timeTable',rows);const sts=[...new Set(f.map(r=>r.RigState))],tr=sts.map(s=>({type:'bar',name:s,x:f.filter(r=>r.RigState===s).map(r=>r.Period),y:f.filter(r=>r.RigState===s).map(r=>Number(r.Duration_hr)),marker:{color:color(s)}}));Plotly.react('timeBars',tr,{template:'plotly_white',barmode:'stack',title:{text:typ+' State Hours',x:.5},yaxis:{title:'Hours'},legend:{orientation:'h'}},{responsive:true})}
let selectedStands=new Set();let standUpdateTimer=null;
function syncStandChecks(){document.querySelectorAll('#standChecks input[type="checkbox"]').forEach(x=>x.checked=selectedStands.has(String(x.value)))}
function scheduleStandComparison(){clearTimeout(standUpdateTimer);standUpdateTimer=setTimeout(updateStandComparison,80)}
function initStands(){const root=document.getElementById('standChecks'),rows=P.stands||[];if(!root)return;const selectableRows=rows.filter(r=>r.Selectable!==false),valid=new Set(selectableRows.map(r=>String(r.Stand)));selectedStands=new Set([...selectedStands].filter(k=>valid.has(k)));if(!selectedStands.size)selectableRows.slice(0,Math.min(5,selectableRows.length)).forEach(r=>selectedStands.add(String(r.Stand)));root.innerHTML=rows.map(r=>{const ok=r.Selectable!==false,status=r.Validation_Status||'Validated',bg=ok?'':'background:#fff7e6;color:#7a4b00;';return `<label class="stand-check-item" style="${bg}"><input type="checkbox" value="${esc(r.Stand)}" ${selectedStands.has(String(r.Stand))?'checked':''} ${ok?'':'disabled'} onchange="toggleStandSelection('${esc(r.Stand)}',this.checked)"><span><b>${esc(r.Stand)}</b> — ${esc(status)}<br>MD ${num(r.Start_MD_ft,0)}–${num(r.End_MD_ft,0)} ft | drilling ${num(r.Drilling_hr,2)} hr | fraction ${num(100*Number(r.Drilling_Fraction||0),0)}% | sensor ROP ${num(r.Avg_ROP_ft_hr,1)} ft/hr<br><span class="small">${esc(r.Boundary_Source||'')}</span></span></label>`}).join('');scheduleStandComparison();drawStandPerformance()}
function toggleStandSelection(k,v){if(v)selectedStands.add(String(k));else selectedStands.delete(String(k));scheduleStandComparison()}
function selectAllStands(v){selectedStands=new Set(v?(P.stands||[]).filter(r=>r.Selectable!==false).map(r=>String(r.Stand)):[]);syncStandChecks();scheduleStandComparison()}
function selectFastestStands(){const rows=[...(P.stands||[])].filter(r=>r.Selectable!==false&&Number(r.Avg_ROP_ft_hr)>0).sort((a,b)=>Number(b.Avg_ROP_ft_hr)-Number(a.Avg_ROP_ft_hr)).slice(0,5);selectedStands=new Set(rows.map(r=>String(r.Stand)));syncStandChecks();scheduleStandComparison()}
function selectSlowestStands(){const rows=[...(P.stands||[])].filter(r=>r.Selectable!==false&&Number(r.Avg_ROP_ft_hr)>0).sort((a,b)=>Number(a.Avg_ROP_ft_hr)-Number(b.Avg_ROP_ft_hr)).slice(0,5);selectedStands=new Set(rows.map(r=>String(r.Stand)));syncStandChecks();scheduleStandComparison()}
function updateStandComparison(){const keys=[...selectedStands],root=document.getElementById('standComparePies');root.querySelectorAll('.js-plotly-plot').forEach(x=>Plotly.purge(x));root.innerHTML='';const tableRows=[],allStates=[];keys.forEach((k,i)=>{const meta=(P.stands||[]).find(r=>String(r.Stand)===String(k)),rows=(P.stand_states||[]).filter(r=>String(r.Stand)===String(k)).map(r=>({state:r.RigState,duration_sec:Number(r.Duration_sec),duration_hr:Number(r.Duration_hr),percentage:Number(r.Percentage),drilled_ft:Number(r.Drilled_ft||0)}));rows.forEach(r=>allStates.push(r.state));const card=document.createElement('div');card.className='panel stand-pie-card';card.innerHTML=`<div id="standPie${i}" class="plot"></div><div class="metric"><b>${esc(k)}</b><br>MD: ${num(meta?.Start_MD_ft,1)}–${num(meta?.End_MD_ft,1)} ft<br>Footage: ${num(meta?.Footage_ft,1)} ft<br>Drilling: ${num(meta?.Drilling_hr,2)} hr<br>Avg ROP: ${num(meta?.Avg_ROP_ft_hr,1)} ft/hr</div>`;root.appendChild(card);pie('standPie'+i,rows,k,{showLegend:false});rows.forEach(r=>tableRows.push({stand:k,...r}))});sharedStateLegend('standSharedLegend',allStates);document.getElementById('standCompareTable').innerHTML='<thead><tr><th>Stand</th><th>State</th><th>Hours</th><th>Minutes</th><th>Percentage</th><th>Drilled depth (ft)</th></tr></thead><tbody>'+tableRows.map(r=>`<tr><td>${esc(r.stand)}</td><td>${esc(r.state)}</td><td>${num(r.duration_hr,2)}</td><td>${num(r.duration_sec/60,1)}</td><td>${num(r.percentage,2)}%</td><td>${num(r.drilled_ft,1)}</td></tr>`).join('')+'</tbody>';drawStandPerformance(keys)}
function drawStandPerformance(selectedKeys=null){const chosen=selectedKeys&&selectedKeys.length?new Set(selectedKeys):null,rows=(P.stands||[]).filter(r=>!chosen||chosen.has(String(r.Stand)));Plotly.react('standBars',[{type:'bar',x:rows.map(r=>r.Stand),y:rows.map(r=>Number(r.Drilling_hr||0)),name:'Drilling time (hr)',marker:{color:rows.map(r=>color(r.Dominant_State||'Unknown'))},customdata:rows.map(r=>[r.Start_MD_ft,r.End_MD_ft,r.Footage_ft,r.Avg_ROP_ft_hr,r.Dominant_State,r.Dominant_State_pct]),hovertemplate:'%{x}<br>MD: %{customdata[0]:.1f}–%{customdata[1]:.1f} ft<br>Footage: %{customdata[2]:.1f} ft<br>Drilling time: %{y:.2f} hr<br>Avg ROP: %{customdata[3]:.1f} ft/hr<br>Dominant: %{customdata[4]} (%{customdata[5]:.1f}%)<extra></extra>'},{type:'scatter',mode:'lines+markers',x:rows.map(r=>r.Stand),y:rows.map(r=>Number(r.Avg_ROP_ft_hr||0)),name:'Average ROP (ft/hr)',yaxis:'y2',line:{color:'#ff7f0e',width:2},marker:{size:6}}],{template:'plotly_white',height:650,title:{text:'Stand Performance — Drilling Time and Average ROP',x:.5},xaxis:{tickangle:-45},yaxis:{title:'Drilling time (hr)',rangemode:'tozero'},yaxis2:{title:'Average ROP (ft/hr)',overlaying:'y',side:'right',rangemode:'tozero'},legend:{orientation:'h',y:1.08,x:.5,xanchor:'center'},margin:{t:90,l:75,r:85,b:120}},{responsive:true})}
function renderBitRuns(){const keys=['run_key','run_name','run_class','start_time','end_time','depth_in_ft','depth_out_ft','footage_ft','duration_hr','bit_size_in','notes'];bitRuns.forEach((r,i)=>{if(!r.run_name)r.run_name=r.run_key;if(r.bit_size_in===undefined)r.bit_size_in='';if(r.notes===undefined)r.notes=''});document.getElementById('bitRunTable').innerHTML='<thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+bitRuns.map((r,i)=>'<tr>'+keys.map(k=>`<td>${['run_name','bit_size_in','notes'].includes(k)?`<input value="${esc(r[k]??'')}" onchange="bitRuns[${i}]['${k}']=this.value;initBitRunSelect();updateMainOverlays()">`:esc(r[k]??'')}</td>`).join('')+'</tr>').join('')+'</tbody>';initBitRunSelect();drawBitRunPerformance()}
let selectedBitRuns=new Set();let bitRunUpdateTimer=null;
function syncBitRunChecks(){document.querySelectorAll('#bitRunChecks input[type="checkbox"]').forEach(x=>x.checked=selectedBitRuns.has(String(x.value)))}
function scheduleBitRunComparison(){clearTimeout(bitRunUpdateTimer);bitRunUpdateTimer=setTimeout(updateBitRunComparison,90)}
function initBitRunSelect(){const root=document.getElementById('bitRunChecks');if(!root)return;const valid=new Set(bitRuns.map(r=>String(r.run_key)));selectedBitRuns=new Set([...selectedBitRuns].filter(k=>valid.has(k)));if(!selectedBitRuns.size)bitRuns.filter(r=>r.selectable!==false).slice(0,Math.min(4,bitRuns.length)).forEach(r=>selectedBitRuns.add(String(r.run_key)));root.innerHTML=bitRuns.map((r,i)=>`<label class="bitrun-check-item"><input type="checkbox" value="${esc(r.run_key)}" ${selectedBitRuns.has(String(r.run_key))?'checked':''} onchange="toggleBitRunSelection('${esc(r.run_key)}',this.checked)"><span><b>${esc(r.run_name||r.run_key)}</b><br>${esc(r.role||r.run_class)} | ${num(r.depth_in_ft,0)}–${num(r.depth_out_ft,0)} ft | ${num(r.footage_ft,0)} ft</span></label>`).join('');scheduleBitRunComparison()}
function toggleBitRunSelection(k,v){if(v)selectedBitRuns.add(String(k));else selectedBitRuns.delete(String(k));scheduleBitRunComparison()}
function selectAllBitRuns(v){selectedBitRuns=new Set(v?bitRuns.filter(r=>r.selectable!==false).map(r=>String(r.run_key)):[]);syncBitRunChecks();scheduleBitRunComparison()}
function selectBitRunClass(cls){selectedBitRuns=new Set(bitRuns.filter(r=>String(r.run_class)===cls).map(r=>String(r.run_key)));syncBitRunChecks();scheduleBitRunComparison()}
function updateBitRunComparison(){const keys=[...selectedBitRuns],root=document.getElementById('bitRunComparePies');root.querySelectorAll('.js-plotly-plot').forEach(x=>Plotly.purge(x));root.innerHTML='';const tableRows=[],allStates=[];keys.forEach((k,i)=>{const run=bitRuns.find(r=>String(r.run_key)===String(k)),rows=(P.bitrun_states||[]).filter(r=>String(r.run_key)===String(k)).map(r=>({state:r.RigState,duration_sec:Number(r.Duration_sec),duration_hr:Number(r.Duration_hr),drilled_ft:Number(r.Drilled_ft||0),percentage:Number(r.Percentage)}));rows.forEach(r=>allStates.push(r.state));const card=document.createElement('div');card.className='panel bitrun-pie-card';card.innerHTML=`<div id="bitPie${i}" class="plot"></div><div class="metric"><b>${esc(run?.run_name||k)}</b><br>${esc(run?.role||run?.run_class||'')}<br>Start: ${esc(run?.start_time||'')}<br>End: ${esc(run?.end_time||'')}<br>MD: ${num(run?.depth_in_ft,1)}–${num(run?.depth_out_ft,1)} ft<br>Drilled: ${num(run?.footage_ft,1)} ft | Total: ${num(run?.duration_hr,2)} hr</div>`;root.appendChild(card);pie('bitPie'+i,rows,(run?.run_name||k),{showLegend:false});rows.forEach(r=>tableRows.push({run:run?.run_name||k,...r}))});sharedStateLegend('bitRunSharedLegend',allStates);document.getElementById('bitRunStateTable').innerHTML='<thead><tr><th>Bit run</th><th>State</th><th>Hours</th><th>Minutes</th><th>Percentage</th><th>Drilled depth (ft)</th></tr></thead><tbody>'+tableRows.map(r=>`<tr><td>${esc(r.run)}</td><td>${esc(r.state)}</td><td>${num(r.duration_hr,2)}</td><td>${num(r.duration_sec/60,1)}</td><td>${num(r.percentage,2)}%</td><td>${num(r.drilled_ft,1)}</td></tr>`).join('')+'</tbody>';drawBitRunPerformance(keys)}
function updateBitRun(){updateBitRunComparison()}
function drawBitRunPerformance(selectedKeys=null){const chosen=selectedKeys&&selectedKeys.length?new Set(selectedKeys):null;const r=bitRuns.filter(x=>x.selectable!==false&&(!chosen||chosen.has(x.run_key))),names=r.map(x=>x.run_name||x.run_key);const tr=[{type:'bar',name:'Avg ROP (ft/hr)',x:names,y:r.map(x=>Number(x.avg_rop_ft_hr||0)),yaxis:'y',marker:{color:'#1f77b4'}}];[['drilling_hr','Drilling hr'],['idle_hr','Idle hr'],['connection_hr','Connection hr'],['trip_hr','Trip hr'],['ream_hr','Ream hr']].forEach(([k,n])=>tr.push({type:'bar',name:n,x:names,y:r.map(x=>Number(x[k]||0)),yaxis:'y2'}));Plotly.react('bitRunPerformance',tr,{template:'plotly_white',height:700,barmode:'group',title:{text:'Bit-Run Performance and State Comparison',x:.5},xaxis:{tickangle:-35},yaxis:{title:'Average ROP (ft/hr)'},yaxis2:{title:'State hours',overlaying:'y',side:'right'},legend:{orientation:'h',y:1.1}},{responsive:true})}
function csv(name,rows){if(!rows.length)return;const keys=Object.keys(rows[0]),txt=[keys.join(',')].concat(rows.map(r=>keys.map(k=>'"'+String(r[k]??'').replace(/"/g,'""')+'"').join(','))).join('\n'),b=new Blob([txt],{type:'text/csv'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=name;a.click();URL.revokeObjectURL(u)}function exportSections(){csv('hole_sections.csv',sections)}function exportLithology(){csv('lithology.csv',lithology)}function exportBitRuns(){csv('automatic_bit_runs.csv',bitRuns)}
function saveEditedHtml(){const state={sections,lithology,bitRuns,selectedBitRuns:[...selectedBitRuns],selectedStands:[...selectedStands]};let tag=document.getElementById('savedState');if(!tag){tag=document.createElement('script');tag.type='application/json';tag.id='savedState';document.body.appendChild(tag)}tag.textContent=JSON.stringify(state).replace(/<\/script/gi,'<\\/script');const b=new Blob(['<!DOCTYPE html>\n'+document.documentElement.outerHTML],{type:'text/html'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=(P.well_name||'well')+'_UNIFIED_FSM_EDITED.html';a.click();URL.revokeObjectURL(u)}
function loadSaved(){const t=document.getElementById('savedState');if(!t||!t.textContent.trim())return;try{const s=JSON.parse(t.textContent);if(s.sections)sections=s.sections;if(s.lithology)lithology=s.lithology;if(s.bitRuns)bitRuns=s.bitRuns;if(Array.isArray(s.selectedBitRuns))selectedBitRuns=new Set(s.selectedBitRuns);if(Array.isArray(s.selectedStands))selectedStands=new Set(s.selectedStands)}catch(e){}}
function init(){try{loadSaved();mainPlot();bitRunDepthPlot();const ov=(P.overall||[]).map(r=>({state:r.RigState,duration_sec:Number(r.Duration_sec),duration_hr:Number(r.Duration_hr),percentage:Number(r.Percentage),drilled_ft:Number(r.Drilled_ft||0)}));pie('overallPie',ov,'Total FSM State Distribution');statTable('overallTable',ov);renderSections();generateSectionCards();renderLithology();document.getElementById('fsmStartTime').value=P.start_time.slice(0,19);document.getElementById('fsmEndTime').value=P.end_time.slice(0,19);document.getElementById('fsmStartDay').value=0;document.getElementById('fsmEndDay').value=((new Date(P.end_time)-new Date(P.start_time))/86400000).toFixed(3);document.getElementById('timeStart').value=P.start_time.slice(0,16);document.getElementById('timeEnd').value=P.end_time.slice(0,16);updateTime();initStands();renderBitRuns();document.getElementById('status').innerHTML='Dashboard loaded. State selection, day/time/depth controls, detailed rig-state tracks, bit-run hover details, multi-run state comparison, and drilled-depth tables are active.'}catch(e){console.error(e);err('Dashboard initialization failed: '+esc(e.message||e))}}
document.addEventListener('DOMContentLoaded',init);
</script></body></html>'''

html=html_template.replace("__PLOTLY_JS__",get_plotlyjs()).replace("__PAYLOAD__",json.dumps(payload,separators=(",",":"),ensure_ascii=False))
unified_html_path=OUTPUT_DIR/"fsm_UNIFIED_linked_operational_bitrun_dashboard.html"
unified_html_path.write_text(html,encoding="utf-8")

fsm_unified_dashboard=unified_html_path
print("\n=== UNIFIED DASHBOARD ===")
print(unified_html_path.resolve())
print("Automatic bit runs detected:",len(fsm_detected_bit_runs))
print("Stand depth bins:",len(fsm_stand_summary))
print("Stand bins requiring review:",len(fsm_stand_gap_report))
print("Unresolved stand-depth discontinuities:",len(fsm_stand_continuity_report))
print("Stand review CSV:",(OUTPUT_DIR/"fsm_stand_gap_report.csv").resolve())
print("Stand continuity CSV:",(OUTPUT_DIR/"fsm_stand_continuity_report.csv").resolve())
print("Open this file, not the earlier separate operational HTML.")
print("Layout version: v22 — expanded state track, matched MSE axes, and protected following content.")

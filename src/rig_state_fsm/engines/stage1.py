
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

# ============================================================
# STAGE 1 ONLY
# Manual column mapping + timestamp construction +
# calculated WOB/ROP/differential pressure/MSE.
# Original and calculated channels are preserved for Stage 2
#
# This file DOES NOT run preprocessing selection or the FSM.
# ============================================================

# -----------------------------
# USER SETTINGS
# -----------------------------
FILE_NAME = os.environ.get("FSM_STAGE1_INPUT", "").strip()

if not FILE_NAME:
    raise FileNotFoundError(
        "No Stage-1 input was supplied. Set FSM_STAGE1_INPUT to the raw CSV/ZIP path."
    )

FALLBACK_FILE_NAMES = [FILE_NAME]

OUTPUT_DIR = Path("fsm_stage1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Calculation assumptions must be supplied for each well. A bit diameter cannot
# be inferred safely from depth alone. If the raw data do not contain a usable
# bit-size channel, provide FSM_BIT_SECTIONS_JSON. An optional single-size
# fallback may be supplied explicitly with FSM_DEFAULT_BIT_DIAMETER_IN.
DEFAULT_BIT_SECTION_SCHEDULE_FT = json.loads(
    os.environ.get("FSM_BIT_SECTIONS_JSON", "[]")
)
_default_bit_text = os.environ.get("FSM_DEFAULT_BIT_DIAMETER_IN", "").strip()
DEFAULT_BIT_DIAMETER_IN = float(_default_bit_text) if _default_bit_text else None

# Used only when BIT_DIAMETER is not mapped and no valid section schedule applies.
# Units expected below:
#   Hook load / WOB: klbf
#   Torque: kft-lbf
#   Standpipe/differential pressure: psi
#   Flow: gpm
#   Depth: ft
#   ROP: ft/hr

MAX_REASONABLE_ROP_FT_HR = 1000.0
MAX_REASONABLE_WOB_KLBF = 200.0
MAX_REASONABLE_DIFF_PRESSURE_PSI = 10000.0

# Optional simple preprocessing for the selected signals.
APPLY_SELECTED_SIGNAL_CLEANING = True
ROLLING_MEDIAN_WINDOW = 5

# Stage 1 does not create comparison HTML. Plotting and user source selection
# belong to Stage 2 preprocessing.
CREATE_STAGE1_COMPARISON_HTML = False

# MSE guards. Very small ROP in the denominator creates nonphysical spikes.
MSE_MIN_ROP_FT_HR = 2.0
MSE_MAX_PSI = 2_000_000.0


# ============================================================
# FILE DISCOVERY
# ============================================================

current_folder = Path.cwd()
parent_folder = current_folder.parent

base_folders = [
    current_folder,
    parent_folder,
    current_folder / "bit_run_detection_outputs",
    parent_folder / "bit_run_detection_outputs",
]

unique_base_folders = []
seen = set()

for folder in base_folders:
    key = str(folder.expanduser())
    if key not in seen:
        unique_base_folders.append(folder)
        seen.add(key)

possible_paths = [
    folder / name
    for name in FALLBACK_FILE_NAMES
    for folder in unique_base_folders
]

print("=== CURRENT WORKING FOLDER ===")
print(current_folder)

print("\n=== CHECKING POSSIBLE FILE LOCATIONS ===")
file_path = None

for path in possible_paths:
    exists = path.exists()
    print(f"{path} -> {exists}")
    if exists and file_path is None:
        file_path = path

if file_path is None:
    raise FileNotFoundError(
        "CSV/ZIP file was not found.\n\nChecked paths:\n"
        + "\n".join(str(path) for path in possible_paths)
    )

print("\nUsing file:")
print(file_path)

df = pd.read_csv(file_path, low_memory=False)
all_cols = list(df.columns)

print("\n=== AVAILABLE COLUMNS ===")
for i, col in enumerate(all_cols):
    print(f"[{i}] {col}")


# ============================================================
# MANUAL COLUMN MAPPING
# ============================================================

def pick_column(target_name):
    """
    Type:
      keyword -> show matching columns
      index   -> select column
      no      -> skip
      all     -> show all columns
    """
    print(f"\n{target_name}")

    while True:
        value = input(
            f"Search or enter index for {target_name} "
            f"(type 'no' if unavailable, 'all' to show all): "
        ).strip()

        if value.lower() == "no":
            print(f"Skipped: {target_name}")
            return None

        if value.lower() == "all":
            for i, col in enumerate(all_cols):
                print(f"[{i}] {col}")
            continue

        if value.isdigit():
            idx = int(value)
            if 0 <= idx < len(all_cols):
                print(f"Selected: [{idx}] {all_cols[idx]}")
                return all_cols[idx]
            print("Invalid index.")
            continue

        matches = [
            (i, col)
            for i, col in enumerate(all_cols)
            if value.lower() in str(col).lower()
        ]

        if matches:
            print(f"\nMatches for '{value}':")
            for i, col in matches:
                print(f"[{i}] {col}")
        else:
            print("No matches. Try again or type 'no'.")


print("\nSelect dataset columns.")

DATE_col              = pick_column("Date column")
TIME_col              = pick_column("Time / timestamp column")
BIT_DEPTH_col         = pick_column("Bit depth")
HOLE_DEPTH_col        = pick_column("Hole depth / total depth")
BLOCK_POSITION_col    = pick_column("Block height / block position")
HOOK_LOAD_col         = pick_column("Hook load")
WOB_col               = pick_column("Recorded WOB")
ROP_col               = pick_column("Recorded ROP")
ROTARY_RPM_col        = pick_column("Rotary RPM")
ROTARY_TORQUE_col     = pick_column("Rotary torque")
PUMP_OUTPUT_col       = pick_column("Pump output / flow rate in")
PUMP_PRESSURE_col     = pick_column("Standpipe pressure / pump pressure")
DIFF_PRESSURE_col     = pick_column("Recorded differential pressure")
BIT_RPM_col           = pick_column("Bit RPM")
BIT_TORQUE_col        = pick_column("Bit torque")
DEPTH_OF_CUT_col      = pick_column("Depth of cut")
BIT_DIAMETER_col      = pick_column("Bit diameter")
MSE_TOTAL_col         = pick_column("Recorded classical / total MSE")
MSE_DOWNHOLE_col      = pick_column("Recorded hydraulic / downhole MSE")

column_mapping = {
    "DATE": DATE_col,
    "TIME": TIME_col,
    "BLOCK_POSITION": BLOCK_POSITION_col,
    "HOOK_LOAD": HOOK_LOAD_col,
    "BIT_DEPTH": BIT_DEPTH_col,
    "HOLE_DEPTH": HOLE_DEPTH_col,
    "PUMP_OUTPUT": PUMP_OUTPUT_col,
    "PUMP_PRESSURE": PUMP_PRESSURE_col,
    "DIFF_PRESSURE_ORIGINAL": DIFF_PRESSURE_col,
    "WOB_ORIGINAL": WOB_col,
    "ROP_ORIGINAL": ROP_col,
    "ROTARY_RPM": ROTARY_RPM_col,
    "BIT_RPM": BIT_RPM_col,
    "ROTARY_TORQUE": ROTARY_TORQUE_col,
    "BIT_TORQUE": BIT_TORQUE_col,
    "DEPTH_OF_CUT": DEPTH_OF_CUT_col,
    "BIT_DIAMETER_IN": BIT_DIAMETER_col,
    "MSE_TEALE_ORIGINAL": MSE_TOTAL_col,
    "MSE_HYDRAULIC_ORIGINAL": MSE_DOWNHOLE_col,
}

column_mapping = {
    new_name: old_name
    for new_name, old_name in column_mapping.items()
    if old_name is not None
}

working_df = df[list(column_mapping.values())].copy()
working_df.columns = list(column_mapping.keys())

column_mapping_df = pd.DataFrame([
    {
        "Canonical column": canonical,
        "Source column": source,
        "Mapping status": "mapped" if source is not None else "missing",
    }
    for canonical, source in {
        "DATE": DATE_col, "TIME": TIME_col,
        "BLOCK_POSITION": BLOCK_POSITION_col, "HOOK_LOAD": HOOK_LOAD_col,
        "BIT_DEPTH": BIT_DEPTH_col, "HOLE_DEPTH": HOLE_DEPTH_col,
        "PUMP_OUTPUT": PUMP_OUTPUT_col, "PUMP_PRESSURE": PUMP_PRESSURE_col,
        "DIFF_PRESSURE_ORIGINAL": DIFF_PRESSURE_col,
        "WOB_ORIGINAL": WOB_col, "ROP_ORIGINAL": ROP_col,
        "ROTARY_RPM": ROTARY_RPM_col, "BIT_RPM": BIT_RPM_col,
        "ROTARY_TORQUE": ROTARY_TORQUE_col, "BIT_TORQUE": BIT_TORQUE_col,
        "DEPTH_OF_CUT": DEPTH_OF_CUT_col,
        "BIT_DIAMETER_IN": BIT_DIAMETER_col,
        "MSE_TEALE_ORIGINAL": MSE_TOTAL_col,
        "MSE_HYDRAULIC_ORIGINAL": MSE_DOWNHOLE_col,
    }.items()
])

print("\n=== COLUMN MAPPING TABLE ===")
display(column_mapping_df)


# ============================================================
# TIMESTAMP HELPERS
# ============================================================

def parse_datetime_robust(series):
    cleaned = series.astype(str).str.strip().replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "NULL": np.nan,
        "null": np.nan,
        "NaT": np.nan,
        "s": np.nan,
        "sec": np.nan,
        "seconds": np.nan,
    })

    cleaned = cleaned.str.replace(
        r"([+-]\d{2}:\d{2})Z$",
        r"\1",
        regex=True,
    )

    try:
        parsed = pd.to_datetime(
            cleaned,
            errors="coerce",
            format="mixed",
            utc=True,
        )
    except TypeError:
        parsed = pd.to_datetime(cleaned, errors="coerce", utc=True)

    try:
        parsed = parsed.dt.tz_convert(None)
    except Exception:
        pass

    return parsed


def parse_elapsed_time_to_timestamp(
    series,
    synthetic_start="1970-01-01 00:00:00",
):
    raw = series.astype(str).str.strip().replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "NULL": np.nan,
        "null": np.nan,
        "s": np.nan,
        "sec": np.nan,
        "seconds": np.nan,
    })

    timedelta_values = pd.to_timedelta(raw, errors="coerce")
    if timedelta_values.notna().mean() > 0.80:
        return pd.Timestamp(synthetic_start) + timedelta_values, "string_timedelta"

    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() < 0.80:
        return pd.Series(pd.NaT, index=series.index), "unparseable"

    sorted_numeric = numeric.dropna().sort_values()
    median_step = sorted_numeric.diff().dropna().median()
    max_value = numeric.max()

    if max_value <= 10 and median_step < 1:
        unit = "D"
        label = "fractional_days"
    elif max_value <= 1000 and median_step <= 1:
        unit = "h"
        label = "hours"
    elif max_value <= 500000 and median_step <= 10:
        unit = "m"
        label = "minutes"
    else:
        unit = "s"
        label = "seconds"

    return (
        pd.Timestamp(synthetic_start) + pd.to_timedelta(numeric, unit=unit),
        label,
    )


def remove_unit_rows(frame):
    out = frame.copy()

    unit_tokens = {
        "s", "sec", "seconds", "min", "minute", "minutes",
        "hr", "hour", "hours", "m", "meter", "meters",
        "ft", "feet", "t", "ton", "tons", "psi", "rpm",
        "gal/min", "gpm", "m/h", "ft/h", "1000 ft.lbf",
    }

    unit_like_count = pd.Series(0, index=out.index)

    for col in out.columns:
        text = out[col].astype(str).str.strip().str.lower()
        unit_like_count += text.isin(unit_tokens).astype(int)

    return out.loc[unit_like_count <= 1].copy()


def build_timestamp(frame):
    out = frame.copy()

    if "DATE" in out.columns and "TIME" in out.columns:
        combined = (
            out["DATE"].astype(str).str.strip()
            + " "
            + out["TIME"].astype(str).str.strip()
        )
        timestamp = parse_datetime_robust(combined)
        if timestamp.notna().mean() > 0.80:
            out["TIMESTAMP"] = timestamp
            return out, "DATE_PLUS_TIME"

    if "TIME" in out.columns:
        timestamp = parse_datetime_robust(out["TIME"])
        if timestamp.notna().mean() > 0.80:
            out["TIMESTAMP"] = timestamp
            return out, "TIME_FULL_DATETIME"

        timestamp, unit_used = parse_elapsed_time_to_timestamp(out["TIME"])
        if timestamp.notna().mean() > 0.80:
            out["TIMESTAMP"] = timestamp
            return out, f"TIME_ELAPSED_{unit_used}"

    if "DATE" in out.columns:
        timestamp = parse_datetime_robust(out["DATE"])
        if timestamp.notna().mean() > 0.80:
            out["TIMESTAMP"] = timestamp
            return out, "DATE_FULL_DATETIME"

    raise ValueError("Could not construct TIMESTAMP from selected DATE/TIME columns.")


working_df = remove_unit_rows(working_df)
working_df, timestamp_source = build_timestamp(working_df)

for col in working_df.columns:
    if col not in {"DATE", "TIME", "TIMESTAMP"}:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

working_df = (
    working_df
    .dropna(subset=["TIMESTAMP"])
    .sort_values("TIMESTAMP")
    .drop_duplicates(subset=["TIMESTAMP"], keep="first")
    .reset_index(drop=True)
)

print("\nTimestamp source:", timestamp_source)
print("Rows after timestamp cleaning:", len(working_df))


# ============================================================
# UNIT CONFIRMATION / CONVERSION
# ============================================================

def ask_choice(prompt, choices, default):
    choice_text = "/".join(choices)
    value = input(f"{prompt} [{choice_text}] (default {default}): ").strip().lower()
    return value if value in choices else default


print("\nConfirm the mapped dataset units.")

depth_unit = ask_choice("Depth unit", ["ft", "m"], "ft")
hookload_unit = ask_choice("Hook load / WOB unit", ["klbf", "lbf", "tonne"], "klbf")
torque_unit = ask_choice("Torque unit", ["kftlbf", "ftlbf", "knm"], "kftlbf")
flow_unit = ask_choice("Flow unit", ["gpm", "lpm"], "gpm")
pressure_unit = ask_choice("Pressure unit", ["psi", "kpa", "bar"], "psi")
rop_unit = ask_choice("Recorded ROP unit", ["fthr", "mhr"], "fthr")
bit_diameter_unit = ask_choice("Mapped bit diameter unit", ["in", "mm"], "in")
mse_unit = ask_choice("Recorded MSE unit", ["psi", "kpa", "bar", "mpa"], pressure_unit if pressure_unit in {"psi", "kpa", "bar"} else "psi")

analysis_df = working_df.copy()

# Convert depth to ft.
for col in ["BIT_DEPTH", "HOLE_DEPTH", "BLOCK_POSITION", "DEPTH_OF_CUT"]:
    if col in analysis_df.columns and depth_unit == "m":
        analysis_df[col] = analysis_df[col] * 3.280839895

# Convert hook load and WOB to klbf.
for col in ["HOOK_LOAD", "WOB_ORIGINAL"]:
    if col not in analysis_df.columns:
        continue
    if hookload_unit == "lbf":
        analysis_df[col] = analysis_df[col] / 1000.0
    elif hookload_unit == "tonne":
        analysis_df[col] = analysis_df[col] * 2.2046226218

# Convert torque to kft-lbf.
for col in ["ROTARY_TORQUE", "BIT_TORQUE"]:
    if col not in analysis_df.columns:
        continue
    if torque_unit == "ftlbf":
        analysis_df[col] = analysis_df[col] / 1000.0
    elif torque_unit == "knm":
        analysis_df[col] = analysis_df[col] * 0.737562149

# Convert flow to gpm.
if "PUMP_OUTPUT" in analysis_df.columns and flow_unit == "lpm":
    analysis_df["PUMP_OUTPUT"] = analysis_df["PUMP_OUTPUT"] * 0.264172052

# Convert pressure to psi.
for col in ["PUMP_PRESSURE", "DIFF_PRESSURE_ORIGINAL"]:
    if col not in analysis_df.columns:
        continue
    if pressure_unit == "kpa":
        analysis_df[col] = analysis_df[col] * 0.145037738
    elif pressure_unit == "bar":
        analysis_df[col] = analysis_df[col] * 14.5037738

# Convert recorded ROP to ft/hr.
if "ROP_ORIGINAL" in analysis_df.columns and rop_unit == "mhr":
    analysis_df["ROP_ORIGINAL"] = analysis_df["ROP_ORIGINAL"] * 3.280839895

# Convert mapped bit diameter to inches.
if "BIT_DIAMETER_IN" in analysis_df.columns and bit_diameter_unit == "mm":
    analysis_df["BIT_DIAMETER_IN"] = analysis_df["BIT_DIAMETER_IN"] / 25.4

# Convert recorded MSE channels to psi for consistent comparison.
for col in ["MSE_TEALE_ORIGINAL", "MSE_HYDRAULIC_ORIGINAL"]:
    if col not in analysis_df.columns:
        continue
    if mse_unit == "kpa":
        analysis_df[col] = analysis_df[col] * 0.145037738
    elif mse_unit == "bar":
        analysis_df[col] = analysis_df[col] * 14.5037738
    elif mse_unit == "mpa":
        analysis_df[col] = analysis_df[col] * 145.037738


# ============================================================
# CALCULATED PARAMETERS
# ============================================================

def infer_sample_seconds(timestamp):
    dt = timestamp.diff().dt.total_seconds()
    positive = dt[(dt > 0) & np.isfinite(dt)]
    if positive.empty:
        return 1.0
    return float(positive.median())


sample_seconds = infer_sample_seconds(analysis_df["TIMESTAMP"])
analysis_df["DT_SECONDS"] = (
    analysis_df["TIMESTAMP"].diff().dt.total_seconds()
)
analysis_df["DT_SECONDS"] = analysis_df["DT_SECONDS"].where(
    (analysis_df["DT_SECONDS"] > 0)
    & (analysis_df["DT_SECONDS"] <= max(60.0, sample_seconds * 20)),
    sample_seconds,
)

# -----------------------------
# Calculated ROP
# -----------------------------
if "HOLE_DEPTH" in analysis_df.columns:
    depth_change_ft = analysis_df["HOLE_DEPTH"].diff()
    rop_calc = depth_change_ft / (analysis_df["DT_SECONDS"] / 3600.0)
    rop_calc = rop_calc.where(
        (rop_calc >= 0)
        & (rop_calc <= MAX_REASONABLE_ROP_FT_HR)
    )
    analysis_df["ROP_CALCULATED"] = (
        rop_calc
        .rolling(ROLLING_MEDIAN_WINDOW, center=True, min_periods=1)
        .median()
    )
else:
    analysis_df["ROP_CALCULATED"] = np.nan

# -----------------------------
# Calculated WOB
# -----------------------------
if {"HOOK_LOAD", "BIT_DEPTH", "HOLE_DEPTH"}.issubset(analysis_df.columns):
    on_bottom_gap = (analysis_df["HOLE_DEPTH"] - analysis_df["BIT_DEPTH"]).abs()

    # Off-bottom reference candidates:
    # bit visibly above bottom, or recorded WOB close to zero when available.
    off_bottom = on_bottom_gap > 5.0

    if "WOB_ORIGINAL" in analysis_df.columns:
        off_bottom = off_bottom | (analysis_df["WOB_ORIGINAL"].fillna(0) <= 1.0)

    reference_hookload = analysis_df["HOOK_LOAD"].where(off_bottom)

    # Time-based rolling quantile would be ideal, but row rolling is more robust
    # for mixed timestamps. This creates a slowly changing off-bottom reference.
    window_rows = max(
        101,
        int(round((30 * 60) / max(sample_seconds, 1.0))),
    )

    reference_hookload = (
        reference_hookload
        .rolling(window_rows, center=True, min_periods=max(10, window_rows // 20))
        .quantile(0.75)
        .interpolate(limit_direction="both")
    )

    wob_calc = reference_hookload - analysis_df["HOOK_LOAD"]
    wob_calc = wob_calc.where(on_bottom_gap <= 5.0, 0.0)
    wob_calc = wob_calc.clip(lower=0.0, upper=MAX_REASONABLE_WOB_KLBF)

    analysis_df["HOOK_LOAD_OFF_BOTTOM_REFERENCE"] = reference_hookload
    analysis_df["WOB_CALCULATED"] = (
        wob_calc
        .rolling(ROLLING_MEDIAN_WINDOW, center=True, min_periods=1)
        .median()
    )
else:
    analysis_df["HOOK_LOAD_OFF_BOTTOM_REFERENCE"] = np.nan
    analysis_df["WOB_CALCULATED"] = np.nan

# -----------------------------
# Calculated differential pressure
# -----------------------------
if {"PUMP_PRESSURE", "PUMP_OUTPUT"}.issubset(analysis_df.columns):
    flow = analysis_df["PUMP_OUTPUT"]
    pressure = analysis_df["PUMP_PRESSURE"]

    positive_flow = flow[flow > 0]
    flow_threshold = (
        max(1.0, float(positive_flow.quantile(0.10)))
        if not positive_flow.empty
        else 1.0
    )

    pumps_on = flow >= flow_threshold

    off_bottom_mask = pumps_on.copy()

    if {"BIT_DEPTH", "HOLE_DEPTH"}.issubset(analysis_df.columns):
        off_bottom_mask = off_bottom_mask & (
            (analysis_df["HOLE_DEPTH"] - analysis_df["BIT_DEPTH"]).abs() > 5.0
        )

    if "WOB_ORIGINAL" in analysis_df.columns:
        off_bottom_mask = off_bottom_mask | (
            pumps_on & (analysis_df["WOB_ORIGINAL"].fillna(0) <= 1.0)
        )

    baseline_pressure = pressure.where(off_bottom_mask)

    window_rows = max(
        101,
        int(round((30 * 60) / max(sample_seconds, 1.0))),
    )

    baseline_pressure = (
        baseline_pressure
        .rolling(window_rows, center=True, min_periods=max(10, window_rows // 20))
        .median()
        .interpolate(limit_direction="both")
    )

    diff_calc = pressure - baseline_pressure
    diff_calc = diff_calc.where(pumps_on, 0.0)
    diff_calc = diff_calc.clip(
        lower=0.0,
        upper=MAX_REASONABLE_DIFF_PRESSURE_PSI,
    )

    analysis_df["PUMP_PRESSURE_OFF_BOTTOM_BASELINE"] = baseline_pressure
    analysis_df["DIFF_PRESSURE_CALCULATED"] = (
        diff_calc
        .rolling(ROLLING_MEDIAN_WINDOW, center=True, min_periods=1)
        .median()
    )
else:
    analysis_df["PUMP_PRESSURE_OFF_BOTTOM_BASELINE"] = np.nan
    analysis_df["DIFF_PRESSURE_CALCULATED"] = np.nan

# -----------------------------
# Bit diameter
# -----------------------------
def assign_bit_diameter_from_sections(hole_depth_ft, section_schedule):
    """Assign an editable hole-section bit diameter from measured depth."""
    diameter = pd.Series(np.nan, index=hole_depth_ft.index, dtype="float64")
    source = pd.Series("unassigned", index=hole_depth_ft.index, dtype="object")

    for row in section_schedule:
        top = float(row["top_depth_ft"])
        base = float(row["base_depth_ft"])
        bit_size = float(row["bit_diameter_in"])
        mask = hole_depth_ft.ge(top) & hole_depth_ft.lt(base)
        diameter.loc[mask] = bit_size
        source.loc[mask] = f"fallback section {top:g}-{base:g} ft"

    return diameter, source


valid_mapped_diameter = (
    "BIT_DIAMETER_IN" in analysis_df.columns
    and pd.to_numeric(analysis_df["BIT_DIAMETER_IN"], errors="coerce").gt(0).mean() >= 0.20
)

if valid_mapped_diameter:
    mapped_diameter = pd.to_numeric(analysis_df["BIT_DIAMETER_IN"], errors="coerce")
    analysis_df["BIT_DIAMETER_SOURCE"] = np.where(
        mapped_diameter.gt(0),
        "mapped data column",
        "missing mapped value",
    )
    analysis_df["BIT_DIAMETER_IN"] = mapped_diameter.where(mapped_diameter.gt(0))
else:
    if "HOLE_DEPTH" in analysis_df.columns:
        fallback_diameter, fallback_source = assign_bit_diameter_from_sections(
            pd.to_numeric(analysis_df["HOLE_DEPTH"], errors="coerce"),
            DEFAULT_BIT_SECTION_SCHEDULE_FT,
        )
        analysis_df["BIT_DIAMETER_IN"] = fallback_diameter
        analysis_df["BIT_DIAMETER_SOURCE"] = fallback_source
    else:
        analysis_df["BIT_DIAMETER_IN"] = np.nan
        analysis_df["BIT_DIAMETER_SOURCE"] = "no hole depth available"

# An explicit single-size fallback may cover rows outside the supplied schedule.
missing_diameter = analysis_df["BIT_DIAMETER_IN"].isna() | analysis_df["BIT_DIAMETER_IN"].le(0)
if missing_diameter.any() and DEFAULT_BIT_DIAMETER_IN is not None:
    analysis_df.loc[missing_diameter, "BIT_DIAMETER_IN"] = DEFAULT_BIT_DIAMETER_IN
    analysis_df.loc[missing_diameter, "BIT_DIAMETER_SOURCE"] = (
        f"explicit default {DEFAULT_BIT_DIAMETER_IN:g} in"
    )

missing_diameter = analysis_df["BIT_DIAMETER_IN"].isna() | analysis_df["BIT_DIAMETER_IN"].le(0)
if missing_diameter.any():
    raise ValueError(
        "Bit diameter is unavailable for some rows. Map a bit-diameter channel, "
        "set FSM_BIT_SECTIONS_JSON to the verified well program, or explicitly "
        "set FSM_DEFAULT_BIT_DIAMETER_IN for a single-size interval."
    )

print("\n=== BIT DIAMETER ASSIGNMENT ===")
bit_section_summary_df = (
    analysis_df[["BIT_DIAMETER_IN", "BIT_DIAMETER_SOURCE"]]
    .value_counts(dropna=False)
    .rename("Rows")
    .reset_index()
)
bit_section_summary_df["Coverage (%)"] = (
    bit_section_summary_df["Rows"] / max(len(analysis_df), 1) * 100.0
).round(3)
print(bit_section_summary_df.to_string(index=False))

# -----------------------------
# Classical Teale MSE
# -----------------------------
def calculate_teale_mse_psi(wob_klbf, torque_kftlbf, rpm, rop_fthr, diameter_in):
    """
    Classical Teale MSE in psi.

    MSE = WOB/A + (120*pi*N*T)/(A*ROP)

    Internally:
      WOB klbf -> lbf
      Torque kft-lbf -> ft-lbf
      Area = pi*D^2/4 in^2
    """
    area_in2 = np.pi * diameter_in.pow(2) / 4.0
    wob_lbf = wob_klbf * 1000.0
    torque_ftlbf = torque_kftlbf * 1000.0

    axial_term = wob_lbf / area_in2
    rotary_term = (
        120.0
        * np.pi
        * rpm
        * torque_ftlbf
        / (area_in2 * rop_fthr)
    )

    result = axial_term + rotary_term
    valid = (
        rop_fthr.ge(MSE_MIN_ROP_FT_HR)
        & diameter_in.gt(0)
        & wob_klbf.ge(0)
        & torque_kftlbf.ge(0)
        & rpm.ge(0)
        & np.isfinite(result)
        & result.ge(0)
        & result.le(MSE_MAX_PSI)
    )
    return result.where(valid)


torque_for_mse = (
    analysis_df["BIT_TORQUE"]
    if "BIT_TORQUE" in analysis_df.columns
    and analysis_df["BIT_TORQUE"].notna().mean() >= 0.20
    else analysis_df.get("ROTARY_TORQUE", pd.Series(np.nan, index=analysis_df.index))
)

rpm_for_mse = (
    analysis_df["BIT_RPM"]
    if "BIT_RPM" in analysis_df.columns
    and analysis_df["BIT_RPM"].notna().mean() >= 0.20
    else analysis_df.get("ROTARY_RPM", pd.Series(np.nan, index=analysis_df.index))
)

# Stage 1 produces an independent, fully derived MSE candidate. It never
# blends original and calculated WOB/ROP/DP row by row. Stage 2 owns final
# preprocessing, comparison, source selection, and selected-input MSE.
analysis_df["MSE_TEALE_CALCULATED"] = calculate_teale_mse_psi(
    analysis_df["WOB_CALCULATED"],
    torque_for_mse,
    rpm_for_mse,
    analysis_df["ROP_CALCULATED"],
    analysis_df["BIT_DIAMETER_IN"],
)

# -----------------------------
# Hydraulic MSE
# -----------------------------
def calculate_hydraulic_mse_psi(
    teale_mse_psi,
    differential_pressure_psi,
    flow_gpm,
    rop_fthr,
    diameter_in,
):
    """
    Hydraulic contribution based on hydraulic power per excavated rock volume.

    Hydraulic term:
      1155 * DeltaP(psi) * Q(gpm) / [A(in^2) * ROP(ft/hr)]

    Result is added to classical Teale MSE.
    """
    area_in2 = np.pi * diameter_in.pow(2) / 4.0

    hydraulic_term = (
        1155.0
        * differential_pressure_psi
        * flow_gpm
        / (area_in2 * rop_fthr)
    )

    result = teale_mse_psi + hydraulic_term

    valid = (
        rop_fthr.ge(MSE_MIN_ROP_FT_HR)
        & diameter_in.gt(0)
        & differential_pressure_psi.ge(0)
        & flow_gpm.ge(0)
        & np.isfinite(result)
        & result.ge(0)
        & result.le(MSE_MAX_PSI)
    )
    return result.where(valid)


if "PUMP_OUTPUT" in analysis_df.columns:
    analysis_df["MSE_HYDRAULIC_CALCULATED"] = calculate_hydraulic_mse_psi(
        analysis_df["MSE_TEALE_CALCULATED"],
        analysis_df["DIFF_PRESSURE_CALCULATED"],
        analysis_df["PUMP_OUTPUT"],
        analysis_df["ROP_CALCULATED"],
        analysis_df["BIT_DIAMETER_IN"],
    )
else:
    analysis_df["MSE_HYDRAULIC_CALCULATED"] = np.nan

analysis_df["MSE_VALID_INPUT_ROW"] = (
    analysis_df["ROP_CALCULATED"].ge(MSE_MIN_ROP_FT_HR)
    & analysis_df["WOB_CALCULATED"].ge(0)
    & analysis_df["BIT_DIAMETER_IN"].gt(0)
    & rpm_for_mse.ge(0)
    & torque_for_mse.ge(0)
)

print("\n=== MSE CALCULATION CHECK ===")
print("Minimum ROP used in MSE:", MSE_MIN_ROP_FT_HR, "ft/hr")
print("Maximum retained MSE:", MSE_MAX_PSI, "psi")
print("Valid MSE input rows:", int(analysis_df["MSE_VALID_INPUT_ROW"].sum()))
print("Calculated Teale MSE coverage:", f"{analysis_df['MSE_TEALE_CALCULATED'].notna().mean():.2%}")
print("Calculated hydraulic MSE coverage:", f"{analysis_df['MSE_HYDRAULIC_CALCULATED'].notna().mean():.2%}")
print("MSE candidate inputs: calculated WOB, ROP, and differential pressure; "
      "available torque/RPM; assigned bit diameter")


# ============================================================
# ORIGINAL VS CALCULATED QC
# ============================================================

def comparison_metrics(original, calculated, name, active_mask=None):
    original = pd.to_numeric(original, errors="coerce")
    calculated = pd.to_numeric(calculated, errors="coerce")

    pair_all = pd.DataFrame({
        "original": original,
        "calculated": calculated,
    }).dropna()

    if active_mask is None:
        active_mask = pd.Series(True, index=original.index)
    active_mask = active_mask.reindex(original.index).fillna(False).astype(bool)

    pair_active = pd.DataFrame({
        "original": original.where(active_mask),
        "calculated": calculated.where(active_mask),
    }).dropna()

    metrics = {
        "Parameter": name,
        "Original coverage": float(original.notna().mean()),
        "Calculated coverage": float(calculated.notna().mean()),
        "Paired rows all": int(len(pair_all)),
        "Paired rows active": int(len(pair_active)),
        "Correlation active": np.nan,
        "MAE active": np.nan,
        "RMSE active": np.nan,
        "Median original active": np.nan,
        "Median calculated active": np.nan,
        "Calculation status": "no paired active rows",
    }

    if len(pair_active) >= 3:
        metrics["Correlation active"] = float(pair_active.corr().iloc[0, 1])
        error = pair_active["calculated"] - pair_active["original"]
        metrics["MAE active"] = float(error.abs().mean())
        metrics["RMSE active"] = float(np.sqrt(np.mean(error.pow(2))))
        metrics["Median original active"] = float(pair_active["original"].median())
        metrics["Median calculated active"] = float(pair_active["calculated"].median())
        metrics["Calculation status"] = "comparison available"
    elif calculated.notna().any():
        metrics["Calculation status"] = "calculated; insufficient paired rows"

    return metrics


qc_rows = []

if "WOB_ORIGINAL" in analysis_df.columns:
    qc_rows.append(comparison_metrics(
        analysis_df["WOB_ORIGINAL"],
        analysis_df["WOB_CALCULATED"],
        "WOB",
        active_mask=(analysis_df["WOB_ORIGINAL"].fillna(0).gt(1.0) | analysis_df["WOB_CALCULATED"].fillna(0).gt(1.0)),
    ))

if "ROP_ORIGINAL" in analysis_df.columns:
    qc_rows.append(comparison_metrics(
        analysis_df["ROP_ORIGINAL"],
        analysis_df["ROP_CALCULATED"],
        "ROP",
        active_mask=(analysis_df["ROP_ORIGINAL"].fillna(0).gt(1.0) | analysis_df["ROP_CALCULATED"].fillna(0).gt(1.0)),
    ))

if "DIFF_PRESSURE_ORIGINAL" in analysis_df.columns:
    qc_rows.append(comparison_metrics(
        analysis_df["DIFF_PRESSURE_ORIGINAL"],
        analysis_df["DIFF_PRESSURE_CALCULATED"],
        "Differential pressure",
        active_mask=(analysis_df["PUMP_OUTPUT"].fillna(0).gt(0) if "PUMP_OUTPUT" in analysis_df.columns else None),
    ))

if "MSE_TEALE_ORIGINAL" in analysis_df.columns:
    qc_rows.append(comparison_metrics(
        analysis_df["MSE_TEALE_ORIGINAL"],
        analysis_df["MSE_TEALE_CALCULATED"],
        "Classical Teale MSE",
        active_mask=analysis_df["MSE_VALID_INPUT_ROW"],
    ))

if "MSE_HYDRAULIC_ORIGINAL" in analysis_df.columns:
    qc_rows.append(comparison_metrics(
        analysis_df["MSE_HYDRAULIC_ORIGINAL"],
        analysis_df["MSE_HYDRAULIC_CALCULATED"],
        "Hydraulic MSE",
        active_mask=analysis_df["MSE_VALID_INPUT_ROW"],
    ))

qc_df = pd.DataFrame(qc_rows)

print("\n=== ORIGINAL VS CALCULATED QC ===")
print(qc_df.to_string(index=False))


# ============================================================
# STAGE-1 OUTPUT — NO SOURCE SELECTION HERE
# ============================================================

# Keep both original and calculated channels. Stage 2 will preprocess,
# compare, and let the user select the source passed to the FSM.
stage1_df = analysis_df.copy()

source_selection_df = pd.DataFrame([
    {"Parameter": "WOB", "Original": "WOB_ORIGINAL", "Calculated": "WOB_CALCULATED"},
    {"Parameter": "ROP", "Original": "ROP_ORIGINAL", "Calculated": "ROP_CALCULATED"},
    {"Parameter": "Differential pressure", "Original": "DIFF_PRESSURE_ORIGINAL", "Calculated": "DIFF_PRESSURE_CALCULATED"},
    {"Parameter": "Classical Teale MSE", "Original": "MSE_TEALE_ORIGINAL", "Calculated": "MSE_TEALE_CALCULATED"},
    {"Parameter": "Hydraulic MSE", "Original": "MSE_HYDRAULIC_ORIGINAL", "Calculated": "MSE_HYDRAULIC_CALCULATED"},
])

# Auditable manifests: coverage is always based on the final Stage-1 row count.
canonical_units = {
    "DATE": "source", "TIME": "source", "TIMESTAMP": "datetime",
    "BLOCK_POSITION": "ft", "BIT_DEPTH": "ft", "HOLE_DEPTH": "ft",
    "DEPTH_OF_CUT": "ft", "HOOK_LOAD": "klbf", "WOB_ORIGINAL": "klbf",
    "WOB_CALCULATED": "klbf", "ROP_ORIGINAL": "ft/hr",
    "ROP_CALCULATED": "ft/hr", "ROTARY_RPM": "rpm", "BIT_RPM": "rpm",
    "ROTARY_TORQUE": "kft-lbf", "BIT_TORQUE": "kft-lbf",
    "PUMP_OUTPUT": "gpm", "PUMP_PRESSURE": "psi",
    "DIFF_PRESSURE_ORIGINAL": "psi", "DIFF_PRESSURE_CALCULATED": "psi",
    "BIT_DIAMETER_IN": "in", "MSE_TEALE_ORIGINAL": "psi",
    "MSE_TEALE_CALCULATED": "psi", "MSE_HYDRAULIC_ORIGINAL": "psi",
    "MSE_HYDRAULIC_CALCULATED": "psi",
}

mapped_source_lookup = dict(zip(
    column_mapping_df["Canonical column"],
    column_mapping_df["Source column"],
))
expected_columns = list(dict.fromkeys(
    column_mapping_df["Canonical column"].tolist()
    + ["TIMESTAMP", "WOB_CALCULATED", "ROP_CALCULATED",
       "DIFF_PRESSURE_CALCULATED", "BIT_DIAMETER_IN",
       "MSE_TEALE_CALCULATED", "MSE_HYDRAULIC_CALCULATED"]
))
column_summary_rows = []
for column in expected_columns:
    exists = column in stage1_df.columns
    non_null_rows = int(stage1_df[column].notna().sum()) if exists else 0
    source_column = mapped_source_lookup.get(column)
    if source_column is not None:
        category = "mapped"
    elif column.endswith("_CALCULATED") or column == "TIMESTAMP":
        category = "calculated"
    else:
        category = "missing"
    column_summary_rows.append({
        "Canonical column": column,
        "Category": category,
        "Source column": source_column,
        "Present": exists,
        "Non-null rows": non_null_rows,
        "Missing rows": int(len(stage1_df) - non_null_rows),
        "Coverage (%)": round(non_null_rows / max(len(stage1_df), 1) * 100.0, 3),
        "Unit selected": canonical_units.get(column, "unitless/metadata"),
    })
column_summary_df = pd.DataFrame(column_summary_rows)
mapped_column_summary_df = column_summary_df.loc[
    column_summary_df["Category"].isin(["mapped", "missing"])
].reset_index(drop=True)

derived_definitions = {
    "TIMESTAMP": "constructed from mapped DATE/TIME",
    "ROP_CALCULATED": "hole-depth change divided by elapsed time",
    "WOB_CALCULATED": "off-bottom hook-load reference minus hook load",
    "DIFF_PRESSURE_CALCULATED": "standpipe pressure minus off-bottom baseline",
    "BIT_DIAMETER_IN": "mapped diameter or editable depth-section schedule",
    "MSE_TEALE_CALCULATED": "Teale MSE from calculated WOB/ROP",
    "MSE_HYDRAULIC_CALCULATED": "Teale MSE plus hydraulic contribution",
}
calculation_rows = []
for column, method in derived_definitions.items():
    computed = int(stage1_df[column].notna().sum()) if column in stage1_df else 0
    calculation_rows.append({
        "Calculated column": column,
        "Method": method,
        "Rows computed": computed,
        "Rows missing": int(len(stage1_df) - computed),
        "Coverage (%)": round(computed / max(len(stage1_df), 1) * 100.0, 3),
        "Unit": canonical_units.get(column, "unitless/metadata"),
        "Status": "complete" if computed == len(stage1_df) else (
            "partial" if computed > 0 else "unavailable"
        ),
    })
calculated_column_summary_df = pd.DataFrame(calculation_rows)
calculation_report_df = calculated_column_summary_df.copy()

stage1_csv = OUTPUT_DIR / "stage1_original_and_calculated_parameters.csv"
qc_csv = OUTPUT_DIR / "stage1_original_vs_calculated_qc.csv"
channel_manifest_csv = OUTPUT_DIR / "stage1_channel_manifest.csv"
column_summary_csv = OUTPUT_DIR / "stage1_column_coverage_summary.csv"
calculation_report_csv = OUTPUT_DIR / "stage1_calculation_report.csv"
bit_section_summary_csv = OUTPUT_DIR / "stage1_bit_diameter_section_summary.csv"

stage1_df.to_csv(stage1_csv, index=False)
qc_df.to_csv(qc_csv, index=False)
source_selection_df.to_csv(channel_manifest_csv, index=False)
column_summary_df.to_csv(column_summary_csv, index=False)
calculation_report_df.to_csv(calculation_report_csv, index=False)
bit_section_summary_df.to_csv(bit_section_summary_csv, index=False)

# ============================================================
# COMPARISON HTML MOVED TO STAGE 2
# ============================================================
# Stage 1 intentionally stops after mapping, calculation, and QC export.
# Linked original-vs-calculated plotting, preprocessing, and user source
# selection will be implemented in Stage 2.

html_path = None

# Compact output summary for notebooks and scripts.
summary_df = pd.DataFrame([{
    "Rows": len(stage1_df),
    "Columns": len(stage1_df.columns),
    "Timestamp source": timestamp_source,
    "Median sample interval (s)": sample_seconds,
    "WOB calculated coverage (%)": round(stage1_df["WOB_CALCULATED"].notna().mean() * 100, 3),
    "ROP calculated coverage (%)": round(stage1_df["ROP_CALCULATED"].notna().mean() * 100, 3),
    "Differential pressure calculated coverage (%)": round(stage1_df["DIFF_PRESSURE_CALCULATED"].notna().mean() * 100, 3),
    "Teale MSE calculated coverage (%)": round(stage1_df["MSE_TEALE_CALCULATED"].notna().mean() * 100, 3),
    "Hydraulic MSE calculated coverage (%)": round(stage1_df["MSE_HYDRAULIC_CALCULATED"].notna().mean() * 100, 3),
}])

summary_csv = OUTPUT_DIR / "stage1_summary.csv"
mapping_csv = OUTPUT_DIR / "stage1_column_mapping.csv"
summary_df.to_csv(summary_csv, index=False)
column_mapping_df.to_csv(mapping_csv, index=False)

# Notebook-visible DataFrame outputs. Keep the complete dataframe preview so
# users can verify every retained Stage-1 channel, not only a curated subset.
stage1_preview_df = stage1_df.head(20).copy()

print("=" * 80)
print("STAGE 1 OUTPUT DATAFRAME")
print("=" * 80)
display(stage1_df.head(20))

print("=" * 80)
print("COLUMN COVERAGE SUMMARY")
print("=" * 80)
display(column_summary_df)

print("=" * 80)
print("ORIGINAL VS CALCULATED QC")
print("=" * 80)
display(qc_df)

print("=" * 80)
print("CHANNEL MANIFEST")
print("=" * 80)
display(source_selection_df)

print("=" * 80)
print("MAPPED-COLUMN SUMMARY")
print("=" * 80)
display(mapped_column_summary_df)

print("=" * 80)
print("CALCULATED-COLUMN SUMMARY")
print("=" * 80)
display(calculated_column_summary_df)

print("=" * 80)
print("BIT-DIAMETER SECTION SUMMARY")
print("=" * 80)
display(bit_section_summary_df)

print("\n=== STAGE 1 COMPLETE ===")
print("Mapped and calculated dataframe:", stage1_csv)
print("Summary table:", summary_csv)
print("Column mapping table:", mapping_csv)
print("QC table:", qc_csv)
print("Channel manifest:", channel_manifest_csv)
print("Column coverage summary:", column_summary_csv)
print("Calculation report:", calculation_report_csv)
print("Bit-diameter section summary:", bit_section_summary_csv)
print("Comparison HTML: deferred to Stage 2 preprocessing")

print("\nNotebook variables available:")
print("  stage1_df          -> complete Stage 1 DataFrame")
print("  stage1_preview_df  -> first 20 rows of important channels")
print("  summary_df         -> Stage 1 summary table")
print("  qc_df              -> original-vs-calculated QC table")
print("  column_mapping_df  -> canonical/source mapping table")
print("  column_summary_df  -> mapped/calculated/missing coverage and units")
print("  calculated_column_summary_df -> derived-channel calculation report")
print("  bit_section_summary_df -> per-section bit-diameter assignment")
print("  source_selection_df-> Stage 2 channel manifest")

print("\nNo source selection has been made in Stage 1.")
print("Stage 2 must compare/preprocess original and calculated channels and create the final standardized FSM inputs.")
print("The FSM has NOT been run.")

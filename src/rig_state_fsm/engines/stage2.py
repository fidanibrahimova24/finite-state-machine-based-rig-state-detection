from __future__ import annotations

"""Generic Stage-2 preprocessing for drilling EDR/FSM workflows.

Dependencies: numpy, pandas, plotly (Plotly is needed only for HTML output).
Notebook use: execute after Stage 1 has created stage1_df/analysis_df/working_df.
CLI use: python FSM_Stage2_Preprocessing_GitHub.py --input stage1.csv
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
import sys
import argparse
import json
import re

import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # preprocessing remains usable without the optional renderer
    go = None
    make_subplots = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print


# ============================================================
# STAGE 2 — PREPROCESSING + DERIVED-SOURCE VALIDATION/SELECTION
# ============================================================
# Input: working_df created by Stage 1.
#
# Important design rules:
#   1. Direct rig sensors are never recalculated here.
#   2. Original/calculated selection is limited to derived parameters:
#        WOB, ROP, differential pressure, and MSE when EDR MSE exists.
#   3. Comparison always occurs BEFORE source selection.
#   4. MSE is recalculated again after WOB/ROP/DP selection using the exact
#      selected inputs, measured torque/RPM/flow, and the mapped/assigned bit size.
#   5. If EDR MSE exists, it is compared with both Stage-1 MSE and the newly
#      selected-input MSE. If EDR MSE does not exist, calculated MSE is used.
#   6. Native units remain visible in plots. Separate FSM columns are converted
#      to a fixed internal unit basis so the FSM thresholds remain portable.
# ============================================================


@dataclass
class PreprocessConfig:
    output_dir: Path = Path("fsm_stage2_outputs")

    # Sampling
    resample_rule: Optional[str] = "auto"  # auto -> observed median interval
    preserve_original_timestamp: bool = True

    # Missing data
    null_sentinels: Tuple[float, ...] = (
        -9999.0, -999.25, -999.0, -8888.0, -7777.0, -1e20, 1e20
    )
    short_gap_limit: int = 10
    interpolation_method: str = "linear"

    # Outliers/noise
    iqr_multiplier: float = 3.0
    outlier_action: str = "flag"  # flag or replace
    outlier_median_window: int = 5
    smoothing_window: int = 5
    smoothing_method: str = "median"  # median or mean

    # Local spike correction. Hampel filtering is applied only to isolated
    # excursions, so sustained and physically valid operating regimes remain.
    spike_action: str = "replace"  # flag or replace
    hampel_window: int = 9
    hampel_sigma: float = 6.0
    maximum_spike_run: int = 3
    final_spike_iterations: int = 3
    use_smoothed_fsm_inputs: bool = True
    strict_quality_gate: bool = False
    minimum_fsm_valid_pct: float = 95.0
    correct_impossible_rop: bool = True
    rop_active_threshold_ft_hr: float = 0.5
    rpm_off_threshold: float = 1.0
    flow_off_threshold_gpm: float = 1.0
    hole_depth_small_decrease_ft: float = 2.0

    # Source selection for WOB/ROP/DP and final MSE
    selection_mode: str = "ask"  # ask, automatic, original, calculated
    minimum_pair_rows: int = 100
    minimum_coverage: float = 0.20
    strong_correlation: float = 0.85
    moderate_correlation: float = 0.60

    # MSE limits in canonical US calculation units
    mse_min_rop_ft_hr: float = 2.0
    mse_max_psi: float = 2_000_000.0

    # Outputs
    make_html: bool = True
    max_plot_points: int = 120_000

    # Override only when automatic unit detection is incorrect.
    unit_overrides: Dict[str, str] = field(default_factory=lambda: {
        "depth": "auto",       # ft, m
        "rop": "auto",         # ft/hr, m/hr
        "pressure": "auto",    # psi, bar, MPa, kPa
        "flow": "auto",        # gpm, L/min, m3/h
        "load": "auto",        # klbf, kN, tonne, lbf
        "torque": "auto",      # kft·lbf, ft·lbf, kN·m, N·m
        "mse": "auto",         # psi, bar, MPa, kPa
    })


BASE_DERIVED = {
    "WOB": {
        "original_candidates": ["WOB_ORIGINAL", "WOB"],
        "calculated_candidates": ["WOB_CALCULATED"],
        "selected": "WOB_SELECTED",
        "unit_family": "load",
    },
    "ROP": {
        "original_candidates": ["ROP_ORIGINAL", "ROP"],
        "calculated_candidates": ["ROP_CALCULATED"],
        "selected": "ROP_SELECTED",
        "unit_family": "rop",
    },
    "DIFF_PRESSURE": {
        "original_candidates": ["DIFF_PRESSURE_ORIGINAL", "DIFF_PRESSURE"],
        "calculated_candidates": ["DIFF_PRESSURE_CALCULATED"],
        "selected": "DIFF_PRESSURE_SELECTED",
        "unit_family": "pressure",
    },
}

MSE_PARAMETERS = {
    "MSE_TEALE": {
        "original_candidates": ["MSE_TEALE_ORIGINAL", "MSE_TEALE", "MSE_TOTAL"],
        "stage1_candidates": ["MSE_TEALE_CALCULATED"],
        "selected_recalculated": "MSE_TEALE_SELECTED_RECALCULATED",
        "final": "MSE_TEALE_SELECTED",
    },
    "MSE_HYDRAULIC": {
        "original_candidates": ["MSE_HYDRAULIC_ORIGINAL", "MSE_HYDRAULIC", "MSE_DOWNHOLE"],
        "stage1_candidates": ["MSE_HYDRAULIC_CALCULATED"],
        "selected_recalculated": "MSE_HYDRAULIC_SELECTED_RECALCULATED",
        "final": "MSE_HYDRAULIC_SELECTED",
    },
}

DIRECT_SENSOR_COLUMNS = [
    "BLOCK_POSITION", "HOOK_LOAD", "BIT_DEPTH", "HOLE_DEPTH",
    "PUMP_OUTPUT", "PUMP_PRESSURE", "ROTARY_RPM", "BIT_RPM",
    "ROTARY_TORQUE", "BIT_TORQUE", "BIT_DIAMETER_IN", "DEPTH_OF_CUT",
]
TIME_COLUMNS = {"DATE", "TIME", "TIMESTAMP", "TIMESTAMP_ORIGINAL"}

# Canonical Stage-2 names and common EDR/Stage-1 aliases.  Matching is
# case-insensitive and punctuation-insensitive. Existing canonical columns are
# never overwritten.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "TIMESTAMP": ("UTC_TIME", "UTCTIME", "DATETIME", "DATE_TIME", "TIME_STAMP"),
    "BIT_DEPTH": ("DBTM", "BITDEPTH", "BIT_DEPTH_MD", "BD"),
    "HOLE_DEPTH": ("DMEA", "HOLEDEPTH", "HOLE_DEPTH_MD", "HD"),
    "HOOK_LOAD": ("HKLD", "HOOKLOAD", "HOOK_LOAD_AVG"),
    "ROTARY_RPM": ("RPMA", "RPM", "SURFACE_RPM", "ROTARY_SPEED"),
    "BIT_RPM": ("BITRPM", "DOWNHOLE_RPM"),
    "PUMP_PRESSURE": ("SPPA", "SPP", "STANDPIPE_PRESSURE"),
    "PUMP_OUTPUT": ("MFIA", "FLOW_IN", "FLOW_RATE", "MUD_FLOW_IN"),
    "ROTARY_TORQUE": ("TQA", "TORQUE", "SURFACE_TORQUE"),
    "BIT_TORQUE": ("TOB", "TORQUE_ON_BIT"),
    "BLOCK_POSITION": ("BPOS", "BLOCK_HEIGHT", "BLOCK_POS"),
    "WOB_ORIGINAL": ("WOBA", "WOB", "WEIGHT_ON_BIT"),
    "ROP_ORIGINAL": ("ROPA", "ROP", "RATE_OF_PENETRATION"),
    "DIFF_PRESSURE_ORIGINAL": ("DIFP", "DIFF_PRESSURE", "DIFFERENTIAL_PRESSURE"),
    "BIT_DIAMETER_IN": ("BIT_SIZE", "BIT_DIAMETER", "HOLE_SIZE_IN"),
}


# ============================================================
# Utilities
# ============================================================

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _normalized_column_name(value: object) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


UNIT_SUFFIXES = {
    "M", "FT", "IN", "MM", "PSI", "KPA", "MPA", "BAR",
    "GPM", "LPM", "LMIN", "M3H", "M3HR", "RPM",
    "KN", "KLBF", "LBF", "TONNE", "TONNES",
    "KNM", "NM", "KFTLBF", "FTLBF",
    "MHR", "MH", "FTHR", "FTH",
}


def _matches_alias_with_optional_unit(column: object, alias: object) -> bool:
    """Match an alias exactly or with one recognized engineering-unit suffix."""
    column_norm = _normalized_column_name(column)
    alias_norm = _normalized_column_name(alias)
    if column_norm == alias_norm:
        return True
    if not column_norm.startswith(alias_norm):
        return False
    return column_norm[len(alias_norm):] in UNIT_SUFFIXES


def canonicalize_core_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add canonical aliases without renaming or overwriting source columns."""
    out = df.copy()
    lookup: Dict[str, str] = {}
    for col in out.columns:
        lookup.setdefault(_normalized_column_name(col), col)
    rows = []
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            rows.append({"canonical_column": canonical, "source_column": canonical, "action": "already canonical"})
            continue
        source = next((
            source_col
            for alias in (canonical, *aliases)
            for source_col in out.columns
            if _matches_alias_with_optional_unit(source_col, alias)
        ), None)
        if source is not None:
            out[canonical] = out[source]
            rows.append({"canonical_column": canonical, "source_column": source, "action": "alias copy"})
        else:
            rows.append({"canonical_column": canonical, "source_column": None, "action": "not found"})
    return out, pd.DataFrame(rows)


def is_continuous_numeric(series: pd.Series) -> bool:
    """True for real numeric sensor channels, never Boolean/flag channels."""
    return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)


def safe_numeric(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    # Explicit float conversion prevents nullable/NumPy Boolean arrays from
    # reaching subtraction, quantile, interpolation, or rolling operations.
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def consecutive_gap_lengths(mask: pd.Series) -> List[int]:
    groups = (mask != mask.shift()).cumsum()
    return mask.groupby(groups).sum()[mask.groupby(groups).first()].astype(int).tolist()




# ============================================================
# INPUT RESOLUTION + FALLBACK DERIVED CALCULATIONS
# ============================================================

def resolve_stage2_input(namespace: Optional[Dict[str, object]] = None) -> Tuple[pd.DataFrame, str]:
    """
    Resolve the richest dataframe produced by Stage 1.

    Priority is critical: Stage 1 stores calculated channels in ``stage1_df``
    (and internally in ``analysis_df``), while ``working_df`` contains only the
    mapped raw/original channels. Earlier Stage 2 versions used ``working_df``
    unconditionally, which made every calculated column appear unavailable.
    """
    ns = namespace if namespace is not None else globals()
    for name in ("stage1_df", "analysis_df", "working_df", "processed_df"):
        obj = ns.get(name)
        if isinstance(obj, pd.DataFrame) and len(obj):
            return obj.copy(), name
    raise NameError(
        "No Stage 1 dataframe was found. Expected stage1_df, analysis_df, or working_df."
    )


def _infer_dt_seconds(timestamp: pd.Series) -> pd.Series:
    dt = pd.to_datetime(timestamp, errors="coerce").diff().dt.total_seconds()
    positive = dt[(dt > 0) & np.isfinite(dt)]
    fallback = float(positive.median()) if not positive.empty else 1.0
    return dt.where((dt > 0) & (dt <= max(60.0, fallback * 20.0)), fallback)


def ensure_stage1_derived_columns(
    df: pd.DataFrame, units: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preserve Stage 1 calculated channels when present. If Stage 2 receives only
    ``working_df``, reconstruct the missing derived channels so the workflow is
    still self-contained and GitHub-friendly. No measured direct sensor is
    recalculated or overwritten.
    """
    out = df.copy()
    units = units or {
        "depth": "ft", "rop": "ft/hr", "pressure": "psi", "flow": "gpm",
        "load": "klbf", "torque": "kft·lbf", "mse": "psi",
    }
    factors = conversion_factors_to_us(units)
    depth_gap_native = 5.0 / factors["depth_to_ft"]
    rows = []
    if "TIMESTAMP" not in out.columns:
        out, _ = build_timestamp_safely(out)
    out = out.sort_values("TIMESTAMP").reset_index(drop=True)
    dt = _infer_dt_seconds(out["TIMESTAMP"])

    # ROP from hole-depth change.
    if "ROP_CALCULATED" not in out.columns or not pd.to_numeric(out["ROP_CALCULATED"], errors="coerce").notna().any():
        if "HOLE_DEPTH" in out.columns:
            depth = pd.to_numeric(out["HOLE_DEPTH"], errors="coerce")
            rop = depth.diff() / (dt / 3600.0)
            rop = rop.where((rop >= 0) & (rop <= 2000.0))
            out["ROP_CALCULATED"] = rop.rolling(5, center=True, min_periods=1).median()
            rows.append({"parameter":"ROP","action":"reconstructed in Stage 2","coverage_pct":out["ROP_CALCULATED"].notna().mean()*100})

    # WOB from off-bottom hook-load reference.
    if "WOB_CALCULATED" not in out.columns or not pd.to_numeric(out["WOB_CALCULATED"], errors="coerce").notna().any():
        if {"HOOK_LOAD","BIT_DEPTH","HOLE_DEPTH"}.issubset(out.columns):
            hook = pd.to_numeric(out["HOOK_LOAD"], errors="coerce")
            bit = pd.to_numeric(out["BIT_DEPTH"], errors="coerce")
            hole = pd.to_numeric(out["HOLE_DEPTH"], errors="coerce")
            gap = (hole-bit).abs()
            off = gap > depth_gap_native
            if "WOB_ORIGINAL" in out.columns:
                off |= pd.to_numeric(out["WOB_ORIGINAL"], errors="coerce").fillna(0).le(1.0)
            sample = float(dt[dt>0].median()) if (dt>0).any() else 1.0
            window = max(101, int(round(1800/max(sample,1.0))))
            ref = hook.where(off).rolling(window, center=True, min_periods=max(10,window//20)).quantile(.75).interpolate(limit_direction="both")
            wob = (ref-hook).where(gap<=depth_gap_native,0.0)
            out["HOOK_LOAD_OFF_BOTTOM_REFERENCE"] = ref
            out["WOB_CALCULATED"] = wob.rolling(5, center=True, min_periods=1).median()
            rows.append({"parameter":"WOB","action":"reconstructed in Stage 2","coverage_pct":out["WOB_CALCULATED"].notna().mean()*100})

    # Differential pressure from pump-on off-bottom baseline.
    if "DIFF_PRESSURE_CALCULATED" not in out.columns or not pd.to_numeric(out["DIFF_PRESSURE_CALCULATED"], errors="coerce").notna().any():
        if {"PUMP_PRESSURE","PUMP_OUTPUT"}.issubset(out.columns):
            spp = pd.to_numeric(out["PUMP_PRESSURE"], errors="coerce")
            flow = pd.to_numeric(out["PUMP_OUTPUT"], errors="coerce")
            pf = flow[flow>0]
            flow_on = flow >= (max(1.0,float(pf.quantile(.10))) if not pf.empty else 1.0)
            off = flow_on.copy()
            if {"BIT_DEPTH","HOLE_DEPTH"}.issubset(out.columns):
                off &= (pd.to_numeric(out["HOLE_DEPTH"],errors="coerce")-pd.to_numeric(out["BIT_DEPTH"],errors="coerce")).abs()>depth_gap_native
            if "WOB_ORIGINAL" in out.columns:
                off |= flow_on & pd.to_numeric(out["WOB_ORIGINAL"],errors="coerce").fillna(0).le(1.0)
            sample = float(dt[dt>0].median()) if (dt>0).any() else 1.0
            window = max(101, int(round(1800/max(sample,1.0))))
            baseline = spp.where(off).rolling(window,center=True,min_periods=max(10,window//20)).median().interpolate(limit_direction="both")
            dp = (spp-baseline).where(flow_on,0.0).clip(lower=0)
            out["PUMP_PRESSURE_OFF_BOTTOM_BASELINE"] = baseline
            out["DIFF_PRESSURE_CALCULATED"] = dp.rolling(5,center=True,min_periods=1).median()
            rows.append({"parameter":"DIFF_PRESSURE","action":"reconstructed in Stage 2","coverage_pct":out["DIFF_PRESSURE_CALCULATED"].notna().mean()*100})

    # Bit diameter fallback. The schedule is editable and used only where mapped
    # values are absent. It is intentionally visible in the output dataframe.
    if "BIT_DIAMETER_IN" not in out.columns:
        out["BIT_DIAMETER_IN"] = np.nan
    dia = pd.to_numeric(out["BIT_DIAMETER_IN"], errors="coerce")
    if dia.gt(0).mean() < .20 and "HOLE_DEPTH" in out.columns:
        depth = pd.to_numeric(out["HOLE_DEPTH"],errors="coerce")
        # The fallback section schedule is defined in feet and converted to the
        # detected native depth unit. It remains visible and should be replaced
        # by a well-specific Stage-1 mapping whenever available.
        dscale = factors["depth_to_ft"]
        schedule = [(0/dscale,2500/dscale,26.0),(2500/dscale,6000/dscale,17.5),
                    (6000/dscale,9000/dscale,12.25),(9000/dscale,float("inf"),8.5)]
        assigned = pd.Series(np.nan,index=out.index,dtype=float)
        source = pd.Series("unassigned",index=out.index,dtype=object)
        for top,base,size in schedule:
            m=depth.ge(top)&depth.lt(base)
            assigned.loc[m]=size; source.loc[m]=f"fallback section {top:g}-{base:g} {units['depth']}"
        out["BIT_DIAMETER_IN"] = assigned.fillna(8.5)
        out["BIT_DIAMETER_SOURCE"] = source.where(assigned.notna(),"default 8.5 in")
    else:
        out["BIT_DIAMETER_IN"] = dia.where(dia>0).interpolate(limit_direction="both").fillna(8.5)
        if "BIT_DIAMETER_SOURCE" not in out.columns:
            out["BIT_DIAMETER_SOURCE"] = np.where(dia.gt(0),"mapped data column","filled mapped gap")

    # Stage-1-style MSE fallback. Inputs are converted to US engineering units
    # for the equations and the result is converted back to the detected native
    # MSE unit. This avoids silently applying US constants to SI channels.
    torque_col = first_existing(out,["BIT_TORQUE","ROTARY_TORQUE"])
    rpm_col = first_existing(out,["BIT_RPM","ROTARY_RPM"])
    # This fallback recreates the independent Stage-1 calculated candidate;
    # it must not silently select original inputs. Final input selection and
    # selected-input MSE recalculation happen later in this Stage-2 pipeline.
    wob_col = first_existing(out,["WOB_CALCULATED"])
    rop_col = first_existing(out,["ROP_CALCULATED"])
    if torque_col and rpm_col and wob_col and rop_col:
        wob=safe_numeric(out,wob_col)*factors["load_to_klbf"]
        rop=safe_numeric(out,rop_col)*factors["rop_to_ft_hr"]
        torque=safe_numeric(out,torque_col)*factors["torque_to_kft_lbf"]
        rpm=safe_numeric(out,rpm_col); d=pd.to_numeric(out["BIT_DIAMETER_IN"],errors="coerce")
        area=np.pi*d.pow(2)/4.0
        teale=(wob*1000/area)+(120*np.pi*rpm*torque*1000/(area*rop))
        valid=rop.ge(2.0)&wob.ge(0)&torque.ge(0)&rpm.ge(0)&d.gt(0)&np.isfinite(teale)&teale.between(0,2e6)
        if "MSE_TEALE_CALCULATED" not in out.columns or not pd.to_numeric(out["MSE_TEALE_CALCULATED"],errors="coerce").notna().any():
            out["MSE_TEALE_CALCULATED"] = pressure_from_psi(teale.where(valid), units["mse"])
            rows.append({"parameter":"MSE_TEALE","action":"reconstructed in Stage 2","coverage_pct":out["MSE_TEALE_CALCULATED"].notna().mean()*100})
        if {"PUMP_OUTPUT","DIFF_PRESSURE_CALCULATED"}.issubset(out.columns):
            dp=first_existing(out,["DIFF_PRESSURE_CALCULATED"])
            flow=safe_numeric(out,"PUMP_OUTPUT")*factors["flow_to_gpm"]
            hydraulic=teale + 1155*(safe_numeric(out,dp)*factors["pressure_to_psi"])*flow/(area*rop)
            hvalid=valid & np.isfinite(hydraulic) & hydraulic.between(0,2e6)
            if "MSE_HYDRAULIC_CALCULATED" not in out.columns or not pd.to_numeric(out["MSE_HYDRAULIC_CALCULATED"],errors="coerce").notna().any():
                out["MSE_HYDRAULIC_CALCULATED"] = pressure_from_psi(hydraulic.where(hvalid), units["mse"])
                rows.append({"parameter":"MSE_HYDRAULIC","action":"reconstructed in Stage 2","coverage_pct":out["MSE_HYDRAULIC_CALCULATED"].notna().mean()*100})

    return out, pd.DataFrame(rows)


# ============================================================
# Timestamp and basic preprocessing
# ============================================================

def build_timestamp_safely(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    source = None

    if "TIMESTAMP" in out.columns:
        ts = pd.to_datetime(out["TIMESTAMP"], errors="coerce", utc=True)
        if ts.notna().mean() >= 0.80:
            out["TIMESTAMP"] = ts
            source = "TIMESTAMP"

    if source is None and {"DATE", "TIME"}.issubset(out.columns):
        ts = pd.to_datetime(
            out["DATE"].astype(str).str.strip() + " " + out["TIME"].astype(str).str.strip(),
            errors="coerce", utc=True,
        )
        if ts.notna().mean() >= 0.80:
            out["TIMESTAMP"] = ts
            source = "DATE + TIME"

    if source is None:
        raise ValueError("Could not construct a reliable TIMESTAMP column.")

    valid = out["TIMESTAMP"].notna()
    qc = pd.DataFrame([{
        "timestamp_source": source,
        "parse_success_pct": valid.mean() * 100,
        "unique_timestamp_ratio_pct": (
            out.loc[valid, "TIMESTAMP"].nunique() / max(int(valid.sum()), 1) * 100
        ),
        "minimum_timestamp": out["TIMESTAMP"].min(),
        "maximum_timestamp": out["TIMESTAMP"].max(),
    }])
    return out, qc


def replace_sentinels_and_coerce_numeric(
    df: pd.DataFrame, config: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []
    for c in out.columns:
        if c in TIME_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(out[c].dtype):
            rows.append({"column": c, "sentinel_values_replaced": 0, "coercion": "preserved Boolean"})
            continue
        candidate = out[c].replace(list(config.null_sentinels), np.nan)
        before = int(out[c].isin(config.null_sentinels).sum())
        numeric = pd.to_numeric(candidate, errors="coerce")
        non_null = int(candidate.notna().sum())
        numeric_ratio = float(numeric.notna().sum() / max(non_null, 1))
        # Numeric dtypes are always retained. Object columns are converted only
        # when nearly all populated values are actually numeric; lineage/state
        # text therefore survives preprocessing.
        if is_continuous_numeric(candidate) or numeric_ratio >= 0.95:
            out[c] = numeric.astype("float64")
            action = "numeric"
        else:
            out[c] = candidate
            action = "preserved nonnumeric"
        rows.append({"column": c, "sentinel_values_replaced": before, "coercion": action})
    return out, pd.DataFrame(rows)


def sort_deduplicate_resample(
    df: pd.DataFrame, config: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    rows_before = len(out)
    duplicate_rows = int(out["TIMESTAMP"].duplicated().sum())

    numeric_cols = [c for c in out.columns if c not in TIME_COLUMNS and is_continuous_numeric(out[c])]
    if duplicate_rows:
        aggregations = {
            c: ("median" if c in numeric_cols else "first")
            for c in out.columns if c != "TIMESTAMP"
        }
        out = out.groupby("TIMESTAMP", as_index=False, sort=True).agg(aggregations)

    dt_before = out["TIMESTAMP"].diff().dt.total_seconds()
    observed_dt = float(dt_before[dt_before > 0].median()) if (dt_before > 0).any() else np.nan

    applied_rule = None
    requested_rule = config.resample_rule
    if requested_rule:
        if str(requested_rule).lower() == "auto":
            if np.isfinite(observed_dt) and observed_dt > 0:
                applied_rule = (
                    f"{int(round(observed_dt))}s"
                    if observed_dt >= 1.0
                    else f"{max(1, int(round(observed_dt * 1000)))}ms"
                )
        else:
            applied_rule = str(requested_rule)

    if applied_rule:
        interval_seconds = pd.to_timedelta(applied_rule).total_seconds()
        span_seconds = (
            out["TIMESTAMP"].max() - out["TIMESTAMP"].min()
        ).total_seconds() if len(out) > 1 else 0.0
        estimated_rows = int(span_seconds / max(interval_seconds, 1e-9)) + 1
        if estimated_rows <= max(len(out) * 5, len(out) + 1000):
            out = out.set_index("TIMESTAMP").resample(applied_rule).asfreq().reset_index()
        else:
            applied_rule = None

    dt_after = out["TIMESTAMP"].diff().dt.total_seconds()
    qc = pd.DataFrame([{
        "rows_before": rows_before,
        "duplicate_timestamps_aggregated": duplicate_rows,
        "observed_median_sample_interval_sec": observed_dt,
        "resample_rule_requested": requested_rule or "preserve observed sampling",
        "resample_rule_applied": applied_rule or "not applied",
        "rows_after": len(out),
        "final_median_sample_interval_sec": float(dt_after[dt_after > 0].median()) if (dt_after > 0).any() else observed_dt,
        "gaps_gt_5_sec": int((dt_after > 5).sum()),
        "gaps_gt_60_sec": int((dt_after > 60).sum()),
        "maximum_sample_interval_sec": float(dt_after.max()) if dt_after.notna().any() else np.nan,
    }])
    return out, qc


# ============================================================
# Unit detection and conversions
# ============================================================

def detect_units(df: pd.DataFrame, config: PreprocessConfig) -> Tuple[Dict[str, str], pd.DataFrame]:
    text = " ".join(str(c).lower() for c in df.columns)
    rows = []
    units: Dict[str, str] = {}

    def set_unit(family: str, candidates: List[Tuple[str, float, str]], fallback: str):
        override = config.unit_overrides.get(family, "auto")
        if override != "auto":
            unit, confidence, reason = override, 1.0, "user override"
        elif candidates:
            unit, confidence, reason = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        else:
            unit, confidence, reason = fallback, 0.30, "fallback; verify manually"
        units[family] = unit
        rows.append({"unit_family": family, "detected_unit": unit, "confidence": confidence, "reason": reason})

    # A corrected Stage-1 dataframe has already been normalized to the US
    # engineering units documented by Stage 1. This signature must outrank
    # ambiguous magnitude heuristics (for example, a shallow 1,000-ft well).
    stage1_signature = {
        "WOB_CALCULATED", "ROP_CALCULATED", "DIFF_PRESSURE_CALCULATED",
        "BIT_DIAMETER_IN",
    }.issubset(df.columns)

    depth_candidates: List[Tuple[str, float, str]] = []
    if stage1_signature:
        depth_candidates.append(("ft", .995, "corrected Stage-1 canonical-unit contract"))
    if re.search(r"(?:^|[_\s(])ft(?:$|[_\s)/-])|\bfeet\b|\bfoot\b", text):
        depth_candidates.append(("ft", .99, "column-name token"))
    # Word boundaries are essential: ``diameter`` contains the letters
    # ``meter`` but is not a metric-unit declaration.
    if re.search(r"(?:^|[_\s(])m(?:$|[_\s)/-])|\bmeters?\b|\bmetres?\b", text):
        depth_candidates.append(("m", .99, "column-name token"))
    max_depth = max([
        safe_numeric(df, c).max() for c in ["BIT_DEPTH", "HOLE_DEPTH"] if c in df.columns
    ] or [np.nan])
    if np.isfinite(max_depth):
        if max_depth > 7000:
            depth_candidates.append(("ft", .72, "depth magnitude"))
        elif max_depth < 5500:
            depth_candidates.append(("m", .58, "depth magnitude"))
    set_unit("depth", depth_candidates, "ft")

    rop_candidates = []
    if stage1_signature:
        rop_candidates.append(("ft/hr", .995, "corrected Stage-1 canonical-unit contract"))
    if "ft/hr" in text or "ft/h" in text:
        rop_candidates.append(("ft/hr", .99, "column-name token"))
    if "m/hr" in text or "m/h" in text:
        rop_candidates.append(("m/hr", .99, "column-name token"))
    rop_candidates.append(("ft/hr" if units["depth"] == "ft" else "m/hr", .70, "matched depth system"))
    set_unit("rop", rop_candidates, "ft/hr")

    pressure_candidates = []
    if stage1_signature:
        pressure_candidates.append(("psi", .995, "corrected Stage-1 canonical-unit contract"))
    for token, unit in [("psi", "psi"), ("mpa", "MPa"), ("kpa", "kPa"), ("bar", "bar")]:
        if token in text:
            pressure_candidates.append((unit, .99, "column-name token"))
    p_col = first_existing(df, ["PUMP_PRESSURE", "DIFF_PRESSURE_ORIGINAL", "DIFF_PRESSURE_CALCULATED"])
    p_med = safe_numeric(df, p_col)
    p_med = float(p_med[p_med > 0].median()) if (p_med > 0).any() else np.nan
    if np.isfinite(p_med):
        if p_med > 1000:
            pressure_candidates.append(("psi", .75, "pressure magnitude"))
        elif p_med < 60:
            pressure_candidates.append(("MPa", .55, "pressure magnitude"))
        elif p_med < 500:
            pressure_candidates.append(("bar", .55, "pressure magnitude"))
    set_unit("pressure", pressure_candidates, "psi")

    flow_candidates = []
    if stage1_signature:
        flow_candidates.append(("gpm", .995, "corrected Stage-1 canonical-unit contract"))
    if "gpm" in text or "gal/min" in text:
        flow_candidates.append(("gpm", .99, "column-name token"))
    if "l/min" in text or "lpm" in text:
        flow_candidates.append(("L/min", .99, "column-name token"))
    if "m3/h" in text or "m³/h" in text:
        flow_candidates.append(("m3/h", .99, "column-name token"))
    set_unit("flow", flow_candidates, "gpm" if units["depth"] == "ft" else "L/min")

    load_candidates = []
    if stage1_signature:
        load_candidates.append(("klbf", .995, "corrected Stage-1 canonical-unit contract"))
    for token, unit in [("klbf", "klbf"), ("klbs", "klbf"), ("kn", "kN"), ("tonne", "tonne")]:
        if token in text:
            load_candidates.append((unit, .99, "column-name token"))
    set_unit("load", load_candidates, "klbf" if units["depth"] == "ft" else "kN")

    torque_candidates = []
    if stage1_signature:
        torque_candidates.append(("kft·lbf", .995, "corrected Stage-1 canonical-unit contract"))
    if any(x in text for x in ["kn.m", "knm", "kn·m"]):
        torque_candidates.append(("kN·m", .99, "column-name token"))
    if any(x in text for x in ["kft", "1000 ft-lbf", "1000 ft·lbf"]):
        torque_candidates.append(("kft·lbf", .99, "column-name token"))
    if any(x in text for x in ["ft-lbf", "ft·lbf"]):
        torque_candidates.append(("ft·lbf", .90, "column-name token"))
    set_unit("torque", torque_candidates, "kft·lbf" if units["depth"] == "ft" else "kN·m")

    mse_candidates = []
    if stage1_signature:
        mse_candidates.append(("psi", .995, "corrected Stage-1 canonical-unit contract"))
    for token, unit in [("psi", "psi"), ("mpa", "MPa"), ("kpa", "kPa"), ("bar", "bar")]:
        if token in text:
            mse_candidates.append((unit, .90, "column-name token"))
    set_unit("mse", mse_candidates, units["pressure"])
    return units, pd.DataFrame(rows)


def conversion_factors_to_us(units: Dict[str, str]) -> Dict[str, float]:
    return {
        "depth_to_ft": {"ft": 1.0, "m": 3.280839895}[units["depth"]],
        "rop_to_ft_hr": {"ft/hr": 1.0, "m/hr": 3.280839895}[units["rop"]],
        "pressure_to_psi": {"psi": 1.0, "bar": 14.5037738, "MPa": 145.037738, "kPa": 0.145037738}[units["pressure"]],
        "mse_to_psi": {"psi": 1.0, "bar": 14.5037738, "MPa": 145.037738, "kPa": 0.145037738}[units["mse"]],
        "flow_to_gpm": {"gpm": 1.0, "L/min": 0.264172052, "m3/h": 4.40286754}[units["flow"]],
        "load_to_klbf": {"klbf": 1.0, "kN": 0.224808943, "tonne": 2.204622622, "lbf": 0.001}[units["load"]],
        "torque_to_kft_lbf": {"kft·lbf": 1.0, "ft·lbf": 0.001, "kN·m": 0.737562149, "N·m": 0.000737562149}[units["torque"]],
    }


def pressure_from_psi(series: pd.Series, unit: str) -> pd.Series:
    factor = {"psi": 1.0, "bar": 14.5037738, "MPa": 145.037738, "kPa": 0.145037738}[unit]
    return series / factor


# ============================================================
# Cleaning
# ============================================================

def apply_conservative_ranges(
    df: pd.DataFrame, units: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    units = units or {
        "depth": "ft", "rop": "ft/hr", "pressure": "psi",
        "flow": "gpm", "load": "klbf", "torque": "kft·lbf",
    }
    f = conversion_factors_to_us({**units, "mse": units.get("mse", units["pressure"])})
    nonnegative = [
        "HOOK_LOAD", "BIT_DEPTH", "HOLE_DEPTH", "PUMP_OUTPUT", "PUMP_PRESSURE",
        "ROTARY_RPM", "BIT_RPM", "WOB_ORIGINAL", "WOB_CALCULATED",
        "ROP_ORIGINAL", "ROP_CALCULATED", "MSE_TEALE_ORIGINAL",
        "MSE_TEALE_CALCULATED", "MSE_HYDRAULIC_ORIGINAL",
        "MSE_HYDRAULIC_CALCULATED", "BIT_DIAMETER_IN",
    ]
    rows = []
    for c in nonnegative:
        if c not in out.columns:
            continue
        bad = safe_numeric(out, c) < 0
        out.loc[bad, c] = np.nan
        rows.append({"column": c, "rule": "nonnegative", "values_set_to_nan": int(bad.sum())})

    # Broad engineering sanity limits remove impossible telemetry spikes while
    # intentionally retaining a wide operating envelope. Limits are defined in
    # canonical US units and converted to the detected native units.
    upper_limits = {
        "ROP_ORIGINAL": 1000.0 / f["rop_to_ft_hr"],
        "ROP_CALCULATED": 1000.0 / f["rop_to_ft_hr"],
        "ROTARY_RPM": 500.0,
        "BIT_RPM": 1000.0,
        "PUMP_PRESSURE": 20_000.0 / f["pressure_to_psi"],
        "PUMP_OUTPUT": 3_000.0 / f["flow_to_gpm"],
        "HOOK_LOAD": 2_000.0 / f["load_to_klbf"],
        "WOB_ORIGINAL": 300.0 / f["load_to_klbf"],
        "WOB_CALCULATED": 300.0 / f["load_to_klbf"],
        "ROTARY_TORQUE": 200.0 / f["torque_to_kft_lbf"],
        "BIT_TORQUE": 200.0 / f["torque_to_kft_lbf"],
    }
    for c, upper in upper_limits.items():
        if c not in out.columns:
            continue
        bad = safe_numeric(out, c).gt(upper)
        out.loc[bad, c] = np.nan
        rows.append({
            "column": c,
            "rule": f"maximum {upper:.6g} {units.get(BASE_DERIVED.get(c.split('_')[0], {}).get('unit_family', ''), 'native units')}",
            "values_set_to_nan": int(bad.sum()),
        })
    return out, pd.DataFrame(rows)


def fill_short_gaps(df: pd.DataFrame, config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []
    numeric_cols = [
        c for c in out.columns if c not in TIME_COLUMNS
        and is_continuous_numeric(out[c])
        and not c.endswith("_OUTLIER_FLAG")
    ]
    for c in numeric_cols:
        missing = out[c].isna()
        before = int(missing.sum())
        gaps = consecutive_gap_lengths(missing)
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        gap_size = missing.groupby(groups).transform("sum")
        eligible = missing & gap_size.le(config.short_gap_limit)
        interpolated = out[c].interpolate(
            method=config.interpolation_method,
            limit_direction="both",
            limit_area="inside",
        )
        # Fill an entire gap only when its total length is short. Never fill
        # the edges of a long gap, because repeated passes could bridge it.
        out.loc[eligible, c] = interpolated.loc[eligible]
        after = int(out[c].isna().sum())
        rows.append({
            "column": c,
            "nan_before": before,
            "nan_after": after,
            "values_filled": before - after,
            "short_gap_count": sum(g <= config.short_gap_limit for g in gaps),
            "long_gap_count": sum(g > config.short_gap_limit for g in gaps),
            "maximum_gap_samples": max(gaps) if gaps else 0,
        })
    return out, pd.DataFrame(rows)


def correct_isolated_spikes(
    df: pd.DataFrame,
    config: PreprocessConfig,
    units: Dict[str, str],
    phase: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and optionally replace short local spikes with a Hampel filter."""
    out = df.copy()
    f = conversion_factors_to_us(units)
    floor_by_column = {
        "BIT_DEPTH": 0.5 / f["depth_to_ft"],
        "HOLE_DEPTH": 0.5 / f["depth_to_ft"],
        "BLOCK_POSITION": 0.5 / f["depth_to_ft"],
        "HOOK_LOAD": 2.0 / f["load_to_klbf"],
        "WOB_ORIGINAL": 2.0 / f["load_to_klbf"],
        "WOB_CALCULATED": 2.0 / f["load_to_klbf"],
        "ROP_ORIGINAL": 10.0 / f["rop_to_ft_hr"],
        "ROP_CALCULATED": 10.0 / f["rop_to_ft_hr"],
        "PUMP_PRESSURE": 100.0 / f["pressure_to_psi"],
        "DIFF_PRESSURE_ORIGINAL": 100.0 / f["pressure_to_psi"],
        "DIFF_PRESSURE_CALCULATED": 100.0 / f["pressure_to_psi"],
        "PUMP_OUTPUT": 25.0 / f["flow_to_gpm"],
        "ROTARY_RPM": 10.0,
        "BIT_RPM": 15.0,
        "ROTARY_TORQUE": 1.0 / f["torque_to_kft_lbf"],
        "BIT_TORQUE": 1.0 / f["torque_to_kft_lbf"],
    }
    candidates = [
        c for c in dict.fromkeys(
            DIRECT_SENSOR_COLUMNS
            + [x for s in BASE_DERIVED.values()
               for x in s["original_candidates"] + s["calculated_candidates"]]
        )
        if c in out.columns and is_continuous_numeric(out[c])
        and safe_numeric(out, c).notna().any()
    ]
    rows = []
    window = max(5, int(config.hampel_window) | 1)
    for c in candidates:
        s = safe_numeric(out, c)
        local_median = s.rolling(window, center=True, min_periods=max(3, window // 2)).median()
        absolute_deviation = (s - local_median).abs()
        local_mad = absolute_deviation.rolling(
            window, center=True, min_periods=max(3, window // 2)
        ).median()
        robust_sigma = 1.4826 * local_mad
        absolute_floor = float(floor_by_column.get(c, 0.0))
        threshold = (config.hampel_sigma * robust_sigma).clip(lower=absolute_floor)
        candidate_mask = (absolute_deviation > threshold).fillna(False)

        groups = candidate_mask.ne(candidate_mask.shift(fill_value=False)).cumsum()
        run_length = candidate_mask.groupby(groups).transform("sum")
        isolated_mask = candidate_mask & run_length.le(config.maximum_spike_run)
        flag_col = f"{c}_SPIKE_FLAG"
        previous = safe_numeric(out, flag_col).fillna(0).astype(bool) if flag_col in out else False
        out[flag_col] = (previous | isolated_mask).astype("int8")
        if config.spike_action == "replace":
            out.loc[isolated_mask, c] = local_median.loc[isolated_mask]
        rows.append({
            "phase": phase,
            "column": c,
            "window_samples": window,
            "sigma_threshold": config.hampel_sigma,
            "candidate_excursions": int(candidate_mask.sum()),
            "isolated_spikes": int(isolated_mask.sum()),
            "long_excursion_rows_preserved": int((candidate_mask & ~isolated_mask).sum()),
            "action": config.spike_action,
        })
    return out, pd.DataFrame(rows)


def redundant_stuck_sensor_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report long unchanged runs; do not remove zeros that may be operationally valid."""
    rows = []
    for c in DIRECT_SENSOR_COLUMNS:
        if c not in df.columns or not is_continuous_numeric(df[c]):
            continue
        s = safe_numeric(df, c)
        same = s.eq(s.shift()) & s.notna()
        groups = same.ne(same.shift(fill_value=False)).cumsum()
        lengths = same.groupby(groups).transform("sum")
        rows.append({
            "column": c,
            "repeated_adjacent_pct": float(same.mean() * 100),
            "maximum_unchanged_run_samples": int(lengths.max()) if len(lengths) else 0,
            "unchanged_runs_ge_300_samples": int(
                same.groupby(groups).sum().ge(300).sum()
            ),
        })
    return pd.DataFrame(rows)


def flag_or_replace_outliers(df: pd.DataFrame, config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []
    requested = (DIRECT_SENSOR_COLUMNS
        + [x for spec in BASE_DERIVED.values()
           for x in spec["original_candidates"] + spec["calculated_candidates"]])
    numeric_cols = [c for c in dict.fromkeys(requested)
                    if c in out and is_continuous_numeric(out[c])]
    for c in numeric_cols:
        s = safe_numeric(out, c)
        valid = s.dropna().astype("float64")
        zero_fraction = float(valid.eq(0).mean()) if len(valid) else np.nan
        positive = valid[valid > 0]
        regime_aware = zero_fraction >= 0.20 and len(positive) >= 100
        reference = positive if regime_aware else valid
        basis = "active positive values" if regime_aware else "all finite values"
        if reference.empty:
            mask = pd.Series(False, index=out.index)
            lower = upper = np.nan
        else:
            # NumPy/Pandas Boolean quantiles fail because subtraction is not
            # defined for bool. `is_continuous_numeric` excludes them and the
            # float cast is a second defensive layer.
            q1, q3 = reference.quantile([0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                mask = pd.Series(False, index=out.index)
                lower = upper = np.nan
            else:
                lower = q1 - config.iqr_multiplier * iqr
                upper = q3 + config.iqr_multiplier * iqr
                mask = ((s < lower) | (s > upper)).fillna(False)
                if regime_aware:
                    mask &= s.gt(0)
        out[f"{c}_OUTLIER_FLAG"] = mask.astype("int8")
        if config.outlier_action == "replace":
            med = s.rolling(config.outlier_median_window, center=True, min_periods=1).median()
            out.loc[mask, c] = med.loc[mask]
        rows.append({"column": c, "diagnostic_basis": basis,
                     "zero_fraction_pct": zero_fraction * 100,
                     "lower_limit": lower, "upper_limit": upper,
                     "outliers_flagged": int(mask.sum()),
                     "acceptance_gate": False, "action": config.outlier_action})
    return out, pd.DataFrame(rows)


def create_smoothed_copies(df: pd.DataFrame, config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    candidates = list(dict.fromkeys(
        DIRECT_SENSOR_COLUMNS
        + [x for s in BASE_DERIVED.values() for x in s["original_candidates"] + s["calculated_candidates"]]
        + [x for s in MSE_PARAMETERS.values() for x in s["original_candidates"] + s["stage1_candidates"]]
    ))
    rows = []
    for c in candidates:
        if (c not in out.columns or not is_continuous_numeric(out[c])
                or not safe_numeric(out, c).notna().any()):
            continue
        smooth_col = f"{c}_SMOOTH"
        roll = out[c].rolling(config.smoothing_window, center=True, min_periods=1)
        out[smooth_col] = roll.mean() if config.smoothing_method == "mean" else roll.median()
        rows.append({"original_column": c, "smoothed_column": smooth_col, "method": config.smoothing_method, "window": config.smoothing_window})
    return out, pd.DataFrame(rows)


# ============================================================
# Comparison and source selection — WOB, ROP, differential pressure
# ============================================================

def active_mask(df: pd.DataFrame, parameter: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if parameter in {"ROP", "DIFF_PRESSURE"} and "PUMP_OUTPUT" in df.columns:
        flow = safe_numeric(df, "PUMP_OUTPUT")
        threshold = max(1.0, float(flow[flow > 0].quantile(.10))) if (flow > 0).any() else 1.0
        mask &= flow >= threshold
    if parameter == "WOB" and {"BIT_DEPTH", "HOLE_DEPTH"}.issubset(df.columns):
        gap = (safe_numeric(df, "HOLE_DEPTH") - safe_numeric(df, "BIT_DEPTH")).abs()
        tolerance = max(2.0, float(gap.dropna().quantile(.20))) if gap.notna().any() else 5.0
        mask &= gap <= tolerance
    return mask.fillna(False)


def comparison_metrics(original: pd.Series, calculated: pd.Series, mask: pd.Series, minimum_pair_rows: int) -> Dict[str, float]:
    pair_all = pd.DataFrame({"original": original, "calculated": calculated}).dropna()
    pair_active = pd.DataFrame({"original": original, "calculated": calculated})[mask].dropna()
    pair = pair_active if len(pair_active) >= minimum_pair_rows else pair_all
    metrics = {
        "paired_rows_all": len(pair_all),
        "paired_rows_active": len(pair_active),
        "comparison_rows_used": len(pair),
        "correlation": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "bias_calculated_minus_original": np.nan,
        "median_calculated_to_original_ratio": np.nan,
        "identical_pair_pct": np.nan,
        "exactly_identical_warning": False,
    }
    if len(pair) >= 3:
        error = pair["calculated"] - pair["original"]
        metrics["correlation"] = float(pair.corr().iloc[0, 1])
        metrics["mae"] = float(error.abs().mean())
        metrics["rmse"] = float(np.sqrt(np.mean(error ** 2)))
        metrics["bias_calculated_minus_original"] = float(error.mean())
        med_o = float(pair["original"].median())
        metrics["median_calculated_to_original_ratio"] = float(pair["calculated"].median() / med_o) if abs(med_o) > 1e-12 else np.nan
        identical = np.isclose(pair["original"], pair["calculated"], rtol=1e-10, atol=1e-12)
        metrics["identical_pair_pct"] = float(identical.mean() * 100)
        metrics["exactly_identical_warning"] = bool(identical.mean() > 0.999)
    return metrics


def compare_base_derived(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    rows = []
    for parameter, spec in BASE_DERIVED.items():
        original_col = first_existing(df, spec["original_candidates"])
        calculated_col = first_existing(df, spec["calculated_candidates"])
        original = safe_numeric(df, original_col)
        calculated = safe_numeric(df, calculated_col)
        activity = active_mask(df, parameter)
        metrics = comparison_metrics(original, calculated, activity, config.minimum_pair_rows)
        original_cov = float(original.notna().mean())
        calculated_cov = float(calculated.notna().mean())
        original_active = original[activity].dropna()
        calculated_active = calculated[activity].dropna()
        original_negative_active_pct = (
            float(original_active.lt(0).mean() * 100) if not original_active.empty else np.nan
        )
        calculated_negative_active_pct = (
            float(calculated_active.lt(0).mean() * 100) if not calculated_active.empty else np.nan
        )

        if original_cov >= config.minimum_coverage and calculated_cov < config.minimum_coverage:
            recommendation, reason = "original", "Only original has sufficient coverage."
        elif calculated_cov >= config.minimum_coverage and original_cov < config.minimum_coverage:
            recommendation, reason = "calculated", "Only calculated has sufficient coverage."
        elif original_cov >= config.minimum_coverage and calculated_cov >= config.minimum_coverage:
            recommendation = "original"
            corr = metrics["correlation"]
            if (parameter == "DIFF_PRESSURE"
                    and np.isfinite(original_negative_active_pct)
                    and original_negative_active_pct > 5.0
                    and (not np.isfinite(calculated_negative_active_pct)
                         or calculated_negative_active_pct <= 5.0)):
                recommendation = "calculated"
                reason = (
                    "Original differential pressure is negative in "
                    f"{original_negative_active_pct:.1f}% of pump-on valid rows; "
                    "hydraulic MSE requires a nonnegative pressure drop."
                )
            elif metrics["exactly_identical_warning"]:
                reason = "Original and calculated are effectively identical. Verify Stage 1 mapping/calculation inputs before accepting this result."
            elif np.isfinite(corr) and corr >= config.strong_correlation:
                reason = "Strong agreement; original remains the conservative default."
            elif np.isfinite(corr) and corr >= config.moderate_correlation:
                reason = "Moderate agreement; inspect plots before selecting calculated."
            else:
                reason = "Weak or unverified agreement; calculated requires technical validation."
        else:
            recommendation, reason = "unavailable", "Neither source has sufficient coverage."

        rows.append({
            "parameter": parameter,
            "original_column": original_col,
            "calculated_column": calculated_col,
            "original_coverage_pct": original_cov * 100,
            "calculated_coverage_pct": calculated_cov * 100,
            "original_negative_active_pct": original_negative_active_pct,
            "calculated_negative_active_pct": calculated_negative_active_pct,
            **metrics,
            "automatic_recommendation": recommendation,
            "recommendation_reason": reason,
        })
    return pd.DataFrame(rows)


def choose_source(prompt_name: str, original_ok: bool, calculated_ok: bool, recommendation: str, config: PreprocessConfig) -> str:
    if original_ok and not calculated_ok:
        return "original"
    if calculated_ok and not original_ok:
        return "calculated"
    if not original_ok and not calculated_ok:
        return "unavailable"
    mode = config.selection_mode.lower()
    if mode in {"original", "calculated"}:
        return mode
    if mode == "automatic":
        return recommendation if recommendation in {"original", "calculated"} else "original"
    print("\n" + "=" * 72)
    print(f"SOURCE SELECTION — {prompt_name}")
    print("=" * 72)
    print(f"Automatic recommendation: {recommendation}")
    print("[1] Original/EDR  [2] Recalculated  [3 or Enter] Recommendation")
    value = input("Selection: ").strip().lower()
    if value in {"1", "o", "original", "edr"}:
        return "original"
    if value in {"2", "c", "calculated", "recalculated"}:
        return "calculated"
    return recommendation if recommendation in {"original", "calculated"} else "original"


def select_base_derived(df: pd.DataFrame, report: pd.DataFrame, config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []
    for parameter, spec in BASE_DERIVED.items():
        r = report.loc[report["parameter"] == parameter].iloc[0]
        original_col = r["original_column"] if pd.notna(r["original_column"]) else None
        calculated_col = r["calculated_column"] if pd.notna(r["calculated_column"]) else None
        original_ok = original_col is not None and safe_numeric(out, original_col).notna().any()
        calculated_ok = calculated_col is not None and safe_numeric(out, calculated_col).notna().any()
        choice = choose_source(parameter, original_ok, calculated_ok, str(r["automatic_recommendation"]), config)
        source_col = original_col if choice == "original" else calculated_col if choice == "calculated" else None
        out[spec["selected"]] = safe_numeric(out, source_col)
        rows.append({
            "parameter": parameter,
            "selected_source": choice,
            "selected_source_column": source_col,
            "selected_output_column": spec["selected"],
            "automatic_recommendation": r["automatic_recommendation"],
            "recommendation_reason": r["recommendation_reason"],
        })
    return out, pd.DataFrame(rows)


# ============================================================
# MSE recalculation after selected WOB/ROP/DP
# ============================================================

def calculate_teale_mse_psi(wob_klbf, torque_kft_lbf, rpm, rop_ft_hr, diameter_in, config: PreprocessConfig):
    area_in2 = np.pi * diameter_in.pow(2) / 4.0
    axial = wob_klbf * 1000.0 / area_in2
    rotary = 120.0 * np.pi * rpm * torque_kft_lbf * 1000.0 / (area_in2 * rop_ft_hr)
    result = axial + rotary
    valid = (
        rop_ft_hr.ge(config.mse_min_rop_ft_hr)
        & diameter_in.gt(0)
        & wob_klbf.ge(0)
        & torque_kft_lbf.ge(0)
        & rpm.ge(0)
        & np.isfinite(result)
        & result.ge(0)
        & result.le(config.mse_max_psi)
    )
    return result.where(valid)


def calculate_hydraulic_mse_psi(teale_psi, dp_psi, flow_gpm, rop_ft_hr, diameter_in, config: PreprocessConfig):
    area_in2 = np.pi * diameter_in.pow(2) / 4.0
    hydraulic = 1155.0 * dp_psi * flow_gpm / (area_in2 * rop_ft_hr)
    result = teale_psi + hydraulic
    valid = (
        rop_ft_hr.ge(config.mse_min_rop_ft_hr)
        & diameter_in.gt(0)
        & dp_psi.ge(0)
        & flow_gpm.ge(0)
        & np.isfinite(result)
        & result.ge(0)
        & result.le(config.mse_max_psi)
    )
    return result.where(valid)


def recalculate_mse_from_selected(df: pd.DataFrame, units: Dict[str, str], config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    f = conversion_factors_to_us(units)

    wob_klbf = safe_numeric(out, "WOB_SELECTED") * f["load_to_klbf"]
    rop_ft_hr = safe_numeric(out, "ROP_SELECTED") * f["rop_to_ft_hr"]
    dp_psi = safe_numeric(out, "DIFF_PRESSURE_SELECTED") * f["pressure_to_psi"]
    flow_gpm = safe_numeric(out, "PUMP_OUTPUT") * f["flow_to_gpm"]
    diameter_in = safe_numeric(out, "BIT_DIAMETER_IN")

    torque_source = first_existing(out, ["BIT_TORQUE", "ROTARY_TORQUE"])
    rpm_source = first_existing(out, ["BIT_RPM", "ROTARY_RPM"])
    torque_kft_lbf = safe_numeric(out, torque_source) * f["torque_to_kft_lbf"]
    rpm = safe_numeric(out, rpm_source)

    teale_psi = calculate_teale_mse_psi(wob_klbf, torque_kft_lbf, rpm, rop_ft_hr, diameter_in, config)
    hydraulic_psi = calculate_hydraulic_mse_psi(teale_psi, dp_psi, flow_gpm, rop_ft_hr, diameter_in, config)

    out["MSE_TEALE_SELECTED_RECALCULATED_PSI"] = teale_psi
    out["MSE_HYDRAULIC_SELECTED_RECALCULATED_PSI"] = hydraulic_psi
    out["MSE_TEALE_SELECTED_RECALCULATED"] = pressure_from_psi(teale_psi, units["mse"])
    out["MSE_HYDRAULIC_SELECTED_RECALCULATED"] = pressure_from_psi(hydraulic_psi, units["mse"])

    report = pd.DataFrame([{
        "torque_source": torque_source,
        "rpm_source": rpm_source,
        "bit_diameter_column": "BIT_DIAMETER_IN" if "BIT_DIAMETER_IN" in out.columns else None,
        "teale_selected_recalculated_coverage_pct": out["MSE_TEALE_SELECTED_RECALCULATED"].notna().mean() * 100,
        "hydraulic_selected_recalculated_coverage_pct": out["MSE_HYDRAULIC_SELECTED_RECALCULATED"].notna().mean() * 100,
        "mse_native_display_unit": units["mse"],
        "calculation_internal_unit": "psi",
    }])
    return out, report


def compare_and_select_final_mse(df: pd.DataFrame, config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    comparison_rows = []
    selection_rows = []

    for parameter, spec in MSE_PARAMETERS.items():
        original_col = first_existing(out, spec["original_candidates"])
        stage1_col = first_existing(out, spec["stage1_candidates"])
        selected_recalc_col = spec["selected_recalculated"] if spec["selected_recalculated"] in out.columns else None

        original = safe_numeric(out, original_col)
        stage1 = safe_numeric(out, stage1_col)
        selected_recalc = safe_numeric(out, selected_recalc_col)

        # MSE exists only while drilling inputs are physically valid. Global
        # coverage across stationary/tripping rows is therefore not a valid
        # availability test.
        mse_active = (
            safe_numeric(out, "ROP_SELECTED").ge(config.mse_min_rop_ft_hr)
            & safe_numeric(out, "BIT_DIAMETER_IN").gt(0)
        ).fillna(False)
        active_rows = int(mse_active.sum())

        def source_stats(series: pd.Series) -> Dict[str, float]:
            valid_all = int(series.notna().sum())
            valid_active = int(series[mse_active].notna().sum())
            return {
                "valid_rows_all": valid_all,
                "valid_rows_active": valid_active,
                "global_coverage_pct": valid_all / max(len(out), 1) * 100.0,
                "active_coverage_pct": valid_active / max(active_rows, 1) * 100.0,
            }

        original_stats = source_stats(original)
        stage1_stats = source_stats(stage1)
        selected_stats = source_stats(selected_recalc)

        def usable(stats: Dict[str, float]) -> bool:
            return (
                stats["valid_rows_active"] >= max(3, config.minimum_pair_rows)
                and stats["active_coverage_pct"] >= config.minimum_coverage * 100.0
            )

        original_ok = usable(original_stats)
        stage1_ok = usable(stage1_stats)
        selected_ok = usable(selected_stats)

        # Report two distinct comparisons. The second one explains why Stage-1
        # MSE can lag selected-input MSE even when no EDR MSE channel exists.
        primary_calc_col = selected_recalc_col or stage1_col
        primary_calc = safe_numeric(out, primary_calc_col)
        edr_metrics = comparison_metrics(
            original,
            primary_calc,
            mse_active,
            config.minimum_pair_rows,
        )
        stage1_selected_metrics = comparison_metrics(
            stage1,
            selected_recalc,
            mse_active,
            config.minimum_pair_rows,
        )

        if original_ok and selected_ok:
            recommendation = "original"
            if edr_metrics["exactly_identical_warning"]:
                reason = "EDR MSE and recalculated MSE are effectively identical; verify that they are not mapped to the same source."
            elif np.isfinite(edr_metrics["correlation"]) and edr_metrics["correlation"] >= config.strong_correlation:
                reason = "Strong agreement; EDR MSE remains conservative default, with recalculated MSE retained for validation."
            else:
                reason = "EDR and recalculated MSE disagree or are insufficiently validated; inspect the plots and formula inputs."
        elif selected_ok:
            recommendation = "calculated"
            reason = "No usable EDR MSE exists; use MSE recalculated from the selected WOB/ROP/DP inputs."
        elif stage1_ok:
            recommendation = "calculated"
            reason = "Selected-input MSE is unavailable; use Stage-1 calculated MSE after checking missing inputs."
        elif original_ok:
            recommendation = "original"
            reason = "Only EDR MSE is available."
        else:
            recommendation = "unavailable"
            reason = "No MSE source has sufficient coverage within physically valid drilling rows."

        def prefixed(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
            return {f"{prefix}_{key}": value for key, value in metrics.items()}

        comparison_rows.append({
            "parameter": parameter,
            "edr_original_column": original_col,
            "stage1_calculated_column": stage1_col,
            "selected_input_recalculated_column": selected_recalc_col,
            "primary_calculated_column_for_validation": primary_calc_col,
            "mse_active_rows": active_rows,
            "edr_valid_rows_active": original_stats["valid_rows_active"],
            "stage1_valid_rows_active": stage1_stats["valid_rows_active"],
            "selected_recalculated_valid_rows_active": selected_stats["valid_rows_active"],
            "edr_active_coverage_pct": original_stats["active_coverage_pct"],
            "stage1_active_coverage_pct": stage1_stats["active_coverage_pct"],
            "selected_recalculated_active_coverage_pct": selected_stats["active_coverage_pct"],
            "edr_global_coverage_pct": original_stats["global_coverage_pct"],
            "stage1_global_coverage_pct": stage1_stats["global_coverage_pct"],
            "selected_recalculated_global_coverage_pct": selected_stats["global_coverage_pct"],
            **prefixed(edr_metrics, "edr_vs_selected"),
            **prefixed(stage1_selected_metrics, "stage1_vs_selected"),
            "stage1_median": float(stage1[mse_active].median()) if stage1[mse_active].notna().any() else np.nan,
            "selected_recalculated_median": float(selected_recalc[mse_active].median()) if selected_recalc[mse_active].notna().any() else np.nan,
            "selected_recalculated_p95": float(selected_recalc[mse_active].quantile(.95)) if selected_recalc[mse_active].notna().any() else np.nan,
            "selected_recalculated_p99": float(selected_recalc[mse_active].quantile(.99)) if selected_recalc[mse_active].notna().any() else np.nan,
            "automatic_recommendation": recommendation,
            "recommendation_reason": reason,
        })

        calculated_ok = selected_ok or stage1_ok
        choice = choose_source(parameter, original_ok, calculated_ok, recommendation, config)

        if choice == "original":
            source_col = original_col
        elif choice == "calculated":
            source_col = selected_recalc_col if selected_ok else stage1_col
        else:
            source_col = None

        out[spec["final"]] = safe_numeric(out, source_col)
        selection_rows.append({
            "parameter": parameter,
            "selected_source": choice,
            "selected_source_column": source_col,
            "selected_output_column": spec["final"],
            "automatic_recommendation": recommendation,
            "recommendation_reason": reason,
        })

    return out, pd.DataFrame(comparison_rows), pd.DataFrame(selection_rows)


# ============================================================
# FSM columns and reports
# ============================================================

def prepare_fsm_input_signals(
    df: pd.DataFrame, units: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the final cleaned/smoothed signal layer and an exclusion mask."""
    out = df.copy()
    candidates = [
        "BIT_DEPTH", "HOLE_DEPTH", "BLOCK_POSITION", "HOOK_LOAD",
        "PUMP_OUTPUT", "PUMP_PRESSURE", "ROTARY_RPM", "BIT_RPM",
        "ROTARY_TORQUE", "BIT_TORQUE", "WOB_SELECTED", "ROP_SELECTED",
        "DIFF_PRESSURE_SELECTED", "MSE_TEALE_SELECTED",
        "MSE_HYDRAULIC_SELECTED",
    ]
    rows = []
    window = max(1, int(config.smoothing_window))
    for source in candidates:
        if source not in out.columns or not safe_numeric(out, source).notna().any():
            continue
        target = f"{source}_FSM_INPUT"
        s = safe_numeric(out, source)
        if config.use_smoothed_fsm_inputs and window > 1:
            rolling = s.rolling(window, center=True, min_periods=1)
            out[target] = rolling.mean() if config.smoothing_method == "mean" else rolling.median()
            method = f"{config.smoothing_method} window={window}"
        else:
            out[target] = s
            method = "cleaned unsmoothed"
        rows.append({
            "source_column": source,
            "fsm_input_column": target,
            "preparation": method,
            "coverage_pct": out[target].notna().mean() * 100,
        })

    required_candidates = [
        "BIT_DEPTH", "HOLE_DEPTH", "HOOK_LOAD", "PUMP_OUTPUT",
        "PUMP_PRESSURE", "ROTARY_RPM", "ROTARY_TORQUE",
        "WOB_SELECTED", "ROP_SELECTED", "DIFF_PRESSURE_SELECTED",
    ]
    required = [f"{c}_FSM_INPUT" for c in required_candidates if f"{c}_FSM_INPUT" in out]
    missing_required = out[required].isna().any(axis=1) if required else pd.Series(True, index=out.index)
    physical_conflict = pd.Series(False, index=out.index)
    if {"BIT_DEPTH_FSM_INPUT", "HOLE_DEPTH_FSM_INPUT"}.issubset(out.columns):
        tolerance = 2.0 if units["depth"] == "ft" else 0.61
        physical_conflict = (
            safe_numeric(out, "BIT_DEPTH_FSM_INPUT")
            - safe_numeric(out, "HOLE_DEPTH_FSM_INPUT")
        ).gt(tolerance).fillna(False)
    out["FSM_INPUT_VALID"] = (~missing_required & ~physical_conflict).astype(bool)
    out["FSM_INVALID_REASON_COUNT"] = (
        missing_required.astype("int8") + physical_conflict.astype("int8")
    )
    rows.append({
        "source_column": "quality gate",
        "fsm_input_column": "FSM_INPUT_VALID",
        "preparation": "exclude long-gap/physically inconsistent rows from automated analysis",
        "coverage_pct": out["FSM_INPUT_VALID"].mean() * 100,
    })
    return out, pd.DataFrame(rows)


def final_fsm_spike_pass(
    df: pd.DataFrame, units: Dict[str, str], config: PreprocessConfig,
    replace_values: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assess/correct the exact signals consumed by the FSM and log every event."""
    out = df.copy()
    f = conversion_factors_to_us(units)
    floors_us = {
        "BIT_DEPTH": 0.5, "HOLE_DEPTH": 0.5, "BLOCK_POSITION": 0.5,
        "HOOK_LOAD": 2.0, "PUMP_OUTPUT": 25.0, "PUMP_PRESSURE": 100.0,
        "ROTARY_RPM": 10.0, "ROTARY_TORQUE": 1.0, "WOB_SELECTED": 2.0,
        "ROP_SELECTED": 10.0, "DIFF_PRESSURE_SELECTED": 100.0,
    }
    divisors = {
        "BIT_DEPTH": f["depth_to_ft"], "HOLE_DEPTH": f["depth_to_ft"],
        "BLOCK_POSITION": f["depth_to_ft"], "HOOK_LOAD": f["load_to_klbf"],
        "PUMP_OUTPUT": f["flow_to_gpm"], "PUMP_PRESSURE": f["pressure_to_psi"],
        "ROTARY_RPM": 1.0, "ROTARY_TORQUE": f["torque_to_kft_lbf"],
        "WOB_SELECTED": f["load_to_klbf"], "ROP_SELECTED": f["rop_to_ft_hr"],
        "DIFF_PRESSURE_SELECTED": f["pressure_to_psi"],
    }
    event_frames, summaries = [], []
    window = max(5, int(config.hampel_window) | 1)
    iterations = max(1, config.final_spike_iterations if replace_values else 1)
    timestamp = out["TIMESTAMP"] if "TIMESTAMP" in out else pd.Series(out.index, index=out.index)
    for iteration in range(1, iterations + 1):
        any_events = False
        for base, floor_us in floors_us.items():
            col = f"{base}_FSM_INPUT"
            if col not in out:
                continue
            s = safe_numeric(out, col)
            median = s.rolling(window, center=True, min_periods=max(3, window // 2)).median()
            deviation = (s - median).abs()
            mad = deviation.rolling(window, center=True, min_periods=max(3, window // 2)).median()
            threshold = (config.hampel_sigma * 1.4826 * mad).clip(
                lower=floor_us / divisors[base]
            )
            candidates = (deviation > threshold).fillna(False)
            groups = candidates.ne(candidates.shift(fill_value=False)).cumsum()
            run = candidates.groupby(groups).transform("sum")
            mask = candidates & run.le(config.maximum_spike_run)
            count = int(mask.sum())
            if count:
                any_events = True
                events = pd.DataFrame({
                    "timestamp": timestamp[mask], "column": col,
                    "iteration": iteration, "value": s[mask],
                    "local_median": median[mask], "threshold": threshold[mask],
                    "absolute_deviation": deviation[mask],
                    "previous_value": s.shift(1)[mask], "next_value": s.shift(-1)[mask],
                    "replacement_value": median[mask],
                    "action": "replaced" if replace_values else "flagged",
                })
                event_frames.append(events.reset_index(drop=True))
                if replace_values:
                    out.loc[mask, col] = median.loc[mask]
            summaries.append({"iteration": iteration, "column": col,
                              "isolated_spikes": count,
                              "action": "replace" if replace_values else "flag"})
        if not any_events or not replace_values:
            break
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(
        columns=["timestamp", "column", "iteration", "value", "local_median",
                 "threshold", "absolute_deviation", "previous_value", "next_value",
                 "replacement_value", "action"])
    return out, pd.DataFrame(summaries), events


def apply_final_logical_corrections(
    df: pd.DataFrame, units: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correct defensible FSM-input conflicts and preserve an event audit trail."""
    out = df.copy()
    f = conversion_factors_to_us(units)
    events = []
    ts = out["TIMESTAMP"] if "TIMESTAMP" in out else pd.Series(out.index, index=out.index)
    needed = {"ROP_SELECTED_FSM_INPUT", "ROTARY_RPM_FSM_INPUT", "PUMP_OUTPUT_FSM_INPUT"}
    if needed.issubset(out.columns):
        rop = safe_numeric(out, "ROP_SELECTED_FSM_INPUT")
        rpm = safe_numeric(out, "ROTARY_RPM_FSM_INPUT")
        flow = safe_numeric(out, "PUMP_OUTPUT_FSM_INPUT")
        mask = (rop * f["rop_to_ft_hr"] > config.rop_active_threshold_ft_hr)
        mask &= rpm <= config.rpm_off_threshold
        mask &= flow * f["flow_to_gpm"] <= config.flow_off_threshold_gpm
        for i in out.index[mask]:
            events.append({"timestamp": ts.loc[i], "event": "ROP while RPM and flow off",
                           "action": "ROP FSM input set to zero", "original_value": rop.loc[i],
                           "replacement_value": 0.0, "magnitude": rop.loc[i]})
        if config.correct_impossible_rop:
            out.loc[mask, "ROP_SELECTED_FSM_INPUT"] = 0.0

    out["FSM_HOLE_DEPTH_LARGE_DECREASE"] = False
    if "HOLE_DEPTH_FSM_INPUT" in out:
        col = "HOLE_DEPTH_FSM_INPUT"
        small_limit = config.hole_depth_small_decrease_ft / f["depth_to_ft"]
        # Sequential processing ensures consecutive tiny reversals do not survive.
        values = safe_numeric(out, col).copy()
        previous = np.nan
        for i in out.index:
            current = values.loc[i]
            if not np.isfinite(current):
                continue
            if np.isfinite(previous) and current < previous:
                drop = previous - current
                small = drop <= small_limit
                events.append({"timestamp": ts.loc[i], "event": "HOLE_DEPTH decrease",
                               "action": "flattened to previous" if small else "flagged invalid",
                               "original_value": current,
                               "replacement_value": previous if small else current,
                               "magnitude": drop})
                if small:
                    values.loc[i] = previous
                    current = previous
                else:
                    out.loc[i, "FSM_HOLE_DEPTH_LARGE_DECREASE"] = True
            previous = current
        out[col] = values
    return out, pd.DataFrame(events, columns=["timestamp", "event", "action",
                                               "original_value", "replacement_value", "magnitude"])


def refresh_fsm_validity(df: pd.DataFrame, units: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    bases = ["BIT_DEPTH", "HOLE_DEPTH", "HOOK_LOAD", "PUMP_OUTPUT", "PUMP_PRESSURE",
             "ROTARY_RPM", "ROTARY_TORQUE", "WOB_SELECTED", "ROP_SELECTED",
             "DIFF_PRESSURE_SELECTED"]
    required = [f"{x}_FSM_INPUT" for x in bases if f"{x}_FSM_INPUT" in out]
    missing_count = out[required].isna().sum(axis=1) if required else pd.Series(1, index=out.index)
    conflict = pd.Series(False, index=out.index)
    if {"BIT_DEPTH_FSM_INPUT", "HOLE_DEPTH_FSM_INPUT"}.issubset(out.columns):
        tolerance = 2.0 if units["depth"] == "ft" else 0.61
        conflict = (safe_numeric(out, "BIT_DEPTH_FSM_INPUT") -
                    safe_numeric(out, "HOLE_DEPTH_FSM_INPUT")).gt(tolerance).fillna(False)
    large_drop = out.get("FSM_HOLE_DEPTH_LARGE_DECREASE", pd.Series(False, index=out.index)).astype(bool)
    out["FSM_MISSING_REQUIRED_COUNT"] = missing_count.astype("int16")
    out["FSM_PHYSICAL_CONFLICT"] = conflict
    out["FSM_INPUT_VALID"] = missing_count.eq(0) & ~conflict & ~large_drop
    out["FSM_INVALID_REASON_COUNT"] = missing_count + conflict.astype(int) + large_drop.astype(int)
    return out


def rop_tail_statistics(df: pd.DataFrame, units: Dict[str, str]) -> pd.DataFrame:
    f = conversion_factors_to_us(units)
    rows = []
    for col in ["ROP_ORIGINAL", "ROP_CALCULATED", "ROP_SELECTED", "ROP_SELECTED_FSM_INPUT", "ROP_FSM_FT_HR"]:
        if col not in df:
            continue
        s = safe_numeric(df, col).dropna()
        if col != "ROP_FSM_FT_HR":
            s = s * f["rop_to_ft_hr"]
        rows.append({"column": col, "unit": "ft/hr", "non_null_rows": len(s),
                     "p95": s.quantile(.95), "p99": s.quantile(.99),
                     "p99_9": s.quantile(.999), "maximum": s.max(),
                     **{f"rows_above_{x}": int((s > x).sum()) for x in [200, 300, 500, 750]}})
    return pd.DataFrame(rows)


def contextual_stuck_sensor_report(df: pd.DataFrame) -> pd.DataFrame:
    """Long flat runs are suspicious only when a related channel indicates activity."""
    rows, window = [], 300
    relationships = {
        "HOLE_DEPTH_FSM_INPUT": ("ROP_SELECTED_FSM_INPUT", 0.0),
        "BIT_DEPTH_FSM_INPUT": ("BLOCK_POSITION_FSM_INPUT", 0.0),
        "HOOK_LOAD_FSM_INPUT": ("BLOCK_POSITION_FSM_INPUT", 0.0),
        "PUMP_PRESSURE_FSM_INPUT": ("PUMP_OUTPUT_FSM_INPUT", 0.0),
        "ROTARY_TORQUE_FSM_INPUT": ("ROTARY_RPM_FSM_INPUT", 1.0),
    }
    for col, (context_col, active_min) in relationships.items():
        if col not in df or context_col not in df:
            continue
        s, context = safe_numeric(df, col), safe_numeric(df, context_col)
        flat = s.diff().abs().rolling(window, min_periods=window).sum().eq(0)
        context_change = context.diff().abs().rolling(window, min_periods=window).sum().gt(0)
        suspect = flat & context_change & context.gt(active_min)
        groups = suspect.ne(suspect.shift(fill_value=False)).cumsum()
        longest = int(suspect.groupby(groups).sum().max()) if len(suspect) else 0
        rows.append({"column": col, "context_column": context_col,
                     "window_samples": window, "contextual_suspect_rows": int(suspect.sum()),
                     "contextual_suspect_pct": suspect.mean() * 100,
                     "longest_suspect_run_samples": longest, "acceptance_gate": False})
    return pd.DataFrame(rows)

def add_fsm_columns(df: pd.DataFrame, units: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    f = conversion_factors_to_us(units)
    rows = []

    def add(source: str, target: str, factor: float, native_unit: str, fsm_unit: str):
        prepared_source = f"{source}_FSM_INPUT" if f"{source}_FSM_INPUT" in out.columns else source
        if prepared_source not in out.columns:
            return
        out[target] = safe_numeric(out, prepared_source) * factor
        out.loc[~out.get("FSM_INPUT_VALID", pd.Series(True, index=out.index)), target] = np.nan
        rows.append({"source_column": prepared_source, "fsm_column": target, "native_unit": native_unit, "fsm_unit": fsm_unit, "factor": factor})

    for c in ["BIT_DEPTH", "HOLE_DEPTH", "BLOCK_POSITION", "DEPTH_OF_CUT"]:
        add(c, f"{c}_FSM_FT", f["depth_to_ft"], units["depth"], "ft")
    add("ROP_SELECTED", "ROP_FSM_FT_HR", f["rop_to_ft_hr"], units["rop"], "ft/hr")
    add("WOB_SELECTED", "WOB_FSM_KLBF", f["load_to_klbf"], units["load"], "klbf")
    add("PUMP_PRESSURE", "PUMP_PRESSURE_FSM_PSI", f["pressure_to_psi"], units["pressure"], "psi")
    add("DIFF_PRESSURE_SELECTED", "DIFF_PRESSURE_FSM_PSI", f["pressure_to_psi"], units["pressure"], "psi")
    add("PUMP_OUTPUT", "PUMP_OUTPUT_FSM_GPM", f["flow_to_gpm"], units["flow"], "gpm")
    add("ROTARY_TORQUE", "ROTARY_TORQUE_FSM_KFT_LBF", f["torque_to_kft_lbf"], units["torque"], "kft·lbf")
    add("BIT_TORQUE", "BIT_TORQUE_FSM_KFT_LBF", f["torque_to_kft_lbf"], units["torque"], "kft·lbf")
    add("MSE_TEALE_SELECTED", "MSE_TEALE_FSM_PSI", f["mse_to_psi"], units["mse"], "psi")
    add("MSE_HYDRAULIC_SELECTED", "MSE_HYDRAULIC_FSM_PSI", f["mse_to_psi"], units["mse"], "psi")
    return out, pd.DataFrame(rows)


def sensor_availability(df: pd.DataFrame) -> pd.DataFrame:
    candidates = list(dict.fromkeys(
        DIRECT_SENSOR_COLUMNS
        + [x for s in BASE_DERIVED.values() for x in s["original_candidates"] + s["calculated_candidates"]]
        + [x for s in MSE_PARAMETERS.values() for x in s["original_candidates"] + s["stage1_candidates"]]
    ))
    return pd.DataFrame([{
        "column": c,
        "available": c in df.columns,
        "coverage_pct": float(df[c].notna().mean() * 100) if c in df.columns else 0.0,
        "non_null_rows": int(df[c].notna().sum()) if c in df.columns else 0,
    } for c in candidates])


def logical_checks(df: pd.DataFrame, units: Dict[str, str]) -> pd.DataFrame:
    rows = []
    f = conversion_factors_to_us(units)
    bit = "BIT_DEPTH_FSM_INPUT" if "BIT_DEPTH_FSM_INPUT" in df else "BIT_DEPTH"
    hole = "HOLE_DEPTH_FSM_INPUT" if "HOLE_DEPTH_FSM_INPUT" in df else "HOLE_DEPTH"
    valid = df.get("FSM_INPUT_VALID", pd.Series(True, index=df.index)).fillna(False)
    if {bit, hole}.issubset(df.columns):
        tol = 2.0 if units["depth"] == "ft" else 0.61
        rows.append({"check": f"BIT_DEPTH exceeds HOLE_DEPTH by > {tol:.2f} {units['depth']}", "violations": int(((safe_numeric(df, bit) - safe_numeric(df, hole) > tol) & valid).sum())})
        rows.append({"check": "HOLE_DEPTH decreases on valid FSM rows", "violations": int(((safe_numeric(df, hole).diff() < 0) & valid & valid.shift(fill_value=False)).sum())})
    rop = "ROP_SELECTED_FSM_INPUT" if "ROP_SELECTED_FSM_INPUT" in df else "ROP_SELECTED"
    rpm = "ROTARY_RPM_FSM_INPUT" if "ROTARY_RPM_FSM_INPUT" in df else "ROTARY_RPM"
    flow = "PUMP_OUTPUT_FSM_INPUT" if "PUMP_OUTPUT_FSM_INPUT" in df else "PUMP_OUTPUT"
    if {rop, rpm, flow}.issubset(df.columns):
        # Use the same engineering thresholds used by the correction step.
        # Tiny positive values below 0.5 ft/hr are numerical/sensor residue,
        # not evidence of drilling activity.
        rows.append({
            "check": "Selected ROP active while RPM and flow are both off",
            "violations": int((
                (safe_numeric(df, rop) * f["rop_to_ft_hr"] > 0.5)
                & (safe_numeric(df, rpm) <= 1.0)
                & (safe_numeric(df, flow) * f["flow_to_gpm"] <= 1.0)
                & valid
            ).sum()),
        })
    return pd.DataFrame(rows)


def build_handoff_report(
    df: pd.DataFrame, residual: pd.DataFrame, logical: pd.DataFrame,
    timestamp_qc: pd.DataFrame, unit_report: pd.DataFrame, config: PreprocessConfig,
) -> pd.DataFrame:
    # Depth, ROP and pressure spikes directly compromise footage, MSE and FSM
    # interpretation. Load/torque/RPM/flow/block-position excursions can be
    # legitimate one-to-three-sample state transitions, so they remain visible
    # as non-blocking review items rather than forcing destructive over-cleaning.
    critical_spike_columns = {
        "BIT_DEPTH_FSM_INPUT", "HOLE_DEPTH_FSM_INPUT",
        "ROP_SELECTED_FSM_INPUT", "PUMP_PRESSURE_FSM_INPUT",
        "DIFF_PRESSURE_SELECTED_FSM_INPUT",
    }
    if residual.empty:
        critical_residual_count = review_residual_count = 0
    else:
        is_critical = residual["column"].isin(critical_spike_columns)
        critical_residual_count = int(residual.loc[is_critical, "isolated_spikes"].sum())
        review_residual_count = int(residual.loc[~is_critical, "isolated_spikes"].sum())
    logical_count = int(logical["violations"].sum()) if not logical.empty else 0
    valid_pct = float(df.get("FSM_INPUT_VALID", pd.Series(False, index=df.index)).mean() * 100)
    timestamp_ok = bool(not timestamp_qc.empty and timestamp_qc.iloc[0].get("parse_success_pct", 100) >= 99)
    unit_ok = bool(not unit_report.empty and unit_report["detected_unit"].notna().all()) if "detected_unit" in unit_report else True
    checks = [
        ("Timestamp parsing", timestamp_ok, "at least 99% parsed", True),
        ("Unit detection", unit_ok, "all reported unit families detected", True),
        ("Critical FSM-input residual spikes", critical_residual_count == 0,
         "0 in depth, ROP and pressure inputs", True),
        ("Transition-sensitive residual excursions", True,
         "reported for review; not automatically removed", False),
        ("Final logical consistency", logical_count == 0, "0 violations on valid rows", True),
        ("FSM valid-row coverage", valid_pct >= config.minimum_fsm_valid_pct,
         f">= {config.minimum_fsm_valid_pct:.1f}%", True),
    ]
    rows = []
    for name, ok, criterion, blocking in checks:
        if name == "FSM valid-row coverage":
            value = valid_pct
        elif name == "Critical FSM-input residual spikes":
            value = critical_residual_count
        elif name == "Transition-sensitive residual excursions":
            value = review_residual_count
        elif "logical" in name.lower():
            value = logical_count
        else:
            value = bool(ok)
        status = "PASS" if ok else "FAIL"
        if name == "Transition-sensitive residual excursions":
            status = "REVIEW" if review_residual_count else "PASS"
        rows.append({"check": name, "value": value, "criterion": criterion,
                     "status": status, "blocking": blocking})
    ready = all(r[1] for r in checks if r[3])
    rows.append({"check": "OVERALL STAGE 2 HANDOFF", "value": ready,
                 "criterion": "all blocking checks pass",
                 "status": "READY FOR FSM" if ready else "NOT READY FOR FSM", "blocking": True})
    return pd.DataFrame(rows)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        if (not is_continuous_numeric(df[c])
                or c.endswith("_OUTLIER_FLAG")):
            continue
        flag_col = f"{c}_OUTLIER_FLAG"
        numeric = safe_numeric(df, c)
        valid = numeric.dropna()
        rows.append({
            "column": c,
            "missing_pct": numeric.isna().mean() * 100,
            "outlier_flagged_pct": df[flag_col].mean() * 100 if flag_col in df.columns else 0.0,
            "mean": float(valid.mean()) if not valid.empty else np.nan,
            "median": float(valid.median()) if not valid.empty else np.nan,
            "std": float(valid.std()) if len(valid) > 1 else np.nan,
            "minimum": float(valid.min()) if not valid.empty else np.nan,
            "maximum": float(valid.max()) if not valid.empty else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# HTML visualization
# ============================================================

def report_to_html(df: pd.DataFrame) -> str:
    """Readable, scroll-contained HTML table with compact numeric formatting."""
    shown = df.copy()
    numeric_cols = shown.select_dtypes(include=[np.number]).columns
    shown[numeric_cols] = shown[numeric_cols].round(4)
    shown = shown.rename(columns={c: c.replace("_", " ") for c in shown.columns})
    return f"<div class='table-wrap'>{shown.to_html(index=False, border=0)}</div>"

def make_comparison_html(
    df: pd.DataFrame,
    units: Dict[str, str],
    base_comparison: pd.DataFrame,
    base_selection: pd.DataFrame,
    mse_comparison: pd.DataFrame,
    mse_selection: pd.DataFrame,
    output_path: Path,
    max_plot_points: int,
    preprocessing_reports: Optional[Dict[str, pd.DataFrame]] = None,
):
    if go is None or make_subplots is None:
        raise ImportError(
            "HTML visualization requires Plotly. Install it with: pip install plotly"
        )
    plot_df = downsample(df, max_plot_points)
    panels = ["WOB", "ROP", "DIFF_PRESSURE", "MSE_TEALE", "MSE_HYDRAULIC"]
    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.060,
        subplot_titles=[f"{p}: source comparison" for p in panels],
    )
    unit_label = {
        "WOB": units["load"], "ROP": units["rop"], "DIFF_PRESSURE": units["pressure"],
        "MSE_TEALE": units["mse"], "MSE_HYDRAULIC": units["mse"],
    }

    for row_idx, parameter in enumerate(panels, start=1):
        traces: List[Tuple[Optional[str], str, float]] = []
        if parameter in BASE_DERIVED:
            spec = BASE_DERIVED[parameter]
            traces = [
                (first_existing(plot_df, spec["original_candidates"]), "solid", 1.1),
                (first_existing(plot_df, spec["calculated_candidates"]), "dot", 1.1),
                (spec["selected"] if spec["selected"] in plot_df.columns else None, "solid", 2.4),
            ]
        else:
            spec = MSE_PARAMETERS[parameter]
            traces = [
                (first_existing(plot_df, spec["original_candidates"]), "solid", 1.1),
                (first_existing(plot_df, spec["stage1_candidates"]), "dot", 1.1),
                (spec["selected_recalculated"] if spec["selected_recalculated"] in plot_df.columns else None, "dash", 1.5),
                (spec["final"] if spec["final"] in plot_df.columns else None, "solid", 2.4),
            ]

        added = 0
        seen = set()
        for col, dash, width in traces:
            if not col or col in seen or col not in plot_df.columns or not safe_numeric(plot_df, col).notna().any():
                continue
            seen.add(col)
            added += 1
            fig.add_trace(go.Scattergl(
                x=plot_df["TIMESTAMP"], y=plot_df[col], mode="lines", name=col,
                line=dict(width=width, dash=dash),
                hovertemplate=f"%{{x}}<br>{col}: %{{y:.4g}}<extra></extra>",
            ), row=row_idx, col=1)

        if added == 0:
            fig.add_annotation(
                text="No usable source is available for this parameter.",
                xref=f"x{row_idx if row_idx > 1 else ''} domain",
                yref=f"y{row_idx if row_idx > 1 else ''} domain",
                x=0.5, y=0.5, showarrow=False,
            )
        fig.update_yaxes(title_text=f"{parameter} ({unit_label[parameter]})", row=row_idx, col=1)

    fig.update_xaxes(rangeslider=dict(visible=True), row=len(panels), col=1)
    fig.update_layout(
        height=390 * len(panels), hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.88)",
        ),
        margin=dict(l=90, r=45, t=190, b=80),
    )

    warnings = []
    for report in [base_comparison, mse_comparison]:
        identity_columns = [c for c in report.columns if c.endswith("exactly_identical_warning")]
        for identity_col in identity_columns:
            for _, r in report[report[identity_col] == True].iterrows():  # noqa: E712
                warnings.append(f"{r['parameter']} ({identity_col}): compared sources are effectively identical. Verify column mapping and calculation lineage.")
    warning_html = "".join(f"<li>{w}</li>" for w in warnings) or "<li>No exact-identity warning was triggered.</li>"

    quality_panels = []
    for title, key in [
        ("Preprocessing overview", "overview"),
        ("Sampling and continuity", "sort_resample_qc"),
        ("Physical-range corrections", "range_report"),
        ("Gap handling", "gap_report"),
        ("Local spike corrections", "spike_correction"),
        ("Final residual-spike assessment", "residual_spike_assessment"),
        ("FSM input preparation and exclusion gate", "fsm_input_preparation"),
        ("Logical consistency checks", "logical_report"),
        ("Processed stuck/redundant sensor assessment", "processed_redundant_stuck_sensors"),
    ]:
        report = (preprocessing_reports or {}).get(key)
        if not isinstance(report, pd.DataFrame) or report.empty:
            continue
        shown = report
        if key == "range_report" and "values_set_to_nan" in shown:
            shown = shown.loc[shown["values_set_to_nan"] > 0]
        elif key == "gap_report" and {"values_filled", "long_gap_count"}.issubset(shown.columns):
            shown = shown.loc[(shown["values_filled"] > 0) | (shown["long_gap_count"] > 0)]
        elif key in {"spike_correction", "residual_spike_assessment"} and "isolated_spikes" in shown:
            shown = shown.loc[
                (shown["isolated_spikes"] > 0)
                | (shown.get("long_excursion_rows_preserved", 0) > 0)
            ]
        elif key == "logical_report" and "violations" in shown:
            shown = shown.loc[shown["violations"] > 0]
        elif key == "processed_redundant_stuck_sensors" and "unchanged_runs_ge_300_samples" in shown:
            shown = shown.loc[shown["unchanged_runs_ge_300_samples"] > 0]
        if shown.empty:
            quality_panels.append(
                f"<div class='panel pass'><h3>{title}</h3><p>No actionable issue remains.</p></div>"
            )
        else:
            quality_panels.append(
                f"<div class='panel'><h3>{title}</h3>{report_to_html(shown)}</div>"
            )
    quality_html = "".join(quality_panels)

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Stage 2 Preprocessing Comparison</title>
<style>
*{{box-sizing:border-box}}
html,body{{max-width:100%;overflow-x:hidden}}
body{{font-family:Arial,sans-serif;margin:18px;color:#172033;background:#fff}}
.panel{{border:1px solid #d0d7de;border-radius:10px;padding:14px;margin:14px 0;background:#fbfdff;max-width:100%;overflow:hidden}}
.table-wrap{{width:100%;overflow-x:auto;padding-bottom:6px}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:12px}}
th,td{{border:1px solid #d0d7de;padding:7px 8px;text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;max-width:360px}}
th{{background:#eef4ff;position:sticky;top:0;z-index:1}}
.warning{{background:#fff8e6;border-color:#e0b84f}}
.pass{{background:#effaf2;border-color:#78b987}}
</style></head><body>
<h2>Stage 2 Preprocessing and Derived-Parameter Validation</h2>
<div class='panel'><h3>Native/display units</h3><pre>{json.dumps(units, indent=2)}</pre>
<p>Plots retain native units. Separate FSM columns use fixed internal units.</p></div>
<div class='panel warning'><h3>Lineage checks</h3><ul>{warning_html}</ul></div>
<h2>Preprocessing and final quality assessment</h2>
{quality_html}
<h2>Derived-parameter validation and selection</h2>
<div class='panel'><h3>WOB / ROP / differential-pressure selections</h3>{report_to_html(base_selection)}</div>
<div class='panel'><h3>WOB / ROP / differential-pressure comparison statistics</h3>{report_to_html(base_comparison)}</div>
<div class='panel'><h3>Final MSE selections</h3>{report_to_html(mse_selection)}</div>
<div class='panel'><h3>MSE comparison statistics</h3>{report_to_html(mse_comparison)}</div>
<div class='panel'><h3>Original/EDR vs recalculated vs selected derived parameters</h3>{fig.to_html(include_plotlyjs=True, full_html=False)}</div>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")


# ============================================================
# Main pipeline
# ============================================================

def preprocess_working_dataframe(
    working_df: pd.DataFrame,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    config = config or PreprocessConfig()
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.outlier_action not in {"flag", "replace"}:
        raise ValueError("outlier_action must be 'flag' or 'replace'.")
    if config.spike_action not in {"flag", "replace"}:
        raise ValueError("spike_action must be 'flag' or 'replace'.")
    if config.selection_mode not in {"ask", "automatic", "original", "calculated"}:
        raise ValueError("selection_mode must be ask, automatic, original, or calculated.")

    original = working_df.copy()
    df, column_mapping_report = canonicalize_core_columns(working_df)
    df, timestamp_qc = build_timestamp_safely(df)
    df, sentinel_report = replace_sentinels_and_coerce_numeric(df, config)
    df, sort_resample_qc = sort_deduplicate_resample(df, config)
    units, unit_report = detect_units(df, config)
    raw_stuck_report = redundant_stuck_sensor_report(df)

    # Phase 1: clean measured/direct and any existing Stage-1 channels before
    # constructing fallbacks. This prevents spikes from propagating into WOB,
    # ROP, differential pressure, and MSE.
    df, range_report_1 = apply_conservative_ranges(df, units)
    range_report_1.insert(0, "phase", "pre-derived direct-channel cleaning")
    df, gap_report_1 = fill_short_gaps(df, config)
    gap_report_1.insert(0, "phase", "pre-derived direct-channel cleaning")
    df, spike_report_1 = correct_isolated_spikes(
        df, config, units, "pre-derived direct-channel cleaning"
    )

    # Fallback calculations occur only after units are known and direct
    # channels have passed physical/gap/spike correction.
    df, fallback_derived_report = ensure_stage1_derived_columns(df, units)

    # Phase 2: validate newly created and pre-existing derived candidates.
    df, range_report_2 = apply_conservative_ranges(df, units)
    range_report_2.insert(0, "phase", "post-derived validation")
    df, spike_report_2 = correct_isolated_spikes(
        df, config, units, "post-derived validation"
    )
    range_report = pd.concat([range_report_1, range_report_2], ignore_index=True)
    gap_report = gap_report_1.reset_index(drop=True)
    spike_report = pd.concat([spike_report_1, spike_report_2], ignore_index=True)

    availability_report = sensor_availability(df)
    df, outlier_report = flag_or_replace_outliers(df, config)
    df, smooth_report = create_smoothed_copies(df, config)

    # Compare before selecting. This prevents selected=original from being
    # mistaken for an independent calculated comparison.
    base_comparison = compare_base_derived(df, config)
    df, base_selection = select_base_derived(df, base_comparison, config)

    # Recalculate MSE from the selected WOB/ROP/DP channels.
    df, mse_recalculation_report = recalculate_mse_from_selected(df, units, config)

    # Compare EDR MSE when present, Stage-1 MSE, and selected-input MSE.
    df, mse_comparison, mse_selection = compare_and_select_final_mse(df, config)

    # Create a final cleaned/smoothed signal layer and exclude long-gap or
    # physically inconsistent rows before fixed-unit FSM conversion.
    df, fsm_input_preparation_report = prepare_fsm_input_signals(df, units, config)

    # Correct and then validate the exact columns consumed by the FSM. Every
    # changed sample is exported for auditability.
    df, final_spike_correction_report, fsm_spike_events = final_fsm_spike_pass(
        df, units, config, replace_values=True
    )
    df, logical_correction_events = apply_final_logical_corrections(df, units, config)
    df = refresh_fsm_validity(df, units)
    df, residual_spike_report, _unused_residual_events = final_fsm_spike_pass(
        df, units, config, replace_values=False
    )
    residual_spike_total = int(residual_spike_report["isolated_spikes"].sum()) if not residual_spike_report.empty else 0
    if config.strict_quality_gate and residual_spike_total:
        raise ValueError(
            f"Final preprocessing quality gate failed: {residual_spike_total} isolated spikes remain."
        )

    # Create internal FSM unit columns only after all final selections.
    df, fsm_conversion_report = add_fsm_columns(df, units)
    logical_report = logical_checks(df, units)
    quality_report = data_quality_report(df)
    processed_stuck_report = redundant_stuck_sensor_report(df)
    contextual_stuck_report = contextual_stuck_sensor_report(df)
    rop_tail_report = rop_tail_statistics(df, units)
    handoff_report = build_handoff_report(
        df, residual_spike_report, logical_report, timestamp_qc, unit_report, config
    )
    stage2_ready = bool(handoff_report.iloc[-1]["value"])

    overview = pd.DataFrame([{
        "rows_original": len(original),
        "rows_processed": len(df),
        "columns_processed": len(df.columns),
        "remaining_nan_cells": int(df.isna().sum().sum()),
        "range_values_set_to_nan": int(range_report["values_set_to_nan"].sum()) if not range_report.empty else 0,
        "short_gap_values_filled": int(gap_report["values_filled"].sum()) if not gap_report.empty else 0,
        "exploratory_regime_aware_outlier_flags": int(outlier_report["outliers_flagged"].sum()) if not outlier_report.empty else 0,
        "isolated_spikes_corrected": int(spike_report.loc[spike_report["action"] == "replace", "isolated_spikes"].sum()) if not spike_report.empty else 0,
        "residual_isolated_spikes": residual_spike_total,
        "fsm_valid_rows_pct": float(df["FSM_INPUT_VALID"].mean() * 100) if "FSM_INPUT_VALID" in df else 0.0,
        "logical_violations": int(logical_report["violations"].sum()) if not logical_report.empty else 0,
        "stage2_ready_for_fsm": stage2_ready,
    }])

    reports: Dict[str, object] = {
        "overview": overview,
        "column_mapping": column_mapping_report,
        "fallback_derived_calculation": fallback_derived_report,
        "timestamp_qc": timestamp_qc,
        "sentinel_report": sentinel_report,
        "sort_resample_qc": sort_resample_qc,
        "unit_detection": unit_report,
        "sensor_availability": availability_report,
        "raw_redundant_stuck_sensors": raw_stuck_report,
        "range_report": range_report,
        "gap_report": gap_report,
        "spike_correction": spike_report,
        "fsm_input_spike_correction": final_spike_correction_report,
        "fsm_input_spike_events": fsm_spike_events,
        "residual_spike_assessment": residual_spike_report,
        "outlier_report": outlier_report,
        "smooth_report": smooth_report,
        "base_derived_comparison": base_comparison,
        "base_derived_selection": base_selection,
        "mse_recalculation": mse_recalculation_report,
        "mse_comparison": mse_comparison,
        "mse_selection": mse_selection,
        "fsm_input_preparation": fsm_input_preparation_report,
        "fsm_conversion": fsm_conversion_report,
        "logical_report": logical_report,
        "logical_correction_events": logical_correction_events,
        "rop_tail_statistics": rop_tail_report,
        "contextual_stuck_sensors": contextual_stuck_report,
        "handoff_readiness": handoff_report,
        "processed_redundant_stuck_sensors": processed_stuck_report,
        "data_quality_score": quality_report,
        "detected_units": units,
    }

    processed_path = config.output_dir / "stage2_processed_fsm_ready.csv"
    df.to_csv(processed_path, index=False)
    reports["processed_csv_path"] = processed_path

    for name, report in reports.items():
        if isinstance(report, pd.DataFrame):
            report.to_csv(config.output_dir / f"stage2_{name}.csv", index=False)
    (config.output_dir / "stage2_detected_units.json").write_text(json.dumps(units, indent=2), encoding="utf-8")

    if config.make_html:
        html_path = config.output_dir / "stage2_original_calculated_selected_comparison_V3.html"
        make_comparison_html(
            df, units, base_comparison, base_selection,
            mse_comparison, mse_selection, html_path, config.max_plot_points,
            preprocessing_reports={
                "overview": overview,
                "sort_resample_qc": sort_resample_qc,
                "range_report": range_report,
                "gap_report": gap_report,
                "spike_correction": spike_report,
                "fsm_input_spike_correction": final_spike_correction_report,
                "residual_spike_assessment": residual_spike_report,
                "fsm_input_preparation": fsm_input_preparation_report,
                "logical_report": logical_report,
                "rop_tail_statistics": rop_tail_report,
                "contextual_stuck_sensors": contextual_stuck_report,
                "handoff_readiness": handoff_report,
                "processed_redundant_stuck_sensors": processed_stuck_report,
            },
        )
        reports["html_path"] = html_path

    return df, reports


# ============================================================
# RUN STAGE 2
# ============================================================
# This section expects `working_df` to already exist from Stage 1.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 EDR/FSM preprocessing")
    parser.add_argument("--input", type=Path, help="Stage-1 CSV or Parquet input")
    parser.add_argument("--output-dir", type=Path, default=Path("fsm_stage2_outputs"))
    parser.add_argument("--selection-mode", choices=["ask", "automatic", "original", "calculated"], default="ask")
    parser.add_argument("--no-html", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.input:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file does not exist: {args.input}")
        suffix = args.input.suffix.lower()
        if suffix == ".csv":
            stage2_input_df = pd.read_csv(args.input, low_memory=False)
        elif suffix in {".parquet", ".pq"}:
            stage2_input_df = pd.read_parquet(args.input)
        else:
            raise ValueError("--input must be a .csv, .parquet, or .pq file")
        stage2_input_name = str(args.input)
    else:
        stage2_input_df, stage2_input_name = resolve_stage2_input(globals())
    print(f"Stage 2 input dataframe: {stage2_input_name} ({len(stage2_input_df):,} rows, {len(stage2_input_df.columns)} columns)")

    config = PreprocessConfig(
        output_dir=args.output_dir,
        resample_rule="auto",
        selection_mode=args.selection_mode,
        outlier_action="flag",
        spike_action="replace",
        make_html=not args.no_html,
        unit_overrides={
            "depth": "auto", "rop": "auto", "pressure": "auto",
            "flow": "auto", "load": "auto", "torque": "auto", "mse": "auto",
        },
    )

    processed_df, reports = preprocess_working_dataframe(stage2_input_df, config)

    print("=" * 72)
    print("STAGE 2 PREPROCESSING SUMMARY")
    print("=" * 72)
    for key in [
        "overview", "column_mapping", "fallback_derived_calculation", "timestamp_qc",
        "sort_resample_qc", "unit_detection", "sensor_availability",
        "raw_redundant_stuck_sensors", "range_report", "gap_report",
        "spike_correction", "residual_spike_assessment",
        "fsm_input_spike_correction", "fsm_input_spike_events",
        "outlier_report", "smooth_report",
        "base_derived_comparison", "base_derived_selection",
        "mse_recalculation", "mse_comparison", "mse_selection",
        "fsm_input_preparation", "fsm_conversion", "logical_report",
        "logical_correction_events", "rop_tail_statistics",
        "contextual_stuck_sensors", "handoff_readiness",
        "processed_redundant_stuck_sensors", "data_quality_score",
    ]:
        print(f"\n{key.upper()}")
        display(reports[key])

    print("\nProcessed CSV:", reports["processed_csv_path"])
    if "html_path" in reports:
        print("Comparison HTML:", reports["html_path"])
    print("\nFINAL DECISION:", reports["handoff_readiness"].iloc[-1]["status"])

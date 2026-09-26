# Development of an Open-Source Finite-State Machine Model for Automated Classification of Drilling Rig Operational States Using Real-Time Surface Drilling Data

An open-source, auditable deterministic finite-state-machine workflow developed for the automated classification of drilling-rig operational states using real-time surface drilling data. The project includes preprocessing and quality control, fixed-unit FSM inputs, connection and stand detection, bit-run analysis, classical and hydraulic MSE, and a linked Plotly dashboard.

## Status

This is a **research release candidate (v0.2.0rc1)**. The reference regression dataset passes the implemented semantic checks. Cross-well classification accuracy must not be claimed until additional wells are processed and compared with independent DDR or manually reviewed labels.

## States

The engine distinguishes drilling, circulating/rotating, connection, stationary, trip, run/pull with pump or rotation, reaming/backreaming, data-gap, and unknown behavior. Exact output names are reported in `fsm_state_counts_corrected.csv`.

## Install

```bash
git clone https://github.com/fidanibrahimova24/finite-state-machine-based-rig-state-detection.git
cd finite-state-machine-based-rig-state-detection
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e ".[test,parquet]"
pytest -q
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run a new well

The workflow is the same for Utah FORGE and other fields. Place each raw EDR export under `data/input/` and keep its results in a separate well-named output directory:

```text
contextual-rig-state-fsm/
├── data/
│   └── input/
│       └── WELL_NAME_raw.csv
└── outputs/
    └── WELL_NAME/
```

### 1. Map channels, verify units and supply the bit-size schedule

If the raw export already contains a reliable bit-diameter channel, run:

```bash
FSM_STAGE1_INPUT="data/input/WELL_NAME_raw.csv" \
python3 -m rig_state_fsm.engines.stage1
```

If it does not, supply the verified well program as JSON. For example:

```bash
export FSM_STAGE1_INPUT="data/input/WELL_NAME_raw.csv"
export FSM_BIT_SECTIONS_JSON='[
  {"top_depth_ft": 0, "base_depth_ft": 1000, "bit_diameter_in": 17.5},
  {"top_depth_ft": 1000, "base_depth_ft": 5000, "bit_diameter_in": 12.25}
]'
python3 -m rig_state_fsm.engines.stage1
```

The values above are examples only. Replace them with the actual well program. For a verified single-size interval, `FSM_DEFAULT_BIT_DIAMETER_IN` may be set explicitly instead.

Stage 1 is intentionally interactive because channel selection, native units and bit sizes must be verified for each well. Its main output is:

```text
fsm_stage1_outputs/stage1_original_and_calculated_parameters.csv
```

### 2. Preprocess and validate the FSM inputs

```bash
rig-state-fsm preprocess \
  "fsm_stage1_outputs/stage1_original_and_calculated_parameters.csv" \
  --selection-mode automatic \
  --output-dir "outputs/WELL_NAME/stage2"
```

Before classification, confirm that `outputs/WELL_NAME/stage2/stage2_handoff_readiness.csv` reports `READY FOR FSM`.

### 3. Classify states and generate the dashboard

```bash
rig-state-fsm classify \
  "outputs/WELL_NAME/stage2/stage2_processed_fsm_ready.csv" \
  --well-name "WELL_NAME" \
  --output-dir "outputs/WELL_NAME/fsm"
```

Open the main interactive result:

```text
outputs/WELL_NAME/fsm/fsm_UNIFIED_linked_operational_bitrun_dashboard.html
```

On macOS, it can be opened from Terminal with:

```bash
open "outputs/WELL_NAME/fsm/fsm_UNIFIED_linked_operational_bitrun_dashboard.html"
```

The dashboard contains linked depth and parameter plots, rig-state intervals, connection markers, stand and bit-run summaries, state distributions, MSE views and high-resolution image export.

Optional hole-section and lithology metadata can be supplied as JSON through `FSM_HOLE_SECTIONS_JSON` and `FSM_LITHOLOGY_JSON`. They affect dashboard annotations only and never change FSM classification.

### Split CSV exports

Combine date- or interval-split files before Stage 1. The combiner sorts by parsed timestamp and removes duplicate timestamps:

```bash
rig-state-fsm combine data/input/WELL_NAME_parts.zip \
  --timestamp "Timestamp" \
  --output data/input/WELL_NAME_combined.csv
```

Use the exact timestamp column name present in the source files.

## Data contract

Stage 2 accepts the Stage-1 table and recognizes canonical names plus common Pason aliases. Stage 3 consumes fixed-unit columns such as:

- `BIT_DEPTH_FSM_FT`, `HOLE_DEPTH_FSM_FT`
- `HOOK_LOAD_FSM_INPUT`
- `PUMP_OUTPUT_FSM_GPM`, `PUMP_PRESSURE_FSM_PSI`
- `ROTARY_RPM_FSM_INPUT`, `ROTARY_TORQUE_FSM_KFT_LBF`
- `WOB_FSM_KLBF`, `ROP_FSM_FT_HR`
- `FSM_INPUT_VALID`

Never infer a well's bit-size schedule from another well. Supply and verify it during Stage 1 because it affects MSE.

## Reproducibility and configuration

- Keep one output directory per well.
- Record the exact input hash, software version/commit, channel mapping, units and bit program.
- Treat the example configuration as a template, not a universal well program.
- The package contains no raw well data or field-specific default bit schedule.

## Scientific safeguards

- MSE is interpretive and never controls state classification.
- Invalid Stage-2 intervals become `Data Gap`; missing inputs are not silently set to zero.
- Short state episodes survive dashboard downsampling.
- Detected connections are preserved as state intervals and dashboard markers/windows.
- Inferred stand bins are separated from connection-validated stands.
- Global IQR flags are exploratory; zero-inflated multimodal channels are not rejected solely by IQR.
- Thresholds are expressed in canonical internal US units after Stage-2 conversion.

## Cross-well validation protocol

For each new well:

1. Record native units, sampling interval, sensor names, and bit-size schedule.
2. Run Stage 1 and retain the mapping/calculation reports.
3. Require the Stage-2 handoff report to say `READY FOR FSM`.
4. Run Stage 3 without retuning thresholds initially.
5. Check semantic contradictions, unknown/data-gap percentage, connection timing, stand continuity, and bit runs.
6. Compare interval labels to DDR/manual ground truth. Report per-state precision, recall, F1, confusion matrix, and duration-weighted accuracy.
7. If thresholds change, justify the change physically and rerun all previously validated wells to detect regression.

Robustness/QC without labels is not classification accuracy.

## Outputs

Key files include `stage2_handoff_readiness.csv`, `fsm_qc_corrected.csv`, `fsm_state_intervals_corrected.csv`, `fsm_connection_candidates_corrected.csv`, `fsm_stand_by_stand_summary.csv`, `fsm_detected_bit_runs.csv`, and `fsm_UNIFIED_linked_operational_bitrun_dashboard.html`.

Run the semantic checks on a labeled result with:

```bash
rig-state-fsm validate outputs/WELL_NAME/fsm/fsm_labeled_compact_corrected.csv
```

## Privacy and data

Raw well data are intentionally excluded from version control. Publish only data you are authorized to redistribute. Small synthetic fixtures may be committed for tests.

## License and citation

Code is MIT licensed. See `CITATION.cff`. Confirm coauthor, university, consortium, and data-provider citation requirements before the public release.

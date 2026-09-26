from pathlib import Path
import pandas as pd
from rig_state_fsm.io import combine_csvs

def test_combine_sorts_and_deduplicates(tmp_path: Path):
    pd.DataFrame({"ts": ["2024-01-01 00:00:01", "2024-01-01 00:00:02"], "x": [1, 2]}).to_csv(tmp_path/"a.csv", index=False)
    pd.DataFrame({"ts": ["2024-01-01 00:00:02", "2024-01-01 00:00:03"], "x": [20, 3]}).to_csv(tmp_path/"b.csv", index=False)
    report = combine_csvs([tmp_path/"b.csv", tmp_path/"a.csv"], tmp_path/"out.csv", "ts")
    out = pd.read_csv(tmp_path/"out.csv")
    assert out["x"].tolist() == [1, 2, 3]
    assert report["duplicate_timestamps_removed"] == 1


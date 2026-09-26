from __future__ import annotations
import pandas as pd

MOVEMENT = {"Trip In", "Trip Out", "Trip-In", "Trip-Out"}
REAM = {"Reaming", "Backreaming"}


def semantic_checks(df: pd.DataFrame) -> dict[str, int]:
    state = df["RigState"].astype(str)
    pump = df["PUMP_ON"].astype(bool)
    rpm = df["RPM_ON"].astype(bool)
    return {
        "trip_with_pump": int((state.isin(MOVEMENT) & pump).sum()),
        "trip_with_rpm": int((state.isin(MOVEMENT) & rpm).sum()),
        "ream_without_pump": int((state.isin(REAM) & ~pump).sum()),
        "ream_without_rpm": int((state.isin(REAM) & ~rpm).sum()),
    }

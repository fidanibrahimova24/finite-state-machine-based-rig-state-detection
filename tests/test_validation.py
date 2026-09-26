import pandas as pd
from rig_state_fsm.validation import semantic_checks

def test_clean_state_semantics():
    df = pd.DataFrame({"RigState": ["Trip In", "Reaming", "Rotary Drilling"],
                       "PUMP_ON": [False, True, True], "RPM_ON": [False, True, True]})
    assert semantic_checks(df) == {"trip_with_pump": 0, "trip_with_rpm": 0,
                                   "ream_without_pump": 0, "ream_without_rpm": 0}

def test_detects_contradiction():
    df = pd.DataFrame({"RigState": ["Trip Out"], "PUMP_ON": [True], "RPM_ON": [False]})
    assert semantic_checks(df)["trip_with_pump"] == 1


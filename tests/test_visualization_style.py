from pathlib import Path


def test_stage3_dashboard_uses_readable_visualization_defaults():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "rig_state_fsm"
        / "engines"
        / "stage3.py"
    ).read_text(encoding="utf-8")

    assert "VISUALIZATION_LINE_WIDTH = 4.0" in source
    assert "VISUALIZATION_STATE_LINE_WIDTH = 5.0" in source
    assert "VISUALIZATION_AXIS_FONT_SIZE = 18" in source
    assert "toImageButtonOptions:{format:'png',scale:3}" in source
    assert "line:{width:4}" in source
    assert "marker:{size:st==='Connection'?14:10" in source
    assert "text:'<b>Con.</b>'" in source
    assert "yaxis15:{overlaying:'y14',side:'right'" in source
    assert "matches:'y14'" in source
    assert "height:2500" in source
    assert "FSM_OUTPUT_DIR" in source


def test_stage3_has_no_field_specific_section_defaults():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "rig_state_fsm"
        / "engines"
        / "stage3.py"
    ).read_text(encoding="utf-8")

    assert '"FSM_HOLE_SECTIONS_JSON"' in source
    assert '"FSM_LITHOLOGY_JSON"' in source
    assert '"hole_size":"26 in"' not in source


def test_stage1_has_no_personal_or_well_specific_fallback():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "rig_state_fsm"
        / "engines"
        / "stage1.py"
    ).read_text(encoding="utf-8")

    assert "/Users/" not in source
    assert "FSM_BIT_SECTIONS_JSON" in source
    assert "78B-32" not in source

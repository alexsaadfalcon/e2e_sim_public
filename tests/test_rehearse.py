"""Hands-off tests for webapp.rehearse's summary.json merge behaviour (no browser, no
torch): a `--only` partial re-render must not discard the rest of a previous full run's
entries (regression for the defect where a 7-entry summary was replaced by a 1-entry
file after `--only <preset>`)."""
from __future__ import annotations

from webapp.rehearse import _merge_summary

ORDER = [
    "thrust1_circuit_knobs",
    "thrust2_feature_reduction_error",
    "thrust3_cold_start_acquisition",
    "thrust4_interconnect_range_profile",
    "thrust5_detector_cfar",
    "thrust5_detector_ml",
    "thrust5_detector_raddetnet",
]


def _full_summary():
    summary = {pid: {"label": pid, "wall_s": 1.0, "rendered_at": "2026-09-24T00:00:00+00:00"}
               for pid in ORDER}
    summary["cancel_journey"] = {"preset": ORDER[0], "wall_s": 2.0,
                                  "rendered_at": "2026-09-24T00:00:00+00:00"}
    return summary


def test_partial_rerun_keeps_the_rest_of_a_full_run():
    existing = _full_summary()
    new = {"thrust5_detector_raddetnet": {"label": "thrust5_detector_raddetnet",
                                           "wall_s": 9.0,
                                           "rendered_at": "2026-09-24T01:00:00+00:00"}}

    merged = _merge_summary(existing, new, ORDER)

    assert list(merged.keys()) == ORDER + ["cancel_journey"]
    # the re-run preset is replaced ...
    assert merged["thrust5_detector_raddetnet"]["wall_s"] == 9.0
    assert merged["thrust5_detector_raddetnet"]["rendered_at"] == "2026-09-24T01:00:00+00:00"
    # ... every other entry, including cancel_journey (not re-run), is untouched
    for pid in ORDER:
        if pid == "thrust5_detector_raddetnet":
            continue
        assert merged[pid] == existing[pid]
    assert merged["cancel_journey"] == existing["cancel_journey"]


def test_partial_rerun_including_cancel_journey_replaces_it():
    existing = _full_summary()
    new = {
        "thrust5_detector_cfar": {"label": "thrust5_detector_cfar", "wall_s": 5.0,
                                   "rendered_at": "2026-09-24T01:00:00+00:00"},
        "cancel_journey": {"preset": "thrust5_detector_cfar", "wall_s": 3.0,
                            "rendered_at": "2026-09-24T01:00:00+00:00"},
    }

    merged = _merge_summary(existing, new, ORDER)

    assert list(merged.keys()) == ORDER + ["cancel_journey"]
    assert merged["thrust5_detector_cfar"] == new["thrust5_detector_cfar"]
    assert merged["cancel_journey"] == new["cancel_journey"]


def test_empty_existing_summary_is_just_the_new_run():
    new = {"thrust1_circuit_knobs": {"label": "thrust1_circuit_knobs", "wall_s": 1.5,
                                      "rendered_at": "2026-09-24T01:00:00+00:00"}}

    merged = _merge_summary({}, new, ORDER)

    assert merged == new


def test_order_follows_preset_order_not_insertion_order():
    # existing was written with keys out of registry order (e.g. hand-edited or from a
    # differently-ordered PRESETS list at generation time); the merge must reorder.
    existing = {ORDER[2]: {"wall_s": 1.0}, ORDER[0]: {"wall_s": 2.0}}
    new = {ORDER[1]: {"wall_s": 3.0}}

    merged = _merge_summary(existing, new, ORDER)

    assert list(merged.keys()) == [ORDER[0], ORDER[1], ORDER[2]]

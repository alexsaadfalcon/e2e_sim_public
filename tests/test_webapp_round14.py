"""Hostile round 14 (2026-09-25): the narration fixes, pinned on REAL runs of the presets.

Every test here drives `webapp.app._run_pipeline` -- the same callback the Run button
calls, both arms -- and, where the screen's words are rewritten at render time, the
same `_render_results` the Results tab calls. The runs use the presets' own frames
(munich Ka / the Ka demo corpus) and skip, by name, on a machine that does not have them.
"""

import copy

import pytest

torch = pytest.importorskip("torch")

from webapp.demo_presets import PRESETS_BY_ID, apply_preset          # noqa: E402


_RUNS = {}


def _run_ab(pid, n_steps=None):
    """Both arms of preset `pid`, through the app's own Run callback. Returns a deep
    copy of the results-store payload (arm A top-level, arm B under "_previous").
    Cached per preset for this module: several tests read one run, and a run is the
    expensive part."""
    key = (pid, n_steps)
    if key not in _RUNS:
        _RUNS[key] = _run_ab_uncached(pid, n_steps)
    return copy.deepcopy(_RUNS[key])


def _run_ab_uncached(pid, n_steps=None):
    from webapp import app as appmod

    preset = PRESETS_BY_ID[pid]
    try:
        data, status, *_ = appmod._run_pipeline(
            1, apply_preset(preset), n_steps or preset.n_steps, "", None)
    except FileNotFoundError as e:           # pragma: no cover - machine-dependent
        pytest.skip(f"{pid}: frames not on this machine ({e})")
    if "_previous" not in data:              # pragma: no cover - machine-dependent
        pytest.skip(f"{pid}: the A/B run did not complete here ({status})")
    return data


def _rendered(data):
    """The Results tab for `data`, after every render-time pass (sharing, the fold
    clause, arm styling). On a deep copy: the render mutates the store it is given."""
    from webapp import app as appmod

    data = copy.deepcopy(data)
    tree = appmod._render_results(data, "tab-results")
    return data, tree


def _all_text(component) -> str:
    if isinstance(component, str):
        return component
    parts = []
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        parts.extend(_all_text(c) for c in children if c is not None)
    elif children is not None:
        parts.append(_all_text(children))
    return " ".join(parts)


def _panel(fig):
    return ((fig.get("layout") or {}).get("meta") or {}).get("panel") or {}


def _strip_text(frame_or_layout):
    """The statistic strip's text in one layout (base or a frame's override)."""
    anns = (frame_or_layout or {}).get("annotations") or []
    for a in anns:
        if str(a.get("name") or "").startswith("stat_strip"):
            return a["text"]
    return ""


# ---------------------------------------------------------------------------------------
# J1 / J5 -- one range crop for the Thrust 5 column, and the clause that says why arm B
# shows anything above it
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("pid", ["thrust5_detector_cfar", "thrust5_detector_raddetnet"])
def test_thrust5_range_doppler_and_objectness_share_the_crop_and_the_scoring_line(pid):
    from webapp import detector_scoreboard

    crop = detector_scoreboard.scoring_max_range_m()
    if crop is None:                          # pragma: no cover - machine-dependent
        pytest.skip("beat_cfar.json not on this machine")
    data = _run_ab(pid)
    det_key = next(k for k in data if k.endswith("_detection"))
    for arm in (data, data["_previous"]):
        rd, det = arm["radar_cube"], arm[det_key]
        assert list(rd["layout"]["yaxis"]["range"]) == list(det["layout"]["yaxis"]["range"])
        assert rd["layout"]["yaxis"]["range"][1] == pytest.approx(50.0)
        for fig in (rd, det):
            ys = [s.get("y0") for s in (fig["layout"].get("shapes") or [])]
            assert crop in ys, (pid, ys)


def test_the_quantisation_clause_is_on_the_3_bit_arm_only_and_is_measured():
    """The clause is an OBSERVATION: it fires only when cells of the drawn 40-50 m band
    clear the clip on this run, and it names the bit depth only when the run's depth is
    below the depth the frames were written at. At 12 bits (arm A, the frames' own
    depth) nothing up there clears the clip, so arm A says the scene is dark."""
    data = _run_ab("thrust5_detector_cfar")
    cap_a = " ".join(_panel(data["radar_cube"])["caption"])
    cap_b = " ".join(_panel(data["_previous"]["radar_cube"])["caption"])
    assert "quantisation" not in cap_a and "dark on purpose" in cap_a
    assert "3-bit quantisation floor, unscored" in cap_b
    assert "above 40 m" in cap_b
    details_b = " ".join(_panel(data["_previous"]["radar_cube"])["details"])
    assert "vanish" not in details_b or "12 bits" in details_b
    assert "12 bits the frames were written at" in details_b


def test_the_if_corner_arm_never_borrows_the_quantisation_mechanism():
    """thrust5_detector_ml turns the IF corner at 12 bits: whatever clears the clip
    above the crop there is NOT a 3-bit product, and the caption must not say it is."""
    data = _run_ab("thrust5_detector_ml")
    for arm in (data, data["_previous"]):
        panel = _panel(arm["radar_cube"])
        assert "quantisation" not in " ".join(panel["caption"])
        assert "12 bits the frames were written at" not in " ".join(panel["details"])


# ---------------------------------------------------------------------------------------
# J4 / K2 / J3 -- each arm's one-line caption is that arm's own facts
# ---------------------------------------------------------------------------------------
def test_thrust5_arm_captions_carry_the_gate_verdict_and_the_placement():
    from webapp.app import _arm_caption

    data = _run_ab("thrust5_detector_cfar")
    cap_a, cap_b = _arm_caption(data), _arm_caption(data["_previous"])
    assert "of 4096 LSB vs stored: bit-identical" in cap_a
    assert "of 8 LSB vs stored: differs (this run's ADC 3-bit)" in cap_b
    for cap in (cap_a, cap_b):
        assert "front end on ifft(CFR)" in cap, cap
        assert len(cap) <= 86, cap


@pytest.mark.parametrize("pid", ["thrust1_circuit_knobs", "thrust2_feature_reduction_error",
                                 "thrust3_cold_start_acquisition"])
def test_munich_arm_captions_state_the_arm_not_the_preset_wide_drive(pid):
    from webapp.app import _arm_caption

    preset = PRESETS_BY_ID[pid]
    data = _run_ab(pid)
    caps = [_arm_caption(data), _arm_caption(data["_previous"])]
    for cap, label in zip(caps, (preset.ab_label_a, preset.ab_label_b)):
        assert "Front-end drive" not in cap, cap
        assert label in cap, (label, cap)
        assert "frame " in cap and " of " in cap, cap
        assert len(cap) <= 86, cap
    # Parallel: the placement is on both arms or on neither.
    assert ("front end on" in caps[0]) == ("front end on" in caps[1]), caps
    if pid == "thrust1_circuit_knobs":
        assert all("front end on the beat record" in c for c in caps), caps
    # ...and the drive caveat is on the page foot, ONCE.
    _, tree = _rendered(data)
    text = _all_text(tree)
    assert text.count("is a DISPLAY choice, not a measured input level (both arms") == 1


# ---------------------------------------------------------------------------------------
# J2 -- the strip's bold number names its scope
# ---------------------------------------------------------------------------------------
def test_detector_strip_says_this_frame_in_the_bold():
    data = _run_ab("thrust5_detector_ml")
    fig = data["ml_detection"]
    for fr in fig.get("frames") or []:
        assert _strip_text(fr.get("layout")).startswith("<b>this frame: ")


# ---------------------------------------------------------------------------------------
# K1 / K4 -- Details names its frame, in the transport's words, and agrees with the strip
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("pid", ["thrust1_circuit_knobs", "thrust6_jsac_resource_split"])
def test_details_statistics_name_the_same_frame_the_strip_does(pid):
    data = _run_ab(pid)
    for arm in (data, data["_previous"]):
        fig = arm["range_az"]
        n = len(fig.get("frames") or [])
        assert n >= 2
        last_strip = _strip_text(fig["frames"][-1]["layout"])
        tag = f"frame {n} of {n}"
        assert tag in last_strip, last_strip
        details = " ".join(_panel(fig)["details"])
        assert f"({tag}, the last)" in details
        # The brightest return: the strip's number on the last frame IS Details' number.
        strip_b = last_strip.split("brightest ", 1)[1].split(" · ", 1)[0]
        db, metres = strip_b.split(" @ ")
        assert f"brightest visible return: {db} at {metres}" in details, (strip_b, details)
        assert f"({tag}; the strip above" in details


def test_thrust6_frame_axes_count_from_one_like_the_transport():
    data = _run_ab("thrust6_jsac_resource_split")
    for arm in (data, data["_previous"]):
        xs = arm["evm"]["data"][0]["x"]
        xs = list(xs) if not isinstance(xs, dict) else None
        if xs is not None:
            assert xs[0] == 1 and xs[-1] == len(xs)
        n = len(arm["range_az"].get("frames") or [])
        assert f"frame {n} of {n}" in _strip_text(arm["evm"]["layout"])


# ---------------------------------------------------------------------------------------
# K3 -- the "shown" window in Details is THIS arm's
# ---------------------------------------------------------------------------------------
def test_thrust6_details_state_each_arms_own_shown_window():
    data = _run_ab("thrust6_jsac_resource_split")
    det_a = " ".join(_panel(data["range_az"])["details"])
    det_b = " ".join(_panel(data["_previous"]["range_az"])["details"])
    assert "0-249.8 m shown of a 499.6 m window" in det_a
    assert "0-249.8 m shown of a" not in det_b
    assert "0-62.4 m shown (this arm's own sensing window)" in det_b


# ---------------------------------------------------------------------------------------
# M2 / L1 -- the fold and the constellation, on the visible captions
# ---------------------------------------------------------------------------------------
def test_thrust6_fold_is_on_arm_bs_caption_and_matches_the_runbook():
    rendered, _ = _rendered(_run_ab("thrust6_jsac_resource_split"))
    cap_b = " ".join(_panel(rendered["_previous"]["range_az"])["caption"])
    cap_a = " ".join(_panel(rendered["range_az"])["caption"])
    assert " folds to " not in cap_a
    fold = next(c for c in _panel(rendered["_previous"]["range_az"])["caption"]
                if " folds to " in c)
    other_m, rest = fold.split(" m folds to ")
    this_m = rest.split(" m ", 1)[0]
    # The card's sentence quotes the SAME numbers, measured on the last frame.
    say = " ".join(PRESETS_BY_ID["thrust6_jsac_resource_split"].say)
    assert f"arm A's {other_m} m return folds to {this_m} m" in say, (fold, say)
    assert len(" · ".join(_panel(rendered["_previous"]["range_az"])["caption"])) <= 94, cap_b


def test_thrust6_constellation_caption_is_measured():
    data = _run_ab("thrust6_jsac_resource_split")
    for arm in (data, data["_previous"]):
        panel = _panel(arm["comm_const"])
        cap = " · ".join(panel["caption"])
        assert " symbols, EVM " in cap and "e-3" in cap, cap
        assert "every point inside its marker" in cap, cap
        assert "px on this panel" in " ".join(panel["details"])
        assert len(cap) <= 94, cap


# ---------------------------------------------------------------------------------------
# L2 -- the offline PR panel's subtitle is printed once
# ---------------------------------------------------------------------------------------
def test_the_offline_pr_caption_is_printed_once():
    from webapp.app import PR_PANEL_KEY

    data = _run_ab("thrust5_detector_cfar")
    if PR_PANEL_KEY not in data:              # pragma: no cover - machine-dependent
        pytest.skip("no stored PR panel on this machine")
    caption = " · ".join(_panel(data[PR_PANEL_KEY])["caption"])
    _, tree = _rendered(data)
    assert caption and _all_text(tree).count(caption) == 1


# ---------------------------------------------------------------------------------------
# Unit-level: the gate's caption and the frame tag
# ---------------------------------------------------------------------------------------
def test_gate_caption_verdicts():
    from webapp.pipeline_runner import _StoredADCGateBlock

    class _Q:
        bits = 3

    g = _StoredADCGateBlock([], quantizer_block=_Q())
    g.n_compared = 5
    g.max_lsb_diff = 0
    assert g.caption(12) == "0 of 8 LSB vs stored: bit-identical"
    g.max_lsb_diff = 1
    assert g.caption(12) == "1 of 8 LSB vs stored: differs (this run's ADC 3-bit)"
    # A difference NOT caused by the bit depth is not attributed to it.
    assert g.caption(3) == "1 of 8 LSB vs stored: differs"
    assert g.caption(None) == "1 of 8 LSB vs stored: differs"
    g.problem = "boom"
    assert "not compared" in g.caption(12)


def test_frame_tag_is_one_based_like_the_transport():
    from webapp.pipeline_runner import _frame_tag, _n_frames_phrase

    assert _frame_tag(0, 5) == "frame 1 of 5"
    assert _frame_tag(4, 5) == "frame 5 of 5"
    assert _n_frames_phrase([3, 4, 5]) == "frames 3-5"
    assert _n_frames_phrase([1]) == "frame 1"
    assert _n_frames_phrase([1, 3]) == "frames 1, 3"


def test_the_drive_foot_line_says_both_arms_only_when_both_ran():
    """Read on the 2026-09-25 cancel render: the one-arm cancelled screen printed
    "(both arms; ...)" because its payload still carries the A/B flag."""
    from webapp.app import _both_arms_ran, _drive_foot_line

    note = ["Front-end drive 3e-05 is a DISPLAY choice, not a measured input level -- x"]
    assert _both_arms_ran({"_ab": True})
    assert not _both_arms_ran({"_ab": True, "_cancelled_chip":
                               "CANCELLED -- 2 of 20 frames; arm B did not run"})
    assert _both_arms_ran({"_ab": True, "_cancelled_chip": "CANCELLED -- arm B ran 2 of 5 frames"})
    assert "both arms" in _drive_foot_line(note, two_arms=True)
    assert "both arms" not in _drive_foot_line(note, two_arms=False)
    assert _drive_foot_line(["Environment 'x'"], two_arms=True) == ""

"""`e2e.ml.cfar_head`: CFAR candidates + a learned rescoring head.

All CPU, all synthetic, seconds. The corpus-scale numbers (AP against CFAR's 0.301, the
four controls) need a GPU and the real manifest and are NOT asserted here -- what is
asserted is the mechanics a reviewer would otherwise have to take on trust:

* the output is the same `[3, R, A]` contract `decode_detections` inverts;
* stage 1 reproduces `baseline.classical_detection_map` EXACTLY, which is the claim the
  whole "compute CFAR inside forward from the rad tensor" design rests on;
* the candidate set contains every CFAR detection above the floor;
* a perfect oracle score really does reorder the candidates into ground truth first
  (i.e. the head has the authority the design says it has);
* the forward pass is deterministic;
* `train.build_model` builds it natively (the registration shim is gone, 2026-09-23).
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from e2e.ml import cfar_head as ch
from e2e.ml.baseline import classical_detection_map
from e2e.ml.dataset import derive_network_input
from e2e.ml.labels import LabelGrid, decode_detections
from e2e.ml.metrics import MatchCriterion, evaluate_dataset, match_detections
from e2e.radar_config import BENCHMARK_V1

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------------
# Fixtures -- a shrunk TDM config and one synthetic frame, on CPU
# --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg():
    """`benchmark_v1` shrunk to 32 chirps x 64 samples (the shrink pattern
    `tests/test_ml_train.py::tiny_manifest_path` uses), so every transform below runs
    in milliseconds while keeping TDM, the virtual array and the zero-Doppler notch
    (max_velocity 9.7 m/s >= NOTCH_MIN_VMAX_MPS) on the real code path."""
    return dataclasses.replace(BENCHMARK_V1, name="cfar_head_tiny",
                               n_chirps=32, n_samples=64)


@pytest.fixture(scope="module")
def grid(cfg):
    return LabelGrid.for_config(cfg)


@pytest.fixture(scope="module")
def adc(cfg):
    g = torch.Generator(device="cpu").manual_seed(7)
    shape = (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
    re = torch.randn(shape, generator=g)
    im = torch.randn(shape, generator=g)
    return (re + 1j * im).to(torch.complex64)


@pytest.fixture(scope="module")
def rad(cfg, adc):
    """The tensor `forward` is actually handed -- from the dataset's own derivation."""
    return derive_network_input(cfg, adc, "rad").to(DEVICE)


@pytest.fixture
def model(cfg, grid, rad):
    a, r, d = rad.shape
    torch.manual_seed(0)
    return ch.CFARHead(a, r, d, grid.n_range, grid.n_azimuth,
                       cfg=cfg, grid=grid).to(DEVICE).eval()


# --------------------------------------------------------------------------------
# Stage 1 -- the claim the design rests on
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["tdm_notch_on", "single_notch_off"])
def test_cfar_from_rad_matches_the_classical_arm(cfg, variant):
    """`forward` cannot call `classical_detection_map` (it has no ADC), so it recovers
    the same objectness from the log-power `rad` cube. That is only legitimate if the
    two agree -- the per-frame median reference cancels in the CFAR ratio.

    Both branches of `classical_detection_map`'s AUTO front end are covered: TDM with the
    zero-Doppler notch on (v_max 9.7 >= NOTCH_MIN_VMAX_MPS), and a single-TX config slow
    enough that the notch and the TDM compensation are both off. A parity that held on
    only one of them would be a parity that held by accident."""
    from e2e.ml.baseline import NOTCH_MIN_VMAX_MPS

    if variant == "single_notch_off":
        # n_tx=1 => mimo "single": no deinterleave, no TDM compensation; a long chirp
        # period drops v_max below the notch gate.
        cfg = dataclasses.replace(cfg, n_tx=1, mimo="single", chirp_period_s=4e-4)
        assert cfg.max_velocity_mps < NOTCH_MIN_VMAX_MPS
    else:
        assert cfg.mimo == "tdm" and cfg.max_velocity_mps >= NOTCH_MIN_VMAX_MPS

    grid = LabelGrid.for_config(cfg)
    g = torch.Generator(device="cpu").manual_seed(7)
    shape = (cfg.n_rx, cfg.n_chirps, cfg.n_samples)
    adc = (torch.randn(shape, generator=g)
           + 1j * torch.randn(shape, generator=g)).to(torch.complex64)
    rad = derive_network_input(cfg, adc, "rad")

    classical = classical_detection_map(cfg, adc, grid)[0]
    recovered = ch.cfar_from_rad(rad, cfg, grid)
    assert recovered.shape == classical.shape == (grid.n_range, grid.n_azimuth)
    # Guard against a vacuous comparison of two all-zero maps: the frame must put a
    # non-trivial number of cells above the candidate floor for the parity to mean
    # anything. MEASURED on these fixtures (2026-09-22): 133 cells for TDM (peak 0.254),
    # 19 for single-TX (peak 0.141 -- a 16-bin angle axis is less peaky than TDM's
    # 64-bin one). Threshold set below both, not at them, so a small numerical change
    # does not read as a broken fixture.
    assert int((classical >= ch.DEFAULT_CFAR_FLOOR).sum()) >= 10, \
        "degenerate fixture: nothing for CFAR to find"

    # THE CLAIM, at the level where it is exactly true: the CFAR FIELD, before
    # peak-grouping. See the next assertion for what grouping does to it.
    ungrouped_classical = classical_detection_map(cfg, adc, grid, peak_grouping=False)[0]
    ungrouped_recovered = ch.cfar_from_rad(rad, cfg, grid, peak_grouping=False)
    assert torch.allclose(ungrouped_recovered, ungrouped_classical, atol=1e-5), \
        f"max |delta| = {float((ungrouped_recovered - ungrouped_classical).abs().max()):.3g}"

    # What survives grouping. `group_peaks` keeps a cell iff `obj >= pooled`, an exact
    # comparison, and the azimuth axis is a nearest-neighbour UPSAMPLE that replicates
    # each angle-FFT bin into `n_azimuth / n_angle` identical columns. On those plateaus
    # a last-bit float difference between the two computation orders decides which
    # replica wins, so grouped maps can differ cell-for-cell without the field differing.
    # MEASURED 2026-09-22 on these fixtures:
    #   TDM, 64 angle bins -> 192 columns (replication 3): no cell flips; grouped maps
    #     agree to 4.5e-7, the same order as the ungrouped field.
    #   single, 16 -> 192 (replication 12): 24 cells flip, max |delta| 0.044.
    # In BOTH cases every flipped cell is below the candidate floor, the delta among
    # cells at or above the floor is <= 4.7e-7, and the candidate SET is identical --
    # which is the property stage 1 actually has to have. The real corpus is the
    # replication-3 case.
    above = (classical >= ch.DEFAULT_CFAR_FLOOR) | (recovered >= ch.DEFAULT_CFAR_FLOOR)
    assert torch.equal(classical >= ch.DEFAULT_CFAR_FLOOR,
                       recovered >= ch.DEFAULT_CFAR_FLOOR), "candidate sets differ"
    assert float((classical - recovered).abs()[above].max()) < 1e-5
    if cfg.mimo == "tdm":
        assert torch.allclose(recovered, classical, atol=1e-5), \
            f"max |delta| = {float((recovered - classical).abs().max()):.3g}"


def test_fixed_threshold_control_is_not_the_cfar_map(cfg, grid, rad):
    """The no-local-adaptivity rung must actually differ from CFAR, or it controls
    for nothing."""
    cfar = ch.cfar_from_rad(rad, cfg, grid)
    glob = ch.global_threshold_objectness(rad, cfg, grid)
    assert glob.shape == cfar.shape
    assert 0.0 <= float(glob.min()) and float(glob.max()) <= 1.0
    assert not torch.allclose(glob, cfar, atol=1e-3)


# --------------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------------
def test_output_shape_and_contract(model, grid, rad):
    out = model(rad[None])
    det = out["detection"]
    assert det.shape == (1, 3, grid.n_range, grid.n_azimuth)
    assert det.dtype == torch.float32
    assert torch.isfinite(det).all()
    # Channel 0 is post-sigmoid objectness in [0, 1] (decode_detections thresholds it
    # directly); channels 1-2 are the raw regression residuals, zero for this head.
    assert float(det[:, 0].min()) >= 0.0 and float(det[:, 0].max()) <= 1.0
    assert torch.equal(det[:, 1:], torch.zeros_like(det[:, 1:]))
    # Non-candidate cells are exactly zero, so the map is sparse by construction.
    rows, cols, _ = out["rows"][0], out["cols"][0], out["cfar"][0]
    nz = (det[0, 0] > 0).nonzero(as_tuple=False)
    assert nz.shape[0] == rows.numel() <= model.max_candidates
    assert set(map(tuple, nz.tolist())) == set(zip(rows.tolist(), cols.tolist()))


def test_batch_of_two_is_scored_independently(model, rad):
    out = model(torch.stack([rad, rad * 0.0]))
    assert out["detection"].shape[0] == 2
    assert len(out["logits"]) == 2
    # An all-zero dB cube is a flat field: CFAR finds no cell above its own annulus.
    assert out["rows"][1].numel() == 0
    assert float(out["detection"][1, 0].abs().max()) == 0.0
    # ...and the REAL frame's map is unchanged by what it was batched with. Without this
    # the test only exercised stage 1 on the degenerate sample and would have passed
    # under batch cross-talk in `_patches`/`_logits` (review finding, 2026-09-22).
    alone = model(rad[None])["detection"][0]
    assert torch.equal(out["detection"][0], alone)
    other = model(torch.stack([rad, rad * 0.5]))["detection"][0]
    assert torch.equal(other, alone)


def test_wrong_input_rank_and_shape_raise(model, rad):
    with pytest.raises(ValueError, match=r"\[B, A, R, D\]"):
        model(rad)
    with pytest.raises(ValueError, match="expected"):
        model(rad[None, :, :, :-1])


def test_decode_round_trips_the_candidate_cells(model, grid, rad):
    """`decode_detections` must find exactly the cells the head scored, at the geometry
    `detections_for` reports -- that is what makes this model's output scoreable by the
    shipped metric without a second decoder."""
    out = model(rad[None])
    rows, cols = out["rows"][0], out["cols"][0]
    scores = torch.sigmoid(out["logits"][0])
    dets = decode_detections(grid, out["detection"][0], threshold=0.0)
    ours = model.detections_for(rows, cols, scores.tolist())
    # decode applies its own 3x3 NMS, so it returns a SUBSET; every returned detection
    # must be one of ours, at identical range / sin_az.
    ours_geom = {(round(r, 9), round(s, 9)) for r, s, _sc, _sr in ours}
    assert dets, "the synthetic frame produced no decodable detection"
    for r, s, _sc, sr in dets:
        assert (round(r, 9), round(s, 9)) in ours_geom
        assert sr == pytest.approx(r)          # zero regression => surface == centre


# --------------------------------------------------------------------------------
# Stage 1 -> stage 2 hand-off
# --------------------------------------------------------------------------------
def test_candidates_contain_every_cfar_detection_above_the_floor(model, cfg, grid, adc, rad):
    """The rescorer can only reorder what stage 1 gives it, so nothing above the floor
    may be dropped (below K). Checked against the CLASSICAL map -- `classical_detection_map`
    on the ADC -- not against the model's own intermediate, so a shared bug cannot hide."""
    obj = classical_detection_map(cfg, adc, grid)[0]
    above = (obj >= ch.DEFAULT_CFAR_FLOOR).nonzero(as_tuple=False)
    assert len(above) > 0
    # K is the ONLY thing allowed to drop a cell above the floor, so the containment
    # claim is tested with K raised above this frame's count (the K-truncation rule is
    # tested separately, below). On pure noise this synthetic frame yields ~133 cells.
    a, r, d = rad.shape
    m = ch.CFARHead(a, r, d, grid.n_range, grid.n_azimuth, cfg=cfg, grid=grid,
                    max_candidates=4 * len(above)).eval()
    rows, cols, scores = m.candidates(rad[None])[0]
    assert set(map(tuple, above.tolist())) == set(zip(rows.tolist(), cols.tolist()))
    assert torch.all(scores >= m.cfar_floor)

    # At the shipped K, the kept set is a strict subset and it is the highest-scoring one.
    krows, kcols, kscores = model.candidates(rad[None])[0]
    assert krows.numel() == model.max_candidates < len(above)
    assert set(zip(krows.tolist(), kcols.tolist())) <= set(zip(rows.tolist(), cols.tolist()))
    assert float(kscores.min()) >= float(torch.sort(scores, descending=True)
                                         .values[model.max_candidates - 1])


def test_candidates_are_truncated_to_k_by_objectness(cfg, grid, rad):
    """With a floor low enough to admit more than K cells, the K kept must be the K
    highest-objectness ones -- and the order must be deterministic under ties."""
    a, r, d = rad.shape
    m = ch.CFARHead(a, r, d, grid.n_range, grid.n_azimuth, cfg=cfg, grid=grid,
                    cfar_floor=0.0, max_candidates=8).eval()
    obj = ch.cfar_from_rad(rad, cfg, grid)
    rows, cols, scores = m.candidates(rad[None])[0]
    assert rows.numel() == 8
    best = torch.sort(obj[obj >= 0.0].flatten(), descending=True).values[:8]
    assert torch.allclose(torch.sort(scores, descending=True).values, best)
    # descending, with a stable (row-major) tie-break
    assert torch.all(scores[:-1] >= scores[1:])


def test_candidate_recall_reports_the_ceiling(model, grid, rad):
    rows, cols, scores = model.candidates(rad[None])[0]
    dets = model.detections_for(rows, cols, scores.tolist())
    # Declare three well-separated candidates to be ground truth; the ceiling is then 1.0.
    picked = _separated(rows, cols, 3)
    targets = [(r, s, "vehicle", r) for r, s, _sc, _sr in
               [dets[i] for i in picked]]
    r = model.candidate_recall([rad], [targets])
    assert r["candidate_recall"] == pytest.approx(1.0)
    assert r["candidate_recall_pre_nms"] == pytest.approx(1.0)
    assert r["n_targets"] == 3.0
    assert r["candidates_per_frame"] == rows.numel()


def test_candidate_recall_ceiling_is_the_post_nms_one(cfg, grid):
    """The reported ceiling must be what a scored arm can actually reach.

    `decode_detections` suppresses a detection within Chebyshev distance 2 of a
    higher-scoring one (`nms_footprint=3`), a COARSER radius than stage 1's
    `group_peaks(radius=1)`. Two candidates 2 cells apart are therefore both proposed and
    only one survives scoring -- so the raw candidate count overstates the ceiling. This
    is the review's finding 1, reproduced as a regression test."""
    a, r_in, d = 8, 16, 4
    m = ch.CFARHead(a, r_in, d, grid.n_range, grid.n_azimuth, cfg=cfg, grid=grid).eval()

    rb, ab = grid.range_bin_m, grid.az_bin
    cells = [(5, 40), (5, 42)]                       # 2 apart: separate peaks, one detection
    targets = [(( i + 0.5) * rb, -1.0 + (j + 0.5) * ab, "vehicle", (i + 0.5) * rb)
               for i, j in cells]
    # Both targets are >2 m / >0.06 sin-az apart? No -- they are 2 azimuth bins apart, so
    # they are inside each other's tolerance. Use two DISTINCT targets that each only one
    # candidate can claim, by placing them exactly on the two cells.
    m.candidate_fn = lambda obj, floor, k: (
        torch.tensor([c[0] for c in cells]), torch.tensor([c[1] for c in cells]),
        torch.tensor([0.99, 0.95]))
    x = torch.zeros((a, r_in, d))
    res = m.candidate_recall([x], [targets])
    assert res["candidate_recall_pre_nms"] > res["candidate_recall"], res
    assert res["candidate_recall"] == pytest.approx(0.5)      # one survives NMS
    assert res["candidate_recall_pre_nms"] == pytest.approx(1.0)


def test_model_stage1_equals_the_standalone_function(model, cfg, grid, rad):
    """`forward`/`candidates` and the standalone `cfar_from_rad` (the function the
    classical-parity test pins) must be the same computation, not two that agree
    today (review finding, 2026-09-22)."""
    assert torch.equal(model.cfar(rad[None])[0], ch.cfar_from_rad(rad, cfg, grid))


# --------------------------------------------------------------------------------
# The head's authority: a perfect oracle must recover ground truth first
# --------------------------------------------------------------------------------
def _separated(rows, cols, n, min_sep=6):
    """Indices of `n` candidates pairwise >= `min_sep` cells apart (Chebyshev), taken
    from the END of the CFAR ordering -- i.e. the ones CFAR ranks WORST, so the oracle
    has something real to fix."""
    picked = []
    for i in reversed(range(rows.numel())):
        ri, ci = int(rows[i]), int(cols[i])
        if all(max(abs(ri - int(rows[j])), abs(ci - int(cols[j]))) >= min_sep
               for j in picked):
            picked.append(i)
        if len(picked) == n:
            break
    assert len(picked) == n, "fixture produced too few separated candidates"
    return picked


def test_oracle_head_reorders_candidates_to_recover_ground_truth_first(model, grid, rad):
    out_cfar = model.candidates(rad[None])[0]
    rows, cols, cfar_scores = out_cfar
    dets = model.detections_for(rows, cols, cfar_scores.tolist())
    picked = _separated(rows, cols, 3)
    targets = [(dets[i][0], dets[i][1], "vehicle", dets[i][3]) for i in picked]

    # Arm A: CFAR's own ordering, scattered into the same [3, R, A] map.
    map_cfar = torch.zeros((3, grid.n_range, grid.n_azimuth))
    map_cfar[0, rows, cols] = cfar_scores

    # Arm B: the same candidates, rescored by a PERFECT oracle (+10 for a cell that
    # matches a target under the metric's own criterion, -10 otherwise).
    want = set(picked)

    def oracle(patches, centre, r_, c_):
        return torch.where(
            torch.tensor([i in want for i in range(r_.numel())]),
            torch.full((r_.numel(),), 10.0), torch.full((r_.numel(),), -10.0))

    model.score_fn = oracle
    try:
        map_head = model(rad[None])["detection"][0]
    finally:
        model.score_fn = None

    kw = dict(score_threshold=0.0, criterion=MatchCriterion())
    ap_cfar = evaluate_dataset([map_cfar], [targets], grid, **kw)["AP"]
    ap_head = evaluate_dataset([map_head], [targets], grid, **kw)["AP"]

    assert ap_head == pytest.approx(1.0), ap_head
    assert ap_head > ap_cfar, (ap_head, ap_cfar)
    # ...and "first" literally: the three highest-scoring decoded detections are the
    # three targets, every one of them matched.
    top3 = decode_detections(grid, map_head, threshold=0.0)[:3]
    matches, _fp, _fn = match_detections(top3, targets, MatchCriterion())
    assert len(matches) == 3


# --------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------
def test_loss_labels_candidates_by_the_metrics_own_matcher(model, grid, rad):
    out = model(rad[None])
    rows, cols, scores = out["rows"][0], out["cols"][0], out["cfar"][0]
    dets = model.detections_for(rows, cols, scores.tolist())
    picked = _separated(rows, cols, 2)
    targets = [(dets[i][0], dets[i][1], "vehicle", dets[i][3]) for i in picked]

    total, parts = model.loss(out, targets=[targets])
    assert total.requires_grad and torch.isfinite(total)
    assert parts["reg"] == 0.0 and parts["cls"] == pytest.approx(float(total))
    total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())

    # The POSITIVE candidate is whichever one the metric's greedy matcher gives the
    # target to -- ranked by CFAR objectness, so it need not be the cell the target was
    # built from (the azimuth tolerance spans ~6 bins, and a stronger neighbour claims
    # it first). That is the matcher's semantics, and the loss must use it, not a
    # nearest-cell rule of its own.
    matches, _fp, _fn = match_detections(dets, targets, MatchCriterion())
    assert len(matches) == 2
    lab = torch.zeros(rows.numel())
    lab[[di for di, _gi in matches]] = 1.0

    right = model.loss({**out, "logits": [(lab * 2 - 1) * 20.0]}, targets=[targets])[0]
    wrong = model.loss({**out, "logits": [(1 - lab * 2) * 20.0]}, targets=[targets])[0]
    assert float(right) < 1e-3 < float(wrong)


def test_loss_falls_back_to_the_dense_label_map(model, grid, rad):
    """Without a target list, the ground truth is recovered from the dense map by
    `decode_detections` -- the documented approximation (no half-extent widening)."""
    out = model(rad[None])
    rows, cols, scores = out["rows"][0], out["cols"][0], out["cfar"][0]
    picked = _separated(rows, cols, 2)
    y = torch.zeros((1, 3, grid.n_range, grid.n_azimuth))
    for i in picked:
        y[0, 0, int(rows[i]), int(cols[i])] = 1.0
    total, parts = model.loss(out, y)
    assert torch.isfinite(total) and parts["reg"] == 0.0

    # It agrees with the exact path when the targets carry no extent (half_extent 0),
    # which is exactly the condition under which the approximation is not one.
    dets = model.detections_for(rows, cols, scores.tolist())
    targets = [(dets[i][0], dets[i][1], "vehicle", dets[i][3]) for i in picked]
    exact, _ = model.loss(out, targets=[targets])
    assert float(total) == pytest.approx(float(exact), abs=1e-6)


def test_loss_survives_a_frame_with_no_candidates(model, rad):
    out = model(torch.zeros_like(rad)[None])
    total, parts = model.loss(out, targets=[[]])
    assert float(total) == 0.0 and parts["cls"] == 0.0
    total.backward()          # must be legal, not an "does not require grad" error


# --------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------
def test_forward_is_deterministic(model, rad):
    x = torch.stack([rad, rad])
    a = model(x)["detection"]
    b = model(x)["detection"]
    assert torch.equal(a, b)
    # and the two identical frames in one batch produce identical maps
    assert torch.equal(a[0], a[1])


def test_candidate_selection_is_deterministic_under_ties():
    """A constant objectness map is all ties: the kept set must be the first K cells in
    row-major order, every time, on every platform."""
    obj = torch.full((6, 6), 0.5)
    for _ in range(3):
        rows, cols, scores = ch.select_candidates(obj, floor=0.1, max_candidates=5)
        assert rows.tolist() == [0, 0, 0, 0, 0]
        assert cols.tolist() == [0, 1, 2, 3, 4]
        assert torch.all(scores == 0.5)


def test_floor_is_inclusive_and_empty_selection_is_clean():
    obj = torch.zeros((4, 4))
    obj[1, 1] = 0.05
    rows, cols, scores = ch.select_candidates(obj, floor=0.05, max_candidates=8)
    assert rows.tolist() == [1] and cols.tolist() == [1]
    rows, cols, scores = ch.select_candidates(obj, floor=0.9, max_candidates=8)
    assert rows.numel() == cols.numel() == scores.numel() == 0


# --------------------------------------------------------------------------------
# Native registration in train.py
# --------------------------------------------------------------------------------
@pytest.fixture
def manifest(cfg, grid):
    return {"config": cfg.to_dict(),
            "grid": {"n_range": grid.n_range, "n_azimuth": grid.n_azimuth,
                     "max_range_m": grid.max_range_m},
            "input_format": "rad"}


def test_train_build_model_builds_the_model(manifest, rad):
    """Edit 1+2 landed natively in `train.py` (2026-09-23); the `register()` shim that
    used to monkey-patch them in is deleted, so this asserts the real branch."""
    from e2e.ml import train as train_mod

    assert ch.MODEL_NAME in train_mod._MODEL_NAMES
    assert ch.MODEL_NAME in train_mod.build_arg_parser()._actions[2].choices

    model = train_mod.build_model(ch.MODEL_NAME, manifest, device=DEVICE)
    assert isinstance(model, ch.CFARHead)
    assert type(model).__name__.lower() == ch.MODEL_NAME
    out = model(rad[None])["detection"]
    assert out.shape == (1, 3, manifest["grid"]["n_range"], manifest["grid"]["n_azimuth"])

    # Every other name still reaches its own branch, and an unknown one still raises.
    assert isinstance(train_mod.build_model("fftradnet", {**manifest, "input_format": "rd"},
                                            device=DEVICE), torch.nn.Module)
    with pytest.raises(ValueError, match="unknown model"):
        train_mod.build_model("nope", manifest, device=DEVICE)


def test_train_loop_dispatches_to_the_models_own_loss(model, rad, grid):
    """Edit 3: the loop calls `model.loss(out, y)` when the model defines one.

    `detection_loss` over this model's map has ZERO gradient outside <= K cells, so the
    check that matters is that a parameter actually moves -- asserted here against the
    source of the loop rather than trusting the branch by reading it."""
    import inspect
    from e2e.ml import train as train_mod

    src = inspect.getsource(train_mod.train)
    assert 'hasattr(model, "loss")' in src and "model.loss(out, y)" in src

    y = torch.zeros((1, 3, grid.n_range, grid.n_azimuth), device=rad.device)
    y[0, 0, 5, 40] = 1.0
    out = model(rad[None])
    loss, parts = model.loss(out, y)
    loss.backward()
    assert parts["reg"] == 0.0
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


@pytest.fixture(scope="module")
def tiny_rad_manifest(tmp_path_factory, cfg):
    """A real on-disk `rad` corpus, 8 frames, so `train.train` can be run END TO END.

    Exists because of a review finding (2026-09-23): the deleted `register()` guard used
    to be exercised by a test that actually CALLED `train.train(..., "cfarhead")`. When
    the guard went away, the only replacement asserted on the source text of the loop,
    which cannot catch a dispatch that runs but trains the wrong thing. This fixture buys
    back the integration-level coverage on the path the CLI actually takes.
    """
    from e2e.ml import dataset as ml_dataset
    from e2e.ml.scenes import DIFFICULTY_TIERS
    from e2e.radar_config import PRESETS

    PRESETS[cfg.name] = cfg
    try:
        out_dir = tmp_path_factory.mktemp("cfar_head_corpus")
        yield ml_dataset.generate_dataset(
            cfg.name, sorted(DIFFICULTY_TIERS)[0], 8, out_dir=out_dir, seed=0,
            device=DEVICE, splits=(0.5, 0.25, 0.25))
    finally:
        PRESETS.pop(cfg.name, None)


def test_train_end_to_end_uses_the_models_own_loss(tiny_rad_manifest, tmp_path,
                                                   monkeypatch):
    """`train.train(..., "cfarhead")` runs the real loop and optimises `CFARHead.loss`.

    Three things are asserted, and each one is a way the dispatch could be wrong while
    still completing: the regression column is EXACTLY zero every epoch (this head has no
    regression output, so a non-zero there means `detection_loss` ran instead); the
    weights actually moved; and the checkpoint reloads through the ordinary evaluation
    seam. `--gamma`/`--reg-weight`/`--cls-normalize` are passed deliberately absurd
    values -- they are inert for this model, and a run whose loss responds to them is
    running the wrong loss.
    """
    from e2e.ml import train as train_mod

    # The sharpest form of the assertion: `detection_loss` must never be reached at all.
    # Checking the loss VALUE instead would be weak -- with `reg_weight=0` the wrong loss
    # also reports a zero regression term.
    def _must_not_run(*_a, **_k):
        raise AssertionError("detection_loss ran; the loop ignored CFARHead.loss")
    monkeypatch.setattr(train_mod, "detection_loss", _must_not_run)

    out = tmp_path / "cfarhead_run"
    history = train_mod.train(tiny_rad_manifest, ch.MODEL_NAME, epochs=2, batch_size=2,
                              input_format="rad", amp=False, seed=0, out_dir=out,
                              reg_weight=0.0, gamma=0.0, cls_normalize="none")

    assert history["train_reg_loss"] == [0.0, 0.0]
    assert all(v > 0.0 for v in history["train_cls_loss"])
    assert history["train_loss"] == history["train_cls_loss"]   # total == cls, no reg term

    model, _m, _grid, fmt = train_mod.load_model_for_eval(
        tiny_rad_manifest, out / "best.pt", device=DEVICE)
    assert fmt == "rad" and isinstance(model, ch.CFARHead)


def test_forward_is_immune_to_autocast(model, rad):
    """`--amp auto` is the CLI default, and fp16 moves stage 1's objectness by up to
    0.127 (module docstring). `forward` disables autocast for its whole body rather than
    trusting the caller to pass `--amp off`, so the SAME output must come back either
    way. Run on CPU autocast (bfloat16), which is available without a GPU."""
    with torch.no_grad():
        plain = model(rad[None])["detection"]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            casted = model(rad[None])["detection"]
    assert casted.dtype == plain.dtype
    assert torch.equal(plain, casted)


def test_train_build_model_refuses_a_non_rad_input_format(manifest):
    from e2e.ml import train as train_mod

    with pytest.raises(ValueError, match="input_format='rad'"):
        train_mod.build_model(ch.MODEL_NAME, {**manifest, "input_format": "rd"},
                              device=DEVICE)


def test_state_dict_round_trips(model, rad):
    a, r, d = rad.shape
    clone = ch.CFARHead(a, r, d, model.n_range_out, model.n_azimuth_out,
                        cfg=model.cfg, grid=model.grid).eval()
    clone.load_state_dict(model.state_dict())
    assert torch.equal(model(rad[None])["detection"], clone(rad[None])["detection"])
    # The control seams are not parameters and must not leak into the checkpoint.
    assert not any("score_fn" in k or "candidate_fn" in k for k in model.state_dict())


def test_output_geometry_must_match_the_grid(cfg, grid, rad):
    a, r, d = rad.shape
    with pytest.raises(ValueError, match="disagrees with the LabelGrid"):
        ch.CFARHead(a, r, d, grid.n_range + 1, grid.n_azimuth, cfg=cfg, grid=grid)
    with pytest.raises(ValueError, match="odd"):
        ch.CFARHead(a, r, d, grid.n_range, grid.n_azimuth, cfg=cfg, grid=grid, patch=4)

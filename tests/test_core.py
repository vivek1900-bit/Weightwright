import pytest
import torch

from tests.conftest import E, Q, W
from weightwright.budget import Budget, BudgetError
from weightwright.evalset import Generation, check_all, EvalItem, has_loop, passes
from weightwright.ops import EditSet
from weightwright.patch import Patch, PatchMismatch, load_patch, make_meta, materialize, save_patch
from weightwright.slicing import parse_index
from weightwright.store import SafetensorsStore


def test_slicing():
    assert parse_index("[:, 0:128]", 2) == (slice(None), slice(0, 128))
    assert parse_index("3", 2) == (3, slice(None))
    assert parse_index(None, 1) == (slice(None),)
    with pytest.raises(ValueError):
        parse_index("__import__('os')", 2)
    with pytest.raises(ValueError):
        parse_index("[0,1,2]", 2)


def test_store_is_lazy_and_indexed(model_dir):
    s = SafetensorsStore(model_dir)
    assert s.total_params() == 64 + 128 + 80
    assert s.info(Q).dtype == "BF16"
    assert s.names("*layers*") == [Q, W] or sorted(s.names("*layers*")) == sorted([Q, W])


def test_scale_edit_is_sparse(model_dir):
    s = SafetensorsStore(model_dir)
    es = EditSet([{"op": "scale", "tensor": W, "factor": 0.3, "index": "[0, 0]"}])
    edits = es.build(s)
    assert len(edits) == 1 and edits[0].count == 1
    assert edits[0].old.item() == 1.0 and abs(edits[0].new.item() - 0.3) < 1e-6


def test_noop_edit_produces_nothing(model_dir):
    s = SafetensorsStore(model_dir)
    assert EditSet([{"op": "scale", "tensor": W, "factor": 1.0}]).build(s) == []


def test_project_out_removes_direction(model_dir):
    s = SafetensorsStore(model_dir)
    op = {"op": "project_out", "tensor": W, "axis": "out",
          "direction": {"source": "inline", "values": [1, 0, 0, 0, 0, 0, 0, 0]}}
    (e,) = EditSet([op]).build(s)
    new = s.get(W).clone().view(-1)
    new[e.idx] = e.new
    assert torch.allclose(new.view(8, 8)[0], torch.zeros(8), atol=1e-6)  # row 0 removed
    assert torch.equal(new.view(8, 8)[1:], s.get(W)[1:])  # others untouched


def test_singular_vector_projection_kills_top_component(model_dir):
    s = SafetensorsStore(model_dir)
    op = {"op": "project_out", "tensor": W, "axis": "out",
          "direction": {"source": "singular_vector", "tensor": W, "which": 0, "side": "left"}}
    (e,) = EditSet([op]).build(s)
    new = s.get(W).clone().view(-1)
    new[e.idx] = e.new
    sv = torch.linalg.svdvals(new.view(8, 8))
    assert sv[0] < 1e-4  # W was rank-1 (all ones); its only direction is gone


def test_budget_rules(model_dir):
    s = SafetensorsStore(model_dir)
    b = Budget(s.total_params(), max_touched_fraction=0.01, deny=["*embed*"])
    with pytest.raises(BudgetError):
        EditSet([{"op": "zero", "tensor": E}]).build(s, b)
    with pytest.raises(BudgetError):  # 64 weights > 1% of 272
        EditSet([{"op": "zero", "tensor": W}]).build(s, b)
    b2 = Budget(s.total_params(), max_touched_fraction=0.01, allow=["*mlp*"])
    with pytest.raises(BudgetError):
        EditSet([{"op": "zero", "tensor": Q, "index": "[0, 0]"}]).build(s, b2)


def test_patch_roundtrip_exact_including_bf16(model_dir, tmp_path):
    s = SafetensorsStore(model_dir)
    ops = [{"op": "scale", "tensor": W, "factor": 0.3, "index": "[0:2, :]"},
           {"op": "scale", "tensor": Q, "factor": 0.5, "index": "[0:4, :]"}]
    edits = EditSet(ops).build(s)
    save_patch(tmp_path / "p", Patch(edits, make_meta(s, edits, ops)))
    patch = load_patch(tmp_path / "p")
    assert patch.touched == sum(e.count for e in edits)

    patched = materialize(s, patch, tmp_path / "patched")
    sp = SafetensorsStore(patched)
    assert not torch.equal(sp.get(Q), s.get(Q))
    assert sp.get(Q).dtype == torch.bfloat16

    reverted = materialize(sp, patch, tmp_path / "reverted", revert=True)
    sr = SafetensorsStore(reverted)
    for n in s.names():
        assert torch.equal(sr.get(n), s.get(n)), n
    assert (reverted / "config.json").exists()


def test_patch_refuses_wrong_state(model_dir, tmp_path):
    s = SafetensorsStore(model_dir)
    ops = [{"op": "scale", "tensor": W, "factor": 0.3}]
    edits = EditSet(ops).build(s)
    patch = Patch(edits, make_meta(s, edits, ops))
    patched = materialize(s, patch, tmp_path / "a")
    with pytest.raises(PatchMismatch):
        materialize(SafetensorsStore(patched), patch, tmp_path / "b")  # already applied
    with pytest.raises(PatchMismatch):
        materialize(s, patch, tmp_path / "c", revert=True)  # not applied yet
    with pytest.raises(ValueError):
        materialize(s, patch, model_dir)  # never overwrite the source


def test_eval_checks():
    assert has_loop("hello " + "step one. " * 6)
    assert not has_loop("a normal answer with no repetition at all, just words.")
    assert passes({"type": "last_number", "value": 42}, Generation("so 40 + 2 = 42."))
    assert passes({"type": "last_number", "value": 1000}, Generation("it is 1,000"))
    assert not passes({"type": "finished"}, Generation("x", hit_limit=True))
    assert passes({"type": "json_valid"}, Generation('{"ok": true}'))
    it = EvalItem("p", "t", {"all": [{"type": "contains", "value": "a"}, {"type": "no_loop"}]})
    assert check_all(it, Generation("a b c"))

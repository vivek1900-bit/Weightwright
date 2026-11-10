"""Tests for full-tensor SHA-256 patch checksums, verification, and `wright verify` CLI."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.conftest import Q, W, ToyBackend
from weightwright.cli import main
from weightwright.ops import TensorEdit
from weightwright.patch import (
    PATCH_VERSION,
    Patch,
    PatchMismatch,
    load_patch,
    make_meta,
    materialize,
    save_patch,
    tensor_sha256,
    verify_patch,
)
from weightwright.runner import run
from weightwright.store import SafetensorsStore


def _make_sample_patch(store: SafetensorsStore) -> Patch:
    w_orig = store.get(W).contiguous()
    q_orig = store.get(Q).contiguous()

    idx_w = torch.tensor([0, 1], dtype=torch.int64)
    old_w = w_orig.view(-1)[idx_w].clone()
    new_w = old_w + 2.5

    idx_q = torch.tensor([0], dtype=torch.int64)
    old_q = q_orig.view(-1)[idx_q].clone()
    new_q = (old_q + 1.0).to(q_orig.dtype)

    edits = [
        TensorEdit(tensor=W, idx=idx_w, old=old_w, new=new_w),
        TensorEdit(tensor=Q, idx=idx_q, old=old_q, new=new_q),
    ]
    meta = make_meta(store, edits, ops=[{"op": "set", "tensor": W}], goal="test")
    return Patch(edits=edits, meta=meta)


def test_tensor_sha256_properties(tmp_path):
    t_f32 = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    t_bf16 = (torch.arange(64, dtype=torch.float32) * 0.25).to(torch.bfloat16).reshape(8, 8)

    st_path = tmp_path / "roundtrip.safetensors"
    save_file({"f32": t_f32, "bf16": t_bf16}, str(st_path), metadata={"format": "pt"})
    loaded = load_file(str(st_path))

    assert tensor_sha256(t_f32) == tensor_sha256(loaded["f32"])
    assert tensor_sha256(t_bf16) == tensor_sha256(loaded["bf16"])

    t_diff = t_f32.clone()
    t_diff[7, 7] += 1e-5
    assert tensor_sha256(t_f32) != tensor_sha256(t_diff)

    t_reshaped = t_f32.reshape(4, 16)
    assert tensor_sha256(t_f32) != tensor_sha256(t_reshaped)

    t_f16 = t_f32.to(torch.float16)
    assert tensor_sha256(t_f32) != tensor_sha256(t_f16)

    empty = torch.empty((0, 8), dtype=torch.float32)
    h_empty = tensor_sha256(empty)
    assert isinstance(h_empty, str) and len(h_empty) == 64


def test_make_meta_v2_and_no_store_mutation(model_dir):
    store = SafetensorsStore(model_dir)
    w_before = store.get(W).clone()
    q_before = store.get(Q).clone()

    patch = _make_sample_patch(store)
    meta = patch.meta

    assert meta["patch_version"] == PATCH_VERSION == 2
    assert meta["goal"] == "test"
    assert "tensor_checks" in meta
    checks = meta["tensor_checks"]
    assert [c["name"] for c in checks] == meta["tensors"] == [W, Q]

    assert torch.equal(store.get(W), w_before)
    assert torch.equal(store.get(Q), q_before)

    assert checks[0]["sha256_before"] == tensor_sha256(w_before)
    w_expected_after = w_before.clone()
    w_expected_after.view(-1)[patch.edits[0].idx] = patch.edits[0].new
    assert checks[0]["sha256_after"] == tensor_sha256(w_expected_after)


def test_apply_and_revert_roundtrip_hashes(tmp_path, model_dir):
    store = SafetensorsStore(model_dir)
    patch = _make_sample_patch(store)

    applied_dir = tmp_path / "applied"
    materialize(store, patch, applied_dir, link_untouched=False)
    applied_store = SafetensorsStore(applied_dir)

    for edit, chk in zip(patch.edits, patch.meta["tensor_checks"]):
        assert tensor_sha256(applied_store.get(edit.tensor)) == chk["sha256_after"]

    reverted_dir = tmp_path / "reverted"
    materialize(applied_store, patch, reverted_dir, revert=True, link_untouched=False)
    reverted_store = SafetensorsStore(reverted_dir)

    for edit, chk in zip(patch.edits, patch.meta["tensor_checks"]):
        assert tensor_sha256(reverted_store.get(edit.tensor)) == chk["sha256_before"]

    for name in store.names():
        t_base = store.get(name).contiguous().view(torch.uint8)
        t_rev = reverted_store.get(name).contiguous().view(torch.uint8)
        assert torch.equal(t_base, t_rev)


def test_untouched_element_tamper_rejected_v2_allowed_v1_and_force(tmp_path, model_dir):
    base_store = SafetensorsStore(model_dir)
    patch = _make_sample_patch(base_store)

    tampered_dir = tmp_path / "tampered_model"
    shutil.copytree(model_dir, tampered_dir)
    shard_file = tampered_dir / "model.safetensors"
    tensors = load_file(str(shard_file))
    w_mod = tensors[W].clone()
    w_mod[7, 7] += 99.0
    tensors[W] = w_mod
    save_file(tensors, str(shard_file), metadata={"format": "pt"})

    tampered_store = SafetensorsStore(tampered_dir)
    report = verify_patch(tampered_store, patch)
    assert report.state == "mismatch"
    w_verdict = next(v for v in report.verdicts if v.name == W)
    assert w_verdict.status == "mismatch"
    assert w_verdict.full_check is True

    out_fail = tmp_path / "should_not_exist"
    with pytest.raises(PatchMismatch) as exc_info:
        materialize(tampered_store, patch, out_fail)
    assert W in str(exc_info.value)
    assert not out_fail.exists()

    v1_meta = {k: v for k, v in patch.meta.items() if k != "tensor_checks"}
    v1_meta["patch_version"] = 1
    v1_patch = Patch(edits=patch.edits, meta=v1_meta)
    v1_report = verify_patch(tampered_store, v1_patch)
    assert v1_report.state == "base"
    assert all(not v.full_check for v in v1_report.verdicts)
    assert all("element-level check only (patch has no checksums)" in v.detail for v in v1_report.verdicts)

    out_v1 = tmp_path / "out_v1"
    materialize(tampered_store, v1_patch, out_v1, link_untouched=False)
    assert (out_v1 / "model.safetensors").is_file()

    out_force = tmp_path / "out_force"
    materialize(tampered_store, patch, out_force, link_untouched=False, force=True)
    assert (out_force / "model.safetensors").is_file()


def test_wrong_state_apply_twice_and_revert_on_base(tmp_path, model_dir):
    store = SafetensorsStore(model_dir)
    patch = _make_sample_patch(store)

    with pytest.raises(PatchMismatch) as exc_rev:
        materialize(store, patch, tmp_path / "bad_revert", revert=True)
    assert W in str(exc_rev.value)

    applied_dir = tmp_path / "applied_once"
    materialize(store, patch, applied_dir, link_untouched=False)
    applied_store = SafetensorsStore(applied_dir)

    with pytest.raises(PatchMismatch) as exc_apply2:
        materialize(applied_store, patch, tmp_path / "applied_twice", revert=False)
    assert W in str(exc_apply2.value)


def test_verify_patch_states_base_patched_mixed_missing_shape_dtype(tmp_path, model_dir):
    store = SafetensorsStore(model_dir)
    patch = _make_sample_patch(store)

    assert verify_patch(store, patch).state == "base"

    applied_dir = tmp_path / "patched_model"
    materialize(store, patch, applied_dir, link_untouched=False)
    patched_store = SafetensorsStore(applied_dir)
    assert verify_patch(patched_store, patch).state == "patched"

    mixed_dir = tmp_path / "mixed_model"
    shutil.copytree(model_dir, mixed_dir)
    tensors = load_file(str(mixed_dir / "model.safetensors"))
    w_t = tensors[W].clone().contiguous()
    w_t.view(-1)[patch.edits[0].idx] = patch.edits[0].new
    tensors[W] = w_t
    save_file(tensors, str(mixed_dir / "model.safetensors"), metadata={"format": "pt"})
    mixed_report = verify_patch(SafetensorsStore(mixed_dir), patch)
    assert mixed_report.state == "mixed"

    missing_dir = tmp_path / "missing_model"
    missing_dir.mkdir()
    tensors_no_w = {k: v for k, v in load_file(str(Path(model_dir) / "model.safetensors")).items() if k != W}
    save_file(tensors_no_w, str(missing_dir / "model.safetensors"), metadata={"format": "pt"})
    missing_report = verify_patch(SafetensorsStore(missing_dir), patch)
    assert missing_report.state == "mismatch"
    assert next(v for v in missing_report.verdicts if v.name == W).status == "missing"

    shape_dir = tmp_path / "shape_model"
    shape_dir.mkdir()
    tensors_shape = load_file(str(Path(model_dir) / "model.safetensors"))
    tensors_shape[W] = torch.ones((4, 16), dtype=torch.float32)
    save_file(tensors_shape, str(shape_dir / "model.safetensors"), metadata={"format": "pt"})
    shape_report = verify_patch(SafetensorsStore(shape_dir), patch)
    assert shape_report.state == "mismatch"
    assert next(v for v in shape_report.verdicts if v.name == W).status == "shape_mismatch"

    dtype_dir = tmp_path / "dtype_model"
    dtype_dir.mkdir()
    tensors_dtype = load_file(str(Path(model_dir) / "model.safetensors"))
    tensors_dtype[W] = tensors_dtype[W].to(torch.float16)
    save_file(tensors_dtype, str(dtype_dir / "model.safetensors"), metadata={"format": "pt"})
    dtype_report = verify_patch(SafetensorsStore(dtype_dir), patch)
    assert dtype_report.state == "mismatch"
    assert next(v for v in dtype_report.verdicts if v.name == W).status == "dtype_mismatch"


def test_empty_patch(tmp_path, model_dir):
    store = SafetensorsStore(model_dir)
    empty_patch = Patch(edits=[], meta=make_meta(store, [], []))
    report = verify_patch(store, empty_patch)
    assert report.state == "empty"

    out_dir = tmp_path / "empty_out"
    materialize(store, empty_patch, out_dir, link_untouched=False)
    assert (out_dir / "model.safetensors").is_file()


def test_cli_verify_and_info(tmp_path, model_dir, capsys):
    store = SafetensorsStore(model_dir)
    patch = _make_sample_patch(store)
    patch_dir = tmp_path / "patch_v2"
    save_patch(patch_dir, patch)

    rc_ok = main(["verify", str(patch_dir), "--model", str(model_dir)])
    out_ok = capsys.readouterr().out
    assert rc_ok == 0
    assert "state: base" in out_ok

    rc_wrong = main(["verify", str(patch_dir), "--model", str(model_dir), "--expect", "patched"])
    capsys.readouterr()
    assert rc_wrong == 2

    rc_json = main(["verify", str(patch_dir), "--model", str(model_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc_json == 0
    assert payload["state"] == "base"
    assert payload["ok"] is True
    assert len(payload["tensors"]) == 2

    rc_info_v2 = main(["info", str(patch_dir)])
    info_v2_out = capsys.readouterr().out
    assert rc_info_v2 == 0
    assert "tensor_checks" not in info_v2_out
    assert "checksums: yes (2 tensors, full SHA-256)" in info_v2_out

    v1_dir = tmp_path / "patch_v1"
    v1_meta = {k: v for k, v in patch.meta.items() if k != "tensor_checks"}
    v1_meta["patch_version"] = 1
    save_patch(v1_dir, Patch(edits=patch.edits, meta=v1_meta))

    rc_info_v1 = main(["info", str(v1_dir)])
    info_v1_out = capsys.readouterr().out
    assert rc_info_v1 == 0
    assert "checksums: no (patch_version 1: element-level check only)" in info_v1_out


def test_runner_integration_saves_v2_checksums(model_dir, make_spec, capsys):
    spec = make_spec()
    store = SafetensorsStore(model_dir)
    res = run(spec, backend=ToyBackend(store), say=lambda *_: None)
    assert res.status == "accepted"
    assert res.patch_dir is not None

    loaded = load_patch(res.patch_dir)
    assert loaded.meta["patch_version"] == 2
    assert isinstance(loaded.meta.get("tensor_checks"), list)
    assert len(loaded.meta["tensor_checks"]) == len(loaded.edits)

    rc = main(["verify", str(res.patch_dir), "--model", str(model_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["state"] == "base"
    assert payload["ok"] is True

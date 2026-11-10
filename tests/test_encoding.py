"""Windows-safety tests: files written by Notepad or PowerShell must load.

Windows Python reads text files with a legacy code page unless told otherwise, and some tools
add a UTF-8 byte-order mark (BOM). The fixture below simulates the legacy default on any OS.
"""
import json
import pathlib

import pytest

from weightwright.evalset import load_evalset
from weightwright.spec import load_spec

BOM = b"\xef\xbb\xbf"
ITEM = {"prompt": "What is H\u2082O? \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 reply in one word.",
        "tag": "target", "check": {"type": "contains", "value": "water"}}


@pytest.fixture
def legacy_default_encoding(monkeypatch):
    """Make Path.read_text() default to cp1252, like Python on Windows without UTF-8 mode."""
    original = pathlib.Path.read_text

    def read_text(self, encoding=None, *args, **kwargs):
        return original(self, encoding or "cp1252", *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)


def test_evalset_with_utf8_bom_loads(tmp_path, legacy_default_encoding):
    p = tmp_path / "e.jsonl"
    p.write_bytes(BOM + json.dumps(ITEM).encode("utf-8") + b"\n")
    (item,) = load_evalset(p)
    assert item.tag == "target" and item.check["value"] == "water"


def test_evalset_with_raw_non_ascii_text_loads_exactly(tmp_path, legacy_default_encoding):
    p = tmp_path / "e.jsonl"
    p.write_bytes(json.dumps(ITEM, ensure_ascii=False).encode("utf-8") + b"\n")
    (item,) = load_evalset(p)
    assert item.prompt == ITEM["prompt"]


def test_evalset_with_bom_and_non_ascii_loads(tmp_path, legacy_default_encoding):
    p = tmp_path / "e.jsonl"
    p.write_bytes(BOM + json.dumps(ITEM, ensure_ascii=False).encode("utf-8") + b"\n")
    (item,) = load_evalset(p)
    assert item.prompt == ITEM["prompt"]


def test_plain_ascii_evalset_still_loads(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_bytes(json.dumps(ITEM).encode("ascii") + b"\n")
    assert len(load_evalset(p)) == 1


SPEC = ("name: enc\nmodel: {path: ./m}\ngoal: \"Fix H\u2082O and \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 answers\"\n"
        "eval: {file: ./e.jsonl}\neditor: {provider: openai_compat, model: x}\n")


@pytest.mark.parametrize("bom", [False, True])
def test_spec_with_non_ascii_goal_loads(tmp_path, legacy_default_encoding, bom):
    p = tmp_path / "s.yaml"
    p.write_bytes((BOM if bom else b"") + SPEC.encode("utf-8"))
    spec = load_spec(p)
    assert spec.goal == "Fix H\u2082O and \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 answers"

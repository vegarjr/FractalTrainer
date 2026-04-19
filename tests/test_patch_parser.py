"""Tests for the vendored patch_parser. Also verifies YAML-patch support."""

import tempfile
from pathlib import Path

from fractaltrainer.repair.patch_parser import PatchParser


def test_parse_no_fix():
    r = PatchParser().parse("NO_FIX_FOUND: cannot determine a fix")
    assert r.no_fix is True
    assert "cannot determine" in (r.no_fix_reason or "")


def test_parse_empty_response_errors():
    r = PatchParser().parse("random text with no patch")
    assert r.no_fix is False
    assert r.parse_error is not None


def test_parse_single_patch_block():
    response = """<<<PATCH
file: configs/hparams.yaml
---old---
learning_rate: 0.1
---new---
learning_rate: 0.05
>>>PATCH"""
    r = PatchParser().parse(response)
    assert not r.no_fix
    assert r.parse_error is None
    assert len(r.patches) == 1
    p = r.patches[0]
    assert p.file_path == "configs/hparams.yaml"
    assert p.old_text == "learning_rate: 0.1"
    assert p.new_text == "learning_rate: 0.05"


def test_validate_patches_allowed_files_scope():
    response = """<<<PATCH
file: src/fractaltrainer/__init__.py
---old---
__version__ = "0.1.0"
---new---
__version__ = "0.2.0"
>>>PATCH"""
    parser = PatchParser()
    r = parser.parse(response)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "hparams.yaml").write_text(
            "learning_rate: 0.1\n")
        (tmp_path / "src" / "fractaltrainer").mkdir(parents=True)
        (tmp_path / "src" / "fractaltrainer" / "__init__.py").write_text(
            '__version__ = "0.1.0"\n')

        errs = parser.validate_patches(
            r.patches, str(tmp_path),
            allowed_files=["configs/hparams.yaml"])
        assert errs, "patch outside allowed_files must be rejected"
        assert any("not in" in e for e in errs)


def test_validate_patches_yaml_passes_ast_skip():
    response = """<<<PATCH
file: configs/hparams.yaml
---old---
learning_rate: 0.1
---new---
learning_rate: 0.05
>>>PATCH"""
    parser = PatchParser()
    r = parser.parse(response)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "hparams.yaml").write_text(
            "learning_rate: 0.1\n")
        errs = parser.validate_patches(
            r.patches, str(tmp_path),
            allowed_files=["configs/hparams.yaml"])
        assert errs == []


def test_validate_patches_old_text_must_be_unique():
    response = """<<<PATCH
file: configs/hparams.yaml
---old---
value: 0
---new---
value: 99
>>>PATCH"""
    parser = PatchParser()
    r = parser.parse(response)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "hparams.yaml").write_text(
            "value: 0\nother: 0\nvalue: 0\n")
        errs = parser.validate_patches(
            r.patches, str(tmp_path),
            allowed_files=["configs/hparams.yaml"])
        assert any("unique" in e for e in errs)


def test_validate_patches_line_budget():
    big_new = "\n".join([f"key_{i}: {i}" for i in range(100)])
    response = f"""<<<PATCH
file: configs/hparams.yaml
---old---
learning_rate: 0.1
---new---
{big_new}
>>>PATCH"""
    parser = PatchParser()
    r = parser.parse(response)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "hparams.yaml").write_text(
            "learning_rate: 0.1\n")
        errs = parser.validate_patches(
            r.patches, str(tmp_path),
            max_total_lines=20,
            allowed_files=["configs/hparams.yaml"])
        assert any("exceeds" in e for e in errs)

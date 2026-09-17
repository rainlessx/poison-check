"""Тесты флага --bundle-check/--no-bundle-check (полное отключение проверки комплекта).

Часть B задачи: с флагом находки MLS-BUNDLE-* не идут ни в отчёт, ни в exit-код
гейта; без флага — прежнее поведение (HIGH, гейт по политике). CLI-флаг имеет
приоритет над enabled_detectors политики и только ВЫКЛЮЧАЕТ.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from poison_check.cli import app

runner = CliRunner()


def _make_bundle(root: Path) -> Path:
    """Комплект: чистые веса + config.json(auto_map) + modeling.py с exec (маркер)."""
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").write_bytes(b'\x08\x00\x00\x00\x00\x00\x00\x00{}      ')
    (bundle / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling.M"}})
    )
    # exec от ЛИТЕРАЛА → видимый код; нагрузка — безобидный маркер.
    (bundle / "modeling.py").write_text(
        "exec(\"open('proof_flag.txt','w').write('triggered')\")\n"
    )
    return bundle


def _bundle_issue_codes(output: str) -> list[str]:
    report = json.loads(output)
    return [
        i["code"]
        for r in report["results"]
        for i in r.get("issues", [])
        if i["code"].startswith("MLS-BUNDLE-")
    ]


class TestBundleCheckFlag:
    """Поведение с флагом и без, взаимодействие с гейтом политики."""

    def test_without_flag_strict_gate_trips(self, tmp_path: Path) -> None:
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            app, ["scan", str(bundle), "--policy", "strict", "--format", "json"]
        )
        # strict: fail_on_severity=high → HIGH-находка BUNDLE-002 валит гейт.
        assert result.exit_code == 1
        codes = _bundle_issue_codes(result.stdout)
        assert "MLS-BUNDLE-002" in codes

    def test_with_flag_disabled_no_findings_and_exit_zero(self, tmp_path: Path) -> None:
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            app,
            ["scan", str(bundle), "--policy", "strict", "--no-bundle-check",
             "--format", "json"],
        )
        # Проверка комплекта полностью отключена: ни находок, ни гейта.
        assert result.exit_code == 0
        assert _bundle_issue_codes(result.stdout) == []

    def test_default_is_enabled(self, tmp_path: Path) -> None:
        # Дефолт — проверка ВКЛЮЧЕНА: без флага BUNDLE-находки присутствуют.
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(app, ["scan", str(bundle), "--format", "json"])
        assert "MLS-BUNDLE-002" in _bundle_issue_codes(result.stdout)

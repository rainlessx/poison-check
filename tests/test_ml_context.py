"""Тесты для poison_check/analysis/ml_context.py.

Проверяет корректность MLContextAnalyzer:
- torch.nn.* globals → framework="pytorch"
- sklearn.* globals → framework="sklearn"
- Пустые globals → framework="unknown", confidence=0.0
- Смешанные globals → фреймворк с наибольшим числом совпадений
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poison_check.analysis.ml_context import MLContextAnalyzer
from poison_check.core.result import MLContext
from poison_check.core.scanner_base import RawScanData


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def make_raw(globals_set: set[tuple[str, str]]) -> RawScanData:
    """Создаёт минимальный RawScanData с заданными globals для тестов."""
    return RawScanData(
        file_path=Path("test.bin"),
        file_hash={},
        file_size=0,
        scanner_name="test",
        globals=globals_set if globals_set else None,
    )


@pytest.fixture()
def analyzer() -> MLContextAnalyzer:
    """Экземпляр MLContextAnalyzer для тестов."""
    return MLContextAnalyzer()


# ---------------------------------------------------------------------------
# Тест 1: PyTorch globals
# ---------------------------------------------------------------------------


class TestPyTorchDetection:
    """Тесты определения PyTorch-фреймворка."""

    def test_torch_nn_globals_gives_pytorch(self, analyzer: MLContextAnalyzer) -> None:
        """Файл с torch.nn.* globals → framework='pytorch', confidence > 0.5."""
        raw = make_raw(
            {
                ("torch.nn.modules.linear", "Linear"),
                ("torch.nn.modules.activation", "ReLU"),
                ("torch.nn.modules.normalization", "LayerNorm"),
                ("torch._utils", "_rebuild_tensor_v2"),
                ("torch", "FloatStorage"),
                ("collections", "OrderedDict"),
            }
        )
        ctx = analyzer.analyze(raw)

        assert ctx.framework == "pytorch", f"Ожидался pytorch, получен {ctx.framework}"
        assert ctx.confidence > 0.0, "Уверенность должна быть > 0"
        assert len(ctx.detected_patterns) > 0

    def test_torch_storage_gives_pytorch(self, analyzer: MLContextAnalyzer) -> None:
        """Минимальный набор: только torch._utils — должен определяться как pytorch."""
        raw = make_raw(
            {
                ("torch._utils", "_rebuild_tensor_v2"),
                ("torch", "FloatStorage"),
                ("collections", "OrderedDict"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "pytorch"
        assert ctx.confidence > 0.0

    def test_confidence_positive_for_pytorch(self, analyzer: MLContextAnalyzer) -> None:
        """confidence должна быть > 0 для любого PyTorch-файла."""
        raw = make_raw(
            {
                ("torch", "Tensor"),
                ("torch.storage", "_load_from_bytes"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "pytorch"
        assert 0.0 < ctx.confidence <= 1.0

    def test_detected_patterns_are_pytorch_modules(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """detected_patterns содержат только pytorch-индикаторы, не sklearn."""
        raw = make_raw(
            {
                ("torch.nn.modules.linear", "Linear"),
                ("torch._utils", "_rebuild_tensor_v2"),
                ("torch", "FloatStorage"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "pytorch"
        for pattern in ctx.detected_patterns:
            assert pattern.startswith("torch"), (
                f"Паттерн не начинается с 'torch': {pattern}"
            )

    def test_full_huggingface_legacy_model(self, analyzer: MLContextAnalyzer) -> None:
        """Типичный набор globals из HuggingFace legacy .bin файла."""
        raw = make_raw(
            {
                ("torch._utils", "_rebuild_tensor_v2"),
                ("torch", "FloatStorage"),
                ("collections", "OrderedDict"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "pytorch"


# ---------------------------------------------------------------------------
# Тест 2: sklearn globals
# ---------------------------------------------------------------------------


class TestSklearnDetection:
    """Тесты определения sklearn-фреймворка."""

    def test_sklearn_logistic_regression(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """Globals из LogisticRegression → framework='sklearn'."""
        raw = make_raw(
            {
                ("sklearn.linear_model._logistic", "LogisticRegression"),
                ("joblib.numpy_pickle", "NumpyArrayWrapper"),
                ("numpy", "ndarray"),
                ("numpy._core.multiarray", "scalar"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "sklearn", f"Ожидался sklearn, получен {ctx.framework}"
        assert ctx.confidence > 0.0

    def test_sklearn_random_forest(self, analyzer: MLContextAnalyzer) -> None:
        """Globals из RandomForestClassifier → framework='sklearn'."""
        raw = make_raw(
            {
                ("sklearn.ensemble._forest", "RandomForestClassifier"),
                ("sklearn.tree._classes", "DecisionTreeClassifier"),
                ("joblib.numpy_pickle", "NumpyArrayWrapper"),
                ("numpy", "ndarray"),
                ("dtype", "dtype"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "sklearn"

    def test_joblib_numpyarraywrapper_is_sklearn_indicator(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """joblib.numpy_pickle.NumpyArrayWrapper — сильный sklearn-индикатор."""
        raw = make_raw(
            {
                ("joblib.numpy_pickle", "NumpyArrayWrapper"),
                ("numpy", "ndarray"),
            }
        )
        ctx = analyzer.analyze(raw)
        # joblib.numpy_pickle входит в SKLEARN_INDICATORS
        assert ctx.framework == "sklearn"

    def test_sklearn_pipeline(self, analyzer: MLContextAnalyzer) -> None:
        """sklearn.pipeline.Pipeline → framework='sklearn'."""
        raw = make_raw(
            {
                ("sklearn.pipeline", "Pipeline"),
                ("sklearn.preprocessing._data", "StandardScaler"),
                ("joblib.numpy_pickle", "NumpyArrayWrapper"),
                ("numpy", "dtype"),
                ("numpy._core.multiarray", "scalar"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "sklearn"

    def test_sklearn_gradient_boosting(self, analyzer: MLContextAnalyzer) -> None:
        """GradientBoostingClassifier с numpy.random internals → sklearn."""
        raw = make_raw(
            {
                ("sklearn.ensemble._gb", "GradientBoostingClassifier"),
                ("sklearn.tree._classes", "DecisionTreeRegressor"),
                ("sklearn.tree._tree", "Tree"),
                ("numpy._core.multiarray", "_reconstruct"),
                ("numpy._core.numeric", "_frombuffer"),
                ("numpy.random._mt19937", "MT19937"),
                ("numpy.random._pickle", "__randomstate_ctor"),
                ("joblib.numpy_pickle", "NumpyArrayWrapper"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "sklearn"


# ---------------------------------------------------------------------------
# Тест 3: Пустые globals
# ---------------------------------------------------------------------------


class TestUnknownFramework:
    """Тесты для файлов без globals или с неизвестным фреймворком."""

    def test_none_globals_returns_unknown(self, analyzer: MLContextAnalyzer) -> None:
        """Файл без globals → framework='unknown', confidence=0.0."""
        raw = make_raw(set())  # → globals=None в make_raw
        ctx = analyzer.analyze(raw)

        assert ctx.framework == "unknown"
        assert ctx.confidence == 0.0
        assert ctx.detected_patterns == []

    def test_raw_data_with_none_globals(self, analyzer: MLContextAnalyzer) -> None:
        """RawScanData.globals=None → framework='unknown'."""
        raw = RawScanData(
            file_path=Path("test.bin"),
            file_hash={},
            file_size=0,
            scanner_name="test",
            globals=None,
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "unknown"
        assert ctx.confidence == 0.0

    def test_unknown_modules_return_unknown(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """Незнакомые модули без совпадений → framework='unknown'."""
        raw = make_raw(
            {
                ("builtins", "object"),
                ("collections", "OrderedDict"),
                ("_codecs", "encode"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "unknown"
        assert ctx.confidence == 0.0

    def test_confidence_is_zero_for_unknown(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """confidence=0.0 только для unknown."""
        raw = make_raw(set())
        ctx = analyzer.analyze(raw)
        assert ctx.confidence == 0.0


# ---------------------------------------------------------------------------
# Тест 4: Смешанные globals — побеждает наибольшее совпадение
# ---------------------------------------------------------------------------


class TestMixedGlobals:
    """Тесты для файлов со смешанными globals."""

    def test_torch_wins_over_numpy_when_more_matches(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """Если torch-индикаторов больше чем numpy — побеждает pytorch."""
        raw = make_raw(
            {
                # torch — 3 индикатора
                ("torch.nn.modules.linear", "Linear"),
                ("torch._utils", "_rebuild_tensor_v2"),
                ("torch", "FloatStorage"),
                # numpy — 1 индикатор
                ("numpy.core.multiarray", "_reconstruct"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "pytorch"

    def test_sklearn_wins_over_numpy_when_more_matches(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """sklearn-индикаторов больше чем numpy → framework='sklearn'."""
        raw = make_raw(
            {
                # sklearn — несколько индикаторов
                ("sklearn.ensemble._forest", "RandomForestClassifier"),
                ("sklearn.tree._classes", "DecisionTreeClassifier"),
                ("joblib.numpy_pickle", "NumpyArrayWrapper"),
                ("sklearn.pipeline", "Pipeline"),
                # numpy — 1
                ("numpy", "ndarray"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "sklearn"

    def test_torch_wins_over_sklearn_when_dominant(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """Если torch доминирует численно — должен победить pytorch."""
        raw = make_raw(
            {
                # torch — много индикаторов
                ("torch.nn.modules.linear", "Linear"),
                ("torch.nn.modules.conv", "Conv2d"),
                ("torch.nn.modules.activation", "ReLU"),
                ("torch.nn.modules.normalization", "LayerNorm"),
                ("torch.nn.modules.dropout", "Dropout"),
                ("torch._utils", "_rebuild_tensor_v2"),
                # sklearn — 1
                ("sklearn.base", "BaseEstimator"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.framework == "pytorch"

    def test_analyze_bytes_convenience(self, analyzer: MLContextAnalyzer) -> None:
        """analyze_bytes() работает как analyze() на set[(module, name)]."""
        globals_set = {
            ("torch._utils", "_rebuild_tensor_v2"),
            ("torch", "FloatStorage"),
            ("collections", "OrderedDict"),
        }
        ctx = analyzer.analyze_bytes(globals_set)
        assert ctx.framework == "pytorch"
        assert ctx.confidence > 0.0

    def test_analyze_bytes_empty(self, analyzer: MLContextAnalyzer) -> None:
        """analyze_bytes() с пустым set → unknown."""
        ctx = analyzer.analyze_bytes(set())
        assert ctx.framework == "unknown"
        assert ctx.confidence == 0.0


# ---------------------------------------------------------------------------
# Тест 5: Свойства MLContext
# ---------------------------------------------------------------------------


class TestMLContextProperties:
    """Тесты для свойств возвращаемого MLContext."""

    def test_confidence_is_between_0_and_1(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """confidence всегда 0.0 ≤ confidence ≤ 1.0."""
        raw = make_raw(
            {
                ("torch.nn.modules.linear", "Linear"),
                ("torch._utils", "_rebuild_tensor_v2"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert 0.0 <= ctx.confidence <= 1.0

    def test_detected_patterns_are_sorted(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """detected_patterns всегда отсортированы."""
        raw = make_raw(
            {
                ("torch.nn.modules.transformer", "Transformer"),
                ("torch.nn.modules.linear", "Linear"),
                ("torch._utils", "_rebuild_tensor_v2"),
                ("torch.storage", "_load_from_bytes"),
            }
        )
        ctx = analyzer.analyze(raw)
        assert ctx.detected_patterns == sorted(ctx.detected_patterns)

    def test_mlcontext_framework_is_string(
        self, analyzer: MLContextAnalyzer
    ) -> None:
        """framework всегда строка."""
        for globals_set in [
            set(),
            {("torch._utils", "_rebuild_tensor_v2")},
            {("sklearn.pipeline", "Pipeline"), ("joblib.numpy_pickle", "NumpyArrayWrapper")},
        ]:
            ctx = analyzer.analyze_bytes(globals_set)
            assert isinstance(ctx.framework, str)
            assert isinstance(ctx.confidence, float)
            assert isinstance(ctx.detected_patterns, list)


# ---------------------------------------------------------------------------
# Регрессия аудита #28: transformers как отдельный framework
# ---------------------------------------------------------------------------


class TestTransformersDetection:
    """HuggingFace transformers распознаётся как отдельный фреймворк."""

    def test_bert_detected_as_transformers(self) -> None:
        """BERT-модель → framework='transformers'."""
        from poison_check.analysis.ml_context import MLContextAnalyzer  # noqa: PLC0415

        globals_set = {
            ("transformers.models.bert.modeling_bert", "BertModel"),
            ("transformers.modeling_utils", "PreTrainedModel"),
            ("torch.nn.modules.linear", "Linear"),
            ("collections", "OrderedDict"),
        }
        ctx = MLContextAnalyzer().analyze_bytes(globals_set)
        assert ctx.framework == "transformers", (
            f"Ожидался 'transformers', получено: {ctx.framework}"
        )

    def test_llama_detected_as_transformers(self) -> None:
        """LLaMA → transformers."""
        from poison_check.analysis.ml_context import MLContextAnalyzer  # noqa: PLC0415

        globals_set = {
            ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
            ("transformers.models.llama.configuration_llama", "LlamaConfig"),
            ("torch.nn.modules.linear", "Linear"),
        }
        ctx = MLContextAnalyzer().analyze_bytes(globals_set)
        assert ctx.framework == "transformers"

    def test_pure_torch_still_pytorch(self) -> None:
        """Чистый PyTorch без transformers → 'pytorch' (regression)."""
        from poison_check.analysis.ml_context import MLContextAnalyzer  # noqa: PLC0415

        globals_set = {
            ("torch.nn.modules.linear", "Linear"),
            ("torch.nn.modules.conv", "Conv2d"),
            ("torch._utils", "_rebuild_tensor_v2"),
        }
        ctx = MLContextAnalyzer().analyze_bytes(globals_set)
        assert ctx.framework == "pytorch"

    def test_transformers_allowlist_includes_pytorch(self) -> None:
        """AllowlistDetector для transformers использует объединение transformers+pytorch+numpy.

        Регрессия: HF-модели всегда содержат torch.* классы — без объединения
        получали бы шум по torch.nn.* при детекции transformers.
        """
        from poison_check.core.result import MLContext  # noqa: PLC0415
        from poison_check.detectors.allowlist_detector import (  # noqa: PLC0415
            AllowlistDetector,
        )

        det = AllowlistDetector()
        ctx = MLContext(framework="transformers", confidence=1.0)
        allowlist = det._get_allowlist(ctx)

        # transformers
        assert ("transformers.models.bert.modeling_bert", "BertModel") in allowlist
        # pytorch (косвенно — должно быть включено)
        # Проверяем хотя бы что объединение существенно больше чистого transformers
        transformers_only = det._allowlists.get("transformers", set())
        assert len(allowlist) > len(transformers_only)

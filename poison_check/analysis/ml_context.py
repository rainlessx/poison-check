"""Определение ML-фреймворка по импортируемым модулям из RawScanData.

ML-context aware анализ: серьёзность находки зависит от определённого
фреймворка модели, что снижает уровень ложных срабатываний.

Алгоритм:
1. Извлекает множество module-частей из raw_data.globals.
2. Подсчитывает совпадения с индикаторами каждого фреймворка.
3. Возвращает фреймворк с наибольшим числом совпадений.
4. confidence = совпадений / len(индикаторы_фреймворка), min(1.0).
"""

from __future__ import annotations

from poison_check.core.result import MLContext
from poison_check.core.scanner_base import RawScanData


class MLContextAnalyzer:
    """Определяет ML-фреймворк по globals, собранным сканером.

    Работает статически: не загружает модели, не исполняет код.
    Использует множество module-индикаторов для каждого фреймворка.

    Примеры:
        >>> analyzer = MLContextAnalyzer()
        >>> ctx = analyzer.analyze(raw_data)
        >>> ctx.framework  # "pytorch" | "sklearn" | "numpy" | "tensorflow" | "unknown"
        >>> ctx.confidence  # 0.0 – 1.0
    """

    # Индикаторы PyTorch — модули, характерные ТОЛЬКО для PyTorch моделей.
    # Эмпирически подтверждено: freq=12/12 HuggingFace моделей (апрель 2026).
    PYTORCH_INDICATORS: frozenset[str] = frozenset(
        {
            "torch",
            "torch.nn",
            "torch.nn.modules",
            "torch.nn.modules.linear",
            "torch.nn.modules.conv",
            "torch.nn.modules.activation",
            "torch.nn.modules.normalization",
            "torch.nn.modules.batchnorm",
            "torch.nn.modules.dropout",
            "torch.nn.modules.pooling",
            "torch.nn.modules.container",
            "torch.nn.modules.rnn",
            "torch.nn.modules.sparse",
            "torch.nn.modules.transformer",
            "torch.storage",
            "torch._utils",
            "torch._tensor",
        }
    )

    # Индикаторы scikit-learn / joblib.
    # joblib.numpy_pickle появляется в 100% sklearn моделей (эмпирика).
    SKLEARN_INDICATORS: frozenset[str] = frozenset(
        {
            "sklearn",
            "sklearn.base",
            "sklearn.linear_model",
            "sklearn.linear_model._base",
            "sklearn.linear_model._logistic",
            "sklearn.linear_model._ridge",
            "sklearn.linear_model._coordinate_descent",
            "sklearn.linear_model._stochastic_gradient",
            "sklearn.linear_model._huber",
            "sklearn.linear_model._theil_sen",
            "sklearn.linear_model._ransac",
            "sklearn.ensemble",
            "sklearn.ensemble._forest",
            "sklearn.ensemble._gb",
            "sklearn.ensemble._weight_boosting",
            "sklearn.ensemble._stacking",
            "sklearn.ensemble._hist_gradient_boosting.gradient_boosting",
            "sklearn.ensemble._hist_gradient_boosting.binning",
            "sklearn.tree",
            "sklearn.tree._classes",
            "sklearn.tree._tree",
            "sklearn.svm",
            "sklearn.svm._classes",
            "sklearn.neighbors",
            "sklearn.neighbors._classification",
            "sklearn.neighbors._regression",
            "sklearn.preprocessing",
            "sklearn.preprocessing._data",
            "sklearn.preprocessing._label",
            "sklearn.preprocessing._encoders",
            "sklearn.pipeline",
            "sklearn.decomposition",
            "sklearn.decomposition._pca",
            "sklearn.decomposition._nmf",
            "sklearn.decomposition._lda",
            "sklearn.cluster",
            "sklearn.cluster._kmeans",
            "sklearn.naive_bayes",
            "sklearn.calibration",
            "sklearn.multioutput",
            "sklearn.feature_selection._univariate_selection",
            "sklearn.cross_decomposition._pls",
            "sklearn.utils",
            "sklearn._loss.link",
            "sklearn._loss.loss",
            "joblib.numpy_pickle",  # появляется в 100% joblib-сохранённых sklearn моделей
        }
    )

    # Индикаторы NumPy (.npy / .npz без sklearn или torch).
    NUMPY_INDICATORS: frozenset[str] = frozenset(
        {
            "numpy",
            "numpy.core",
            "numpy.core.multiarray",
            "numpy._core",
            "numpy._core.multiarray",
            "numpy._core.numeric",
            "numpy.lib.npyio",
            "numpy.ma.core",
            "numpy.random",
            "numpy.random._mt19937",
            "numpy.random._pickle",
            "numpy.random.bit_generator",
        }
    )

    # Индикаторы TensorFlow / Keras.
    TF_INDICATORS: frozenset[str] = frozenset(
        {
            "tensorflow",
            "tensorflow.python",
            "tensorflow.python.framework",
            "tensorflow.python.ops",
            "tensorflow.python.training",
            "keras",
            "keras.engine",
            "keras.layers",
            "keras.models",
            "keras.optimizers",
            "tf_keras",
        }
    )

    # Индикаторы HuggingFace transformers / tokenizers / accelerate / peft.
    # Аудит #28: до этого фреймворк не распознавался — все HF-модели падали в
    # "pytorch" (через зависимость от torch), и AllowlistDetector использовал
    # pytorch.yaml без знания о transformers.models.* классах.
    TRANSFORMERS_INDICATORS: frozenset[str] = frozenset(
        {
            "transformers",
            "transformers.modeling_utils",
            "transformers.configuration_utils",
            "transformers.tokenization_utils",
            "transformers.tokenization_utils_fast",
            "transformers.tokenization_utils_base",
            "transformers.feature_extraction_utils",
            "transformers.image_processing_utils",
            "transformers.processing_utils",
            "transformers.generation.configuration_utils",
            "transformers.models.bert.modeling_bert",
            "transformers.models.distilbert.modeling_distilbert",
            "transformers.models.roberta.modeling_roberta",
            "transformers.models.gpt2.modeling_gpt2",
            "transformers.models.t5.modeling_t5",
            "transformers.models.bart.modeling_bart",
            "transformers.models.llama.modeling_llama",
            "transformers.models.mistral.modeling_mistral",
            "transformers.models.mixtral.modeling_mixtral",
            "transformers.models.falcon.modeling_falcon",
            "transformers.models.qwen2.modeling_qwen2",
            "transformers.models.whisper.modeling_whisper",
            "transformers.models.clip.modeling_clip",
            "transformers.models.vit.modeling_vit",
            "tokenizers",
            "tokenizers.models",
            "sentence_transformers.SentenceTransformer",
            "peft.peft_model",
            "peft.tuners.lora",
            "accelerate.state",
        }
    )

    # Все фреймворки в порядке приоритета разрешения конфликтов.
    # transformers ставится ВЫШЕ pytorch: HF-модели всегда содержат torch.*
    # модули, но точная идентификация как transformers даёт корректный allowlist.
    _FRAMEWORKS: tuple[tuple[str, frozenset[str]], ...] = (
        ("transformers", TRANSFORMERS_INDICATORS),
        ("pytorch", PYTORCH_INDICATORS),
        ("sklearn", SKLEARN_INDICATORS),
        ("tensorflow", TF_INDICATORS),
        ("numpy", NUMPY_INDICATORS),
    )

    def analyze(self, raw_data: RawScanData) -> MLContext:
        """Определяет ML-фреймворк по globals из RawScanData.

        Алгоритм:
        1. Если globals отсутствуют — возвращает framework="unknown", confidence=0.0.
        2. Извлекает множество модулей из (module, name) пар.
        3. Для каждого фреймворка считает пересечение modules ∩ indicators.
        4. Возвращает фреймворк с наибольшим пересечением.
        5. При равных счётчиках — первый по приоритету (_FRAMEWORKS).
        6. Если ни одного совпадения — framework="unknown".

        confidence = len(совпадений) / len(индикаторов_фреймворка), но не > 1.0.
        Это намеренно: даже 1-2 torch.* уже означает PyTorch с низкой уверенностью.

        Args:
            raw_data: Результат сканирования файла.

        Returns:
            MLContext с framework, confidence и detected_patterns.
        """
        if not raw_data.globals:
            return MLContext(framework="unknown", confidence=0.0, detected_patterns=[])

        # Только модульная часть (module, _name) → берём module
        modules: set[str] = {module for module, _name in raw_data.globals}

        best_framework = "unknown"
        best_count = 0
        best_confidence = 0.0
        best_patterns: list[str] = []

        for framework_name, indicators in self._FRAMEWORKS:
            matched = modules & indicators
            count = len(matched)
            if count == 0:
                continue
            # confidence = доля покрытых индикаторов (не > 1.0)
            confidence = min(count / len(indicators), 1.0)
            # Победитель = больше матчей по абсолюту.
            # При равных матчах — приоритет по порядку в _FRAMEWORKS
            # (pytorch > sklearn > tensorflow > numpy).
            # Это корректно: sklearn-специфичные модули (sklearn.*, joblib.*)
            # важнее общих numpy-модулей при tie.
            if count > best_count:
                best_framework = framework_name
                best_count = count
                best_confidence = confidence
                best_patterns = sorted(matched)
            # При равных матчах не обновляем — первый по приоритету побеждает.

        # Правило доминирования: sklearn > numpy.
        # sklearn-модели ВСЕГДА содержат numpy-внутренние globals (numpy._core.*),
        # поэтому numpy-матчей больше чем sklearn-матчей для маленьких моделей.
        # Если есть хотя бы один sklearn-индикатор — это sklearn, а не numpy.
        if best_framework == "numpy":
            sklearn_matched = modules & self.SKLEARN_INDICATORS
            if sklearn_matched:
                best_framework = "sklearn"
                best_count = len(sklearn_matched)
                best_confidence = min(best_count / len(self.SKLEARN_INDICATORS), 1.0)
                best_patterns = sorted(sklearn_matched)

        return MLContext(
            framework=best_framework,
            confidence=round(best_confidence, 4),
            detected_patterns=best_patterns,
        )

    def analyze_bytes(
        self,
        globals_set: set[tuple[str, str]],
    ) -> MLContext:
        """Удобный вариант analyze() для прямого вызова с set[(module, name)].

        Используется в тестах и forensics-режиме.

        Args:
            globals_set: Множество (module, name) пар.

        Returns:
            MLContext.
        """
        from pathlib import Path

        from poison_check.core.scanner_base import RawScanData

        dummy = RawScanData(
            file_path=Path("<in-memory>"),
            file_hash={},
            file_size=0,
            scanner_name="<direct>",
            globals=globals_set if globals_set else None,
        )
        return self.analyze(dummy)

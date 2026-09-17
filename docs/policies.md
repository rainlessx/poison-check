# Политики

Политики контролируют поведение poison-check в конкретном контексте: какие детекторы включены, с какого severity находка требует внимания, при каком severity проваливать CI/CD, и какие compliance-требования проверять.

---

## Два независимых порога

| Ключ | Роль | Что делает | Чего НЕ делает |
|---|---|---|---|
| `severity_threshold` | порог **внимания** | находки ниже порога помечаются подпороговыми во всех форматах вывода | не удаляет находки из отчёта, не влияет на exit code |
| `fail_on_severity` | порог **действия** | с этого уровня CLI возвращает exit code 1 | не влияет на состав и подачу отчёта |

`severity_threshold` — это порог ВНИМАНИЯ, а не фильтр отображения. Находка ниже
порога **остаётся в отчёте** и лишь помечается: пользователь видит её, но она не
претендует на немедленное действие. Скрывать находки нельзя: политика по
умолчанию имеет `severity_threshold: medium`, а «файл не удалось разобрать»
(`MLS-PARSE-001`) — это LOW, то есть при трактовке «скрывать» самый частый
сценарий «файл не проверен» исчезал бы из отчёта по умолчанию.

Как выглядит пометка в каждом формате:

| Формат | Проявление подпороговой находки |
|---|---|
| console | отдельный компактный блок «Ниже порога внимания» под основными находками + строка в итогах |
| JSON | `issues[].below_threshold: true`, счётчик `summary.below_threshold`, блок `tool.thresholds` |
| SARIF | result остаётся в `runs[].results` c исходным `level`, добавляется `suppressions[0]` (`kind: external`) и `properties.below_threshold` |
| SBOM (CycloneDX) | `vulnerabilities[].analysis.state: in_triage` c пояснением + свойство `poison-check:below_threshold` |

**Что порог не подавляет никогда:**

* находки уровня CRITICAL и HIGH — даже при `severity_threshold: critical`;
* факты уровня файла: parse-error (`MLS-PARSE-001`), обрыв разбора
  (`MLS-PKL-006`), бомбы (`MLS-BOMB-001`, `MLS-CMP-*`, `MLS-JOBLIB-003`),
  непроверенный файл (`MLS-KERAS-003`, `MLS-JOBLIB-001/002`), аномалии
  заголовка GGUF (`MLS-GGUF-005/006`). Их severity низкий по шкале, но они
  сообщают, что проверка **не состоялась**;
* ошибку разбора `FileResult.error` — она идёт отдельным каналом
  (в SARIF это `toolExecutionNotifications`) и порогу не подчиняется вовсе;
* счётчики по severity в `summary` — порог не вычитает находки из статистики.

Реализация: `poison_check/output/severity_threshold.py`, тесты —
`tests/test_severity_threshold.py`.

---

## Встроенные политики

### `default` — стандартная

Баланс между полнотой и уровнем шума. Подходит для большинства задач.

```yaml
name: default
description: "Стандартная политика — баланс между полнотой и уровнем шума"
enabled_detectors:
  - blocklist
  - allowlist
  - cve
  - secrets
  - network
  - executable
  - compression
severity_threshold: medium    # порог внимания: MEDIUM и выше
fail_on_severity: critical    # exit 1 только при CRITICAL
compliance: []
```

### `banking` — финансовый сектор

Строгий режим для банков и финансовых организаций. Соответствие ФСТЭК, ГОСТ.

```yaml
name: banking
enabled_detectors: [blocklist, allowlist, cve, secrets, network, executable, compression]
severity_threshold: low          # выше порога внимания — всё, кроме INFO
fail_on_severity: high           # exit 1 при HIGH и выше
compliance: [fstec, owasp_ml, gost_56939_2024]
extra_rules:
  no_external_urls: true         # любой внешний URL → HIGH
  no_unverified_files: true      # непроверенный файл → HIGH
  strict_format_detection: true  # формат ≠ расширение → HIGH (MLS-FMT-001)
  require_safetensors: true      # формат исполняет код → HIGH (MLS-FMT-002)
  max_file_size_gb: 50
```

### `government` — государственный сектор

Строгий режим с маппингом на БДУ ФСТЭК и ГОСТ Р 56939-2024. Для организаций под регуляторными требованиями.

```yaml
name: government
enabled_detectors: [blocklist, allowlist, cve, secrets, network, executable, compression]
severity_threshold: low
fail_on_severity: high
compliance: [fstec, gost_56939_2024, owasp_ml]
extra_rules:
  no_external_urls: true
  no_unverified_files: true
  strict_format_detection: true  # формат ≠ расширение → HIGH (MLS-FMT-001)
  require_safetensors: true      # формат исполняет код → HIGH (MLS-FMT-002)
  max_file_size_gb: 100
```

> Политики `research` в поставке нет: встроенных политик четыре — `default`,
> `banking`, `government`, `strict` (см. `poison-check doctor`). Мягкий режим
> для R&D делается пользовательским YAML по шаблону ниже.

### `strict` — максимальная строгость

Порог внимания на нуле (важно всё), гейт валится начиная с MEDIUM. Для production-систем с нулевой толерантностью.

```yaml
name: strict
enabled_detectors: [blocklist, allowlist, cve, secrets, network, executable, compression]
severity_threshold: info    # порога внимания фактически нет — важно всё
fail_on_severity: medium    # exit 1 при MEDIUM и выше
compliance: [fstec, owasp_ml, gost_56939_2024]
extra_rules:
  no_external_urls: true
  no_unverified_files: true
  strict_format_detection: true  # формат ≠ расширение → HIGH (MLS-FMT-001)
  require_safetensors: true      # формат исполняет код → HIGH (MLS-FMT-002)
  max_file_size_gb: 10
```

---

## Правила формата: `strict_format_detection` и `require_safetensors`

Оба ключа выключены в `default` и включены в `banking` / `government` /
`strict`. Оба относятся к формату файла, а не к его содержимому, поэтому
срабатывают даже на файле, в котором ничего вредоносного не нашли.

### `strict_format_detection` → `MLS-FMT-001` (HIGH)

Расширению файла не доверяем: формат определяется по содержимому (magic-байты
или структура заголовка). Если содержимое не совпадает с тем, что обещает
расширение, — это находка.

```text
Формат файла не совпадает с расширением: расширение '.safetensors' обещает
safetensors, а по содержимому это pickle (определено по признаку: опкод PROTO
(0x80) + номер протокола 2 по смещению 0).
```

Вектор атаки: вредоносный pickle называют `model.safetensors`, потребитель
видит «безопасный формат» и загружает файл небезопасным путём.

Граница: файл, формат которого определить **не удалось**, расхождением не
считается — это по-прежнему `MLS-PARSE-001` (сбой разбора) либо
«неподдерживаемый формат» в `FileResult.error`. Расширения, которые ничего не
обещают (`.bin`), из проверки исключены.

Ключ **не влияет** на выбор сканера — только на отчёт.

### `require_safetensors` → `MLS-FMT-002` (HIGH)

Допустимы только форматы, не исполняющие код при загрузке. Находка эмитится по
самому факту формата: чистый сегодня pickle остаётся исполняемой программой.

| Формат | Класс | Почему |
|---|---|---|
| safetensors | безопасный | JSON-заголовок + сырые байты тензоров |
| GGUF | безопасный | типизированные KV + тензоры, объекты не десериализуются |
| NumPy `.npy` / `.npz` | безопасный* | массивы фиксированного dtype читаются как байты |
| pickle | code-bearing | опкоды `GLOBAL` + `REDUCE` вызывают произвольную функцию |
| joblib | code-bearing | тот же pickle, скрытый компрессией |
| PyTorch `.pt`/`.pth`/`.ckpt` | code-bearing | ZIP + `data.pkl`, разбираемый pickle |
| Keras `.keras`/`.h5` | code-bearing | слои `Lambda` с python-байткодом (CVE-2025-1550) |
| ONNX | code-bearing | кастомные операторы подгружаются как нативные библиотеки |
| ZIP / TAR | не определён | контейнер общего назначения — класс задаёт содержимое |
| неопознанный | не определён | зона `MLS-PARSE-001`, а не политики формата |

\* `.npy` с `dtype=object` — **code-bearing**: такой массив NumPy сериализует
через pickle. Факт фиксирует `NumpyScanner`, класс формата понижается
автоматически.

Реестр классификации — `poison_check/scanners/format_facts.py`
(`FORMAT_SAFETY`); мета-тест не даёт добавить формат или сканер, не
классифицировав его.

### Карта «ключ → статус → точка применения → тест»

| Ключ | Статус | Точка применения | Тест |
|---|---|---|---|
| `strict_format_detection` | ENFORCED | `policies:policy_strict_format_detection` → `policies:detector_kwargs_for` → `FormatPolicyDetector.analyze` | `tests/test_format_policy.py::TestStrictFormatDetection` |
| `require_safetensors` | ENFORCED | `policies:policy_require_safetensors` → `policies:detector_kwargs_for` → `FormatPolicyDetector.analyze` | `tests/test_format_policy.py::TestRequireSafetensors` |

Обе находки — обычные находки политики, а не факты уровня файла: порог
внимания применяется к ним как ко всем прочим (HIGH не понижается никогда), но
в список «неподавляемых» они не входят — при них файл проверен полностью.

Находка по формату не подавляет находку по содержимому: pickle с `os.system`
под `banking` даёт и `MLS-PATTERN-OS-SYSTEM` (CRITICAL), и `MLS-FMT-002` (HIGH).

---

## Зарезервированный ключ: `require_model_signature`

Ключ объявлен в `government.yaml` со статусом **RESERVED** и помечен
`[not enforced yet]`. Что это означает практически:

* значение ключа **не влияет ни на что** — ни на состав находок, ни на severity,
  ни на exit code. `true` и `false` дают одинаковый результат сканирования;
* находок `MLS-SIG-*` сегодня не существует;
* отсутствие подписи у модели **не проверяется вообще** — ни в одной политике.

Дизайн проработан и зафиксирован в [`docs/design/model_signature.md`](design/model_signature.md):
модель угроз, формат подписи (отделённый подписанный манифест), хранение
доверенных ключей и офлайн-отзыв, размещение по слоям, коды и severity,
деградация при ненастроенном окружении, тест-план и поэтапный план внедрения.
Там же — итоговая рекомендация оставить ключ RESERVED до появления заказчика с
собственным процессом подписания, и условия пересмотра этого решения (§9.3).

Важное для тех, кто ждёт эту функцию: подпись подтверждает **происхождение и
целостность**, но не безопасность содержимого — доверенный поставщик может
подписать модель с Lambda-RCE. Поэтому в дизайне подпись заложена как
дополнительная ось, которая не подавляет ни одной находки по содержимому и не
отключает ни одного детектора.

---

## Использование политик

### CLI

```bash
# Именованная политика
poison-check scan model.pt --policy banking

# Кастомный YAML-файл политики
poison-check scan model.pt --policy /path/to/my_policy.yaml
```

### Python API

```python
from poison_check import Scanner

# Именованная политика
result = Scanner(policy="banking").scan("models/")

# Проверка compliance report
for tag in result.compliance_report.fstec_ubi:
    print(f"Требование ФСТЭК: {tag}")
```

---

## Создание кастомной политики

### Шаблон

Создайте YAML-файл в `policies/` или в любом другом месте:

```yaml
# policies/my_policy.yaml
name: my_policy
description: "Кастомная политика для [описание]"

# Список детекторов для включения.
# Допустимые значения: blocklist, allowlist, cve, secrets, network, executable, compression
enabled_detectors:
  - blocklist
  - cve
  - secrets
  - executable

# Порог ВНИМАНИЯ: находки ниже него остаются в отчёте, но помечаются
# подпороговыми. Допустимые значения: info, low, medium, high, critical
severity_threshold: medium

# Severity, при котором CLI возвращает exit code 1 (fail CI).
# Допустимые значения: info, low, medium, high, critical
fail_on_severity: high

# Compliance-требования для маппинга.
# Допустимые значения: fstec, owasp_ml, gost_56939_2024
compliance:
  - fstec
  - owasp_ml

# Дополнительные правила (опционально)
# Применяются только ключи из реестра EXTRA_RULE_KEYS
# (poison_check/policies.py). Незнакомый ключ CLI выведет предупреждением
# «ключи не распознаны и не применяются» и проигнорирует.
extra_rules:
  no_external_urls: false        # true = внешний URL → повышение severity
  no_unverified_files: false     # true = непроверенный файл → HIGH
  strict_format_detection: false # true = формат ≠ расширение → HIGH
  require_safetensors: false     # true = code-bearing формат → HIGH
  max_file_size_gb: 10           # лимит размера файла модели
```

### Использование кастомной политики

```bash
# Через CLI
poison-check scan model.pt --policy policies/my_policy.yaml

# Через Python API
from poison_check import Scanner
result = Scanner(policy="policies/my_policy.yaml").scan("model.pt")
```

---

## Как порог применяется к отчёту

Порог применяется на слое **Output**, а не в сканерах и детекторах: детектор
всегда возвращает находку, а политика управляет только тем, как эта находка
подаётся. Пороги едут вместе с результатом (`ScanResult.policy_thresholds`),
поэтому Python API даёт тот же отчёт, что и CLI, без дополнительных аргументов:

```python
from poison_check import Scanner
from poison_check.output.json_format import JsonFormatter

result = Scanner(policy="banking").scan("models/")
report = JsonFormatter().format(result)   # пометки порога уже применены
```

Собранный вручную `ScanResult` без `policy_thresholds` форматируется так же,
как до появления ключа: ни одна находка не помечается подпороговой.

Отфильтровать подпороговые находки в своём коде можно явно — но это уже
решение потребителя, а не инструмента:

```python
from poison_check.output.severity_threshold import partition_issues, threshold_of

threshold = threshold_of(result)
for path, file_result in result.results_per_file.items():
    primary, below = partition_issues(file_result.issues, threshold)
```

---

## Compliance-маппинг

Когда политика включает compliance-требования, каждый issue получает дополнительные теги:

```python
issue.compliance_tags  # ["owasp-ml:ml03", "fstec:ubi-067", "gost:56939-2024"]
issue.references       # [Reference(type="bdu", id="УБИ.067"), ...]
```

Маппинг реализован в:
- `poison_check/compliance/fstec_mapping.py` — ФСТЭК БДУ
- `poison_check/compliance/gost_mapping.py` — ГОСТ Р 56939-2024
- `poison_check/compliance/owasp_ml_top10.py` — OWASP ML Top 10

---

## Exit codes

| Код | Значение |
|---|---|
| `0` | Нет findings выше `fail_on_severity` |
| `1` | Найдены findings выше `fail_on_severity` |
| `2` | Ошибка сканирования (не удалось разобрать файл, неверные аргументы) |

Exit codes совместимы со стандартными SAST-инструментами и CI/CD-гейтами.

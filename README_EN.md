# poison-check

**English** · [Русский](README.md)

A static security scanner for ML model files: it inspects the model *and its bundle* without loading them.

---

## The problem

A model file is not just weights. The most common serialization format in the Python ecosystem is pickle (and the formats built on it — `.pt`/`.pth`, joblib, numpy object arrays). On load, pickle is not "read" but **executed**: the opcode stream can contain calls to `os.system`, `subprocess`, `eval`/`exec` and other primitives that fire the moment you call `torch.load` / `joblib.load` / `pickle.load` — before you do anything with the model. This is a classic RCE vector through the ML supply chain.

But checking the weights file alone is no longer enough. Modern models ship as a **bundle**: code lives next to the weights (`modeling_*.py`, `configuration_*.py`), and `config.json` points — via `auto_map` / `custom_pipeline` — to which of those files to import. When loaded with `trust_remote_code=True`, that code runs as ordinary Python, even if the weights themselves are in the safe `safetensors` format. The danger moves from the file into the directory, and a file scanner never sees it.

poison-check starts from both facts: it parses the structure of every model file **without loading it** (reading bytes only, walking pickle opcodes, parsing headers), and separately checks the model **bundle** — executable code next to the weights and the references to it from `config.json`. The tool reports the constructs it finds as facts for human review; it does not pass a "malicious" verdict on your behalf.

---

## Capabilities

Everything below matches the actual set of scanners and detectors (`poison-check list-scanners`, `poison-check doctor`).

### Supported formats

| Scanner | Extensions | What it parses |
|---|---|---|
| `pickle` | `.pkl`, `.pickle`, `.dill`, `.pt` | The pickle opcode stream (via `pickletools`, without executing it) |
| `pytorch` | `.pt`, `.pth`, `.bin`, `.ckpt` | The PyTorch ZIP container, nested pickles recursively |
| `joblib` | `.joblib`, `.pkl` | joblib streams, including zlib / lz4 / zstd compression |
| `numpy` | `.npy`, `.npz` | Magic bytes, object-dtype arrays, `.npy` inside `.npz` |
| `safetensors` | `.safetensors` | The JSON header and tensor metadata |
| `gguf` | `.gguf`, `.ggml` | GGUF/GGML metadata (llama.cpp, Ollama) |
| `keras` | `.keras`, `.h5`, `.hdf5` | Model architecture without loading keras/tensorflow |

When scanning a directory, the **model bundle** is checked additionally: code next to the weights and the references to it from `config.json` (`auto_map` / `custom_pipeline`).

### What it detects

Fourteen detectors extract the facts. The main vectors:

- **Dangerous pickle globals.** An `allowlist-first` model: anything outside the list of trusted globals is flagged as suspicious (`MLS-ALW-001/002`). Known dangerous globals and CVEs are handled separately (`MLS-PKL-*`, `MLS-CVE-*`).
- **Code-execution primitives in pickle.** `os.system` / `os.popen`, `subprocess.*`, `eval`, `exec` and others (`MLS-PATTERN-*`). Known CVEs come from rules in `rules/cve/` (for example, CVE-2025-1550 — Lambda RCE in Keras; CVE-2025-32434).
- **Executable code in the model bundle.** A `.py` file referenced by `config.json` (`trust_remote_code`) and the dangerous constructs in it, weighted by position — module-level / load point vs. an ordinary method (`MLS-BUNDLE-001…004`).
- **Embedded executables** inside the model (`MLS-EXE-001`).
- **Format disguise and code-bearing formats** — for example, a pickle masquerading as safetensors (`MLS-FMT-001/002`).
- **Decompression bombs** (`MLS-CMP-*`, `MLS-BOMB-001`).
- **Network indicators** — URLs and IP addresses in the content (`MLS-NET-*`).
- **Secrets** — API keys, tokens, passwords (`MLS-SEC-001`).
- **Format metadata** — GGUF (`MLS-GGUF-*`), joblib (`MLS-JOBLIB-*`), numpy object-dtype (`MLS-NPY-001`).
- **Parse errors** — a file that fails to parse, or is incomplete, is not lost but flagged (`MLS-PARSE-001`).

### Reports

Output formats (`--format`): `console` (default), `json` (schema v1.0), `sarif` (SARIF 2.1.0), `sbom` (CycloneDX 1.4), `html`, `pdf` (requires the `[pdf]` extra). Messages come in Russian or English (`--locale ru|en`). The tool runs offline: no network, no telemetry.

> **Output language.** `--locale en` produces an English CLI (labels, tables, summary) and English descriptions for CVE/PATTERN findings (`MLS-CVE-*`, `MLS-PATTERN-*`) — the most common pickle-RCE vectors. Descriptions from the remaining detectors (bundle, network, secrets, GGUF, and so on) are currently in Russian; full English localization is in progress.

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/rainlessx/poison-check.git
cd poison-check
pip install .
```

The required dependencies (including `lz4`, `zstandard`, `h5py`) are installed automatically — format support does not depend on what happens to be in your environment.

Optional extras:

```bash
pip install ".[pdf]"   # PDF reports (Jinja2 + WeasyPrint, pulls in system libraries)
pip install ".[7z]"    # 7z container extraction (py7zr)
```

Check the environment and the set of rules/scanners:

```bash
poison-check doctor
```

---

## Usage

### Commands

```bash
poison-check scan <path>        # scan a file or a directory
poison-check list-scanners      # list supported formats
poison-check doctor             # diagnose the environment, rules, policies
```

### Main `scan` flags

| Flag | Purpose |
|---|---|
| `--policy <name\|path>` | Policy: `default`, `banking`, `government`, `strict`, or a path to your own YAML |
| `--format <console\|json\|sarif\|sbom\|html\|pdf>` | Report format |
| `--output <file>` | Save the report to a file (works for all formats) |
| `-r`, `--recursive` | Descend into subdirectories (off by default) |
| `--no-bundle-check` | Fully disable the model-bundle check (`MLS-BUNDLE-*`) |
| `--locale <ru\|en>` | Output language |
| `--max-file-size <GB>` | Override the file-size limit |
| `--no-emoji` | ASCII icons `[!]/[*]/[i]` instead of emoji |
| `-v`, `--verbose` | Per-file globals breakdown on stderr (false-positive diagnostics) |
| `--client`, `--auditor` | Details for the report header |
| `--audit-log <file>` | Append one JSON Lines record per scan |

Full list: `poison-check scan --help`.

### Exit codes

- `0` — no findings at or above the action threshold.
- `1` — at least one finding with severity `>=` the selected policy's `fail_on_severity`. Findings below the threshold appear in the report but do not fail the gate.
- `2` — scan error (the file could not be parsed, etc.).

### Example: a pickle with a system call

```console
$ poison-check scan ./payload.pkl --locale en --no-emoji
╭────────────────────────────╮
│ Starting scan: payload.pkl │
╰────────────────────────────╯
Scanning file: payload.pkl

─────────────────────────────── payload.pkl ────────────────────────────────
[!] [CRITICAL] MLS-PATTERN-OS-SYSTEM — os.system/os.popen call detected in
pickle stream
  Location: payload.pkl:offset 15
  Why dangerous: The pickle stream contains a call to os.system or os.popen,
which allows executing arbitrary operating-system commands when the file is
loaded. This is a classic RCE (Remote Code Execution) vector through ML
files, observed in real malicious models on HuggingFace (ReversingLabs
report, 2023).
  Remediation: Do not load this file. Ask the source for a safetensors or
ONNX file. Verify the model's origin for signs of compromise.
  References: CWE:CWE-502, CWE:CWE-78

    Scan summary
┏━━━━━━━━━━┳━━━━━━━┓
┃ Severity ┃ Count ┃
┡━━━━━━━━━━╇━━━━━━━┩
│ CRITICAL │     1 │
│ HIGH     │     0 │
│ MEDIUM   │     0 │
│ LOW      │     0 │
│ INFO     │     0 │
└──────────┴───────┘
Files scanned: 1
Scan completed in 165 ms
⚠️  Issues found: 1
🚨 CRITICAL: 1 critical vulnerabilities
```

Real output of `poison-check scan ./payload.pkl --locale en --no-emoji` (terminal width 76). The only non-deterministic field is the timing line (`Scan completed in N ms`).

For the **model-bundle** case (safe `safetensors` weights + a `modeling.py` with a module-level `exec` referenced from `config.json`), scanning the directory raises `MLS-BUNDLE-002` (HIGH) — a file scanner looking only at the weights would miss it. Under the `default` policy (`fail_on_severity: critical`) that finding still exits `0`; under `strict` (`fail_on_severity: medium`) it exits `1`.

### Python API

```python
from poison_check import Scanner

result = Scanner(policy="default", locale="en").scan("model.pt")

if result.has_critical:
    print(f"Critical findings: {result.summary.critical}")

for path, file_result in result.results_per_file.items():
    for issue in file_result.issues:
        print(f"[{issue.severity.value}] {path.name}: {issue.message}")
        if issue.decompiled_code:
            print(f"  Code: {issue.decompiled_code}")
```

`Scanner(policy=..., locale=...)`; methods `.scan(path, recursive=False)` and `.scan_bytes(data, filename=...)`.

---

## Severity levels

The severity level is determined by the class of the detected construct and its position:

| Level | Meaning |
|---|---|
| `CRITICAL` | A direct code-execution construct detected at a position where it fires on load — maximum RCE potential |
| `HIGH` | A dangerous construct detected with high exploitation potential |
| `MEDIUM` | A suspicious construct that needs manual review |
| `LOW` | Low risk potential, informational |
| `INFO` | For completeness |

Severity reflects the *potential* danger of the detected construct and its position, not *proven* exploitability — the final verdict is the reviewer's.

Policies operate with **two** independent thresholds, and this matters:

- `severity_threshold` — the **attention threshold**: from which level a finding is shown prominently in the report (lower ones go to a "below the attention threshold" section but are not dropped).
- `fail_on_severity` — the **action threshold**: from which level the scan exits with code `1` (a CI gate failure).

Built-in policies:

| Policy | `severity_threshold` | `fail_on_severity` | Purpose |
|---|---|---|---|
| `default` | medium | critical | A standard balance of coverage and noise |
| `banking` | low | high | Financial organizations, elevated strictness |
| `government` | low | high | Public sector, emphasis on FSTEC compliance |
| `strict` | info | medium | Maximum strictness for production CI/CD |

You can define your own policy in a YAML file and pass it via `--policy path.yaml`.

---

## Boundaries and principles

These boundaries are a deliberate choice, not a shortcoming. A security tool must be precise about what it does and does not do.

- **Static analysis only. The model's code is never executed.** Parsing goes through a `pickletools` opcode walk and structure/AST parsing. `pickle.load`, `torch.load`, `joblib.load`, and `eval` are never called on user data.
- **It detects the presence of a dangerous construct — it does not prove maliciousness.** This is not a SAST tool with data-flow analysis: it sees that, say, `requests` or `exec` is present in the load-time code, but it does not prove the call is malicious. A legitimate model may legally reach out to the network on load. The final verdict is made by a human at review — messages are phrased as facts, not a sentence.
- **Fail-closed on unsupported or unreadable input.** A file that could not be parsed (unknown format, corruption, a partial download, a missing dependency) does not pass as "clean": it is flagged with a dedicated finding (`MLS-PARSE-001` and related). "Not checked" never looks like "safe".
- **Offline.** No network, no calls to external services, no telemetry.

What the tool deliberately does **not** do:

- it does not execute or sandbox the model — no dynamic analysis;
- it does not prove exploitability and does not build data-flow;
- it does not guarantee detection of every possible threat — it is one supply-chain check, not a replacement for manual review of untrusted code;
- it does not "fix" models or strip findings from files.

---

## License

Distributed under the **PolyForm Noncommercial License 1.0.0**.

Noncommercial use — evaluation, testing, research, personal and educational projects — is **freely permitted**. Any commercial use requires a separate commercial license.

For commercial licensing: lawnmover58@gmail.com.

The full license text is in the [LICENSE](LICENSE) file. Copyright © 2026 Artem Vorobev.

---

## Feedback

Pull requests are **not accepted** — the project is developed by a single author. But **issues are welcome**, especially reports of hits on real models. A useful report includes:

- the model (link or description) and the file format;
- what the scanner said (finding code, severity, message);
- what you expected (a false positive, or conversely a miss).

Such reports are especially valuable: they directly improve detector accuracy.

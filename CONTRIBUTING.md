# Contributing to TinyRAG

Thank you for your interest in TinyRAG! This document explains how to contribute, what we expect from contributors, and how we keep the project healthy.

> **Note for the capstone author:** This file is here both for you (so you have professional-grade contribution hygiene as you build it) and for any external contributors who find the repo useful.

---

## 1. Code of Conduct

All participants are expected to follow our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). In short: be respectful, be constructive, assume good faith.

---

## 2. Project Structure & Where to Contribute

```
tinyrag/
├── src/tinyrag/         ← production source code
│   ├── ingestion/       ← PDF → chunks → embeddings → FAISS
│   ├── retrieval/       ← query → top-k chunks
│   ├── generation/      ← prompt + retrieved → answer (talks to llama.cpp)
│   ├── sensors/         ← SensorSource protocol + concrete impls
│   ├── storage/         ← FAISS + SQLite
│   ├── api/             ← FastAPI routes
│   └── ui/              ← static web assets
├── tests/               ← pytest unit tests (mirror src/ layout)
├── scripts/             ← operational scripts (ingest, evaluate, benchmark)
├── docs/                ← planning + design docs
└── reports/             ← generated outputs (gitignored)
```

**Where your change goes:**

| If your change... | Put it in... |
|-------------------|--------------|
| Adds a new module under src/ | `src/tinyrag/<package>/` + matching `tests/test_<package>.py` |
| Adds a CLI script | `scripts/<verb>_<noun>.py` |
| Adds a planning doc | `docs/` (and update `AGENT.md` roadmap table) |
| Updates a dependency | bump `pyproject.toml` AND `requirements.txt`; justify in commit |
| Touches a Protocol interface | update **both** the Protocol and ALL concrete implementations + tests |

---

## 3. Development Setup

```bash
# Clone
git clone https://github.com/marajul/tinyrag.git
cd tinyrag

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install (deps + dev tools)
pip install -r requirements.txt
pip install -r requirements-dev.txt  # when present

# Run tests
pytest

# Lint
ruff check src tests
ruff format src tests
```

---

## 4. Workflow

We follow a **trunk-based development** model with short-lived feature branches:

```
main  ───●─────────────●──────────────►  (always green)
           \           /
            ●─────●───●  feat/<name>     (short-lived, ≤ 1 week)
```

1. **Branch from `main`** using the naming convention below.
2. **Make atomic commits** — one logical change per commit.
3. **Write tests first or alongside** — every new module ships with `tests/test_<name>.py`.
4. **Run `pytest` and `ruff`** before pushing.
5. **Push your branch** and open a Pull Request.
6. **Merge to `main`** once CI is green and you've self-reviewed.

### 4.1 Branch naming

| Prefix | Use case | Example |
|--------|----------|---------|
| `feat/` | New feature | `feat/sensor-mqtt-source` |
| `fix/` | Bug fix | `fix/retriever-empty-query` |
| `docs/` | Documentation only | `docs/update-roadmap-phase-5` |
| `refactor/` | Code restructuring, no behavior change | `refactor/protocol-sensor-source` |
| `test/` | Add or improve tests | `test/chunker-fixtures` |
| `chore/` | Tooling, deps, configs | `chore/bump-llama-cpp-2026-07` |

### 4.2 When to open a Pull Request

- The change is **complete** (not "WIP, please review").
- All tests pass locally.
- `ruff check` is clean.
- Commit messages follow §5.

> For solo capstone work, the student may commit directly to `main` if preferred — but feature branches are still recommended for non-trivial changes so you can review your own diff before merging.

---

## 5. Commit Message Convention

We use **Conventional Commits** (https://www.conventionalcommits.org/). This makes the git log greppable, enables automatic changelog generation, and signals intent.

### 5.1 Format

```
<type>(<scope>): <short summary in imperative mood>

<body — explain WHY, not what (the diff shows what)>

<footer — references, breaking changes, etc.>
```

### 5.2 Types

| Type | Use for |
|------|---------|
| `feat` | New user-facing feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting (no code change) |
| `refactor` | Code restructuring, no behavior change |
| `test` | Add or fix tests |
| `chore` | Tooling, deps, configs |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |

### 5.3 Scopes (for this project)

`ingestion`, `retrieval`, `generation`, `sensors`, `storage`, `api`, `ui`, `llamacpp`, `docs`, `eval`, `setup`.

### 5.4 Examples

```
feat(retrieval): add optional cross-encoder reranker

The reranker is gated behind the new `retrieval.rerank.enabled`
flag in config.yaml so it stays off by default on the Pi.

Closes #42
```

```
fix(sensors): handle missing columns in SimulatedCSVSource

Previously raised KeyError if the synthetic CSV omitted the
`motion` column. Now degrades gracefully and emits a warning.

Refs docs/03_architecture_v1.md §4
```

```
docs(roadmap): mark Step 3.1 complete in AGENT.md
```

---

## 6. Pull Request Checklist

Before opening a PR, confirm:

- [ ] Tests pass: `pytest -q`
- [ ] Lint passes: `ruff check src tests`
- [ ] No new files committed that should be gitignored (models, data, logs, .env)
- [ ] If you changed a Protocol, all implementations are updated
- [ ] If you added a new dependency, `pyproject.toml` AND `requirements.txt` are updated
- [ ] Commit messages follow Conventional Commits
- [ ] `AGENT.md` is updated if you changed project status / decisions

---

## 7. Reporting Bugs

Open a GitHub Issue with:

1. **What you did** (the exact command / query)
2. **What you expected** (the correct behavior)
3. **What happened** (the actual behavior, including the full error message)
4. **Environment** (`python --version`, OS, model being used)
5. **Reproducibility** (does it happen every time? on first run? after X?)

---

## 8. Security Issues

**Do not open a public issue for security vulnerabilities.** See [`SECURITY.md`](SECURITY.md) for responsible disclosure.

---

## 9. License

By contributing, you agree that your contributions will be licensed under the project's MIT license. See [`LICENSE`](LICENSE).

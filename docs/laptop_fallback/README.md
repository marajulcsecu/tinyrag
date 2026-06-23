# Laptop Fallback Documentation

> **Purpose:** This directory contains all information needed to run, develop, and deploy TinyRAG on a regular laptop when the Raspberry Pi 5 is unavailable. The same code, models, and architecture apply — only the deployment configuration changes.

---

## When to Use This Path

Use the **laptop path** when:
- ❌ The Raspberry Pi 5 has not been provided by the lab
- ❌ The Pi 5 is damaged or unavailable for an extended period
- ❌ You need a faster development loop (laptop compiles llama.cpp in minutes vs. hours on Pi)
- ❌ You want to do initial development before the Pi arrives

**Rule of thumb:** Develop and test on the laptop. Deploy to the Pi only in the final 1–2 weeks.

---

## Target Machine Profile

| Spec | Value |
|------|-------|
| Model | Dell Inspiron 15 3520 |
| CPU | Intel Core i5-1235U (12th gen, 10 cores: 2P + 8E, up to 4.4 GHz) |
| RAM | 8 GB DDR4-3200 |
| Storage | 512 GB NVMe SSD |
| GPU | Integrated Intel Iris Xe (not used for LLM — too weak for serious LLM inference) |
| OS | **Ubuntu 24.04.4 LTS** (confirmed) — Wayland, GNOME 46, kernel 6.17.0-35-generic |

> **OS confirmed.** Native Ubuntu 24.04 — no WSL2 needed. We get full Linux performance, easy llama.cpp build, and direct browser access at `http://localhost:8000/`.

---

## Why the Laptop is a Good Fallback

| Aspect | Raspberry Pi 5 | Dell Inspiron |
|--------|---------------|---------------|
| CPU | 4× Cortex-A76 @ 2.4 GHz | 10× Intel hybrid (2P@4.4GHz + 8E@3.3GHz) |
| RAM | 8 GB LPDDR4X | 8 GB DDR4 |
| Memory bandwidth | ~17 GB/s | ~50 GB/s |
| LLM tokens/sec (Phi-3 Q4) | ~8–12 tok/s | ~25–35 tok/s |
| Storage | microSD (~80 MB/s) | NVMe SSD (~2000 MB/s) |
| Power draw | ~5W idle, ~15W load | ~10W idle, ~45W load |
| Boot/setup time | Slow (especially llama.cpp compile) | Fast |

**Translation:** The laptop is **3–4× faster** for LLM inference than the Pi. Same code, same models, same UI.

---

## What Stays the Same (vs. Pi)

- ✅ All source code in `src/tinyrag/`
- ✅ All model files (Phi-3, TinyLlama, Llama 3.2)
- ✅ All config keys — only the *values* differ in `config.yaml`
- ✅ All documentation
- ✅ All test scripts
- ✅ Evaluation methodology and gold-set

## What Differs (vs. Pi)

| Aspect | Pi 5 path | Laptop path |
|--------|-----------|-------------|
| OS | Raspberry Pi OS 64-bit | Ubuntu 22.04 / WSL2 |
| llama.cpp build flags | `-DGGML_OPENBLAS=OFF -mcpu=cortex-a76` | `-DGGML_OPENBLAS=ON -march=native` (uses Intel MKL/OpenBLAS for ~2× speedup) |
| Sensor I/O | Real GPIO (DHT22 via libgpiod) | Simulated only (no GPIO on laptop) |
| Model serving | llama.cpp HTTP server on `localhost:8080` | Same |
| Web UI access | `http://<pi-ip>:8000/` from any LAN device | `http://localhost:8000/` from same machine |
| systemd service | Yes (auto-start on boot) | No (run via `run.sh` or background process) |
| Power-loss resilience | Needs read-only filesystem or UPS | Not a concern |

---

## Setup Differences

The `setup.sh` script will detect the platform at runtime and adapt. Roughly:

```bash
# On the laptop (Ubuntu 22.04)
sudo apt install -y python3.10 python3-pip python3-venv build-essential cmake
# llama.cpp build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build -DGGML_OPENBLAS=ON && cmake --build build --config Release -j
# Python deps
pip install -r requirements.txt
# Models (same as Pi)
python scripts/download_models.py
```

## Performance Expectations on the Laptop

| Metric | Expected on Laptop | Expected on Pi 5 |
|--------|-------------------|------------------|
| First-token latency | 0.8 – 1.5 s | 1.5 – 3.0 s |
| End-to-end (200-token answer) | 2 – 4 s | 4 – 7 s |
| Idle RAM after model load | 1.0 GB | 1.5 GB |
| Peak RAM during inference | 2.0 GB | 3.0 GB |
| Embedding throughput | 200+ chunks/sec | 50+ chunks/sec |

These numbers will be measured and reported in the final benchmarks.

---

## Files in This Directory

- `README.md` — this file
- `02_srs_v1.md` — laptop-fallback mirror of the SRS (with laptop-specific notes prepended)
- `setup_laptop.md` — laptop-specific setup steps (to be written)
- `config_laptop.yaml` — reference config file tuned for the laptop (to be written)
- `performance_notes.md` — measured vs. expected performance (to be written)

> **Duplication rule:** every main planning doc (scope, SRS, architecture, DB, tech stack, roadmap) is mirrored here with a laptop-specific banner. If the Pi path diverges in implementation detail, the laptop path is the documented fallback. The canonical version always lives in `docs/`.

---

## When You Eventually Get the Pi

Switching from laptop to Pi is **a config change, not a code change**:

1. Copy `src/`, `data/`, `models/`, `config.yaml` to the Pi.
2. Change `config.yaml`:
   - `deployment.target: raspberry_pi`
   - `llama.cpp.openblas: false`
   - `sensors.source: real_serial` (if lab sensor is connected)
3. Run `setup_pi.sh` (different from `setup_laptop.sh`).
4. Done.

This is the **clean architecture** payoff — the laptop and Pi paths are decoupled.

---

*End of laptop_fallback README. Detailed setup scripts to be added once the student confirms their OS.*

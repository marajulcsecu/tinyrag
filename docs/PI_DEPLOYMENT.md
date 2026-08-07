# Raspberry Pi 5 Deployment

Field notes from the first real deployment (2026-08-04). TinyRAG runs on a
Raspberry Pi 5 (8 GB) end to end: retrieval, generation, and citation, with
no cloud calls.

The Pi uses the **same `setup.sh` and `run.sh` as the laptop** — there is no
Pi-specific code path. Everything below is environment setup and the
gotchas that cost time, recorded so the next deployment does not rediscover
them.

---

## 1. Verified hardware and OS

| | |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 8 GB |
| Storage | 64 GB microSD (~12 GB used after full install) |
| OS | Debian 13 (trixie), 64-bit, kernel 6.18 |
| Arch | `aarch64`, 4 cores |

---

## 2. Python 3.13 is a blocker — install 3.12 first

**This is the one step that will stop you.** Debian 13 ships **Python 3.13
only**, and the pinned dependency versions have no 3.13 wheels:

| Package | Pinned | Highest wheel |
|---|---|---|
| `torch` | 2.4.1 | cp312 |
| `numpy` | 1.26.4 | cp312 |

Without 3.12, `pip` falls back to building torch from source — hours of
compilation on 4 ARM cores, and likely an out-of-memory failure.

Upgrading the pins instead does **not** work: as of this writing
`faiss-cpu`, `PyMuPDF`, and `sentence-transformers` publish **no aarch64
cp313 wheels at any version**. Python 3.12 is the only path today.

`apt` cannot help — trixie has 3.13, bookworm has 3.11. Use a prebuilt
standalone CPython instead (no compilation, ~80 MB):

```bash
mkdir -p ~/.local/opt && cd ~/.local/opt
curl -fsSL -o py312.tar.gz \
  "https://github.com/astral-sh/python-build-standalone/releases/download/20260728/cpython-3.12.13+20260728-aarch64-unknown-linux-gnu-install_only.tar.gz"
tar -xzf py312.tar.gz
~/.local/opt/python/bin/python3.12 --version    # Python 3.12.13
```

Then create the venv from it **before** running `setup.sh`, which detects
the existing venv and installs into it rather than building a new one from
system Python:

```bash
~/.local/opt/python/bin/python3.12 -m venv ~/venvs/tinyrag
```

System Python 3.13 is left untouched. Verify the wheels resolve as
prebuilt binaries:

```bash
~/venvs/tinyrag/bin/python -m pip install --dry-run --no-deps \
  torch==2.4.1 numpy==1.26.4 faiss-cpu==1.8.0.post1 PyMuPDF==1.24.10
```

Every line should say `...aarch64.whl`. If anything mentions building a
wheel, stop — the interpreter version is wrong.

**Why keep the pins rather than upgrade them:** the laptop and Pi then run
byte-identical dependency versions, so evaluation results are directly
comparable. That comparability is the point of the Phase 5 numbers.

---

## 3. Full install

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
    libopenblas-dev liblapack-dev sqlite3

git clone https://github.com/marajulcsecu/tinyrag.git
cd tinyrag
bash setup.sh      # ~30-45 min: llama.cpp compile + 1.9 GB model
bash run.sh
```

Set the deployment target in `config.yaml`:

```yaml
deployment:
  target: raspberry_pi
```

`setup.sh` is idempotent — if it is interrupted, re-run it.

### The corpus does not come with the clone

`data/` is gitignored, so a fresh Pi starts with an empty index. Rebuild it
per [`data/evaluation/corpus/README.md`](../data/evaluation/corpus/README.md).
Ingestion **must** use `--embedder real`; the fake embedder produces
SHA-256 vectors that are not comparable to query vectors, and retrieval
silently returns noise while still looking like it works.

Expect 209 doc chunks, matching the laptop exactly.

---

## 4. Reaching the UI from another machine

Both services bind `127.0.0.1`, so a headless Pi is not reachable by
default. Prefer an SSH tunnel — no configuration change, nothing exposed:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<pi-ip>
# then open http://127.0.0.1:8000/ on your own machine
```

`API_HOST=0.0.0.0 bash run.sh` also works on a trusted network, but the UI
has **no authentication** — anyone on the network can use it and read your
documents. Leave `LLM_HOST` at `127.0.0.1` regardless.

For a graphical desktop, enable VNC with `sudo raspi-config nonint do_vnc 0`.
Note that **Remmina fails to connect**: Pi OS uses `wayvnc`, which requires
TLS/RSA-AES that Remmina's VNC plugin does not negotiate. Use TigerVNC
(`sudo apt install tigervnc-viewer`), which authenticates via PAM with the
normal Pi login.

---

## 5. Measured performance

First query on the Pi, 1730-token prompt:

| Stage | Time | Rate |
|---|---|---|
| Prompt eval | 70.3 s / 1730 tok | 24.6 tok/s |
| Generation | 7.7 s / 28 tok | 3.6 tok/s |
| **Total** | **78 s** | |

The laptop answers the same class of query in 15–25 s, so the Pi is roughly
**3–5× slower** — expected for Cortex-A76 versus a 12-thread i5.

**Prompt processing is ~90% of the total.** Anything that shrinks the prompt
helps far more here than on the laptop, which is why the 0.15 relevance
floor from the pre-4.25 verification pass matters doubly on this hardware.

Memory is not a constraint: **487 MB resident of 8 GB**. A 4 GB board would
likely also work, though it has not been tested. A later run over the larger
2,983-chunk corpus measured 545 MB, so corpus size barely moves this.

**These timings assume the default 4 threads.** On an inadequate supply the
board must be throttled to `LLAMA_THREADS=2` to stay up, which roughly
doubles latency to ~172 s for an 1,800-token prompt. See
[`PI_POWER_RUNBOOK.md`](PI_POWER_RUNBOOK.md).

Verified working end to end:

> **Q:** What processor does the Raspberry Pi 5 use?
> **A:** The Raspberry Pi 5 uses a Broadcom BCM2712 2.4GHz quad-core 64-bit
> Arm Cortex-A76 CPU.
> — cites `raspberry-pi-5-product-brief.pdf` p.2, score 0.807

---

## 6. Things that went wrong (and are now fixed)

Recorded because each cost real time and the fixes are easy to lose.

**`setup.sh` reported "deps verified" with a broken install.** A power cut
mid-`pip install` left torch missing entirely and numpy segfaulting on
import — yet the four sentinel packages the guard probed were all small
pure-Python packages that had installed fine. It skipped the repair and the
failure surfaced much later. Fixed by adding `numpy` and `torch` to the
sentinel list: the large binary wheels are precisely the ones an
interruption leaves half-written.

**A stale `.git` broke the llama.cpp build.** An interrupted clone left a
48 KB `.git` in `llama.cpp/` whose config still pointed at the TinyRAG
remote. The build script checked only that *some* origin existed, then
failed with `couldn't find remote ref master` — and the error handler
blamed the build toolchain. Fixed by verifying the origin URL actually
names llama.cpp, and re-cloning when it does not.

**HTTP 403 on the Pi 5 product brief.** `datasheets.raspberrypi.com`
rejects urllib's default `Python-urllib/3.x` user-agent. Fixed by sending a
normal User-Agent header.

**`setup.sh` failed when invoked by absolute path.** It computed
`REPO_ROOT` but never `cd`'d there, so `make` ran in the caller's directory.
Fixed in `c5bdb99` before this deployment.

---

## 7. If the Pi reboots unexpectedly

**This is now diagnosed and measured — see
[`PI_POWER_RUNBOOK.md`](PI_POWER_RUNBOOK.md).** The suspicion recorded here
was correct: the supply sags under sustained multi-core load, dropping the
5 V input to ~4.96 V against a 4.63 V trip point. Read that document for the
rail measurements, the `LLAMA_THREADS=2` workaround, safe large-document
ingestion, and the PSU migration steps.

First checks:

```bash
vcgencmd get_throttled    # 0x0 = healthy; bit 16 = undervoltage occurred
vcgencmd measure_temp     # throttles at 80°C
uptime -p                 # low uptime = it crashed, even if it pings fine
```

Ours consistently read `0x0` at ~47 °C with no OOM kills and no filesystem
errors — meaning power vanished with **no warning to the OS at all**, which
rules out both undervoltage detection and thermal shutdown. Later
confirmation: a phone adapter reproduced the crash immediately, and the
board also died while completely idle. Use the official 27 W supply.

Enable persistent journaling so the next crash leaves evidence behind. Note
that on Raspberry Pi OS a vendor drop-in at
`/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf` sets
`Storage=volatile` and **silently wins** over edits to
`/etc/systemd/journald.conf`, so override it with a higher-numbered drop-in:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\n' | \
  sudo tee /etc/systemd/journald.conf.d/99-persistent-storage.conf
sudo systemctl restart systemd-journald
journalctl --list-boots    # more than one line means it is working
```

Good news: ext4 and SQLite (WAL) survived every one of these cleanly.
Re-running `setup.sh` and re-ingesting was always sufficient.

**Correction to an earlier claim:** this section previously stated the FAISS
index never showed a dangling or orphaned vector afterwards. That was wrong
— a later audit found 4 vectors in `doc.faiss` with no UUID mapping in the
metadata store, which can surface as retrieval hits with no retrievable
text. Re-ingesting on top of a damaged index preserves those orphans; only a
clean rebuild removes them. Always run the three-store consistency check in
[`PI_POWER_RUNBOOK.md`](PI_POWER_RUNBOOK.md) §4 after a crash.

# Raspberry Pi 5 Power and Operations Runbook

What to do when the Pi shuts down under load, how to run TinyRAG safely on
an inadequate supply, and what changes once a proper PSU arrives.

Written 2026-08-07 after diagnosing repeated hard shutdowns while ingesting
a 1,313-page PDF. This supersedes the guesswork in
[`PI_DEPLOYMENT.md`](PI_DEPLOYMENT.md) §7 — that section suspected the
supply; this one measures it.

**The short version:** it was never the tool, the document size, or the
code. The supply sags under sustained multi-core load. Thread count is the
lever that controls it.

---

## 1. How to start TinyRAG on the Pi

**On the current (inadequate) supply — use this:**

```bash
cd ~/tinyrag && LLAMA_THREADS=2 bash run.sh
```

**After the official 27 W PSU is installed:**

```bash
cd ~/tinyrag && bash run.sh
```

The `LLAMA_THREADS=2` is not optional on a weak supply. Without it `run.sh`
starts llama-server with `--threads 4`, all four cores saturate during
generation, and the board dies mid-answer.

| | Weak PSU (`LLAMA_THREADS=2`) | Official 27 W (default 4) |
|---|---|---|
| Answer latency | ~172 s | ~78 s |
| Stability | Survives, ~0.35 V margin | Full headroom |
| Config needed | `timeout_s: 600` in `config.yaml` | Default fine |

---

## 2. Root cause: the supply sags under sustained load

The Pi 5 PMIC exposes per-rail voltage and current. Sampling it once per
second through a bounded 4-core load gives the whole story:

| State | 5 V input | Core rail | Temp |
|---|---|---|---|
| Idle | **5.30 V** | 0.57 A @ 0.75 V | 48 °C |
| 4 cores pinned | **4.96 V** (min) | 5.78 A @ 0.88 V | 75 °C |
| 2 cores pinned | **4.97 V** (min) | 5.50 A @ 0.88 V | 69 °C |

Read the rails yourself with:

```bash
vcgencmd pmic_read_adc | grep -E "EXT5V_V|VDD_CORE"
```

The Pi 5's documented undervoltage trip is **4.63 V**. Under sustained load
the input sits around 4.96–4.98 V, leaving roughly **0.33 V of margin** —
and that was measured with no SD writes, no Wi-Fi bursts, and no fan ramp.
A real ingest does all three *at the same time* as the pinned cores. Any
transient on top of that steady-state sag punches through 4.63 V, the PMIC
cuts the rail instantly, and you get the red light.

### Why this explains every symptom

- **Only large PDFs failed.** A 2,735-chunk book means ~5 minutes of pinned
  cores plus continuous SD writes. A 100–120 page document finishes in
  20–30 s with far fewer transients, so it squeaks through. It was never a
  page limit — it was *exposure time under load*.
- **Death was instant, with no warning.** A recorder writing `fsync`'d
  samples every 0.5 s showed healthy 5.30 V / 48 °C right up to the final
  sample, then nothing. Zero degraded samples. Software faults do not do
  that; a supply collapse does.
- **It crashed while completely idle.** No workload can cause that. A
  marginal adapter dropping out can.
- **It crashed during an unrelated `apt` update.** Different workload, same
  profile — sustained CPU plus heavy SD writes — same silent death.
- **`get_throttled` read `0x0` throughout, with no OOM kills and no thermal
  event.** The OS never saw it coming because the OS was never involved.
- **A phone adapter reproduced it immediately.** Phone chargers target
  5.0 V rather than the 5.1 V the Pi 5 expects, and have poor transient
  response — they sag exactly when a burst load arrives.

Temperature is a secondary concern, not the cause. 75 °C at 45 s and still
climbing means a 5-minute ingest would reach the 80 °C throttle point. But
throttling only slows the Pi, and a genuine thermal shutdown is *clean* and
leaves logs. These crashes left none.

---

## 3. What 4 threads does, and why 2 fixes it

`run.sh` defaults llama-server to `--threads 4` — one per core. That is the
right default on a healthy supply and the wrong one here.

Current draw scales with how many cores are simultaneously active. Four
pinned cores pull 5.78 A on the core rail; two pull 5.50 A while running at
a higher boost clock, so the saving is smaller than "half" but the
**transient profile** is what matters. Two cores leave slack for the SD
controller, Wi-Fi radio, and fan to draw their bursts without the combined
total crossing the trip point.

The evidence is clean, because the two workloads differ only in core count:

| Workload | Threads | Duration | Result |
|---|---|---|---|
| Cormen ingest | 2 | ~10 min sustained | **Survived**, 4.97 V floor |
| Query / generation | 4 | ~5 min | **Hard crash** |
| Query / generation | 2 | 172 s | **Survived**, 4.98 V floor |

Set it per-run as an environment variable rather than editing `run.sh`, so
the laptop keeps its full-speed default:

```bash
LLAMA_THREADS=2 bash run.sh
```

### The GUI upload path ignores LLAMA_THREADS

Important caveat. `LLAMA_THREADS` controls llama-server only. Uploading
through the admin GUI spawns its **own** embedder inside the API process,
which is not covered by that variable and will run at full speed. Until the
PSU is replaced, ingest large documents with the throttled CLI command in §4
instead of the GUI.

---

## 4. Ingesting a large document on a weak supply

Pin the ingest to two cores with `taskset` and run it detached so an SSH
drop cannot kill it mid-embed:

```bash
cd ~/tinyrag && source ~/venvs/tinyrag/bin/activate
setsid nohup taskset -c 0,1 python3 scripts/ingest.py \
  ~/Downloads/"Cormen Introduction to Algorithms.pdf" \
  --embedder real --doc-type manual --quiet \
  > ~/ingest.log 2>&1 < /dev/null &
```

Confirm the pin actually took effect — 210 % CPU is exactly two saturated
cores, and `Cpus_allowed_list` should read `0,1`:

```bash
top -b -n1 | grep ingest
grep Cpus_allowed_list /proc/$(pgrep -f scripts/ingest.py)/status
```

Watch the rails while it runs. Anything at or below 4.7 V is a warning sign:

```bash
watch -n1 'vcgencmd pmic_read_adc | grep EXT5V_V; vcgencmd measure_temp'
```

**Verified throughput at 2 cores:** 1,313 pages → 2,735 chunks in ~8 min.
The 262-page Data Structures book took 39 s for 235 chunks. Peak RSS was
~490 MB of 8 GB, so memory was never close to a constraint.

**Do not split large PDFs.** There is no page limit in the code — the only
hard cap is the 50 MB upload size. Splitting treats a symptom and
fragments your citation page numbers.

### Verify all three stores agree afterwards

This is the check that matters most, and the one that caught a real bug.
The Documents table reads SQLite; the admin panel's "DOC CHUNKS" reads the
live FAISS index. They can disagree:

```bash
cd ~/tinyrag && source ~/venvs/tinyrag/bin/activate && python3 - <<'EOF'
import faiss, json, sqlite3
db = sqlite3.connect("data/metadata.db")
n_db = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
idx = faiss.read_index("data/vector_store/doc.faiss")
meta = json.load(open("data/vector_store/doc.faiss.meta.json"))
n_map = len(meta["id_to_uuid"])
print(f"DB chunks    : {n_db}")
print(f"faiss ntotal : {idx.ntotal}")
print(f"meta mapped  : {n_map}")
print(f"CONSISTENT   : {n_db == idx.ntotal == n_map}")
EOF
```

If those numbers differ, retrieval is broken even though the UI looks fine.
Two ways it happened here:

- **Wrong index path.** An ingest run with `--index-path /tmp/...` wrote
  metadata to the real DB but vectors to a scratch file. The book was
  registered and unsearchable — DB said 2,735, live index said 12.
- **Orphaned vectors.** The old index held 4 vectors with no UUID mapping,
  which can surface as hits with no retrievable text. Re-ingesting on top
  preserves them; a clean rebuild removes them.

---

## 5. After you get the official 27 W PSU

Work through these in order.

**1. Swap the supply.** Use the official Raspberry Pi 27 W USB-C PSU
(5.1 V / 5 A). Not a phone charger, not a generic USB-C brick. Seat the
plug firmly — a partially seated connector reproduces the same symptoms.

**2. Confirm the Pi negotiated the 5 A profile.**

```bash
vcgencmd get_config usb_max_current_enable   # expect 1
vcgencmd pmic_read_adc | grep EXT5V_V        # expect ~5.1 V at idle
```

With the official PSU the firmware negotiates this automatically. **Do not
set `usb_max_current_enable=1` while still on a weak adapter** — it tells
the firmware to *assume* 5 A is available and removes your last safety
margin, making crashes worse.

**3. Drop the thread throttle.** Start normally:

```bash
cd ~/tinyrag && bash run.sh
```

**4. Re-measure under 4-core load and confirm the margin.**

```bash
# in one shell
for i in 1 2 3 4; do timeout 45 sh -c 'while :; do :; done' & done
# in another
watch -n1 'vcgencmd pmic_read_adc | grep EXT5V_V; vcgencmd measure_temp; vcgencmd get_throttled'
```

Expect the 5 V rail to hold near 5.1 V and `get_throttled` to stay `0x0`.
If it still sags toward 4.7 V, the problem is the cable or the connector,
not the adapter.

**5. Optionally relax the LLM timeout back down.** `timeout_s: 600` in
`config.yaml` is harmless to keep — it is a ceiling, not a delay. At 4
threads answers return in ~78 s regardless.

**6. Confirm latency recovered.** A ~1,800-token prompt should answer in
roughly 78 s rather than 172 s.

---

## 6. The LLM timeout bug this exposed

Throttling to 2 threads surfaced a real deployment bug worth knowing about,
since it looks like a crash but is not.

`LlamaCppClient` defaults to `timeout_s=120.0`, calibrated on the laptop
where generation finishes well inside a minute. Neither `main.py`'s
composition root nor `scripts/ask.py` passed the parameter, so that default
was effectively hardcoded for **every** deployment.

At 2 threads a 1,816-token prompt needs 172 s of llama-server time, so the
request died at 120 s with:

```
llm_failed: llama-server timed out after 120.0s
```

Generation was proceeding normally — only the client gave up. Fixed in
`594340a` by adding `LLMSettings.timeout_s` (default 600 s, validated
`0 < t <= 3600`) and threading it through both construction sites. Set it in
`config.yaml` under `llm:`:

```yaml
llm:
  context_size: 4096
  timeout_s: 600.0
```

If you see `llama-server timed out`, that is this bug or a too-low
`timeout_s` — **not** a power fault. A power fault leaves no error at all.

---

## 7. Diagnosing a future crash

Persistent journaling is now enabled, so crashes leave evidence on disk
rather than vanishing with the RAM. Verify it survived:

```bash
journalctl --list-boots        # more than one line means it is working
journalctl -b -1 -n 50         # last 50 lines before the previous crash
```

If it reverts to volatile, the cause is a vendor drop-in that overrides
`/etc/systemd/journald.conf`. Check for it and override the override:

```bash
grep -rn Storage= /usr/lib/systemd/journald.conf.d/ /etc/systemd/journald.conf.d/
sudo mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\n' | \
  sudo tee /etc/systemd/journald.conf.d/99-persistent-storage.conf
sudo systemctl restart systemd-journald
```

### Reading the result

| Evidence | Means |
|---|---|
| No log entries at all before the cut | **Power loss.** The OS was never involved. |
| `Under-voltage detected` / `get_throttled` bit 16 | Supply sagging past 4.63 V |
| `Out of memory: Killed process` | OOM — a process died, board stayed up |
| Clean `systemd-shutdown` sequence | Thermal or software shutdown, not power |
| `temp>80'C` in throttled output | Thermal throttling (slows, does not power off) |

Two traps that cost time here. Both of these are boot-time kernel
registration lines, **not** crash evidence — do not read them as warnings:

```
thermal_sys: Registered thermal governor 'step_wise'
bcm2835-wdt: Poweroff handler already present!
```

And `ping` succeeding does not mean the board never died. Always check
`uptime -p` — a low uptime means it crashed and rebooted between your
checks.

---

## 8. Data integrity after a hard power cut

Reassuring, and verified repeatedly: **the corpus survived every hard cut
intact.** SQLite runs in WAL mode, which is doing its job.

After any unexpected shutdown:

```bash
cd ~/tinyrag
sqlite3 data/metadata.db "PRAGMA integrity_check;"   # expect: ok
sudo dmesg | grep -ci "ext4.*error"                  # expect: 0
```

Then re-run the three-store consistency check in §4. A rebuild of 2,983
chunks that was interrupted by a power cut came back fully consistent with
`integrity_check: ok` and zero filesystem errors.

---

## 9. Verified working state

Reference numbers from 2026-08-07, `LLAMA_THREADS=2`, for comparison after
any change:

```
doc_chunk_count : 2983   (FAQ 5 + product brief 8 + Data Structures 235 + Cormen 2735)
DB / faiss / meta : 2983 / 2983 / 2983  — CONSISTENT
llama            : up
RAM              : 545 MB of 8 GB
throttled        : 0x0
```

> **Q:** What is the running time of merge sort and why?
> **A:** The running time of merge sort is roughly equal to c2n lg n, where
> c2 is a constant that does not depend on n. This is because the merge sort
> algorithm divides the input into two halves, sorts each half recursively,
> and then merges the two sorted halves…
> — 5 citations from `Cormen Introduction to Algorithms.pdf` (p.33, 819, 33, 169, 61)

Stage breakdown: retrieve 54 ms, prompt build 3 ms, LLM 172.8 s, total
172.8 s, 1,816 prompt + 79 completion tokens.

**Retrieval is 54 ms.** Effectively all latency is generation, so thread
count and the PSU are the only levers that move it.


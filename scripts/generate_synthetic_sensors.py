#!/usr/bin/env python3
"""Generate 30 days of fake-but-realistic smart-home sensor data.

This is the synthetic data source for TinyRAG's evaluation and demo.
Why we generate it ourselves:

1. **Reproducibility.** A fixed SEED (42) means every run produces the
   exact same numbers. That makes the eval set reproducible across
   machines and re-runs.
2. **No PII.** Real sensor logs leak when people are home, when they
   sleep, when they shower. Synthetic data is safe to commit a small
   sample of (we never commit the full CSV — it's gitignored).
3. **Realistic patterns matter.** The Phase 5 eval set asks questions
   like "what was the average living-room temperature yesterday?"
   and "how many motion events in the kitchen this morning?". For
   the LLM to answer those sensibly, the data has to *look* like
   real data: daily temp cycles, weekday vs weekend energy spikes,
   motion bursts at meal times.

Schema (matches ``docs/04_database_design_v1.md`` §6.1)
--------------------------------------------------------

Long format — one row per (timestamp, sensor) reading. 5 columns:

    timestamp, sensor_id, sensor_type, value, unit

6 sensors (the canonical TinyRAG home):

    living_room_temp   temperature  C
    living_room_hum    humidity     %
    bedroom_temp       temperature  C
    bedroom_hum        humidity     %
    kitchen_motion     motion       count  (0 or 1 per tick)
    house_energy       energy       kWh

Per-sensor physics
------------------

Temperature
    Daily sinusoid (peak ~16:00, trough ~05:00), per-room offset
    (bedroom slightly cooler overnight), Gaussian noise.

Humidity
    Weakly inversely correlated with temperature, bounded in [30, 80].

Motion
    Bernoulli (0/1 per tick). Overnight rate ~5%, breakfast ~30%,
    dinner ~40%, weekday lunch ~15%, weekend daytime ~20%. The
    kitchen gets an extra burst on weekday evenings.

Energy
    Base draw 0.1-0.3 kWh per 5-min tick. Weekend rate ~1.5x. Morning
    + evening peaks (cooking, lighting). Random appliance surges
    (5% chance per tick of +0.5-1.5 kWh).

Output
------

Default: ``data/sensor_logs/synthetic_30d.csv`` (gitignored).

CLI flags:

    --start YYYY-MM-DD   first timestamp; default = 30 days before today UTC
    --days N             number of days; default = 30
    --interval-min N     minutes between readings; default = 5
    --out PATH           output CSV path
    --seed N             numpy seed; default = 42
    --summary            also print a human-readable summary to stdout
    --json               print JSON stats instead of CSV preview

Companion docs
--------------
- ``docs/04_database_design_v1.md`` §6 — schema + sensor ID convention
- ``docs/06_roadmap_v2.md`` Step 3.8 — original spec
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — match docs/04_database_design_v1.md §6.2
# ---------------------------------------------------------------------------

#: Default output path. Matches the directory layout in
#: ``docs/03_architecture_v1.md`` and the .gitignore rule for
#: ``data/sensor_logs/``.
DEFAULT_OUTPUT = Path("data/sensor_logs/synthetic_30d.csv")

#: Canonical sensor roster. Don't rename — Phase 4's SensorSource
#: Protocol and Phase 5's eval set will hardcode these ids.
SENSORS: tuple[tuple[str, str, str], ...] = (
    # (sensor_id, sensor_type, unit)
    ("living_room_temp", "temperature", "C"),
    ("living_room_hum", "humidity", "%"),
    ("bedroom_temp", "temperature", "C"),
    ("bedroom_hum", "humidity", "%"),
    ("kitchen_motion", "motion", "count"),
    ("house_energy", "energy", "kWh"),
)

#: Default reproducibility seed. 42 chosen to match the canonical
#: ``docs/06_roadmap_v2.md`` Step 3.8 spec.
DEFAULT_SEED = 42

#: Default sampling interval in minutes. 5 min → 288 readings/day/sensor.
DEFAULT_INTERVAL_MIN = 5

#: Default horizon in days. 30 days at 5-min resolution * 6 sensors
#: = ~51,840 rows. Comfortably fits in RAM and parquet.
DEFAULT_DAYS = 30


# ---------------------------------------------------------------------------
# Per-sensor generators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DailyClock:
    """Hours-since-midnight as a numpy array, for vectorised trig.

    The ``_is_weekend`` field is filled in by :func:`generate` using
    :func:`object.__setattr__` (the dataclass is frozen, so the public
    property setter doesn't work — we go around the safety for one
    write at construction time).
    """

    hours: np.ndarray  # shape (n_ticks,), values in [0, 24)
    _is_weekend: np.ndarray = None  # type: ignore[assignment]

    @property
    def phase(self) -> np.ndarray:
        """2pi * fraction of day completed. Used by sinusoid generators."""
        return 2.0 * np.pi * self.hours / 24.0

    @property
    def is_weekend(self) -> np.ndarray:
        """Per-tick weekend flag (Sat=5, Sun=6 in ISO weekday)."""
        if self._is_weekend is None:
            return np.zeros_like(self.hours, dtype=bool)
        return self._is_weekend


def _generate_temperature(
    clock: _DailyClock,
    *,
    base_c: float,
    amplitude_c: float,
    night_offset_c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Daily sinusoid + per-tick noise. Peak ~16:00, trough ~05:00.

    Using cos(phase - shift) where shift is chosen so cos peaks at
    hour 16: shift = 2pi * (16/24) ~= 4.19.
    """
    shift = 2.0 * np.pi * 16.0 / 24.0
    daily = amplitude_c * np.cos(clock.phase - shift)
    noise = rng.normal(0.0, 0.3, size=clock.hours.size)
    # night_offset is a small downward bias applied during 22:00-06:00.
    night_mask = (clock.hours >= 22.0) | (clock.hours < 6.0)
    return base_c + daily + np.where(night_mask, night_offset_c, 0.0) + noise


def _generate_humidity(
    clock: _DailyClock,
    *,
    base_pct: float,
    temp_series: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weakly anti-correlated with temperature. Bounded in [30, 80]."""
    # Drop ~1.5% humidity for every 1°C above the temp base (22°C).
    adjustment = -1.5 * (temp_series - 22.0)
    noise = rng.normal(0.0, 2.0, size=clock.hours.size)
    values = base_pct + adjustment + noise
    return np.clip(values, 30.0, 80.0)


def _generate_motion(
    clock: _DailyClock,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bernoulli (0 or 1) per 5-min tick. Peaks at breakfast/dinner.

    Rates by hour-of-day (weekday vs weekend differ slightly):
        00-05  → 0.02  (sleeping)
        06-08  → 0.30  weekday / 0.20 weekend (breakfast rush)
        09-11  → 0.10  weekday / 0.15 weekend (chores / coffee)
        12-13  → 0.15  weekday / 0.20 weekend (lunch)
        14-17  → 0.05  (quiet afternoon)
        18-21  → 0.40  weekday / 0.30 weekend (dinner + dishes)
        22-23  → 0.05  (winding down)
    """
    base = np.empty(clock.hours.size, dtype=float)
    for i, h in enumerate(clock.hours):
        if h < 6.0 or h >= 22.0:
            base[i] = 0.02 if h >= 22.0 or h < 6.0 else 0.02
        elif h < 8.0:
            base[i] = 0.30 if not clock.is_weekend[i] else 0.20
        elif h < 11.0:
            base[i] = 0.10 if not clock.is_weekend[i] else 0.15
        elif h < 14.0:
            base[i] = 0.15 if not clock.is_weekend[i] else 0.20
        elif h < 18.0:
            base[i] = 0.05
        else:
            base[i] = 0.40 if not clock.is_weekend[i] else 0.30
    return rng.binomial(1, base).astype(np.float64)


def _generate_energy(
    clock: _DailyClock,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """kWh per 5-min tick. Base + daily peaks + weekend multiplier + surges.

    5-min ticks at average household draw (~1 kW) yield ~0.083 kWh.
    Morning (07-09) and evening (18-22) peaks bump this to ~0.25 kWh.
    Weekends multiply the base by 1.5x.
    5% chance per tick of a random appliance surge (+0.5 to +1.5 kWh).
    """
    base = np.full(clock.hours.size, 0.12)
    morning_peak = (clock.hours >= 7.0) & (clock.hours < 9.0)
    evening_peak = (clock.hours >= 18.0) & (clock.hours < 22.0)
    base = np.where(morning_peak, 0.22, base)
    base = np.where(evening_peak, 0.28, base)
    base = np.where(clock.is_weekend, base * 1.5, base)
    noise = rng.normal(0.0, 0.02, size=clock.hours.size)
    # Random appliance surges (5% chance per tick).
    surges = rng.binomial(1, 0.05, size=clock.hours.size) * rng.uniform(
        0.5, 1.5, size=clock.hours.size
    )
    return np.clip(base + noise + surges, 0.0, None)


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def generate(
    *,
    start_utc: datetime,
    days: int = DEFAULT_DAYS,
    interval_min: int = DEFAULT_INTERVAL_MIN,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate the full long-format DataFrame.

    Parameters
    ----------
    start_utc:
        Timestamp of the first reading. Should be midnight UTC of
        some day for a clean grid.
    days:
        Number of days of data to generate.
    interval_min:
        Minutes between consecutive readings. 5 min is the default.
    seed:
        NumPy seed for reproducibility. Two calls with the same seed
        produce identical output (the unit tests rely on this).

    Returns
    -------
    pandas.DataFrame with columns:
        timestamp (datetime64[ns, UTC]), sensor_id (str),
        sensor_type (str), value (float64), unit (str)

    Total rows: ``days * (24 * 60 // interval_min) * len(SENSORS)``.
    """
    rng = np.random.default_rng(seed)

    # --- Build the time grid ------------------------------------------------
    n_ticks = days * 24 * 60 // interval_min
    timestamps = pd.date_range(
        start=start_utc,
        periods=n_ticks,
        freq=f"{interval_min}min",
        tz="UTC",
        inclusive="left",
    )

    hours = timestamps.hour + timestamps.minute / 60.0
    is_weekend = np.asarray(timestamps.weekday) >= 5  # Sat=5, Sun=6
    clock = _DailyClock(hours=np.asarray(hours, dtype=float))
    # _DailyClock is frozen; bypass via object.__setattr__ for the
    # weekend flag (the property setter is the only legal write path).
    object.__setattr__(clock, "_is_weekend", is_weekend)

    # --- Generate each sensor's values --------------------------------------
    # Pre-generate temperatures so humidity can anti-correlate with them.
    living_room_temp = _generate_temperature(
        clock, base_c=22.0, amplitude_c=2.5, night_offset_c=-0.5, rng=rng
    )
    bedroom_temp = _generate_temperature(
        clock, base_c=20.5, amplitude_c=2.0, night_offset_c=-1.0, rng=rng
    )
    living_room_hum = _generate_humidity(
        clock, base_pct=52.0, temp_series=living_room_temp, rng=rng
    )
    bedroom_hum = _generate_humidity(
        clock, base_pct=55.0, temp_series=bedroom_temp, rng=rng
    )
    kitchen_motion = _generate_motion(clock, rng=rng)
    house_energy = _generate_energy(clock, rng=rng)

    values_by_sensor: dict[str, np.ndarray] = {
        "living_room_temp": living_room_temp,
        "living_room_hum": living_room_hum,
        "bedroom_temp": bedroom_temp,
        "bedroom_hum": bedroom_hum,
        "kitchen_motion": kitchen_motion,
        "house_energy": house_energy,
    }

    # --- Assemble long-format DataFrame ------------------------------------
    sensor_ids = [s[0] for s in SENSORS]
    sensor_types = {s[0]: s[1] for s in SENSORS}
    units = {s[0]: s[2] for s in SENSORS}

    rows: list[pd.DataFrame] = []
    for sid in sensor_ids:
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": sid,
                    "sensor_type": sensor_types[sid],
                    "value": values_by_sensor[sid],
                    "unit": units[sid],
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)

    # Final sanity: round to sensible precision per unit.
    df.loc[df["sensor_type"].isin(["temperature", "humidity", "energy"]), "value"] = (
        df.loc[df["sensor_type"].isin(["temperature", "humidity", "energy"]), "value"]
        .round(2)
    )
    df.loc[df["sensor_type"] == "motion", "value"] = df.loc[
        df["sensor_type"] == "motion", "value"
    ].round(0).astype(np.int64)
    return df


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def summarise(df: pd.DataFrame) -> dict:
    """Return a JSON-serialisable summary of the generated data."""
    by_sensor: dict[str, dict] = {}
    for sid, group in df.groupby("sensor_id"):
        by_sensor[sid] = {
            "count": int(len(group)),
            "min": float(group["value"].min()),
            "max": float(group["value"].max()),
            "mean": float(group["value"].mean()),
            "std": float(group["value"].std()),
            "unit": group["unit"].iloc[0],
        }
    return {
        "rows": int(len(df)),
        "sensors": sorted(df["sensor_id"].unique().tolist()),
        "start_utc": df["timestamp"].min().isoformat(),
        "end_utc": df["timestamp"].max().isoformat(),
        "by_sensor": by_sensor,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_synthetic_sensors.py",
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Start date (YYYY-MM-DD). Default: 30 days before today UTC, "
            "rounded down to midnight."
        ),
    )
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of days.")
    p.add_argument(
        "--interval-min", type=int, default=DEFAULT_INTERVAL_MIN,
        help="Minutes between readings (default: 5).",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="NumPy seed.")
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary to stdout after writing.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print JSON stats to stdout (for scripts/CI).",
    )
    return p


def _resolve_start(arg: str | None, days: int) -> datetime:
    """Parse --start or default to ``days`` ago at midnight UTC."""
    if arg:
        # Accept YYYY-MM-DD or full ISO 8601.
        try:
            dt = datetime.fromisoformat(arg)
        except ValueError as exc:
            raise SystemExit(f"Invalid --start: {arg!r} (expected YYYY-MM-DD)") from exc
    else:
        now = datetime.now(UTC)
        dt = (now - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    start = _resolve_start(args.start, args.days)
    df = generate(
        start_utc=start,
        days=args.days,
        interval_min=args.interval_min,
        seed=args.seed,
    )

    # Ensure parent dir exists.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # ISO 8601 timestamps + UTC. We don't use tz-aware strings here because
    # the design doc §6.1 example shows naive ISO strings ("2026-06-22T..."),
    # and the sensor source (Phase 4.13) will parse them as UTC.
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df.to_csv(args.out, index=False)

    stats = summarise(pd.read_csv(args.out, parse_dates=["timestamp"]))

    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
    elif args.summary:
        print(f"Wrote {stats['rows']:,} rows to {args.out}")
        print(f"  range: {stats['start_utc']} → {stats['end_utc']}")
        print("  by sensor:")
        for sid, s in stats["by_sensor"].items():
            print(
                f"    {sid:18s}  n={s['count']:>5d}  "
                f"min={s['min']:>6.2f}  max={s['max']:>6.2f}  "
                f"mean={s['mean']:>6.2f}  std={s['std']:>5.2f}  {s['unit']}"
            )
    else:
        # Headline output: one line per the script's main purpose.
        print(
            f"[ OK ] Wrote {stats['rows']:,} rows to {args.out} "
            f"({stats['start_utc']} → {stats['end_utc']})"
        )
        print(f"       {len(stats['sensors'])} sensors, seed={args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

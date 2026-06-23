"""Tests for scripts/generate_synthetic_sensors.py.

These tests lock down the *contract* the rest of TinyRAG relies on:

1. **Reproducibility.** SEED=42 produces the same numbers every run.
   Without this, the Phase 5 eval set can't compare across machines.
2. **Schema conformance.** Columns + dtypes match
   ``docs/04_database_design_v1.md`` §6.1 exactly.
3. **Realistic value ranges.** Temperature 15-30°C, humidity 30-80%,
   motion 0/1, energy ≥ 0. If these slip, the eval questions start
   producing nonsense answers.
4. **Canonical sensor roster.** Exactly 6 sensors, with the canonical
   ids Phase 4+ will hardcode.
5. **Row count is deterministic.** ``days * (24*60/interval) * 6``.

We import the script's ``generate`` and ``summarise`` functions directly
(no subprocess) so the tests are fast and hermetic.

Location: ``tests/test_generate_synthetic_sensors.py``
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Make the script importable as a module.
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_synthetic_sensors import (  # noqa: E402
    DEFAULT_DAYS,
    DEFAULT_INTERVAL_MIN,
    DEFAULT_SEED,
    SENSORS,
    generate,
    summarise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_ids() -> set[str]:
    return {sid for sid, _, _ in SENSORS}


def _expected_rows(days: int, interval_min: int) -> int:
    return days * (24 * 60 // interval_min) * len(SENSORS)


# A fixed start so all tests are deterministic regardless of when they run.
FIXED_START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Schema & shape
# ---------------------------------------------------------------------------


class TestSchema:
    """The DataFrame must match docs/04_database_design_v1.md §6.1."""

    def test_columns_exact(self) -> None:
        df = generate(start_utc=FIXED_START)
        assert list(df.columns) == [
            "timestamp",
            "sensor_id",
            "sensor_type",
            "value",
            "unit",
        ]

    def test_no_nan_anywhere(self) -> None:
        """A NaN in a sensor reading breaks aggregation downstream."""
        df = generate(start_utc=FIXED_START)
        assert not df.isna().any().any(), f"NaN found in:\n{df[df.isna().any(axis=1)]}"

    def test_canonical_sensors_only(self) -> None:
        df = generate(start_utc=FIXED_START)
        assert set(df["sensor_id"].unique()) == _canonical_ids()

    def test_canonical_sensor_types(self) -> None:
        df = generate(start_utc=FIXED_START)
        # The four types from §6.1 — exactly.
        assert set(df["sensor_type"].unique()) == {
            "temperature",
            "humidity",
            "energy",
            "motion",
        }

    def test_canonical_units(self) -> None:
        df = generate(start_utc=FIXED_START)
        # The four units from §6.1 — exactly.
        assert set(df["unit"].unique()) == {"C", "%", "kWh", "count"}

    def test_unit_matches_sensor_type(self) -> None:
        """Per §6.1 the unit must match the sensor_type."""
        df = generate(start_utc=FIXED_START)
        expected_unit = {"temperature": "C", "humidity": "%", "energy": "kWh", "motion": "count"}
        for _, row in df.iterrows():
            assert row["unit"] == expected_unit[row["sensor_type"]], (
                f"sensor {row['sensor_id']} has type={row['sensor_type']} "
                f"but unit={row['unit']}"
            )


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------


class TestRowCounts:
    """Row counts are deterministic from (days, interval_min)."""

    def test_default_30_days_5min(self) -> None:
        df = generate(start_utc=FIXED_START)
        # 30 days * 288 ticks/day * 6 sensors = 51,840
        assert len(df) == _expected_rows(DEFAULT_DAYS, DEFAULT_INTERVAL_MIN)

    def test_seven_days(self) -> None:
        df = generate(start_utc=FIXED_START, days=7)
        assert len(df) == 7 * 288 * len(SENSORS)

    def test_one_hour_interval(self) -> None:
        df = generate(start_utc=FIXED_START, days=2, interval_min=60)
        # 2 days * 24 hours * 6 sensors = 288
        assert len(df) == 288

    def test_per_sensor_count_equal(self) -> None:
        """Every sensor gets exactly the same number of readings."""
        df = generate(start_utc=FIXED_START)
        counts = df.groupby("sensor_id").size()
        assert counts.nunique() == 1, f"uneven counts:\n{counts}"
        assert counts.iloc[0] == 288 * DEFAULT_DAYS


# ---------------------------------------------------------------------------
# Value ranges — the "looks like real data" guarantee
# ---------------------------------------------------------------------------


class TestValueRanges:
    """Sanity bounds: values must look realistic, not random or constant."""

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return generate(start_utc=FIXED_START)

    def test_temperature_in_realistic_range(self, df: pd.DataFrame) -> None:
        temps = df[df["sensor_type"] == "temperature"]["value"]
        assert temps.min() >= 15.0, f"too cold: {temps.min()}"
        assert temps.max() <= 30.0, f"too hot: {temps.max()}"
        # Mean should be near 20-23°C (typical indoor).
        assert 18.0 <= temps.mean() <= 24.0

    def test_humidity_in_realistic_range(self, df: pd.DataFrame) -> None:
        hums = df[df["sensor_type"] == "humidity"]["value"]
        assert hums.min() >= 30.0
        assert hums.max() <= 80.0
        assert 40.0 <= hums.mean() <= 65.0

    def test_motion_is_binary(self, df: pd.DataFrame) -> None:
        motion = df[df["sensor_type"] == "motion"]["value"]
        assert set(motion.unique()).issubset({0, 1})

    def test_motion_rate_is_realistic(self, df: pd.DataFrame) -> None:
        """Kitchen motion should fire on 5-30% of ticks (not constant)."""
        motion = df[df["sensor_id"] == "kitchen_motion"]["value"]
        rate = motion.mean()
        assert 0.05 <= rate <= 0.30, f"motion rate {rate:.2f} outside realistic band"

    def test_energy_is_nonnegative(self, df: pd.DataFrame) -> None:
        energy = df[df["sensor_type"] == "energy"]["value"]
        assert energy.min() >= 0.0

    def test_energy_per_tick_realistic(self, df: pd.DataFrame) -> None:
        """5-min kWh at ~1 kW average is ~0.083; peaks with appliances
        can hit ~2 kWh. Mean should be 0.1-0.5."""
        energy = df[df["sensor_type"] == "energy"]["value"]
        assert 0.1 <= energy.mean() <= 0.5, f"energy mean {energy.mean():.2f}"
        # Max should be under 5 kWh (5-min spike of a 60 kW appliance — unrealistic).
        assert energy.max() < 5.0


# ---------------------------------------------------------------------------
# Daily patterns — the data has structure, not just noise
# ---------------------------------------------------------------------------


class TestDailyPatterns:
    """Temperatures should peak in the afternoon; motion should peak at meals."""

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return generate(start_utc=FIXED_START)

    def test_temperature_peaks_afternoon(self, df: pd.DataFrame) -> None:
        """Mean 14:00-17:00 temperature > mean 03:00-06:00 temperature."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        lr = df[df["sensor_id"] == "living_room_temp"]
        afternoon = lr[(lr["hour"] >= 14) & (lr["hour"] <= 17)]["value"].mean()
        early_am = lr[(lr["hour"] >= 3) & (lr["hour"] <= 6)]["value"].mean()
        assert afternoon > early_am, (
            f"afternoon {afternoon:.2f} should exceed early AM {early_am:.2f}"
        )

    def test_motion_peaks_at_dinner(self, df: pd.DataFrame) -> None:
        """Mean motion 18:00-21:00 > mean motion 02:00-05:00."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        kitchen = df[df["sensor_id"] == "kitchen_motion"]
        dinner = kitchen[(kitchen["hour"] >= 18) & (kitchen["hour"] <= 21)]["value"].mean()
        sleeping = kitchen[(kitchen["hour"] >= 2) & (kitchen["hour"] <= 5)]["value"].mean()
        assert dinner > sleeping, (
            f"dinner {dinner:.3f} should exceed sleeping {sleeping:.3f}"
        )


# ---------------------------------------------------------------------------
# Reproducibility — SEED=42 is the contract
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Two runs with the same seed must produce byte-identical output."""

    def test_same_seed_same_output(self) -> None:
        df_a = generate(start_utc=FIXED_START, seed=42)
        df_b = generate(start_utc=FIXED_START, seed=42)
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_different_seed_different_output(self) -> None:
        """Sanity check that the seed is actually doing something."""
        df_a = generate(start_utc=FIXED_START, seed=42)
        df_b = generate(start_utc=FIXED_START, seed=43)
        # The two DataFrames must NOT be equal.
        assert not df_a.equals(df_b)

    def test_default_seed_is_documented_value(self) -> None:
        """The default seed is part of the public contract."""
        assert DEFAULT_SEED == 42


# ---------------------------------------------------------------------------
# Summary stats helper
# ---------------------------------------------------------------------------


class TestSummarise:
    """The summarise() helper is what --json / --summary print."""

    def test_returns_jsonable_dict(self) -> None:
        df = generate(start_utc=FIXED_START)
        stats = summarise(df)
        # Must be JSON-serialisable (no numpy scalars left over).
        import json

        json.dumps(stats)  # raises if not serialisable

    def test_summary_has_required_keys(self) -> None:
        df = generate(start_utc=FIXED_START)
        stats = summarise(df)
        for k in ("rows", "sensors", "start_utc", "end_utc", "by_sensor"):
            assert k in stats, f"missing key: {k}"

    def test_summary_by_sensor_covers_all_six(self) -> None:
        df = generate(start_utc=FIXED_START)
        stats = summarise(df)
        assert set(stats["by_sensor"].keys()) == _canonical_ids()

    def test_summary_row_count_matches_df(self) -> None:
        df = generate(start_utc=FIXED_START)
        stats = summarise(df)
        assert stats["rows"] == len(df)


# ---------------------------------------------------------------------------
# Time-grid correctness
# ---------------------------------------------------------------------------


class TestTimeGrid:
    """Timestamps must be on a perfect 5-min (or chosen-interval) grid."""

    def test_timestamps_on_grid(self) -> None:
        df = generate(start_utc=FIXED_START, interval_min=5)
        # Parse back; check all deltas are exactly 5 minutes.
        ts = pd.to_datetime(df["timestamp"]).sort_values().unique()
        # Drop the first; deltas[i] = ts[i+1] - ts[i]
        deltas = ts[1:] - ts[:-1]
        assert (deltas == pd.Timedelta(minutes=5)).all(), deltas[deltas != pd.Timedelta(minutes=5)]

    def test_start_timestamp_matches_input(self) -> None:
        df = generate(start_utc=FIXED_START)
        ts = pd.to_datetime(df["timestamp"])
        # Every (start_utc, sensor) combination appears exactly once.
        starts = ts[ts == pd.Timestamp(FIXED_START)]
        assert len(starts) == len(SENSORS)

    def test_no_duplicate_timestamp_per_sensor(self) -> None:
        """One row per (timestamp, sensor_id)."""
        df = generate(start_utc=FIXED_START)
        assert not df.duplicated(subset=["timestamp", "sensor_id"]).any()


# ---------------------------------------------------------------------------
# Defaults are sane
# ---------------------------------------------------------------------------


class TestDefaults:
    """The module-level defaults are part of the CLI surface."""

    def test_default_interval_is_5_minutes(self) -> None:
        assert DEFAULT_INTERVAL_MIN == 5

    def test_default_days_is_30(self) -> None:
        assert DEFAULT_DAYS == 30

    def test_default_seed_is_42(self) -> None:
        assert DEFAULT_SEED == 42

    def test_six_sensors_in_canonical_order(self) -> None:
        """The exact order is part of the public surface (Phase 5+)."""
        ids = [s[0] for s in SENSORS]
        assert ids == [
            "living_room_temp",
            "living_room_hum",
            "bedroom_temp",
            "bedroom_hum",
            "kitchen_motion",
            "house_energy",
        ]


# ---------------------------------------------------------------------------
# Custom start date
# ---------------------------------------------------------------------------


class TestCustomStart:
    """The --start flag must work and be honoured exactly."""

    def test_custom_start_date_appears_in_output(self) -> None:
        custom = datetime(2025, 6, 1, 0, 0, 0, tzinfo=UTC)
        df = generate(start_utc=custom)
        ts = pd.to_datetime(df["timestamp"])
        assert ts.min() == pd.Timestamp(custom)

    def test_custom_start_30_days_later(self) -> None:
        custom = datetime(2025, 6, 1, 0, 0, 0, tzinfo=UTC)
        df = generate(start_utc=custom, days=30)
        ts = pd.to_datetime(df["timestamp"])
        # The last reading is at custom + 30 days - 5 min.
        expected_end = custom + timedelta(days=30) - timedelta(minutes=5)
        assert ts.max() == pd.Timestamp(expected_end)

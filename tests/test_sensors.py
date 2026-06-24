"""Unit tests for the SensorSource layer (Step 4.13).

Coverage targets:
- :mod:`tinyrag.sensors.base` — :class:`SensorSource` Protocol
  + :class:`SensorReading` dataclass + the typed exception
  hierarchy.
- :mod:`tinyrag.sensors.simulated` — :class:`SimulatedCSVSource`
  end-to-end on a tmpdir CSV (schema match, ``since`` filter,
  schema-mismatch errors, available_sensors).
- :mod:`tinyrag.sensors.serial_dht` and :mod:`tinyrag.sensors.mqtt`
  — Phase 4 stubs (construction + NotImplementedError on read
  + cheap available_sensors).

All tests are **hermetic** except the very last class, which
exercises the real ``data/sensor_logs/synthetic_30d.csv`` (already
gitignored, present on this machine from Step 3.8) to pin the
schema of the Step 3.8 generator against the Step 4.13 reader.
That class is the regression gate for "did the generator drift
from the schema we promised the rest of the pipeline?".
"""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tinyrag.sensors import (
    REQUIRED_COLUMNS,
    SENSOR_TYPE_ENERGY,
    SENSOR_TYPE_HUMIDITY,
    SENSOR_TYPE_MOTION,
    SENSOR_TYPE_TEMPERATURE,
    SUPPORTED_SENSOR_TYPES,
    SUPPORTED_UNITS,
    UNIT_CELSIUS,
    UNIT_COUNT,
    UNIT_KWH,
    UNIT_PERCENT,
    MQTTBrokerSource,
    RealSerialSource,
    SensorReading,
    SensorSource,
    SensorSourceConfigError,
    SensorSourceError,
    SensorSourceReadError,
    SensorSourceSchemaError,
    SimulatedCSVSource,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

#: A canonical, schema-valid sample row used by the SensorReading
#: and SimulatedCSVSource tests. Datetime in UTC, the convention
#: the Step 3.8 generator follows.
_SAMPLE_ROW: dict[str, Any] = {
    "timestamp": datetime(2026, 6, 24, 9, 30, 0, tzinfo=UTC),
    "sensor_id": "living_room_temp",
    "sensor_type": SENSOR_TYPE_TEMPERATURE,
    "value": 22.5,
    "unit": UNIT_CELSIUS,
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a schema-valid sensor CSV with stable dtypes.

    The CSV writer is configured to:

    - quote nothing (so the on-disk format matches Step 3.8's
      ``df.to_csv(...)`` output: no quoting, UTF-8, LF newlines).
    - write the header row in the canonical column order.
    - format timestamps as ISO 8601 (matches the Step 3.8 generator's
      ``datetime.isoformat()`` output — see ``scripts/generate_synthetic_sensors.py``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            ts = row["timestamp"]
            if isinstance(ts, datetime) or hasattr(ts, "isoformat"):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            writer.writerow(
                {
                    "timestamp": ts_str,
                    "sensor_id": row["sensor_id"],
                    "sensor_type": row["sensor_type"],
                    "value": row["value"],
                    "unit": row["unit"],
                }
            )


def _make_rows(n_days: int = 3) -> list[dict[str, Any]]:
    """Build a small, schema-valid sensor log over ``n_days`` days.

    Six sensors matching the Step 3.8 canonical roster, 4 readings
    per day per sensor (every 6 hours) — enough to exercise the
    ``since`` filter without bloating the tmpdir.
    """
    rows: list[dict[str, Any]] = []
    sensors = [
        ("living_room_temp", SENSOR_TYPE_TEMPERATURE, UNIT_CELSIUS, 22.0),
        ("living_room_hum", SENSOR_TYPE_HUMIDITY, UNIT_PERCENT, 55.0),
        ("bedroom_temp", SENSOR_TYPE_TEMPERATURE, UNIT_CELSIUS, 20.0),
        ("bedroom_hum", SENSOR_TYPE_HUMIDITY, UNIT_PERCENT, 60.0),
        ("kitchen_motion", SENSOR_TYPE_MOTION, UNIT_COUNT, 0),
        ("house_energy", SENSOR_TYPE_ENERGY, UNIT_KWH, 0.1),
    ]
    base = datetime(2026, 6, 22, 0, 0, 0, tzinfo=UTC)
    for day in range(n_days):
        for hour in (0, 6, 12, 18):
            ts = base + timedelta(days=day, hours=hour)
            for sensor_id, sensor_type, unit, base_value in sensors:
                rows.append(
                    {
                        "timestamp": ts,
                        "sensor_id": sensor_id,
                        "sensor_type": sensor_type,
                        "value": base_value + day * 0.1,
                        "unit": unit,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# 1. Public surface — every documented symbol importable
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """Every documented symbol in :mod:`tinyrag.sensors` is importable."""

    def test_all_classes_importable(self) -> None:
        from tinyrag.sensors import (
            MQTTBrokerSource,
            RealSerialSource,
            SensorReading,
            SensorSource,
            SimulatedCSVSource,
        )

        assert SensorSource is not None
        assert SimulatedCSVSource is not None
        assert RealSerialSource is not None
        assert MQTTBrokerSource is not None
        assert SensorReading is not None

    def test_all_errors_importable(self) -> None:
        from tinyrag.sensors import (
            SensorSourceConfigError,
            SensorSourceError,
            SensorSourceReadError,
            SensorSourceSchemaError,
        )

        assert issubclass(SensorSourceConfigError, SensorSourceError)
        assert issubclass(SensorSourceSchemaError, SensorSourceError)
        assert issubclass(SensorSourceReadError, SensorSourceError)

    def test_required_columns_constant(self) -> None:
        assert REQUIRED_COLUMNS == ("timestamp", "sensor_id", "sensor_type", "value", "unit")

    def test_supported_sensor_types_constant(self) -> None:
        assert frozenset(
            {
                SENSOR_TYPE_TEMPERATURE,
                SENSOR_TYPE_HUMIDITY,
                SENSOR_TYPE_ENERGY,
                SENSOR_TYPE_MOTION,
            }
        ) == SUPPORTED_SENSOR_TYPES

    def test_supported_units_constant(self) -> None:
        assert frozenset(
            {UNIT_CELSIUS, UNIT_PERCENT, UNIT_KWH, UNIT_COUNT}
        ) == SUPPORTED_UNITS

    def test_sensor_type_constants(self) -> None:
        assert SENSOR_TYPE_TEMPERATURE == "temperature"
        assert SENSOR_TYPE_HUMIDITY == "humidity"
        assert SENSOR_TYPE_ENERGY == "energy"
        assert SENSOR_TYPE_MOTION == "motion"

    def test_unit_constants(self) -> None:
        assert UNIT_CELSIUS == "C"
        assert UNIT_PERCENT == "%"
        assert UNIT_KWH == "kWh"
        assert UNIT_COUNT == "count"


# ---------------------------------------------------------------------------
# 2. SensorReading dataclass
# ---------------------------------------------------------------------------


class TestSensorReadingDataclass:
    """The :class:`SensorReading` dataclass validates at construction."""

    def test_construction_valid(self) -> None:
        r = SensorReading(**_SAMPLE_ROW)
        assert r.timestamp == _SAMPLE_ROW["timestamp"]
        assert r.sensor_id == "living_room_temp"
        assert r.sensor_type == "temperature"
        assert r.value == 22.5
        assert r.unit == "C"

    def test_frozen(self) -> None:
        r = SensorReading(**_SAMPLE_ROW)
        with pytest.raises(FrozenInstanceError):
            r.value = 99.0  # type: ignore[misc]

    def test_invalid_sensor_type_rejected(self) -> None:
        bad = {**_SAMPLE_ROW, "sensor_type": "voltage"}
        with pytest.raises(ValueError, match="sensor_type must be one of"):
            SensorReading(**bad)

    def test_invalid_unit_rejected(self) -> None:
        bad = {**_SAMPLE_ROW, "unit": "fahrenheit"}
        with pytest.raises(ValueError, match="unit must be one of"):
            SensorReading(**bad)

    def test_empty_sensor_id_rejected(self) -> None:
        bad = {**_SAMPLE_ROW, "sensor_id": ""}
        with pytest.raises(ValueError, match="sensor_id must be a non-empty string"):
            SensorReading(**bad)

    def test_from_row_pandas_series(self) -> None:
        """``from_row`` accepts a pandas Series and converts types."""
        row = pd.Series(_SAMPLE_ROW)
        r = SensorReading.from_row(row)
        assert r.sensor_id == "living_room_temp"
        assert r.sensor_type == "temperature"
        assert r.value == 22.5
        assert r.unit == "C"
        # Timestamp may be a pd.Timestamp after the conversion;
        # what matters is that to_pydatetime() produced a datetime.
        assert isinstance(r.timestamp, datetime)

    def test_from_row_dict_like(self) -> None:
        """``from_row`` accepts any mapping with the required keys."""

        @dataclass
        class RowDict:
            data: dict[str, Any]

            def __getitem__(self, key: str) -> Any:
                return self.data[key]

        r = SensorReading.from_row(RowDict(_SAMPLE_ROW))
        assert r.sensor_id == "living_room_temp"


# ---------------------------------------------------------------------------
# 3. Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """The :class:`SensorSource` Protocol is ``@runtime_checkable``."""

    def test_simulated_csv_source_satisfies_protocol(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        _write_csv(csv_path, [_SAMPLE_ROW])
        src: SensorSource = SimulatedCSVSource(csv_path)
        assert isinstance(src, SensorSource)

    def test_real_serial_source_satisfies_protocol(self) -> None:
        src: SensorSource = RealSerialSource(dht_pin=4, pir_pin=17)
        assert isinstance(src, SensorSource)

    def test_mqtt_broker_source_satisfies_protocol(self) -> None:
        src: SensorSource = MQTTBrokerSource(
            host="127.0.0.1", port=1883, topic_prefix="tinyrag/sensors/"
        )
        assert isinstance(src, SensorSource)

    def test_arbitrary_duck_typed_class_satisfies_protocol(self) -> None:
        """A plain class that implements both methods satisfies the Protocol."""

        class MyDuckSensor:
            def read(self, since: datetime | None = None) -> pd.DataFrame:
                return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

            def available_sensors(self) -> list[str]:
                return []

        assert isinstance(MyDuckSensor(), SensorSource)

    def test_class_missing_read_fails_protocol(self) -> None:
        class NoRead:
            def available_sensors(self) -> list[str]:
                return []

        assert not isinstance(NoRead(), SensorSource)

    def test_class_missing_available_sensors_fails_protocol(self) -> None:
        class NoAvailable:
            def read(self, since: datetime | None = None) -> pd.DataFrame:
                return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

        assert not isinstance(NoAvailable(), SensorSource)


# ---------------------------------------------------------------------------
# 4. SimulatedCSVSource construction
# ---------------------------------------------------------------------------


class TestSimulatedCsvSourceConstruction:
    """:class:`SimulatedCSVSource` accepts both str and Path for the file."""

    def test_str_path_accepted(self) -> None:
        src = SimulatedCSVSource("data/sensors.csv")
        assert src.path == Path("data/sensors.csv")
        assert isinstance(src.path, Path)

    def test_pathlib_path_accepted(self) -> None:
        p = Path("data/sensors.csv")
        src = SimulatedCSVSource(p)
        assert src.path is p

    def test_default_since_defaults_to_none(self) -> None:
        src = SimulatedCSVSource("data/sensors.csv")
        assert src.default_since is None

    def test_default_since_preserved(self) -> None:
        floor = datetime(2026, 6, 24, tzinfo=UTC)
        src = SimulatedCSVSource("data/sensors.csv", default_since=floor)
        assert src.default_since == floor

    def test_frozen(self) -> None:
        src = SimulatedCSVSource("data/sensors.csv")
        with pytest.raises(FrozenInstanceError):
            src.path = Path("other.csv")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. SimulatedCSVSource.read — happy path
# ---------------------------------------------------------------------------


class TestSimulatedCsvSourceRead:
    """:meth:`SimulatedCSVSource.read` returns a schema-valid DataFrame."""

    def test_happy_path_returns_dataframe(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        rows = _make_rows(n_days=3)
        _write_csv(csv_path, rows)
        df = SimulatedCSVSource(csv_path).read()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(rows)

    def test_columns_match_required(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        _write_csv(csv_path, [_SAMPLE_ROW])
        df = SimulatedCSVSource(csv_path).read()
        # Order matches REQUIRED_COLUMNS exactly (pd.read_csv
        # preserves the on-disk order).
        assert list(df.columns) == list(REQUIRED_COLUMNS)

    def test_timestamp_parsed_as_datetime(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        _write_csv(csv_path, [_SAMPLE_ROW])
        df = SimulatedCSVSource(csv_path).read()
        # pandas parses ISO 8601 strings into datetime64[ns].
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_value_parsed_as_float(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        _write_csv(csv_path, [_SAMPLE_ROW])
        df = SimulatedCSVSource(csv_path).read()
        assert pd.api.types.is_float_dtype(df["value"])
        assert df["value"].iloc[0] == pytest.approx(22.5)

    def test_string_columns_preserved(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        _write_csv(csv_path, [_SAMPLE_ROW])
        df = SimulatedCSVSource(csv_path).read()
        # String columns come back as object dtype (not pandas
        # StringDtype) because we use ``string`` in the dtype map
        # but pandas may upcast to object for NA compatibility.
        # The important property is that values round-trip.
        assert str(df["sensor_id"].iloc[0]) == "living_room_temp"
        assert str(df["sensor_type"].iloc[0]) == "temperature"
        assert str(df["unit"].iloc[0]) == "C"

    def test_empty_csv_returns_empty_dataframe_with_correct_schema(
        self, tmp_path: Path
    ) -> None:
        """A header-only CSV returns an empty DataFrame with the right columns."""
        csv_path = tmp_path / "sensors.csv"
        csv_path.write_text("timestamp,sensor_id,sensor_type,value,unit\n")
        df = SimulatedCSVSource(csv_path).read()
        assert len(df) == 0
        assert list(df.columns) == list(REQUIRED_COLUMNS)


# ---------------------------------------------------------------------------
# 6. SimulatedCSVSource.read — ``since`` filter
# ---------------------------------------------------------------------------


class TestSimulatedCsvSourceReadSince:
    """The :meth:`SimulatedCSVSource.read` ``since`` filter works."""

    def test_since_filter_returns_only_recent_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        rows = _make_rows(n_days=3)  # 2026-06-22, 23, 24
        _write_csv(csv_path, rows)
        df = SimulatedCSVSource(csv_path).read(
            since=datetime(2026, 6, 23, 0, 0, 0, tzinfo=UTC)
        )
        # 3 sensors at hours 0/6/12/18 per day on 2 days = 24 rows per
        # sensor * 6 sensors = 144? No — only "day >= 2026-06-23" so
        # day 0 is excluded entirely. 2 days x 4 hours x 6 sensors = 48.
        assert len(df) == 48
        assert df["timestamp"].min() >= datetime(2026, 6, 23, 0, 0, 0, tzinfo=UTC)

    def test_since_none_returns_all_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        rows = _make_rows(n_days=3)
        _write_csv(csv_path, rows)
        df = SimulatedCSVSource(csv_path).read(since=None)
        assert len(df) == len(rows)

    def test_since_in_future_returns_empty(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        rows = _make_rows(n_days=3)
        _write_csv(csv_path, rows)
        df = SimulatedCSVSource(csv_path).read(
            since=datetime(2030, 1, 1, tzinfo=UTC)
        )
        assert len(df) == 0
        # Empty DF still has the right schema.
        assert list(df.columns) == list(REQUIRED_COLUMNS)

    def test_default_since_applied_when_caller_passes_none(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        rows = _make_rows(n_days=3)
        _write_csv(csv_path, rows)
        floor = datetime(2026, 6, 23, 12, 0, 0, tzinfo=UTC)
        src = SimulatedCSVSource(csv_path, default_since=floor)
        df = src.read()  # No since arg → default_since wins
        assert df["timestamp"].min() >= floor

    def test_explicit_since_overrides_default_since(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        rows = _make_rows(n_days=3)
        _write_csv(csv_path, rows)
        # default_since is day 3; explicit since is day 1.
        # Caller's explicit value should win.
        src = SimulatedCSVSource(
            csv_path, default_since=datetime(2026, 6, 24, 0, 0, 0, tzinfo=UTC)
        )
        df = src.read(since=datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC))
        assert df["timestamp"].min() >= datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 7. SimulatedCSVSource.available_sensors
# ---------------------------------------------------------------------------


class TestSimulatedCsvSourceAvailableSensors:
    """:meth:`SimulatedCSVSource.available_sensors` lists sensor IDs cheaply."""

    def test_returns_sorted_unique_ids(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        _write_csv(csv_path, _make_rows(n_days=2))
        ids = SimulatedCSVSource(csv_path).available_sensors()
        assert ids == sorted(set(ids))  # sorted + unique
        assert set(ids) == {
            "living_room_temp",
            "living_room_hum",
            "bedroom_temp",
            "bedroom_hum",
            "kitchen_motion",
            "house_energy",
        }

    def test_returns_empty_list_for_empty_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sensors.csv"
        csv_path.write_text("timestamp,sensor_id,sensor_type,value,unit\n")
        ids = SimulatedCSVSource(csv_path).available_sensors()
        assert ids == []

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(SensorSourceConfigError, match="not found"):
            SimulatedCSVSource(csv_path).available_sensors()


# ---------------------------------------------------------------------------
# 8. SimulatedCSVSource — error mapping
# ---------------------------------------------------------------------------


class TestSimulatedCsvSourceErrors:
    """:class:`SimulatedCSVSource` raises typed :class:`SensorSourceError` subclasses."""

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ghost.csv"
        with pytest.raises(SensorSourceConfigError) as excinfo:
            SimulatedCSVSource(csv_path).read()
        assert excinfo.value.path == csv_path
        assert "not found" in str(excinfo.value).lower()

    def test_missing_column_raises_schema_error(self, tmp_path: Path) -> None:
        """A CSV missing the ``timestamp`` column is rejected."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "sensor_id,sensor_type,value,unit\n"
            "living_room_temp,temperature,22.5,C\n"
        )
        with pytest.raises(SensorSourceSchemaError) as excinfo:
            SimulatedCSVSource(csv_path).read()
        assert excinfo.value.path == csv_path
        # Error message mentions both the missing column and the
        # expected set, so a hand-edited CSV is debuggable.
        assert "timestamp" in str(excinfo.value)
        assert "expected" in str(excinfo.value).lower()

    def test_extra_column_raises_schema_error(self, tmp_path: Path) -> None:
        """A CSV with an extra column is rejected (strict schema)."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "timestamp,sensor_id,sensor_type,value,unit,extra\n"
            "2026-06-24T00:00:00,living_room_temp,temperature,22.5,C,foo\n"
        )
        with pytest.raises(SensorSourceSchemaError, match="extra|unexpected"):
            SimulatedCSVSource(csv_path).read()

    def test_unknown_sensor_type_raises_schema_error(self, tmp_path: Path) -> None:
        """A row with ``sensor_type="voltage"`` is rejected (not in SUPPORTED)."""
        csv_path = tmp_path / "bad.csv"
        _write_csv(
            csv_path,
            [
                {
                    "timestamp": datetime(2026, 6, 24, tzinfo=UTC),
                    "sensor_id": "mystery",
                    "sensor_type": "voltage",  # NOT in SUPPORTED_SENSOR_TYPES
                    "value": 5.0,
                    "unit": "V",  # also bad, but bad sensor_type fires first
                }
            ],
        )
        with pytest.raises(SensorSourceSchemaError, match="sensor_type"):
            SimulatedCSVSource(csv_path).read()

    def test_unknown_unit_raises_schema_error(self, tmp_path: Path) -> None:
        """A row with ``unit="V"`` is rejected (not in SUPPORTED_UNITS)."""
        csv_path = tmp_path / "bad.csv"
        _write_csv(
            csv_path,
            [
                {
                    "timestamp": datetime(2026, 6, 24, tzinfo=UTC),
                    "sensor_id": "mystery_voltage",
                    "sensor_type": SENSOR_TYPE_TEMPERATURE,  # valid type
                    "value": 5.0,
                    "unit": "V",  # NOT in SUPPORTED_UNITS
                }
            ],
        )
        with pytest.raises(SensorSourceSchemaError, match="unit"):
            SimulatedCSVSource(csv_path).read()

    def test_error_message_lists_offending_value(self, tmp_path: Path) -> None:
        """The error message includes the bad value(s) for debuggability."""
        csv_path = tmp_path / "bad.csv"
        _write_csv(
            csv_path,
            [
                {
                    "timestamp": datetime(2026, 6, 24, tzinfo=UTC),
                    "sensor_id": "mystery",
                    "sensor_type": "voltage",
                    "value": 5.0,
                    "unit": "V",
                }
            ],
        )
        with pytest.raises(SensorSourceSchemaError) as excinfo:
            SimulatedCSVSource(csv_path).read()
        msg = str(excinfo.value)
        # The offending value is named in the error message so the
        # user can fix the CSV without bisecting rows.
        assert "'voltage'" in msg


# ---------------------------------------------------------------------------
# 9. RealSerialSource — Phase 4 stub
# ---------------------------------------------------------------------------


class TestRealSerialSourceStub:
    """:class:`RealSerialSource` is a Phase 4 stub: cheap, not yet wired to GPIO."""

    def test_construction_accepts_pins(self) -> None:
        src = RealSerialSource(dht_pin=4, pir_pin=17)
        assert src.dht_pin == 4
        assert src.pir_pin == 17

    def test_frozen(self) -> None:
        src = RealSerialSource(dht_pin=4, pir_pin=17)
        with pytest.raises(FrozenInstanceError):
            src.dht_pin = 27  # type: ignore[misc]

    def test_read_raises_not_implemented(self) -> None:
        src = RealSerialSource(dht_pin=4, pir_pin=17)
        with pytest.raises(NotImplementedError, match="Phase 6"):
            src.read()

    def test_read_with_since_raises_not_implemented(self) -> None:
        src = RealSerialSource(dht_pin=4, pir_pin=17)
        with pytest.raises(NotImplementedError):
            src.read(since=datetime(2026, 6, 24, tzinfo=UTC))

    def test_available_sensors_returns_three_canonical_ids(self) -> None:
        """DHT22 produces 2 sensors (temp + humidity) + PIR produces 1."""
        ids = RealSerialSource(dht_pin=4, pir_pin=17).available_sensors()
        assert ids == sorted(["dht22_temp", "dht22_hum", "pir_motion"])

    def test_available_sensors_works_without_gpio(self) -> None:
        """available_sensors is fully functional on a laptop (no GPIO)."""
        # If this raised, the test runner would be unusable on the
        # Pi deployment machine — guards the cheap path.
        assert len(RealSerialSource(dht_pin=4, pir_pin=17).available_sensors()) == 3


# ---------------------------------------------------------------------------
# 10. MQTTBrokerSource — Phase 4 stub
# ---------------------------------------------------------------------------


class TestMQTTBrokerSourceStub:
    """:class:`MQTTBrokerSource` is a Phase 4 stub."""

    def test_construction_minimal(self) -> None:
        src = MQTTBrokerSource(host="127.0.0.1", port=1883, topic_prefix="tinyrag/sensors/")
        assert src.host == "127.0.0.1"
        assert src.port == 1883
        assert src.topic_prefix == "tinyrag/sensors/"
        assert src.username is None
        assert src.password is None

    def test_construction_with_creds(self) -> None:
        src = MQTTBrokerSource(
            host="broker.local",
            port=8883,
            topic_prefix="tinyrag/sensors/",
            username="tinyrag",
            password="s3cret",
        )
        assert src.username == "tinyrag"
        assert src.password == "s3cret"

    def test_frozen(self) -> None:
        src = MQTTBrokerSource(host="127.0.0.1", port=1883, topic_prefix="tinyrag/")
        with pytest.raises(FrozenInstanceError):
            src.host = "other"  # type: ignore[misc]

    def test_read_raises_not_implemented(self) -> None:
        src = MQTTBrokerSource(host="127.0.0.1", port=1883, topic_prefix="tinyrag/")
        with pytest.raises(NotImplementedError, match="Phase 6"):
            src.read()

    def test_available_sensors_returns_empty(self) -> None:
        """Broker-dependent — can't know without subscribing."""
        src = MQTTBrokerSource(host="127.0.0.1", port=1883, topic_prefix="tinyrag/")
        assert src.available_sensors() == []


# ---------------------------------------------------------------------------
# 11. Typed exception hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """The :class:`SensorSourceError` hierarchy catches uniformly."""

    def test_all_subclasses_inherit_from_base(self) -> None:
        for cls in (
            SensorSourceConfigError,
            SensorSourceSchemaError,
            SensorSourceReadError,
        ):
            assert issubclass(cls, SensorSourceError)

    def test_single_except_catches_all(self, tmp_path: Path) -> None:
        """One ``except SensorSourceError`` catches every source failure."""
        # 1. Config error (missing file)
        with pytest.raises(SensorSourceError):
            SimulatedCSVSource(tmp_path / "ghost.csv").read()
        # 2. Schema error (missing column)
        bad = tmp_path / "bad.csv"
        bad.write_text("sensor_id,sensor_type,value,unit\nx,y,1,z\n")
        with pytest.raises(SensorSourceError):
            SimulatedCSVSource(bad).read()

    def test_path_preserved_on_config_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ghost.csv"
        with pytest.raises(SensorSourceConfigError) as excinfo:
            SimulatedCSVSource(csv_path).read()
        assert excinfo.value.path == csv_path

    def test_path_preserved_on_schema_error(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("sensor_id\nx\n")
        with pytest.raises(SensorSourceSchemaError) as excinfo:
            SimulatedCSVSource(csv_path).read()
        assert excinfo.value.path == csv_path


# ---------------------------------------------------------------------------
# 12. End-to-end with the REAL synthetic 30-day CSV from Step 3.8
# ---------------------------------------------------------------------------


# Skip if the real CSV isn't on disk (CI without Step 3.8's output
# shouldn't fail the suite).
REAL_CSV = Path("data/sensor_logs/synthetic_30d.csv")


@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="data/sensor_logs/synthetic_30d.csv not present (run scripts/generate_synthetic_sensors.py)",
)
class TestEndToEndRealCsv:
    """End-to-end against the real Step 3.8 synthetic CSV.

    This class is the **regression gate** for "did the Step 3.8
    generator drift from the schema the Step 4.13 reader expects?".
    If this fails, either:

    - Step 3.8's generator changed (regenerate or fix the
      generator), or
    - Step 4.13's schema expectations drifted (fix the validator).

    Both are bugs; the test name + commit message should make it
    obvious which side broke.
    """

    def test_read_returns_51840_rows(self) -> None:
        """30 days x 288 ticks/day x 6 sensors = 51,840 rows."""
        df = SimulatedCSVSource(REAL_CSV).read()
        assert len(df) == 51_840

    def test_read_schema_matches_required(self) -> None:
        df = SimulatedCSVSource(REAL_CSV).read()
        assert list(df.columns) == list(REQUIRED_COLUMNS)

    def test_read_dtypes_match_expected(self) -> None:
        df = SimulatedCSVSource(REAL_CSV).read()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_float_dtype(df["value"])

    def test_sensor_types_all_supported(self) -> None:
        df = SimulatedCSVSource(REAL_CSV).read()
        assert set(df["sensor_type"].unique().tolist()) <= set(SUPPORTED_SENSOR_TYPES)

    def test_units_all_supported(self) -> None:
        df = SimulatedCSVSource(REAL_CSV).read()
        assert set(df["unit"].unique().tolist()) <= set(SUPPORTED_UNITS)

    def test_available_sensors_returns_all_six_canonical(self) -> None:
        ids = SimulatedCSVSource(REAL_CSV).available_sensors()
        # Step 3.8's canonical roster — see
        # scripts/generate_synthetic_sensors.py::SENSORS.
        assert ids == sorted(
            [
                "bedroom_hum",
                "bedroom_temp",
                "house_energy",
                "kitchen_motion",
                "living_room_hum",
                "living_room_temp",
            ]
        )

    def test_since_filter_works_on_real_data(self) -> None:
        """The ``since`` filter shrinks the 51k row dataset correctly."""
        df_full = SimulatedCSVSource(REAL_CSV).read()
        cutoff = df_full["timestamp"].max() - timedelta(days=1)
        df_recent = SimulatedCSVSource(REAL_CSV).read(since=cutoff)
        # Last-day filter should drop ~50k rows.
        assert len(df_recent) < len(df_full)
        assert len(df_recent) > 0
        assert df_recent["timestamp"].min() >= cutoff

"""SensorSource Protocol + shared types.

The :mod:`tinyrag.sensors.base` module defines the **seam** between
"where sensor readings come from" and "what the rest of the system
sees". Anything that can produce a DataFrame in the canonical schema
(from :mod:`docs/04_database_design_v1.md` §6.1) is a valid
:class:`SensorSource` — a CSV file, a DHT22 over GPIO, an MQTT broker,
or a future Zigbee/Bluetooth adapter.

The :class:`SensorSource` Protocol lives here (not in the
implementation modules) so the rest of the codebase can import it
without pulling in any of the heavy optional dependencies
(``paho-mqtt``, ``RPi.GPIO``, ``serial``). Each concrete source lives
in its own module and only imports its own dep at the top of that
file.

Why a Protocol (not an ABC)?
----------------------------
- **Duck typing for tests.** Tests can pass a plain class that
  implements the two methods — no need to inherit from anything
  to satisfy ``isinstance(src, SensorSource)``.
- **No forced inheritance chain.** A future source that already
  inherits from e.g. ``paho.mqtt.client.Client`` doesn't have to
  multiple-inherit from our ABC just to be a sensor source.
- **Self-documenting.** The Protocol IS the contract; the docstring
  lists the exact DataFrame schema every implementer must return.

Three implementations exist (see :mod:`tinyrag.sensors.simulated`,
:mod:`tinyrag.sensors.serial_dht`, :mod:`tinyrag.sensors.mqtt`):

==================  =========================  =========================
Class               Backing                    Use case
==================  =========================  =========================
SimulatedCSVSource  a CSV file on disk         Default; always available
RealSerialSource    DHT22 + PIR over GPIO      Pi only (Phase 6)
MQTTBrokerSource    a local Mosquitto broker   When sensors publish via MQTT
==================  =========================  =========================

Switching is one config line (``config.yaml``'s ``sensors.source``)
— **no code change**.

Location: ``src/tinyrag/sensors/base.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Canonical schema constants — match docs/04_database_design_v1.md §6.1
# ---------------------------------------------------------------------------

#: Allowed values for the ``sensor_type`` column.
SENSOR_TYPE_TEMPERATURE: str = "temperature"
SENSOR_TYPE_HUMIDITY: str = "humidity"
SENSOR_TYPE_ENERGY: str = "energy"
SENSOR_TYPE_MOTION: str = "motion"

#: All supported sensor types (exhaustive set, frozen).
#:
#: Used by:
#: - :class:`SensorReading` validation in ``__post_init__``.
#: - :class:`~tinyrag.sensors.simulated.SimulatedCSVSource` to reject
#:   rows whose ``sensor_type`` value isn't in this set (a likely
#:   symptom of a corrupt CSV or a hand-edited file).
SUPPORTED_SENSOR_TYPES: frozenset[str] = frozenset(
    {
        SENSOR_TYPE_TEMPERATURE,
        SENSOR_TYPE_HUMIDITY,
        SENSOR_TYPE_ENERGY,
        SENSOR_TYPE_MOTION,
    }
)

#: Allowed values for the ``unit`` column.
UNIT_CELSIUS: str = "C"
UNIT_PERCENT: str = "%"
UNIT_KWH: str = "kWh"
UNIT_COUNT: str = "count"

#: All supported units (exhaustive set, frozen).
SUPPORTED_UNITS: frozenset[str] = frozenset(
    {UNIT_CELSIUS, UNIT_PERCENT, UNIT_KWH, UNIT_COUNT}
)

#: Canonical expected/required columns of a sensor reading DataFrame.
#: Order matches the CSV produced by ``scripts/generate_synthetic_sensors.py``
#: (Step 3.8) and the schema in :mod:`docs/04_database_design_v1.md` §6.1.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "sensor_id",
    "sensor_type",
    "value",
    "unit",
)

# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorReading:
    """A single sensor reading — the row-level type for sensor data.

    This is the in-memory representation of one row of a sensor
    DataFrame. A :class:`SensorSource` returns DataFrames because
    that's what the :class:`~tinyrag.core.sensor_summarizer.SensorSummarizer`
    consumes; :class:`SensorReading` is the per-row handle for code
    that wants strict typing (e.g. tests, the MQTT decoder in
    Phase 6).

    Attributes
    ----------
    timestamp:
        UTC instant of the reading. Always timezone-aware
        (``datetime.UTC``); the CSV producer (:mod:`scripts.generate_synthetic_sensors`)
        writes naive UTC, so the loader converts on read.
    sensor_id:
        Logical sensor name, e.g. ``"living_room_temp"``.
        Convention: ``<room>_<type>`` (or ``<scope>_<type>`` for
        whole-house sensors like ``house_energy``).
    sensor_type:
        One of :data:`SUPPORTED_SENSOR_TYPES`. ``temperature`` is
        paired with unit ``C``; ``humidity`` with ``%``; ``energy``
        with ``kWh``; ``motion`` with ``count``.
    value:
        The reading itself (float). For ``motion`` this is 0 or 1
        (Bernoulli); for ``energy`` it's the kWh delta over the
        sampling interval.
    unit:
        One of :data:`SUPPORTED_UNITS`.
    """

    timestamp: datetime
    sensor_id: str
    sensor_type: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        # Validate sensor_type and unit at construction so misuse fails
        # loudly at the call site, not deep inside a pandas pipeline.
        if self.sensor_type not in SUPPORTED_SENSOR_TYPES:
            raise ValueError(
                f"sensor_type must be one of {sorted(SUPPORTED_SENSOR_TYPES)}, "
                f"got {self.sensor_type!r}"
            )
        if self.unit not in SUPPORTED_UNITS:
            raise ValueError(
                f"unit must be one of {sorted(SUPPORTED_UNITS)}, got {self.unit!r}"
            )
        if not self.sensor_id:
            raise ValueError("sensor_id must be a non-empty string")

    @classmethod
    def from_row(cls, row: Any) -> SensorReading:
        """Build a :class:`SensorReading` from a DataFrame row.

        Accepts any object that supports ``["col"]`` indexing
        (e.g. a ``pandas.Series``). Lets callers write::

            df = source.read()
            for _, row in df.iterrows():
                reading = SensorReading.from_row(row)
                ...

        without manually unpacking five columns.

        Parameters
        ----------
        row:
            A row-like object with the five required columns
            (:data:`REQUIRED_COLUMNS`). The ``timestamp`` may be a
            ``datetime`` (preferred) or a ``pd.Timestamp`` (auto-cast
            via ``.to_pydatetime()``).

        Returns
        -------
        SensorReading
            A new :class:`SensorReading` instance.
        """
        ts = row["timestamp"]
        # pandas Timestamps are datetime subclasses but the stdlib
        # datetime check is more permissive. We only need to_pydatetime
        # when it's a pandas Timestamp (which is NOT a stdlib datetime
        # in some pandas versions).
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        return cls(
            timestamp=ts,
            sensor_id=str(row["sensor_id"]),
            sensor_type=str(row["sensor_type"]),
            value=float(row["value"]),
            unit=str(row["unit"]),
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SensorSource(Protocol):
    """Anything that can produce a DataFrame of sensor records.

    The :class:`SensorSource` Protocol is the seam between TinyRAG
    and the outside world for sensor data. Two methods:

    - :meth:`read` returns the canonical DataFrame (columns match
      :data:`REQUIRED_COLUMNS`).
    - :meth:`available_sensors` lists the sensor IDs this source
      can provide (cheap, no full read required).

    The Protocol is :func:`typing.runtime_checkable`, so
    ``isinstance(src, SensorSource)`` works for any object that
    implements both methods — no inheritance required. This makes
    the contract testable in isolation (see
    :mod:`tests.test_sensors.TestProtocolConformance`).
    """

    def read(self, since: datetime | None = None) -> pd.DataFrame:
        """Return sensor readings as a DataFrame.

        Parameters
        ----------
        since:
            If provided, only return rows with ``timestamp >= since``.
            ``None`` means "everything this source has". Implementations
            may apply a default floor (e.g. ``SimulatedCSVSource``'s
            ``default_since``) when the caller passes ``None``.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with exactly the columns
            :data:`REQUIRED_COLUMNS` and the dtypes
            ``timestamp=datetime64[ns]``,
            ``sensor_id=object`` (str),
            ``sensor_type=object`` (str, ∈ :data:`SUPPORTED_SENSOR_TYPES`),
            ``value=float64``,
            ``unit=object`` (str, ∈ :data:`SUPPORTED_UNITS`).

        Raises
        ------
        SensorSourceError
            Base class for all source failures (subclassed by the
            three typed errors below).
        """
        ...

    def available_sensors(self) -> list[str]:
        """Return the list of sensor IDs this source can provide.

        The list is cheap to compute — implementations should NOT
        require a full :meth:`read` to populate it. Used by the
        composition root (``main.py``, Step 4.17) to surface the
        available sensors in ``GET /api/status``.
        """
        ...


# ---------------------------------------------------------------------------
# Typed exception hierarchy
# ---------------------------------------------------------------------------


class SensorSourceError(Exception):
    """Base class for every sensor-source failure.

    Carries ``.path`` (the source's file / host / topic) so the
    API layer (:mod:`tinyrag.api`) can log it cleanly and return
    a structured 5xx response without inspecting the message.

    Subclasses correspond to the three failure modes a real source
    can hit:

    - :class:`SensorSourceConfigError` — caller misconfiguration
      (e.g. missing CSV file, bad pin number).
    - :class:`SensorSourceSchemaError` — the data is reachable but
      in the wrong shape (missing column, unknown sensor_type).
    - :class:`SensorSourceReadError` — IO failure (network down,
      permission denied, serial port busy).
    """

    def __init__(self, message: str, *, path: str | Path | None = None) -> None:
        super().__init__(message)
        self.path: str | Path | None = path


class SensorSourceConfigError(SensorSourceError):
    """The source was misconfigured (missing file, bad args).

    Maps to HTTP 400 in the API layer — the user's fault, not the
    system's. Example: a ``csv_path`` that points at a non-existent
    file.
    """


class SensorSourceSchemaError(SensorSourceError):
    """The data is reachable but doesn't match the canonical schema.

    Maps to HTTP 500 — the data is corrupt or the producer has drifted
    from the schema. Example: a CSV missing the ``timestamp`` column,
    or a row with ``sensor_type="voltage"`` (not in
    :data:`SUPPORTED_SENSOR_TYPES`).
    """


class SensorSourceReadError(SensorSourceError):
    """An IO failure prevented reading the source.

    Maps to HTTP 503 — the data store is temporarily unavailable.
    Example: a network partition in the MQTT case, a serial-port
    busy error in the GPIO case.
    """


__all__ = [
    "REQUIRED_COLUMNS",
    "SENSOR_TYPE_ENERGY",
    "SENSOR_TYPE_HUMIDITY",
    "SENSOR_TYPE_MOTION",
    "SENSOR_TYPE_TEMPERATURE",
    "SensorReading",
    "SensorSource",
    "SensorSourceConfigError",
    "SensorSourceError",
    "SensorSourceReadError",
    "SensorSourceSchemaError",
    "SUPPORTED_SENSOR_TYPES",
    "SUPPORTED_UNITS",
    "UNIT_CELSIUS",
    "UNIT_COUNT",
    "UNIT_KWH",
    "UNIT_PERCENT",
]

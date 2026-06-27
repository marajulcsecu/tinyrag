"""SimulatedCSVSource — reads sensor readings from a canonical CSV file.

The :class:`SimulatedCSVSource` is the default
:class:`~tinyrag.sensors.base.SensorSource` for the laptop path
(``config.yaml``'s ``sensors.source: simulated``). It reads the
long-format CSV produced by ``scripts/generate_synthetic_sensors.py``
(Step 3.8) and returns a :class:`pandas.DataFrame` whose columns
exactly match :data:`~tinyrag.sensors.base.REQUIRED_COLUMNS`.

Why this is the default
-----------------------
- **Always available.** The CSV is generated at setup time by
  ``make generate-sensors``. No GPIO, no network, no broker.
- **Reproducible.** The generator pins ``SEED=42`` so every fresh
  setup produces the same numbers — critical for the Phase 5 eval
  set ("what was the average living-room temperature yesterday?").
- **Realistic enough for retrieval.** The generator uses daily
  sinusoids + per-room offsets + weekend motion multipliers, so the
  data has the patterns the LLM needs to actually answer sensor
  questions meaningfully.

Schema (matches ``docs/04_database_design_v1.md`` §6.1)
--------------------------------------------------------

::

    timestamp,sensor_id,sensor_type,value,unit
    2026-05-24T00:00:00,living_room_temp,temperature,20.34,C
    2026-05-24T00:00:00,living_room_hum,humidity,58.10,%
    2026-05-24T00:00:00,kitchen_motion,motion,0,count
    2026-05-24T00:00:00,house_energy,energy,0.08,kWh

:class:`SimulatedCSVSource` is the single point where that CSV
becomes a typed DataFrame. Validation is **strict** by design —
wrong columns, unknown sensor types, or bad units are rejected with
:class:`~tinyrag.sensors.base.SensorSourceSchemaError`, not silently
coerced. The eval set depends on the data being exactly what the
schema promises.

Location: ``src/tinyrag/sensors/simulated.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tinyrag.sensors.base import (
    REQUIRED_COLUMNS,
    SUPPORTED_SENSOR_TYPES,
    SUPPORTED_UNITS,
    SensorSourceConfigError,
    SensorSourceSchemaError,
)

#: The CSV dtype map. Centralised here (not inside ``read()``) so
#: it's grep-able from a REPL probe and stable across calls.
#: ``sensor_id`` / ``sensor_type`` / ``unit`` are kept as plain
#: Python strings (object dtype in pandas) so the
#: ``isin(SUPPORTED_*)`` validation that follows can report the
#: offending literal value back to the user.
_CSV_DTYPES: dict[str, str] = {
    "sensor_id": "string",
    "sensor_type": "string",
    "value": "float64",
    "unit": "string",
}


@dataclass(frozen=True)
class SimulatedCSVSource:
    """A :class:`SensorSource` backed by a CSV file on disk.

    The class is a frozen dataclass (matches the project's pattern
    of immutable, dependency-injected seams — :class:`Chunker`,
    :class:`Retriever`, :class:`MetadataStore`). The ``path`` is
    required; ``default_since`` is an optional floor applied when
    the caller passes ``since=None`` to :meth:`read`.

    Parameters
    ----------
    path:
        Filesystem path to the CSV. Both :class:`str` and
        :class:`pathlib.Path` are accepted; ``Path`` is the
        canonical form internally.
    default_since:
        If provided, every call to :meth:`read` with
        ``since=None`` applies this floor. Useful for the
        ``scripts/ingest_sensors.py`` CLI (Step 4.15) which wants
        to ingest "everything from the last 7 days" without
        recomputing the cutoff on every run.

    Attributes
    ----------
    path
    default_since

    Examples
    --------
    >>> from pathlib import Path
    >>> from datetime import datetime, timedelta, UTC
    >>> src = SimulatedCSVSource("data/sensor_logs/synthetic_30d.csv")
    >>> df = src.read(since=datetime.now(UTC) - timedelta(days=1))
    >>> df.columns.tolist()  # doctest: +SKIP
    ['timestamp', 'sensor_id', 'sensor_type', 'value', 'unit']
    """

    path: Path
    default_since: datetime | None = None

    def __post_init__(self) -> None:
        # Normalise to Path so downstream code never has to handle
        # both str and Path. The dataclass is frozen so we use
        # object.__setattr__ to mutate the field.
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))

    # ------------------------------------------------------------------
    # SensorSource protocol
    # ------------------------------------------------------------------

    def read(self, since: datetime | None = None) -> pd.DataFrame:
        """Read sensor readings from the CSV, optionally filtered by time.

        Steps:

        1. Verify the file exists. → :class:`SensorSourceConfigError`
           if not (maps to HTTP 400 — the user's fault, not the
           data's).
        2. Read the CSV with :func:`pandas.read_csv` and explicit
           dtypes. ``parse_dates=["timestamp"]`` so the column
           comes back as ``datetime64[ns]`` ready for
           ``.dt.floor(...)`` / ``.dt.hour`` access in the
           summarizer.
        3. Validate the column set is **exactly**
           :data:`REQUIRED_COLUMNS` (no missing, no extras).
           → :class:`SensorSourceSchemaError` otherwise.
        4. Validate that every ``sensor_type`` value is in
           :data:`SUPPORTED_SENSOR_TYPES` and every ``unit`` value
           is in :data:`SUPPORTED_UNITS`. Vectorised via
           ``.isin(...)``; the offending distinct value is included
           in the error message so a user with a hand-edited CSV
           can fix it without bisecting.
        5. Apply the ``since`` filter (``since`` arg, falling back to
           ``default_since``). Returns the **filtered** DataFrame;
           does not sort (the CSV is append-only per
           ``docs/04_database_design_v1.md`` §6.3 — caller order is
           preserved).

        Parameters
        ----------
        since:
            Optional floor on ``timestamp``. Rows with
            ``timestamp < since`` are dropped. ``None`` means "no
            filter unless ``default_since`` is set".

        Returns
        -------
        pandas.DataFrame
            Filtered, schema-validated DataFrame. May be empty
            (e.g. when ``since > max(timestamp)``); an empty
            DataFrame is a valid result, not an error.
        """
        if not self.path.exists():
            raise SensorSourceConfigError(
                f"Sensor CSV not found at {self.path}",
                path=self.path,
            )

        # Column validation BEFORE pd.read_csv — the reader raises
        # an opaque "Missing column provided to 'parse_dates'" if
        # we let it try to parse a column that isn't there, and the
        # user can't tell that a typo in the header is the cause.
        # A one-line preview read with usecols=REQUIRED_COLUMNS
        # catches the column set cheaply (pandas only parses the
        # named columns).
        try:
            preview_cols = pd.read_csv(
                self.path, nrows=0
            ).columns.tolist()
        except (OSError, ValueError) as exc:
            raise SensorSourceSchemaError(
                f"Failed to read sensor CSV header at {self.path}: {exc}",
                path=self.path,
            ) from exc

        actual_cols = set(preview_cols)
        expected_cols = set(REQUIRED_COLUMNS)
        if actual_cols != expected_cols:
            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols
            parts: list[str] = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if extra:
                parts.append(f"unexpected={sorted(extra)}")
            raise SensorSourceSchemaError(
                f"Sensor CSV at {self.path} has wrong columns "
                f"({', '.join(parts)}); expected exactly "
                f"{list(REQUIRED_COLUMNS)}",
                path=self.path,
            )

        # Read. We deliberately use pd.read_csv directly (not a
        # helper) so the parse_dates + dtype kwargs are visible at
        # the call site — future readers can trace exactly how the
        # DataFrame is constructed.
        try:
            df = pd.read_csv(
                self.path,
                dtype=_CSV_DTYPES,
                parse_dates=["timestamp"],
            )
        except (OSError, ValueError) as exc:
            # pd.read_csv raises OSError for IO failures and
            # ValueError for malformed bytes; both map to a
            # sensor-source schema error because they mean
            # "the data is there but not readable as a CSV".
            raise SensorSourceSchemaError(
                f"Failed to parse sensor CSV at {self.path}: {exc}",
                path=self.path,
            ) from exc

        # sensor_type / unit validation — vectorised .isin().
        # ``unique()`` then sorted-list for a stable error message.
        bad_types = sorted(
            df.loc[~df["sensor_type"].isin(SUPPORTED_SENSOR_TYPES), "sensor_type"]
            .unique()
            .tolist()
        )
        if bad_types:
            raise SensorSourceSchemaError(
                f"Sensor CSV at {self.path} contains unknown "
                f"sensor_type value(s) {bad_types}; allowed: "
                f"{sorted(SUPPORTED_SENSOR_TYPES)}",
                path=self.path,
            )
        bad_units = sorted(
            df.loc[~df["unit"].isin(SUPPORTED_UNITS), "unit"]
            .unique()
            .tolist()
        )
        if bad_units:
            raise SensorSourceSchemaError(
                f"Sensor CSV at {self.path} contains unknown "
                f"unit value(s) {bad_units}; allowed: "
                f"{sorted(SUPPORTED_UNITS)}",
                path=self.path,
            )

        # Apply the since floor. ``default_since`` only fires when
        # the caller passes ``since=None`` — an explicit ``since``
        # always wins so a user can override the source's default.
        floor = since if since is not None else self.default_since
        if floor is not None:
            # Align ``floor``'s tz-awareness with the column's
            # tz-awareness so the comparison succeeds. The CSV
            # may be loaded as tz-naive (``datetime64[ns]``) or
            # tz-aware (``datetime64[ns, UTC]``) depending on the
            # pandas version + the timestamp strings in the file;
            # comparing a tz-aware ``floor`` against a tz-naive
            # column (or vice-versa) raises
            # ``TypeError: Invalid comparison between dtype=...``.
            # We always localise-naive → localise to UTC when the
            # column is tz-aware, and localise-aware → strip tz
            # when the column is tz-naive.
            col_is_tz_aware = bool(
                getattr(df["timestamp"].dt, "tz", None) is not None
            )
            if col_is_tz_aware and floor.tzinfo is None:
                floor = floor.replace(tzinfo=UTC)
            elif (not col_is_tz_aware) and floor.tzinfo is not None:
                floor = floor.replace(tzinfo=None)
            df = df.loc[df["timestamp"] >= floor].reset_index(drop=True)

        return df

    def available_sensors(self) -> list[str]:
        """Return the sorted, unique list of sensor IDs in the CSV.

        Cheap — reads only the ``sensor_id`` column (pandas is
        smart enough not to materialise the rest). Used by the
        API ``/api/status`` endpoint (Step 4.17) so the UI can show
        which sensors are online without doing a full read.

        Returns
        -------
        list[str]
            Sorted list of distinct sensor IDs (e.g.
            ``["bedroom_hum", "bedroom_temp", "house_energy",
            "kitchen_motion", "living_room_hum",
            "living_room_temp"]`` for the synthetic 30-day CSV).

        Raises
        ------
        SensorSourceConfigError
            If the CSV file does not exist. (We don't bother
            validating the rest of the schema here — that's
            :meth:`read`'s job; this method exists to answer "is
            there a file at all?" cheaply.)
        """
        if not self.path.exists():
            raise SensorSourceConfigError(
                f"Sensor CSV not found at {self.path}",
                path=self.path,
            )

        # Read only the one column. ``usecols`` makes pandas skip
        # parsing the other four (faster + lower memory for the
        # 51k-row synthetic CSV).
        df = pd.read_csv(self.path, usecols=["sensor_id"], dtype={"sensor_id": "string"})
        return sorted(df["sensor_id"].dropna().unique().tolist())


__all__ = ["SimulatedCSVSource"]

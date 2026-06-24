"""SensorSummarizer — turns a sensor DataFrame into human-readable text Chunks.

The :class:`SensorSummarizer` is the **chunking** step of the sensor
ingest pipeline. It consumes the canonical DataFrame produced by a
:class:`~tinyrag.sensors.base.SensorSource` (Step 4.13) and emits
a list of :class:`~tinyrag.core.chunker.Chunk` objects ready for
embedding + indexing in the sensor FAISS index (Step 4.15).

Why a dedicated summarizer (not just embed-the-rows)?
-----------------------------------------------------
Raw sensor rows ("22.34, 22.31, 22.45, ...") embed poorly:

- A 5-minute temperature tick and a midnight temperature tick
  look almost identical to an embedding model — there's no
  contextual signal for "this is yesterday's living room".
- A user query like *"What was the temperature yesterday?"*
  matches against a row's literal value, not against its time
  window.
- Per-row embedding explodes the index size — 51,840 rows x 384
  dims = 80 MB just for the synthetic 30-day CSV.

Summarising to per-day, per-sensor text chunks ("On 2026-06-15, the
living room temperature averaged 24.3°C, peaking at 27.1°C at
16:00, and reaching a minimum of 19.2°C at 05:30.") gives the
embedding model the **time + room + statistic** context it needs
to match natural-language questions, while shrinking the index
by ~300x (30 days x 6 sensors = 180 chunks instead of 51,840 rows).

The output format follows the examples in
:mod:`docs/04_database_design_v1.md` §6.4 verbatim — the LLM in
Step 4.16 will answer "What was the temperature yesterday?" by
citing the matching summary chunk.

Why daily-only (not hourly)?
----------------------------
The architecture doc lists hourly summaries as a future option
("for high-frequency queries"). The roadmap Step 4.14 spec says:
*"Default mode: per-day, per-sensor-type summaries (avg, min,
max, peak time). Special handling for `motion` (event-based, not
stats)."* The summarizer carries a ``window`` parameter as a hook
for a future "hourly" mode, but the implementation is
daily-only — the Step 5 eval set will reveal if hourly is
actually needed and we can extend without changing the public
API.

Pure-function module
--------------------
Like every module in :mod:`tinyrag.core`, this one has **no I/O**
and **no dependency on** :mod:`tinyrag.sensors` (the
``core/__init__.py`` docstring enforces this one-way dep rule).
The summarizer re-declares the small sensor-type set it needs
locally (see :data:`_NUMERIC_SENSOR_TYPES` and
:data:`_MOTION_SENSOR_TYPE`) — a future contributor adding a
sensor type must update both this file AND
:mod:`tinyrag.sensors.base`.

Location: ``src/tinyrag/core/sensor_summarizer.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd
import tiktoken

from tinyrag.core.chunker import Chunk

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Sensor-type constants (re-declared locally — see module docstring)
# ---------------------------------------------------------------------------
#
# These must stay in lockstep with :mod:`tinyrag.sensors.base`. The
# core package is not allowed to import from sensors (one-way dep
# rule in :mod:`tinyrag.core.__init__`), so we re-declare the small
# subset the summarizer needs. Adding a new sensor type? Update
# BOTH places.

#: Sensor types summarised with the numeric path (avg / min / max / peak time).
#: ``motion`` is the only excluded type (handled separately because
#: it's event-based, not statistical).
_NUMERIC_SENSOR_TYPES: frozenset[str] = frozenset({"temperature", "humidity", "energy"})

#: The single sensor type that uses the event-log summary path.
_MOTION_SENSOR_TYPE: str = "motion"

#: Sensor-id suffix → human-readable room name. The synthetic CSV
#: (and real-world deployments) follow the convention
#: ``<room>_<type>`` (e.g. ``living_room_temp`` → room="living room").
#: Whole-house sensors like ``house_energy`` fall back to the
#: literal ``sensor_id`` because there's no room suffix to strip.
_ROOM_SUFFIXES: tuple[str, ...] = ("_temp", "_hum", "_motion", "_energy")

#: Default tiktoken encoding — same default as
#: :class:`~tinyrag.core.chunker.Chunker`. Keeping the two in sync
#: means a chunk's ``token_count`` field matches the chunker's count
#: for the same text, which keeps the prompt builder's budget math
#: consistent across doc and sensor chunks.
_DEFAULT_ENCODING = "cl100k_base"

#: Cap on the number of motion timestamps enumerated verbatim. With
#: this many or fewer active ticks, we list each ``at HH:MM, ...``
#: in the summary text; with more, we fall back to the
#: "N events, first at ..., last at ..." form to avoid token bloat.
_MOTION_VERBATIM_LIMIT: int = 5

#: Sentinel value used when a numeric group has no values to take
#: the mean / argmax / argmin of. Should never trigger in practice
#: (groupby drops empty groups), but the defensive default keeps
#: the renderer from raising on a malformed input.
_EMPTY_FALLBACK_FLOAT: float = 0.0
_EMPTY_FALLBACK_TIME_STR: str = "00:00"


# ---------------------------------------------------------------------------
# Typed exception hierarchy
# ---------------------------------------------------------------------------


class SensorSummarizerError(Exception):
    """Base class for every :class:`SensorSummarizer` failure.

    Subclasses correspond to the two ways a summarizer call can
    fail:

    - :class:`SensorSummarizerSchemaError` — the DataFrame is
      missing one of the required columns or has the wrong dtypes.
    - :class:`SensorSummarizerEmptyError` — the DataFrame is well-
      formed but produced zero chunks (all rows were filtered out
      by the summary logic). Empty results are usually a caller
      bug (a ``since`` filter too tight, a wrong date range), so
      we raise rather than return ``[]`` — callers shouldn't have
      to defensively check the return value.
    """


class SensorSummarizerSchemaError(SensorSummarizerError):
    """The DataFrame doesn't match the canonical sensor schema."""


class SensorSummarizerEmptyError(SensorSummarizerError):
    """The DataFrame produced zero summary chunks.

    Maps cleanly to "no data" in the API layer; the caller
    usually wants to surface "no sensor readings found for that
    time range" rather than silently producing an empty list.
    """


# ---------------------------------------------------------------------------
# The summarizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorSummarizer:
    """Pure-function summarizer: DataFrame → list[:class:`Chunk`].

    Frozen dataclass (matches the project's pattern: immutable,
    stateless, dependency-free). All behaviour is parameterised
    by the constructor arguments so tests can pin the output
    format exactly.

    Parameters
    ----------
    window:
        Summary window. Currently only ``"daily"`` is implemented;
        the kwarg exists so a future ``"hourly"`` mode is a single
        if-branch away without breaking the API.
    time_fmt:
        :func:`datetime.strftime` pattern used to render the
        "peaked at HH:MM" / "minimum at HH:MM" timestamps inside
        numeric summaries, and the "at HH:MM, HH:MM, ..." motion
        event list. Default ``"%H:%M"`` (24-hour, no seconds).
        12-hour with AM/PM is a one-kwarg change.
    date_fmt:
        :func:`datetime.strftime` pattern used to render the
        "On YYYY-MM-DD, ..." date prefix. Default ``"%Y-%m-%d"``
        (ISO 8601 date).
    source_label:
        The :attr:`Chunk.source` value applied to every emitted
        chunk. Default ``"sensor-summary"`` matches the
        ``doc_type='sensor_summary'`` value in
        :mod:`docs/04_database_design_v1.md` §5.
    encoding_name:
        tiktoken encoding used for the ``token_count`` field on
        every emitted :class:`Chunk`. Default ``"cl100k_base"`` —
        same as :class:`~tinyrag.core.chunker.Chunker` so the
        prompt builder's budget math is consistent across doc and
        sensor chunks.

    Attributes
    ----------
    window
    time_fmt
    date_fmt
    source_label
    encoding_name

    Notes
    -----
    Validation is deliberately **minimal**: a bad ``window`` value
    raises :class:`ValueError` at construction (so a misconfig
    fails fast at startup, not at first call), but the format
    strings and tiktoken encoding are looked up lazily inside
    :meth:`summarize` so a bad value surfaces with the line of
    pandas context that triggered it. tiktoken errors are
    wrapped in :class:`SensorSummarizerSchemaError` to match the
    other "schema is wrong" failures.
    """

    window: Literal["daily"] = "daily"
    time_fmt: str = "%H:%M"
    date_fmt: str = "%Y-%m-%d"
    source_label: str = "sensor-summary"
    encoding_name: str = _DEFAULT_ENCODING

    # Cached tiktoken encoder — lazily resolved on first call so a
    # misconfig doesn't fail at construction. The encoder is
    # process-global in tiktoken (encoding_for_model returns the
    # same instance per name), so caching is safe.
    _encoder: tiktoken.Encoding = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.window != "daily":
            raise ValueError(
                f"SensorSummarizer.window must be 'daily' (the only "
                f"implemented mode); got {self.window!r}"
            )
        # Pre-resolve the encoder so a bad encoding name fails here
        # with a clean ValueError, not at first summarize() call.
        try:
            object.__setattr__(
                self, "_encoder", tiktoken.get_encoding(self.encoding_name)
            )
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Unknown tiktoken encoding {self.encoding_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, df: pd.DataFrame) -> list[Chunk]:
        """Convert a sensor DataFrame into per-day summary :class:`Chunk`\\s.

        Algorithm (architecture §6.4, "daily summary chunk"):

        1. **Validate** the input has the 5 required columns
           (``timestamp``, ``sensor_id``, ``sensor_type``, ``value``,
           ``unit``). → :class:`SensorSummarizerSchemaError` if not.
        2. **Coerce** ``timestamp`` to ``datetime64[ns]`` (defensive —
           the canonical :class:`~tinyrag.sensors.simulated.SimulatedCSVSource`
           already does this, but the API contract says "any
           DataFrame").
        3. **Group** by ``(date, sensor_id)`` (the "per-day, per-sensor-
           type" the roadmap requires).
        4. **Dispatch** each group to a renderer:

           - ``temperature`` / ``humidity`` / ``energy`` → numeric
             summary (mean, min, max, peak time, trough time).
           - ``motion`` → event log (active timestamps; collapsed to
             "N events" form when there are more than
             :data:`_MOTION_VERBATIM_LIMIT`).
           - any other ``sensor_type`` → silently skipped (the
             loader is responsible for rejecting unknown types;
             the summarizer's defensive ``continue`` keeps a
             future schema drift from crashing ingest).

        5. **Build** one :class:`Chunk` per group with:
           ``text`` = rendered summary, ``source`` = ``source_label``,
           ``page`` = ``None``, ``chunk_index`` = global ordinal
           0..N-1, ``char_offset`` = 0, ``token_count`` = tiktoken-
           encoded length of ``text``.
        6. **Return** the list. Empty result → :class:`SensorSummarizerEmptyError`
           so a caller with a too-tight ``since`` filter gets a
           clear error instead of a silent zero-length answer.

        Parameters
        ----------
        df:
            A DataFrame in the canonical sensor schema (see
            :class:`~tinyrag.sensors.base.SensorSource`).
            Additional columns are tolerated and ignored.

        Returns
        -------
        list[Chunk]
            One :class:`Chunk` per ``(date, sensor_id)`` group,
            ordered by ``(date, sensor_id)`` for stable citation
            ordering downstream. ``chunk_index`` is the global
            ordinal (0..N-1) — same invariant the document chunker
            and ingest CLI enforce.
        """
        self._validate_columns(df)
        df = self._ensure_timestamp(df)

        # Day partition. ``.dt.date`` returns Python ``date``
        # objects (not timestamps) — groupby on those is robust
        # against timezone edge cases.
        work = df.copy()
        work["_date"] = work["timestamp"].dt.date

        chunks: list[Chunk] = []
        chunk_index = 0
        # Stable ordering: sort by (date, sensor_id) before grouping
        # so the output order is reproducible regardless of the
        # caller's row order. The CSV is append-only, but a future
        # in-memory SensorSource may insert out of order.
        for (date_value, _sensor_id), group in (
            work.groupby(["_date", "sensor_id"], sort=True)
        ):
            sensor_type = str(group["sensor_type"].iloc[0])
            if sensor_type in _NUMERIC_SENSOR_TYPES:
                text = self._summarize_numeric(group, date_value)
            elif sensor_type == _MOTION_SENSOR_TYPE:
                text = self._summarize_motion(group, date_value)
            else:
                # Unknown type — defensive skip. Shouldn't happen
                # if the loader validated, but a future contributor
                # adding a sensor type shouldn't have to update
                # the summarizer AND the validator atomically.
                continue

            token_count = len(self._encoder.encode(text))
            chunks.append(
                Chunk(
                    text=text,
                    source=self.source_label,
                    page=None,
                    chunk_index=chunk_index,
                    char_offset=0,
                    token_count=token_count,
                )
            )
            chunk_index += 1

        if not chunks:
            raise SensorSummarizerEmptyError(
                "SensorSummarizer produced 0 chunks from the input "
                "DataFrame — check the data range and sensor types."
            )

        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Reject DataFrames missing the canonical 5 columns."""
        required = {"timestamp", "sensor_id", "sensor_type", "value", "unit"}
        actual = set(df.columns)
        missing = required - actual
        if missing:
            raise SensorSummarizerSchemaError(
                f"SensorSummarizer.summarize requires columns "
                f"{sorted(required)}; missing {sorted(missing)} "
                f"from input DataFrame (got {sorted(actual)})"
            )

    @staticmethod
    def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce ``timestamp`` to ``datetime64[ns]`` if it isn't already.

        The canonical :class:`~tinyrag.sensors.simulated.SimulatedCSVSource`
        already produces ``datetime64[ns]``, but the summarizer's
        API contract says "any DataFrame", and a hand-built test
        fixture might pass strings or ``pd.Timestamp`` objects.
        Defensive coercion keeps the method easy to use in REPL
        probes and ad-hoc scripts.
        """
        col = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(col):
            # Even when the dtype is right, NaT entries mean a
            # prior coercion failed silently (or a user built the
            # DataFrame with explicit NaT). Treat them as bad
            # input and fail loudly rather than producing
            # nonsensical summaries.
            if col.isna().any():
                n_bad = int(col.isna().sum())
                raise SensorSummarizerSchemaError(
                    f"SensorSummarizer.summarize: {n_bad} row(s) have "
                    f"unparseable timestamps; fix the input DataFrame."
                )
            return df
        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        # ``errors='coerce'`` turns unparseable values into NaT;
        # those rows would silently produce wrong results. Drop
        # them explicitly so a bad input fails loudly downstream
        # rather than producing wrong summaries.
        if work["timestamp"].isna().any():
            n_bad = int(work["timestamp"].isna().sum())
            raise SensorSummarizerSchemaError(
                f"SensorSummarizer.summarize: {n_bad} row(s) have "
                f"unparseable timestamps; fix the input DataFrame."
            )
        return work

    def _summarize_numeric(self, group: pd.DataFrame, date_value: object) -> str:
        """Render a numeric (temp/humidity/energy) summary for one ``(date, sensor)`` group."""
        # Reset the group's index so ``idxmax()`` returns a
        # positional index into a 0..N-1 RangeIndex — safer than
        # ``iloc[values.idxmax()]`` which can IndexError when the
        # group inherited a non-contiguous index from the parent
        # DataFrame.
        group = group.reset_index(drop=True)
        values = group["value"].astype(float)
        unit = str(group["unit"].iloc[0])
        sensor_type = str(group["sensor_type"].iloc[0])
        room = self._humanize_room(str(group["sensor_id"].iloc[0]))
        date_str = pd.Timestamp(date_value).strftime(self.date_fmt)

        if values.empty:
            # Defensive — groupby already drops empty groups, but
            # an explicit guard makes the function safe to call
            # outside the groupby loop too (e.g. from tests).
            return (
                f"On {date_str}, no readings were recorded for the "
                f"{room} {sensor_type}."
            )

        mean_val = float(values.mean())
        max_val = float(values.max())
        min_val = float(values.min())

        # ``idxmax`` / ``idxmin`` are now safe positional indices
        # into the reset-index group.
        peak_idx = int(values.idxmax())
        trough_idx = int(values.idxmin())
        peak_time_str = group["timestamp"].iloc[peak_idx].strftime(self.time_fmt)
        trough_time_str = group["timestamp"].iloc[trough_idx].strftime(self.time_fmt)

        # Format values with one decimal (the synthetic data has
        # ~0.01 precision, but precision past the first decimal
        # is noise for a human reader). Units that conventionally
        # attach to the number without a space (``%`` and ``°``)
        # are written tight (``"57.7%"``, ``"22.5°C"``); other
        # units get a thin space (``"20.2 C"``, ``"0.3 kWh"``).
        # The choice is made per-unit, not per-sensor-type, so
        # adding a new unit is a one-line change in the table.
        val_str_mean = self._format_value_with_unit(mean_val, unit)
        val_str_max = self._format_value_with_unit(max_val, unit)
        val_str_min = self._format_value_with_unit(min_val, unit)

        return (
            f"On {date_str}, the {room} {sensor_type} averaged "
            f"{val_str_mean}, peaking at {val_str_max} "
            f"at {peak_time_str}, and reaching a minimum of "
            f"{val_str_min} at {trough_time_str}."
        )

    def _summarize_motion(self, group: pd.DataFrame, date_value: object) -> str:
        """Render a motion event-log summary for one ``(date, sensor)`` group.

        Three forms:

        - 0 active ticks → "no motion was detected in the kitchen".
        - 1..N (≤ :data:`_MOTION_VERBATIM_LIMIT`) active ticks →
          "motion was detected in the kitchen at 14:23, 15:47, and 22:03".
        - > :data:`_MOTION_VERBATIM_LIMIT` active ticks → "N events,
          the first at 06:12, the last at 22:45" (avoids token bloat
          on busy days).
        """
        group = group.reset_index(drop=True)
        room = self._humanize_room(str(group["sensor_id"].iloc[0]))
        date_str = pd.Timestamp(date_value).strftime(self.date_fmt)
        active = group.loc[group["value"].astype(float) > 0, "timestamp"].sort_values()
        n_active = int(len(active))

        if n_active == 0:
            return f"On {date_str}, no motion was detected in the {room}."

        if n_active <= _MOTION_VERBATIM_LIMIT:
            timestamps = [ts.strftime(self.time_fmt) for ts in active]
            # English list-joining: "a", "a and b", "a, b, and c"
            if n_active == 1:
                joined = timestamps[0]
            elif n_active == 2:
                joined = f"{timestamps[0]} and {timestamps[1]}"
            else:
                joined = ", ".join(timestamps[:-1]) + f", and {timestamps[-1]}"
            return (
                f"On {date_str}, motion was detected in the {room} "
                f"at {joined}."
            )

        # n_active > _MOTION_VERBATIM_LIMIT — collapse to a count.
        first = active.iloc[0].strftime(self.time_fmt)
        last = active.iloc[-1].strftime(self.time_fmt)
        return (
            f"On {date_str}, the {room} detected {n_active} motion "
            f"events, the first at {first}, the last at {last}."
        )

    @staticmethod
    def _humanize_room(sensor_id: str) -> str:
        """Convert ``"living_room_temp"`` → ``"living room"``.

        Strips a recognised ``_temp`` / ``_hum`` / ``_motion`` /
        ``_energy`` suffix and replaces underscores in the
        remainder with spaces. Whole-house sensors without a
        recognised suffix (e.g. ``"house_energy"``) fall back to
        the literal ``sensor_id`` so a reader sees the raw id
        rather than a misleading empty string.
        """
        for suffix in _ROOM_SUFFIXES:
            if sensor_id.endswith(suffix):
                room_part = sensor_id[: -len(suffix)]
                return room_part.replace("_", " ")
        return sensor_id

    @staticmethod
    def _format_value_with_unit(value: float, unit: str) -> str:
        """Render ``(value, unit)`` with the conventional spacing.

        Per-unit spacing rules (SI / typographic convention):

        - ``%`` (percent) and ``°C`` (degrees Celsius): **no space**
          — "57.7%", "22.5°C". The percent sign is a modifier of
          the number, not a separate word; the degree sign is
          part of the unit symbol.
        - everything else (``C``, ``kWh``, ``count``, …): **space**
          — "20.2 C", "0.3 kWh". These units are read as separate
          tokens in English prose.

        Adding a new unit? Update :data:`_TIGHT_UNITS` — the
        rest of the formatter needs no change.
        """
        # Units that conventionally attach to the number with no
        # space. ``%`` matches the canonical ``SUPPORTED_UNITS``
        # value exactly; ``°C`` isn't a supported unit (the
        # canonical unit for temperature is plain ``C``) but
        # ``°C`` is included as a courtesy in case a future
        # schema upgrade switches to it.
        tight_units = {"%", "°C"}
        separator = "" if unit in tight_units else " "
        return f"{value:.1f}{separator}{unit}"


__all__ = [
    "SensorSummarizer",
    "SensorSummarizerError",
    "SensorSummarizerSchemaError",
    "SensorSummarizerEmptyError",
]

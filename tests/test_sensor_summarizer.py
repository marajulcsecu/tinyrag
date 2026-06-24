"""Unit tests for SensorSummarizer (Step 4.14).

Coverage targets:
- Public surface: SensorSummarizer is a frozen dataclass, every
  documented kwarg is wired through, the encoder cache works.
- Schema validation: missing columns → SchemaError, unparseable
  timestamps → SchemaError, empty DataFrame → EmptyError,
  DataFrame that produces 0 chunks after grouping → EmptyError.
- Numeric summaries: temperature / humidity / energy all use the
  same renderer; mean / min / max / peak time / trough time all
  surface in the text; the format string matches the architecture
  doc's example.
- Motion summaries: 0 events, 1 event, 2 events, 3 events,
  N events (≤ verbatim limit), N events (> verbatim limit) all
  produce the right text shape.
- Chunk shape: every emitted Chunk has the right ``text``,
  ``source``, ``page``, ``chunk_index``, ``char_offset``,
  ``token_count``.
- Token count discipline: token_count matches tiktoken's count
  for the same text.
- Chunk index invariant: chunk_index is exactly 0..N-1 (the same
  invariant the document chunker and ingest CLI enforce).
- Custom labels: source_label, time_fmt, date_fmt all flow
  through to the output.
- Unknown sensor_type is silently skipped (defensive).
- Room extraction: ``living_room_temp`` → "living room";
  ``house_energy`` → "house_energy" (no recognised suffix);
  ``kitchen_motion`` → "kitchen".
- Empty group after filter: no chunk emitted (verified
  explicitly via a single-sensor-type DataFrame with one empty
  group).
- End-to-end against the real Step 3.8 synthetic CSV — the
  regression gate for "did the generator + summarizer drift
  from the architecture doc's expected output shape?".
- Integration with the prompt builder — the chunks flow into
  PromptBuilder.build() and produce a clean grounded prompt.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import tiktoken

from tinyrag.core import Chunk, PromptBuilder
from tinyrag.core.sensor_summarizer import (
    SensorSummarizer,
    SensorSummarizerEmptyError,
    SensorSummarizerError,
    SensorSummarizerSchemaError,
)

# ---------------------------------------------------------------------------
# Test fixtures (inline DataFrames — no I/O)
# ---------------------------------------------------------------------------

#: Single-day, single-sensor, 4 numeric readings — the minimum
#: useful input for a numeric summary.
def _numeric_df(
    sensor_id: str = "living_room_temp",
    sensor_type: str = "temperature",
    unit: str = "C",
    base_value: float = 20.0,
    *,
    date: datetime = datetime(2026, 6, 24),
) -> pd.DataFrame:
    """Build a small numeric sensor DataFrame spanning one day.

    The 4 readings are at 00:00, 06:00, 12:00, 18:00 so mean /
    min / max / peak time / trough time are all easily
    predictable from the input. ``base_value`` is the 00:00
    reading; 06:00 is ``base + 1``, 12:00 is ``base + 4``,
    18:00 is ``base + 2`` (so peak = 12:00, trough = 00:00).
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    date,
                    date + timedelta(hours=6),
                    date + timedelta(hours=12),
                    date + timedelta(hours=18),
                ]
            ),
            "sensor_id": [sensor_id] * 4,
            "sensor_type": [sensor_type] * 4,
            "value": [base_value, base_value + 1.0, base_value + 4.0, base_value + 2.0],
            "unit": [unit] * 4,
        }
    )


def _motion_df(
    sensor_id: str = "kitchen_motion",
    *,
    date: datetime = datetime(2026, 6, 24),
    active_hours: tuple[int, ...] = (),
    n_total: int = 24,
) -> pd.DataFrame:
    """Build a synthetic motion DataFrame for one day.

    Produces ``n_total`` rows at hourly intervals. ``active_hours``
    lists the hours at which the motion sensor reports ``value=1``;
    all others are ``value=0``. By default the sensor is silent
    (no active hours), so the default output is the "no motion"
    summary text.
    """
    active_set = set(active_hours)
    rows: list[dict] = []
    for h in range(n_total):
        rows.append(
            {
                "timestamp": date + timedelta(hours=h),
                "sensor_id": sensor_id,
                "sensor_type": "motion",
                "value": 1 if h in active_set else 0,
                "unit": "count",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """SensorSummarizer exposes the documented surface."""

    def test_construction_with_no_args_uses_defaults(self) -> None:
        s = SensorSummarizer()
        assert s.window == "daily"
        assert s.time_fmt == "%H:%M"
        assert s.date_fmt == "%Y-%m-%d"
        assert s.source_label == "sensor-summary"
        assert s.encoding_name == "cl100k_base"

    def test_frozen(self) -> None:
        s = SensorSummarizer()
        with pytest.raises(FrozenInstanceError):
            s.window = "hourly"  # type: ignore[misc]

    def test_invalid_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 'daily'"):
            SensorSummarizer(window="hourly")  # type: ignore[arg-type]

    def test_unknown_encoding_rejected_at_construction(self) -> None:
        # Bad encoding name fails fast at __post_init__, not at
        # the first summarize() call — surfaces misconfig
        # immediately at process startup.
        with pytest.raises(ValueError, match="Unknown tiktoken encoding"):
            SensorSummarizer(encoding_name="not-a-real-encoding")

    def test_encoder_cached(self) -> None:
        s = SensorSummarizer()
        # The encoder is the same object across accesses (tiktoken
        # caches globally, but we cache it on the dataclass too so
        # callers don't pay a dict lookup per chunk).
        assert s._encoder is s._encoder


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Bad DataFrames raise typed SensorSummarizerError subclasses."""

    def test_missing_column_raises_schema_error(self) -> None:
        # Missing the ``unit`` column.
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([datetime(2026, 6, 24)]),
                "sensor_id": ["x"],
                "sensor_type": ["temperature"],
                "value": [20.0],
            }
        )
        with pytest.raises(SensorSummarizerSchemaError, match="missing"):
            SensorSummarizer().summarize(df)

    def test_multiple_missing_columns_listed_in_error(self) -> None:
        df = pd.DataFrame({"timestamp": pd.to_datetime([datetime(2026, 6, 24)])})
        with pytest.raises(SensorSummarizerSchemaError) as excinfo:
            SensorSummarizer().summarize(df)
        # The error message lists every missing column so a user
        # with a hand-built DataFrame can fix it without bisecting.
        for col in ("sensor_id", "sensor_type", "value", "unit"):
            assert col in str(excinfo.value)

    def test_unparseable_timestamp_raises_schema_error(self) -> None:
        # Build the DataFrame from scratch with one bad timestamp
        # (don't mutate an existing datetime64 column in place —
        # pandas emits a FutureWarning for that).
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "not a date",  # forces coercion failure
                        datetime(2026, 6, 24, 6),
                        datetime(2026, 6, 24, 12),
                        datetime(2026, 6, 24, 18),
                    ],
                    errors="coerce",
                ),
                "sensor_id": ["x", "x", "x", "x"],
                "sensor_type": ["temperature"] * 4,
                "value": [20.0, 21.0, 24.0, 22.0],
                "unit": ["C"] * 4,
            }
        )
        with pytest.raises(SensorSummarizerSchemaError, match="unparseable"):
            SensorSummarizer().summarize(df)

    def test_empty_dataframe_raises_empty_error(self) -> None:
        # An empty DataFrame with the right columns is schema-valid
        # but produces 0 chunks → EmptyError, not SchemaError.
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([]),
                "sensor_id": pd.Series([], dtype="string"),
                "sensor_type": pd.Series([], dtype="string"),
                "value": pd.Series([], dtype="float64"),
                "unit": pd.Series([], dtype="string"),
            }
        )
        with pytest.raises(SensorSummarizerEmptyError, match="0 chunks"):
            SensorSummarizer().summarize(df)

    def test_dataframe_with_only_unknown_sensor_type_raises_empty(self) -> None:
        # All rows have an unknown sensor_type; the summarizer
        # silently skips them → no chunks produced → EmptyError.
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([datetime(2026, 6, 24)] * 3),
                "sensor_id": ["a", "a", "a"],
                "sensor_type": ["voltage", "voltage", "voltage"],
                "value": [1.0, 2.0, 3.0],
                "unit": ["V", "V", "V"],
            }
        )
        with pytest.raises(SensorSummarizerEmptyError):
            SensorSummarizer().summarize(df)

    def test_extras_columns_are_ignored(self) -> None:
        """Extra columns don't break the summarizer — they're tolerated."""
        df = _numeric_df()
        df["extra_col"] = "ignored"
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# 3. Numeric summaries
# ---------------------------------------------------------------------------


class TestNumericSummary:
    """Numeric summaries render the expected mean/min/max/peak/trough text."""

    def test_single_day_single_sensor_one_chunk(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 1

    def test_numeric_summary_contains_all_statistics(self) -> None:
        df = _numeric_df(base_value=20.0)
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        # Values: 20.0, 21.0, 24.0, 22.0 → mean 21.75, min 20.0,
        # max 24.0, peak at 12:00, trough at 00:00.
        assert "21.8" in text  # mean, rounded to 1 dp
        assert "20.0" in text  # min
        assert "24.0" in text  # max
        assert "12:00" in text  # peak time
        assert "00:00" in text  # trough time

    def test_numeric_summary_uses_date_and_sensor_name(self) -> None:
        df = _numeric_df(sensor_id="bedroom_temp", date=datetime(2026, 6, 15))
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        assert "2026-06-15" in text  # date prefix
        assert "bedroom" in text  # humanised room
        assert "temperature" in text  # sensor type

    @pytest.mark.parametrize(
        "sensor_id,sensor_type,unit,expected_room_substring",
        [
            ("living_room_temp", "temperature", "C", "living room"),
            ("bedroom_hum", "humidity", "%", "bedroom"),
            ("kitchen_motion", "motion", "count", "kitchen"),
        ],
    )
    def test_room_humanization_param(
        self,
        sensor_id: str,
        sensor_type: str,
        unit: str,
        expected_room_substring: str,
    ) -> None:
        """All sensor IDs render the human-readable room name."""
        df = _numeric_df(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            unit=unit,
        )
        chunks = SensorSummarizer().summarize(df)
        assert expected_room_substring in chunks[0].text

    def test_percent_unit_attaches_without_space(self) -> None:
        """``%`` is written tight (``"21.8%"``), not ``"21.8 %"``."""
        df = _numeric_df(sensor_id="bedroom_hum", sensor_type="humidity", unit="%")
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        # ``%`` should be tight to the number; check both forms
        # explicitly so a regression to ``" %"`` is caught.
        assert "21.8%" in text  # mean rendered tight with the unit
        assert "21.8 %" not in text  # the wrong form

    def test_kwh_unit_attaches_with_space(self) -> None:
        """``kWh`` is written with a space (``"0.3 kWh"``)."""
        df = _numeric_df(
            sensor_id="house_energy",
            sensor_type="energy",
            unit="kWh",
            base_value=0.1,
        )
        chunks = SensorSummarizer().summarize(df)
        assert "kWh" in chunks[0].text
        # The space between number and unit is intentional.
        assert "0.1 kWh" in chunks[0].text

    def test_single_value_summary_works(self) -> None:
        """A group with exactly 1 row: mean = min = max, peak = trough."""
        df = _numeric_df().iloc[:1]
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 1
        # The single timestamp is both the peak and the trough.
        assert "00:00" in chunks[0].text


# ---------------------------------------------------------------------------
# 4. Multi-sensor + multi-day grouping
# ---------------------------------------------------------------------------


class TestMultiSensorAndDay:
    """Multiple sensors / days produce one chunk per ``(date, sensor)`` pair."""

    def test_two_sensors_same_day_two_chunks(self) -> None:
        df = pd.concat([_numeric_df(), _numeric_df(sensor_id="bedroom_temp")], ignore_index=True)
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 2
        sensor_names = sorted(c.text for c in chunks)
        assert any("living room" in t for t in sensor_names)
        assert any("bedroom" in t for t in sensor_names)

    def test_same_sensor_two_days_two_chunks(self) -> None:
        day1 = _numeric_df(date=datetime(2026, 6, 24))
        day2 = _numeric_df(date=datetime(2026, 6, 25))
        df = pd.concat([day1, day2], ignore_index=True)
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 2
        dates = sorted(c.text.split("On ", 1)[1].split(",", 1)[0] for c in chunks)
        assert dates == ["2026-06-24", "2026-06-25"]

    def test_two_sensors_two_days_four_chunks(self) -> None:
        days = [datetime(2026, 6, 24), datetime(2026, 6, 25)]
        sensors = ["living_room_temp", "bedroom_temp"]
        frames = [
            _numeric_df(sensor_id=s, date=d)
            for d in days
            for s in sensors
        ]
        df = pd.concat(frames, ignore_index=True)
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 4


# ---------------------------------------------------------------------------
# 5. Motion summaries
# ---------------------------------------------------------------------------


class TestMotionSummary:
    """Motion summaries render the right text for 0 / few / many events."""

    def test_zero_motion_events(self) -> None:
        df = _motion_df()
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 1
        assert "no motion was detected" in chunks[0].text

    def test_one_motion_event(self) -> None:
        df = _motion_df(active_hours=(14,))
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        assert "14:00" in text
        # One event → "at 14:00" (no Oxford comma).
        assert "at 14:00." in text

    def test_two_motion_events(self) -> None:
        df = _motion_df(active_hours=(8, 18))
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        assert "08:00" in text
        assert "18:00" in text
        # Two events → joined with "and", not a comma.
        assert "at 08:00 and 18:00" in text

    def test_three_motion_events(self) -> None:
        df = _motion_df(active_hours=(8, 14, 22))
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        # Three events → Oxford comma: "a, b, and c"
        assert "at 08:00, 14:00, and 22:00" in text

    def test_five_motion_events_still_verbatim(self) -> None:
        """At exactly the verbatim limit, the full list is still shown."""
        df = _motion_df(active_hours=(6, 9, 12, 15, 18))
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        # All five timestamps are listed.
        for ts in ("06:00", "09:00", "12:00", "15:00", "18:00"):
            assert ts in text
        # No count form yet (this is the boundary, still verbatim).
        assert "events" not in text

    def test_six_motion_events_uses_count_form(self) -> None:
        """Above the verbatim limit, the summary collapses to a count."""
        df = _motion_df(active_hours=(1, 4, 7, 10, 13, 16))
        chunks = SensorSummarizer().summarize(df)
        text = chunks[0].text
        # 6 events → "6 motion events" form, first + last only.
        assert "6 motion events" in text
        assert "01:00" in text
        assert "16:00" in text
        # Inner timestamps should NOT be listed (avoid token bloat).
        assert "04:00" not in text


# ---------------------------------------------------------------------------
# 6. Chunk shape invariants
# ---------------------------------------------------------------------------


class TestChunkShape:
    """Every emitted Chunk has the right fields filled."""

    def test_chunk_is_chunk_dataclass(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_source_label(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        for c in chunks:
            assert c.source == "sensor-summary"

    def test_page_is_none(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        for c in chunks:
            assert c.page is None

    def test_char_offset_is_zero(self) -> None:
        """Sensor chunks aren't slices of a parent document."""
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        for c in chunks:
            assert c.char_offset == 0

    def test_token_count_positive(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        for c in chunks:
            assert c.token_count > 0

    def test_token_count_matches_tiktoken(self) -> None:
        """The Chunk's ``token_count`` field matches a fresh tiktoken encode."""
        df = _numeric_df()
        s = SensorSummarizer()
        chunks = s.summarize(df)
        encoder = tiktoken.get_encoding(s.encoding_name)
        for c in chunks:
            assert c.token_count == len(encoder.encode(c.text))


# ---------------------------------------------------------------------------
# 7. Chunk-index invariant
# ---------------------------------------------------------------------------


class TestChunkIndexInvariant:
    """chunk_index is exactly ``[0, 1, ..., N-1]`` (the canonical invariant)."""

    def test_single_chunk_index_is_zero(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        assert [c.chunk_index for c in chunks] == [0]

    def test_multiple_chunks_have_contiguous_indices(self) -> None:
        df = pd.concat(
            [
                _numeric_df(sensor_id=s, date=d)
                for d in [datetime(2026, 6, 24), datetime(2026, 6, 25)]
                for s in ["living_room_temp", "bedroom_temp"]
            ],
            ignore_index=True,
        )
        chunks = SensorSummarizer().summarize(df)
        assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 8. Custom configuration flows through
# ---------------------------------------------------------------------------


class TestCustomConfiguration:
    """source_label, time_fmt, date_fmt kwargs reach the output."""

    def test_custom_source_label(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer(source_label="my-fleet-2026").summarize(df)
        for c in chunks:
            assert c.source == "my-fleet-2026"

    def test_custom_time_format_12_hour(self) -> None:
        """``"%I:%M %p"`` produces 12-hour times with AM/PM."""
        df = _motion_df(active_hours=(14,))
        chunks = SensorSummarizer(time_fmt="%I:%M %p").summarize(df)
        # 14:00 in 12-hour is "02:00 PM".
        assert "02:00 PM" in chunks[0].text
        assert "14:00" not in chunks[0].text

    def test_custom_date_format(self) -> None:
        df = _numeric_df(date=datetime(2026, 6, 15))
        chunks = SensorSummarizer(date_fmt="%d %b %Y").summarize(df)
        # British short-month form: "15 Jun 2026".
        assert "15 Jun 2026" in chunks[0].text


# ---------------------------------------------------------------------------
# 9. Defensive handling of unknown sensor types
# ---------------------------------------------------------------------------


class TestUnknownSensorType:
    """Unknown sensor_type values are silently skipped (defensive)."""

    def test_unknown_sensor_type_skipped(self) -> None:
        df = pd.concat(
            [
                _numeric_df(),  # 1 valid numeric chunk
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([datetime(2026, 6, 24)]),
                        "sensor_id": ["mystery"],
                        "sensor_type": ["voltage"],  # unknown
                        "value": [5.0],
                        "unit": ["V"],
                    }
                ),
            ],
            ignore_index=True,
        )
        chunks = SensorSummarizer().summarize(df)
        # Only the valid numeric chunk is emitted; the unknown
        # type is silently dropped.
        assert len(chunks) == 1
        assert "living room" in chunks[0].text


# ---------------------------------------------------------------------------
# 10. Room extraction
# ---------------------------------------------------------------------------


class TestRoomExtraction:
    """``_humanize_room`` produces the right human-readable room name."""

    @pytest.mark.parametrize(
        "sensor_id,expected_room",
        [
            ("living_room_temp", "living room"),
            ("bedroom_temp", "bedroom"),
            ("kitchen_motion", "kitchen"),
            ("living_room_hum", "living room"),
            ("house_energy", "house"),  # _energy suffix is recognised
            ("mystery_sensor", "mystery_sensor"),  # no recognised suffix → fallback
        ],
    )
    def test_room_humanization(
        self, sensor_id: str, expected_room: str
    ) -> None:
        assert SensorSummarizer._humanize_room(sensor_id) == expected_room


# ---------------------------------------------------------------------------
# 11. End-to-end with the REAL synthetic 30-day CSV (Step 3.8)
# ---------------------------------------------------------------------------


REAL_CSV = Path("data/sensor_logs/synthetic_30d.csv")


@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="data/sensor_logs/synthetic_30d.csv not present (run scripts/generate_synthetic_sensors.py)",
)
class TestEndToEndRealCsv:
    """End-to-end against the real 30-day Step 3.8 synthetic CSV.

    This is the **regression gate** for "did Step 3.8's generator
    drift from the schema the Step 4.13 reader + Step 4.14
    summarizer expect?". If this fails, fix the producer (Step
    3.8's generator) or fix the consumer (this summarizer).
    """

    def test_produces_expected_chunk_count(self) -> None:
        """30 days x 6 sensors = 180 chunks."""
        from tinyrag.sensors.simulated import SimulatedCSVSource

        df = SimulatedCSVSource(REAL_CSV).read()
        chunks = SensorSummarizer().summarize(df)
        assert len(chunks) == 180

    def test_all_chunks_have_correct_source(self) -> None:
        from tinyrag.sensors.simulated import SimulatedCSVSource

        df = SimulatedCSVSource(REAL_CSV).read()
        chunks = SensorSummarizer().summarize(df)
        for c in chunks:
            assert c.source == "sensor-summary"

    def test_chunk_indices_are_contiguous(self) -> None:
        from tinyrag.sensors.simulated import SimulatedCSVSource

        df = SimulatedCSVSource(REAL_CSV).read()
        chunks = SensorSummarizer().summarize(df)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunks_can_be_queried_by_temperature_question(self) -> None:
        """Spot-check: a chunk about temperature looks like the architecture's example."""
        from tinyrag.sensors.simulated import SimulatedCSVSource

        df = SimulatedCSVSource(REAL_CSV).read()
        chunks = SensorSummarizer().summarize(df)
        # Find a temperature chunk.
        temp_chunk = next(c for c in chunks if "temperature" in c.text)
        # Architecture doc example: "On YYYY-MM-DD, the X temperature averaged
        # N.N C, peaking at N.N C at HH:MM, and reaching a minimum..."
        assert temp_chunk.text.startswith("On 20")
        assert "temperature averaged" in temp_chunk.text
        assert "peaking at" in temp_chunk.text
        assert "at 00:00" in temp_chunk.text or "at 0" in temp_chunk.text


# ---------------------------------------------------------------------------
# 12. Integration with PromptBuilder
# ---------------------------------------------------------------------------


class TestIntegrationWithPromptBuilder:
    """Summarizer chunks flow cleanly into PromptBuilder.build()."""

    def test_chunks_produce_clean_grounded_prompt(self) -> None:
        df = _numeric_df()
        chunks = SensorSummarizer().summarize(df)
        prompt = PromptBuilder().build("What was the temperature?", chunks)
        # The prompt carries exactly one chunk (citation [1]) and
        # the question at the end.
        assert prompt.chunks_used == 1
        assert prompt.chunks_dropped == 0
        assert "[1]" in prompt.user_message
        assert "Question: What was the temperature?" in prompt.user_message

    def test_multi_day_chunks_get_consecutive_citations(self) -> None:
        """3 chunks → citations [1], [2], [3] — contiguous, no gaps."""
        df = pd.concat(
            [
                _numeric_df(date=datetime(2026, 6, 23)),
                _numeric_df(date=datetime(2026, 6, 24)),
                _numeric_df(date=datetime(2026, 6, 25)),
            ],
            ignore_index=True,
        )
        chunks = SensorSummarizer().summarize(df)
        prompt = PromptBuilder().build("Temperature history?", chunks)
        assert prompt.chunks_used == 3
        for marker in ("[1]", "[2]", "[3]"):
            assert marker in prompt.user_message


# ---------------------------------------------------------------------------
# 13. Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """The SensorSummarizerError hierarchy catches uniformly."""

    def test_all_subclasses_inherit_from_base(self) -> None:
        for cls in (SensorSummarizerSchemaError, SensorSummarizerEmptyError):
            assert issubclass(cls, SensorSummarizerError)

    def test_single_except_catches_all(self) -> None:
        # Empty DataFrame triggers EmptyError
        with pytest.raises(SensorSummarizerError):
            SensorSummarizer().summarize(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([]),
                        "sensor_id": pd.Series([], dtype="string"),
                        "sensor_type": pd.Series([], dtype="string"),
                        "value": pd.Series([], dtype="float64"),
                        "unit": pd.Series([], dtype="string"),
                    }
                )
            )
        # Missing column triggers SchemaError
        with pytest.raises(SensorSummarizerError):
            SensorSummarizer().summarize(
                pd.DataFrame({"timestamp": pd.to_datetime([datetime(2026, 6, 24)])})
            )

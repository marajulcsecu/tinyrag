"""Tests for src/tinyrag/config.py — the typed configuration loader.

These tests cover every requirement the config layer must satisfy:
- FR-49: single config.yaml at startup.
- FR-50: required fields are all represented on Settings.
- FR-51: invalid config fails loudly with a clear error.
- FR-52: deployment.target is constrained to {laptop, raspberry_pi}.

Test layout
-----------
- TestPublicSurface     — every Pydantic section class instantiates
  with sane defaults (catches missing-field drift).
- TestEnumCoercion      — string-to-enum coercion works for every
  enum field.
- TestLoadSettings      — happy path: load the real config.yaml
  and assert all FR-50 fields are populated.
- TestLoadSettingsErrors — every error path produces a typed
  ConfigError subclass with an actionable message.
- TestCrossFieldValidation — FR-18 [L] + the chunking overlap rule.
- TestResolve           — Settings.resolve() anchors relative paths
  to the project root; absolute paths pass through.
- TestConfigYamlMatchesSpec — the real config.yaml matches the
  schema declared in docs/02_srs_v1.md Appendix B and
  docs/04_database_design_v1.md §config.
- TestFROrNumbers       — explicit FR-49..FR-52 traceability.

Location: ``tests/test_config.py``
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from tinyrag.config import (
    ChunkingSettings,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    DeploymentSettings,
    DeploymentTarget,
    EmbeddingDevice,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    LogLevel,
    PathsSettings,
    RetrievalSettings,
    SensorSettings,
    SensorSource,
    ServerSettings,
    Settings,
    load_settings,
)

# ----------------------------------------------------------------------------
# Path to the real config.yaml (used by TestLoadSettings +
# TestConfigYamlMatchesSpec). Every test that needs it should use
# this fixture rather than hard-coding the path.
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ----------------------------------------------------------------------------
# Sample-config fixture. Returns a fresh dict each call so tests
# can mutate freely without leaking state.
# ----------------------------------------------------------------------------


def _sample_config_dict() -> dict:
    """A complete, valid config.yaml-shaped dict for tests.

    Used by tests that need to load a Settings without hitting the
    real file (so they can pass a tmp_path-style override). The
    schema mirrors docs/04_database_design_v1.md §config.
    """
    return {
        "deployment": {"target": "laptop"},
        "server": {"host": "127.0.0.1", "port": 8000},
        "llm": {
            "model_path": "models/phi-3-mini.gguf",
            "server_url": "http://127.0.0.1:8080",
            "context_size": 4096,
            "temperature": 0.0,
            "max_tokens": 512,
            "gpu_layers": 0,
        },
        "embedding": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "cache_dir": "models/_hf_cache",
        },
        "chunking": {
            "chunk_size": 400,
            "chunk_overlap": 50,
            "encoding": "cl100k_base",
        },
        "retrieval": {
            "doc_index_path": "data/vector_store/doc.faiss",
            "sensor_index_path": "data/vector_store/sensor.faiss",
            "doc_top_k": 3,
            "sensor_top_k": 2,
            "similarity_threshold": 0.3,
            "index_type": "faiss",
        },
        "sensors": {
            "source": "simulated",
            "csv_path": "data/sensor_logs/synthetic_30d.csv",
            "dht_pin": 4,
            "pir_pin": 17,
            "mqtt_broker": "localhost",
            "mqtt_port": 1883,
            "mqtt_topic_prefix": "tinyrag/sensors/",
        },
        "logging": {
            "level": "INFO",
            "json_format": True,
            "path": "logs/tinyrag.log",
        },
        "paths": {
            "documents_dir": "data/documents/",
            "metadata_db": "data/metadata.db",
            "sensor_logs_dir": "data/sensor_logs/",
            "logs_dir": "logs/",
        },
    }


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Write the sample config to a tmp file and return its path."""
    p = tmp_path / "config.yaml"
    import yaml  # local import to keep top-level import list small

    p.write_text(yaml.safe_dump(_sample_config_dict()), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestPublicSurface:
    """Every Pydantic sub-model can be instantiated with defaults."""

    @pytest.mark.parametrize(
        "cls",
        [
            DeploymentSettings,
            ServerSettings,
            LLMSettings,
            EmbeddingSettings,
            ChunkingSettings,
            RetrievalSettings,
            SensorSettings,
            LoggingSettings,
            PathsSettings,
        ],
    )
    def test_default_construction(self, cls) -> None:
        """Defaults exist for every sub-model — no required-arg drift."""
        instance = cls()
        # Every sub-model should have at least one populated attribute.
        assert len(instance.model_dump()) > 0


class TestEnumCoercion:
    """String-to-enum coercion works for every enum-typed field."""

    def test_deployment_target_string(self) -> None:
        assert DeploymentSettings(target="laptop").target is DeploymentTarget.LAPTOP
        assert (
            DeploymentSettings(target="raspberry_pi").target
            is DeploymentTarget.RASPBERRY_PI
        )

    def test_deployment_target_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            DeploymentSettings(target="workstation")

    def test_sensor_source_string(self) -> None:
        assert SensorSettings(source="simulated").source is SensorSource.SIMULATED
        assert SensorSettings(source="real_serial").source is SensorSource.REAL_SERIAL
        assert SensorSettings(source="mqtt").source is SensorSource.MQTT

    def test_log_level_string(self) -> None:
        assert LoggingSettings(level="DEBUG").level is LogLevel.DEBUG
        assert LoggingSettings(level="INFO").level is LogLevel.INFO

    def test_embedding_device_string(self) -> None:
        assert EmbeddingSettings(device="cpu").device is EmbeddingDevice.CPU


class TestLoadSettings:
    """The happy path: load a config and assert every field is populated."""

    def test_load_real_config_yaml(self) -> None:
        """The real config.yaml in the repo root validates + loads."""
        s = load_settings(REAL_CONFIG_PATH)
        assert isinstance(s, Settings)
        # Spot-check a field from each of the 9 sections.
        assert s.deployment.target is DeploymentTarget.LAPTOP
        assert s.server.port == 8000
        assert s.llm.model_path == "models/phi-3-mini.gguf"
        assert s.llm.temperature == 0.0
        assert s.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.chunking.chunk_size == 400
        assert s.chunking.chunk_overlap == 50
        assert s.retrieval.doc_top_k == 3
        assert s.retrieval.similarity_threshold == 0.3
        assert s.sensors.source is SensorSource.SIMULATED
        assert s.logging.level is LogLevel.INFO
        assert s.paths.documents_dir == "data/documents/"

    def test_load_sample_config(self, sample_config_path: Path) -> None:
        """A separately-written config also loads cleanly."""
        s = load_settings(sample_config_path)
        assert s.llm.model_path == "models/phi-3-mini.gguf"
        assert s.deployment.target is DeploymentTarget.LAPTOP

    def test_load_is_idempotent(self) -> None:
        """Loading the same config twice gives equal Settings."""
        a = load_settings(REAL_CONFIG_PATH)
        b = load_settings(REAL_CONFIG_PATH)
        assert a == b

    def test_settings_is_frozen(self) -> None:
        """Mutating a loaded Settings raises (FR-49: config is read-only)."""
        s = load_settings(REAL_CONFIG_PATH)
        with pytest.raises(ValidationError):
            s.llm.temperature = 0.7  # type: ignore[misc]

    def test_settings_resolve_uses_project_root(self) -> None:
        """Settings.resolve() anchors relative paths to config dir."""
        s = load_settings(REAL_CONFIG_PATH)
        resolved = s.resolve(s.llm.model_path)
        assert resolved.is_absolute()
        assert resolved == s.project_root() / "models" / "phi-3-mini.gguf"

    def test_settings_resolve_absolute_passthrough(self) -> None:
        """Settings.resolve() leaves absolute paths alone."""
        s = load_settings(REAL_CONFIG_PATH)
        abs_path = "/tmp/some-absolute-path.bin"
        assert s.resolve(abs_path) == Path(abs_path)


class TestLoadSettingsErrors:
    """Every error path produces a typed exception with a useful message."""

    def test_missing_file_raises_typed_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigNotFoundError) as ei:
            load_settings(tmp_path / "nope.yaml")
        assert "config file not found" in str(ei.value)

    def test_missing_file_is_a_config_error(self, tmp_path: Path) -> None:
        """ConfigNotFoundError is a ConfigError subclass (FR-51 catchable)."""
        with pytest.raises(ConfigError):
            load_settings(tmp_path / "nope.yaml")

    def test_malformed_yaml_raises_validation_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("not: valid: yaml: [", encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "is not valid YAML" in str(ei.value)

    def test_missing_section_raises_validation_error(self, tmp_path: Path) -> None:
        """A config that omits a required section fails at load time."""
        p = tmp_path / "partial.yaml"
        p.write_text("deployment:\n  target: laptop\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "missing required section" in str(ei.value).lower()
        # Multiple sections are missing — confirm the message lists them.
        assert "sensors" in str(ei.value)

    def test_wrong_type_raises_validation_error(self, tmp_path: Path) -> None:
        """server.port: 'not-a-number' is a type error."""
        import yaml

        cfg = _sample_config_dict()
        cfg["server"]["port"] = "not-a-number"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "server.port" in str(ei.value)
        assert "integer" in str(ei.value).lower()

    def test_out_of_range_raises_validation_error(self, tmp_path: Path) -> None:
        """llm.temperature: 5.0 violates the le=2.0 constraint."""
        import yaml

        cfg = _sample_config_dict()
        cfg["llm"]["temperature"] = 5.0
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "llm.temperature" in str(ei.value)

    def test_unknown_deployment_target_rejected(self, tmp_path: Path) -> None:
        """FR-52: deployment.target is constrained to {laptop, raspberry_pi}."""
        import yaml

        cfg = _sample_config_dict()
        cfg["deployment"]["target"] = "workstation"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "deployment.target" in str(ei.value)

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        """``extra=forbid`` means a typo'd key is caught."""
        import yaml

        cfg = _sample_config_dict()
        cfg["llm"]["model_paht"] = "typo.gguf"  # missing 't' in 'path'
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "llm.model_paht" in str(ei.value)

    def test_invalid_yaml_top_level_type(self, tmp_path: Path) -> None:
        """A YAML list at the top level is rejected."""
        p = tmp_path / "bad.yaml"
        p.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        assert "must be a YAML mapping" in str(ei.value)

    def test_empty_yaml_rejected_as_missing_sections(self, tmp_path: Path) -> None:
        """An empty YAML file is rejected as missing all sections.

        Rationale: an empty file is almost always a mistake (a copy
        that didn't complete, a wrong path, etc.) — silently
        defaulting everything is more dangerous than failing loud.
        The user is forced to explicitly write `{}` for each
        section if they really want all defaults.
        """
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        msg = str(ei.value).lower()
        assert "missing required section" in msg

    def test_yaml_with_all_empty_sections_loads(self, tmp_path: Path) -> None:
        """A YAML with `{}` for every section is allowed and uses defaults."""
        p = tmp_path / "empty_sections.yaml"
        p.write_text(
            dedent(
                """\
                deployment: {}
                server: {}
                llm: {}
                embedding: {}
                chunking: {}
                retrieval: {}
                sensors: {}
                logging: {}
                paths: {}
                """
            ),
            encoding="utf-8",
        )
        s = load_settings(p)
        assert s.llm.temperature == 0.0
        assert s.chunking.chunk_size == 400


class TestCrossFieldValidation:
    """FR-18 [L] + the chunking-overlap invariant."""

    def test_laptop_rejects_real_serial_sensor(self, tmp_path: Path) -> None:
        """The headline cross-field rule from FR-18 [L]."""
        import yaml

        cfg = _sample_config_dict()
        cfg["deployment"]["target"] = "laptop"
        cfg["sensors"]["source"] = "real_serial"
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        msg = str(ei.value).lower()
        assert "real_serial" in msg
        assert "laptop" in msg
        assert "gpio" in msg

    def test_raspberry_pi_allows_real_serial_sensor(self) -> None:
        """The same combo is valid on the Pi target."""
        s = Settings(
            deployment=DeploymentSettings(target=DeploymentTarget.RASPBERRY_PI),
            sensors=SensorSettings(source=SensorSource.REAL_SERIAL),
        )
        assert s.sensors.source is SensorSource.REAL_SERIAL

    def test_laptop_allows_simulated_sensor(self) -> None:
        """The laptop default is the simulated source — no error."""
        s = Settings(
            deployment=DeploymentSettings(target=DeploymentTarget.LAPTOP),
            sensors=SensorSettings(source=SensorSource.SIMULATED),
        )
        assert s.sensors.source is SensorSource.SIMULATED

    def test_laptop_allows_mqtt_sensor(self) -> None:
        """MQTT is also valid on the laptop (uses the network, not GPIO)."""
        s = Settings(
            deployment=DeploymentSettings(target=DeploymentTarget.LAPTOP),
            sensors=SensorSettings(source=SensorSource.MQTT),
        )
        assert s.sensors.source is SensorSource.MQTT

    def test_chunk_overlap_must_be_strictly_less_than_size(self) -> None:
        """chunk_overlap >= chunk_size is rejected."""
        with pytest.raises(ValidationError):
            ChunkingSettings(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValidationError):
            ChunkingSettings(chunk_size=100, chunk_overlap=200)

    def test_chunk_overlap_just_under_size_ok(self) -> None:
        """chunk_overlap == chunk_size - 1 is the legal boundary."""
        c = ChunkingSettings(chunk_size=100, chunk_overlap=99)
        assert c.chunk_size == 100
        assert c.chunk_overlap == 99


class TestConfigYamlMatchesSpec:
    """The real config.yaml matches the spec docs (SRS Appendix B +
    database design §config).

    This is the traceability test: if a future contributor changes
    the schema in either direction, this test breaks until both
    sides are updated together.
    """

    def test_real_config_has_every_required_section(self) -> None:
        import yaml

        data = yaml.safe_load(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        for section in (
            "deployment",
            "server",
            "llm",
            "embedding",
            "chunking",
            "retrieval",
            "sensors",
            "logging",
            "paths",
        ):
            assert section in data, f"config.yaml missing section: {section}"

    def test_real_config_defaults_match_srs_appendix_b(self) -> None:
        """Spot-check the SRS Appendix B defaults (FR-50)."""
        s = load_settings(REAL_CONFIG_PATH)
        # From SRS Appendix B:
        assert s.deployment.target is DeploymentTarget.LAPTOP
        assert s.server.host == "127.0.0.1"
        assert s.server.port == 8000
        assert s.llm.temperature == 0.0
        assert s.llm.max_tokens == 512
        assert s.llm.gpu_layers == 0
        assert s.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.embedding.batch_size == 32
        assert s.chunking.chunk_size == 400
        assert s.chunking.chunk_overlap == 50
        assert s.retrieval.doc_top_k == 3
        assert s.retrieval.sensor_top_k == 2
        assert s.retrieval.similarity_threshold == 0.3
        assert s.sensors.source is SensorSource.SIMULATED
        assert s.logging.level is LogLevel.INFO

    def test_real_config_has_paths_from_database_design(self) -> None:
        """Spot-check the paths from docs/04_database_design_v1.md §config."""
        s = load_settings(REAL_CONFIG_PATH)
        assert s.retrieval.doc_index_path == "data/vector_store/doc.faiss"
        assert s.retrieval.sensor_index_path == "data/vector_store/sensor.faiss"
        assert s.paths.documents_dir == "data/documents/"
        assert s.paths.metadata_db == "data/metadata.db"
        assert s.paths.sensor_logs_dir == "data/sensor_logs/"
        assert s.paths.logs_dir == "logs/"


class TestFROrNumbers:
    """Explicit traceability for FR-49..FR-52 from docs/02_srs_v1.md."""

    def test_fr49_single_config_yaml(self) -> None:
        """FR-49: runtime config comes from a single config.yaml."""
        s = load_settings(REAL_CONFIG_PATH)
        # If the file is the only source, the Settings it produces
        # must be sufficient to drive every downstream module.
        # Spot-check that no field is None / empty that would force
        # a downstream module to fall back to a hard-coded value.
        assert s.llm.model_path
        assert s.embedding.model_name
        assert s.chunking.chunk_size > 0
        assert s.chunking.chunk_overlap >= 0
        assert s.retrieval.doc_top_k > 0
        assert s.retrieval.similarity_threshold >= 0.0
        assert s.llm.temperature >= 0.0
        assert s.server.port > 0
        assert s.sensors.source is not None
        assert s.deployment.target is not None

    def test_fr50_required_fields_all_present(self) -> None:
        """FR-50: every field listed in the SRS is on Settings."""
        s = load_settings(REAL_CONFIG_PATH)
        required = {
            "llm.model_path",
            "embedding.model_name",
            "chunking.chunk_size",
            "chunking.chunk_overlap",
            "retrieval.doc_top_k",
            "retrieval.similarity_threshold",
            "llm.temperature",
            "server.port",
            "sensors.source",
            "deployment.target",
        }
        dump = s.model_dump()
        for dotted in required:
            node = dump
            for part in dotted.split("."):
                node = node[part]
            assert node is not None, f"FR-50: {dotted} is None"

    def test_fr51_invalid_config_fails_loudly(self, tmp_path: Path) -> None:
        """FR-51: missing required key exits with a clear error."""
        p = tmp_path / "missing_section.yaml"
        # Has only deployment + server — sensors, llm, etc. all missing.
        p.write_text(
            dedent(
                """\
                deployment:
                  target: laptop
                server:
                  host: 127.0.0.1
                  port: 8000
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigValidationError) as ei:
            load_settings(p)
        # The error must name the missing sections by name (not just
        # raise a generic exception). This is what makes the error
        # "clear" per FR-51.
        msg = str(ei.value).lower()
        assert "missing required section" in msg
        assert "sensors" in msg

    def test_fr52_deployment_target_values(self) -> None:
        """FR-52: deployment.target accepts {laptop, raspberry_pi} only."""
        # Both legal values parse to the right enum.
        assert (
            DeploymentSettings(target="laptop").target is DeploymentTarget.LAPTOP
        )
        assert (
            DeploymentSettings(target="raspberry_pi").target
            is DeploymentTarget.RASPBERRY_PI
        )
        # Anything else is rejected.
        for bogus in ("workstation", "LAPTOP", "raspberry-pi", ""):
            with pytest.raises(ValidationError):
                DeploymentSettings(target=bogus)

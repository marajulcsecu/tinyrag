"""Typed configuration loader — the single source of runtime truth.

This module is the ONLY place in TinyRAG that reads ``config.yaml``.
Every other module receives a :class:`Settings` instance (typically
constructed once in :mod:`tinyrag.main` and passed via FastAPI's
dependency injection or a simple module-level singleton for scripts).
This pattern satisfies three requirements from the SRS:

- **FR-49** — single ``config.yaml`` read at startup.
- **FR-50** — required fields (model path, embedding name, chunk size,
  chunk overlap, top-k, similarity threshold, LLM temperature, server
  ports, sensor source, deployment target) are all represented as
  typed, required fields on the Settings object.
- **FR-51** — invalid config fails loudly at startup, never silently
  in production. See :class:`ConfigValidationError`.
- **FR-52** — ``deployment.target`` controls runtime defaults
  (``gpu_layers``, ``sensors.source`` defaults) and is validated
  against the allowed set ``{"laptop", "raspberry_pi"}``.

Why Pydantic v2 (not ``dataclasses`` or ``pydantic-settings.BaseSettings``)?
----------------------------------------------------------------------------
- Pydantic v2 gives us free type coercion + structural validation +
  immutable-by-default (``frozen=True``). A misconfigured field is a
  startup error, never a runtime ``AttributeError`` deep inside the
  retrieval pipeline.
- We deliberately do NOT use ``pydantic_settings.BaseSettings``'s
  env-var loading. TinyRAG is single-process and single-config;
  mixing env vars and YAML is a recipe for "which one wins?"
  confusion. YAML is the single source of truth.

Why a custom loader instead of ``BaseSettings._build_config_values()``?
----------------------------------------------------------------------
- We want YAML, not env vars. ``BaseSettings`` is designed for
  env-first loading with YAML as an optional override — the wrong
  default for this project.
- A custom loader is ~30 lines, fully testable, and easy to swap if
  the team later decides to add TOML or JSON support.

Why are all fields required?
----------------------------
- A missing field is almost certainly a typo or a forgotten config
  upgrade. Hiding the bug behind a "use default" silent fallback
  makes debugging config drift painful.
- The defaults that DO exist (e.g. ``gpu_layers: 0``,
  ``chunk_overlap: 50``) are baked into the Pydantic schema, so a
  config file that omits them gets the schema default with zero
  surprises. The schema IS the documentation of "what's the
  expected value if the user doesn't override".

Public surface
--------------
- :class:`Settings` — the root config object.
- :class:`DeploymentSettings`, :class:`ServerSettings`,
  :class:`LLMSettings`, :class:`EmbeddingSettings`,
  :class:`ChunkingSettings`, :class:`RetrievalSettings`,
  :class:`SensorSettings`, :class:`LoggingSettings`,
  :class:`PathsSettings` — the per-section models.
- :func:`load_settings` — load ``config.yaml`` from a path (or the
  default project-root location) and return a validated
  :class:`Settings`.
- :class:`ConfigError`, :class:`ConfigValidationError` — the typed
  exceptions callers may catch.

Location: ``src/tinyrag/config.py``
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class ConfigError(Exception):
    """Base class for any configuration-loading failure.

    Subclasses:
    - :class:`ConfigNotFoundError` — ``config.yaml`` doesn't exist.
    - :class:`ConfigValidationError` — file exists but a field has
      the wrong type or is missing.

    Catching ``ConfigError`` catches both, which is what
    :mod:`tinyrag.main` does to translate config errors into a
    friendly startup error message.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class ConfigNotFoundError(ConfigError):
    """The ``config.yaml`` file does not exist at the requested path."""


class ConfigValidationError(ConfigError):
    """The config file has a structural problem (wrong type, missing key).

    The original :class:`pydantic.ValidationError` is preserved on
    ``self.original`` for programmatic inspection (the test suite
    uses it to assert on the exact failing field).
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        original: ValidationError | None = None,
    ) -> None:
        self.original = original
        super().__init__(message, path=path)


# ----------------------------------------------------------------------------
# Enums (for type-safe literal fields)
# ----------------------------------------------------------------------------


class DeploymentTarget(str, Enum):
    """Allowed values for ``deployment.target``.

    - ``laptop`` — Dell Inspiron 15 3520 (Ubuntu 24.04). No GPIO,
      simulated sensor default, CPU-only LLM.
    - ``raspberry_pi`` — Raspberry Pi 5 / 8 GB. GPIO available,
      real_serial sensor default, CPU-only LLM (no usable GPU on Pi
      for LLMs).
    """

    LAPTOP = "laptop"
    RASPBERRY_PI = "raspberry_pi"


class SensorSource(str, Enum):
    """Allowed values for ``sensors.source``.

    - ``simulated`` — reads the synthetic CSV. Always works.
    - ``real_serial`` — DHT22 + PIR over GPIO. Pi-only.
    - ``mqtt`` — subscribes to an MQTT broker.
    """

    SIMULATED = "simulated"
    REAL_SERIAL = "real_serial"
    MQTT = "mqtt"


class LogLevel(str, Enum):
    """Standard library log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class EmbeddingDevice(str, Enum):
    """Allowed values for ``embedding.device``.

    TinyRAG is CPU-only by design — ``cuda`` is accepted for
    future-proofing (a developer laptop with a discrete NVIDIA
    GPU could enable it) but not used by the default install.
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


# ----------------------------------------------------------------------------
# Sub-models (one per section of config.yaml)
# ----------------------------------------------------------------------------


class DeploymentSettings(BaseModel):
    """Hardware profile + the cross-cutting defaults it implies.

    Setting ``target`` to ``raspberry_pi`` permits
    ``sensors.source: real_serial``; setting it to ``laptop``
    rejects it (FR-18 [L] — no GPIO on the laptop). The actual
    per-target sensor default lives in :class:`SensorSettings`; this
    class only carries the identity of the target.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: DeploymentTarget = Field(
        default=DeploymentTarget.LAPTOP,
        description="Hardware profile: 'laptop' or 'raspberry_pi'.",
    )


class ServerSettings(BaseModel):
    """FastAPI bind address + port."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(
        default="127.0.0.1",
        description="Bind address. Use 0.0.0.0 to accept remote connections.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port for the FastAPI app.",
    )


class LLMSettings(BaseModel):
    """LLM model + llama-server endpoint + sampling parameters."""

    # ``protected_namespaces = ()`` silences Pydantic's warning that
    # ``model_path`` collides with the ``model_`` reserved namespace
    # (used by Pydantic for ``model_validate``, ``model_dump``, etc.).
    # The name ``model_path`` is correct for our domain — it really
    # does refer to the on-disk LLM model — so we don't want
    # Pydantic to forbid it.
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )

    model_path: str = Field(
        default="models/phi-3-mini.gguf",
        description=(
            "On-disk GGUF path (relative to project root). Must match a "
            "model id in `src/tinyrag/models/registry.py`."
        ),
    )
    server_url: str = Field(
        default="http://127.0.0.1:8080",
        description="OpenAI-compatible endpoint served by llama-server.",
    )
    context_size: int = Field(
        default=4096,
        ge=128,
        le=32768,
        description="Context window passed to llama-server's --ctx-size.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 = deterministic.",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Hard cap on generated tokens per response.",
    )
    gpu_layers: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Layers to offload to GPU. 0 = CPU-only.",
    )


class EmbeddingSettings(BaseModel):
    """sentence-transformers configuration."""

    # See LLMSettings for the explanation of protected_namespaces=().
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        protected_namespaces=(),
    )

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="sentence-transformers model id (HF hub or local path).",
    )
    device: EmbeddingDevice = Field(
        default=EmbeddingDevice.CPU,
        description="Inference device. TinyRAG is CPU-only by default.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size when embedding multiple chunks at once.",
    )
    cache_dir: str = Field(
        default="models/_hf_cache",
        description="Where sentence-transformers caches its downloaded model.",
    )


class ChunkingSettings(BaseModel):
    """Token-based chunking parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(
        default=400,
        ge=32,
        le=8192,
        description="Target tokens per chunk.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=4096,
        description="Overlap between consecutive chunks (tokens).",
    )
    encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding name used to count tokens.",
    )

    @model_validator(mode="after")
    def _overlap_less_than_size(self) -> ChunkingSettings:
        """Overlap must be strictly less than chunk_size (otherwise you
        loop forever).
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunking.chunk_overlap ({self.chunk_overlap}) must be "
                f"strictly less than chunking.chunk_size ({self.chunk_size})."
            )
        return self


class RetrievalSettings(BaseModel):
    """FAISS-backed retrieval parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_index_path: str = Field(
        default="data/vector_store/doc.faiss",
        description="Where the document-chunk FAISS index is persisted.",
    )
    sensor_index_path: str = Field(
        default="data/vector_store/sensor.faiss",
        description="Where the sensor-summary FAISS index is persisted.",
    )
    doc_top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        description="How many document chunks to retrieve per query.",
    )
    sensor_top_k: int = Field(
        default=2,
        ge=0,
        le=50,
        description="How many sensor-summary chunks to retrieve per query.",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a chunk to count.",
    )
    index_type: Literal["faiss"] = Field(
        default="faiss",
        description="Vector index backend. Only 'faiss' supported in v1.",
    )


class SensorSettings(BaseModel):
    """Pluggable sensor source configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: SensorSource = Field(
        default=SensorSource.SIMULATED,
        description="Active sensor source: simulated | real_serial | mqtt.",
    )
    csv_path: str = Field(
        default="data/sensor_logs/synthetic_30d.csv",
        description="Path to the synthetic sensor CSV (source=simulated).",
    )
    dht_pin: int = Field(
        default=4,
        ge=0,
        le=40,
        description="BCM pin for the DHT22 data line (source=real_serial).",
    )
    pir_pin: int = Field(
        default=17,
        ge=0,
        le=40,
        description="BCM pin for the PIR motion sensor (source=real_serial).",
    )
    mqtt_broker: str = Field(
        default="localhost",
        description="MQTT broker hostname (source=mqtt).",
    )
    mqtt_port: int = Field(
        default=1883,
        ge=1,
        le=65535,
        description="MQTT broker port (source=mqtt).",
    )
    mqtt_topic_prefix: str = Field(
        default="tinyrag/sensors/",
        description="MQTT topic prefix (source=mqtt). Sensor-specific "
        "suffixes are appended at runtime.",
    )


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level: DEBUG | INFO | WARNING | ERROR.",
    )
    json_format: bool = Field(
        default=True,
        description="If true, logs are JSON per line. If false, pretty.",
    )
    path: str | None = Field(
        default="logs/tinyrag.log",
        description="Where logs are written. None disables file logging.",
    )


class PathsSettings(BaseModel):
    """On-disk data layout (gitignored subdirectories of the project root)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents_dir: str = Field(
        default="data/documents/",
        description="Where uploaded documents (PDF/TXT/MD) are stored.",
    )
    metadata_db: str = Field(
        default="data/metadata.db",
        description="Where the SQLite metadata DB is stored.",
    )
    sensor_logs_dir: str = Field(
        default="data/sensor_logs/",
        description="Where the sensor CSVs live.",
    )
    logs_dir: str = Field(
        default="logs/",
        description="Where the rotating log file lives.",
    )


# ----------------------------------------------------------------------------
# Root Settings
# ----------------------------------------------------------------------------


class Settings(BaseModel):
    """The full typed config — every section of config.yaml in one object.

    All fields are required in the *type* sense (they all have defaults
    baked into the schema) but you must provide a config file with all
    nine sections present — :func:`load_settings` rejects a YAML file
    that omits an entire section to catch "you forgot to upgrade your
    config" drift.

    The instance is frozen (immutable). To "change" a setting in tests
    or at runtime, use :meth:`Settings.model_copy` with ``update=``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Private attribute set by load_settings() after construction. It
    # is the project root (parent of config.yaml) used to resolve
    # relative paths like model_path, csv_path, etc. via resolve().
    _project_root: Path | None = PrivateAttr(default=None)

    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    sensors: SensorSettings = Field(default_factory=SensorSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)

    # Note: the laptop-vs-real_serial cross-field check (FR-18 [L])
    # lives in load_settings() rather than as a @model_validator
    # here. Pydantic short-circuits at the first nested-model
    # failure and may not run a Settings-level model_validator, so
    # the user would sometimes miss the cross-field error. Doing it
    # in load_settings() after a successful Settings.model_validate
    # guarantees the check runs every time. The check is also
    # covered by tests/test_config.py::TestCrossFieldValidation.

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def project_root(self) -> Path:
        """Return the directory the config was loaded from.

        This is needed to resolve the many relative paths in the
        config (model_path, csv_path, doc_index_path, etc.) against
        the same anchor. The anchor is set by :func:`load_settings`
        via :attr:`_project_root`; this method is the public accessor.
        """
        if self._project_root is None:
            raise RuntimeError(
                "Settings.project_root() called on a Settings instance "
                "that wasn't created by load_settings(). Construct via "
                "load_settings('config.yaml') instead."
            )
        return self._project_root

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path from config against the project root.

        Absolute paths are returned unchanged. Relative paths are
        joined onto :meth:`project_root`.
        """
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root() / p


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


# Default location of config.yaml: the project root. ``config.yaml`` is
# at the repo root (not under ``src/``) because it's the only file a
# deployer needs to edit.
DEFAULT_CONFIG_PATH = Path("config.yaml")

# Top-level sections that MUST be present in a v1 config.yaml. We
# require them explicitly so that "you forgot to add the new sensors
# section when you upgraded" is caught at startup instead of as a
# confusing ``AttributeError`` deep inside the sensor module.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "deployment",
    "server",
    "llm",
    "embedding",
    "chunking",
    "retrieval",
    "sensors",
    "logging",
    "paths",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read ``path`` as YAML and return the top-level dict.

    Raises :class:`ConfigNotFoundError` if the file is missing,
    :class:`ConfigValidationError` if the YAML is malformed or the
    top-level structure is not a mapping.
    """
    if not path.is_file():
        raise ConfigNotFoundError(
            f"config file not found: {path}\n"
            f"  Hint: copy config.yaml.example (if present) to "
            f"{path.name}, or pass an explicit path to load_settings().",
            path=path,
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}", path=path) from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"config file {path} is not valid YAML: {exc}", path=path
        ) from exc
    if loaded is None:
        # Empty YAML file. Treat as "all defaults" — but warn the user
        # by listing the required sections explicitly.
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigValidationError(
            f"config file {path} must be a YAML mapping at the top level, "
            f"got {type(loaded).__name__}.",
            path=path,
        )
    return loaded


def _check_required_sections(data: dict[str, Any], path: Path) -> None:
    """Reject a config that omits an entire section.

    This is intentionally stricter than letting Pydantic fill in
    defaults: a missing top-level key usually means the config file
    was authored against an older schema, and silently using the new
    default is a debugging nightmare.
    """
    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConfigValidationError(
            f"config file {path} is missing required section(s): "
            f"{', '.join(missing)}. Each of these must be present (even "
            f"if empty `{{}}`) — see config.yaml in the repo for the "
            f"canonical schema.",
            path=path,
        )


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Load and validate ``config.yaml``, returning a :class:`Settings`.

    Parameters
    ----------
    path:
        Path to the config file. ``None`` (the default) loads
        ``config.yaml`` from the *current working directory*. Pass an
        explicit path to load a test fixture or a deployment-specific
        config (e.g. ``/etc/tinyrag/laptop.yaml``).

    Returns
    -------
    Settings
        A fully-validated, immutable Settings object. Every nested
        section is populated (either from the YAML or from the
        Pydantic schema default).

    Raises
    ------
    ConfigNotFoundError
        The file does not exist.
    ConfigValidationError
        The file exists but is malformed YAML, has the wrong
        top-level type, omits a required section, or has a field with
        the wrong type / value (FR-51).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = _read_yaml(config_path)
    _check_required_sections(data, config_path)

    # First pass: collect all field-level errors from Pydantic.
    field_errors: list[str] = []
    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        field_errors = _summarise_pydantic_errors(exc).splitlines()
        # Build a Settings from the validated sections where
        # possible — default-fill anything that didn't validate.
        # This lets us run cross-field checks even when individual
        # fields are broken.
        settings = _build_partial_settings(data)

    # Second pass: cross-field rules on whatever Settings we
    # managed to construct. We always run this — even if there
    # are field errors — because the user benefits from seeing
    # all problems at once.
    cross_errors: list[str] = []
    if (
        settings.deployment.target is DeploymentTarget.LAPTOP
        and settings.sensors.source is SensorSource.REAL_SERIAL
    ):
        cross_errors.append(
            "  - <root>: sensors.source: real_serial is not allowed "
            "when deployment.target is 'laptop' — the Dell laptop "
            "has no GPIO pins. Either set deployment.target to "
            "'raspberry_pi' or set sensors.source to 'simulated' "
            "or 'mqtt'."
        )

    if field_errors or cross_errors:
        all_lines = field_errors + cross_errors
        raise ConfigValidationError(
            f"config file {config_path} failed validation:\n"
            + "\n".join(all_lines),
            path=config_path,
        )

    # Stash the project root so `Settings.resolve()` can anchor
    # relative paths. ``object.__setattr__`` is needed because
    # Settings is frozen via ``model_config = ConfigDict(frozen=True)``;
    # PrivateAttr bypasses the frozen check, so this is the standard
    # way to set a private attribute after construction.
    project_root = config_path.resolve().parent
    object.__setattr__(settings, "_project_root", project_root)
    return settings


def _build_partial_settings(data: dict[str, Any]) -> Settings:
    """Build a Settings from ``data``, default-filling invalid sections.

    This is a recovery helper used by :func:`load_settings` to run
    cross-field checks when some sections of the YAML are invalid.
    It tries each top-level section independently; any section that
    raises a ValidationError is replaced with its schema defaults.
    """
    field_to_class: dict[str, type[BaseModel]] = {
        "deployment": DeploymentSettings,
        "server": ServerSettings,
        "llm": LLMSettings,
        "embedding": EmbeddingSettings,
        "chunking": ChunkingSettings,
        "retrieval": RetrievalSettings,
        "sensors": SensorSettings,
        "logging": LoggingSettings,
        "paths": PathsSettings,
    }
    kwargs: dict[str, Any] = {}
    for field_name, cls in field_to_class.items():
        section = data.get(field_name, {})
        try:
            kwargs[field_name] = cls.model_validate(section)
        except ValidationError:
            # This section is broken; use defaults so the cross-field
            # check still runs on whatever we can salvage.
            kwargs[field_name] = cls()
    return Settings(**kwargs)


def _summarise_pydantic_errors(exc: ValidationError) -> str:
    """Render a Pydantic ValidationError as a beginner-friendly summary.

    Pydantic's default error format is precise but noisy. This
    helper produces one line per failing field, in the same
    ``dot.path: message`` format that ``mypy`` and ``ruff`` use.
    """
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) if err["loc"] else "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Optional: a field validator that demonstrates enum-coercion
# ----------------------------------------------------------------------------


# Pydantic v2 already coerces enum-valued strings to the enum
# automatically when the field type is the enum class. For example,
# a YAML file containing ``deployment.target: laptop`` is loaded
# into ``settings.deployment.target`` as a :class:`DeploymentTarget`
# enum member, not the raw string. No explicit ``field_validator``
# is needed.

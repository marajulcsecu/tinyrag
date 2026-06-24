"""Pluggable sensor data sources — simulated, serial, MQTT.

The :mod:`tinyrag.sensors` subpackage defines the
:class:`~tinyrag.sensors.base.SensorSource` Protocol and provides
concrete implementations. The
:class:`~tinyrag.core.sensor_summarizer.SensorSummarizer` consumes
any :class:`SensorSource` — it doesn't care whether the data is
coming from a CSV file on disk, a DHT22 over GPIO on a Raspberry
Pi, or an MQTT broker.

Modules
-------
- :mod:`tinyrag.sensors.base` — :class:`SensorSource` Protocol
  + the shared :class:`SensorReading` dataclass + the typed
  :class:`SensorSourceError` hierarchy.
- :mod:`tinyrag.sensors.simulated` — :class:`SimulatedCSVSource`
  (reads the synthetic 30-day CSV from Step 3.8). The default
  for the laptop path.
- :mod:`tinyrag.sensors.serial_dht` — :class:`RealSerialSource`
  (DHT22 temperature/humidity + PIR motion, over GPIO on the Pi).
  **Phase 4 stub** — raises :class:`NotImplementedError` from
  :meth:`read`; full implementation lands in Phase 6.
- :mod:`tinyrag.sensors.mqtt` — :class:`MQTTBrokerSource`
  (subscribes to an MQTT topic for wireless sensors). **Phase 4
  stub** — same intent as ``serial_dht``.

Why a subpackage and not a single file?
---------------------------------------
- The three concrete sources have wildly different dependencies
  (``paho-mqtt``, ``RPi.GPIO``, ``serial``). Only the simulated
  one is importable on a developer laptop. Splitting them means
  a missing optional dep on the laptop only fails when the user
  *tries* to instantiate the affected source, not on import.
- Adding a new sensor type (e.g. Zigbee, Bluetooth) is a one-file
  change in this subpackage plus a one-line addition to the
  ``SensorSource`` factory in :mod:`tinyrag.config`.

Why stubs for the Pi-only sources?
-----------------------------------
The architecture doc (``docs/03_architecture_v1.md`` §6.1) lists
three implementations; all three modules must be importable so the
``config.yaml`` factory can resolve any of the three values of
``sensors.source`` without ``ModuleNotFoundError``. The stubs
satisfy the :class:`SensorSource` Protocol (the cheap
:meth:`available_sensors` methods work on a laptop) and fail
clearly on :meth:`read` with a pointer to the Phase 6 roadmap
step.

Location: ``src/tinyrag/sensors/``
"""

from __future__ import annotations

from tinyrag.sensors.base import (
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
    SensorReading,
    SensorSource,
    SensorSourceConfigError,
    SensorSourceError,
    SensorSourceReadError,
    SensorSourceSchemaError,
)
from tinyrag.sensors.mqtt import MQTTBrokerSource
from tinyrag.sensors.serial_dht import RealSerialSource
from tinyrag.sensors.simulated import SimulatedCSVSource

__all__ = [
    # base — Protocol + types + errors
    "MQTTBrokerSource",
    "RealSerialSource",
    "REQUIRED_COLUMNS",
    "SENSOR_TYPE_ENERGY",
    "SENSOR_TYPE_HUMIDITY",
    "SENSOR_TYPE_MOTION",
    "SENSOR_TYPE_TEMPERATURE",
    "SUPPORTED_SENSOR_TYPES",
    "SUPPORTED_UNITS",
    "SensorReading",
    "SensorSource",
    "SensorSourceConfigError",
    "SensorSourceError",
    "SensorSourceReadError",
    "SensorSourceSchemaError",
    "SimulatedCSVSource",
    "UNIT_CELSIUS",
    "UNIT_COUNT",
    "UNIT_KWH",
    "UNIT_PERCENT",
]

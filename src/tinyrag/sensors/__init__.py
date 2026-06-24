"""Pluggable sensor data sources — simulated, serial, MQTT.

The :mod:`tinyrag.sensors` subpackage defines the ``SensorSource``
Protocol and provides concrete implementations. The
:class:`~tinyrag.core.sensor_summarizer.SensorSummarizer` consumes any
``SensorSource`` — it doesn't care whether the data is coming from a
CSV file on disk, a DHT22 over GPIO on a Raspberry Pi, or an MQTT
broker.

Modules (to be added in later Phase 4 steps)
--------------------------------------------
- :mod:`tinyrag.sensors.base` — ``SensorSource`` Protocol + the
  shared :class:`SensorReading` dataclass.
- :mod:`tinyrag.sensors.simulated` — ``SimulatedCSVSource`` (reads
  the synthetic 30-day CSV from Step 3.8).
- :mod:`tinyrag.sensors.serial_dht` — ``RealSerialSource`` (DHT22
  temperature/humidity + PIR motion, over GPIO on the Pi).
- :mod:`tinyrag.sensors.mqtt` — ``MQTTBrokerSource`` (subscribes to
  an MQTT topic for wireless sensors).

Why a subpackage and not a single file?
---------------------------------------
- The three concrete sources have wildly different dependencies
  (``paho-mqtt``, ``serial``, ``RPi.GPIO``). Only the simulated one
  is importable on a developer laptop. Splitting them means a missing
  optional dep on the laptop only fails when the user *tries* to
  instantiate the affected source, not on import.
- Adding a new sensor type (e.g. Zigbee, Bluetooth) is a one-file
  change in this subpackage plus a one-line addition to the
  ``SensorSource`` factory in :mod:`tinyrag.config`.

Location: ``src/tinyrag/sensors/``
"""

from __future__ import annotations

# Subpackage is currently a placeholder. Modules will be re-exported
# here as they are implemented in later Phase 4 steps (4.13).

"""MQTTBrokerSource — subscribes to a local Mosquitto broker for wireless sensors.

**Stub for Phase 4.** The Phase 6 deployment step will fill in the
real implementation; for now this module exists so the architecture
is complete (``docs/03_architecture_v1.md`` §6.1 lists three
implementations; all three must be importable) and the
``config.yaml`` factory can resolve ``sensors.source: mqtt`` without
``ImportError``.

Why the stub now
----------------
- Mirrors :mod:`tinyrag.sensors.serial_dht`: the real implementation
  needs ``paho-mqtt`` (not currently pinned) and a running broker
  (not available on the laptop path). Having a stub means a misconfig
  fails with a clear ``NotImplementedError`` pointing at the
  roadmap, not a missing-module surprise.
- Lets the unit-test suite import :class:`MQTTBrokerSource` and
  exercise the construction surface without a network dep.

Topic convention (MQTT)
-----------------------
- Single topic prefix: ``config.yaml``'s ``sensors.mqtt_topic_prefix``
  (default ``tinyrag/sensors/``).
- Per-sensor topic: ``{prefix}{sensor_id}`` — e.g.
  ``tinyrag/sensors/living_room_temp``.
- Payload: JSON ``{"timestamp": "ISO8601", "value": float, "unit": "C"}``
  (matches the canonical CSV schema's :data:`~tinyrag.sensors.base.REQUIRED_COLUMNS`).

These will be enforced by the real implementation in Phase 6; the
stub carries the config keys but doesn't use them yet.

Why a placeholder for ``available_sensors``
-------------------------------------------
Unlike :class:`~tinyrag.sensors.serial_dht.RealSerialSource` (which
knows its 3 IDs up front because the hardware is fixed), the set
of sensors an MQTT broker carries is broker-dependent — we can't
return a useful list without subscribing. The stub returns
``[]`` (a value :meth:`read` consumers can detect as "don't trust
this list, the broker decides").

Location: ``src/tinyrag/sensors/mqtt.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class MQTTBrokerSource:
    """A :class:`SensorSource` backed by an MQTT broker subscription.

    **Phase 4 stub.** Construction is allowed; :meth:`read` raises
    :class:`NotImplementedError` until Phase 6 lands.
    :meth:`available_sensors` returns an empty list because the set
    of broker-published sensors is broker-dependent and can only
    be discovered by subscribing (the real implementation will
    return the subscribed set after ``connect()``).

    Parameters
    ----------
    host:
        Broker hostname or IP (e.g. ``"127.0.0.1"``, ``"mosquitto.local"``).
    port:
        Broker port. Default MQTT port is 1883 (plaintext) or 8883
        (TLS). Range 1-65535.
    topic_prefix:
        MQTT topic prefix. Per-sensor topics are ``{prefix}{sensor_id}``.
        Convention: trailing slash, e.g. ``"tinyrag/sensors/"``.
    username:
        Optional. Most local Mosquitto setups are unauthenticated;
        the lab broker may require creds.
    password:
        Optional. Paired with ``username``.

    Attributes
    ----------
    host
    port
    topic_prefix
    username
    password

    Notes
    -----
    No port-range or host-shape validation here — that's the real
    implementation's job (a Phase 6 ``connect()`` will fail with a
    clear ``paho`` exception that the API layer can map to 503).
    Adding redundant validation in the stub risks drift from the
    real implementation's view of the same constraints.
    """

    host: str
    port: int
    topic_prefix: str
    username: str | None = None
    password: str | None = None

    def read(self, since: datetime | None = None) -> pd.DataFrame:
        """Subscribe to the broker and return cached readings — Phase 6 only.

        Raises
        ------
        NotImplementedError
            Always. The Phase 6 deployment step will replace this
            body with a ``paho-mqtt`` client that subscribes to
            ``{topic_prefix}#``, buffers incoming payloads in a
            small in-memory deque, and returns the buffered DataFrame.
        """
        raise NotImplementedError(
            "MQTTBrokerSource is a Phase 6 implementation. "
            "See docs/06_roadmap_v2.md — Phase 6 (Pi deployment). "
            "Use SimulatedCSVSource for laptop development."
        )

    def available_sensors(self) -> list[str]:
        """Return an empty list (broker-dependent — unknown until subscribed).

        Returns ``[]`` because the set of sensors an MQTT broker
        carries is broker-dependent and can only be discovered
        by subscribing. A consumer that needs an up-front list
        should use the CLI config (``sensors.mqtt_topic_prefix``)
        instead.
        """
        return []


__all__ = ["MQTTBrokerSource"]

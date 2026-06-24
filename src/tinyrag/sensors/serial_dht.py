"""RealSerialSource — reads DHT22 + PIR over GPIO (Raspberry Pi 5 only).

**Stub for Phase 4.** The Phase 6 deployment step will fill in the
real implementation; for now this module exists so the architecture
is complete (``docs/03_architecture_v1.md` §6.1 lists three
implementations; all three must be importable) and the
``config.yaml`` factory can resolve ``sensors.source: real_serial``
without ``ImportError``.

Why the stub now
----------------
- The composition root (``main.py``, Step 4.17) imports every
  concrete source lazily at startup so a misconfigured source
  fails with a clear message. With no module at all, the
  ``sensors.source: real_serial`` config value would cause a
  ``ModuleNotFoundError`` at startup that's harder to diagnose.
- The class still implements the :class:`SensorSource` Protocol
  (:meth:`available_sensors` works on the laptop) so unit tests
  can construct it and check its surface without needing GPIO.

Why a separate module file (not a ``NotImplementedError`` raise in
:mod:`tinyrag.sensors.simulated`)
------------------------------------------------------------------
The real implementation will need optional hardware deps
(``RPi.GPIO``, ``adafruit-circuitpython-dht``, ``libgpiod``) that
must NOT be importable on a laptop. Splitting the file means a
``from tinyrag.sensors.serial_dht import RealSerialSource`` on a
laptop succeeds (the class is here, just not wired to GPIO), but a
``RealSerialSource(pin=...).read()`` call fails cleanly with
``NotImplementedError`` pointing at the Phase 6 step.

Pin convention (BCM numbering, per Raspberry Pi docs)
-----------------------------------------------------
- ``dht_pin``: the DHT22 data line. DHT22 is a single-wire protocol;
  any free GPIO works. Common choices: GPIO4 (default for the
  AdaFruit tutorial), GPIO17, GPIO27.
- ``pir_pin``: the PIR motion sensor's data line. Active HIGH;
  the BISS0001 chip in most PIR modules pulls it HIGH on motion.
  Any free GPIO works. Common: GPIO17, GPIO22, GPIO23.

These will be configurable via ``config.yaml``'s
``sensors.dht_pin`` / ``sensors.pir_pin`` (already in the schema
from Step 4.2).

Location: ``src/tinyrag/sensors/serial_dht.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class RealSerialSource:
    """A :class:`SensorSource` backed by a DHT22 + PIR over GPIO.

    **Phase 4 stub.** Construction is allowed; :meth:`read` raises
    :class:`NotImplementedError` until Phase 6 lands. :meth:`available_sensors`
    returns the three sensor IDs the real implementation will produce
    so the API ``/api/status`` endpoint can show "DHT22 + PIR are
    configured" on a laptop without crashing.

    Parameters
    ----------
    dht_pin:
        BCM pin number for the DHT22 data line. Range 2-27 (excluding
        0, 1, 9, 10, 11, 14, 15, 20-27 are reserved on some Pi models
        — see the Phase 6 implementation for the full guard).
    pir_pin:
        BCM pin number for the PIR motion sensor's data line.

    Attributes
    ----------
    dht_pin
    pir_pin

    Notes
    -----
    Pin validation is intentionally **not** done in the stub — the
    real implementation in Phase 6 will need to check against the
    Pi's actual pin map (which varies by board revision). Adding
    the check here would be the kind of premature code that
    becomes wrong the moment a new Pi model ships.
    """

    dht_pin: int
    pir_pin: int

    def read(self, since: datetime | None = None) -> pd.DataFrame:
        """Read live readings from the DHT22 + PIR — Phase 6 only.

        Raises
        ------
        NotImplementedError
            Always. The Phase 6 deployment step
            (``docs/06_roadmap_v2.md``) will replace this body with
            a libgpiod-backed read loop. The error message points
            at the right roadmap step so a future contributor can
            find the spec immediately.
        """
        raise NotImplementedError(
            "RealSerialSource is a Phase 6 implementation. "
            "See docs/06_roadmap_v2.md — Phase 6 (Pi deployment). "
            "Use SimulatedCSVSource for laptop development."
        )

    def available_sensors(self) -> list[str]:
        """Return the sensor IDs the real implementation will produce.

        Returns the three canonical IDs the Phase 6 implementation
        will yield — DHT22 reports temperature + humidity on a
        single physical wire, and the PIR reports motion. The
        order is sorted for stable assertions in tests.

        Unlike :meth:`read`, this method is fully functional on a
        laptop (it doesn't touch GPIO), so unit tests can use it
        to verify the class structure.
        """
        return ["dht22_hum", "dht22_temp", "pir_motion"]


__all__ = ["RealSerialSource"]

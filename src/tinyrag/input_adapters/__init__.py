"""Pluggable user input adapters — text (always), voice (stretch).

The :mod:`tinyrag.input_adapters` subpackage defines the
``InputAdapter`` Protocol and provides concrete implementations. The
chat page and the CLI both depend on ``InputAdapter``; they don't care
whether the user typed the query or spoke it.

Modules (to be added in later Phase 4 steps)
--------------------------------------------
- :mod:`tinyrag.input_adapters.base` — ``InputAdapter`` Protocol +
  shared types.
- :mod:`tinyrag.input_adapters.text` — ``TextInputAdapter`` (always
  works; passes the typed string through).
- :mod:`tinyrag.input_adapters.voice` — ``VoiceInputAdapter``
  (stretch goal only; depends on Whisper.cpp + a working microphone).
  Commented in the architecture doc as ``# voice.py`` because it is
  optional.

Why a subpackage and not a single file?
---------------------------------------
- Voice input pulls in heavy native deps (Whisper.cpp, a sound
  library). Keeping it in its own file means a laptop without a
  microphone can still ``import tinyrag.input_adapters`` cleanly.
- Adding a new input mode (e.g. a CLI flag, a Telegram bot) is a
  one-file change in this subpackage.

Location: ``src/tinyrag/input_adapters/``
"""

from __future__ import annotations

# Subpackage is currently a placeholder. Modules will be re-exported
# here as they are implemented in later Phase 4 steps (4.19+).

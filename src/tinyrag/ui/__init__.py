"""Web UI — static HTML, CSS, and JavaScript served by FastAPI.

The :mod:`tinyrag.ui` subpackage holds the front-end assets: the
``index.html`` (chat page), ``admin.html`` (document management page),
their shared ``style.css``, and the per-page JavaScript files
(``chat.js``, ``admin.js``).

These are *static* files. They are served by FastAPI's
``StaticFiles`` mount (configured in :mod:`tinyrag.main`) and contain
no Python. The ``__init__.py`` exists only so ``tinyrag.ui`` is a
proper Python subpackage — it has no runtime symbols.

Subdirectories
--------------
- ``ui/static/`` — ``style.css``, ``chat.js``, ``admin.js`` (no
  build step, no React/Vue, no npm).
- ``ui/templates/`` — ``index.html``, ``admin.html`` (Jinja2
  templates; the chat page is server-rendered once and then JS takes
  over).

Location: ``src/tinyrag/ui/``
"""

from __future__ import annotations

# No Python symbols to re-export — this subpackage is a static asset
# container. The presence of the ``__init__.py`` is what makes the
# directory importable as a subpackage; it does not define a Python API.

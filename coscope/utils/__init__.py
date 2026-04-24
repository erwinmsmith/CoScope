"""
Deprecated utilities namespace — contents migrated.

All real modules have been moved to their proper subsystem:

- Dataset loaders    -> :mod:`coscope.io.loaders`
- Serializer / IO   -> :mod:`coscope.io`
- Subset split      -> :mod:`coscope.evaluation.split`
- LLM clients       -> :mod:`coscope.llm`

This package is kept as an empty shell so that old wheel-cached imports do
not produce an ImportError. Remove it once all consumers have been updated.
"""

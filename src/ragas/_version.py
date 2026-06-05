# Static version stand-in for the vendored ragas library.
#
# Upstream ragas generates this file via setuptools-scm at build time
# (pyproject: version_file = "src/ragas/_version.py"). This AISC plugin builds
# the vendored copy with hatchling, which does NOT run setuptools-scm, so the
# file is never generated and `from ragas._version import __version__` (used by
# ragas/_analytics.py, ragas/prompt/base.py, ragas/prompt/pydantic_prompt.py —
# all on the core eval path) raises ModuleNotFoundError. Shipping it statically
# keeps the vendored library importable. The value is informational only
# (analytics + prompt-format-compat warnings); exact number is not significant.
__version__ = version = "0.3.0"
__version_tuple__ = version_tuple = (0, 3, 0)

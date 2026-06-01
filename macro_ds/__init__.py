"""macro_ds — shared library for the macro deep-search distillation pipeline.

The same code drives both teacher trace generation (OpenAI) and student serving
(vLLM), guaranteeing train/serve parity. See docs/plans/ for the design.
"""

__version__ = "0.1.0"

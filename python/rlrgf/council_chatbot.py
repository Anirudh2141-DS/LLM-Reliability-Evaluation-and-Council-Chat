"""
Compatibility entrypoint for the legacy council chatbot.

This module now delegates to the canonical Streamlit dashboard to avoid
maintaining duplicate council-chat logic paths.
"""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("rlrgf.dashboard", run_name="__main__")


if __name__ == "__main__":
    main()

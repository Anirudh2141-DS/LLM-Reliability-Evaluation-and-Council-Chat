"""
Main entry point for the rlrgf package.
"""

import asyncio
import sys

from .run_experiment import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

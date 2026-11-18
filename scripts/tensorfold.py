#!/usr/bin/env python3
"""Run the TensorFold engineering adapter without installing SGLang."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specpilot_tensorfold.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

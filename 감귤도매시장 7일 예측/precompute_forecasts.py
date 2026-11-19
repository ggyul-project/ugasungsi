"""Create the saved forecast CSV consumed by the Streamlit dashboard."""

import os
import runpy
from pathlib import Path


os.environ["CITRUS_PRECOMPUTE_ONLY"] = "1"
app_path = Path(__file__).with_name("app.py")

try:
    runpy.run_path(str(app_path), run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise

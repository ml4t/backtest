"""
Zipline extension to register the validation bundle.

This file should be copied or symlinked to ~/.zipline/extension.py
or $ZIPLINE_ROOT/extension.py
"""

import sys
from pathlib import Path

# Add bundle directory to path so we can import the ingest function
BUNDLE_DIR = Path(__file__).parent
sys.path.append(str(BUNDLE_DIR))

from zipline.data.bundles import register
from validation_ingest import validation_to_bundle

# Register the validation bundle
register(
    'validation',
    validation_to_bundle(),
    calendar_name='NYSE',  # Using standard NYSE calendar
)

print("✅ Registered 'validation' bundle for Zipline")

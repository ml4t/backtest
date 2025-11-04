"""
Zipline extension to register the validation bundle.

This file should be copied or symlinked to ~/.zipline/extension.py
or $ZIPLINE_ROOT/extension.py
"""

import sys
import os
from pathlib import Path

# When loaded via exec() by Zipline, __file__ is not defined
# Use __name__ and sys.modules to find the extension location
try:
    BUNDLE_DIR = Path(__file__).parent
except NameError:
    # Fallback: use ZIPLINE_ROOT environment variable
    zipline_root = os.environ.get('ZIPLINE_ROOT')
    if zipline_root:
        BUNDLE_DIR = Path(zipline_root)
    else:
        # Last resort: use current working directory
        BUNDLE_DIR = Path.cwd() / 'bundles' / '.zipline_root'

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

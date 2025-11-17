"""Ingest test_data bundle."""

import sys
sys.path.insert(0, '.')

# Import and register the bundle
from bundles.test_data_bundle import *

# Now ingest it
from zipline.data.bundles import ingest

print("Ingesting test_data bundle...")
ingest('test_data', show_progress=True)
print("✓ Bundle ingested successfully!")

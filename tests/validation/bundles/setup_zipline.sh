#!/bin/bash
# Setup script for Zipline validation bundle

set -e  # Exit on error

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIPLINE_ROOT="$BUNDLE_DIR/.zipline_root"

echo "Setting up Zipline validation bundle..."
echo "Bundle directory: $BUNDLE_DIR"
echo "Zipline root: $ZIPLINE_ROOT"
echo

# Create Zipline root directory
mkdir -p "$ZIPLINE_ROOT"

# Copy extension.py
echo "Copying extension.py..."
cp "$BUNDLE_DIR/extension.py" "$ZIPLINE_ROOT/"

# Set environment variable
export ZIPLINE_ROOT="$ZIPLINE_ROOT"

echo "✅ Zipline root configured"
echo
echo "To use this bundle, run:"
echo "  export ZIPLINE_ROOT=$ZIPLINE_ROOT"
echo "  zipline ingest -b validation"
echo
echo "Or add to your shell profile:"
echo "  echo 'export ZIPLINE_ROOT=$ZIPLINE_ROOT' >> ~/.bashrc"

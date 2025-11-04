#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit

cd ..
rm -rf build
rm -rf vectorbtpro.egg-info
python -m build

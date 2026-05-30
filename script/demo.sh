#!/usr/bin/env bash
set -euo pipefail

python main.py
python post/demo_tspn_analysis.py

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "VAA FlowVision CrowdFlow V6 Complete — http://localhost:8501"
python validate_v6.py
python -m streamlit run app.py

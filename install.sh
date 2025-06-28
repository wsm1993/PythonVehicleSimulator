#!/bin/bash
# setup.sh

# Create virtual environment
python3 -m venv ship_env
source ship_env/bin/activate

pip install -e .
pip install -r requirements.txt

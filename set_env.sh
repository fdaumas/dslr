#!/usr/bin/sh

# to call: `source set_env.sh`

python3 -m venv ENV_dslr
source ENV_dslr/bin/activate
pip install -r requirements.txt
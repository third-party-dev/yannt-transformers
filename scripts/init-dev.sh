#!/usr/bin/env bash

export PROJ_PATH=$(pwd)
mkdir -p ./cache/venv
python3 -m venv ./cache/venv/hft
[ $? -ne 0 ] && { echo "Failed to create venv"; exit 1; }

./cache/venv/hft/bin/pip install -U pip
./cache/venv/hft/bin/pip install -U setuptools wheel build pytest
./cache/venv/hft/bin/pip install -U -e .

echo
echo "The environment is now ready. Try 'hft --help' for information."

# Include yannt tab completion.
TMP_RC="$(mktemp)"
cat >> "$TMP_RC" <<'EOF'
[ -f "$HOME/.bashrc" ] && source $HOME/.bashrc
source ${PROJ_PATH}/cache/venv/hft/bin/activate
#eval "$(register-python-argcomplete hft)"
EOF

exec bash --rcfile "$TMP_RC" -i
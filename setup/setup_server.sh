#!/bin/bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

echo "=== Setting up imitation learning environment ==="

# Create isolated venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install imitation with Atari extras
pip install -e ".[atari]"

# Download Atari ROMs (non-interactive)
autorom --accept-license

# Verify ALE registration
python3 -c "
import gymnasium as gym
env = gym.make('PongNoFrameskip-v4')
print('ALE environment OK:', env.observation_space)
env.close()
print('Setup complete!')
"

#!/usr/bin/env bash
# File: start_harmony_tmux.sh
# Starts a tmux session with 2 windows:
#   1. Python server
#   2. ngrok tunnel

SESSION_NAME="harmony"
APP_DIR="${HARMONY_APP_DIR:-harmony}"
PORT="${PORT:-7000}"
HARMONY_NGROK_URL="${HARMONY_NGROK_URL:-harmony.ngrok.app}"

# If the session already exists, just attach
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to existing tmux session '$SESSION_NAME'..."
  exec tmux attach -t "$SESSION_NAME"
fi

echo "Creating tmux session '$SESSION_NAME'..."

# Create new detached session
tmux new-session -d -s "$SESSION_NAME" -n "server" \
  "PYTHONUNBUFFERED=1 nix run .#harmony; read"

# Second window: ngrok
if [[ -n "$HARMONY_NGROK_URL" ]]; then
  tmux new-window -t "$SESSION_NAME" -n "ngrok" \
    "ngrok http $PORT; read"
else
  tmux new-window -t "$SESSION_NAME" -n "ngrok" \
    "ngrok http $PORT --url https://$HARMONY_NGROK_URL; read"
fi

# Show both windows
tmux select-window -t "$SESSION_NAME:0"
tmux attach -t "$SESSION_NAME"

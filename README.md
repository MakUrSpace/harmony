# Harmony
Harmony integrates tabletop gaming utilities into a game recording and referreeing system.

## Setup

### Dependencies -- [Nix](https://nixos.org/download/)

Harmony's primary dependency is nix which manages packaging and running Harmony. To use Harmony, install Nix and enable both flakes and nix-commands.

1. Install Nix: `sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon`
1. Add to `/etc/nix/nix.conf`: `experimental-features = flakes nix-command`

### Execute Harmony

`nix run "github:makurspace/harmony#harmony-ngrok" --refresh`

### Host Harmony with [ngrok](https://ngrok.com/)

Using ngrok allows exposing a Harmony server to a public URL with HTTPS for sharing.

#### Configure ngrok with token:

`NIXPKGS_ALLOW_UNFREE=1 nix run --impure "github:makurspace/harmony#register-ngrok" -- {ngrok token}`

#### Launch Harmony and ngrok

The following command starts both ngrok and Harmony in tmux session:

`NIXPKGS_ALLOW_UNFREE=1 nix run --impure "github:makurspace/harmony#harmony-ngrok" --refresh`

### Using a locally cloned Harmony repo

You can execute the above commands with a locally cloned repo by replacing `github:makurspace/harmony#` with `.#` when executing the `nix` commands from a directory in the repo.

## Developer Setup

### devShell

Harmony provides a nix devShell for development. The devShell provides an environment with all dependencies to execute Harmony and the shell's PYTHONPATH updated to use the application code directly (bypassing the Nix package).

1. Start the devShell with: `nix develop`
1. Start harmony: `python harmony/harmonyServer.py`

### [Jupyter](https://jupyter.org/)

Harmony provides a Jupyter development environment. This can started with `nix run .#jupyter`, which will start a Jupyter server at [https://localhost:8888](https://localhost:8888) with the token phrase "harmony".

## Harmony is built on:
* [htmx](https://htmx.org/)
* [flask](https://flask.palletsprojects.com/en/stable/)
* [python 3](https://www.python.org/)
* [SQLite](https://docs.python.org/3/library/sqlite3.html)
* [OpenCV](https://opencv.org/)
* [nix](https://nixos.org/guides/how-nix-works/)


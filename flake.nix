{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [];
      systems = [ "x86_64-linux" "aarch64-linux" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: 
      with pkgs;
      with python3Packages;
      let 
        opencv-contrib-python = buildPythonPackage rec {
          pname = "opencv-contrib-python";
          version = "4.12.0.88";
          pyproject = true;
          build-system = [ setuptools ];
          build-inputs = [ scikit-learn ];
          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-Dx4igjqs4JBnuaDo4rS6bXoe8IgH1s6+oxXzEz9Bmg4=";
          };
        };

        ipynb = buildPythonPackage rec {
          pname = "ipynb";
          version = "0.5.1";
          pyproject = true;
          build-system = [ setuptools ];
          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-jYNMd3yjiFKJk4cozDgvCByGpY6Slh6G8KumDJaTjOU=";
          };
        };

	harmony-deps = [
          jupyterlab
          requests
          flask
          gunicorn
          ipynb
          ipynbname
          nest-asyncio
          opencv4
          imutils
          matplotlib
          uvicorn
          graphviz
          pyzbar
          aiohttp 
        ];

	harmony = buildPythonPackage {
          pname = "harmony";
          version = "0.0.1";
          src = ./.;
          pyproject = true;
          build-system = [ setuptools setuptools-scm ];
          propagatedBuildInputs = harmony-deps;
        };

    	harmony-dev-env = python.withPackages (ps: with ps; harmony-deps ++ [ pytest pytest-cov ]);
      in {
        packages.default = harmony;
        packages.harmony = harmony;
        packages.jupyter = writeShellApplication {
          name = "jupyter";
          runtimeInputs = [ harmony-dev-env ];
          text = ''
            #!/usr/bin/env bash
            # Run from whatever dir `nix run` is invoked in.
            # If your code lives in ./src, expose it to Python:
            export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  
            echo "Starting Jupyter Lab in $PWD"
            echo "PYTHONPATH=$PYTHONPATH"
  
            exec jupyter lab --notebook-dir="." --ip=0.0.0.0 --no-browser --NotebookApp.token="harmony"
          '';
        };
        packages.register-ngrok = writeShellApplication {
          name = "register-ngrok";
          runtimeInputs = [ ngrok ];
          text = ''
            set -euo pipefail
    
            if [ "$#" -ne 1 ]; then
              echo "Usage: nix run .#ngrok-auth -- <NGROK_AUTHTOKEN>" >&2
              exit 1
            fi
    
            exec ngrok config add-authtoken "$1"
          '';
        };
        packages.harmony-ngrok = writeShellApplication {
          name = "harmony-ngrok";
          runtimeInputs = [ self'.packages.harmony ngrok tmux ];
          text = ''
            # Starts a tmux session with 3 panes:
            #   1. Python server (port 7000/7001)
            #   2. ngrok tunnel for Harmony (7000)
            #   3. ngrok tunnel for Admin (7001)
            
            SESSION_NAME="harmony"
            PORT="''${PORT:-7000}"
            ADMIN_PORT="''${ADMIN_PORT:-7001}"
            
            # Arguments for URLs
            HARMONY_URL="''${1:-harmony.ngrok.app}"
            ADMIN_URL="''${2:-harmony-admin.ngrok.app}"
            
            # If the session already exists, just attach
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
              echo "Attaching to existing tmux session '$SESSION_NAME'..."
              exec tmux attach -t "$SESSION_NAME"
            fi
            
            echo "Creating tmux session '$SESSION_NAME'..."
            
            # Create new detached session with the server
            tmux new-session -d -s "$SESSION_NAME" -n "harmony-stack" \
              "PYTHONUNBUFFERED=1 harmony; read"
            
            # Split for Harmony Ngrok (horizontal split)
            tmux split-window -t "$SESSION_NAME:0" -h \
              "echo 'Exposing Harmony on $HARMONY_URL -> $PORT'; ngrok http $PORT --domain=$HARMONY_URL; read"
            
            # Split for Admin Ngrok (vertical split of the new pane)
            tmux split-window -t "$SESSION_NAME:0.1" -v \
              "echo 'Exposing Admin on $ADMIN_URL -> $ADMIN_PORT'; ngrok http $ADMIN_PORT --domain=$ADMIN_URL; read"
            
            # Reorganize tiles
            tmux select-layout -t "$SESSION_NAME:0" tiled
            
            # Attach
            tmux attach -t "$SESSION_NAME"
          '';
        };
        packages.tests = writeShellApplication {
          name = "run-tests";
          runtimeInputs = [ harmony-dev-env self'.packages.harmony ];
          text = ''
            pytest tests/ "$@"
          '';
        };
        devShells.default = pkgs.mkShell {
          name = "harmonyShell";
          packages = [ harmony-dev-env ];
          shellHook = ''
            PROJECT_NAME="$(basename "$PWD")"
            export PYTHONPATH="$PWD/harmony:$PWD/observer:${PYTHONPATH:-}"
            export PS1="\[\e[1;36m\][$PROJECT_NAME]\[\e[0m\] \w \$ "
          '';
        };
      };
      flake = {};
    };
}

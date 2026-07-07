{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
  };

  outputs = inputs@{ flake-parts, crane, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [];
      systems = [ "x86_64-linux" "aarch64-linux" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: 
      with pkgs;
      with python3Packages;
      let 
        craneLib = crane.mkLib pkgs;
        harmony-rs = craneLib.buildPackage {
          pname = "harmony-web";
          version = "0.1.0";
          src = pkgs.lib.cleanSourceWith {
            src = craneLib.path ./.;
            filter = path: type:
              (builtins.match ".*static/.*" path != null) ||
              (builtins.match ".*html$" path != null) ||
              (craneLib.filterCargoSources path type);
          };
          strictDeps = true;
          buildInputs = [
            pkgs.opencv4
            pkgs.clang
          ];
          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.clang
            pkgs.rustPlatform.bindgenHook
            pkgs.makeWrapper
          ];
          postInstall = ''
            mkdir -p $out/share/harmony-web
            cp -r harmony-web/static $out/share/harmony-web/static
            
            wrapProgram $out/bin/harmony-web \
              --set HARMONY_STATIC_DIR $out/share/harmony-web/static
          '';
        };

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
          fastapi
          jinja2
          python-multipart
          graphviz
          pyzbar
          aiohttp 
          hypercorn
        ];

	harmony = buildPythonPackage {
          pname = "harmony";
          version = "0.0.1";
          src = ./.;
          pyproject = true;
          build-system = [ setuptools setuptools-scm ];
          propagatedBuildInputs = harmony-deps;
        };

    	harmony-dev-env = python.withPackages (ps: with ps; harmony-deps ++ [ pytest pytest-cov pytest-playwright ]);
      in {
        packages.default = harmony;
        packages.harmony = harmony;
        packages.harmony-rs = harmony-rs;
        packages.jupyter = writeShellApplication {
          name = "jupyter";
          runtimeInputs = [ harmony-dev-env pkgs.nodejs_20 pkgs.nodePackages.npm ];
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
          runtimeInputs = [ self'.packages.harmony-rs ngrok tmux ];
          text = ''
            # Starts a tmux session with 3 panes:
            #   1. Rust server (port 8081/8080)
            #   2. ngrok tunnel for Harmony (8081)
            #   3. ngrok tunnel for Admin (8080)
            
            SESSION_NAME="harmony"
            PORT="''${PORT:-8081}"
            ADMIN_PORT="''${ADMIN_PORT:-8080}"
            
            # Arguments for URLs
            ADMIN_URL="''${1:-harmony-admin.ngrok.app}"
            HARMONY_URL="''${2:-harmony.ngrok.app}"
            
            # If the session already exists, just attach
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
              echo "Attaching to existing tmux session '$SESSION_NAME'..."
              exec tmux attach -t "$SESSION_NAME"
            fi
            
            echo "Creating tmux session '$SESSION_NAME'..."
            
            # Create new detached session with the server
            tmux new-session -d -s "$SESSION_NAME" -n "harmony-stack" \
              "harmony-web; read"
            
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
          runtimeInputs = [ harmony-dev-env pkgs.nodejs_20 pkgs.nodePackages.npm pkgs.playwright-driver ];
          text = ''
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

            # Ensure node_modules are present for vitest
            if [ ! -d "node_modules" ]; then
              echo "node_modules not found, running npm install..."
              npm install
            fi

            EXIT_CODE=0
            echo "--- Running Python tests (pytest) ---"
            if ! pytest tests/ --cov=harmony --cov=observer --cov-report=term-missing "$@"; then
              EXIT_CODE=1
            fi

            echo ""
            echo "--- Running UI Javascript tests (vitest) ---"
            if ! npm test; then
              EXIT_CODE=1
            fi

            exit "$EXIT_CODE"
          '';
        };
        packages.test = self'.packages.tests;
        formatter = pkgs.writeShellApplication {
          name = "nix-fmt";
          runtimeInputs = [ pkgs.treefmt pkgs.ruff ];
          text = ''
            exec treefmt "$@"
          '';
        };
        devShells.default = pkgs.mkShell {
          name = "harmonyShell";
          packages = [ harmony-dev-env pkgs.nodejs_20 pkgs.nodePackages.npm pkgs.playwright-driver pkgs.treefmt pkgs.ruff pkgs.rustc pkgs.cargo pkgs.pkg-config pkgs.opencv4 pkgs.clang pkgs.rustPlatform.bindgenHook ];
          shellHook = ''
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

            PROJECT_NAME="$(basename "$PWD")"
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
            export PYTHONPATH="$PWD/harmony:$PWD/observer:${PYTHONPATH:-}"
            export PS1="\[\e[1;36m\][$PROJECT_NAME]\[\e[0m\] \w \$ "
          '';
        };
      };
      flake = {};
    };
}

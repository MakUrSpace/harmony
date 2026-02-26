{
  description = "Nervous System Management";

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

	nervous-deps = [
          requests
          flask
          gunicorn
          opencv4
          imutils
          matplotlib
          uvicorn
          graphviz
          aiohttp 
        ];

	nervous = buildPythonPackage {
          pname = "nervous";
          version = "0.0.1";
          src = ./.;
          pyproject = true;
          build-system = [ setuptools setuptools-scm ];
          propagatedBuildInputs = nervous-deps;
        };

    	nervous-dev-env = python.withPackages (ps: with ps; nervous-deps ++ [ pytest pytest-cov ]);
      in {
        packages.default = nervous;
        packages.nervous = nervous;
        packages.observer = writeShellApplication {
          name = "observer";
          runtimeInputs = [ self'.packages.nervous ];
          text = ''
            observer "$@"
          '';
        };
        packages.nervous-ngrok = writeShellApplication {
          name = "nervous-ngrok";
          runtimeInputs = [ self'.packages.nervous ngrok tmux ];
          text = ''
            # Starts a tmux session with 2 panes:
            #   1. Python server (port 9101)
            #   2. ngrok tunnel for Nervous Observer (9101)
            
            SESSION_NAME="nervous"
            PORT="''${PORT:-9101}"
            
            # Arguments for URLs
            NERVOUS_URL="''${1:-leyline.ngrok.app}"
            
            # If the session already exists, just attach
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
              echo "Attaching to existing tmux session '$SESSION_NAME'..."
              exec tmux attach -t "$SESSION_NAME"
            fi
            
            echo "Creating tmux session '$SESSION_NAME'..."
            
            # Create new detached session with the server
            tmux new-session -d -s "$SESSION_NAME" -n "nervous-stack" \
              "NERVOUS=$PORT PYTHONUNBUFFERED=1 observer; read"
            
            # Split for nervous Ngrok (horizontal split)
            tmux split-window -t "$SESSION_NAME:0" -h \
              "echo 'Exposing nervous on $NERVOUS_URL -> $PORT'; ngrok http $PORT --domain=$NERVOUS_URL; read"
            
            # Reorganize tiles
            tmux select-layout -t "$SESSION_NAME:0" tiled
            
            # Attach
            tmux attach -t "$SESSION_NAME"
          '';
        };
        packages.tests = writeShellApplication {
          name = "run-tests";
          runtimeInputs = [ nervous-dev-env self'.packages.nervous ];
          text = ''
            pytest tests/ "$@"
          '';
        };
        devShells.default = pkgs.mkShell {
          name = "nervousShell";
          packages = [ nervous-dev-env ];
          shellHook = ''
            PROJECT_NAME="$(basename "$PWD")"
            export PYTHONPATH="$PWD/nervous:$PWD/observer:${PYTHONPATH:-}"
            export PS1="\[\e[1;36m\][$PROJECT_NAME]\[\e[0m\] \w \$ "
          '';
        };
      };
      flake = {};
    };
}

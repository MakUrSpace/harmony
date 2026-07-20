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
            pkgs.openssl
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
        
        test-env = pkgs.python3.withPackages (ps: with ps; [ pytest pytest-playwright requests numpy ]);
      in {
        packages.default = harmony-rs;
        packages.harmony-rs = harmony-rs;
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
            # Starts a tmux session with 5 panes:
            #   1. Rust server (port 8081/8080/8082/8083)
            #   2. ngrok tunnel for Harmony (8081)
            #   3. ngrok tunnel for Admin (8080)
            #   4. ngrok tunnel for Discord (8082)
            #   5. ngrok tunnel for VR (8083)
            
            SESSION_NAME="harmony"
            PORT="''${PORT:-8081}"
            ADMIN_PORT="''${ADMIN_PORT:-8080}"
            DISCORD_PORT="''${DISCORD_PORT:-8082}"
            VR_PORT="''${VR_PORT:-8083}"
            
            # Arguments for URLs
            ADMIN_URL="''${1:-harmony-admin.ngrok.app}"
            HARMONY_URL="''${2:-harmony.ngrok.app}"
            DISCORD_URL="''${3:-harmony-discord.ngrok.app}"
            VR_URL="''${4:-harmony-vr.ngrok.app}"
            
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
            
            # Split for Discord Ngrok
            tmux split-window -t "$SESSION_NAME:0.2" -v \
              "echo 'Exposing Discord Activity on $DISCORD_URL -> $DISCORD_PORT'; ngrok http $DISCORD_PORT --domain=$DISCORD_URL; read"

            # Split for VR Ngrok
            tmux split-window -t "$SESSION_NAME:0.3" -v \
              "echo 'Exposing VR on $VR_URL -> $VR_PORT'; ngrok http $VR_PORT --domain=$VR_URL; read"
            
            # Reorganize tiles
            tmux select-layout -t "$SESSION_NAME:0" tiled
            
            # Attach
            tmux attach -t "$SESSION_NAME"
          '';
        };

        packages.tests = writeShellApplication {
          name = "run-tests";
          runtimeInputs = [ test-env pkgs.nodejs_20 pkgs.nodePackages.npm pkgs.playwright-driver pkgs.cargo pkgs.rustc pkgs.pkg-config pkgs.openssl pkgs.opencv4 pkgs.clang pkgs.rustPlatform.bindgenHook ];
          text = ''
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

            # Ensure node_modules are present for vitest
            if [ ! -d "node_modules" ]; then
              echo "node_modules not found, running npm install..."
              npm install
            fi

            EXIT_CODE=0
            echo "--- Running Rust tests (cargo test) ---"
            if ! nix develop . -c cargo test --workspace; then
              EXIT_CODE=1
            fi

            echo "--- Running UI Python E2E tests (pytest) ---"
            if ! pytest tests/test_ui_playwright.py; then
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
          runtimeInputs = [ pkgs.treefmt ];
          text = ''
            exec treefmt "$@"
          '';
        };
        devShells.default = pkgs.mkShell {
          name = "harmonyShell";
          packages = [ pkgs.nodejs_20 pkgs.nodePackages.npm pkgs.playwright-driver pkgs.treefmt pkgs.rustc pkgs.cargo pkgs.pkg-config pkgs.openssl pkgs.opencv4 pkgs.clang pkgs.rustPlatform.bindgenHook ];
          shellHook = ''
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

            PROJECT_NAME="$(basename "$PWD")"
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
            export PS1="\[\e[1;36m\][$PROJECT_NAME]\[\e[0m\] \w \$ "
          '';
        };
      };
      flake = {};
    };
}

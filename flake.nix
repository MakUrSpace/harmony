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

    	harmony-dev-env = python.withPackages (ps: with ps; harmony-deps);
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
            export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
  
            echo "Starting Jupyter Lab in $PWD"
            echo "PYTHONPATH=$PYTHONPATH"
  
            exec jupyter lab --notebook-dir="." --ip=0.0.0.0 --no-browser
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

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
        importnb = pkgs.python3Packages.buildPythonPackage rec {
          pname = "importnb";
          version = "0.5.3";
          pyproject = true;
          build-system = [ pkgs.python3Packages.setuptools ];
          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-IWQ7Kf3hrbDF2Q1yqxlOdDdUhwZM2jgmO4Tyoryeomo="; 
          };
        };
	harmony = (
          with pkgs;
          with python3Packages;
          buildPythonPackage {
            pname = "harmony";
            version = "0.0.1";
            src = ./harmony;
            pyproject = true;
            build-system = [ setuptools ];
            buildInputs = [ pytest ];
            propagatedBuildInputs = [
              requests
              flask
              gunicorn
              importnb
              opencv4Full
              imutils
              matplotlib
              uvicorn
              graphviz
              importnb
              pyzbar
              aiohttp 
            ];
          }
        );
      in {
        packages.default = harmony;
        packages.harmony = harmony;
      };
      flake = {};
    };
}

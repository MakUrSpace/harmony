# Harmony
Harmony integrates tabletop gaming utilities into a game recording and referreeing system.

## Setup

### Dependencies -- [Nix](https://nixos.org/download/)

Harmony's primary dependency is nix which manages packaging and running Harmony. To use Harmony, install Nix and enable both flakes and nix-commands.

1. Install Nix: `sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon`
1. Add to `/etc/nix/nix.conf`: `experimental-features = flakes nix-command`

### Execute Harmony

`nix run --impure "github:makurspace/harmony#harmony" --refresh`

Harmony will be available at [http://localhost:7000](http://localhost:7000).

#### Configure Cameras

To configure Harmony, navigate to [http://localhost:7000/configurator](http://localhost:7000/configurator) (accessible from the `Configurator` button from the Harmony homepage).

##### Add Camera

Provide:
* Name -- a simple name for the camera
* RTSP -- enable the RTSP camera streaming module
* Rotation -- degrees of rotation to apply to camera stream
* Address -- IP address for camera, like `192.168.2.185`. An RTSP address might look like `192.168.2.185:554/11`
* Auth -- authentication for the IP camera, provided as a comma separated list like `{username},{password}` 

###### SV3C RTSP Stream

If using a SV3C camera, you can use the RTSP camera stream module by clicking the `RTSP` checkbox and providing the RTSP address, which is typically `{Camera IP}:554/11`

##### Grid Configuration

Harmony projects a hexagonal grid across its perspectives. The grid cell size and the starting coordinate of the grid can be adjusted in this section.

##### Calibrator

Harmony's Calibrator is a utility for mapping cameras. It's accessed by clicking the `Calibrate Cameras` button on the Configurator page or by navigating to [http://localhost:7000/configurator/calibrator](http://localhost:7000/configurator/calibrator). Use this page to confirm the camera stream, and then click `First (Reset)` in the `Capture Control` section. 

1. Confirm camera stream
1. Select `First (Reset)` button in `Capture Control` section
1. Place calibration triangle (60mmX80mmX100mm triangle) onto the camera
1. Confirm Harmony recognized the triangle
1. Click `Commit Calibration`
1. Navigate to `Configurator`
1. Click `Save State` button
  1. This will create an `observerConfiguration.json` in Harmony's working directory

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


# Harmony
Harmony integrates tabletop gaming utilities into a game recording and referreeing system.

## Setup

1. ```Flash Raspbian```
1. ```sudo apt update && sudo apt upgrade ```
1. ```sudo apt install python3 python3-pip python3-dev```
1. ```sudo apt-get install graphviz libgraphviz-dev pkg-config```
1. ```python3 -m venv venv```
1. ```source vennv/bin/activate```
1. ```pip install -r requirements.txt```

### NeoPixel Strip

1. Connect NeoPixel to {NeoPixelPins}
1. Link service definition file: `ln -s ./observer/observerEyes.service /etc/systemd/system/observerEyes.service`
1. Reload systemd daemon: `sudo systemctl daemon-reload`
1. Enable observerEyes NeoPixel service: `sudo systemctl enable observerEyes`
1. Start observerEyes: `sudo systemctl start observerEyes`

### Camera Setup

Any USB cameras need to be setup as IP cameras for Harmony to access them. Harmony suggests [ayufan's camera-streamer](https://github.com/ayufan/camera-streamer) and offers systemd service files for it.

#### Install camera-streamer
Instructions for debian package install from [https://github.com/ayufan/camera-streamer/releases/tag/v0.2.6](https://github.com/ayufan/camera-streamer/releases/tag/v0.2.6)
```
https://github.com/ayufan/camera-streamer/releases/tag/v0.2.6if [[ -e /etc/default/raspberrypi-kernel ]]; then
  PACKAGE=camera-streamer-raspi_0.2.6.bullseye_$(dpkg --print-architecture).deb
else
  PACKAGE=camera-streamer-generic_0.2.6.bullseye_$(dpkg --print-architecture).deb
fi
wget "https://github.com/ayufan/camera-streamer/releases/download/v0.2.6/$PACKAGE"
sudo apt install "$PWD/$PACKAGE](https://github.com/ayufan/camera-streamer/releases/tag/v0.2.6)
```
https://github.com/ayufan/camera-streamer/releases/tag/v0.2.6if [[ -e /etc/default/raspberrypi-kernel ]]; then
  PACKAGE=camera-streamer-raspi_0.2.6.bullseye_$(dpkg --print-architecture).deb
else
  PACKAGE=camera-streamer-generic_0.2.6.bullseye_$(dpkg --print-architecture).deb
fi
wget "https://github.com/ayufan/camera-streamer/releases/download/v0.2.6/$PACKAGE"
sudo apt install "$PWD/$PACKAGE"
```

#### Configure Cameras

With `camera-streamer` installed, now each camera needs its service file installed and enabled. For each camera:
1. Link camera serivce file: `ln -s ./observer/dmaCameraServices/cam{camNum}-camera-streamer.service /etc/systemd/system/`
1. Reload systemctl daemon: `sudo systemctl daemon-reload`
1. Enable camera service: `sudo systemctl enable cam{camNum}-camera-streamer.service`
1. Start camera service: `sudo systemctl start cam{camNum}-camera-streamer.service`

Harmony is built on:
* [CogDB](https://cogdb.io/intro/)
* [OpenCV](https://opencv.org/)


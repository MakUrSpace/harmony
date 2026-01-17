### Camera Setup

Any USB cameras need to be setup as IP cameras for Harmony to access them. Harmony suggests [ayufan's camera-streamer](https://github.com/ayufan/camera-streamer) and offers systemd service files for it.

#### Install camera-streamer
Instructions for debian package install from [https://github.com/ayufan/camera-streamer/releases/tag/v0.2.6](https://github.com/ayufan/camera-streamer/releases/tag/v0.2.8)
```
PACKAGE=camera-streamer-$(test -e /etc/default/raspberrypi-kernel && echo raspi || echo generic)_0.2.8.$(. /etc/os-release; echo $VERSION_CODENAME)_$(dpkg --print-architecture).deb
wget "https://github.com/ayufan/camera-streamer/releases/download/v0.2.8/$PACKAGE"
sudo apt install "$PWD/$PACKAGE"
```

#### Configure Cameras

With `camera-streamer` installed, now each camera needs its service file installed and enabled. For each camera:
1. Install camera serivce file: `cp ./observer/dmaCameraServices/cam{camNum}-camera-streamer.service /etc/systemd/system/`
1. Reload systemctl daemon: `sudo systemctl daemon-reload`
1. Enable camera service: `sudo systemctl enable cam{camNum}-camera-streamer.service`
1. Start camera service: `sudo systemctl start cam{camNum}-camera-streamer.service`


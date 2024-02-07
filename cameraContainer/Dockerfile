FROM python:3.11

WORKDIR /opt

# Download camera-streamer
#run apt install libavcodec58 libavformat58 libavutil56 libssl1.1
run apt update && apt upgrade
env PACKAGE=camera-streamer-generic_0.2.8.bookworm_arm64.deb
run wget "https://github.com/ayufan/camera-streamer/releases/download/v0.2.8/$PACKAGE"
run apt install -y "$PWD/$PACKAGE"

CMD ["camera-streamer", "--camera-path=/dev/video0", "--camera-format=JPEG", "--camera-width=1920", "--camera-height=1080", "--camera-fps=30", "--camera-nbufs=2", "--camera-snapshot.height=1080", "--camera-video.height=720", "--http-listen=0.0.0.0"]

import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta
import base64

from flask import Flask, render_template, Response, request


CAMERAS = []
app = Flask(__name__)
PORT = int(os.getenv("OBS_IMPORT_PORT", "7000"))


def identify_cameras():
    cameraOutputs = os.listdir("output")
    cameras = [fn.split(".")[0][3:] for fn in cameraOutputs]
    return cameras


@app.route('/observer/<observer_name>', methods=['POST'])
def receive_images(observer_name):
    imported_images = request.get_json()
    deltas = imported_images.pop("DELTAS")
    for cam, frames in imported_images.items():
        os.makedirs(f"imported/{observer_name}", exist_ok=True)
        for idx, image in enumerate(frames):
            imageBin = base64.b64decode(image)
            with open(f"imported/{observer_name}/{observer_name}_cam{cam}_frame{idx}_{datetime.utcnow().strftime('%Y-%m-%dT%H_%M')}.jpg", "wb") as f:
               f.write(imageBin)
    for cam, deltaImages in deltas.items():
        for idx, image in enumerate(deltaImages):
            imageBin = base64.b64decode(image)
            with open(f"imported/{observer_name}/deltas_cam{cam}_{datetime.utcnow().strftime('%Y-%m-%dT%H_%M')}.jpg", "wb") as f:
               f.write(imageBin)

    return "Success!!"


@app.route('/camera_list')
def camera_list():
    return json.dumps(CAMERAS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=PORT)


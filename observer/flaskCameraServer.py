import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta
import base64

from flask import Flask, render_template, Response


CAMERAS = []
app = Flask(__name__)


def identify_cameras():
    cameraOutputs = os.listdir("output")
    cameras = [fn.split(".")[0][3:] for fn in cameraOutputs]
    return cameras


def gen_frames(camera_idx):
    while True:
        with open(f"output/cam{camera_idx}.jpg", "rb") as f:
            jpg = f.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


@app.route('/video_feed/<camera_idx>')
def video_feed(camera_idx):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(int(camera_idx)), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_list')
def camera_list():
    return Response(json.dumps(CAMERAS).encode())


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', cameras=CAMERAS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    CAMERAS = identify_cameras()
    print(f"Supporting Cameras: {CAMERAS}")
    app.run(debug=True, host="0.0.0.0", use_reloader=False)


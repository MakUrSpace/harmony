import os
import cv2
import numpy as np
import threading
import time

from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for
from traceback import format_exc

from nervous.camera import CaptureConfiguration

app = None

PORT = int(os.getenv("NERVOUS", "9003"))


nervous = Blueprint('observer', __name__, template_folder='templates')


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

import json
from flask import current_app
from nervous.camera import RTSPCamera

@nervous.route('/', methods=['GET'])
def nervousDashboard():
    with open(f"{os.path.dirname(__file__)}/templates/Nervous.html") as f:
        page = f.read()
    return page

@nervous.route('/config', methods=['GET'])
def getConfig():
    return json.dumps(current_app.cc.buildConfiguration())


class CameraStream:
    def __init__(self, cam):
        self.cam = cam
        self.thread = None
        self.clients = 0
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition(self.lock)
        self.latest_frame_bytes = None
        self.running = False

    def start(self):
        with self.lock:
            self.clients += 1
            if self.thread is None:
                self.running = True
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()

    def stop(self):
        with self.lock:
            self.clients -= 1
            if self.clients <= 0:
                self.clients = 0
                self.running = False
                self.thread = None

    def _loop(self):
        while self.running:
            try:
                frame = self.cam.collectImage()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        with self.lock:
                            self.latest_frame_bytes = buffer.tobytes()
                            self.frame_cond.notify_all()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Stream error in background thread for {self.cam.camName}: {e}")
                time.sleep(0.1)

stream_managers = {}

def get_stream_manager(cam_name, cam):
    if cam_name not in stream_managers:
        stream_managers[cam_name] = CameraStream(cam)
    return stream_managers[cam_name]

def generateFrames(cam_name, cam):
    manager = get_stream_manager(cam_name, cam)
    manager.start()
    try:
        last_frame = None
        while True:
            with manager.lock:
                if manager.latest_frame_bytes == last_frame:
                    manager.frame_cond.wait(timeout=1.0)
                frame_bytes = manager.latest_frame_bytes
                
            if frame_bytes is not None:
                last_frame = frame_bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    finally:
        manager.stop()

@nervous.route('/stream/<cam_name>')
def streamCamera(cam_name):
    print(f"Request stream for: {cam_name}")
    print(f"Cameras in CC: {list(current_app.cc.cameras.keys())}")
    cam = current_app.cc.cameras.get(cam_name)
    if not cam:
        cam = current_app.cc.cameras.get(cam_name.replace("RTSPCamera", ""))
        
    if not cam:
        return "Not found", 404
        
    return Response(generateFrames(cam_name, cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@nervous.route('/static/<path:filename>')
def serve_static(filename):
    # Serve from the root nervous-system directory to resolve install images
    root_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(root_dir, f"observer/{filename}")
    print(f"Serving file: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
        # simple mime type mapping
        ext = filename.split('.')[-1].lower()
        mimetype = f"image/{ext}" if ext in ('jpg', 'jpeg', 'png') else "application/octet-stream"
        return Response(content, mimetype=mimetype)
    return "Not found", 404

def main():
    app = Flask(__name__)
    
    @app.before_request
    def require_auth():
        # Do not require auth for static files or preflight OPTIONS
        if request.method == 'OPTIONS' or request.endpoint in ['getBSCSS', 'getBSJS', 'getHTMX', 'static']:
            return
            
        auth = request.authorization
        # Check authentication against the stored password
        if not auth or getattr(auth, 'password', None) != app.cc.password:
            # Setting WWW-Authenticate prompts the browser for credentials
            return make_response(
                'Could not verify your access level for that URL.\n'
                'You have to login with proper credentials', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})
    app.cc = CaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(nervous, url_prefix='/observer')

    @app.route('/')
    def index():
        return redirect('/observer', code=303)

    @app.route('/style.css', methods=['GET'])
    def getBSCSS():
        with open(f"{os.path.dirname(__file__)}/templates/style.css", "r") as f:
            bscss = f.read()
        return Response(bscss, mimetype="text/css")
    
    @app.route('/bootstrap.min.js', methods=['GET'])
    def getBSJS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
            bsjs = f.read()
        return Response(bsjs, mimetype="application/javascript")
    
    @app.route('/htmx.min.js', methods=['GET'])
    def getHTMX():
        with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
            htmx = f.read()
        return Response(htmx, mimetype="application/javascript")
    
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, DELETE, PUT'
        response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
        return response

    print(f"Launching Nervous Server on {PORT}")
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np

from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for
from traceback import format_exc

from nervous.camera import CaptureConfiguration

app = None

PORT = int(os.getenv("NERVOUS", "9002"))


nervous = Blueprint('nervous', __name__, template_folder='templates')


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

@nervous.route('/camera', methods=['POST'])
def addCamera():
    try:
        data = request.form
        name = data.get('name')
        if not name:
            return "Name required", 400
            
        addr = data.get('addr')
        username = data.get('username', '')
        password = data.get('password', '')
        auth = [username, password] if username and password else None
        
        # Rotation could be handled here if needed, keeping it as string for now
        rot = data.get('rot', '')
        
        # Default active zone
        az = np.float32([[0,0], [1920,0], [1920,1080], [0,1080]])
        
        cam = RTSPCamera(address=addr, activeZone=az, camName=name, auth=auth)
        if hasattr(cam, 'rotate'):
            cam.rotate = (rot == 'true' or rot == 'True')
            
        current_app.cc.cameras[name] = cam
        current_app.cc.saveConfiguration()
        
        return "OK", 200
    except Exception as e:
        print(f"Error adding camera: {e}")
        return str(e), 500

@nervous.route('/camera/<cam_name>', methods=['DELETE'])
def removeCamera(cam_name):
    try:
        target_name = cam_name
        if target_name not in current_app.cc.cameras:
            target_name = cam_name.replace("RTSPCamera", "")
            
        if target_name in current_app.cc.cameras:
            del current_app.cc.cameras[target_name]
            current_app.cc.saveConfiguration()
            return "OK", 200
        return "Not found", 404
    except Exception as e:
        return str(e), 500

def generateFrames(cam):
    # Using our blocking collectImage method
    while True:
        try:
            frame = cam.collectImage()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Stream error for {cam.camName}: {e}")
            pass

@nervous.route('/stream/<cam_name>')
def streamCamera(cam_name):
    print(f"Request stream for: {cam_name}")
    print(f"Cameras in CC: {list(current_app.cc.cameras.keys())}")
    cam = current_app.cc.cameras.get(cam_name)
    if not cam:
        cam = current_app.cc.cameras.get(cam_name.replace("RTSPCamera", ""))
        
    if not cam:
        return "Not found", 404
        
    return Response(generateFrames(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    app.register_blueprint(nervous, url_prefix='/nervous')

    @app.route('/')
    def index():
        return redirect('/nervous', code=303)

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

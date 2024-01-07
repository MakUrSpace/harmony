import json
import cv2
import numpy as np
from flask import Blueprint, render_template, abort, request, Response

from ipynb.fs.full.Observer import CaptureConfiguration, RemoteCamera


configurator = Blueprint('config', __name__, template_folder='templates')


def buildConfigurator():
    cameraConfigRows = []
    clickSubs = []
    for cam in cc.cameras.values():
        if cam is None:
            continue
        activeZone = json.dumps(cam.activeZone.tolist())
        optionValues = f"""<option value="field" selected>Game Field</option><option value="dice">Dice</option>"""
        cameraConfigRows.append(f"""
            <div class="row justify-content-center text-center">
                <h3 class="mt-5">Camera {cam.camName} <input type="button" value="Delete" class="btn-error" hx-post="/config/delete_cam/{cam.camName}" hx-swap="outerHTML"></h3>
                <img src="/config/camera/{cam.camName}" title="{cam.camName} Capture" height="375" id="cam{cam.camName}" onclick="camClickListener('{cam.camName}', event)">
                <label for="az">Active Zone</label><br>
                <div class="container">
                    <div class="row">
                        <div class="col">
                            <input type="text" name="az" id="cam{cam.camName}_ActiveZone" value="{activeZone}" size="50" hx-post="/config/cam{cam.camName}_activezone" hx-swap="none">   
                            <input type="button" name="clearCam{cam.camName}AZ" value="Clear AZ" onclick="clearCamAZ('{cam.camName}', event)">
                        </div>
                        <div class="col">    
                            <label>Camera Type</label>
                            <select name="camType" hx-post="/config/cam{cam.camName}_type" hx-swap="none">
                              {optionValues}
                            </select>
                        </div>
                    </div>
                </div>
            </div>""")

    with open("templates/Configuration.html") as f:
        template = f.read()
    
    return template.replace(
        "{cameraConfigRows}", "\n".join(cameraConfigRows))

    
@configurator.route('/', methods=['GET'])
def config():
    return buildConfigurator()


@configurator.route('/', methods=['POST'])
def updateConfig():
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = "Saved Configuration"
    cc.saveConfiguration()
    return "success"


def genCameraFullViewWithActiveZone(camName):
    while True:
        try:
            cam = cc.cameras[str(camName)]
            img = cam.drawActiveZone(cam.mostRecentFrame)
            ret, img = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Failed genCameraFullViewWithActiveZone for {camName} -- {e}")
            yield (b'--frame\r\nContent-Type: image/jpg\r\n\r\n\r\n')
    
    
@configurator.route('/camera/<camName>')
def cameraActiveZoneWithObjects(camName):
    return Response(genCameraFullViewWithActiveZone(str(camName)), mimetype='multipart/x-mixed-replace; boundary=frame')


@configurator.route('/cam<camName>_activezone', methods=['POST'])
def updateCamActiveZone(camName):
    global CONSOLE_OUTPUT
    print(f"Received update Active Zone request for {camName}")
    try:
        az = np.float32(json.loads(request.form.get(f"az")))
        cam = cc.cameras[camName]
        cam.setActiveZone(az)
    except Exception as e:
        print(f"Unrecognized data: {camName} - {az} - {e}")
    CONSOLE_OUPUT = f"Updated {camName} AZ"
    return "success"


@configurator.route('/cam<camName>_type', methods=['POST'])
def updateCamType(camName):
    global CONSOLE_OUTPUT
    print(f"Received update Active Zone request for {camName}")
    try:
        camType = str(request.form.get(f"camType"))
        cam = cameras[camName]
        cam.camType = camType 
    except:
        print(f"Unrecognized data: {camName} - {camType}")
    CONSOLE_OUPUT = f"Updated {camName} type to {camType}"
    return "success"


@configurator.route('/new_camera', methods=['GET'])
def getNewCameraForm():
    with open("templates/NewCamera.html", "r") as f:
        template = f.read()
    return template


@configurator.route('/new_camera', methods=['POST'])
def addNewCamera():
    global CONSOLE_OUTPUT
    camName = request.form.get("camName")
    camRot = request.form.get("camRot")
    camAddr = request.form.get("camAddr")
    cc.cameras[camName] = RemoteCamera(address=camAddr, activeZone=[[0, 0], [0, 1], [1, 1,], [1, 0]], camName=camName, rotate=camRot)
    cc.rsc = None
    cc.saveConfiguration()
    cc.capture()
    CONSOLE_OUTPUT = f"Added Camera {camName}"
    return """<script>window.location = '/config';</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/config'">"""


@configurator.route('/delete_cam/<camName>', methods=['POST'])
def deleteCamera(camName):
    global CONSOLE_OUTPUT
    cc.cameras.pop(camName)
    cc.rsc = None
    cc.saveConfiguration()
    CONSOLE_OUTPUT = f"Deleted Camera {camName}"
    return """<script>window.location.reload();</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/config'">"""


if __name__ == "__main__":
    from flask import Flask
    cc = CaptureConfiguration()
    cc.capture()
    
    app = Flask(__name__)
    app.register_blueprint(configurator, url_prefix="/config")

    @app.route('/<page>')
    def getPage(page):
        try:
            with open(f"templates/{page}") as page:
                page = page.read()
        except Exception as e:
            print(f"Failed to find page: {e}")
            page = "Not found!"
        return page
    app.run(debug=True, host='0.0.0.0', port=7000)

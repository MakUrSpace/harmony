import os
import json
import cv2
import numpy as np
import math
from flask import Blueprint, render_template, abort, request, Response, url_for

from ipynb.fs.full.Observer import RemoteCamera, RTSPCamera
from ipynb.fs.full.CalibratedObserver import CalibratedCaptureConfiguration, CalibrationObserver
from ipynb.fs.full.HexObserver import HexGridConfiguration

# from calibrator import calibrator, registerCaptureService, setCalibratorApp # Deprecated



configurator = Blueprint('configurator', __name__, template_folder='templates')
# configurator.register_blueprint(calibrator, url_prefix='/calibrator') # Deprecated



def buildConfigurator():
    cameraConfigRows = []
    clickSubs = []
    # Inject camera names for JS
    cameraNames = list(app.cc.cameras.keys())
    cameraNamesJson = json.dumps(cameraNames)
    
    for cam in app.cc.cameras.values():
        if cam is None:
            continue
        activeZone = json.dumps(cam.activeZone.tolist())
        optionValues = f"""<option value="field" selected>Game Field</option><option value="dice">Dice</option>"""
        cameraConfigRows.append(f"""
            <div class="row justify-content-center text-center">
                <h3 class="mt-5">Camera {cam.camName} <input type="button" value="Delete" class="btn btn-danger" hx-post="{url_for('.config')}delete_cam/{cam.camName}" hx-swap="outerHTML"></h3>
                <div id="camContainer_{cam.camName}" style="position: relative; display: inline-block;">
                    <img src="{url_for('.config')}camera/{cam.camName}" title="{cam.camName} Capture" height="375" id="cam{cam.camName}" onclick="camClickListener('{cam.camName}', event)">
                    <canvas id="overlayCanvas_{cam.camName}" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair;"
                        onmousedown="onMouseDown('{cam.camName}', event)"
                        onmousemove="onMouseMove('{cam.camName}', event)"
                        ></canvas>
                </div>
                <div class="text-center mt-2">
                    <button class="btn btn-sm btn-warning" onclick="clearShape('{cam.camName}')">Clear Manual Points</button>
                    <!-- AZ controls below -->
                </div>
                <label for="az">Active Zone</label><br>
                <div class="container">
                    <div class="row">
                        <div class="col">
                            <input type="text" name="az" id="cam{cam.camName}_ActiveZone" value="{activeZone}" size="50" hx-post="{url_for('.config')}cam{cam.camName}_activezone" hx-swap="none">   
                            <input type="button" class="btn btn-secondary" name="clearCam{cam.camName}AZ" value="Clear AZ" onclick="clearCamAZ('{cam.camName}', event)">
                        </div>
                        <div class="col">    
                            <label>Camera Type</label>
                            <select name="camType" hx-post="{url_for('.config')}/cam{cam.camName}_type" hx-swap="none">
                              {optionValues}
                            </select>
                        </div>
                    </div>
                </div>
            </div>""")

    # Fix for Table Population on Load:
    # app.cm.calibrationPts isn't persisted, but app.cc.rsc is.
    # If table is empty but we have calibration, reconstruct it for display.
    if not app.cm.calibrationPts and hasattr(app.cc, 'rsc') and app.cc.rsc is not None:
        reconstructed = []
        try:
            # app.cc.rsc.realCamSpacePairs is [(camName, [camPts, realPts]), ...]
            for cN, (camPts, realPts) in app.cc.rsc.realCamSpacePairs:
                # helper to sanitize numpy arrays to lists
                def to_list_recursive(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, list):
                        return [to_list_recursive(x) for x in obj]
                    return obj

                rec_entry = {cN: [to_list_recursive(camPts), to_list_recursive(realPts)]}
                reconstructed.append(rec_entry)
            
            app.cm.calibrationPts = reconstructed
        except Exception as e:
            print(f"Failed to reconstruct calibration table: {e}")

    calibrationPtsJson = json.dumps(app.cm.calibrationPts)

    with open(f"{os.path.dirname(__file__)}/templates/Configurator.html") as f:
        template = f.read()
    
    return template.replace(
        "{configuratorURL}", url_for(".config")).replace(
        "{cameraConfigRows}", "\n".join(cameraConfigRows)).replace(
        "{size}", str(app.cc.hex.size)).replace(
        "__CAMERA_NAMES_JSON__", cameraNamesJson).replace(
        "__CALIBRATION_PTS_JSON__", calibrationPtsJson).replace(
        "{calibratorURL}", url_for(".config")) # Redirect calibratorURL to configurator base for now


    
@configurator.route('/', methods=['GET'])
def config():
    return buildConfigurator()


@configurator.route('/', methods=['POST'])
def updateConfig():
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = "Saved Configuration"
    app.cc.saveConfiguration()
    return "Success"


def draw_dynamic_grid(camName):
    try:
        if not hasattr(app.cc, 'rsc') or app.cc.rsc is None:
            return None
        if not hasattr(app.cc, 'hex') or app.cc.hex is None:
            return None
            
        cam = app.cc.cameras[str(camName)]
        h, w = cam.mostRecentFrame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        converter = app.cc.rsc.converters[str(camName)][0]
        
        # Define 4 corners of the camera frame
        corners_cam = [(0, 0), (w, 0), (w, h), (0, h)]
        
        # Project to Real Space
        corners_real = []
        for p in corners_cam:
            rp = converter.convertCameraToRealSpace(p)
            corners_real.append(rp)
            
        # Convert to Axial to find range
        qs = []
        rs = []
        for cr in corners_real:
            q, r = app.cc.pixel_to_axial(cr[0], cr[1])
            qs.append(q)
            rs.append(r)
            
        min_q, max_q = min(qs), max(qs)
        min_r, max_r = min(rs), max(rs)
        
        # Add padding to ensure coverage
        pad = 2
        
        # Iterate and draw
        # Use white color for grid lines, typical for overlay
        grid_color = (255, 255, 255)
        
        for q in range(min_q - pad, max_q + pad + 1):
            for r in range(min_r - pad, max_r + pad + 1):
                # Calculate Hex in Real Space
                # hex_at_axial returns shape (N, 1, 2) roughly, lets verify usage
                # It returns poly_i which is rounded int array. 
                # We need exact float coords preferably for smooth projection but hex_at_axial does round.
                # However, convertRealToCameraSpace expects tuple (x, y).
                
                real_poly = app.cc.hex_at_axial(q, r) # numpy array (6, 1, 2)
                
                cam_pts = []
                for pt in real_poly:
                    # pt is [x, y]
                    x, y = pt[0]
                    cx, cy = converter.convertRealToCameraSpace((x, y))
                    cam_pts.append([int(round(cx)), int(round(cy))])
                
                poly_cam = np.array(cam_pts, dtype=np.int32).reshape((-1, 1, 2))
                
                # Draw
                cv2.polylines(overlay, [poly_cam], True, grid_color, 1, cv2.LINE_AA)
                
        return overlay
        
    except Exception as e:
        print(f"Error drawing dynamic grid: {e}")
        # import traceback
        # traceback.print_exc()
        return None


def genCameraFullViewWithActiveZone(camName):
    while True:
        try:
            cam = app.cc.cameras[str(camName)]
            img = cam.drawActiveZone(cam.mostRecentFrame)
            




            # Overlay Hex Grid if configured and calibration exists
            if hasattr(app.cc, 'rsc') and app.cc.rsc is not None:
                try:
                    # grid_overlay = app.cc.cameraGriddle(str(camName))
                    grid_overlay = draw_dynamic_grid(camName)
                    
                    if grid_overlay is not None:
                        if grid_overlay.shape[:2] == img.shape[:2]:
                            cv2.addWeighted(grid_overlay, 0.5, img, 1.0, 0.0, dst=img)
                except Exception as e:
                    print(f"Grid overlay error: {e}")
                    # import traceback
                    # traceback.print_exc()
            
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
        cam = app.cc.cameras[camName]
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
    with open(f"{os.path.dirname(__file__)}/templates/NewCamera.html", "r") as f:
        template = f.read()
    return template.replace("{configuratorURL}", url_for(".config"))


@configurator.route('/new_camera', methods=['POST'])
def addNewCamera():
    global CONSOLE_OUTPUT
    camName = request.form.get("camName")
    camRot = request.form.get("camRot")
    rtspCam = request.form.get("rtspCam")
    camAddr = request.form.get("camAddr")
    try:
        camAuth = [part.strip() for part in request.form.get("camAuth").split(",")]
        if len(camAuth) != 2:
            raise Exception(f"Unrecognized auth format: {request.form.get('camAuth')}")
    except:
        camAuth = None

    builder = RTSPCamera if rtspCam else RemoteCamera
    app.cc.cameras[camName] = builder(address=camAddr, activeZone=[[0, 0], [0, 1], [1, 1,], [1, 0]], camName=camName, rotate=camRot, auth=camAuth)
    app.cc.rsc = None
    app.cc.saveConfiguration()
    app.cc.capture()
    CONSOLE_OUTPUT = f"Added Camera {camName}"
    return f"""<script>window.location = '{url_for('.config')}';</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='{url_for('.config')}'">"""


@configurator.route('/manual_calibration', methods=['POST'])
def manualCalibration():
    # Expects a dict of {camName: points}
    data = request.json
    print(f"Received manual calibration data: {data}")
    if not data:
        return "No data received", 400
        
    added_count = 0
    # We will accumulate all points into a single "calibration point" entry
    # which is a dict of {camName: [pixelPoints, realPoints]}
    
    # Structure of calibrationPts entry: {camName: [pixelPoints, realPoints], camName2: ...}
    new_calib_entry = {}
    
    for camName, points in data.items():
        if len(points) != 3:
            continue
            
        if camName not in app.cc.cameras:
            continue

        # Get camera resolution
        cam = app.cc.cameras[camName]
        try:
            h, w = cam.mostRecentFrame.shape[:2]
        except AttributeError:
             print(f"Camera {camName} has no frame.")
             continue
        
        # Convert normalized points to pixel coordinates
        pixelPoints = [[int(p[0] * w), int(p[1] * h)] for p in points]
        
        # Add to entry
        # Assumes "first" triangle position as the target for manual calibration.
        if app.cm.first_triangle:
             new_calib_entry[camName] = [pixelPoints, app.cm.first_triangle]
             added_count += 1
        else:
             print("app.cm.first_triangle is not defined")
        
    if added_count > 0:
        # Overwrite previous calibration (User Request)
        app.cm.calibrationPts = []
        
        # Add to calibration points list
        app.cm.calibrationPts.append(new_calib_entry)
        app.cm.buildRealSpaceConverter()
        app.cc.saveConfiguration()
        return f"Added manual calibration for {added_count} cameras", 200
    else:
        return "No valid calibration points found", 400



@configurator.route('/delete_cam/<camName>', methods=['POST'])
def deleteCamera(camName):
    global CONSOLE_OUTPUT
    app.cc.cameras.pop(camName)
    app.cc.rsc = None
    app.cc.saveConfiguration()
    CONSOLE_OUTPUT = f"Deleted Camera {camName}"
    return f"""<script>window.location.reload();</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='{url_for('.config')}'">"""


@configurator.route('/grid_configuration', methods=['POST'])
def configureGrid():
    app.cc.hex = HexGridConfiguration(
        size=float(request.form.get("size"))
    )
    return f"""
        <input type="number" class="form-check-input bg-info" name="size" min="10" max="60" value="{app.cc.hex.size}" style="width:4em">
        <label class="form-check-label" for="size">Size</label><br>
        <input type="submit" class="btn btn-primary bg-secondary" value="Submit Grid Configuration">"""


def setConfiguratorApp(newApp):
    # setCalibratorApp(newApp) # Deprecated

    global app
    app = newApp


if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.cc = CalibratedCaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(configurator, url_prefix='/configurator')
    app.cm = CalibrationObserver(app.cc)
    setCalibratorApp(app)

    @app.route('/<page>')
    def getPage(page):
        try:
            with open(f"{os.path.dirname(__file__)}/templates/{page}") as page:
                page = page.read()
        except Exception as e:
            print(f"Failed to find page: {e}")
            page = "Not found!"
        return page

    registerCaptureService(app)
    print(f"Launching Observer Server on Port 7000")
    app.run(host="0.0.0.0", port=7000)

import os
import cv2
from math import ceil
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import argparse
import json
from io import BytesIO
from ipynb.fs.full.Observer import cc, cameras, pov, Camera, CalibrationBox
from flask import Flask, render_template, Response, request, make_response
from traceback import format_exc
from ultralytics import YOLO


observerApp = Flask(__name__)
PORT = int(os.getenv("OBSERVER_PORT", "7000"))

CONSOLE_OUTPUT = "No Output Yet"


humanInferenceModel = YOLO("yolov8n.pt")


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


@observerApp.route('/bootstrap.min.css', methods=['GET'])
def getBSCSS():
    with open("templates/bootstrap.min.css", "r") as f:
        bscss = f.read()
    return Response(bscss, mimetype="text/css")


@observerApp.route('/bootstrap.min.js', methods=['GET'])
def getBSJS():
    with open("templates/bootstrap.min.js", "r") as f:
        bsjs = f.read()
    return Response(bsjs
                    , mimetype="application/javascript")


@observerApp.route('/trackedObjects', methods=['GET'])
def trackedObjects():
    return json.dumps([{obj.name: obj.changeBoxes} for obj in cc.captureObjects])


@observerApp.route('/mostRecentFrames', methods=['GET'])
def mostRecentFrames():
    return json.dumps({camNum: imageToBase64(cam.mostRecentFrame) for camNum, cam in cameras.items()})


@observerApp.route('/mostRecentFrame/<camNum>', methods=['GET'])
def camMostRecentFrame(camNum):
    camNum = int(camNum)
    return imageToBase64(cameras[camNum].mostRecentFrame)


@observerApp.route('/capture', methods=['GET'])
def capture():
    captures = cc.capture()
    return json.dumps({
        camNum: {
            "capture": base64.b64encode(cv2.imencode('.jpg', camCap[0])[1]).decode(),
            "changeBoxes": list(camCap[1])
        } for camNum, camCap in captures.items()
    })


@observerApp.route('/calibrate', methods=['GET'])
def calibrate():
    return "Calibrated!"


@observerApp.route('/add_object', methods=['GET'])
def add_object():
    return "Object located"


@observerApp.route('/remove_object', methods=['GET'])
def remove_object():
    return "Object removed"


@observerApp.route('/move/select_object', methods=['GET'])
def select_object():
    return "Object selected"


@observerApp.route('/move/place_object', methods=['GET'])
def move_object():
    "Object placed"


def genCameraFullViewWithActiveZone(camNum):
    while True:
        cam = cameras[int(camNum)]
        cam.capture()
        camImage = cam.drawActiveZone(cc.drawObjectsOnCam(cam))
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.imshow(camImage)
        camFrame = BytesIO()
        FigureCanvas(fig).print_png(camFrame)
        camFrame.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camFrame.read() + b'\r\n')
    
    
@observerApp.route('/config/camera/<camNum>')
def cameraActiveZoneWithObjects(camNum):
    return Response(genCameraFullViewWithActiveZone(int(camNum)), mimetype='multipart/x-mixed-replace; boundary=frame')
    return response


def genCameraInference(camNum):
    cam = cameras[int(camNum)]
    cam.capture()
    baseIm = cam.maskFrameToActiveZone(cam.imageBuffer[0])
    while True:
        cam.capture()
        contouredImages = None
        for idx in range(0, len(cam.imageBuffer[:-1]), 3):
            deltaIm = cam.imageBuffer[idx] 
            deltaIm = cam.maskFrameToActiveZone(deltaIm)
            contours = cam.contoursBetween(baseIm, deltaIm)

            filteredContours = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if 1000000 > area > 1000:
                    filteredContours.append(c)

            contours = [c for c in contours if (x, y, w, h := cv2.boundingRect(c))[-1] * w]
            baseIm = cam.referenceFrame
            contoured = cv2.drawContours(cam.maskFrameToActiveZone(baseIm.copy()), filteredContours, -1, (0,int((255 / 10) * idx),255), 3)
            if cam.interactionDetection(deltaIm):
                contoured = cv2.line(contoured, (0, 0), (1800, 1080), (0, 0, 255), 5)
                contoured = cv2.line(contoured, (0, 1080), (1800, 0), (0, 0, 255), 5)

            if contouredImages is None:
                contouredImages = contoured
            else:
                contouredImages = np.concatenate((contouredImages, contoured), axis=1)
        visImage = np.concatenate((contouredImages, baseIm), axis=1)
        
        # TODO: Filter for persistent contours
        # TODO: Create persistent contoured image
        # TODO: Combine statck image with large persistent, contoured image
        
        ret, visImage = cv2.imencode('.jpg', visImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + visImage.tobytes() + b'\r\n')
    
    
@observerApp.route('/debug/camera/<camNum>')
def cameraInference(camNum):
    return Response(genCameraInference(int(camNum)), mimetype='multipart/x-mixed-replace; boundary=frame')
    return response


def genCameraActiveZoneWithObjectsAndDeltas(camNum):
    while True:
        cam = cameras[camNum]
        cam.capture()
        camImage = cam.maskFrameToActiveZone(cam.drawActiveZone(cc.drawDeltasOnCam(cam, cc.drawObjectsOnCam(cam))))
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/camera/<camNum>')
def cameraActiveZoneWithObjectsAndDeltas(camNum):
    return Response(genCameraActiveZoneWithObjectsAndDeltas(int(camNum)), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
def buildConfigurator():
    cameraConfigRows = []
    for cam in cameras.values():
        activeZone = json.dumps(cam.activeZone.tolist())
        currentCalibration = f"Calibrated to: {cam.MCalibratedTo}" if cam.M is not None else "Not Calibrated"
        cameraConfigRows.append(f"""
            <div class="row">
                <div class="col-lg-8  offset-lg-2">
                    <h3 class="mt-5">Camera {cam.camNum}</h3>
                    <div>
                        <iframe src="/config/camera/{cam.camNum}" title="{cam.camNum} Capture" width="100%" height="500"></iframe>
                    </div>
                    <form method="post">
                      <label for="min_width">Minimum Width of Tracked Objects</label><br>
                      <input type="number" id="min_width" name="min_width" value="{cam.minimumWidth}"><br>
                      <label for="az">Active Zone</label><br>
                      <input type="text" id="az" name="az" value="{activeZone}" size="50"><br>
                      <input type="hidden" id="camNum" name="camNum" value="{cam.camNum}">
                      <input type="submit" value="Update Cam {cam.camNum}">
                      <input type="hidden" id="configType" name="configType" value="activeZone">
                    </form>
                    <hr>
                    {currentCalibration}
                    <form method="post"
                        <label>Calibration Coordinates:&nbsp;&nbsp;(</label>
                        <label for="calibrate_x">X:&nbsp;&nbsp;</label>
                        <input type="number" id="calibrate_x" name="calibrate_x" placeholder="0.0" min="0" max="10000">
                        <label for="calibrate_y">, Y:&nbsp;&nbsp;</label>
                        <input type="number" id="calibrate_y" name="calibrate_y" placeholder="0.0" min="0" max="10000">&nbsp;&nbsp;)</p>
                        <input type="hidden" id="camNum" name="camNum" value="{cam.camNum}">
                        <input type="hidden" id="configType" name="configType" value="calibrate">
                        <input type="submit" value="Calibrate Cam {cam.camNum}">
                    </form>
                </div>
            </div>""")
    with open("templates/Configuration.html") as f:
        template = f.read()
    return template.replace("{cameraConfigRows}", "\n".join(cameraConfigRows))
    
    
@observerApp.route('/config', methods=['GET'])
def config():
    cc.capture()
    return buildConfigurator()
    
    
@observerApp.route('/config', methods=['POST'])
def updateConfig():
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = ""
    print(f"Received update config request")
    
    if (config_type := request.form.get("config_type")) == 'save_state':
        cc.saveState()
        CONSOLE_OUTPUT = "State Saved!"
        print("State Saved!")
    elif config_type == 'capture_cameras':
        cc.capture()
        CONSOLE_OUTPUT = "Cameras Captured"
    elif config_type == 'load_state':
        cc.recoverState()
        CONSOLE_OUTPUT = "State Recovered!"
        print("State Recovered!")
    else:
        camNum = int(request.form.get("camNum"))
        if (configType := request.form.get("configType")) == "activeZone":
            az = np.float32(json.loads(request.form.get('az')))
            min_width = float(request.form.get("min_width"))

            cam = cameras[camNum]
            cam.setActiveZone(az)
            cam.minimumWidth = min_width
        elif configType == "calibrate":
            xPosition = float(request.form.get('calibrate_x'))
            yPosition = float(request.form.get('calibrate_y'))
            try:
                qrCodes = cameras[camNum].calibrate((xPosition, yPosition))
                CONSOLE_OUTPUT = f"Camera {camNum} Calibrated to ({xPosition}, {yPosition}). Saw {qrCodes} calibration boxes"
            except Exception as e:
                print(f"Failed Calibration: {e}")
                CONSOLE_OUTPUT = f"Failed Calibration: {e}"

    print(f"Rebuilding configurator")
    return buildConfigurator()



@observerApp.route('/captureObjects', methods=['GET'])
def captureObjectsTable():
    capObjTableRows = "\n".join([f"""
        <tr>
            <td>{capObj.name}</td>
            <td>{capObj.realCoords}</td>
            <td>{capObj.camBoxes}</td>
        </tr>
    """ for capObj in cc.captureObjects.values()])
    
    captureObjTable = f"""
<html lang="en">
<head>
    <style>
        table, th, td {{
          border: 1px solid black;
          padding: 15px;
        }}
    </style>
</head>
<body>
    <table style="width:100%">
        <thead>
            <tr>
                <th>Object Name</th>
                <th>Real Coordinates</th>
                <th>Object Camera Boxes</th>
            </tr>
        </thead>
        <tbody>
            {capObjTableRows}
        </tbody>
    </table>
</body>
</html>
"""
    
    return captureObjTable
    

SELECTED = None
LASTCHANGE = None


@observerApp.route('/control', methods=['GET', 'POST'])
def controller():
    global CONSOLE_OUTPUT, LASTCHANGE, SELECTED
    CONSOLE_OUTPUT = ""
    captureType = request.form.get("capture_type")
    if (captureType) is not None:
        if captureType == 'commit':
            LASTCHANGE[0](*LASTCHANGE[1:])
            CONSOLE_OUTPUT = "Last Change committed"
            LASTCHANGE = None
        elif captureType == "reject":
            LASTCHANGE = None
            for cam in cameras.values():
                cam.imageBuffer[0] = cam.imageBuffer[1]
            CONSOLE_OUTPUT = "Last capture rejected"
        elif captureType == 'Null':
            LASTCHANGE = None
            SELECTED = None
            captures = cc.capture()
            cc.setReference()
            changeBoxes = {cam: camCap[1] for cam, camCap in captures.items()}
            CONSOLE_OUTPUT = f"Captured: {changeBoxes}"
        elif captureType == 'Add':
            objName = request.form.get("add_obj_name")
            movable = request.form.get("movable")
            targetable = request.form.get("targetable")
            captures = cc.capture()
            changeBoxes = {cam: camCap[1] for cam, camCap in captures.items()}
            CONSOLE_OUTPUT = f"Captured: {changeBoxes}"
            LASTCHANGE = (cc.defineObject, objName, captures)
        elif captureType == 'Delete':
            objName = request.form.get("del_obj_name")
            assert objName in cc.captureObjects, f"Unrecognized object: {objName}"
            cc.captureObjects.pop(objName)
            CONSOLE_OUTPUT = f"{objName} deleted"
        elif captureType == "Select to Move":
            LASTCHANGE = None
            captures = cc.capture()
            try:
                selectedObj = cc.identifySelectedObject(captures)[0]
                startingPoints = [cameras[cam].convertCameraToRealSpace(Camera.boxToBasePoint(camCap[1][0]))
                                  for cam, camCap in captures.items()
                                  if len(camCap[1]) > 0]
                SELECTED = (selectedObj.name, startingPoints)
                CONSOLE_OUTPUT = f"Selected {SELECTED[0]}"
            except Exception as e:
                CONSOLE_OUTPUT = f"Failed to identify selected object: {e}<br>{format_exc()}"
        elif captureType == 'Move':
            assert SELECTED is not None, "No Mech selected to move"
            captures = cc.capture()
            endingPoints = [cameras[cam].convertCameraToRealSpace(Camera.boxToBasePoint(camCap[1][0]))
                            for cam, camCap in captures.items()
                            if len(camCap[1]) > 0]
            CONSOLE_OUTPUT = f"Moved {SELECTED[0]} to Estimated Placement Coordinates:<br>{endingPoints}"
            avgDistanceEstimate = sum(distances := [
                Camera.distanceFormula(sP, eP) * CalibrationBox.millimetersPerPixel
                for eP in endingPoints
                for sP in SELECTED[1]]) / len(distances)
            CONSOLE_OUTPUT += f"<br>Distance Moved Estimate: {avgDistanceEstimate}"
            changeBoxes = {cam: camCap[1] for cam, camCap in captures.items()}
            LASTCHANGE = (cc.updateObject, SELECTED[0], changeBoxes)
            SELECTED = None
        elif captureType == 'Object Range':
            aggressor = request.form.get("rng_obj_name")
            objDistances = "<br>".join([f"{objName:15}: {cc.distanceBetween(aggressor, objName)}"
                                        for objName, obj in cc.captureObjects.items()
                                        if aggressor != objName])
            CONSOLE_OUTPUT = f"Target Distances from {aggressor}:<br>{objDistances}"
        elif captureType == "POV":
            res, image = cv2.imencode('.jpg', pov.collectImage())
            if res != True:
                CONSOLE_OUTPUT = f'Failed to encode image'
            else:
                povCap = base64.b64encode(image.tobytes()).decode()
                CONSOLE_OUTPUT = f'<img src="data:image/jpg;base64,{povCap}" width="100%"/>'
        else:
            raise Exception(f"Unrecognized capture type: {captureType}")
            
    if LASTCHANGE is not None:
        CONSOLE_OUTPUT += f"<br><br>LASTCHANGE: {LASTCHANGE[1]} - {LASTCHANGE[0].__name__}"

    captureImage = '<div width="100%">\n' + "\n".join([f"""
        <div width="100%">
            <h3 class="mt-5">Camera {cam.camNum}</h3>
            <img src="/control/camera/{cam.camNum}" title="{cam.camNum} Capture" width="100%">
        </div>
    """ for cam in cameras.values()]) + "\n</div>"
      
    with open("templates/Controller.html") as f:
        template = f.read()
    template = template.replace("{captureImage}",  captureImage)
    return template


@observerApp.route('/console', methods=['GET'])
def getConsole():
    return CONSOLE_OUTPUT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    cc.capture()
    print(f"Launching Observer Server on {PORT}")
    observerApp.run(host="0.0.0.0", port=PORT)

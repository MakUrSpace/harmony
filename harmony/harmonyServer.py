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
from ipynb.fs.full.HarmonyEye import cc, cm, mc, cameras, hStackImages, vStackImages

import threading
import atexit
from flask import Flask, render_template, Response, request, make_response
from traceback import format_exc

CONSOLE_OUTPUT = "No Output Yet"


POOL_TIME = 0.1 #Seconds
PORT = int(os.getenv("OBSERVER_PORT", "7000"))

ENABLE_CYCLE = True

data_lock = threading.Lock()

captureTimer = threading.Timer(0,lambda x: None,())    
def createCaptureApp():
    app = Flask(__name__)

    def interrupt():
        global captureTimer
        captureTimer.cancel()

    def cycleMachine():
        with data_lock:
            if ENABLE_CYCLE:
                cm.cycle()
        # Set the next timeout to happen
        captureTimer = threading.Timer(POOL_TIME, cycleMachine, ())
        captureTimer.start()   

    def initialize():
        global captureTimer
        captureTimer = threading.Timer(POOL_TIME, cycleMachine, ())
        captureTimer.start()

    initialize()
    # When you kill Flask (SIGTERM), cancels the timer
    atexit.register(interrupt)
    return app


observerApp = createCaptureApp()


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(cameras.values())[0]
        shape = (400, 400)
        mid = [int(d / 2) for d in shape]
        zeros = np.zeros(shape, dtype="uint8")
        
        consoleImage = cv2.putText(zeros, f'Status: {cm.mode:7} -- {cm.state:10}',
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Cycle {cm.cycleCounter}',
            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        if len(cm.transitions) > 0:
            lastObj = cm.transitions[-1]['obj']
            currentLocation = [f"{pt:7.2f}" for pt in cc.rsc.changeSetToRealCenter(lastObj)]
            consoleImage = cv2.putText(zeros, f'Last Action',
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

            try:
                lastMem = cm.memory[cm.memory.index(lastObj)]
                objectIdentifier = f'{lastMem.objectType} {lastMem.name}'
            except ValueError:
                objectIdentifier = f'Unclassified Object'
            objectLocation = f'at {currentLocation}'
    
            consoleImage = cv2.putText(zeros, objectIdentifier,
                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            consoleImage = cv2.putText(zeros, objectLocation,
                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            if not lastObj.isNewObject:
                lastLocation = [f"{d:7.2f}" for d in cc.rsc.changeSetToRealCenter(lastObj.previousVersion())]
                distanceMoved = cc.rsc.trackedObjectLastDistance(lastObj)
                consoleImage = cv2.putText(zeros, f'Moved {distanceMoved:6.2f} mm',
                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
                consoleImage = cv2.putText(zeros, f'From {lastLocation}',
                    (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
            else:
                consoleImage = cv2.putText(zeros, f'Added',
                    (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
   
        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@observerApp.route('/console', methods=['GET'])
def getConsoleImage():
    return Response(renderConsole(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/bootstrap.min.css', methods=['GET'])
def getBSCSS():
    with open("templates/bootstrap.min.css", "r") as f:
        bscss = f.read()
    return Response(bscss, mimetype="text/css")


@observerApp.route('/bootstrap.min.js', methods=['GET'])
def getBSJS():
    with open("templates/bootstrap.min.js", "r") as f:
        bsjs = f.read()
    return Response(bsjs, mimetype="application/javascript")


@observerApp.route('/htmx.min.js', methods=['GET'])
def getHTMX():
    with open("templates/htmx.min.js", "r") as f:
        htmx = f.read()
    return Response(htmx, mimetype="application/javascript")


def genCameraFullViewWithActiveZone(camNum):
    while True:
        cam = cameras[int(camNum)]
        img = cam.drawActiveZone(cam.mostRecentFrame)

        h, w, _ = img.shape
        dy, dx = 100, 100
        rows, cols = int(h / dy), int(w / dx)
        
        # Draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=(0, 0, 255), thickness=2)
        
        # Draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=(0, 0, 255), thickness=2)
        
        gridImage = 255 * np.ones([1300, 2220, 3], dtype="uint8")
        gridImage[100:1180, 200:2120] = img
        
        # Annotate with pixel numbers
        for i in range(rows):
            y = int(round(i * dy + dy)) + 120
            x = 50
            cv2.putText(gridImage, f"{(i+1) * dy:4}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        for j in range(cols):
            if j % 2 != 0:
                continue
            x = int(round(j * dx + dx / 2)) + 180
            y = 1250
            cv2.putText(gridImage, f"{(j+1) * dx:4}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 2, cv2.LINE_AA)

        ret, gridImage = cv2.imencode('.jpg', gridImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + gridImage.tobytes() + b'\r\n')
    
    
@observerApp.route('/config/camera/<camNum>')
def cameraActiveZoneWithObjects(camNum):
    return Response(genCameraFullViewWithActiveZone(int(camNum)), mimetype='multipart/x-mixed-replace; boundary=frame')


def buildConfigurator():
    cameraConfigRows = []
    clickSubs = []
    for cam in cameras.values():
        activeZone = json.dumps(cam.activeZone.tolist())
        cameraConfigRows.append(f"""
            <div class="row justify-content-center text-center">
                <h3 class="mt-5">Camera {cam.camNum} <input type="button" value="Delete" class="btn-error"></h3>
                <img src="/config/camera/{cam.camNum}" title="{cam.camNum} Capture" height="375" id="cam{cam.camNum}" onclick="camClickListener({cam.camNum}, event)">
                <label for="az">Active Zone</label><br>
                <input type="text" name="az" id="cam{cam.camNum}_ActiveZone" value="{activeZone}" size="50" hx-post="/config/cam{cam.camNum}_activezone" hx-swap="none"><br>
            </div>""")
    with open("templates/Configuration.html") as f:
        template = f.read()
    return template.replace(
        "{cameraConfigRows}", "\n".join(cameraConfigRows)
    )

    
@observerApp.route('/config', methods=['GET'])
def config():
    return buildConfigurator()


@observerApp.route('/config', methods=['POST'])
def updateConfig():
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = "Saved Configuration"
    cc.saveConfiguration()
    return "success"


@observerApp.route('/config/cam<camNum>_activezone', methods=['POST'])
def updateCamActiveZone(camNum):
    global CONSOLE_OUTPUT
    CONSOLE_OUTPUT = ""
    print(f"Received update Active Zone request for {camNum}")
    try:
        az = np.float32(json.loads(request.form.get(f"az")))
        cam = cameras[int(camNum)]
        cam.setActiveZone(az)
    except:
        print(f"Unrecognized data: {camNum} - {az}")
    CONSOLE_OUPUT = f"Updated {camNum} AZ"
    return "success"


@observerApp.route('/calibrator/start_calibration')
def startCalibration():
    with data_lock:
        cm.startCalibration()
    return "success"
        
        
@observerApp.route('/calibrator/abort_calibration')
def abortCalibration():
    with data_lock:
        cm.abortCalibration()
    return "success"


@observerApp.route('/calibrator', methods=['GET'])
def buildCalibrator():
    if cm.mode != "calibrate":
        calibrationMonitor = f"""<p>Calibration in not in Progress<p>"""
    else:
        formattedPts = [f"> {pt}<br>" for pt in cm.calibrationPts]
        calibrationMonitor = f"""
            <p>Calibration in Progress: {cm.mode == "calibrate"}<br>Progress {len(cm.calibrationPts)} / {cm.NUM_CALIB_PTS}<br>{formattedPts}<p>
        """
        
    if cm.cc.rsc == None:
        calibratorResult = "Not Calibrated"
    else:
        ims = []
        for camNum, cons in cm.cc.rsc.converters.items():
            if len(cons) == 1:
                camImage = cons[0].showUnwarpedImage()
            else:
                camImage = hStackImages([c.showUnwarpedImage() for c in cons])
            ims.append(camImage)
        unWarped = imageToBase64(vStackImages(ims))
        calibratorResult = f'{cm.cc.rsc}<br><img alt="Unwarped Camera Views" src="data:image/jpg;base64,{unWarped}">'

    with open("templates/Calibrator.html") as f:
        template = f.read()
    return template.replace(
        "{calibratorResult}", calibratorResult).replace(
        "{calibrationMonitor}", calibrationMonitor)

    if (start_calibration := request.form.get("calib_type")) == 'start_calibration':
        with data_lock:
            cm.startCalibration()
    elif (abort_calibration := request.form.get("calib_type")) == 'abort_calibration':
        with data_lock:
            cm.abortCalibration()
    return buildCalibrator()


def genCombinedCamerasView():
    while True:
        camImages = []
        for camNum in cameras.keys():
            camImage = cameras[camNum].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCameraWithChangesView(camNum):
    camNum = int(camNum)
    cam = cameras[camNum]
    while True:
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            print("Has changes")
        if cm.lastClassification is not None:
            print("Has class")
        camImage = cam.mostRecentFrame.copy()
        # Paint known objects blue
        for memObj in cm.memory:
            if memObj.changeSet[camNum].changeType not in ['delete', None]:
                memContour = np.array([memObj.changeSet[camNum].changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
        # Paint last changes red
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            lastChange = cm.lastChanges.changeSet[camNum]
            if lastChange is not None and lastChange.changeType not in ['delete', None]:
                lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
        # Paint classification green
        if cm.lastClassification is not None and not cm.lastClassification.empty:
            lastClass = cm.lastClassification.changeSet[camNum]
            if lastClass is not None and lastClass.changeType not in ['delete', None]:
                lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/camWithChanges/<camNum>')
def cameraViewWithChangesResponse(camNum):
    return Response(genCameraWithChangesView(camNum), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        camImages = []
        if cm.lastChanges is not None and not cm.lastChanges.empty:
            print("Has changes")
        if cm.lastClassification is not None:
            print("Has class")
        for camNum in cameras.keys():
            camImage = cameras[camNum].mostRecentFrame.copy()
            # Paint known objects blue
            for memObj in cm.memory:
                if memObj.changeSet[camNum].changeType not in ['delete', None]:
                    memContour = np.array([memObj.changeSet[camNum].changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
            # Paint last changes red
            if cm.lastChanges is not None and not cm.lastChanges.empty:
                lastChange = cm.lastChanges.changeSet[camNum]
                if lastChange is not None and lastChange.changeType not in ['delete', None]:
                    lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
            # Paint classification green
            if cm.lastClassification is not None and not cm.lastClassification.empty:
                lastClass = cm.lastClassification.changeSet[camNum]
                if lastClass is not None and lastClass.changeType not in ['delete', None]:
                    lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@observerApp.route('/control/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def minimapGenerator():
    while True:
        camImage = cm.buildMiniMap()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
    

@observerApp.route('/control/minimap')
def minimapResponse():
    return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@observerApp.route('/control/set_passive')
def controlSetPassive():
    with data_lock:
        cm.passiveMode()
    return "success"
        

@observerApp.route('/control/set_add')
def controlSetAdd():
    with data_lock:
        cm.addMode()
    return "success"
        

@observerApp.route('/control/set_move')
def controlSetMove():
    with data_lock:
        cm.moveMode()
    return "success"
        

@observerApp.route('/control/set_action')
def controlSetAction():
    with data_lock:
        cm.actionMode()
    return "success"


@observerApp.route('/control')
def buildController():
    with open("templates/Controller.html", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'<input type="button" value="Camera {camNum}" onclick="liveCameraClick({camNum})">' for camNum in cameras.keys()])
    template = template.replace("{cameraButtons}", cameraButtons)
    return template


def buildObjectTable():
    changeRows = []
    captureModals = []
    print(f"Aware of {len(cm.memory)} objects")
    for capture in cm.memory:
        encodedBA = imageToBase64(capture.visual())
        center = ", ".join([f"{pt:.2f}" for pt in cc.rsc.changeSetToRealCenter(capture)])
        changeRow = f"""
            <div class="row mb-1">
                <div class="col">
                    <p>{"" if capture.name is None else capture.name}</p>
                </div>
                <div class="col">
                    <p>({center})</p>
                </div> 
                <div class="col">
                    <img class="img-fluid border border-secondary" alt="Capture Image" src="data:image/jpg;base64,{encodedBA}" style="border-radius: 10px;">
                </div>
            </div>"""
        changeRows.append(changeRow)
    if len(cm.newObjectBuffer) > 0:
        changeRows.insert(0, """
            <div class="row">
                <button type="button" class="btn btn-warning btn-sm" onclick="location.href='/objectsettings';">Assign New Objects</button>
            </div>
        """)
    return " ".join(changeRows)


@observerApp.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable()


def buildObjectSettingsTable():
    changeRows = []
    captureModals = []
    print(f"Aware of {len(cm.memory)} objects")
    for idx, capture in enumerate(cm.newObjectBuffer):
        encodedBA = imageToBase64(capture.visual())
        center = [f"{pt:.2f}" for pt in cc.rsc.changeSetToRealCenter(capture)]
        changeRow = f"""
            <div class="row mb-1">
                <div class="col">
                    <form method="post">
                      <div class="form-group">
                        <label for="objectName">Object Name</label>
                        <input type="text" name="objectName" value="" required>
                      </div>
                      <div class="form-group">
                        <label for="objectType">Object Type</label>
                        <input name="objectType" list="objectTypeDataList" value="" required>
                      </div>
                      <input type="hidden" name="objectId" value="{capture.oid}">
                      <button type="submit" class="btn btn-secondary">Update Object</button>
                    </form>
                </div>
                <div class="col">
                    <p>{center}</p><br>
                    <img class="img-fluid border border-secondary" alt="Capture Image" src="data:image/jpg;base64,{encodedBA}" style="border-radius: 10px;">
                </div>
            </div>"""
        changeRows.append(changeRow)
    objOptions = "\n".join([f'<option value="{objType}">' for objType in mc.ObjectFactories.keys()])
    changeRows.insert(0, f"""
            <datalist id="objectTypeDataList">
                {objOptions}
            </datalist>""")
    with open("templates/ChangeTable.html", "r") as f:
        template = f.read()
    return template.replace("{changeTableBody}", " ".join(changeRows))


@observerApp.route('/objectsettings', methods=['GET'])
def getObjectSettingsTable():
    return buildObjectSettingsTable()
    
    
@observerApp.route('/objectsettings', methods=['POST'])
def postObjectSettings():
    objName = request.form.get("objectName")
    objType = request.form.get("objectType")
    oid = request.form.get("objectId")
    cm.annotateObject(oid, objName, objType)
    return buildObjectSettingsTable()


if __name__ == "__main__":
    cm.cycle()
    print(f"Launching Observer Server on {PORT}")
    observerApp.run(host="0.0.0.0", port=PORT)

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

import threading
import atexit
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for
from traceback import format_exc

from observer.configurator import configurator, setConfiguratorApp
from observer.calibrator import calibrator, CalibratedObserver, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT


harmony = Blueprint('harmony', __name__, template_folder='harmony_templates')


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(app.cc.cameras.values())[0]
        shape = (170, 400)
        mid = [int(d / 2) for d in shape]
        zeros = np.zeros(shape, dtype="uint8")

        consoleImage = cv2.putText(zeros, f'Cycle {app.cm.cycleCounter}',
            (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Mode: {app.cm.mode:7}',
            (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'Board State: {app.cm.state:10}',
            (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
        consoleImage = cv2.putText(zeros, f'LO: {CONSOLE_OUTPUT}',
            (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')


@harmony.route('/harmony_console', methods=['GET'])
def getConsoleImage():
    return Response(renderConsole(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCamerasView():
    while True:
        camImages = []
        for camName in app.cc.cameras.keys():
            camImage = app.cc.cameras[camName].mostRecentFrame.copy()
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/combinedCameras')
def combinedCamerasResponse():
    return Response(genCombinedCamerasView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCameraWithChangesView(camName):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
            print("Has changes")
        if app.cm.lastClassification is not None:
            print("Has class")
        camImage = cam.cropToActiveZone(cam.mostRecentFrame.copy())
        # Paint known objects blue
        for memObj in app.cm.memory:
            if memObj.changeSet[camName].changeType not in ['delete', None]:
                memContour = np.array([memObj.changeSet[camName].changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
        # Paint last changes red
        if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
            lastChange = app.cm.lastChanges.changeSet[camName]
            if lastChange is not None and lastChange.changeType not in ['delete', None]:
                lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
        # Paint classification green
        if app.cm.lastClassification is not None and not app.cm.lastClassification.empty:
            lastClass = app.cm.lastClassification.changeSet[camName]
            if lastClass is not None and lastClass.changeType not in ['delete', None]:
                lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/camWithChanges/<camName>')
def cameraViewWithChangesResponse(camName):
    if camName == "VirtualMap":
        return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(genCameraWithChangesView(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def fullCam(camName):
    camName = str(camName)
    cam = app.cc.cameras[camName]
    while True:
        camImage = cam.mostRecentFrame.copy()
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/fullCam/<camName>')
def genFullCam(camName):
    return Response(fullCam(camName), mimetype='multipart/x-mixed-replace; boundary=frame')


def genCombinedCameraWithChangesView():
    while True:
        camImages = []
        if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
            print("Has changes")
        if app.cm.lastClassification is not None:
            print("Has class")
        for camName in app.cc.cameras.keys():
            camImage = app.cc.cameras[camName].mostRecentFrame.copy()
            # Paint known objects blue
            for memObj in app.cm.memory:
                if memObj.changeSet[camName].changeType not in ['delete', None]:
                    memContour = np.array([memObj.changeSet[camName].changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, memContour, -1, (255, 0, 0), -1)
            # Paint last changes red
            if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
                lastChange = app.cm.lastChanges.changeSet[camName]
                if lastChange is not None and lastChange.changeType not in ['delete', None]:
                    lastChangeContour = np.array([lastChange.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastChangeContour, -1 , (0, 0, 255), -1)
            # Paint classification green
            if app.cm.lastClassification is not None and not app.cm.lastClassification.empty:
                lastClass = app.cm.lastClassification.changeSet[camName]
                if lastClass is not None and lastClass.changeType not in ['delete', None]:
                    lastClassContour = np.array([lastClass.changePoints], dtype=np.int32)
                    camImage = cv2.drawContours(camImage, lastClassContour, -1 , (0, 255, 0), -1)
            camImages.append(camImage)
        camImage = vStackImages(camImages)
        camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


@harmony.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


@harmony.route('/reset')
def resetharmony():
    with DATA_LOCK:
        app.cm = CalibratedObserver(app.cc)
    return 'success'


@harmony.route('/set_passive')
def controlSetPassive():
    with DATA_LOCK:
        app.cm.passiveMode()
    return buildModeController()
        

@harmony.route('/set_track')
def controlSetTrack():
    with DATA_LOCK:
        app.cm.trackMode()
    return buildModeController()


def buildModeController():
    return """  <div class="btn-group" role="group" aria-label="harmony Capture Mode Control Buttons">
                  <input type="radio" class="btn-check" name="btnradio" id="passive" autocomplete="off" {passiveChecked}hx-get="{harmonyURL}set_passive" hx-target="#modeController">
                  <label class="btn btn-outline-primary" for="passive">Passive</label>
                  <input type="radio" class="btn-check" name="btnradio" id="track" autocomplete="off" {activeChecked}hx-get="{harmonyURL}set_track" hx-target="#modeController">
                  <label class="btn btn-outline-primary" for="track">Track</label>
                </div>""".replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{passiveChecked}", 'checked=""' if app.cm.mode == "passive" else '').replace(
        "{activeChecked}", 'checked=""' if app.cm.mode == "track" else '')


@harmony.route('/get_mode_controller')
def getModeController():
    return buildModeController()


@harmony.route('/')
def buildHarmony():
    if type(app.cm) is not CalibratedObserver:
        with DATA_LOCK:
            app.cm = CalibratedObserver(app.cc)
    with open("harmony_templates/Harmony.html", "r") as f:
        template = f.read()
    cameraButtons = '<input type="button" value="Virtual Map" onclick="liveCameraClick(\'VirtualMap\')">' + ' '.join([f'''<input type="button" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in app.cc.cameras.keys()])
    defaultCam = [camName for camName, cam in app.cc.cameras.items()][0]
    return template.replace(
        "{defaultCamera}", defaultCam).replace(
        "{cameraButtons}", cameraButtons).replace(
        "{harmonyURL}", url_for('.buildHarmony')).replace(
        "{configuratorURL}", '/configurator')


def captureToChangeRow(capture):
    encodedBA = imageToBase64(capture.visual())
    name = "None"
    objType = "None"
    center = ", ".join(["0", "0"])
    health = ""
    numHits = 0
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "
    with open("harmony_templates/TrackedObjectRow.html") as f:
        changeRowTemplate = f.read()
    moveDistance = app.cm.cc.rsc.trackedObjectLastDistance(capture)
    moveDistance = "None" if moveDistance is None else f"{moveDistance:6.0f} mm"
    changeRow = changeRowTemplate.replace(
        "{objectName}", capture.oid).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in app.cm.cc.rsc.changeSetToRealCenter(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{encodedBA}", imageToBase64(capture.visual()))
    return changeRow


def buildObjectTable(filter=None):
    changeRows = []
    print(f"Aware of {len(app.cm.memory)} objects")
    for capture in app.cm.memory:
        if filter is not None and getattr(capture, 'objectType', None) != filter:
            continue
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@harmony.route('/objects', methods=['GET'])
def getObjectTable():
    filter = request.args.get('filter', None)
    return buildObjectTable(filter=filter)


def buildObjectsFilter(filter=None):
    assert filter in [None, 'None', 'Terrain', 'Building', 'Unit'], f"Unrecognized filter type: {filter}"

    noneSelected, terrainSelected, buildingSelected, unitSelected = "", "", "", ""
    if filter == 'None' or filter is None:
        filterQuery = ''
        noneSelected = ' checked="checked"'
    elif filter == 'Terrain':
        filterQuery = '?filter=Terrain'
        terrainSelected = ' checked="checked"'
    elif filter == 'Building':
        filterQuery = '?filter=Building'
        buildingSelected = ' checked="checked"'
    elif filter == 'Unit':
        filterQuery = '?filter=Unit'
        unitSelected = ' checked="checked"'
        
    template = """
    <div id="objectFilter"class="btn-group" role="group" aria-label="Objects Table Filter Select Buttons">
      <input type="radio" class="btn-check" name="objectFilterradio" id="None" autocomplete="off" {noneSelected} hx-get="{harmonyURL}objects_filter" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="None">None</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Terrain" autocomplete="off" {terrainSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Terrain" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Terrain">Terrain</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Building" autocomplete="off" {buildingSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Building" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Building">Building</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Unit" autocomplete="off" {unitSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Unit" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Unit">Unit</label>
    </div>
    <div id="objectsTable" hx-get="{harmonyURL}objects{filter}" hx-trigger="every 1s">{objectRows}</div>"""
    template = template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{filter}", filterQuery).replace(
        "{noneSelected}", noneSelected).replace(
        "{terrainSelected}", terrainSelected).replace(
        "{buildingSelected}", buildingSelected).replace(
        "{unitSelected}", unitSelected).replace(
        "{objectRows}", buildObjectTable(filter))
    return template


@harmony.route('/objects_filter', methods=['GET'])
def getObjectTableContainer():
    return buildObjectsFilter(request.args.get('objectFilter', None))
    
    
@harmony.route('/objects/<objectId>', methods=['GET'])
def getObjectSettings(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404

    objectName = cap.oid
    harmonyURL = url_for(".buildHarmony")
    with open("harmony_templates/TrackedObjectUpdater.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", harmonyURL).replace(
        "{objectName}", cap.oid)


@harmony.route('/objects/<objectId>', methods=['POST'])
def updateObjectSettings(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404
    newName = request.form["objectName"]
    if newName != cap.oid:
        cap.oid = newName
    for key, value in request.form.items():
        if key == 'objectName':
            continue
        setattr(cap, key, value)
    return buildObjectsFilter()
    
    
@harmony.route('/objects/<objectId>', methods=['DELETE'])
def deleteObjectSettings(objectId):
    app.cm.deleteObject(objectId)
    return buildObjectsFilter()


@harmony.route('/objects/<objectId>/type', methods=['POST'])
def updateObjectType(objectId):
    newType = request.form["objectType"]
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if newType == "Terrain":
        settings = {"Rating": getattr(cap, "Rating", "1"), "Elevation": getattr(cap, "Elevation", "5")}
    elif newType == "Building":
        settings = {"Class": getattr(cap, "Class", "Residential"), "Elevation": getattr(cap, "Elevation", "5")}
    elif newType == "Unit":
        settings = {"Class": getattr(cap, "Class", "Atlas")}
    else:
        return f"Invalid Object Type Request", 405
    return "<br>".join([
        f"""<label for="{key}">{key}</label><input type="text" class="form-control" name="{key}" value="{value}">"""
        for key, value in settings.items()])
    


@harmony.route('/object_distances/<objectId>', methods=['GET'])
def getObjectDistances(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404

    with open("harmony_templates/ObjectDistanceCard.html") as f:
        cardTemplate = f.read()
    objDistCards = []
    for target in app.cm.memory:
        if target.oid == cap.oid:
            continue
        else:
            objDistCards.append(cardTemplate.replace(
                "{targetName}", target.oid).replace(
                "{encodedBA}", imageToBase64(target.visual())).replace(
                "{objectDistance}", f"{app.cm.cc.rsc.distanceBetweenObjects(cap, target):6.0f} mm"))
                
    with open("harmony_templates/ObjectDistanceTable.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{objectDistanceCards}", "\n".join(objDistCards))


def minimapGenerator():
    while True:
        camImage = app.cm.cc.buildMiniMap(
            blueObjects=app.cm.memory,
            greenObjects=[app.cm.lastClassification] if app.cm.lastClassification is not None else None)
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
    

@harmony.route('/minimap')
def minimapResponse():
    return Response(minimapGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def setHarmonyApp(newApp):
    global app
    app = newApp


if __name__ == "__main__":
    from flask import Flask, redirect, Response, Blueprint
    from observer.observer import CalibratedCaptureConfiguration, observer, configurator, CalibratedObserver, registerCaptureService, setConfiguratorApp, setObserverApp
    
    app = Flask(__name__)
    app.cc = CalibratedCaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(configurator, url_prefix='/configurator')
    app.register_blueprint(harmony, url_prefix='/harmony')
    app.cm = CalibratedObserver(app.cc)
    setConfiguratorApp(app)
    setObserverApp(app)
    
    @app.route('/')
    def index():
        return redirect('/harmony', code=303)
    
    @app.route('/bootstrap.min.css', methods=['GET'])
    def getBSCSS():
        with open("templates/bootstrap.min.css", "r") as f:
            bscss = f.read()
        return Response(bscss, mimetype="text/css")
    
    @app.route('/bootstrap.min.js', methods=['GET'])
    def getBSJS():
        with open("templates/bootstrap.min.js", "r") as f:
            bsjs = f.read()
        return Response(bsjs, mimetype="application/javascript")
    
    @app.route('/htmx.min.js', methods=['GET'])
    def getHTMX():
        with open("templates/htmx.min.js", "r") as f:
            htmx = f.read()
        return Response(htmx, mimetype="application/javascript")
    
    registerCaptureService(app)
    PORT = 7000
    print(f"Launching harmony Server on {PORT}")
    app.run(host="0.0.0.0", port=PORT)

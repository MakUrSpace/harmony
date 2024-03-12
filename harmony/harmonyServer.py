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
from dataclasses import dataclass

import threading
import atexit
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for
from traceback import format_exc

from observer.configurator import configurator, setConfiguratorApp
from observer.calibrator import calibrator, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT
from ipynb.fs.full.HarmonyMachine import HarmonyMachine 


harmony = Blueprint('harmony', __name__, template_folder='harmony_templates')
ROUND = 0


class GameState:
    states = ["Add", "Movement", "Declare", "Resolve"]
    state = "Add"
    round = 0
    declaredActions = {}

    @classmethod
    def reset(cls):
        cls.round = 0
        cls.state = "Add"
        cls.declaredActions = {}

    @classmethod
    def nextState(cls, currentState=None):
        if currentState is not None:
            assert currentState == cls.state, "GameState mismatch. Cannot progress in this manner"
        currentState = cls.state
        newState = currentState
        if newState == "Add":
            newState = "Movement"
            cls.movements = []
        elif newState == "Movement":
            newState = "Declare"
            cls.declaredActions = {}
        elif newState == "Declare":
            newState = "Resolve"
        elif newState == "Resolve":
            newState = "Movement"
            cls.declaredActions = {}
            cls.round += 1
        cls.state = newState

    @classmethod
    def gameStateButton(cls):
        gs = cls.state
        if gs == "Add":
            button = """<input type="button" class="btn btn-info" name="commitAdditions" id="passive" hx-get="{harmonyURL}commit_additions" hx-target="#objectInteractor" value="Start Game">"""
        elif gs == "Movement":
            button = """<input type="button" class="btn btn-info" name="commitMovement" id="passive" hx-get="{harmonyURL}commit_movement" hx-target="#objectInteractor" value="Commit Movement">"""
        elif gs == "Declare":
            button = """<input type="button" class="btn btn-info" name="declareActions" id="passive" hx-get="{harmonyURL}declare_actions" hx-target="#objectInteractor" value="Declare Actions">"""
        elif gs == "Resolve":
            button = """<input type="button" class="btn btn-danger" name="resolveActions" id="passive" hx-get="{harmonyURL}resolve_actions" hx-target="#objectInteractor" value="Resolve Actions">"""
        return button.replace("{harmonyURL}", url_for(".buildHarmony"))


@harmony.route('/get_game_controller')
def getGameController():
    return GameState.gameStateButton()


@harmony.route('/commit_additions')
def commitAdditions():
    with DATA_LOCK:
        GameState.nextState("Add")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/commit_movement', methods=['GET'])
def commitMovement():
    with DATA_LOCK:
        app.cm.passiveMode()
        GameState.nextState("Movement")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/declare_actions', methods=['GET'])
def declareActions():
    with DATA_LOCK:
        app.cm.passiveMode()
        GameState.nextState("Declare")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/resolve_actions', methods=['GET'])
def resolveActions():
    with DATA_LOCK:
        app.cm.trackMode()
        GameState.nextState("Resolve")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(app.cc.cameras.values())[0]
        shape = (200, 400)
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
        consoleImage = cv2.putText(zeros, f'Round: {GameState.round:3}-{GameState.state}',
            (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

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
        app.cm = HarmonyMachine(app.cc)
        app.gm = GameState
        app.gm.reset()
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
    if type(app.cm) is not HarmonyMachine:
        with DATA_LOCK:
            app.cm = HarmonyMachine(app.cc)
            app.gm = GameState
            app.gm.reset()
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
        "{encodedBA}", imageToBase64(capture.visual())).replace(
        "{actions}", "" if getattr(capture, 'objectType', None) != "Unit" else f"""<button class="btn btn-primary" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}/actions">Object Actions</button>""").replace(
        "{edit}", "" if GameState.state != "Add" else f"""<button class="btn btn-info" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}">Edit</button>""")
    return changeRow


def buildObjectTable(filter=None):
    changeRows = []
    for capture in app.cm.memory:
        if filter is not None and getattr(capture, 'objectType', None) != filter:
            continue
        changeRows.append(captureToChangeRow(capture))
    return " ".join(changeRows)


@harmony.route('/objects', methods=['GET'])
def getObjectTable():
    filter = request.args.get('filter', None)
    return buildObjectTable(filter=filter)


def buildObjectActionResolver():
    return f"""<div id="objectActionResolver" hx-get="{url_for(".buildHarmony")}objects_filter" hx-trigger="every 1s"><p>{json.dumps(GameState.declaredActions, indent=4)}</p></div>"""


def buildObjectsFilter(filter=None):
    if GameState.state == "Resolve":
        return buildObjectActionResolver()

    assert filter in [None, 'None', 'Terrain', 'Building', 'Unit'], f"Unrecognized filter type: {filter}"
    buildObjectsFilter.filter = filter
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
    return """<div class="row ">
                <h2 class="mt-5">Object Tracker</h2>
            </div>""" + template


def getInteractor():
    if GameState.state == "Resolve":
        return buildObjectActionResolver()
    else:
        return buildObjectsFilter


@harmony.route('/objects_filter', methods=['GET'])
def getObjectTableContainer():
    if GameState.state != "Resolve":
        return buildObjectsFilter(request.args.get('objectFilter', None))
    else:
        return buildObjectActionResolver()
    
    
@harmony.route('/objects/<objectId>', methods=['GET'])
def getObject(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404

    objectName = cap.oid
    with open("harmony_templates/TrackedObjectUpdater.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{objectSettings}", buildObjectSettings(cap))


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
    
    
@harmony.route('/objects/<objectId>/declare_attack/<targetId>', methods=['POST'])
def declareAttackOnTarget(objectId, targetId):
    GameState.declaredActions[objectId] = {"target": targetId}
    return buildObjectActions(objectId)
    
    
@harmony.route('/objects/<objectId>/declare_no_action', methods=['POST'])
def declareNoAction(objectId):
    GameState.declaredActions[objectId] = {}
    return buildObjectActions(objectId)


def buildObjectSettings(obj):
    objType = getattr(obj, 'objectType', 'None')
    terrainSelected, buildingSelected, unitSelected = '', '', ''
    if objType == "Terrain":
        settings = {"Rating": getattr(obj, "Rating", "1"), "Elevation": getattr(obj, "Elevation", "5")}
        terrainSelected = " selected='selected'"
    elif objType == "Building":
        settings = {"Class": getattr(obj, "Class", "Residential"), "Elevation": getattr(obj, "Elevation", "5")}
        buildingSelected = " selected='selected'"
    elif objType == "Unit":
        settings = {"Class": getattr(obj, "Class", "Atlas"), "Skill": str(getattr(obj, "Skill", 4))}
        unitSelected = " selected='selected'"
    else:
        settings = {}

    objectSettings = "<br>".join([
        f"""<label for="{key}">{key}</label><input type="text" class="form-control" name="{key}" value="{value}">"""
        for key, value in settings.items()])

    return """
    <select name="objectType" id="objectType" hx-target="#objectSettings" hx-post='{harmonyURL}objects/{objectName}/type'>
      <option value="None">None</option>
      <option value="Terrain" {terrainSelected}>Terrain</option>
      <option value="Building" {buildingSelected}>Building</option>
      <option value="Unit" {unitSelected}>Unit</option>
    </select><br>
    {objectSettings}
    """.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", obj.oid).replace(
        "{objectSettings}", objectSettings).replace(
        "{terrainSelected}", terrainSelected).replace(
        "{buildingSelected}", buildingSelected).replace(
        "{unitSelected}", unitSelected)


@harmony.route('/objects/<objectId>/settings', methods=['GET'])
def getObjectSettings(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return 404
    return buildObjectSettings(cap)


@harmony.route('/objects/<objectId>/type', methods=['POST'])
def updateObjectType(objectId):
    newType = request.form["objectType"]
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return 404

    assert newType in ["None", "Terrain", "Building", "Unit"], f"Unrecognized object type: {newType}"

    with DATA_LOCK:
        cap.objectType = newType
    return buildObjectSettings(cap)
    

def buildObjectActions(objectId):
    cap = None
    for capture in app.cm.memory:
        if capture.oid == objectId:
            cap = capture
            break
    if cap is None:
        return f"{objectId} Not found", 404

    with open("harmony_templates/ObjectActionCard.html") as f:
        cardTemplate = f.read()
    objActCards = []
    objMovement = app.cm.cc.rsc.trackedObjectLastDistance(cap)
    aMM = -1 if objMovement is None or objMovement < 10 else 1
    for target in app.cm.memory:
        if target.oid == cap.oid:
            continue
        else:
            declare = ""
            if GameState.state == "Declare":
                disabled = "" if cap.oid not in GameState.declaredActions or target.oid != GameState.declaredActions[cap.oid].get('target', None) else "disabled"
                declare = f"""
                <div class="col">
                    <input {disabled} type="button" class="btn btn-warning" value="Declare Attack" id="declare_{target.oid}" hx-target="#objectInteractor" hx-post="{url_for(".buildHarmony")}objects/{cap.oid}/declare_attack/{target.oid}">
                </div>"""

            targetDistance = app.cm.cc.rsc.distanceBetweenObjects(cap, target)
            targetRange = "short" if targetDistance < 155 else "medium" if targetDistance < 610 else "long"
            rangeModifier = 0 if "short" else 2 if "medium" else 4

            tMM = app.cm.cc.rsc.trackedObjectLastDistance(target)
            tMM = -1 if tMM is None or tMM < 10 else 1
            targetNumber = int(cap.Skill) + aMM + tMM + 0 + rangeModifier
            objActCards.append(cardTemplate.replace(
                "{harmonyURL}", url_for(".buildHarmony")).replace(
                "{objectName}", cap.oid).replace(
                "{targetName}", target.oid).replace(
                "{encodedBA}", imageToBase64(target.visual())).replace(
                "{objectDistance}", f"{targetRange.capitalize()} ({targetDistance / 25.4:6.1f} in)").replace(
                "{declare}", declare).replace(
                "{skill}", cap.Skill).replace(
                "{attackerMovementModifier}", str(aMM)).replace(
                "{targetMovementModifier}", str(tMM)).replace(
                "{range}", str(rangeModifier)).replace(
                "{other}", "0").replace(
                "{targetNumber}", str(targetNumber)))

    if GameState.state == "Declare":
        disabled = "" if cap.oid not in GameState.declaredActions or GameState.declaredActions[cap.oid] != {} else "disabled"
        objActCards.append(f"""
        <div class="row mb-1 border border-secondary border-2">
            <input {disabled} type="button" class="btn btn-danger" value="Take No Action" id="no_action" hx-target="#objectInteractor" hx-post="{url_for(".buildHarmony")}objects/{cap.oid}/declare_no_action">
        </div>
        """.replace("{harmonyURL}", url_for(".buildHarmony")).replace("{objectName}", cap.oid))
    
    with open("harmony_templates/ObjectActions.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{objectIcon}", imageToBase64(cap.icon)).replace(
        "{objectActionCards}", "\n".join(objActCards))


@harmony.route('/objects/<objectId>/actions', methods=['GET'])
def getObjectActions(objectId):
    return buildObjectActions(objectId)


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
    from observer.observer import CalibratedCaptureConfiguration, observer, configurator, registerCaptureService, setConfiguratorApp, setObserverApp
    
    app = Flask(__name__)
    app.cc = CalibratedCaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(configurator, url_prefix='/configurator')
    app.register_blueprint(harmony, url_prefix='/harmony')
    app.cm = HarmonyMachine(app.cc)
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

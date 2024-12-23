import os
from math import ceil
import base64
import argparse
import json
from io import BytesIO
from dataclasses import dataclass
from typing import Callable
from functools import wraps
import threading
import atexit
from traceback import format_exc

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for

from observer.configurator import configurator, setConfiguratorApp
from observer.observer import CalibratedCaptureConfiguration, observer, configurator, registerCaptureService, setConfiguratorApp, setObserverApp
from observer.calibrator import calibrator, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT
from ipynb.fs.full.HarmonyMachine import HarmonyMachine, ObjectAction


harmony = Blueprint('harmony', __name__, template_folder='harmony_templates')


def gameStateButton(gameState):
    if gameState == "Add":
        button = """<input type="button" class="btn btn-info" name="commitAdditions" id="passive" hx-get="{harmonyURL}commit_additions" hx-target="#objectInteractor" value="Start Game">"""
    elif gameState == "Move":
        button = """<input type="button" class="btn btn-info" name="commitMovement" id="passive" hx-get="{harmonyURL}commit_movement" hx-target="#objectInteractor" value="Commit Movement">"""
    elif gameState == "Declare":
        button = """<input type="button" class="btn btn-info" name="declareActions" id="passive" hx-get="{harmonyURL}declare_actions" hx-target="#objectInteractor" value="Declare Actions">"""
    elif gameState == "Action":
        button = """<input type="button" class="btn btn-info" name="commitActions" id="passive" hx-get="{harmonyURL}commit_actions" hx-target="#objectInteractor" value="Commit Actions">"""
    else:
        print(f"Unknown gameState: {gameState}")
        button = ""
        print(f"Hmmm -- {app.cm.GameState.getPhase()}")
    return button.replace("{harmonyURL}", url_for(".buildHarmony"))


@harmony.route('/get_game_controller')
def getGameController():
    return gameStateButton(app.cm.GameState.getPhase())


@harmony.route('/commit_additions')
def commitAdditions():
    with DATA_LOCK:
        app.cm.GameState.newPhase("Add")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/commit_movement', methods=['GET'])
def commitMovement():
    with DATA_LOCK:
        app.cm.passiveMode()
        app.cm.GameState.newPhase("Move")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/declare_actions', methods=['GET'])
def declareActions():
    with DATA_LOCK:
        app.cm.passiveMode()
        GameState.nextState("Declare")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/commit_actions', methods=['GET'])
def commitActions():
    with DATA_LOCK:
        app.cm.activeMode()
        app.cm.GameState.newPhase("Action")
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
        consoleImage = cv2.putText(zeros, f'Round: {app.cm.GameState.getRoundCount():3}-{app.cm.GameState.getPhase()}',
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

        camImage = app.cm.getCameraImagesWithChanges(cameraKeys=[camName])[camName]
        ret, camImage = cv2.imencode('.jpg', camImage)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')


def oldGenCameraWithChangesView(camName):
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
def resetHarmony():
    with DATA_LOCK:
        app.cm = HarmonyMachine(app.cc)
        app.cm.reset()
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
                  <input type="radio" class="btn-check" name

@harmony.route('/declare_actions', methods=['GET'])
def declareActions():
    with DATA_LOCK:
        app.cm.passiveMode()
        GameState.nextState("Declare")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))="btnradio" id="track" autocomplete="off" {activeChecked}hx-get="{harmonyURL}set_track" hx-target="#modeController">
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
        resetHarmony()
    with open("harmony_templates/Harmony.html", "r") as f:
        template = f.read()
    cameraButtons = '<input type="button" value="Virtual Map" onclick="liveCameraClick(\'VirtualMap\')">' + ' '.join([f'''<input type="button" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in app.cc.cameras.keys()])
    defaultCam = [camName for camName, cam in app.cc.cameras.items()]
    if len(defaultCam) == 0:
        defaultCam = "None"
    else:
        defaultCam = defaultCam[0]
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
        "{encodedBA}", imageToBase64(capture.visual(withContours=False))).replace(
        "{actions}", "" if getattr(capture, 'objectType', None) != "Unit" else f"""<button class="btn btn-primary" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}/actions">Object Actions</button>""").replace(
        "{edit}", "" if app.cm.GameState.getPhase() != "Add" else f"""<button class="btn btn-info" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}">Edit</button>""")
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
    table = ""
    for objAction in qs.GameState.getDeclaredEvents():
        value = "" if objAction.result is None else objAction.result
        table += f"""<input class="number" id="objectActionResolver" hx-post="{url_for(".buildHarmony")}objects/{objAction.cap.oid}/take_action" value="{value}">"""
    return table


def buildObjectsFilter(filter=None):
    if app.cm.GameState.getPhase() == "Resolve":
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
    if app.cm.GameState.getPhase() == "Resolve":
        return buildObjectActionResolver()
    else:
        return buildObjectsFilter


@harmony.route('/objects_filter', methods=['GET'])
def getObjectTableContainer():
    if app.cm.GameState.getPhase() != "Resolve":
        return buildObjectsFilter(request.args.get('objectFilter', None))
    else:
        return buildObjectActionResolver()


def findObjectIdOr404(objectId_endpoint: Callable) -> Callable:
    @wraps(objectId_endpoint)
    def findOr404_endpoint(**kwargs):
        try:
            objectId = kwargs.pop("objectId")
            return objectId_endpoint(cap=app.cm.findObject(objectId=objectId), **kwargs)
        except KeyError as ke:
            error = f"{objectId} Not found"
            print(error)
            return error, 404
    return findOr404_endpoint
    
    
@harmony.route('/objects/<objectId>', methods=['GET'])
@findObjectIdOr404
def getObject(cap):
    footprint_enabled = request.args.get('footprint', "false") == "true"

    objectName = cap.oid
    with open("harmony_templates/TrackedObjectUpdater.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{objectSettings}", buildObjectSettings(cap)).replace(
        "{footprintToggleState}", str(not footprint_enabled).lower()).replace(
        "{encodedBA}", imageToBase64(cap.visual(withContours=footprint_enabled)))


@harmony.route('/objects/<objectId>', methods=['POST'])
@findObjectIdOr404
def updateObjectSettings(cap):
    newName = request.form["objectName"]
    #TODO: rename game object
    if newName != cap.oid:
        cap.oid = newName
    objectKwargs = {}
    cm.classifyObject(objectId=cap.oid, objectType=, objectSubType=, objectKwargs=)
    for key, value in request.form.items():
        if key == 'objectName':
            continue
        setattr(cap, key, value)
    # TODO: create/update MC objects
    return buildObjectsFilter()
    
    
@harmony.route('/objects/<objectId>', methods=['DELETE'])
@findObjectIdOr404
def deleteObjectSettings(cap):
    app.cm.deleteObject(cap.oid)
    return buildObjectsFilter()
    
    
@harmony.route('/objects/<objectId>/declare_attack/<targetId>', methods=['POST'])
@findObjectIdOr404
def declareAttackOnTarget(cap, targetId):
    target = findObject(targetId)
    app.cm.declareEvent(eventType="Attack", eventFaction="Undeclared", eventObject=cap.oid, eventValue=targetId, eventResult="null")
    return buildObjectActions(cap)
    
    
@harmony.route('/objects/<objectId>/declare_no_action', methods=['POST'])
@findObjectIdOr404
def declareNoAction(cap):
    app.cm.GameState.declareAction(ObjectAction(cap=cap, target=None, result=None))
    return buildObjectActions(cap)


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
@findObjectIdOr404
def getObjectSettings(cap):
    return buildObjectSettings(cap)


@harmony.route('/objects/<objectId>/type', methods=['POST'])
@findObjectIdOr404
def updateObjectType(cap):
    newType = request.form["objectType"]

    assert newType in ["None", "Terrain", "Building", "Unit"], f"Unrecognized object type: {newType}"

    with DATA_LOCK:
        cap.objectType = newType
    return buildObjectSettings(cap)    

def buildObjectActions(cap):
    with open("harmony_templates/ObjectActionCard.html") as f:
        cardTemplate = f.read()
    objActCards = []
    for target in app.cm.memory:
        if target.oid == cap.oid:
            continue
        else:
            declare = ""
            if app.cm.GameState.getPhase() == "Declare":
                # TODO: fix with new gamestate disabled = "" if cap.oid not in app.cm.GameState.declaredEvents or target.oid != getattr(app.cm.GameState.declaredEvents[cap.oid], 'target', None) else "disabled"
                disabled = ""
                declare = f"""
                <div class="col">
                    <input {disabled} type="button" class="btn btn-warning" value="Declare Attack" id="declare_{target.oid}" hx-target="#objectInteractor" hx-post="{url_for(".buildHarmony")}objects/{cap.oid}/declare_attack/{target.oid}">
                </div>"""
            action = ObjectAction(cap, target)
            objActCards.append(cardTemplate.replace(
                "{harmonyURL}", url_for(".buildHarmony")).replace(
                "{objectName}", cap.oid).replace(
                "{targetName}", target.oid).replace(
                "{encodedBA}", imageToBase64(target.visual())).replace(
                "{objectDistance}", f"{targetRange.capitalize()} ({action.targetDistance / 25.4:6.1f} in)").replace(
                "{declare}", declare).replace(
                "{skill}", cap.Skill).replace(
                "{attackerMovementModifier}", str(action.aMM)).replace(
                "{targetMovementModifier}", str(action.tMM)).replace(
                "{range}", str(action.rangeModifier)).replace(
                "{other}", "0").replace(
                "{targetNumber}", str(action.targetNumber)))

    if app.cm.GameState.getPhase() == "Action":
        # TODO: fix with new game state disabled = "" if cap.oid not in GameState.declaredActions or GameState.declaredActions[cap.oid] != {} else "disabled"
        disabled= ""
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
@findObjectIdOr404
def getObjectActions(cap):
    return buildObjectActions(cap)


@harmony.route('/objects/<objectId>/take_action', methods=['POST'])
@findObjectIdOr404
def objectTakeAction(cap):
    result = request.form["actionResult"]
    return buildObjectActions(cap)
    

@harmony.route('/objects/<objectId>/footprint_editor', methods=['GET'])
@findObjectIdOr404
def buildFootprintEditor(cap):
    from flask import Flask, redirect, Response, Blueprint
    from observer.observer import CalibratedCaptureConfiguration, observer, configurator, registerCaptureService, setConfiguratorApp, setObserverApp
    objectName = cap.oid
    with open("harmony_templates/TrackedObjectFootprintEditor.html") as f:
        template = f.read()

    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{encodedBA}", imageToBase64(cap.visual(withContours=True, margin=50))).replace(
        "{camName}", '0')
    return ""
    

@harmony.route('/objects/<objectId>/margin', methods=['POST'])
@findObjectIdOr404
def objectVisualWithMargin(cap):
    margin = int(request.form["objectMargin"])
    cap.margin = margin
    return f"""<img class="img-fluid border border-info border-2" id="objectImage" alt="Capture Image" src="data:image/jpg;base64,{imageToBase64(cap.visual(withContours=True, margin=50))}" style="border-radius: 10px;"  onclick="imageClickListener(event)">"""
    

@harmony.route('/objects/<objectId>/project_addition', methods=['POST'])
@findObjectIdOr404
def projectAdditionOntoObject(cap):
    #margin = getattr(cap, "margin", 5)
    margin = 50
    addition_points = json.loads(request.form["additionPolygon"])
    addition_points = np.int32(addition_points)
    print(f"Received addition points: {addition_points}")
    projected_image = cap.visual(withContours=True, margin=margin)
    projected_image = cv2.polylines(projected_image, [addition_points], isClosed=True, color=(255,255,0), thickness=3)
    return f"""<img class="img-fluid border border-info border-2" id="objectImage" alt="Capture Image" src="data:image/jpg;base64,{imageToBase64(projected_image)}" style="border-radius: 10px;"  onclick="imageClickListener(event)">"""
    

@harmony.route('/objects/<objectId>/submit_addition', methods=['POST'])
@findObjectIdOr404
def combineObjectWithAddition(cap):
    # margin = getattr(cap, "margin", 5)
    margin = 50
    addition_points = json.loads(request.form["additionPolygon"])
    camName = request.form["camName"]
    cam = app.cc.cameras[camName]
    change = cap.changeSet[camName]
    x, y, w, h = cap.changeSet['0'].clipBox
    adjusted_addition_points = [[pt[0] + x - margin, pt[1] + y - margin] for pt in addition_points]
    adjusted_addition_points = np.int32(adjusted_addition_points)
    height, width = cam.mostRecentFrame.shape[:2]
    blank_image = np.zeros([height, width], np.uint8)
    contour_image = cv2.drawContours(blank_image.copy(), change.changeContours, -1, (255), -1)
    projected_addition_and_contours = cv2.polylines(contour_image, [adjusted_addition_points], isClosed=True, color=(255), thickness=3)
    _, thresh = cv2.threshold(projected_addition_and_contours, 127, 255, 0)
    newChangeContours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cap.changeSet[camName].overrideChangeContours(newChangeContours)
    return f"""<img class="img-fluid border border-info border-2" id="objectImage" alt="Capture Image" src="data:image/jpg;base64,{imageToBase64(cap.visual(withContours=True, margin=margin))}" style="border-radius: 10px;">"""


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


@harmony.route('/undo_last_change')
def undoLastChange():
    app.cm.undoLastChange()
    return "Success"


def setHarmonyApp(newApp):
    global app
    app = newApp


def create_harmony_app():
    global app
    
    app = Flask(__name__)
    app.cc = CalibratedCaptureConfiguration()
    app.cc.capture()
    app.register_blueprint(configurator, url_prefix='/configurator')
    app.register_blueprint(harmony, url_prefix='/harmony')
    resetHarmony()
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

    return app


if __name__ == "__main__":
    create_harmony_app()
    PORT = 7000
    print(f"Launching harmony Server on {PORT}")
    app.run(host="0.0.0.0", port=PORT)

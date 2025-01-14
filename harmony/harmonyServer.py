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
from ipynb.fs.full.HarmonyMachine import HarmonyMachine, HarmonyObject, ObjectAction, GameEvent, Mech, qs, mc


harmony = Blueprint('harmony', __name__, template_folder='harmony_templates')


def gameStateButton(gameState):
    if gameState == "Add":
        button = """<input type="button" class="btn btn-info" name="commitAdditions" id="passive" hx-get="{harmonyURL}commit_additions" hx-target="#objectInteractor" value="Start Game">"""
    elif gameState == "Move":
        button = """<input type="button" class="btn btn-info" name="commitMovement" id="passive" hx-get="{harmonyURL}commit_movement" hx-target="#objectInteractor" value="Commit Movement">"""
    elif gameState == "Declare":
        button = """<input type="button" class="btn btn-info" name="declareActions" id="passive" hx-get="{harmonyURL}declare_actions" hx-target="#objectInteractor" value="Declare Actions">"""
    elif gameState == "Action":
        button = ""
    else:
        print(f"Unknown gameState: {gameState}")
        button = ""
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
        app.cm.GameState.newPhase("Declare")
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/resolve_actions', methods=['POST'])
def resolveActionForm():
    try:
        actionData = {key.split("|||||")[0]: result for key, result in request.form.items()}
    except Exception as e:
        raise Exception(f"Unrecognized form data: {request.form}") from e
    print(f"Received actions: {actionData} -- {request.form}")
    for actor, result in actionData.items():
        try:
            objectActionEvent = GameEvent.get_existing_declarations(actor)[0]
        except Exception as e:
            raise Exception(f"Failed to recover {actor} actions") from e
        with DATA_LOCK:
            GameEvent.set_result(gameEvent=objectActionEvent.meta_anchor, newResult=result)

    unresolved_actions = [ge for ge in GameEvent.get_declared_events() if ge.GameEventResult.terminant in ['null', None] and ge.GameEventType.terminant != "NoAction"]

    if unresolved_actions:
        console_message = f"{len(unresolved_actions)} Unresolved actions"
        with DATA_LOCK:
            global CONSOLE_OUTPUT
            CONSOLE_OUTPUT = console_message
        return buildObjectActionResolver()
    else:
        with DATA_LOCK:
            app.cm.trackMode()
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
        try:
            roundCount = app.cm.GameState.getRoundCount()
        except TypeError:
            roundCount = "N/A"
        consoleImage = cv2.putText(zeros, f'Round: {roundCount:3}-{app.cm.GameState.getPhase()}',
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


def generateGameGraph():
    graph = qs.render()
    yield (b'--frame\r\n'
           b'Content-Type: image/svg\r\n\r\n' + graph.pipe(encoding='utf-8') + b'\r\n')


@harmony.route('/gamegraph')
def game_graph_stream():
    return Response(generateGameGraph(), mimetype='multipart/x-mixed-replace; boundary=frame')



@harmony.route('/reset')
def resetHarmony():
    with DATA_LOCK:
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
        resetHarmony()
    with open("harmony_templates/Harmony.html", "r") as f:
        template = f.read()
    cameraButtons = ' '.join([f'''<input type="button" class="btn btn-info" value="Camera {camName}" onclick="liveCameraClick('{camName}')">''' for camName in app.cc.cameras.keys()])
    cameraButtons = f"""<input type="button" class="btn btn-info" value="Virtual Map" onclick="liveCameraClick(\'VirtualMap\')">{cameraButtons}<input type="button" class="btn btn-info" value="Game Graph" onclick="gameGraph()">"""
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
    # moveDistance = app.cm.lastMoveDistance(capture) -- movement distance this round
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
    if isinstance(capture, HarmonyObject):
        changeRow = changeRow.replace(
            "{armorPlating}", f"{mc.ArmorPlating(capture.oid).terminant - mc.ArmorPlatingDamage(capture.oid).terminant}/{mc.ArmorPlating(capture.oid).terminant}").replace(
            "{armorStructural}", f"{mc.ArmorStructural(capture.oid).terminant - mc.ArmorStructuralDamage(capture.oid).terminant}/{mc.ArmorStructural(capture.oid).terminant}")
    else:
        changeRow = changeRow.replace(
            "{armorPlating}", "N/A").replace(
            "{armorStructural}", "N/A")
        
    return changeRow


def buildObjectTable(filter=None):
    changeRows = []
    if filter == "Event":
        for de in GameEvent.get_declared_events():
            # TODO: Add cancel event button
            changeRows.append(f"""
                <div class="row mb-1 border border-secondary border-2">
                    <h5>{de.GameEventObject.terminant}</h5><h4>{de.GameEventType.terminant}</h4><h5>{de.GameEventValue.terminant}</h5>
                </div>""")
    else:
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
    for objAction in GameEvent.get_declared_events():
        if objAction.GameEventType.terminant in [None, "NoAction"]:
            continue
        value = "" if objAction.GameEventResult.terminant in [None, "null"] else objAction.GameEventResult.terminant
        gameObject = objAction.GameEventObject.terminant
        eventType = objAction.GameEventType.terminant
        gameValue = objAction.GameEventValue.terminant
        table += f"""
            <div class="row">
                <label class="btn btn-outline-primary" for="{gameObject}Resolver">{gameObject} {eventType} {gameValue}</label>
                <input type="number" name="{gameObject}|||||Resolver" max="100" value="{value}"><br>
            </div>"""
    return f"""<div class="row">
                <h2 class="mt-5">Object Action Resolver</h2>
                <div id="objectsSummary" class="text-warning" hx-get="{url_for(".buildHarmony")}objects_summary" hx-trigger="every 1.3s"></div>
            </div>
            <div class="row">
                <div class="container">
                    <form hx-post="{url_for(".buildHarmony")}resolve_actions" hx-target="#objectInteractor">
                        {table}
                        <input type="submit" class="btn btn-danger" value="Commit Actions">
                    </form>
                </div>
            </div>"""


@harmony.route('/objects_summary', methods=['GET'])
def getObjectSummary():
    num_units = len(Mech.entities())
    objects_summary = ""
    match app.cm.GameState.getPhase():
        case "Move":
            num_moved = len(GameEvent.get_declared_events())
            objects_summary = f"{num_moved}/{num_units} Movements Declared"
        case "Declare":
            declared_actions = len(GameEvent.get_declared_events())
            objects_summary = f"{declared_actions}/{num_units} Actions Declared"
        case "Action":
            declared_actions = len(GameEvent.get_declared_events())
            objects_summary = f"{declared_actions} Actions to Resolve"
    
    return f"""<h4>{objects_summary}</h4>"""


def buildObjectsFilter(filter=None):
    if app.cm.GameState.getPhase() == "Action":
        return buildObjectActionResolver()

    assert filter in [None, 'None', 'Terrain', 'Building', 'Unit', 'Event'], f"Unrecognized filter type: {filter}"
    buildObjectsFilter.filter = filter
    noneSelected, terrainSelected, buildingSelected, unitSelected, eventSelected = "", "", "", "", ""
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
    elif filter == 'Event':
        filterQuery = '?filter=Event'
        eventSelected = ' checked="checked"'
        
    template = """
    <div class="row ">
        <h2 class="mt-5">Object Tracker</h2>
        <div id="objectsSummary" class="text-warning" hx-get="{harmonyURL}objects_summary" hx-trigger="every 1.3s"></div>
    </div>
    <div id="objectFilter"class="btn-group" role="group" aria-label="Objects Table Filter Select Buttons">
      <input type="radio" class="btn-check" name="objectFilterradio" id="None" autocomplete="off" {noneSelected} hx-get="{harmonyURL}objects_filter" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="None">None</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Terrain" autocomplete="off" {terrainSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Terrain" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Terrain">Terrain</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Building" autocomplete="off" {buildingSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Building" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Building">Building</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Unit" autocomplete="off" {unitSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Unit" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Unit">Unit</label>
      <input type="radio" class="btn-check" name="objectFilterradio" id="Event" autocomplete="off" {eventSelected} hx-get="{harmonyURL}objects_filter?objectFilter=Event" hx-target="#objectInteractor">
      <label class="btn btn-outline-primary" for="Event">Event</label>
    </div>
    <div id="objectsTable" hx-get="{harmonyURL}objects{filter}" hx-trigger="every 1s">{objectRows}</div>"""
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{filter}", filterQuery).replace(
        "{noneSelected}", noneSelected).replace(
        "{terrainSelected}", terrainSelected).replace(
        "{buildingSelected}", buildingSelected).replace(
        "{unitSelected}", unitSelected).replace(
        "{objectRows}", buildObjectTable(filter))


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
        except KeyError as ke:
            error = f"{objectId} Not found"
            print(error)
            return error, 404
        return objectId_endpoint(cap=app.cm.findObject(objectId=objectId), **kwargs)
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
    objectKwargs = request.form.to_dict()
    newName = objectKwargs.pop("objectName")

    with DATA_LOCK:
        objectType = objectKwargs.pop('objectType')
        objectSubType = objectKwargs.pop('objectSubType')
        cap = app.cm.classifyObject(
            objectId=cap.oid,
            objectType=objectType,
            objectSubType=objectSubType,
            objectKwargs=objectKwargs)
    
        if newName != cap.oid:
            cap.rename(newName)

    print(f"{cap.oid} created!!!")
    return buildObjectsFilter()
    
    
@harmony.route('/objects/<objectId>', methods=['DELETE'])
@findObjectIdOr404
def deleteObjectSettings(cap):
    with DATA_LOCK:
        app.cm.deleteObject(cap.oid)
    return buildObjectsFilter()
    
    
@harmony.route('/objects/<objectId>/declare_attack/<targetId>', methods=['POST'])
@findObjectIdOr404
def declareAttackOnTarget(cap, targetId):
    with DATA_LOCK:
        app.cm.declareEvent(eventType="Attack", eventFaction="Undeclared", eventObject=cap.oid, eventValue=targetId, eventResult="null")
    return buildObjectActions(cap)
    
    
@harmony.route('/objects/<objectId>/declare_no_action', methods=['POST'])
@findObjectIdOr404
def declareNoAction(cap):
    with DATA_LOCK:
        app.cm.declareEvent(eventType="NoAction", eventFaction="Undeclared", eventObject=cap.oid, eventValue="null", eventResult="null")
    return buildObjectActions(cap)


def buildObjectSettings(obj):
    objType = getattr(obj, 'objectType', 'None')
    terrainSelected, buildingSelected, unitSelected = '', '', ''

    text_box_template = """<label for="{key}">{key}</label><input type="text" class="form-control" name="{key}" value="{value}">"""

    objectSettings = []
    if objType == "Terrain":
        objectSettings.append(text_box_template.format(key="Rating", value=1))
        objectSettings.append(text_box_template.format(key="Elevation", value=5))
        terrainSelected = " selected='selected'"
    elif objType == "Building":
        objectSettings.append(text_box_template.format(key="Elevation", value=5))
        
        structureTypes = """<select name="objectSubType" id="objectSubType">"""
        for structureType in HarmonyObject.objectFactories['Structure'].keys(): 
            structureTypes += f"""<option value="{structureType}">{structureType}</option>"""
        structureTypes += "</select>"
        objectSettings.append(structureTypes)
        
        buildingSelected = " selected='selected'"
    elif objType == "Unit":
        unitTypes = """<select name="objectSubType" id="objectSubType">"""
        for unitType in HarmonyObject.objectFactories['Unit'].keys(): 
            unitTypes += f"""<option value="{unitType}">{unitType}</option>"""
        unitTypes += "</select>"
        objectSettings.append(unitTypes)
        objectSettings.append(text_box_template.format(key="Skill", value=4))

        unitSelected = " selected='selected'"
    else:
        settings = {}

    objectSettings = "<br>".join(objectSettings)

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
    try:
        capDeclaredAction = GameEvent.get_existing_declarations(cap.oid)[0]
        capDeclaredTarget = capDeclaredAction.GameEventValue.terminant
        if capDeclaredTarget is not None:
            capDeclaredTarget = app.cm.findObject(capDeclaredTarget)
    except IndexError:
        capDeclaredAction = None
        capDeclaredTarget = None
    for target in app.cm.memory:
        if target.oid == cap.oid:
            continue
        else:
            declare = ""
            action = ObjectAction(cap, target)
            selected = capDeclaredAction is not None and capDeclaredTarget == target
            if app.cm.GameState.getPhase() == "Declare":
                disabled = "disabled" if selected else ""
                declare = f"""
                <div class="col">
                    <input {disabled} type="button" class="btn btn-warning" value="Declare Attack" id="declare_{target.oid}" hx-target="#objectInteractor" hx-post="{url_for(".buildHarmony")}objects/{cap.oid}/declare_attack/{target.oid}">
                </div>"""
            objActCards.append(cardTemplate.replace(
                "{harmonyURL}", url_for(".buildHarmony")).replace(
                "{cardBorder}", "" if not selected else "border border-secondary border-3").replace(
                "{objectName}", cap.oid).replace(
                "{targetName}", target.oid).replace(
                "{encodedBA}", imageToBase64(target.visual())).replace(
                "{objectDistance}", f"{action.targetRange.capitalize()} ({action.targetDistance / 25.4:6.1f} in)").replace(
                "{declare}", declare).replace(
                "{skill}", str(action.skill)).replace(
                "{attackerMovementModifier}", str(action.aMM)).replace(
                "{targetMovementModifier}", str(action.tMM)).replace(
                "{range}", str(action.rangeModifier)).replace(
                "{other}", "0").replace(
                "{targetNumber}", str(action.targetNumber)))

    if app.cm.GameState.getPhase() == "Declare":
        selected = capDeclaredAction is not None and capDeclaredTarget is None
        disabled = "disabled" if selected else ""
        objActCards.append(f"""
        <div class="row mb-1 {"border border-secondary border-5" if selected else ""}">
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
    with DATA_LOCK:
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
    app.cm = HarmonyMachine(app.cc)
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

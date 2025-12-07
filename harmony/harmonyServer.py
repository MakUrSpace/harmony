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
import time

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, Blueprint, render_template, Response, request, make_response, redirect, url_for

from observer.configurator import configurator, setConfiguratorApp
from observer.observerServer import CalibratedCaptureConfiguration, observer, configurator, registerCaptureService, setConfiguratorApp, setObserverApp
from observer.calibrator import calibrator, CalibratedCaptureConfiguration, registerCaptureService, DATA_LOCK, CONSOLE_OUTPUT
from ipynb.fs.full.HarmonyMachine import HarmonyMachine, HarmonyObject, ObjectAction, mc, INCHES_TO_MM


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
    with DATA_LOCK:
        controllerHTML =  gameStateButton(app.cm.getPhase())
    return controllerHTML


@harmony.route('/get_undo_button')
def getUndoButton():
    with DATA_LOCK:
        if app.cm.getPhase() in ["Add", "Move"]:
            undoHTML = f"""<input type="button" class="btn btn-danger float-end" name="undoChange" id="passive" hx-get="{url_for(".buildHarmony")}undo_last_change" hx-swap="none" value="Undo Last Change">"""
        else:
            undoHTML = ""
    return undoHTML


@harmony.route('/commit_additions')
def commitAdditions():
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        app.cm.commitAdditions()
        CONSOLE_OUTPUT = ""
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/commit_movement', methods=['GET'])
def commitMovement():
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        try:
            app.cm.commitMovement()
            CONSOLE_OUTPUT = ""
        except Exception as e:
            CONSOLE_OUTPUT = str(e)
            raise
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/declare_actions', methods=['GET'])
def declareActions():
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        app.cm.passiveMode()
        app.cm.GameState.newPhase("Declare")
        CONSOLE_OUTPUT = ""
    return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))


@harmony.route('/resolve_actions', methods=['POST'])
def resolveActionForm():
    global CONSOLE_OUTPUT
    try:
        actionData = {key.split("|||||")[0]: result for key, result in request.form.items()}
    except Exception as e:
        raise Exception(f"Unrecognized form data: {request.form}") from e
    print(f"Received actions: {actionData} -- {request.form}")
    for actor, result in actionData.items():
        try:
            objectActionEvent = app.cm.GameEvents.get_existing_declarations(actor)[0]
        except Exception as e:
            raise Exception(f"Failed to recover {actor} actions") from e
        with DATA_LOCK:
            app.cm.GameEvents.set_result(gameEvent=objectActionEvent.entity, newResult=result)

    unresolved_actions = [ge for ge in app.cm.GameEvents.get_declared_events() if ge.GameEventResult.terminant in ['null', None] and ge.GameEventType.terminant != "NoAction"]

    if unresolved_actions:
        console_message = f"{len(unresolved_actions)} Unresolved actions"
        with DATA_LOCK:
            global CONSOLE_OUTPUT
            CONSOLE_OUTPUT = console_message
        return buildObjectActionResolver()
    else:
        with DATA_LOCK:
            app.cm.resolveRound()
            CONSOLE_OUTPUT = ""
        return buildObjectsFilter(getattr(buildObjectsFilter, "filter", None))
    

def imageToBase64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()


def renderConsole():
    while True:
        cam = list(app.cc.cameras.values())[0]
        shape = (200, 600)
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
        with DATA_LOCK:
            try:
                roundCount = app.cm.getRoundCount()
            except TypeError:
                roundCount = "N/A"
            consoleImage = cv2.putText(zeros, f'Round: {roundCount:3}-{app.cm.getPhase()}',
                (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

        ret, consoleImage = cv2.imencode('.jpg', zeros)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + consoleImage.tobytes() + b'\r\n')
        time.sleep(0.1)


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

        with DATA_LOCK:
            camImage = app.cm.getCameraImagesWithChanges(cameraKeys=[camName])[camName]
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
        with DATA_LOCK:
            camImages = []
            if app.cm.lastChanges is not None and not app.cm.lastChanges.empty:
                print("Has changes")
            if app.cm.lastClassification is not None:
                print("Has class")
            camImages = app.cm.getCameraImagesWithChanges(app.cc.cameras.keys()).values()
            camImage = vStackImages(camImages)
            camImage = cv2.resize(camImage, [480, 640], interpolation=cv2.INTER_AREA)
            ret, camImage = cv2.imencode('.jpg', camImage)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + camImage.tobytes() + b'\r\n')
        time.sleep(0.1)


@harmony.route('/combinedCamerasWithChanges')
def combinedCamerasWithChangesResponse():
    return Response(genCombinedCameraWithChangesView(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generateGameGraph():
    yield (b'--frame\r\n'
           b'Content-Type: image/svg\r\n\r\n' + qs.render().pipe(encoding='utf-8') + b'\r\n')


@harmony.route('/gamegraph')
def game_graph_stream():
    return Response(generateGameGraph(), mimetype='multipart/x-mixed-replace; boundary=frame')


@harmony.route('/reset')
def resetHarmony():
    with DATA_LOCK:
        app.cm = HarmonyMachine(app.cc)
        app.cm.reset()
        CONSOLE_OUTPUT = "Harmony reset."
    return 'success'


@harmony.route('/load', methods=["POST"])
def loadHarmony():
    game_name = request.form["game_name"]
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        CONSOLE_OUTPUT = "Loading game state..."
    time.sleep(3)
    with DATA_LOCK:
        app.cm.loadGame(gameName=game_name)        
        CONSOLE_OUTPUT = "Game state reloaded."
    return 'success'


@harmony.route('/save', methods=["POST"])
def saveHarmony():
    game_name = request.form["game_name"]
    global CONSOLE_OUTPUT
    with DATA_LOCK:
        app.cm.saveGame(gameName=game_name)
        CONSOLE_OUTPUT = "Saved game state."
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
    cameraButtons = f"""<input type="button" class="btn btn-info" value="Virtual Map" onclick="liveCameraClick(\'VirtualMap\')">{cameraButtons}"""
    # cameraButtons += """<input type="button" class="btn btn-info" value="Game Graph" onclick="gameGraph()">"""
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
    name = "None"
    objType = "None"
    center = ", ".join(["0", "0"])
    health = ""
    numHits = 0
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "

    objectName = capture.oid
    objectActions = ""
    objectMovement = ""
    moveDistance = "None"
    borderType = "border-secondary border-2"
    editObject = ""
    armorPlating = "N/A"
    armorStructural = "N/A"
    rowClass = ""
    with open("harmony_templates/TrackedObjectRow.html") as f:
        changeRowTemplate = f.read()

    encodedBA = imageToBase64(app.cm.object_visual(capture, withContours=False))

    gamePhase = app.cm.getPhase()
    
    if isinstance(capture, HarmonyObject):
        if getattr(capture, 'objectType', None) == "Unit":
            moveDistance = app.cm.objectLastDistance(capture)
            movementSpeed = f"{app.cm.objectSpeed(capture.oid):4.1f}{app.cm.objectJumpJets(capture.oid) or ''}"
            moveDistance = f"0 /{movementSpeed} in" if moveDistance is None or moveDistance < 0.3 else f"{moveDistance:4.1f} /{movementSpeed} in"
        
            if app.cm.obj_destroyed(capture.oid):
                objectName = f"<s>{capture.oid}</s>"
                borderType = "border-danger border-5 x-box"
            else:
                if app.cm.objectCouldInteract(capture):
                    borderType = "border-success border-3"
                    interactiveStyle = True
                elif app.cm.objectViolatingRules(capture):
                    borderType = "border-danger border-3"
                    interactiveStyle = True
                else:
                    interactiveStyle = False
                objectActions = f"""<div class="col"><button class="btn btn-{'success' if gamePhase == 'Declare' and interactiveStyle else 'primary'}" style="width:100%" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}/actions">Actions</button></div>"""
                objectMovement = f"""<div class="col"><button class="btn btn-{'success' if gamePhase == 'Move' and interactiveStyle else 'primary'}" style="width:100%" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}/movement">Movement</button></div>"""
    
        if getattr(capture, 'objectType', None) in ["Unit", "Structure"]:
            armorPlating = f"{mc.ArmorPlating(capture.oid).terminant - mc.ArmorPlatingDamage(capture.oid).terminant}/{mc.ArmorPlating(capture.oid).terminant}"
            armorStructural = f"{mc.ArmorStructural(capture.oid).terminant - mc.ArmorStructuralDamage(capture.oid).terminant}/{mc.ArmorStructural(capture.oid).terminant}"
        
    if gamePhase == "Add":
        editObject = f"""<div class="col"><button class="btn btn-info" style="width:100%" hx-target="#objectInteractor" hx-get="{url_for(".buildHarmony")}objects/{capture.oid}">Edit</button></div>"""
    
    changeRow = changeRowTemplate.replace(
        "{borderType}", borderType).replace(
        "{objectName}", objectName).replace(
        "{realCenter}", ", ".join([f"{dim:6.0f}" for dim in app.cm.captureRealCoord(capture)])).replace(
        "{moveDistance}", moveDistance).replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{encodedBA}", encodedBA).replace(
        "{actions}", objectActions).replace(
        "{movement}", objectMovement).replace(
        "{edit}", editObject).replace(
        "{armorPlating}", armorPlating).replace(
        "{armorStructural}", armorStructural)

    return changeRow


def captureToActionRow(capture):
    name = "None"
    objType = "None"
    center = ", ".join(["0", "0"])
    health = ""
    numHits = 0
    for i in range(numHits):
        health += "[x] "
    for i in range(3 - numHits):
        health += "[ ] "

    objectName = capture.oid
    objectActions = ""
    objectMovement = ""
    moveDistance = "None"
    borderType = "border-secondary border-2"
    editObject = ""
    armorPlating = "N/A"
    armorStructural = "N/A"
    rowClass = ""

    gamePhase = app.cm.getPhase()

    with open("harmony_templates/TrackedObjectActionRow.html") as f:
        actionRowTemplate = f.read()

    with open("harmony_templates/ObjectActionCard.html") as f:
        cardTemplate = f.read()

    objectVisual = imageToBase64(app.cm.object_visual(capture, withContours=False))

    objectActions = []
    if app.cm.object_type(capture.oid) == "Unit" and not app.cm.obj_destroyed(capture.oid):
        visibilityMap = f"""<img class="img-fluid border border-info border-2" alt="Visibility Map" src="data:image/jpg;base64,{imageToBase64(app.cm.buildVisibilityMap(capture))}" style="border-radius: 10px;">"""
        try:
            capDeclaredAction = app.cm.GameEvents.get_existing_declarations(capture.oid)[0]
            capDeclaredTarget = capDeclaredAction.GameEventValue.terminant
            if capDeclaredTarget == "null":
                capDeclaredTarget = None
            if capDeclaredTarget is not None:
                capDeclaredTarget = app.cm.findObject(capDeclaredTarget)
        except IndexError:
            print(f"Failed to locate existing action declarations on {capture.oid}")
            capDeclaredAction = None
            capDeclaredTarget = None

        for target in app.cm.memory:
            if not app.cm.can_target(capture, target):
                continue
    
            declare = ""
            action = ObjectAction(capture, target)
            selected = capDeclaredAction is not None and capDeclaredTarget == target
            if gamePhase == "Declare":
                disabled = "disabled" if selected else ""
                declare = f"""
                <div class="col">
                    <input {disabled} type="button" class="btn btn-warning" value="Declare Attack" id="declare_{target.oid}" hx-target="#objectInteractor" hx-post="{url_for(".buildHarmony")}objects/{capture.oid}/declare_attack/{target.oid}/object_table">
                </div>"""
                
            objectActions.append(cardTemplate.replace(
                "{harmonyURL}", url_for(".buildHarmony")).replace(
                "{cardBorder}", "" if not selected else "border border-secondary border-3").replace(
                "{objectName}", capture.oid).replace(
                "{targetName}", target.oid).replace(
                "{targetVisual}", imageToBase64(target.icon)).replace(
                "{objectDistance}", f"{action.targetRange.capitalize()} ({action.targetDistance / 25.4:6.1f} in )").replace(
                "{declare}", declare).replace(
                "{skill}", str(action.skill)).replace(
                "{attackerMovementModifier}", str(action.aMM)).replace(
                "{targetMovementModifier}", str(action.tMM)).replace(
                "{rangeModifier}", str(action.rangeModifier)).replace(
                "{other}", str(action.otherModifiers)).replace(
                "{targetNumber}", str(action.targetNumber)))
    
        if gamePhase == "Declare":
            selected = capDeclaredAction is not None and capDeclaredTarget is None
            disabled = "disabled" if selected else ""
            objectActions.append(f"""
            <div class="row mb-1 {"border border-secondary border-5" if selected else ""}">
                <input {disabled} type="button" class="btn btn-danger" value="Take No Action" id="no_action" hx-target="#objectInteractor" hx-post="{url_for(".buildHarmony")}objects/{capture.oid}/declare_no_action/object_table">
            </div>
            """.replace("{harmonyURL}", url_for(".buildHarmony")).replace("{objectName}", capture.oid))
    else:
        visibilityMap = ""
        
    
    actionRow = actionRowTemplate.replace(
        "{borderType}", borderType).replace(
        "{objectName}", objectName).replace(
        "{objectVisual}", objectVisual).replace(
        "{visibilityMap}", visibilityMap).replace(
        "{objectActions}", "\n".join(objectActions))

    return actionRow
    


def buildObjectTable(faction_filter=None, type_filter=None):
    changeRows = []

    if type_filter == "Event":
        for de in app.cm.GameEvents.get_declared_events():
            # TODO: Add cancel event button
            changeRows.append(f"""
                <div class="row mb-1 border border-secondary border-2">
                    <h5>{de.GameEventObject.terminant}</h5><h4>{de.GameEventType.terminant}</h4><h5>{de.GameEventValue.terminant}</h5>
                </div>""")
    else:
        print(f"Structure object table with filters: Faction: {faction_filter} -- Type: {type_filter}")
        gamePhase = app.cm.getPhase()
        rowRenderer = captureToChangeRow if gamePhase in ["Add", "Move"] else captureToActionRow
        for capture in app.cm.memory:
            print(f"Cap {capture.oid} -- Faction: {app.cm.faction(capture.oid)} -- Type: {app.cm.object_type(capture.oid)}")
            if gamePhase == "Declare" and app.cm.obj_destroyed(capture.oid):
                continue
            if ((faction_filter is not None and faction_filter != "All" and app.cm.faction(capture.oid) != faction_filter)
                or (type_filter is not None and type_filter != "All" and app.cm.object_type(capture.oid) != type_filter)
            ):
                continue
            changeRows.append(rowRenderer(capture))
    return " ".join(changeRows)


@harmony.route('/objects', methods=['GET'])
def getObjectTable():
    return buildObjectTable(
        request.args.get('faction_filter', None),
        request.args.get('type_filter', None))


def buildObjectActionResolver():
    table = f"""
        <div class="row border-bottom border-info border-3">
            <div class="col"><h3>Actor</h3></div>
            <div class="col"><h3>Target</h3></div>
            <div class="col-6"><h3>Attack Roll</h3></div>
        </div>"""
    for objAction in app.cm.GameEvents.get_declared_events():
        if objAction.GameEventType.terminant in [None, "NoAction"]:
            continue
        actionResult = "" if objAction.GameEventResult.terminant in [None, "null"] else objAction.GameEventResult.terminant
        gameObject = objAction.GameEventObject.terminant
        eventType = objAction.GameEventType.terminant
        gameValue = objAction.GameEventValue.terminant
        actionTarget = objAction.GameEventTarget.terminant
        actor = app.cm.findObject(gameObject)
        target = app.cm.findObject(gameValue)

        actorIcon = imageToBase64(actor.icon)
        targetIcon = imageToBase64(target.icon)
        
        
        table += f"""
            <div class="row border-bottom border-primary border-3">
                <div class="col">
                    <h4>{gameObject}</h4>
                    <img class="img-fluid border border-primary border-2" alt="Actor Image" src="data:image/jpg;base64,{actorIcon}" style="border-radius: 10px">
                </div>
                <div class="col">
                    <h4>{gameValue}</h4>
                    <img class="img-fluid border border-danger border-2" alt="Target Image" src="data:image/jpg;base64,{targetIcon}" style="border-radius: 10px">
                </div>
                <div class="col-6">
                    <label for="{gameObject}Resolver">Attack Roll: 
                    <input type="number" name="{gameObject}|||||Resolver" max="12" style="max-width:50px" value="{actionResult}">/ {actionTarget}</label><br>
                </div>
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
    with DATA_LOCK:
        num_units = len(app.cm.units())
        active_units = len(app.cm.active_units())
        destroyed = num_units - active_units
        objects_summary = ""
        match app.cm.getPhase():
            case "Move":
                num_moved = len(app.cm.unitsMovedThisRound())
                objects_summary = f"{num_moved}/{active_units} Movements Declared"
            case "Declare":
                declared_actions = len(app.cm.unitsDeclaredThisRound())
                objects_summary = f"{declared_actions}/{active_units} Actions Declared"
            case "Action":
                declared_actions = len(app.cm.unitsDeclaredThisRound())
                no_actions = len(app.cm.GameEvents.get_declared_no_action_events())
                objects_summary = f"{declared_actions - no_actions} Actions to Resolve"
        if destroyed > 0:
            objects_summary += f". {destroyed} Destroyed"
        
    return f"""<h4>{objects_summary}</h4>"""


def buildObjectsFilter(faction_filter=None, type_filter=None):
    with DATA_LOCK:
        gamePhase = app.cm.getPhase()
        if gamePhase == "Action":
            return buildObjectActionResolver()

    factions = app.cm.factions()
    assert faction_filter in [None, 'All', *factions], f"Unrecognized faction: {faction_filter}"
    assert type_filter in [None, 'All', 'Terrain', 'Structure', 'Unit', 'Event'], f"Unrecognized object type: {type_filter}"
    faction_filter = faction_filter or getattr(buildObjectsFilter, "faction_filter", None)
    type_filter = type_filter or getattr(buildObjectsFilter, "type_filter", None)
    with DATA_LOCK:
        setattr(buildObjectsFilter, 'faction_filter', faction_filter)
        setattr(buildObjectsFilter, 'type_filter', type_filter)
    
    allFactionsSelected = ' checked="checked"' if faction_filter == "All" or faction_filter is None else ''

    factionButtons = {}
    filters = []
    factionButtonTemplate = """
        <input type="radio" class="btn-check" name="factionFilterradio" id="{faction}Radio" autocomplete="off"{checked} hx-get="{harmonyURL}objects_filter{param}" hx-target="#objectInteractor">
        <label class="btn btn-outline-secondary" for="{faction}Radio">{faction}</label>"""
    for faction in factions:
        if faction_filter == faction:
            checked = ' checked ="checked"'
            filters.append(f'faction_filter={faction}')
        else:
            checked = ''
        factionButtons[faction] = factionButtonTemplate.replace(
            "{faction}", faction).replace(
            "{param}", f"?faction_filter={faction}").replace(
            "{checked}", checked).replace(
            "{harmonyURL}", url_for(".buildHarmony"))

    allTypesSelected, terrainSelected, structureSelected, unitSelected, eventSelected = "", "", "", "", ""
    if type_filter == 'All' or type_filter is None:
        allTypesSelected = ' checked="checked"'
    elif type_filter == 'Terrain':
        filters.append('type_filter=Terrain')
        terrainSelected = ' checked="checked"'
    elif type_filter == 'Structure':
        filters.append('type_filter=Structure')
        structureSelected = ' checked="checked"'
    elif type_filter == 'Unit':
        filters.append('type_filter=Unit')
        unitSelected = ' checked="checked"'
    elif type_filter == 'Event':
        filters.append('type_filter=Event')
        eventSelected = ' checked="checked"'

    filters = f"?{'&'.join(filters)}" if len(filters) > 0 else ''
    
    with open("harmony_templates/FilteredObjectsTable.html", "r") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{allFactionsSelected}", allFactionsSelected).replace(
        "{factionFilters}", "".join(factionButtons.values())).replace(
        "{allTypesSelected}", allTypesSelected).replace(
        "{terrainSelected}", terrainSelected).replace(
        "{structureSelected}", structureSelected).replace(
        "{unitSelected}", unitSelected).replace(
        "{eventSelected}", eventSelected).replace(
        "{filter}", filters).replace(
        "{objectRows}", buildObjectTable(faction_filter, type_filter))


def getInteractor():
    with DATA_LOCK:
        gameState = app.cm.getPhase()
    if gameState == "Resolve":
        return buildObjectActionResolver()
    else:
        return buildObjectsFilter()


@harmony.route('/objects_filter', methods=['GET'])
def getObjectTableContainer():
    with DATA_LOCK:
        gameState = app.cm.getPhase()
    if gameState != "Resolve":
        return buildObjectsFilter(
            faction_filter=request.args.get('faction_filter', None),
            type_filter=request.args.get('type_filter', None))
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
        "{encodedBA}", imageToBase64(app.cm.object_visual(cap, withContours=footprint_enabled)))


@harmony.route('/objects/<objectId>', methods=['POST'])
@findObjectIdOr404
def updateObjectSettings(cap):
    if app.cm.getPhase() != "Add":
        return "Cannot update object settings outside of Add Phase", 400
    objectKwargs = request.form.to_dict()
    newName = objectKwargs.pop("objectName")

    with DATA_LOCK:
        objectType = objectKwargs.pop('objectType')
        objectSubType = objectKwargs.pop('objectSubType')
        if 'elevation' in objectKwargs:
            objectKwargs['elevation'] = int(float(elevation) * INCHES_TO_MM)
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
    if app.cm.getPhase() != "Add":
        return "Cannot remove object outside of Add Phase", 400
    with DATA_LOCK:
        app.cm.deleteObject(cap.oid)
    return buildObjectsFilter()
    
    
@harmony.route('/objects/<objectId>/declare_attack/<targetId>', methods=['POST'])
@findObjectIdOr404
def declareAttackOnTarget(cap, targetId):
    with DATA_LOCK:
        try:
            target = app.cm.findObject(targetId)
            action = ObjectAction(cap, target)
            app.cm.declareEvent(eventType="Attack", eventFaction=app.cm.faction(cap.oid), eventObject=cap.oid, eventValue=targetId, eventTarget=action.targetNumber, eventResult="null")
        except Exception as e:
            print(f"Failed to declare attack: {cap.oid} on {targetId} -- {e}")
    return buildObjectActions(cap)
    
    
@harmony.route('/objects/<objectId>/declare_attack/<targetId>/object_table', methods=['POST'])
@findObjectIdOr404
def declareAttackOnTargetReturnTable(cap, targetId):
    with DATA_LOCK:
        try:
            target = app.cm.findObject(targetId)
            action = ObjectAction(cap, target)
            app.cm.declareEvent(eventType="Attack", eventFaction=app.cm.faction(cap.oid), eventObject=cap.oid, eventValue=targetId, eventTarget=action.targetNumber, eventResult="null")
        except Exception as e:
            print(f"Failed to declare attack: {cap.oid} on {targetId} -- {e}")
    return buildObjectsFilter()


@harmony.route('/objects/<objectId>/declare_no_action', methods=['POST'])
@findObjectIdOr404
def declareNoAction(cap):
    with DATA_LOCK:
        app.cm.declareEvent(eventType="NoAction", eventFaction=app.cm.faction(cap.oid), eventObject=cap.oid, eventValue="null", eventTarget=0, eventResult="null")
    return buildObjectActions(cap)


@harmony.route('/objects/<objectId>/declare_no_action/object_table', methods=['POST'])
@findObjectIdOr404
def declareNoActionReturnTable(cap):
    with DATA_LOCK:
        app.cm.declareEvent(eventType="NoAction", eventFaction=app.cm.faction(cap.oid), eventObject=cap.oid, eventValue="null", eventTarget=0, eventResult="null")
    return buildObjectsFilter()


def buildObjectSettings(obj, objType=None):
    if objType is None:
        objType = app.cm.object_type(obj.oid)
    terrainSelected, structureSelected, unitSelected = '', '', ''

    text_box_template = """<label for="{key}">{key}</label><input type="text"{datalist} class="form-control" name="{key}" value="{value}">"""

    objectSettings = []
    if objType == "Terrain":
        objectSettings.append(text_box_template.format(key="Elevation", value=app.cm.objectElevation(obj), datalist=""))
        objectSettings.append(text_box_template.format(key="Difficulty", value=1, datalist=""))
        objectSettings.append("""<input type="text" hidden class="form-control" name="objectSubType" value="UniformElevation">""")
        terrainSelected = " selected='selected'"
    elif objType == "Structure":
        objectSettings.append(text_box_template.format(key="Elevation", value=1, datalist=""))
        
        structureTypes = """<select name="objectSubType" id="objectSubType">"""
        for structureType in HarmonyObject.objectFactories['Structure'].keys(): 
            structureTypes += f"""<option value="{structureType}">{structureType}</option>"""
        structureTypes += "</select>"
        objectSettings.append(text_box_template.format(key="Faction", value="Unaligned", datalist=""))
        objectSettings.append(structureTypes)
        
        structureSelected = " selected='selected'"
    elif objType == "Unit":
        unitTypes = """<select name="objectSubType" id="objectSubType">"""
        for unitType in HarmonyObject.objectFactories['Unit'].keys(): 
            unitTypes += f"""<option value="{unitType}">{unitType}</option>"""
        unitTypes += "</select>"
        objectSettings.append(unitTypes)

        object_faction = app.cm.faction(obj.oid) or "Unaligned"
        faction_options = [f'<option value="{faction}">{faction}</option>' for faction in app.cm.factions()]
        factionInput = text_box_template + """
        <datalist id="existing_factions">
            {faction_options}
        </datalist>"""
        
        objectSettings.append(factionInput.format(
            key="Faction",
            value=object_faction,
            datalist=' datalist="existing_factions"',
            faction_options="\n".join(faction_options)))
            
        mechSkill = app.cm.mech_skill(obj.oid) or 4
        objectSettings.append(text_box_template.format(key="Skill", value=mechSkill, datalist=""))

        unitSelected = " selected='selected'"
    else:
        settings = {}

    objectSettings = "<br>".join(objectSettings)

    return """
    <select name="objectType" id="objectType" hx-target="#objectSettings" hx-post='{harmonyURL}objects/{objectName}/type'>
      <option value="None">None</option>
      <option value="Terrain" {terrainSelected}>Terrain</option>
      <option value="Structure" {structureSelected}>Structure</option>
      <option value="Unit" {unitSelected}>Unit</option>
    </select><br>
    {objectSettings}
    """.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", obj.oid).replace(
        "{objectSettings}", objectSettings).replace(
        "{terrainSelected}", terrainSelected).replace(
        "{structureSelected}", structureSelected).replace(
        "{unitSelected}", unitSelected)


@harmony.route('/objects/<objectId>/settings', methods=['GET'])
@findObjectIdOr404
def getObjectSettings(cap):
    return buildObjectSettings(cap)


@harmony.route('/objects/<objectId>/type', methods=['POST'])
@findObjectIdOr404
def updateObjectType(cap):
    newType = request.form["objectType"]
    assert newType in ["None", "Terrain", "Structure", "Unit"], f"Unrecognized object type: {newType}"
    return buildObjectSettings(cap, objType=newType)    


def buildObjectActions(cap):
    with open("harmony_templates/ObjectActionCard.html") as f:
        cardTemplate = f.read()
    objActCards = []
    with DATA_LOCK:
        try:
            if app.cm.getPhase() == "Declare":
                capDeclaredAction = app.cm.GameEvents.get_existing_declarations(cap.oid)[0]
                capDeclaredTarget = capDeclaredAction.GameEventValue.terminant
            else:
                capDeclaredAction = None
                capDeclaredTarget = None
                
            if capDeclaredTarget == "null":
                capDeclaredTarget = None
            if capDeclaredTarget is not None:
                capDeclaredTarget = app.cm.findObject(capDeclaredTarget)
        except IndexError:
            print(f"Failed to locate existing action declarations on {cap.oid}")
            capDeclaredAction = None
            capDeclaredTarget = None
        cap_faction = app.cm.faction(cap.oid)
        for target in app.cm.memory:
            if not app.cm.can_target(cap, target) or app.cm.obj_destroyed(target.oid):
                continue

            declare = ""
            action = ObjectAction(cap, target)
            selected = capDeclaredAction is not None and capDeclaredTarget == target
            if app.cm.getPhase() == "Declare":
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
                "{targetVisual}", imageToBase64(target.icon)).replace(
                "{objectDistance}", f"{action.targetRange.capitalize()} ({action.targetDistance / 25.4:6.1f} in )").replace(
                "{declare}", declare).replace(
                "{skill}", str(action.skill)).replace(
                "{attackerMovementModifier}", str(action.aMM)).replace(
                "{targetMovementModifier}", str(action.tMM)).replace(
                "{range}", action.targetRange).replace(
                "{rangeModifier}", str(action.rangeModifier)).replace(
                "{other}", str(action.otherModifiers)).replace(
                "{targetNumber}", str(action.targetNumber)))
    
        if app.cm.getPhase() == "Declare":
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
        "{visibilityMap}", imageToBase64(app.cm.buildVisibilityMap(cap))).replace(
        "{objectActionCards}", "\n".join(objActCards))


@harmony.route('/objects/<objectId>/actions', methods=['GET'])
@findObjectIdOr404
def getObjectActions(cap):
    return buildObjectActions(cap)


def buildObjectMovement(cap):
    with open("harmony_templates/HarmonyObjectMovement.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{encodedBA}", imageToBase64(app.cm.buildMovementMap(cap))).replace(
        "{newLocation}", json.dumps(getattr(cap, "newLocation", "[]")))
    

@harmony.route('/objects/<objectId>/movement', methods=['GET'])
@findObjectIdOr404
def getObjectMovement(cap):
    with DATA_LOCK:
        return buildObjectMovement(cap)


@harmony.route('/objects/<objectId>/movement', methods=['POST'])
@findObjectIdOr404
def setMovementLocation(cap):
    newLocation = json.loads(request.form["newLocation"])
    movementMap = app.cm.buildMovementMap(cap)
    if newLocation:
        cap.newLocation = newLocation
        cv2.circle(movementMap, [int(d) for d in newLocation], 10, (255, 255, 0), -1)
    return f"""<div id="objectImageDiv">
            <img class="img-fluid border border-info border-2" alt="Object Movement Display" id="movementMap" src="data:image/jpg;base64,{imageToBase64(movementMap)}" onclick="movementImageClickListener(event)" style="border-radius: 10px; width:100%; height:100%; object-fit:contain;">
        </div>"""


@harmony.route('/objects/<objectId>/request_movement', methods=['POST'])
@findObjectIdOr404
def requestObjectMovement(cap):
    global CONSOLE_OUTPUT
    requestedLocation = [int(d) for d in json.loads(request.form["newLocation"])]
    x, y, w, h = app.cm.cc.realSpaceBoundingBox()
    with DATA_LOCK:
        cap.newLocation = [requestedLocation[0] + x, requestedLocation[1] + y]
        print(f"Requesting {cap.oid} move to {cap.newLocation}")
        CONSOLE_OUTPUT = f"RQ-{cap.oid} move to {cap.newLocation}"
        app.cm.expectObjectMovement(cap, cap.newLocation)
    return getObjectTableContainer()
    

@harmony.route('/objects/<objectId>/visibility', methods=['GET'])
@findObjectIdOr404
def getObjectVisibility(cap):
    with open("harmony_templates/HarmonyObjectMovement.html") as f:
        template = f.read()
    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{encodedBA}", imageToBase64(app.cm.buildVisibilityMap(cap))).replace(
        "{newLocation}", getattr(cap, "newLocation", ""))


def buildFootprintEditor(cap):
    with open("harmony_templates/TrackedObjectFootprintEditor.html") as f:
        template = f.read()

    if getattr(cap, "selection_polygon", None) is not None:
        selectionPolygon = cap.selection_polygon
        selection_points = np.int32(json.loads(selectionPolygon))
        projected_image = app.cm.object_visual(cap, withContours=True, margin=50)
        projected_image = cv2.polylines(projected_image, [selection_points], isClosed=True, color=(255,255,0), thickness=3)
        encodedBA = imageToBase64(projected_image)
    else:
        encodedBA = imageToBase64(app.cm.object_visual(cap, withContours=True, margin=50))
        selectionPolygon = "[]"

    return template.replace(
        "{harmonyURL}", url_for(".buildHarmony")).replace(
        "{objectName}", cap.oid).replace(
        "{selectionPolygon}", selectionPolygon).replace(
        "{encodedBA}", encodedBA).replace(
        "{camName}", '0')
    return ""


@harmony.route('/objects/<objectId>/footprint_editor', methods=['GET'])
@findObjectIdOr404
def getFootprintEditor(cap):
    return buildFootprintEditor(cap)


@harmony.route('/objects/<objectId>/project_selection', methods=['POST'])
@findObjectIdOr404
def projectSelectionOntoObject(cap):
    margin = 50
    selection_polygon = json.loads(request.form["additionPolygon"])
    cap.selection_polygon = request.form["additionPolygon"]
    return buildFootprintEditor(cap)


@harmony.route('/objects/<objectId>/submit_addition', methods=['POST'])
@findObjectIdOr404
def combineObjectWithAddition(cap):
    # margin = getattr(cap, "margin", 5)
    margin = 50
    addition_points = json.loads(request.form["additionPolygon"])
    camName = request.form["camName"]
    with DATA_LOCK:
        app.cm.add_footprint(cap, camName, addition_points)
        cap.selection_polygon = None
    return buildFootprintEditor(cap)


@harmony.route('/objects/<objectId>/submit_subtraction', methods=['POST'])
@findObjectIdOr404
def combineObjectWithSubtraction(cap):
    # margin = getattr(cap, "margin", 5)
    margin = 50
    subtraction_points = json.loads(request.form["subtractionPolygon"])
    camName = request.form["camName"]
    with DATA_LOCK:
        app.cm.subtract_footprint(cap, camName, subtraction_points)
        cap.selection_polygon = None
    return buildFootprintEditor(cap)


def minimapGenerator():
    while True:
        with DATA_LOCK:
            camImage = app.cm.buildMiniMap()
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


def main():
    create_harmony_app()
    PORT = 7000
    print(f"Launching harmony Server on {PORT}")
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()


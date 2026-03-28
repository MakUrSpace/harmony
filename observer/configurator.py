import os
import json
import cv2
import numpy as np
import math
import time
from typing import Optional
from dataclasses import dataclass

from fastapi import APIRouter, Request, Response, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse

from observer.Observer import RemoteCamera, RTSPCamera
from observer.CalibratedObserver import CalibratedCaptureConfiguration, CalibrationObserver, distanceFormula
from observer.HexObserver import HexGridConfiguration


configurator = APIRouter(prefix='/configurator', tags=['configurator'])


def buildConfigurator(request: Request):
    cc = request.app.state.cc
    cm = request.app.state.cm
    cameraConfigRows = []
    # Inject camera names for JS
    cameraNames = list(cc.cameras.keys())
    cameraNamesJson = json.dumps(cameraNames)
    
    for cam in cc.cameras.values():
        if cam is None:
            continue
        activeZone = json.dumps(cam.activeZone.tolist())
        optionValues = f"""<option value="field" selected>Game Field</option><option value="dice">Dice</option>"""
        cameraConfigRows.append(f"""
            <div class="row justify-content-center text-center">
                <h3 class="mt-5">Camera {cam.camName} <input type="button" value="Delete" class="btn btn-danger" hx-post="/configurator/delete_cam/{cam.camName}" hx-swap="outerHTML"></h3>
                <div id="camContainer_{cam.camName}" style="position: relative; display: inline-block;">
                    <img src="/configurator/camera/{cam.camName}" title="{cam.camName} Capture" height="375" id="cam{cam.camName}" onclick="camClickListener('{cam.camName}', event)">
                    <canvas id="overlayCanvas_{cam.camName}" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair;"
                        onmousedown="onMouseDown('{cam.camName}', event)"
                        onmousemove="onMouseMove('{cam.camName}', event)"
                        ></canvas>
                </div>
                <div class="text-center mt-2 d-flex flex-wrap justify-content-center align-items-center gap-2">
                    <button class="btn btn-sm btn-warning" onclick="clearShape('{cam.camName}')">Clear Manual Points</button>
                    <button class="btn btn-sm btn-info" onclick="advanceColumn('{cam.camName}')">Next Column</button>
                    <span class="d-inline-flex align-items-center gap-1 ms-2" title="Axial coordinate increment per click within a column, and per new column">
                        <small class="text-muted">Row&nbsp;&Delta;(q,r):</small>
                        <input type="number" id="rowIncrQ_{cam.camName}" value="8" style="width:4em" class="form-control form-control-sm d-inline-block">
                        <input type="number" id="rowIncrR_{cam.camName}" value="0" style="width:4em" class="form-control form-control-sm d-inline-block">
                        <small class="text-muted ms-2">Col&nbsp;&Delta;(q,r):</small>
                        <input type="number" id="colIncrQ_{cam.camName}" value="-2" style="width:4em" class="form-control form-control-sm d-inline-block">
                        <input type="number" id="colIncrR_{cam.camName}" value="4" style="width:4em" class="form-control form-control-sm d-inline-block">
                    </span>
                    <!-- AZ controls below -->
                </div>
                
                <div class="mt-3" id="activeCalibTableContainer_{cam.camName}" style="display:none; max-width: 600px; margin: auto;">
                    <h5>Calibration Points (Editable)</h5>
                    <div class="table-responsive">
                        <table class="table table-sm table-bordered text-center" id="activeCalibTable_{cam.camName}">
                            <thead><tr><th>Point</th><th>Pixel X</th><th>Pixel Y</th><th>Axial Q</th><th>Axial R</th></tr></thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>

                <label for="az">Active Zone</label><br>
                <div class="container">
                    <div class="row">
                        <div class="col">
                            <input type="text" name="az" id="cam{cam.camName}_ActiveZone" value="{activeZone}" size="50" hx-post="/configurator/cam{cam.camName}_activezone" hx-swap="none">   
                            <input type="button" class="btn btn-secondary" name="clearCam{cam.camName}AZ" value="Clear AZ" onclick="clearCamAZ('{cam.camName}', event)">
                        </div>
                        <div class="col">    
                            <label>Camera Type</label>
                            <select name="camType" hx-post="/configurator/cam{cam.camName}_type" hx-swap="none">
                              {optionValues}
                            </select>
                        </div>
                    </div>
                </div>
            </div>""")

    # Fix for Table Population on Load:
    # app.cm.calibrationPts isn't persisted, but app.cc.rsc is.
    # If table is empty but we have calibration, reconstruct it for display.
    calibrationPts = getattr(cm, 'calibrationPts', [])
    if not calibrationPts and hasattr(cc, 'rsc') and cc.rsc is not None:
        reconstructed = []
        try:
            # cc.rsc.realCamSpacePairs is [(camName, coordList), ...]
            for cN, coordList in cc.rsc.realCamSpacePairs:
                # helper to sanitize numpy arrays to lists
                def to_list_recursive(obj):
                    if isinstance(obj, np.ndarray):
                        return to_list_recursive(obj.tolist())
                    if isinstance(obj, (list, tuple)):
                        return [to_list_recursive(x) for x in obj]
                    if hasattr(obj, 'item'):
                        return obj.item()
                    return obj

                rec_entry_coords = [to_list_recursive(coordList[0]), to_list_recursive(coordList[1])]
                if len(coordList) > 2:
                    rec_entry_coords.append(to_list_recursive(coordList[2]))
                rec_entry = {cN: rec_entry_coords}
                reconstructed.append(rec_entry)
            
            cm.calibrationPts = reconstructed
            calibrationPts = cm.calibrationPts
        except Exception as e:
            print(f"Failed to reconstruct calibration table: {e}")

    calibrationPtsJson = json.dumps(calibrationPts)

    with open(f"{os.path.dirname(__file__)}/templates/Configurator.html") as f:
        template = f.read()
    
    return template.replace(
        "{configuratorURL}", "/configurator/").replace(
        "{cameraConfigRows}", "\n".join(cameraConfigRows)).replace(
        "{size}", str(cc.hex.size)).replace(
        "__CAMERA_NAMES_JSON__", cameraNamesJson).replace(
        "__CALIBRATION_PTS_JSON__", calibrationPtsJson).replace(
        "{calibratorURL}", "/configurator/") # Redirect calibratorURL to configurator base for now


    
@configurator.get('/')
@configurator.get('')
async def config(request: Request):
    return HTMLResponse(buildConfigurator(request))


@configurator.post('/')
@configurator.post('')
async def updateConfig(request: Request):
    cc = request.app.state.cc
    cc.saveConfiguration()
    return Response("Success")


def draw_dynamic_grid(cc, camName):
    try:
        rsc = getattr(cc, 'rsc', None)
        hex_grid = getattr(cc, 'hex', None)
        if rsc is None or hex_grid is None:
            return None
            
        cam = cc.cameras[str(camName)]
        if cam.mostRecentFrame is None:
            return None
        h, w = cam.mostRecentFrame.shape[:2]
        
        # Caching logic
        if not hasattr(cc, '_grid_cache'):
            cc._grid_cache = {}
        
        cache_key = (camName, id(rsc), hex_grid.size, w, h)
        if cache_key in cc._grid_cache:
            return cc._grid_cache[cache_key]

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Define 4 corners of the camera frame
        corners_cam = [(0, 0), (w, 0), (w, h), (0, h)]
        
        # Project to Real Space using the converter closest to each corner
        corners_real = []
        for p in corners_cam:
            converter = rsc.closestConverterToCamCoord(str(camName), p)
            rp = converter.convertCameraToRealSpace(p)
            corners_real.append(rp)
            
        # Convert to Axial to find range
        
        # Collect base bounding box from calibrated coordinates to prevent mapping
        # the camera corners across a perspective vanishing line which inflates hex ranges
        cm = getattr(cc, 'cm', None) # We might not have cm directly in cc, but wait it's passed? No, cc has no cm.
        # draw_dynamic_grid doesn't get cm. It gets cc and camName.
        # However, cm is available if imported or we can look inside cc.rsc
        qs = []
        rs = []
        
        # Read the RealSpaceConverter configuration to gather mapped axial coordinates
        if hasattr(cc, 'rsc') and cc.rsc and hasattr(cc.rsc, 'realCamSpacePairs'):
            for cName, coordPairs in cc.rsc.realCamSpacePairs:
                if cName == camName:
                    if len(coordPairs) > 2:
                        axials = coordPairs[2]
                        for pt in axials:
                            if hasattr(pt, '__iter__') and len(pt) >= 2:
                                qs.append(int(pt[0]))
                                rs.append(int(pt[1]))
        
        if qs and rs:
            min_q, max_q = int(min(qs)) - 15, int(max(qs)) + 15
            min_r, max_r = int(min(rs)) - 10, int(max(rs)) + 10
        else:
            # Fallback to corner projection
            for cr in corners_real:
                q, r = cc.pixel_to_axial(cr[0], cr[1])
                qs.append(q)
                rs.append(r)
            min_q, max_q = int(min(qs)), int(max(qs))
            min_r, max_r = int(min(rs)), int(max(rs))
        
        # Add padding to ensure coverage
        pad = 2
        
        # Safety bounds to prevent infinite loops or extreme memory usage
        if (max_q - min_q) > 200 or (max_r - min_r) > 200:
            print(f"Grid range too large: {max_q-min_q}x{max_r-min_r}. Check calibration.")
            return None

        # Iterate and draw
        # Use white color for grid lines, typical for overlay
        grid_color = (255, 255, 255)
        
        for q in range(min_q - pad, max_q + pad + 1):
            for r in range(min_r - pad, max_r + pad + 1):
                # Calculate Hex in Real Space
                real_poly = cc.hex_at_axial(q, r, apply_affine=False) # numpy array (6, 1, 2)
                
                cam_pts = []
                for pt in real_poly:
                    # pt is [x, y]
                    x, y = pt[0]
                    converter = rsc.closestConverterToRealCoord(str(camName), (x, y))
                    cx, cy = converter.convertRealToCameraSpace((x, y))
                    cam_pts.append([int(round(cx)), int(round(cy))])
                
                poly_cam = np.array(cam_pts, dtype=np.int32).reshape((-1, 1, 2))
                
                # Draw
                cv2.polylines(overlay, [poly_cam], True, grid_color, 1, cv2.LINE_AA)
        
        # Cache the result
        cc._grid_cache[cache_key] = overlay
        return overlay
        
    except Exception as e:
        print(f"Error drawing dynamic grid: {e}")
        # import traceback
        # traceback.print_exc()
        return None


def genCameraFullViewWithActiveZone(cc, cm, camName):
    while True:
        try:
            cam = cc.cameras[str(camName)]
            img = cam.drawActiveZone(cam.mostRecentFrame)

            # Overlay Hex Grid if configured and calibration exists
            if hasattr(cc, 'rsc') and cc.rsc is not None:
                if getattr(cc, 'show_grid', True):
                    try:
                        grid_overlay = draw_dynamic_grid(cc, camName)
                        
                        if grid_overlay is not None:
                            if grid_overlay.shape[:2] == img.shape[:2]:
                                cv2.addWeighted(grid_overlay, 0.5, img, 1.0, 0.0, dst=img)
                    except Exception as e:
                        print(f"Grid overlay error: {e}")

            # Overlay Calibration Objects
            if getattr(cc, 'show_objects', True) and hasattr(cm, 'calibrationPts') and cm.calibrationPts:
                for idx, calibObj in enumerate(cm.calibrationPts):
                    if camName in calibObj:
                        pts = calibObj[camName][0]  # pixelPoints
                        if pts and len(pts) > 0:
                            poly_cam = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(img, [poly_cam], True, (0, 255, 0), 2, cv2.LINE_AA)
                            for i, p in enumerate(pts):
                                x, y = int(p[0]), int(p[1])
                                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                                cv2.putText(img, str(i + 1), (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, img = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + img.tobytes() + b'\r\n')
            
            # Rate limiting: ~10 FPS is plenty for configurator preview
            time.sleep(0.1)
        except Exception as e:
            print(f"Failed genCameraFullViewWithActiveZone for {camName} -- {e}")
            yield (b'--frame\r\nContent-Type: image/jpg\r\n\r\n\r\n')
            time.sleep(1.0) # Back off on error
    

@configurator.post('/set_overlays')
async def setConfiguratorOverlays(request: Request, show_grid: bool = Form(...), show_objects: bool = Form(...)):
    cc = request.app.state.cc
    cc.show_grid = show_grid
    cc.show_objects = show_objects
    return Response("Success")
    
    
@configurator.get('/camera/{camName}')
async def cameraActiveZoneWithObjects(request: Request, camName: str):
    cc = request.app.state.cc
    cm = request.app.state.cm
    return StreamingResponse(genCameraFullViewWithActiveZone(cc, cm, str(camName)), media_type='multipart/x-mixed-replace; boundary=frame')


@configurator.post('/cam{camName}_activezone')
async def updateCamActiveZone(request: Request, camName: str, az: str = Form(...)):
    cc = request.app.state.cc
    print(f"Received update Active Zone request for {camName}")
    try:
        az_arr = np.float32(json.loads(az))
        cam = cc.cameras[camName]
        cam.setActiveZone(az_arr)
    except Exception as e:
        print(f"Unrecognized data: {camName} - {az} - {e}")
    return Response("success")


@configurator.post('/cam{camName}_type')
async def updateCamType(request: Request, camName: str, camType: str = Form(...)):
    cc = request.app.state.cc
    print(f"Received update Camera Type request for {camName}")
    try:
        cam = cc.cameras[camName]
        cam.camType = camType 
    except Exception as e:
        print(f"Unrecognized data: {camName} - {camType} - {e}")
    return Response("success")


@configurator.get('/new_camera')
async def getNewCameraForm(request: Request):
    with open(f"{os.path.dirname(__file__)}/templates/NewCamera.html", "r") as f:
        template = f.read()
    return HTMLResponse(template.replace("{configuratorURL}", "/configurator/"))


@configurator.post('/new_camera')
async def addNewCamera(
    request: Request,
    camName: str = Form(...),
    camRot: str = Form(...),
    rtspCam: Optional[str] = Form(None),
    camAddr: str = Form(...),
    camAuth: Optional[str] = Form(None)
):
    cc = request.app.state.cc
    try:
        if camAuth:
            auth_parts = [part.strip() for part in camAuth.split(",")]
            if len(auth_parts) != 2:
                raise Exception(f"Unrecognized auth format: {camAuth}")
            auth = auth_parts
        else:
            auth = None
    except:
        auth = None

    builder = RTSPCamera if rtspCam else RemoteCamera
    cc.cameras[camName] = builder(address=camAddr, activeZone=[[0, 0], [0, 1], [1, 1,], [1, 0]], camName=camName, rotate=camRot, auth=auth)
    cc.rsc = None
    cc.saveConfiguration()
    cc.capture()
    return HTMLResponse(f"""<script>window.location = '/configurator/';</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/configurator/'">""")


@dataclass
class CalibrationPoint:
    axial: list
    pixel: list
    real: list


def order_points_clockwise(calibPts: list[CalibrationPoint]):
    camPts = np.array([pt.pixel for pt in calibPts])
    centroid = np.mean(camPts, axis=0)
    angles = np.arctan2(camPts[:, 1] - centroid[1], camPts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return [calibPts[i] for i in order]


@configurator.post('/manual_calibration')
async def manualCalibration(request: Request):
    cc = request.app.state.cc
    cm = request.app.state.cm
    # Expects a dict of {camName: points}
    data = await request.json()
    print(f"Received manual calibration data: {data}")
    if not data:
        return Response("No data received", status_code=400)
        
    added_count = 0
    new_calib_entry = {}

    cm.calibrationPts = []
    for camName, payload in data.items():
        if not isinstance(payload, dict) or 'pixel' not in payload or 'axial' not in payload:
            continue
            
        pixel_data = payload['pixel']
        axial_data = payload['axial']
        
        if len(pixel_data) < 3 or len(axial_data) < 3:
            continue
            
        if camName not in cc.cameras:
            continue

        # Get camera resolution
        cam = cc.cameras[camName]
        try:
            h, w = cam.mostRecentFrame.shape[:2]
        except AttributeError:
             print(f"Camera {camName} has no frame.")
             continue

        unique_pts = {}
        for pixel, axial in zip(pixel_data, axial_data):
            pixel_full = [int(pixel[0] * w), int(pixel[1] * h)]
            real = cc.axial_to_pixel(*axial)
            calib_pt = CalibrationPoint(pixel=pixel_full, axial=axial, real=real)
            # Use tuple of axial as unique key to deduplicate
            axial_tup = tuple(axial)
            if axial_tup not in unique_pts:
                unique_pts[axial_tup] = calib_pt

        calib_pts = list(unique_pts.values())
        calib_pt_columns = {}
        for calib_pt in calib_pts:
            if calib_pt.axial[1] not in calib_pt_columns:
                calib_pt_columns[calib_pt.axial[1]] = []
            calib_pt_columns[calib_pt.axial[1]].append(calib_pt)
        
        col_values = sorted(list(calib_pt_columns.keys()))
        
        # Sort each column's points by row coordinate q (axial[0])
        for r_val in col_values:
            calib_pt_columns[r_val].sort(key=lambda pt: pt.axial[0])

        valid_blocks = []
        seen_blocks = set()
        for col_idx, col in enumerate(col_values[:-1]):
            next_col = col_values[col_idx + 1]
            col_pts = calib_pt_columns[col]
            next_col_pts = calib_pt_columns[next_col]
            # Only form blocks where both columns have a consecutive pair of points
            max_idx = min(len(col_pts), len(next_col_pts)) - 1
            for idx in range(max_idx):
                calib_pt = col_pts[idx]
                next_same_r = col_pts[idx + 1]
                next_q = next_col_pts[idx:idx + 2]

                if len(next_q) < 2:
                    continue  # Safety guard — should not occur given max_idx, but be explicit

                block = [calib_pt, next_same_r, *next_q]
                block = order_points_clockwise(block)

                block_sig = tuple(sorted([tuple(p.axial) for p in block]))
                if block_sig not in seen_blocks:
                    seen_blocks.add(block_sig)
                    valid_blocks.append(block)

        for block in valid_blocks:
            pixel_chunk = [p.pixel for p in block]
            axial_chunk = [p.axial for p in block]
            real_chunk = [p.real for p in block]
                
            def sanitize_native(obj):
                if isinstance(obj, np.ndarray):
                    return sanitize_native(obj.tolist())
                if isinstance(obj, (list, tuple)):
                    return [sanitize_native(x) for x in obj]
                if hasattr(obj, 'item'):
                    return obj.item()
                if isinstance(obj, float):
                    return float(obj)
                if isinstance(obj, int):
                    return int(obj)
                return obj
            
            p_chunk = sanitize_native(pixel_chunk)
            r_chunk = sanitize_native(real_chunk)
            a_chunk = sanitize_native(axial_chunk)
                
            cm.calibrationPts.append({camName: [p_chunk, r_chunk, a_chunk]})
            added_count += 1
        
    if added_count > 0:
        if hasattr(cm, 'buildRealSpaceConverter'):
            cm.buildRealSpaceConverter()
        else:
            from observer.CalibratedObserver import RealSpaceConverter
            cc.rsc = RealSpaceConverter([cNCoordPair 
                                               for cPtGrp in cm.calibrationPts
                                               for cNCoordPair in list(cPtGrp.items())])
        cc.saveConfiguration()
        return Response(f"Added manual calibration for {added_count} cameras")
    else:
        return Response("No valid calibration points found", status_code=400)



@configurator.post('/delete_calibration/{index}')
async def deleteCalibrationEndpoint(request: Request, index: int):
    cc = request.app.state.cc
    cm = request.app.state.cm
    try:
        if 0 <= index < len(cm.calibrationPts):
            cm.calibrationPts.pop(index)
            if len(cm.calibrationPts) > 0:
                if hasattr(cm, 'buildRealSpaceConverter'):
                    cm.buildRealSpaceConverter()
                else:
                    from observer.CalibratedObserver import RealSpaceConverter
                    cc.rsc = RealSpaceConverter([cNCoordPair 
                                                       for cPtGrp in cm.calibrationPts
                                                       for cNCoordPair in list(cPtGrp.items())])
            else:
                cc.rsc = None
            cc.saveConfiguration()
            return Response("Success")
        else:
            return Response("Invalid index", status_code=400)
    except Exception as e:
        print(f"Error deleting calibration object: {e}")
        return Response(str(e), status_code=500)


@configurator.post('/delete_cam/{camName}')
async def deleteCamera(request: Request, camName: str):
    cc = request.app.state.cc
    cc.cameras.pop(camName)
    cc.rsc = None
    cc.saveConfiguration()
    return HTMLResponse(f"""<script>window.location.reload();</script>
              <input class="btn-secondary" type="button" value="Configuration" onclick="window.location.href='/configurator/'">""")


@configurator.post('/grid_configuration')
async def configureGrid(request: Request, size: float = Form(...)):
    cc = request.app.state.cc
    cc.hex = HexGridConfiguration(size=size)
    return HTMLResponse(f"""
        <input type="number" class="form-check-input bg-info" name="size" min="10" max="60" value="{cc.hex.size}" style="width:4em">
        <label class="form-check-label" for="size">Size</label><br>
        <input type="submit" class="btn btn-primary bg-secondary" value="Submit Grid Configuration">""")



if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    app = FastAPI()
    app.state.cc = CalibratedCaptureConfiguration()
    app.state.cc.capture()
    app.state.cm = CalibrationObserver(app.state.cc)
    app.include_router(configurator)

    @app.get('/{page}')
    async def getPage(page: str):
        try:
            with open(f"{os.path.dirname(__file__)}/templates/{page}") as f:
                content = f.read()
            if page.endswith(".html"):
                return HTMLResponse(content)
            elif page.endswith(".css"):
                return Response(content, media_type="text/css")
            elif page.endswith(".js"):
                return Response(content, media_type="application/javascript")
            return Response(content)
        except Exception as e:
            print(f"Failed to find page: {e}")
            return HTMLResponse("Not found!", status_code=404)

    print(f"Launching Configurator Server on Port 7000")
    uvicorn.run(app, host="0.0.0.0", port=7000)

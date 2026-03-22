#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import sleep
from dataclasses import dataclass
from collections import defaultdict
from math import cos, acos, degrees, radians
import random


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[3]:


from observer.Observer import Observer, CaptureConfiguration, CameraChange, Camera, hStackImages, vStackImages, ChangeSet, TrackedObject, Transition, clipImage


# In[4]:


def distanceFormula(pt0, pt1):
    if len(pt0) != len(pt1):
        raise Exception(f"Cannot compute distance for dimensions: {len(pt0)} vs. {len(pt1)}")
    return sum([(pt1[i] - pt0[i])**2 for i in range(len(pt0))]) ** 0.5


def scale_contour(contour, scale_factor):
    """
    Scales a contour by a given factor while keeping it centered.

    :param contour: NumPy array of shape (N, 1, 2) containing contour points.
    :param scale_factor: The factor by which to scale the contour.
    :return: Scaled contour.
    """
    # Compute the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:  # Avoid division by zero
        cx, cy = 0, 0
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    # Convert contour to centered points
    centered_contour = contour - [cx, cy]

    # Scale the contour
    scaled_contour = centered_contour * scale_factor

    # Shift back to original position
    scaled_contour += [cx, cy]

    return scaled_contour.astype(np.int32)  # Convert back to integer


def order_points_clockwise(camPts, realPts):
    camPts = np.array(camPts)
    realPts = np.array(realPts)
    centroid = np.mean(camPts, axis=0)
    angles = np.arctan2(camPts[:, 1] - centroid[1], camPts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return camPts[order], realPts[order]

# In[5]:


@dataclass
class CameraRealSpaceConverter:
    camName: int
    camTriPts: np.array
    realTriPts: np.array
    M: np.array = None

    def __post_init__(self):
        self.camTriPts = np.array(self.camTriPts)
        self.realTriPts = np.array(self.realTriPts)
        
        if len(self.camTriPts) == 3:
            self.camRect = self.triangleToSquare(self.camTriPts)
            self.realRect = self.triangleToSquare(self.realTriPts)
            # Rectangles are in the order of Pt-A-90*; Pt-B-60*; Pt-C-Projected; Pt-D-30*
            self.tuneToCalibrationBox(self.camRect, self.realRect)
        elif len(self.camTriPts) == 4:
            camPts, realPts = order_points_clockwise(self.camTriPts, self.realTriPts)
            self.camRect = np.float32(camPts)
            self.realRect = np.float32(realPts)
            self.tuneToCalibrationBox(self.camRect, self.realRect)
        else: # >= 5 points
            camPts, realPts = order_points_clockwise(self.camTriPts, self.realTriPts)
            self.camRect = np.float32(camPts)
            self.realRect = np.float32(realPts)
            self.M, _ = cv2.findHomography(self.camRect, self.realRect)

    @property
    def camSpaceCentroid(self):
        xA = int(sum([pt[0] for pt in self.camRect]) / len(self.camRect))
        yA = int(sum([pt[1] for pt in self.camRect]) / len(self.camRect))
        return np.array([xA, yA], dtype="int32")

    @property
    def realSpaceCentroid(self):
        xA = int(sum([pt[0] for pt in self.realRect]) / len(self.realRect))
        yA = int(sum([pt[1] for pt in self.realRect]) / len(self.realRect))
        return np.array([xA, yA], dtype="int32")

    @staticmethod
    def getAngle(pt0, pt1, pt2):
        u = [pt1[0] - pt0[0], pt1[1] - pt0[1]]
        v = [pt2[0] - pt0[0], pt2[1] - pt0[1]]
        duv = u[0] * v[0] + u[1] * v[1]
        mu = (u[0] ** 2 + u[1] ** 2) ** 0.5
        mv = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return degrees(acos( duv / (mu * mv) ))

    @classmethod
    def triangleToSquare(cls, triPts):
        angles = {}
        triPts = np.array(triPts)
        for idx, pt in enumerate(triPts):
            pt = np.array(pt)
            vectors = []
            try:
                otherPts = [op for op in triPts if not (pt == op).all()]
            except:
                print("Failed triangle to square")
                raise
            angles[idx] = cls.getAngle(pt, *otherPts)

        ptA_ninety = triPts[sorted([(key, abs(90 - angle)) for key, angle in angles.items()], key=lambda x: x[1])[0][0]]
        ptB_sixty = triPts[sorted([(key, abs(60 - angle)) for key, angle in angles.items()], key=lambda x: x[1])[0][0]]
        ptD_thirty = triPts[sorted([(key, abs(30 - angle)) for key, angle in angles.items()], key=lambda x: x[1])[0][0]]

        ninetyThirtyDiff = [d1 - d0 for d1, d0 in zip(ptD_thirty, ptA_ninety)]
        ptC_projected = np.float32([pt + d for pt, d in zip(ptB_sixty, ninetyThirtyDiff)])

        squarePts = np.float32([ptA_ninety, ptB_sixty, ptC_projected, ptD_thirty])
        return squarePts

    def tuneToCalibrationBox(self, cameraRectangle, realRectangle):
        self.M = cv2.getPerspectiveTransform(cameraRectangle, realRectangle)

    def convertCameraToRealSpace(self, p):
        assert not (self.M is None), "Must calibrate camera before converting coordinates"
        M = self.M
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        return (px, py)

    def convertRealToCameraSpace(self, p):
        assert not (self.M is None), "Must calibrate camera before converting coordinates"
        # Compute the inverse of the perspective transform matrix
        M_inv = np.linalg.inv(self.M)

        # Apply the transformation
        px = (M_inv[0][0] * p[0] + M_inv[0][1] * p[1] + M_inv[0][2]) / (M_inv[2][0] * p[0] + M_inv[2][1] * p[1] + M_inv[2][2])
        py = (M_inv[1][0] * p[0] + M_inv[1][1] * p[1] + M_inv[1][2]) / (M_inv[2][0] * p[0] + M_inv[2][1] * p[1] + M_inv[2][2])
        return (px, py)   

    def showUnwarpedImage(self, cam):
        warp = cv2.warpPerspective(cam.cropToActiveZone(cam.mostRecentFrame), self.M, (1200, 1200))
        return warp


# In[6]:


class RealSpaceConverter:
    def __init__(self,  realCamSpacePairs: dict):
        self.realCamSpacePairs = realCamSpacePairs
        #  {camName: [[camSpaceTriPts, realSpaceTriPts], ...], ...}
        self.converters = defaultdict(list)
        for camName, coordPairs in self.realCamSpacePairs:
            camSpaceTriPts, realSpaceTriPts = coordPairs[:2]
            converter = CameraRealSpaceConverter(camName, camSpaceTriPts, realSpaceTriPts)
            self.converters[camName].append(converter)

    def closestConverterToCamCoord(self, camName, camCoord):
        converters = self.converters[camName]
        closest = converters[0]
        minDistance = distanceFormula(closest.camSpaceCentroid, camCoord)
        for converter in converters[1:]:
            distance = distanceFormula(converter.camSpaceCentroid, camCoord)
            if distance < minDistance:
                closest = converter
                minDistance = distance
        return closest

    def closestConverterToRealCoord(self, camName, realCoord):
        converters = self.converters[camName]
        closest = converters[0]
        minDistance = distanceFormula(closest.realSpaceCentroid, realCoord)
        for converter in converters[1:]:
            distance = distanceFormula(converter.realSpaceCentroid, realCoord)
            if distance < minDistance:
                closest = converter
                minDistance = distance
        return closest

    def camCoordToRealSpace(self, camName, camCoord):
        closest = self.closestConverterToCamCoord(camName, camCoord)
        return closest.convertCameraToRealSpace(camCoord)

    def realSpaceToCamCoord(self, realCoord, camName):
        closest = self.closestConverterToRealCoord(camName, realCoord)
        return closest.convertRealToCameraSpace(realCoord)

    def changeSetCenterPoints(self, changeSet: ChangeSet):
        return {cN: self.camCoordToRealSpace(cN, change.center)
                for cN, change in changeSet.changeSet.items()
                if change is not None and change.changeType not in ["delete", None]}

    def changeSetToRealCenter(self, changeSet: ChangeSet):
        centerPoints = list(self.changeSetCenterPoints(changeSet).values())
        try:
            xS, yS = zip(*centerPoints)
        except Exception:
            from traceback import format_exc
            print(f"Failed to find Real Center for: {changeSet}")
            print(format_exc())
            raise

        if len(centerPoints) == 1:
            return xS[0], yS[0]

        medDists = []
        for idx, ctrPt in enumerate(centerPoints):
            distances = [distanceFormula(ctrPt, centerPoints[i]) for i in range(len(centerPoints)) if i != idx]
            medDists.append(sorted(distances)[int(len(distances) / 2)])

        weights = [1/max(1, md) for md in medDists]
        avgX = sum([x * w for x, w in zip(xS, weights)]) / sum(weights)
        avgY = sum([y * w for y, w in zip(yS, weights)]) / sum(weights)
        return avgX, avgY

    def changeSetCenterDeltas(self, changeSet: ChangeSet):
        centerPoints = self.changeSetCenterPoints(changeSet)
        center = self.changeSetToRealCenter(changeSet)
        return {cN: [d0 - d1 for d0, d1 in zip(center, camCenter)]
                for cN, camCenter in centerPoints.items()}

    def changeSetWithinSameRealSpace(self, changeSet: ChangeSet, tolerance=30):
        centerDeltas = self.changeSetCenterDeltas(changeSet)
        for camName, delta in centerDeltas.items():
            distance = sum([d**2 for d in delta]) ** 0.5
            if distance > tolerance:
                return False
        return True

    def trackedObjectLastDistance(self, trackedObject: TrackedObject):
        previousChangeSet = trackedObject.previousVersion()
        if previousChangeSet.empty:
            return 0
        currentLocation = self.changeSetToRealCenter(trackedObject)
        previousLocation = self.changeSetToRealCenter(previousChangeSet)
        return distanceFormula(previousLocation, currentLocation)

    def distanceBetweenObjects(self, firstObject, secondObject):
        firstCenter = self.changeSetToRealCenter(firstObject)
        secondCenter = self.changeSetToRealCenter(secondObject)
        return distanceFormula(firstCenter, secondCenter)

    def cameraRealSpaceOverlap(self, cameras):
        im = None
        warps = []
        for camName, camConverters in self.converters.items():
            cam = cameras[camName]
            for converter in camConverters:
                warp = cv2.warpPerspective(cam.cropToActiveZone(cam.mostRecentFrame), converter.M, (1200, 1200))
                warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                warps.append(warp)
        avg_im = sum([warp * (1 / len(warps)) for warp in warps]).astype("uint8")
        avg_im = cv2.threshold(avg_im, 64, 255, cv2.THRESH_BINARY)[1]
        realSpaceContour = cv2.findContours(avg_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        return realSpaceContour

    def cameraRealSpaceOverlap(self, cameras, out_size=(1200, 1200)):
        warps = []
        for camName, camConverters in self.converters.items():
            cam = cameras[camName]
            if cam.mostRecentFrame is not None:
                h, w = cam.mostRecentFrame.shape[:2]
            else:
                # Use activeZone bounding box if frame is not available
                pts = np.int32(cam.activeZone)
                x, y, w, h = cv2.boundingRect(pts)
                # Ensure we have at least some dimensions, 1920x1080 is common
                w = max(w + x, 1920)
                h = max(h + y, 1080)

            frame = np.zeros((h, w), dtype=np.uint8)
            pts = np.int32(cam.activeZone)
            cv2.fillPoly(frame, [pts], 255)

            for converter in camConverters:
                warp = cv2.warpPerspective(
                    frame,
                    converter.M,
                    out_size
                )
                warps.append(warp)

        if not warps:
            return []

        # --- max-sum union of warped masks ---
        # sum first (uint16 to avoid overflow), then clamp
        summed = np.sum(
            np.stack(warps, axis=0).astype(np.uint16),
            axis=0
        )
        union = np.clip(summed, 0, 255).astype(np.uint8)

        # --- binarize + extract contour ---
        _, union = cv2.threshold(union, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            union,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        return contours


    def unwarpedOverlaidCameras(self, cameras):
        im = None
        for camName, camConverters in self.converters.items():
            cam = cameras[camName]
            for converter in camConverters:
                warp = cv2.warpPerspective(cam.cropToActiveZone(cam.mostRecentFrame), converter.M, (1200, 1200))
                im = warp if im is None else cv2.addWeighted(im, 0.6, warp, 0.3, 0)
        return im


# In[7]:


class CalibrationObserver(Observer):
    def __init__(self, config: CaptureConfiguration):
        super().__init__(config)

    def reset(self):
        super().reset()
        self.calibrated = False
        self.next_triangle = [[0, 0], [0, 0], [0, 0]]
        self.first_triangle = [[500, 500], [460, 500], [500, 430]]
        self.calibrationPts = []

    def calibrateToObject(self, calibObj, dowel_position):
        print("Calibrating")
        calibPtLoc = self.next_triangle[0]
        expectedCams = list(self.cc.cameras.keys())

        transition = self.first_triangle
        self.next_triangle = [[d0 + d1 for d0, d1 in zip(tri, trans)] for tri, trans in zip(self.next_triangle, transition)]
        realTriPts = self.next_triangle

        calibTriPts = {}
        for camName, change in calibObj.changeSet.items():
            if camName not in expectedCams:
                continue
            calibContour = change.changeContours[0]
            polyPoints = cv2.approxPolyDP(calibContour, 0.053 * cv2.arcLength(calibContour, True), True)
            assert len(polyPoints) == 3, f"Failed to find triangle for {camName} at CalibPt {calibPtLoc}. Found {len(polyPoints)} points"
            calibTriPts[camName] = [[pt[0] for pt in polyPoints], realTriPts]
        print(f"Storing calibration point {calibPtLoc}")
        calibObj.calibTriPts = calibTriPts
        self.calibrationPts.append(calibTriPts)

    def buildRealSpaceConverter(self):
        self.cc.rsc = RealSpaceConverter([cNCoordPair 
                                          for cPtGrp in self.calibrationPts
                                          for cNCoordPair in list(cPtGrp.items())])


@dataclass
class MiniMapObject:
    object: TrackedObject
    color: tuple[int, int, int]


class CalibratedCaptureConfiguration(CaptureConfiguration):
    def buildConfiguration(self):
        config = super().buildConfiguration()
        if self.rsc is not None:
            def sanitize_recursive(obj):
                if isinstance(obj, np.ndarray):
                    return sanitize_recursive(obj.tolist())
                if isinstance(obj, (list, tuple)):
                    return [sanitize_recursive(x) for x in obj]
                if hasattr(obj, 'item'):
                    return obj.item()
                return obj

            config['rsc'] = []
            for cN, coordList in self.rsc.realCamSpacePairs:
                entry = [sanitize_recursive(coordList[0]), sanitize_recursive(coordList[1])]
                if len(coordList) > 2:
                    entry.append(sanitize_recursive(coordList[2]))
                config['rsc'].append([cN, entry])
        return config

    def loadConfiguration(self, path="observerConfiguration.json"):
        super().loadConfiguration(path="observerConfiguration.json")
        try:
            config = self.readConfigFile()
        except Exception:
            config = {}
        self.rsc = config.get("rsc", None)
        if self.rsc is not None:
            self.rsc = [
                [cN, [[np.array(pt, dtype="int32") for pt in cL] for cL in coordList]]
                for cN, coordList in self.rsc]
            self.rsc = RealSpaceConverter(self.rsc)
            self.realSpaceContours = self.rsc.cameraRealSpaceOverlap(self.cameras)
            
            # Calculate _realSpaceBoundingBox once
            x, y, w, h = 1200, 1200, 1, 1
            if not self.realSpaceContours:
                 self._realSpaceBoundingBox = (0, 0, 1200, 1200)
            else:
                for contour in self.realSpaceContours:
                    cbX, cbY, cbW, cbH = cv2.boundingRect(contour)
                    cb_maxX = cbX + cbW
                    cb_maxY = cbY + cbH
                    x = min(x, cbX)
                    y = min(y, cbY)
                    w = max(w, cb_maxX - x)
                    h = max(h, cb_maxY - y)
                self._realSpaceBoundingBox = (x, y, w, h)
        else:
            self.realSpaceContours = None
            self._realSpaceBoundingBox = (0, 0, 1200, 1200)

    def objectToHull(self, obj: TrackedObject, color=(255, 255, 255)):
        assert self.rsc is not None, "Calibration information needed"
        newContour = {}
        for camName, cS in obj.changeSet.items():
            if cS.changeType not in ['add', 'move']:
                continue
            for cnt in cS.changeContours:
                realSpaceConverter = self.rsc.closestConverterToCamCoord(camName, cnt[0][0])
                converted = []
                for pt in cnt:
                    converted.append([realSpaceConverter.convertCameraToRealSpace(pt[0])])
            newContour[camName] = np.array([converted], dtype="int32")

        contourBase = np.zeros([1200, 1200], dtype="uint8")
        contourOverlap = None
        cSDeltas = self.rsc.changeSetCenterDeltas(obj)
        for index, (camName, contours) in enumerate(newContour.items()):
            tx, ty = cSDeltas[camName]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            for cnt in contours:
                filledContour = cv2.drawContours(contourBase.copy(), [cnt], -1, 255, -1)
                shiftedFilledContour = cv2.warpAffine(filledContour, M, (1200, 1200))
                if contourOverlap is None:
                    contourOverlap = shiftedFilledContour
                else:
                    contourOverlap = cv2.bitwise_or(contourOverlap, shiftedFilledContour)
        contours, _ = cv2.findContours(contourOverlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[-1]

    def realSpaceBoundingBox(self):
        return getattr(self, "_realSpaceBoundingBox", (0, 0, 1200, 1200))

    def buildMiniMap(self, objectsAndColors: list[MiniMapObject]):
        image = np.zeros([1200, 1200, 3], dtype="uint8")
        if self.rsc is None:
            return image

        x, y, w, h = self.realSpaceBoundingBox()
        cv2.drawContours(image, self.realSpaceContours, -1, (125, 125, 125), 2)

        print(f"Minimap has {len(objectsAndColors)} objects")
        if len(objectsAndColors) == 0:
            return image[y:y+h, x:x+w]
        drawnObjs = []
        
        for objAndColor in objectsAndColors[::-1]:
            obj = objAndColor.object
            color = objAndColor.color
            if obj in drawnObjs:
                continue
            drawnObjs.append(obj)
            hull = self.objectToHull(obj)
            image = cv2.drawContours(image, [hull], -1, color, -1)
        return image[y:y+h, x:x+w]

    def line_of_sight(self, objectA: TrackedObject, objectB: TrackedObject, allObjects: list[TrackedObject]):
        """ Determines if there is a clear line of sight between objectA and objectB. """
        # 1) Compute the convex hull containing both objects
        points = []

        # Get the bounding box points for both objects
        for obj in [objectA, objectB]:
            points.extend([p[0] for p in self.objectToHull(obj)])

        # Compute the convex hull
        objects_hull = cv2.convexHull(np.array(points))

        # 2) Subtract all other objects from the convex hull
        mask = np.zeros([1200, 1200, 1], dtype="uint8")
        cv2.fillPoly(mask, [objects_hull], 255)  # Fill the hull with white

        for obj in allObjects:
            if obj == objectA or obj == objectB:
                continue  # Skip the two main objects
            hull = self.objectToHull(obj)
            hull = scale_contour(hull, 1.2)
            mask = cv2.drawContours(mask, [hull], -1, 0, thickness=-1)  # Subtract objects by filling them black

        # 3) Count the remaining objects within the hull
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4) If only one contour (the hull remains), return True, else return False
        return len(contours) == 1


class CalibratedObserver(Observer):
    def __init__(self, config: CalibratedCaptureConfiguration):
        super().__init__(config)

    def distinct_colors(self, n=15, new=False):
        """ Generate n distinct colors that stand out against brown. """
        if not new and n > 0 and len(getattr(self, "_colors", {n: []})[n]) == n:
            return self._colors[n]
        colors = set()
        contrast_colors = [(139, 69, 19), (204, 100, 2), (0, 255, 0)]
        calc_contrast = lambda r, g, b, bg: abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
        while len(colors) < n:
            color = None
            for i in range(100):
                r, g, b = (random.randint(50, 230) for _ in range(3))  # Exclude near black/white
                contrasts = [calc_contrast(r, g, b, c) for c in contrast_colors + list(colors)]
                if sum([contrast > 200 for contrast in contrasts]) >= len(contrast_colors + list(colors)):  # Ensure high contrast
                    color = (r, g, b)
                    break
            if color is None:
                raise Exception(f"Failed to generate color within 100 cycles")
            colors.add((r, g, b))
        self._colors = getattr(self, "_colors", {})
        self._colors[n] = list(colors)
        return self._colors[n]

    @property
    def objectsAndColors(self):
        memColors = {mem.oid: color for mem, color in zip(self.memory, self.distinct_colors(len(self.memory)))}
        return [
            MiniMapObject(
                mem,
                memColors[mem.oid]
            ) for mem in self.memory]

    def buildMiniMap(self, objectsAndColors=None):
        return self.cc.buildMiniMap(objectsAndColors=objectsAndColors or [])

    def object_visual(self, changeSet: ChangeSet, withContours=True, margin=0):
        cameras = self.cc.cameras
        if changeSet.empty:
            return np.zeros([10, 10], dtype="float32")

        images = {cam: change.after for cam, change in changeSet.changeSet.items() if change.changeType not in ["delete", None]}
        # # Add snippet of object from MiniMap
        # miniMap = self.buildMiniMap()
        # hull = self.cc.objectToHull(changeSet)
        # mmcb_x, mmcb_y, mmcb_w, mmcb_h = cv2.boundingRect(np.array(hull, dtype=np.int32))

        # images['minimap'] = miniMap[mmcb_y: mmcb_y + mmcb_h, mmcb_x: mmcb_x + mmcb_w]

        maxHeight =  max([im.shape[0] + margin * 2 for im in images.values()])
        filler = np.zeros((maxHeight, 50, 3), np.uint8)

        margins = [-margin, -margin, margin * 2, margin * 2]

        for camName, change in changeSet.changeSet.items():
            if change.changeType == "delete":
                images[camName] = filler
            else:
                if withContours:
                    images[camName] = clipImage(cv2.addWeighted(
                            cameras[camName].mostRecentFrame.copy(),
                            0.6,
                            cv2.drawContours(
                                cameras[camName].mostRecentFrame.copy(),
                                change.changeContours, 
                                -1,
                                (255, 0, 0),
                                -1
                            ),
                            0.4,
                            0
                        ),
                        [dim + m for dim, m in zip(change.clipBox, margins)]
                    )
                else:
                    images[camName] = clipImage(self.cc.cameras[camName].mostRecentFrame.copy(), [dim + m for dim, m in zip(change.clipBox, margins)])

        return hStackImages(images.values())

    @staticmethod
    def circle_to_contour(center, radius, num_points=50):
        """
        Converts a circle into an OpenCV contour.

        :param center: (x, y) center of the circle.
        :param radius: Radius of the circle.
        :param num_points: Number of points to approximate the circle.
        :return: OpenCV contour (NumPy array of shape (N, 1, 2)).
        """
        x, y = center
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # Evenly spaced angles
        contour = np.array([[
            [int(x + radius * np.cos(theta)), int(y + radius * np.sin(theta))]
        ] for theta in angles], dtype=np.int32)

        return contour


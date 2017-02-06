#!/usr/bin/python
"""
roadManager.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/07: Initial version converted to a class
"""

import numpy as np
import cv2
import math
from p5lib.cameraCal import CameraCal
from p5lib.imageFilters import ImageFilters
from p5lib.projectionManager import ProjectionManager
from p5lib.lane import Lane
from p5lib.line import Line
from p5lib.vehicleDetection import VehicleDetection
from p5lib.vehicle import Vehicle
from p5lib.vehicleTracking import VehicleTracking


class RoadManager():
    # Initialize roadManager

    def __init__(self, camCal, keepN=10, debug=False, scrType=0):
        # for both left and right lines
        # set debugging
        self.debug = debug
        self.scrType = scrType

        # frameNumber
        self.curFrame = None

        # keep last N
        self.keepN = keepN

        # our own copy of the camera calibration results
        self.mtx, self.dist, self.img_size = camCal.get()

        # normal image size
        self.x, self.y = self.img_size

        # mid point
        self.mid = int(self.y / 2)

        # create our own projection manager
        self.projMgr = ProjectionManager(camCal, keepN=keepN, debug=self.debug)

        # sides definitions
        self.left = 1
        self.right = 2

        # default left-right lane masking
        self.maskDelta = 5

        # road statistics
        # the left and right lanes curvature measurement could be misleading -
        # need a threshold to indicate straight road.
        self.roadStraight = False
        # radius of curvature of the line in meters
        self.radiusOfCurvature = None

        # distance in meters of vehicle center is off from road center
        self.lineBasePos = None

        # ghosting of lane lines (for use in trouble spots - i.e: bridge or in
        # harder challenges)
        self.lastNEdges = None

        # array of lines - we will create a set of empty lines at the beginning
        # and have the first lane initialize them
        leftLine = Line(self.left, self.x, self.y, self.projMgr.projectedX,
                        self.projMgr.projectedY, self.maskDelta)
        rightLine = Line(self.right, self.x, self.y, self.projMgr.projectedX,
                         self.projMgr.projectedY, self.maskDelta)
        self.lines = [leftLine, rightLine]

        # array of lanes - we will create one at the begining and add more
        # during processing of the first frame.
        self.lanes = [Lane(self.x, self.y,
                           self.projMgr.projectedX,
                           self.projMgr.projectedY,
                           self.maskDelta, self.lines)]
        self.mainLaneIdx = 0
        self.resized = False

        # road overhead and unwarped views
        self.roadsurface = np.zeros(
            (self.projMgr.projectedY, self.projMgr.projectedX, 3),
            dtype=np.uint8)
        self.roadunwarped = None
        self.lastTop = 0
        self.restartCount = 0
        self.resetProjectionCount = 0

        # cloudy mode
        self.cloudyMode = False

        # pixel offset from direction of travel
        self.lastLeftRightOffset = 0

        # boosting
        self.boosting = 0.0

        # vehicles
        self.versionName = "CHOGRGB4"
        self.cspace = 'RGB'
        self.orient = 8
        self.pix_per_cell = 4
        self.cell_per_block = 2
        self.hog_channel = 0
        # self.threshold = 10.15
        # self.threshold = 20.0
        # self.threshold = 15.0
        # self.threshold = 27.0
        # self.threshold = 40.0
        self.threshold = 50.0
        self.vehicleDetection = VehicleDetection(
                                    self.projMgr.projectedX,
                                    self.projMgr.projectedY,
                                    self.versionName,
                                    self.cspace, self.orient,
                                    self.pix_per_cell,
                                    self.cell_per_block,
                                    self.hog_channel,
                                    self.threshold)
        self.vehicleTracking = VehicleTracking(
                                    self.x,
                                    self.y,
                                    self.projMgr.projectedX,
                                    self.projMgr.projectedY,
                                    self.lanes)

        self.possibleVehicleWindows = []
        self.vehicles = []

        # special effects image
        self.specialProjectedEffects = np.zeros(
            (self.projMgr.projectedY, self.projMgr.projectedX, 3),
            dtype=np.uint8)
        self.specialPerspectiveEffects = np.zeros(
            (self.y, self.x, 3), dtype=np.uint8)

        # activeWindows
        self.activeWindows = []

        # resulting image
        self.final = None

        # for debugging only
        if self.debug:
            self.diag1 = np.zeros((self.y, self.x, 3), dtype=np.float32)

    def addLaneLeft(self, curLane):
        faint = -(curLane.maskvalue / 128)
        if self.resized:
            faint *= 0.75
        if ((np.absolute(faint) > 0.5 or
                (self.resized and np.absolute(faint) > 0.4)) and
                curLane.adjacentLeft and
                curLane.adjacentLLane is None):
            # print("threshold:", faint)
            newLeftLine = Line(self.left, self.x, self.y,
                               self.projMgr.projectedX,
                               self.projMgr.projectedY,
                               self.maskDelta)
            newLeftLine.createPolyFitLeft(
                self.curImgFtr, curLane, faint=faint, resized=self.resized)
            if newLeftLine.detected and newLeftLine.lineClassified:
                # print("adding left lane")
                newLeftLine.findBottomOfLine(self.curImgFtr)
                self.lines.insert(0, newLeftLine)
                for lane in self.lanes:
                    leftIdx, rightIdx = lane.getLineIndex()
                    lane.setLineIndex(leftIdx + 1, rightIdx + 1)
                newLane = Lane(self.x, self.y,
                               self.projMgr.projectedX,
                               self.projMgr.projectedY,
                               self.maskDelta, self.lines,
                               maskvalue=curLane.maskvalue - 28)
                newLane.adjacentRLane = curLane
                newLane.adjacentRight = True
                newLane.adjacentLeft = newLeftLine.adjacentLeft
                newLane.adjacentLLane = None
                newLane.leftprojection = newLeftLine.applyLineMask(
                    self.curImgFtr.getEdgeProjection())
                newLane.rightprojection = curLane.leftprojection
                self.lanes.insert(0, newLane)
                curLane.adjacentLLane = newLane
                curLane.adjacentLeft = True
                self.mainLaneIdx += 1
            else:
                # print("left lane not detected")
                curLane.adjacentLeft = False

    def addLaneRight(self, curLane):
        faint = (curLane.maskvalue / 128)
        if self.resized:
            faint *= 0.75
        if ((faint > 0.5 or
                (self.resized and faint > 0.4)) and
                curLane.adjacentRight and
                curLane.adjacentRLane is None):
            # print("threshold:", faint)
            newRightLine = Line(self.right, self.x, self.y,
                                self.projMgr.projectedX,
                                self.projMgr.projectedY, self.maskDelta)
            newRightLine.createPolyFitRight(
                self.curImgFtr,  curLane, faint=faint, resized=self.resized)
            if newRightLine.detected and newRightLine.lineClassified:
                # print("adding right lane")
                newRightLine.findBottomOfLine(self.curImgFtr)
                self.lines.append(newRightLine)
                newLane = Lane(self.x, self.y,
                               self.projMgr.projectedX,
                               self.projMgr.projectedY,
                               self.maskDelta, self.lines,
                               len(self.lines) - 2,
                               len(self.lines) - 1,
                               curLane.maskvalue - 24)
                newLane.adjacentLLane = curLane
                newLane.adjacentRLane = None
                newLane.adjacentLeft = True
                newLane.adjacentRight = newRightLine.adjacentRight
                newLane.leftprojection = curLane.rightprojection
                newLane.rightprojection = newRightLine.applyLineMask(
                    self.curImgFtr.getEdgeProjection())
                self.lanes.append(newLane)
                curLane.adjacentRLane = newLane
                curLane.adjacentRight = True
            else:
                # print("right lane not detected")
                curLane.adjacentRight = False

    def updateLaneLeft(self, curLane):
        if curLane.adjacentLeft and curLane.adjacentLLane is not None:
            leftLine = curLane.adjacentLLane.lines[curLane.adjacentLLane.left]
            leftLine.updatePolyFitLeft(curLane)
            leftLine.findBottomOfLine(self.curImgFtr)

    def updateLaneRight(self, curLane):
        if curLane.adjacentRight and curLane.adjacentRLane is not None:
            rightLine = curLane.adjacentRLane.lines[
                curLane.adjacentRLane.right]
            rightLine.updatePolyFitRight(curLane)
            rightLine.findBottomOfLine(self.curImgFtr)

    def findLanes(self, img, resized=False):
        self.resized = resized
        if self.curFrame is None:
            self.curFrame = 0
        else:
            self.curFrame += 1
            # print("############################################")
            # print("Frame #", self.curFrame)
            # print("############################################")

        if resized and self.curFrame < 2:
            # self.threshold = 5.0
            # self.threshold = 10.0
            # self.threshold = 12.0
            # self.threshold = -0.25
            self.threshold = 30.0
            self.vehicleDetection.set_threshold(self.threshold)

        self.curImgFtr = ImageFilters(self.projMgr.camCal,
                                      self.projMgr.projectedX,
                                      self.projMgr.projectedY,
                                      debug=True)
        self.projMgr.set_image_filter(self.curImgFtr)
        self.curImgFtr.imageQ(img)
        mainLane = self.lanes[self.mainLaneIdx]

        # Experimental
        # if self.curImgFtr.visibility < -30 or \
        #   self.curImgFtr.skyImageQ == 'Sky Image: overexposed' or \
        #   self.curImgFtr.skyImageQ == 'Sky Image: underexposed':
        #    self.curImgFtr.balanceEx()

        # detected cloudy condition!
        if (self.curImgFtr.skyText == 'Sky Condition: cloudy' and
                self.curFrame == 0):
            self.cloudyMode = True

        # choose a default filter based on weather condition
        # line class can update filter based on what it wants too (different
        # for each lane line).
        if self.cloudyMode:
            self.curImgFtr.applyFilter3()
            self.maskDelta = 10
            mainLane.setMaskDelta(self.maskDelta)
        elif (self.curImgFtr.skyText == 'Sky Condition: clear' or
              self.curImgFtr.skyText == 'Sky Condition: tree shaded'):
            self.curImgFtr.applyFilter2()
        elif self.curFrame < 2:
            self.curImgFtr.applyFilter4()
        else:
            self.curImgFtr.applyFilter5()

        self.curImgFtr.horizonDetect(debug=True)

        # low confidence?
        if mainLane.confidence() < 0.5 or self.curFrame < 2:
            self.restartCount += 1
            self.projMgr.findInitialRoadCorners(self.curImgFtr)

            # Is our destination projection too high for our source?
            # update destination rect
            # experimental
            projectionTopPixel = \
                self.curImgFtr.projectionThrowDistanceDetect(debug=self.debug)

            # Experimental
            # print("projectionTopPixel", projectionTopPixel)
            # if projectionTopPixel > self.projMgr.projectedY/5:
            #    self.resetProjectionCount += 1
            #    self.lastTop = projectionTopPixel
            #    self.projMgr.resetDestTop(self.lastTop)

            #    # this is important enough that we want to reproject
            #    # right away!
            #    self.projMgr.project(self.curImgFtr,
            #                         self.lastLeftRightOffset, sameFrame=True)
            #    projectionTopPixel = \
            #        self.curImgFtr.projectionThrowDistanceDetect(
            #            debug=self.debug)
            #    # print("2nd Time projectionTopPixel", projectionTopPixel)

            # we may be too conservative...
            # Just move up the projection in the next frame
            # elif projectionTopPixel == 0 and self.lastTop > 10:
            #    self.resetProjectionCount += 1
            #    self.lastTop -= 10
            #    self.projMgr.resetDestTop(self.lastTop)

            # Use visibility to lower the FoV
            # adjust source for perspective projection accordingly
            self.initialGradient = self.projMgr.curGradient
            if (self.curImgFtr.horizonFound and
                    self.projMgr.curGradient is not None):
                self.roadHorizonGap = (self.projMgr.curGradient -
                                       self.curImgFtr.roadhorizon)
                newTop = self.curImgFtr.roadhorizon + self.roadHorizonGap
                self.projMgr.setSrcTop(
                    newTop - self.curImgFtr.visibility,
                    self.curImgFtr.visibility)

            # find main lane lines left and right
            self.lastNEdges = self.curImgFtr.curRoadEdge
            mainLane.findInitialLines(self.curImgFtr, resized=self.resized)

            # find lane lines left and right
            curLane = mainLane
            while (curLane is not None and
                   curLane.adjacentLeft and
                   curLane.adjacentLLane is None):
                self.addLaneLeft(curLane)
                curLane = curLane.adjacentLLane
            curLane = mainLane
            while (curLane is not None and
                   curLane.adjacentRight and
                   curLane.adjacentRLane is None):
                self.addLaneRight(curLane)
                curLane = curLane.adjacentRLane
        else:
            # Apply Boosting...
            # For Challenges ONLY
            if self.cloudyMode:
                if self.curImgFtr.skyText == 'Sky Condition: cloudy':
                    self.boosting = 0.4
                    self.lastNEdges = self.curImgFtr.miximg(
                        self.curImgFtr.curRoadEdge, self.lastNEdges, 1.0, 0.4)
                else:
                    self.boosting = 1.0
                    self.lastNEdges = self.curImgFtr.miximg(
                        self.curImgFtr.curRoadEdge, self.lastNEdges, 1.0, 1.0)
                self.curImgFtr.curRoadEdge = self.lastNEdges
            elif (self.curImgFtr.skyText ==
                  'Sky Condition: surrounded by trees'):
                self.boosting = 0.0
                # self.lastNEdges = self.curImgFtr.miximg(
                #     self.curImgFtr.curRoadEdge, self.lastNEdges, 1.0, 0.4)
                # self.curImgFtr.curRoadEdge = self.lastNEdges

            # project the new frame to a plane for further analysis.
            self.projMgr.project(self.curImgFtr, self.lastLeftRightOffset)

            # Is our destination projection too high for our source? update
            # destination rect
            projectionTopPixel = self.curImgFtr.projectionThrowDistanceDetect(
                debug=self.debug)
            # experimental
            # print("projectionTopPixel", projectionTopPixel)
            # if projectionTopPixel > self.projMgr.projectedY/4:
            #    self.resetProjectionCount += 1
            #    self.lastTop = projectionTopPixel
            #    self.projMgr.resetDestTop(self.lastTop)

            #    # this is important enough that we want to reproject
            #    # right away!
            #    self.projMgr.project(self.curImgFtr,
            #                         self.lastLeftRightOffset, sameFrame=True)
            #    projectionTopPixel = \
            #        self.curImgFtr.projectionThrowDistanceDetect(
            #            debug=self.debug)
            #    print("2nd Time projectionTopPixel", projectionTopPixel)

            # we may be too conservative...
            # Just move up the projection in the next frame
            # elif projectionTopPixel == 0 and self.lastTop > 10:
            #    self.resetProjectionCount += 1
            #    self.lastTop -= 10
            #    self.projMgr.resetDestTop(self.lastTop)

            # find main lane lines left and right again
            mainLane.findExistingLines(self.curImgFtr)

            # Update location for FoV
            # if mainLane.leftLineLastTop is not None and
            #    mainLane.rightLineLastTop is not None and
            #    self.curImgFtr.visibility < -30:
            #    self.lastLeftRightOffset = int((self.x/2) -
            #        (mainLane.leftLineLastTop[0] +
            #         mainLane.rightLineLastTop[0])/2)
            #    print("lastLeftRightOffset: ", self.lastLeftRightOffset)

            # update lane lines left and right
            curLane = mainLane
            while curLane.adjacentLeft and curLane.adjacentLLane is not None:
                self.updateLaneLeft(curLane)
                curLane = curLane.adjacentLLane
            curLane = mainLane
            while curLane.adjacentRight and curLane.adjacentRLane is not None:
                self.updateLaneRight(curLane)
                curLane = curLane.adjacentRLane

        # Update Stats and Top points for next frame.
        self.leftLineLastTop = mainLane.leftLineLastTop
        self.rightLineLastTop = mainLane.rightLineLastTop
        self.leftLinePoints = mainLane.leftLinePoints
        self.rightLinePoints = mainLane.rightLinePoints

        # Update road statistics for display
        self.lineBasePos = mainLane.getLineBasePos()
        self.radiusOfCurvature, self.roadStraight = \
            mainLane.getRadiusOfCurvature()

        # Scan for vehicles
        self.vehicleScan = np.copy(self.curImgFtr.getRoadProjection())
        if self.curFrame > 1:
            if self.curFrame == 2:
                if resized:
                    # self.threshold = 12.0
                    # self.threshold = 15.0
                    self.threshold = 30.0
                    self.vehicleDetection.set_threshold(self.threshold)
                else:
                    self.threshold = 25.0
                    self.vehicleDetection.set_threshold(self.threshold)

            self.roadGrid = self.vehicleDetection.slidingWindows(
                self.lines, self.mainLaneIdx, False)
            allPossibleWindows = self.roadGrid.getAllWindows()
        else:
            self.roadGrid = self.vehicleDetection.slidingWindows(
                self.lines, self.mainLaneIdx, True)
            allPossibleWindows = self.roadGrid.getAllWindows()

        if self.scrType & 16 == 16:
            self.vehicleDetection.collectData(
                self.curFrame, self.vehicleScan, allPossibleWindows)
        else:
            # add our vehicles into the search space
            # to reduce search area
            for vehIdx in range(len(self.vehicles)):
                if self.vehicles[vehIdx].mode > 2:
                    self.roadGrid = self.vehicles[vehIdx].updateVehicle(
                        self.roadGrid, np.copy(self.curImgFtr.curImage))
                elif len(self.vehicles[vehIdx].windows) > 0:
                    lane = self.vehicles[vehIdx].lane
                    yidx = self.vehicles[vehIdx].yidx
                    window = self.vehicles[vehIdx].windows[0]
                    # print("vehIdx: ", lane, yidx, window, vehIdx)
                    self.roadGrid.insertTrackedObject(
                        lane, yidx, window, vehIdx)

            # detect any vehicles
            self.roadGrid = self.vehicleDetection.detectVehicles(
                self.vehicleScan, self.roadGrid)

            # get updated location for our existing vehicles
            # and check to make sure our vehicles are still there
            vehicleDropList = []
            for vehIdx in range(len(self.vehicles)):
                self.roadGrid = self.vehicles[vehIdx].updateVehicle(
                    self.roadGrid, np.copy(self.curImgFtr.curImage))
                if not self.vehicleTracking.isVehicleThere(
                        np.copy(self.curImgFtr.curImage),
                        self.roadGrid, self.mainLaneIdx,
                        self.vehicles, vehIdx):
                    vehicleDropList.append(self.vehicles[vehIdx])

            self.possibleVehicleWindows = \
                self.roadGrid.getFoundAndNotOccludedWindows()
            self.hiddenVehicleWindows = \
                self.roadGrid.getOccludedWindows()

            # did we lose any vehicles?
            for dropped_vehicle in vehicleDropList:
                self.vehicles.remove(dropped_vehicle)

            # did we get any new vehicles?
            for objIdx in range(self.roadGrid.getNumObjects()):
                found = False
                boxes = self.roadGrid.getObjectList(objIdx)
                for vehIdx in range(len(self.vehicles)):
                    if self.vehicles[vehIdx].objectIsVehicle(
                            boxes, self.roadGrid):
                        self.roadGrid.setVehicle(boxes, vehIdx)
                        found = True
                if not found:
                    ID = len(self.vehicles)
                    vehicle = Vehicle(
                        ID, self.lanes, self.projMgr, self.roadGrid, objIdx,
                        np.copy(self.curImgFtr.curImage), self.mainLaneIdx)

                    # TODO - need depth ordering here..
                    self.vehicles.append(vehicle)

            # re-index our vehicles
            for vehIdx in range(len(self.vehicles)):
                self.vehicles[vehIdx].vehIdx = vehIdx

        # Experimental
        # adjust source for perspective projection accordingly
        # attempt to dampen bounce
        # if (self.leftLineLastTop is not None and
        #     self.rightLineLastTop is not None):
        #    x = int((self.x -
        #            (self.leftLineLastTop[0] +
        #             self.rightLineLastTop[0]))/4)
        #    self.projMgr.setSrcTopX(x)

        # create road mask polygon for reprojection back onto perspective view.
        roadmask = np.zeros(
            (self.projMgr.projectedY, self.projMgr.projectedX), dtype=np.uint8)

        leftLinemask = np.zeros(
            (self.projMgr.projectedY, self.projMgr.projectedX), dtype=np.uint8)
        rightLinemask = np.zeros(
            (self.projMgr.projectedY, self.projMgr.projectedX), dtype=np.uint8)
        for i in range(len(self.lanes)):
            self.lanes[i].drawLanePoly(roadmask)
            leftLinemask = self.curImgFtr.miximg(
                leftLinemask, self.lanes[i].leftprojection, 1.0, 1.0)
            rightLinemask = self.curImgFtr.miximg(
                rightLinemask, self.lanes[i].rightprojection, 1.0, 1.0)

        self.roadsurface[:, :, 0] = leftLinemask
        self.roadsurface[:, :, 1] = roadmask
        self.roadsurface[:, :, 2] = rightLinemask

        # generate wireframe scanner rendering.
        self.specialProjectedEffects = self.curImgFtr.miximg(
            self.roadsurface * 0, self.specialProjectedEffects, 1.0, 0.9)
        self.specialPerspectiveEffects = \
            self.curImgFtr.miximg(
                self.specialPerspectiveEffects * 0,
                self.specialPerspectiveEffects, 1.0, 0.9)
        self.sweepLane = self.projMgr.sweep(
            self.specialProjectedEffects, self.curFrame, self.lines)

        self.roadsquares = self.projMgr.wireframe(
            self.roadsurface, self.curFrame, self.lanes[self.mainLaneIdx])
        self.roadsurface = self.curImgFtr.miximg(
            self.roadsurface, self.specialProjectedEffects, 1.0, 1.0)
        self.projMgr.sweep(self.roadsurface, self.curFrame, self.lines)

        # draw the possible vehicles detected
        self.vehicleDetection.draw_boxes(
            self.roadsurface, self.possibleVehicleWindows)
        self.vehicleDetection.draw_boxes(
            self.roadsurface, self.hiddenVehicleWindows, color=(255, 0, 0))

        # draw closing circle when we first initialize
        # and then track
        for vehicle in self.vehicles:
            vehicle.drawClosingCircle(
                self.sweepLane,
                self.specialProjectedEffects,
                self.roadsurface)

        # unwarp the roadsurface
        self.roadunwarped = self.projMgr.curUnWarp(
            self.curImgFtr, self.roadsurface)

        # print("self.roadunwarped:", self.roadunwarped.shape)
        # print("self.curImgFtr.curImage:", self.curImgFtr.curImage.shape)

        # create the final image
        self.final = self.curImgFtr.miximg(
            self.curImgFtr.curImage, self.roadunwarped, 0.95, 0.75)

        # add vehicle detection and tracking visuals
        font = cv2.FONT_HERSHEY_COMPLEX
        self.projMgr.drawRoadSquares(self.final, self.roadsquares)

        self.final = self.curImgFtr.miximg(
            self.final, self.specialPerspectiveEffects, 1.0, 1.0)

        # vehicle info - left or right side?
        if self.mainLaneIdx == 0:
            startx = 30
        else:
            startx = self.x - 285 - 300

        # draw vehicle info
        for vehIdx in range(len(self.vehicles)):
            if self.vehicles[vehIdx].contourInPerspective is not None:
                self.specialPerspectiveEffects = self.curImgFtr.miximg(
                    self.specialPerspectiveEffects,
                    self.vehicles[vehIdx].contourInPerspective,
                    1.0, 0.20)
            self.vehicles[vehIdx].drawScanning(
                self.specialPerspectiveEffects, self.final)

            # calculate vehicle info positions
            y1 = 150 + 120*vehIdx
            y2 = y1 + 80
            # print("y1, y2: ", y1, y2)
            if y2 < 720:
                if len(self.vehicles[vehIdx].selfie.shape) > 2:
                    self.final[y1:y2, startx:startx+240] = \
                        cv2.resize(
                            self.vehicles[vehIdx].selfie,
                            (240, 80), interpolation=cv2.INTER_AREA)
                else:
                    selfie = self.vehicles[vehIdx].selfie
                    self.final[y1:y2, startx:startx+240] = \
                        cv2.resize(
                        np.dstack((selfie, selfie, selfie)),
                        (240, 80), interpolation=cv2.INTER_AREA)

                Text = self.vehicles[vehIdx].getTextStats()
                i = 0
                for text in Text.split('\n'):
                    if i == 0:
                        cv2.putText(
                            self.final, text,
                            (startx, y1-10), font, 0.60,
                            self.vehicles[vehIdx].statusColor, 2)
                    else:
                        cv2.putText(
                            self.final, text,
                            (startx+255, y1-10+15*(i-1)),
                            font, 0.50,
                            (192, 192, 192), 2)
                    i += 1

        # draw dots and polyline
        if self.debug:
            # our own diag screen
            self.diag1 = np.copy(self.projMgr.diag4)
            cv2.putText(self.projMgr.diag4, 'Frame: %d' %
                        (self.curFrame), (30, 30), font, 1, (255, 255, 0), 2)
            cv2.putText(self.diag1, 'Frame: %d' %
                        (self.curFrame), (30, 30), font, 1, (255, 0, 0), 2)
            for i in range(len(self.lines)):
                self.lines[i].scatter_plot(self.diag1)
                self.lines[i].polyline(self.diag1)
                self.lines[i].scatter_plot(self.projMgr.diag4)
                self.lines[i].polyline(self.projMgr.diag4)
                # self.projMgr.diag3 = \
                #     self.lines[i].applyReverseLineMask(self.projMgr.diag3)

                # draw bottom of lane line point if not at bottom
                if self.lines[i].bottomProjectedY < self.projMgr.projectedY:
                    cv2.circle(self.projMgr.diag4,
                               (int(self.lines[i].pixelBasePos),
                                int(self.lines[i].bottomProjectedY)),
                               10, (64, 64, 255), 10)
                    linetext = "x%d,y%d: %d,%d" % \
                               (i, i,
                                int(self.lines[i].pixelBasePos),
                                int(self.lines[i].bottomProjectedY))
                    if self.lines[i].side == self.left:
                        cv2.putText(self.projMgr.diag4, linetext,
                                    (int(self.lines[i].pixelBasePos - 275),
                                     int(self.lines[i].bottomProjectedY) - 15),
                                    font, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(self.projMgr.diag4, linetext,
                                    (int(self.lines[i].pixelBasePos + 25),
                                     int(self.lines[i].bottomProjectedY) - 15),
                                    font, 1, (0, 0, 255), 2)
                text = 'Line %d: %d count,  %4.1f%% confidence, detected: %r'
                cv2.putText(self.projMgr.diag4, text %
                            (i, len(self.lines[i].allY),
                             self.lines[i].confidence * 100,
                             self.lines[i].detected),
                            (30, 60 + 90 * i), font, 1, (255, 255, 0), 2)
                if (self.lines[i].radiusOfCurvature is not None and
                        self.lines[i].lineBasePos is not None):
                    text = 'Line %d: RoC: %fm, DfVC: %fcm'
                    cv2.putText(self.projMgr.diag4, text %
                                (i, self.lines[i].radiusOfCurvature,
                                 self.lines[i].lineBasePos * 100),
                                (30, 90 + 90 * i), font, 1, (255, 255, 0), 2)
                elif self.lines[i].radiusOfCurvature is not None:
                    text = 'Line %d: RoC: %fm, DfVC: UNKNOWN'
                    cv2.putText(self.projMgr.diag4, text %
                                (i, self.lines[i].radiusOfCurvature),
                                (30, 90 + 90 * i), font, 1, (255, 255, 0), 2)
                elif self.lines[i].lineBasePos is not None:
                    text = 'Line %d: RoC: UNKNOWN, DfVC: %fcm'
                    cv2.putText(self.projMgr.diag4, text %
                                (i, self.lines[i].lineBasePos * 100),
                                (30, 90 + 90 * i), font, 1, (255, 255, 0), 2)
                else:
                    text = 'Line %d: RoC: UNKNOWN, DfVC: UNKNOWN'
                    cv2.putText(self.projMgr.diag4, text % (i),
                                (30, 90 + 90 * i), font, 1, (255, 255, 0), 2)
                linetext = 'Line %d: is %s %s, '
                linetext = linetext + 'more left: %r  more right: %r'
                linetext = linetext % (i, self.lines[i].lineType,
                                       self.lines[i].line_color,
                                       self.lines[i].adjacentLeft,
                                       self.lines[i].adjacentRight)
                cv2.putText(self.projMgr.diag4, linetext,
                            (30, 120 + 90 * i), font, 1, (255, 255, 0), 2)

            if self.boosting > 0.0:
                y = 90 * len(self.lines)
                cv2.putText(self.projMgr.diag4, 'Boosting @ %f%%' % (
                    self.boosting),
                    (30, 60 + y), font, 1, (128, 128, 192), 2)

            # print("self.roadsurface:", self.roadsurface.shape)
            # print("self.roadunwarped:", self.roadunwarped.shape)
            # print("self.projMgr.diag4:", self.projMgr.diag4.shape)
            # print("self.projMgr.diag2:", self.projMgr.diag2.shape)

            self.projMgr.diag4 = self.curImgFtr.miximg(
                self.projMgr.diag4, self.roadsurface, 1.0, 2.0)
            self.projMgr.diag2 = self.curImgFtr.miximg(
                self.projMgr.diag2, self.roadunwarped, 1.0, 0.5)
            self.projMgr.diag1 = self.curImgFtr.miximg(
                self.projMgr.diag1, self.roadunwarped[
                    self.mid:self.y, :, :], 1.0, 2.0)

            # generate the window positions
            if self.scrType & 5 == 5:
                windows = self.vehicleDetection.slidingWindows(
                    self.lines, self.mainLaneIdx, False)
                print("sentinal_windows=", windows)
                # self.vehicleDetection.draw_boxes(self.projMgr.diag3, windows)
            elif self.scrType & 4 == 4:
                windows = self.vehicleDetection.slidingWindows(
                    self.lines, self.mainLaneIdx, True)
                print("complete_scan_windows=", windows)
                # self.vehicleDetection.draw_boxes(self.projMgr.diag3, windows)
            else:
                if mainLane.lines[mainLane.left].adjacentLeft:
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.left].pixelBasePos, 0)
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.left].pixelBasePos,
                        -mainLane.distance)
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.left].pixelBasePos,
                        -mainLane.distance * 2)
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.left].pixelBasePos,
                        -mainLane.distance * 3)
                else:
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.left].pixelBasePos, 0)

                if mainLane.lines[mainLane.right].adjacentRight:
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.right].pixelBasePos, 0)
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.right].pixelBasePos,
                        mainLane.distance)
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.right].pixelBasePos,
                        mainLane.distance * 2)
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.right].pixelBasePos,
                        mainLane.distance * 3)
                else:
                    self.projMgr.draw_estimated_lane_line_location(
                        self.projMgr.diag3,
                        mainLane.lines[mainLane.right].pixelBasePos, 0)
                self.vehicleDetection.draw_boxes(
                    self.projMgr.diag3, self.possibleVehicleWindows)

            # let's try to draw some cubes...
            # self.projMgr.drawAxisOnLane(self.projMgr.diag2)
            # self.projMgr.drawCalibrationCube(self.projMgr.diag2)
            # self.projMgr.drawCubes(self.projMgr.diag2,
            #                        self.possibleVehicleWindows)

            self.projMgr.drawRoadSquares(self.projMgr.diag2, self.roadsquares)

            self.projMgr.diag2 = self.curImgFtr.miximg(
                self.projMgr.diag2, self.specialPerspectiveEffects, 1.0, 1.0)

            # vehicle info - left or right side?
            if self.mainLaneIdx == 0:
                startx = 30
            else:
                startx = self.x - 285 - 300

            # diagnostics vehicle info
            for vehIdx in range(len(self.vehicles)):
                if self.vehicles[vehIdx].contourInPerspective is not None:
                    self.specialPerspectiveEffects = self.curImgFtr.miximg(
                        self.specialPerspectiveEffects,
                        self.vehicles[vehIdx].contourInPerspective,
                        1.0, 0.20)
                self.vehicles[vehIdx].drawScanning(
                    self.specialPerspectiveEffects, self.projMgr.diag2)

                # experimental
                # draw cube cross-section - if enough points
                # if len(self.vehicles[vehIdx].cube_intersect) == 4:
                #     print("cube_intersect: ",
                #         self.vehicles[vehIdx].cube_intersect)
                #     cv2.drawContours(
                #         self.projMgr.diag2,
                #         [self.vehicles[vehIdx].cube_intersect[:4]],
                #         -1, (255, 0, 255), 3)

                # calculate vehicle info positions
                y1 = 275 + 120*vehIdx
                y2 = y1 + 80
                # print("y1, y2: ", y1, y2)
                if y2 < 720:
                    if len(self.vehicles[vehIdx].selfie.shape) > 2:
                        self.projMgr.diag2[y1:y2, startx:startx+240] = \
                            cv2.resize(
                                self.vehicles[vehIdx].selfie,
                                (240, 80), interpolation=cv2.INTER_AREA)
                    else:
                        selfie = self.vehicles[vehIdx].selfie
                        self.projMgr.diag2[y1:y2, startx:startx+240] = \
                            cv2.resize(
                                np.dstack((selfie, selfie, selfie)),
                                (240, 80), interpolation=cv2.INTER_AREA)

                    font = cv2.FONT_HERSHEY_COMPLEX
                    Text = self.vehicles[vehIdx].getTextStats()
                    i = 0
                    for text in Text.split('\n'):
                        if i == 0:
                            cv2.putText(
                                self.projMgr.diag2, text,
                                (startx, 265+120*vehIdx), font, 0.60,
                                self.vehicles[vehIdx].statusColor, 2)
                        else:
                            cv2.putText(
                                self.projMgr.diag2, text,
                                (startx+255, 265+120*vehIdx+15*(i-1)),
                                font, 0.50,
                                (192, 192, 192), 2)
                        i += 1

    def drawLaneStats(self, color=(224, 192, 0)):
        font = cv2.FONT_HERSHEY_COMPLEX
        if self.roadStraight:
            text = 'Estimated lane curvature: road nearly straight'
            cv2.putText(self.final, text,
                        (30, 60), font, 1, color, 2)
        elif self.radiusOfCurvature > 0.0:
            text = 'Estimated lane curvature: center is %fm to the right'
            cv2.putText(self.final, text % (
                self.radiusOfCurvature), (30, 60), font, 1, color, 2)
        else:
            text = 'Estimated lane curvature: center is %fm to the left'
            cv2.putText(self.final, text %
                        (-self.radiusOfCurvature), (30, 60), font, 1, color, 2)

        if self.lineBasePos < 0.0:
            text = 'Estimated left of center: %5.2fcm'
            cv2.putText(self.final, text %
                        (-self.lineBasePos * 100), (30, 90), font, 1, color, 2)
        elif self.lineBasePos > 0.0:
            text = 'Estimated right of center: %5.2fcm'
            cv2.putText(self.final, text % (
                self.lineBasePos * 100), (30, 90), font, 1, color, 2)
        else:
            text = 'Estimated at center of road'
            cv2.putText(self.final, text, (30, 90), font, 1, color, 2)

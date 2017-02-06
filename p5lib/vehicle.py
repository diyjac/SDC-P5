#!/usr/bin/python
"""
vehicle.py: version 0.1.0

History:
2017/01/31: Use webcolor to name rgb color of vehicle:
            http://stackoverflow.com/questions/9694165/\
            convert-rgb-color-to-english-color-name-like-green
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/17: Initial version converted to a class
"""

import numpy as np
import cv2
import webcolors


class Vehicle():

    # initialization
    def __init__(
            self, ID, lanes, projMgr, roadGrid,
            objIdx, perspectiveImage, mainLaneIdx):
        self.vehIdx = ID
        self.vehStr = '%d' % (ID)
        self.projMgr = projMgr
        self.roadGrid = roadGrid
        self.projectedX = projMgr.projectedX
        self.projectedY = projMgr.projectedY
        self.middle = self.projectedX/2
        self.x = projMgr.x
        self.y = projMgr.y
        self.lanes = lanes
        self.mainLaneIdx = mainLaneIdx
        self.selfieX = 640
        self.selfieY = 240

        # special effects
        # closing circle sweep
        self.sweepDone = False
        self.sweepDeltaFrame = 0
        # scanning sweep
        self.scanDone = False
        self.scanDeltaFrame = 0

        # this would be the width, height, depth of the vehicle
        # if we could see it in birds-eye view
        # we will calculate this during 3D reconstruction
        self.boundingShape = np.array([0.0, 0.0, 0.0]).astype(np.float32)

        # estimated x,y location in birds-eye projected view
        self.xcenter = 0
        self.ycenter = 0

        # estimated initial size 64x64
        self.deltaX = 32
        self.deltaY = 32

        # use the projection manager's estimated height - our z value
        self.z = projMgr.z * 1.2

        # initial windows during detection
        self.lastObjList = roadGrid.getObjectList(objIdx)
        self.initialWindows = roadGrid.getObjectListWindows(objIdx)

        # windows
        self.windows = roadGrid.getFoundAndNotOccludedWindowsInObject(objIdx)

        # boxes
        self.boxes = roadGrid.getFoundAndNotOccludedBoxesInObject(objIdx)

        # lane and location in the voxel grid the vehicle is on
        if len(self.boxes) > 0:
            self.lane, self.yidx = roadGrid.gridCoordinates(self.boxes[0])
            self.box = self.boxes[0]
            self.xcenter, self.ycenter = self.windowCenter(
                roadGrid.getBoxWindow(self.box))

        else:
            self.lane = None
            self.yidx = None
            self.box = None
            self.initialMaskVector = None

        # was the vehicle detected in the last iteration?
        self.detected = False

        # percentage confidence
        self.detectConfidence = 0.0
        self.detectConfidence_base = 0.0
        self.initFrames = 0
        self.graceFrames = 10
        self.exitFrames = 0
        self.traveled = False

        # contour of vehicle
        self.contourInPerspective = None

        # mask of vehicle
        self.maskedProfile = None
        self.vehicleHeatMap = np.zeros(
            (self.selfieY, self.selfieX), dtype=np.float32)
        self.vehicleMaskInPerspective = None

        # vehicle status and statistics
        self.vehicleClassified = False
        self.color = (0, 0, 0)
        self.colorpoints = 0
        self.webColorName = None
        self.statusColor = None
        self.status = "Not Found"
        self.vehicleInLane = None
        self.previousboxes = []

        # could be one of:
        # DetectionPhase:
        #     0:Initialized
        #     1:DetectionConfirmed
        # TrackingPhase:
        #     2:Scanning
        #     3:VehicleAcquired
        #     4:VehicleLocked
        #     5:VehicleOccluded
        #     6:VehicleLeaving
        #     7:VehicleLosted
        self.mode = 0

        # array of 3d and 2d points for bounding cube
        # do the calculations for the 2d and 3d bounding box
        self.cube3d, self.cube2d = \
            self.calculateRoughBoundingCubes(self.windows)

        # create the rough masked image for projection.
        self.maskVertices, self.maskedImage = \
            self.calculateMask(np.copy(perspectiveImage))

        # project the image for verification
        self.selfie = self.takeProfileSelfie(self.maskedImage)

    # update vehicle status before tracking.
    def updateVehicle(
            self, roadGrid, perspectiveImage, x=None, y=None, lane=None):
        self.roadGrid = roadGrid
        if lane is not None:
            self.lane = lane

        # lane and location in the voxel grid the vehicle is on
        if x is not None and y is not None and self.lane is not None:
            self.ycenter = y
            self.xcenter = self.lanes[self.lane].calculateXCenter(y)
            self.window = \
                ((self.xcenter - self.deltaX, self.ycenter - self.deltaY),
                 (self.xcenter + self.deltaX, self.ycenter + self.deltaY))
            self.windows = [self.window]
            if lane is not None:
                self.lane = lane
            if self.lane is not None:
                yidx = self.roadGrid.calculateObjectPosition(
                    self.lane, self.ycenter)
                if yidx > 0:
                    self.yidx = yidx
            self.box = self.roadGrid.getKey(self.lane, self.yidx)
            self.boxes = [self.box]
            self.roadGrid.insertTrackedObject(
                self.lane, self.yidx, self.window, self.vehIdx, tracking=True)

        elif self.mode > 2 and self.mode < 7:
            # for testing without tracking.
            # self.ycenter -= 0.5
            self.xcenter = self.lanes[self.lane].calculateXCenter(self.ycenter)
            self.window = \
                ((self.xcenter - self.deltaX, self.ycenter - self.deltaY),
                 (self.xcenter + self.deltaX, self.ycenter + self.deltaY))
            self.windows = [self.window]
            if lane is not None:
                self.lane = lane
            if self.lane is not None:
                yidx = self.roadGrid.calculateObjectPosition(
                    self.lane, self.ycenter)
                if yidx > 0:
                    self.yidx = yidx
            newbox = self.roadGrid.getKey(self.lane, self.yidx)

            # save last ten voxels for voxel trigger subpression
            if newbox != self.box:
                self.previousboxes.insert(0, self.box)
                self.previousboxes = self.previousboxes[:10]
                self.box = newbox
            self.boxes = [self.box]
            for oldbox in self.previousboxes:
                self.roadGrid.setOccluded(oldbox)
            self.roadGrid.insertTrackedObject(
                self.lane, self.yidx, self.window, self.vehIdx, tracking=True)

        else:
            # initial windows during detection
            # print("self.roadGrid.vehicle_list",
            #       self.vehStr, self.roadGrid.vehicle_list)
            if self.vehStr in self.roadGrid.vehicle_list:
                self.box = self.roadGrid.vehicle_list[self.vehStr]
            else:
                self.roadGrid.vehicle_list[self.vehStr] = self.box

            # windows
            self.windows = \
                self.roadGrid.getFoundAndNotOccludedWindowsInVehicle(
                    self.vehIdx)

            # boxes
            self.boxes = \
                self.roadGrid.getFoundAndNotOccludedBoxesInVehicle(
                    self.vehIdx)

            if len(self.boxes) > 0:
                self.lane, self.yidx = \
                    self.roadGrid.gridCoordinates(self.box)
                self.xcenter, self.ycenter = self.windowCenter(
                    self.roadGrid.getBoxWindow(self.box))

        # was the vehicle detected in the last iteration?
        self.detected = True

        # This is automatic now.  Voxel will reject if not found.
        if self.mode == 0:
            self.mode = 1

        # array of 3d and 2d points for bounding cube
        # do the calculations for the 2d and 3d bounding box
        self.cube3d, self.cube2d = \
            self.calculateRoughBoundingCubes(self.windows)

        # create the rough masked image for projection.
        self.maskVertices, self.maskedImage = \
            self.calculateMask(np.copy(perspectiveImage))

        # project the image for verification
        self.selfie = self.takeProfileSelfie(self.maskedImage)
        return self.roadGrid

    # classify the vehicle by its main color components
    def closest_colour(self, requested_colour):
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    # get a name match for the closest color
    def get_colour_name(self, requested_colour):
        try:
            closest_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = self.closest_colour(requested_colour)
        return closest_name

    def modeColor(self):
        # unknown state black
        color = (0, 0, 0)

        # DetectionPhase:
        #     0:Initialized
        if self.mode == 0:
            # yellow
            self.statusColor = (255, 255, 0)
            self.status = "Initializing..."
        #     1:DetectionConfirmed
        elif self.mode == 1:
            # cyan
            self.statusColor = (0, 192, 192)
            self.status = "Detected!"

        # TrackingPhase:
        #     2:Scanning
        elif self.mode == 2:
            # blue
            self.statusColor = (0, 0, 255)
            self.status = "Scanning..."

        #     3:VehicleAcquired
        elif self.mode == 3:
            # white
            self.statusColor = (255, 255, 255)
            self.status = "Vehicle Acquired"

        #     4:VehicleLocked
        elif self.mode == 4:
            # green
            self.statusColor = (0, 255, 0)
            self.status = "Vehicle Locked"

        #     5:VehicleOccluded
        elif self.mode == 5:
            # orange
            self.statusColor = (255, 165, 0)
            self.status = "Vehicle Occluded"

        #     6:VehicleLeaving
        elif self.mode == 6:
            # red
            self.statusColor = (255, 0, 0)
            self.status = "Vehicle Leaving..."

        #     7:VehicleLosted
        elif self.mode == 6:
            # black
            self.statusColor = (0, 0, 0)
            self.status = "Vehicle Losted"

        return self.statusColor

    def distance(self):
        xoffset = (self.middle - self.xcenter)
        yoffset = (self.projectedY - self.ycenter)
        return np.sqrt(xoffset*xoffset+yoffset*yoffset)

    def sortByDistance(self):
        return self.distance()

    # function to project the undistorted camera image to a plane
    # at a side of the vehicle bounding cube - to take a selfie!
    def unwarp_vehicle(self, img, src, dst, mtx):
        # Pass in your image, 4 source points:
        #     src = np.float32([[,],[,],[,],[,]])
        # and 4 destination points:
        #     dst = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        # use cv2.warpPerspective() to warp your image to a side
        # view of vehicle bounding box

        self.src2dstM = cv2.getPerspectiveTransform(src, dst)
        img_size = (self.selfieX, self.selfieY)
        warped = cv2.warpPerspective(
            img, self.src2dstM, img_size, flags=cv2.INTER_LINEAR)

        # warped = gray
        return warped, self.src2dstM

    # function to project the undistorted camera image to a plane at the side.
    # of a vehicle bounding cube - we will use this to project augmentation
    # back on to the vehicle.
    def unwarp_vehicle_back(self, img, src, dst, mtx):
        # Pass in your image, 4 source points:
        #     src = np.float32([[,],[,],[,],[,]])
        # and 4 destination points:
        #     dst = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        # use cv2.warpPerspective() to warp your image to a side
        # view of the vehicle bounding box.

        self.dst2srcM = cv2.getPerspectiveTransform(src, dst)
        img_size = (self.x, self.y)
        warped = cv2.warpPerspective(
            img, self.dst2srcM, img_size, flags=cv2.INTER_LINEAR)

        # warped = gray
        return warped, self.dst2srcM

    # function to find center of projection
    def findCenter(self, masked_projection):
        try:
            points = np.nonzero(masked_projection)
            x = int(np.average(points[1]))
            y = int(np.average(points[0]))
            # print("findCenter: ", x, y)
        except:
            h, w = masked_projection.shape[:2]
            x = int(w/2)
            y = int(h/2)
        return x, y

    # function to find center of max color
    def findMaxColor(self, masked_projection):
        xhistogram = np.sum(masked_projection.astype(np.float32), axis=0)
        yhistogram = np.sum(masked_projection.astype(np.float32), axis=1)
        x = np.argmax(xhistogram)
        y = np.argmax(yhistogram)
        # print("findMaxColor:", masked_projection.shape, "x,y", x, y)
        return x, y

    # function to find color of vehicle
    def sampleColor(self, img):
        # default to black
        red = 0
        green = 0
        blue = 0

        # experimental
        # get a center patch of the image
        # midw, midh = self.findMaxColor(img)
        midw, midh = self.findCenter(img)
        imgR = img[
            midh-20:midh+20,
            midw-40:midw+40, 0].astype(np.uint8)
        imgG = img[
            midh-20:midh+20,
            midw-40:midw+40, 1].astype(np.uint8)
        imgB = img[
            midh-20:midh+20,
            midw-40:midw+40, 2].astype(np.uint8)
        if imgR.shape[1] > 0 and imgG.shape[1] > 0 and imgB.shape[1] > 0:
            red1 = np.min(imgR)
            green1 = np.min(imgG)
            blue1 = np.min(imgB)
            red2 = np.max(imgR)
            green2 = np.max(imgG)
            blue2 = np.max(imgB)
            cv2.circle(img, (midw, midh), 22, (0, 0, 0), 2)
            cv2.circle(img, (midw, midh), 24, (255, 255, 255), 2)
        else:
            # get a center patch of the image
            h, w = img.shape[:2]
            midh = int(h/2)
            midw = int(w/2)
            imgR = img[
                midh-20:midh+20,
                midw-40:midw+40, 0].astype(np.uint8)
            imgG = img[
                midh-20:midh+20,
                midw-40:midw+40, 1].astype(np.uint8)
            imgB = img[
                midh-20:midh+20,
                midw-40:midw+40, 2].astype(np.uint8)
            red1 = np.min(imgR)
            green1 = np.min(imgG)
            blue1 = np.min(imgB)
            red2 = np.max(imgR)
            green2 = np.max(imgG)
            blue2 = np.max(imgB)
            cv2.circle(img, (midw, midh), 42, (0, 0, 0), 2)
            cv2.circle(img, (midw, midh), 44, (255, 255, 255), 2)

        # set the vehicle's color
        rgb1 = (red1, green1, blue1)
        rgbm = (
            int((red1+red2)/2),
            int((green1+green2)/2),
            int((blue1+blue2)/2))
        rgb2 = (red2, green2, blue2)
        colorpalet = np.array([[rgb1, rgbm, rgb2]]).reshape(3, 1, 3)
        vehicle_grays = \
            cv2.cvtColor(colorpalet.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if vehicle_grays[2] > 200:
            self.vehicle_rgb = rgb2
            self.vehicle_gray = vehicle_grays[2]
        elif vehicle_grays[1] < 55:
            self.vehicle_rgb = rgb1
            self.vehicle_gray = vehicle_grays[0]
        else:
            self.vehicle_rgb = rgbm
            self.vehicle_gray = vehicle_grays[1]
        self.webColorName = self.get_colour_name(self.vehicle_rgb)

    def getTextStats(self):
        meterDistance = self.distance() * self.projMgr.pixel2Meter()

        # check for bad lane setting
        if self.lane is None:
            lane = 'Unknown'
        else:
            lane = '%d' % (self.lane)
        if self.box is None:
            voxel = 'Unknown'
        else:
            voxel = ''.join(self.box.split('+'))

        text = 'Vehicle %d Visuals:\n' % (self.vehIdx + 1)
        text += 'color: %s\n' % (self.webColorName)
        text += 'Status: %s\n' % (self.status)
        text += 'occupies lane: %s\n' % (lane)
        text += 'tracking voxel: %s\n' % (voxel)
        text += 'tracking distance:\n'
        text += '   %fm' % (meterDistance)
        return text

    # calculate center of window
    def windowCenter(self, window):
        x = int((window[0][0] + window[1][0]) / 2)
        y = int((window[0][1] + window[1][1]) / 2)
        return (x, y)

    def vehicleInBox(self, box, roadgrid):
        if self.lane is not None:
            yidx = self.roadGrid.calculateObjectPosition(
                self.lane, self.ycenter)
            vehbox = roadgrid.getKey(self.lane, yidx)
            if vehbox == box:
                return True
        return False

    def objectIsVehicle(self, boxlist, roadgrid):
        if self.box is not None:
            if self.box in boxlist:
                return True
        else:
            if self.lane is not None:
                yidx = self.roadGrid.calculateObjectPosition(
                    self.lane, self.ycenter)
                vehbox = roadgrid.getKey(self.lane, yidx)
                if vehbox in boxlist:
                    return True
        return False

    # Augmentation Special Effects -
    # default full closing circle sweep takes
    # about two seconds 52 frames - video is 26fps
    def drawClosingCircle(
            self, sweepLane, projectionFX, roadProjection,
            color=[0, 0, 255], sweepedcolor=[128, 128, 255],
            sweepThick=5, fullsweepFrame=20):
        if self.lane == sweepLane:
            ccolor = sweepedcolor
        else:
            ccolor = color
        if not self.sweepDone:
            # calculate sweep radius
            radius = (fullsweepFrame -
                      (self.sweepDeltaFrame % fullsweepFrame)) * 10
            self.sweepDeltaFrame += 1

            # closingCircle sweep
            cv2.circle(
                projectionFX, (int(self.xcenter), int(self.ycenter)),
                radius, ccolor, 10)
            cv2.circle(
                roadProjection, (int(self.xcenter), int(self.ycenter)),
                radius, ccolor, 10)

            if self.sweepDeltaFrame == fullsweepFrame:
                self.sweepDone = True
        else:
            if self.mode < 2:
                self.mode = 2
            radius = self.deltaX*2
            cv2.circle(
                roadProjection, (int(self.xcenter), int(self.ycenter)),
                radius, ccolor, 10)

    def calculateRoughBoundingCubes(self, windows):

        # need to calculate?
        if self.boundingShape[0] == 0:

            # yes
            height = self.z
            nWindows = len(windows)
            ymin = 0
            xmin = self.projectedX
            for i in range(nWindows):
                if xmin > windows[i][0][0]:
                    xmin = windows[i][0][0]
                if ymin < windows[i][0][1]:
                    ymin = windows[i][0][1]
            xmin += 12
            xmax = xmin + 40
            ymax = ymin + 100

            # set the boundingShape: width, height, depth of the vehicle
            self.boundingShape[0] = xmax - xmin
            self.boundingShape[1] = ymax - ymin
            self.boundingShape[2] = height

        # nope!  restore from last estimate
        else:
            xmin = self.xcenter - self.boundingShape[0]/2
            xmax = xmin + self.boundingShape[0]
            ymin = self.ycenter - self.boundingShape[1]/2
            ymax = ymin + self.boundingShape[1]

            # height seems to be non-linear closer to the vanishing point
            # attempting to adjust height so that we can still have good
            # visual of the tracked vehicle
            if self.ycenter < (self.projectedY*0.3):
                height = self.boundingShape[2]*1.5
            elif self.ycenter < (self.projectedY*0.5):
                height = self.boundingShape[2]*1.2
            elif self.ycenter < (self.projectedY*0.7):
                height = self.boundingShape[2]*1.1
            else:
                height = self.boundingShape[2]

        cube3d = np.float32(
            [[xmin, ymin, 0],
             [xmax, ymin, 0],
             [xmax, ymax, 0],
             [xmin, ymax, 0],
             [xmin, ymin, height],
             [xmax, ymin, height],
             [xmax, ymax, height],
             [xmin, ymax, height]]).reshape(-1, 3)

        cube2d = self.projMgr.projectPoints(cube3d)
        cube2d = np.int32(cube2d).reshape(-1, 2)
        return cube3d, cube2d

    # calculate perspective mask location from birds-eye view
    def calculateMask(self, perspectiveImage):
        # defining a blank mask to start with
        mask = np.zeros_like(perspectiveImage)

        # defining a 3 channel or 1 channel color to fill the mask with
        # depending on the input image
        if len(perspectiveImage.shape) > 2:
            # i.e. 3 or 4 depending on your image
            channel_count = perspectiveImage.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # right side mask
        if self.lane is None or self.lane > self.mainLaneIdx:
            # collect the 6 points that matters
            # p0, p4, p5, p6, p2, p3
            vertices = np.array([[
                (self.cube2d[0][0], self.cube2d[0][1]),
                (self.cube2d[4][0], self.cube2d[4][1]),
                (self.cube2d[5][0], self.cube2d[5][1]),
                (self.cube2d[6][0], self.cube2d[6][1]),
                (self.cube2d[2][0], self.cube2d[2][1]),
                (self.cube2d[3][0], self.cube2d[3][1])]], dtype=np.int32)

        # straight ahead mask
        elif self.lane == self.mainLaneIdx:
            # collect the 4 points that matters
            # p7, p6, p2, p3
            vertices = np.array([[
                (self.cube2d[7][0], self.cube2d[7][1]),
                (self.cube2d[6][0], self.cube2d[6][1]),
                (self.cube2d[2][0], self.cube2d[2][1]),
                (self.cube2d[3][0], self.cube2d[3][1])]], dtype=np.int32)

        # left side mask
        else:
            # collect the 6 points that matters
            # p7, p4, p5, p1, p2, p3
            vertices = np.array([[
                (self.cube2d[7][0], self.cube2d[7][1]),
                (self.cube2d[4][0], self.cube2d[4][1]),
                (self.cube2d[5][0], self.cube2d[5][1]),
                (self.cube2d[1][0], self.cube2d[1][1]),
                (self.cube2d[2][0], self.cube2d[2][1]),
                (self.cube2d[3][0], self.cube2d[3][1])]], dtype=np.int32)

        # print("vertices", vertices)
        # print("mask", mask.shape)
        # print("ignore_mask_color", ignore_mask_color)

        # filling pixels inside the polygon defined by
        # "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        maskedImage = cv2.bitwise_and(perspectiveImage, mask)
        return vertices, maskedImage

    def draw3DBoundingCube(self, perspectiveImage):
        # draw bottom of cube
        cv2.drawContours(
            perspectiveImage, [self.cube2d[:4]], -1, self.statusColor, 2)
        # draw sides of cube
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(
                perspectiveImage,
                tuple(self.cube2d[i]), tuple(self.cube2d[j]),
                self.statusColor, 2)
        # draw top of cube
        cv2.drawContours(
            perspectiveImage, [self.cube2d[4:]], -1, self.statusColor, 2)

    # Augmentation Special Effects
    # - default vehicle scanning sweep takes less
    # than a second 20 frames - video is 26fps
    def drawScanning(
            self, projectionFX, roadProjection,
            color=[0, 0, 255], sweepThick=2, fullsweepFrame=26):

        if self.sweepDone and not self.scanDone:
            # calculate scanning height
            height = self.scanDeltaFrame
            window = self.roadGrid.getBoxWindow(self.box)
            be_cube = np.float32(
                [[window[0][0], window[0][1], height],
                 [window[1][0], window[0][1], height],
                 [window[1][0], window[1][1], height],
                 [window[0][0], window[1][1], height],
                 [window[0][0], window[0][1], height + 1],
                 [window[1][0], window[0][1], height + 1],
                 [window[1][0], window[1][1], height + 1],
                 [window[0][0], window[1][1], height + 1]])
            cube = self.projMgr.projectPoints(
                be_cube.reshape(-1, 3))
            imgpts = np.int32(cube).reshape(-1, 2)

            # draw bottom of cube
            cv2.drawContours(
                projectionFX, [imgpts[:4]],
                -1, (0, 0, 255), 3)
            cv2.drawContours(
                roadProjection, [imgpts[:4]],
                -1, (0, 0, 255), 3)
            # draw top of cube
            cv2.drawContours(
                projectionFX, [imgpts[4:]],
                -1, (128, 128, 255), 3)
            cv2.drawContours(
                roadProjection, [imgpts[4:]],
                -1, (128, 128, 255), 3)

            # check if done.
            self.scanDeltaFrame += 1
            if height > fullsweepFrame:
                self.scanDone = True
                if self.mode < 3:
                    self.mode = 3
        elif self.sweepDone and self.scanDone:
            if self.mode < 3:
                self.mode = 3
            self.draw3DBoundingCube(projectionFX)

    def takeProfileSelfie(self, perspectiveImage, newheightFactor=1.0):
        # give up if we are not sure which lane we are on
        # if it is a bad detect the vehicle tracking module will reject it
        if self.lane is None:
            # generate an empty cube intersect
            self.cube_intersect = np.float32([])
            # generate an empty mask profile
            # self.maskedProfile = np.array(
            #     (self.selfieY, self.selfieX), dtype=np.uint8)
            # return an empty vehicle image
            projected_carImage = np.array(
                (self.selfieY, self.selfieX, 3), dtype=np.uint8)
            cv2.rectangle(
                projected_carImage, (5, 5), (635, 235), self.statusColor, 5)
            return projected_carImage

        # for debugging and diagnostics without vehicle tracker
        # if self.mode == 3:
        #     self.ycenter -= 0.35

        # calculate the plane for selfie
        height = self.boundingShape[2]*newheightFactor

        # if we are looking at it straight ahead - in the same lane
        if self.lane == self.mainLaneIdx:
            ymin = self.ycenter
            ymax = self.ycenter
            x1 = self.lanes[self.lane].calculateXCenter(ymin) - self.deltaX
            x2 = self.lanes[self.lane].calculateXCenter(ymax) + self.deltaX

        # we are looking at it from an angle - different lane
        else:
            ymin = self.ycenter - self.deltaY*2
            ymax = self.ycenter + self.deltaY*3
            x1 = self.lanes[self.lane].calculateXCenter(ymin) - self.deltaX*1.5
            x2 = self.lanes[self.lane].calculateXCenter(ymax) + self.deltaX*1.5

        # slice into the middle of the cube
        if self.lane >= self.mainLaneIdx:
            # perpendicular from our view port from the left or straight ahead
            cube_intersect = np.float32(
                [[x1, ymin, 0],
                 [x1, ymin, height],
                 [x2, ymax, height],
                 [x2, ymax, 0]]).reshape(-1, 3)

            # put it here in case we need to change it for left
            # set up cross section projection destination
            dstVehicleCorners = np.float32([
                [-200, 195], [-200, 35],
                [self.selfieX+200, 35], [self.selfieX+200, 195]])

        else:
            # perpendicular from our view port from the right
            cube_intersect = np.float32(
                [[x1, ymax, 0],
                 [x1, ymax, height],
                 [x2, ymin, height],
                 [x2, ymin, 0]]).reshape(-1, 3)

            # put it here in case we need to change it for right
            # set up cross section projection destination
            dstVehicleCorners = np.float32([
                [-200, 195], [-200, 35],
                [self.selfieX+200, 35], [self.selfieX+200, 195]])

        cube_intersect = self.projMgr.projectPoints(cube_intersect)
        self.cube_intersect = np.int32(cube_intersect).reshape(-1, 2)

        # project the car image
        projected_carImage, M = self.unwarp_vehicle(
            np.copy(perspectiveImage), cube_intersect.astype(np.float32),
            dstVehicleCorners, self.projMgr.mtx)

        # generate stats
        if self.mode < 3:
            self.sampleColor(projected_carImage)
        self.modeColor()

        # genrate mask from detected color
        self.maskedProfile = cv2.cvtColor(
            projected_carImage, cv2.COLOR_RGB2GRAY)
        self.maskedProfile = self.maskedProfile.astype(np.uint8)

        # print("self.vehicle_gray", self.vehicle_gray)
        if self.vehicle_gray > 224:
            self.maskedProfile[
                (self.maskedProfile < (self.vehicle_gray - 48))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        elif self.vehicle_gray < 64:
            self.maskedProfile[(self.maskedProfile == 0)] = 128
            self.maskedProfile = 255 - self.maskedProfile
            self.maskedProfile[
                (self.maskedProfile < (255-(self.vehicle_gray)))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        elif self.vehicle_gray > 192:
            self.maskedProfile[
                (self.maskedProfile < (self.vehicle_gray - 24))] = 0
            self.maskedProfile[
                (self.maskedProfile > (self.vehicle_gray + 24))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        elif self.vehicle_gray > 128:
            self.maskedProfile[
                (self.maskedProfile < (self.vehicle_gray - 24))] = 0
            self.maskedProfile[
                (self.maskedProfile > (self.vehicle_gray + 24))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        elif self.vehicle_gray > 96:
            self.maskedProfile[
                (self.maskedProfile < (self.vehicle_gray - 24))] = 0
            self.maskedProfile[
                (self.maskedProfile > (self.vehicle_gray + 24))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        elif self.vehicle_gray > 64:
            self.maskedProfile[
                (self.maskedProfile < (self.vehicle_gray - 24))] = 0
            self.maskedProfile[
                (self.maskedProfile > (self.vehicle_gray + 24))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        elif self.vehicle_gray > 32:
            self.maskedProfile[
                (self.maskedProfile < (self.vehicle_gray - 16))] = 0
            self.maskedProfile[
                (self.maskedProfile > (self.vehicle_gray + 16))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255
        else:
            self.maskedProfile[
                (self.maskedProfile > (self.vehicle_gray + 16))] = 0
            self.maskedProfile[(self.maskedProfile > 0)] = 255

        try:
            if self.mode < 3:
                points = np.nonzero(self.maskedProfile)
                if len(points[0]) > self.colorpoints:
                    self.color = self.vehicle_rgb
                    self.gray = self.vehicle_gray
                else:
                    self_vehicle_rgb = self.color
                    self.vehicle_gray = self.gray
        except:
            self_vehicle_rgb = self.color
            self.vehicle_gray = self.gray

        # get the contour of the vehicle from the mask
        img2, self.contours, hierarchy = cv2.findContours(
            self.maskedProfile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        edges = np.copy(self.maskedProfile)*0

        # draw a filled contour for our mask
        cv2.drawContours(edges, self.contours, -1, 255, -1)

        # return np.dstack((edges, edges, edges))

        # mask our image
        # projected_carImage = cv2.bitwise_and(
        #     projected_carImage, projected_carImage,
        #     mask=self.maskedProfile)

        # draw the contours on top
        cv2.drawContours(
            projected_carImage, self.contours, -1, self.statusColor, 2)
        vehicle_contour = np.copy(projected_carImage) * 0
        cv2.drawContours(
            vehicle_contour, self.contours, -1, self.statusColor, 2)

        # draw our tracking points
        # projected_carImage[:,0:50] = [128, 64, 64]
        # projected_carImage[:,self.selfieX-50:self.selfieX] = [128, 64, 64]

        cv2.rectangle(
            projected_carImage, (5, 5), (635, 235), self.statusColor, 5)

        # unwarp the car mask
        self.contourInPerspective, M = self.unwarp_vehicle_back(
            vehicle_contour, dstVehicleCorners,
            self.cube_intersect.astype(np.float32), self.projMgr.mtx)

        # debugging masks...
        # newedge = np.dstack((edges, edges, edges))
        # cv2.drawContours(
        #     newedge, self.contours, -1, self.statusColor, 2)
        # cv2.rectangle(
        #     newedge, (5, 5), (635, 235), self.statusColor, 5)
        # return newedge

        return projected_carImage

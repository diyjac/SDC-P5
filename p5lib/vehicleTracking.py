#!/usr/bin/python
"""
vehicleTracking.py: version 0.1.0

History:
2017/01/29: coding style phase1:
            reformat to python-guide.org code style
            http://docs.python-guide.org/en/latest/writing/style/
            which uses PEP 8 as a base: http://pep8.org/.
2017/01/25: Initial version converted to a class
"""

import numpy as np
import cv2


class VehicleTracking():

    # initialize
    def __init__(
            self, x, y, projectedX, projectedY, lanes):
        self.x = x
        self.y = y
        self.projectedX = projectedX
        self.projectedY = projectedY
        self.lanes = lanes

    def isVehicleThere(
            self, perspectiveImage, roadGrid, mainLaneIdx, vehicles, vehIdx):
        # something is wrong - we should not have a vehicle in this state
        # with a maskedProfile that is not there!
        # reject the vehicle
        if vehicles[vehIdx] is None:
            # print("rejecting ", vehIdx, "has has no entry in vehicles array")
            return False

        if vehicles[vehIdx].maskedProfile is None:
            # print("rejecting ", vehIdx, "has no maskedProfile")
            return False

        masked_vehicle = vehicles[vehIdx].maskedProfile
        vehicle_points = len(np.nonzero(masked_vehicle)[0])
        midw, midh = vehicles[vehIdx].findCenter(masked_vehicle)
        # print("mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx)

        # initialization?
        if vehicles[vehIdx].mode < 3:

            # collect points counts
            vehicles[vehIdx].detectConfidence_base += vehicle_points
            vehicles[vehIdx].initFrames += 1

            # calculate running confidence for quick elimination
            # during peer scan pruning
            vehicles[vehIdx].confidence = \
                vehicle_points / (
                    vehicles[vehIdx].detectConfidence_base /
                    vehicles[vehIdx].initFrames)

            # merge masks into heatmaps
            if masked_vehicle is not None:
                vehicles[vehIdx].vehicleHeatMap = \
                    vehicles[vehIdx].projMgr.curImgFtr.miximg(
                        vehicles[vehIdx].vehicleHeatMap,
                        masked_vehicle, 0.5, 0.1)

            # return true until we get out of the scanning phase
            # print(
            #     "mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx,
            #     "confidence base: ", vehicles[vehIdx].detectConfidence_base,
            #     "current confidence:", vehicles[vehIdx].confidence)
            return True

        # scanning done - need to set confidence and start tracking.
        # if pasted threshold - make sure we actually have a confiremd vehicle!
        elif vehicles[vehIdx].mode == 3:
            vehicles[vehIdx].detectConfidence_base /= \
                vehicles[vehIdx].initFrames
            # print(
            #     "mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx,
            #     "confidence base: ", vehicles[vehIdx].detectConfidence_base,
            #     "current confidence:", vehicles[vehIdx].confidence)
            if vehicles[vehIdx].detectConfidence_base > 100:
                # calculate confident
                vehicles[vehIdx].confidence = \
                    vehicle_points / vehicles[vehIdx].detectConfidence_base
                if vehicles[vehIdx].confidence > 0.5:
                    # set found
                    vehicles[vehIdx].detected = True

                    # set start of tracking...
                    vehicles[vehIdx].mode = 4
                    # print("new mode: ", vehicles[vehIdx].mode,
                    #       " tracking:", vehIdx)
                    return True

                # low confidence...
                else:
                    # give up and drop it - must be a false-positive
                    vehicles[vehIdx].detected = False
                    vehicles[vehIdx].mode = 7
                    # print("new mode: ", vehicles[vehIdx].mode,
                    #       " tracking:", vehIdx)
                    return False

            # can't find it at all!
            else:
                # give up and drop it - must be a false-positive
                vehicles[vehIdx].detected = False
                vehicles[vehIdx].mode = 7
                # print("new mode: ", vehicles[vehIdx].mode,
                #       " tracking:", vehIdx)
                return False

        # tracking mode - need to check confidence.
        elif vehicles[vehIdx].mode == 4:
            # print(
            #     "mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx,
            #     "confidence base: ", vehicles[vehIdx].detectConfidence_base,
            #     "current confidence:", vehicles[vehIdx].confidence)
            # check for occlusion!
            if roadGrid.isOccluded(vehicles[vehIdx].box):
                # reduce confidence base during this crisis
                # and set grace frame to 50
                vehicles[vehIdx].detectConfidence_base /= 4
                vehicles[vehIdx].graceFrame = 50
                vehicles[vehIdx].mode = 5
                # print(
                #     "new mode: ", vehicles[vehIdx].mode,
                #     " tracking:", vehIdx)
                return True

            # check the counts against confidence
            # drop the vehicle to scanning if we are not 50% confident
            vehicles[vehIdx].confidence = \
                vehicle_points / vehicles[vehIdx].detectConfidence_base
            if vehicles[vehIdx].confidence < 0.5 and \
               vehicles[vehIdx].graceFrame == 0:
                vehicles[vehIdx].detected = False
                vehicles[vehIdx].mode = 2
                # print(
                #   "new mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx,
                #   "confidence base:", vehicles[vehIdx].detectConfidence_base,
                #   "current confidence:", vehicles[vehIdx].confidence)
                return False
            elif vehicles[vehIdx].confidence < 0.5:
                # ten graceFrame to get back our confidence
                vehicles[vehIdx].graceFrame -= 1
                # print(
                #     "new grace:", vehicles[vehIdx].graceFrame,
                #     "tracking:", vehIdx)
                vehicles[vehIdx].takeProfileSelfie(perspectiveImage, 2.0)
            elif vehicles[vehIdx].confidence > 0.5:
                vehicles[vehIdx].graceFrame = 10
                # print(
                #     "new grace:", vehicles[vehIdx].graceFrame,
                #     "tracking:", vehIdx)

            # get some info on how far shifted is the car image
            shift = int(vehicles[vehIdx].selfieX/2) - midw
            laneIdx = vehicles[vehIdx].lane

            # check direction of travel
            # shift would be positive for right and negative for left
            #    if travel is forward
            # shift would be negative for right and positive for left
            #    if travel is backward
            if ((shift > 3 and laneIdx > mainLaneIdx) or
                    (shift < -3 and laneIdx < mainLaneIdx)):
                # forward travel - compensate
                # increment center y as a delta percentage
                delta = np.absolute(shift)/60.0
                # print("shift:", shift, "moving vehicle", vehIdx,
                #       "old position", vehicles[vehIdx].ycenter,
                #       "new position", vehicles[vehIdx].ycenter - delta,
                #       "decrement by", delta)
                vehicles[vehIdx].ycenter -= delta

                if vehicles[vehIdx].ycenter < 1500.0:
                    vehicles[vehIdx].traveled = True

                # vehicle leaving?
                if vehicles[vehIdx].ycenter < 200:
                    vehicles[vehIdx].mode = 6
                    # print("new mode: ", vehicles[vehIdx].mode,
                    #       " tracking:", vehIdx)

            if ((shift < -3 and laneIdx > mainLaneIdx) or
                    (shift > 3 and laneIdx < mainLaneIdx)):
                # backward travel - compensate
                # decrement center y as a delta percentage
                # going backward needs to be faster it seems
                delta = np.absolute(shift)/15.0
                # print("shift: ", shift, "moving vehicle", vehIdx,
                #       "old position", vehicles[vehIdx].ycenter,
                #       "new position", vehicles[vehIdx].ycenter - delta,
                #       "increment by", delta)
                vehicles[vehIdx].ycenter += delta

                # vehicle leaving?
                laneIdx = vehicles[vehIdx].lane
                if vehicles[vehIdx].ycenter > self.lanes[laneIdx].bottomY():
                    if vehicles[vehIdx].traveled:
                        vehicles[vehIdx].mode = 6
                        # print(
                        #     "new mode: ", vehicles[vehIdx].mode,
                        #     "tracking:", vehIdx)

            # returning from mode == 4
            return True

        # occluded - need to check if we are out of occlusion
        elif vehicles[vehIdx].mode == 5:
            # print(
            #     "mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx,
            #     "confidence base: ", vehicles[vehIdx].detectConfidence_base,
            #     "current confidence:", vehicles[vehIdx].confidence)
            # check for occlusion!
            if not roadGrid.isOccluded(vehicles[vehIdx].box):
                vehicles[vehIdx].mode = 4
                # print("new mode: ", vehicles[vehIdx].mode,
                #       " tracking:", vehIdx)
                return True

            # check the counts against confidence
            # drop the vehicle to scanning if we are not 50% confident
            vehicles[vehIdx].confidence = \
                vehicle_points / vehicles[vehIdx].detectConfidence_base
            if vehicles[vehIdx].confidence < 0.5 and \
               vehicles[vehIdx].graceFrame == 0:
                vehicles[vehIdx].detected = False
                vehicles[vehIdx].mode = 2
                # print(
                #   "new mode:", vehicles[vehIdx].mode, " tracking:", vehIdx,
                #   "confidence base:", vehicles[vehIdx].detectConfidence_base,
                #   "current confidence:", vehicles[vehIdx].confidence)
                return False
            elif vehicles[vehIdx].confidence < 0.5:
                # ten graceFrame to get back our confidence
                vehicles[vehIdx].graceFrame -= 1
                # print(
                #     "new grace:", vehicles[vehIdx].graceFrame,
                #     "tracking:", vehIdx)
                vehicles[vehIdx].takeProfileSelfie(perspectiveImage, 2.0)
            elif vehicles[vehIdx].confidence > 0.5:
                vehicles[vehIdx].graceFrame = 50
                # print(
                #     "new grace:", vehicles[vehIdx].graceFrame,
                #     "tracking:", vehIdx)

            # get some info on how far shifted is the car image
            shift = int(vehicles[vehIdx].selfieX/2) - midw
            laneIdx = vehicles[vehIdx].lane

            # check direction of travel
            # shift would be positive for right and negative for left
            #    if travel is forward
            # shift would be negative for right and positive for left
            #    if travel is backward
            if ((shift > 3 and laneIdx > mainLaneIdx) or
                    (shift < -3 and laneIdx < mainLaneIdx)):
                # forward travel - compensate
                # increment center y as a delta percentage
                delta = np.absolute(shift)/60.0
                # print(
                #     "shift: ", shift, "moving vehicle", vehIdx,
                #     "old position", vehicles[vehIdx].ycenter,
                #     "new position", vehicles[vehIdx].ycenter - delta,
                #     "decrement by", delta)
                vehicles[vehIdx].ycenter -= delta

                if vehicles[vehIdx].ycenter < 1500.0:
                    vehicles[vehIdx].traveled = True

                # vehicle leaving?
                if vehicles[vehIdx].ycenter < 200:
                    vehicles[vehIdx].mode = 6
                    # print("new mode: ", vehicles[vehIdx].mode,
                    #       " tracking:", vehIdx)

            if ((shift < -3 and laneIdx > mainLaneIdx) or
                    (shift > 3 and laneIdx < mainLaneIdx)):
                # backward travel - compensate
                # decrement center y as a delta percentage
                # going backward needs to be faster it seems
                delta = np.absolute(shift)/15.0
                # print("shift: ", shift, "moving vehicle", vehIdx,
                #       "old position", vehicles[vehIdx].ycenter,
                #       "new position", vehicles[vehIdx].ycenter - delta,
                #       "increment by", delta)
                vehicles[vehIdx].ycenter += delta

                # vehicle leaving?
                laneIdx = vehicles[vehIdx].lane
                if vehicles[vehIdx].ycenter > self.lanes[laneIdx].bottomY():
                    if vehicles[vehIdx].traveled:
                        vehicles[vehIdx].mode = 6
                        # print("new mode: ", vehicles[vehIdx].mode,
                        #       " tracking:", vehIdx)

            # returning from mode == 5
            return True

        # vehicle leaving the scene
        elif vehicles[vehIdx].mode == 6:
            # set timer for 26 frames (1 second) and then drop the vehicle
            vehicles[vehIdx].exitFrames += 1
            if vehicles[vehIdx].exitFrames > 170:
                vehicles[vehIdx].detected = False
                vehicles[vehIdx].mode = 7
                # print("new mode: ", vehicles[vehIdx].mode,
                #       " tracking:", vehIdx)
                return False
            else:
                vehicles[vehIdx].ycenter += 0.40
                return True

        # vehicle losted (no tracking)
        elif vehicles[vehIdx].mode == 7:
            # print("new mode: ", vehicles[vehIdx].mode,
            #       " tracking:", vehIdx)
            return False

        # unknown state - return False
        else:
            # print("unknown mode: ", vehicles[vehIdx].mode,
            #       " tracking:", vehIdx)
            return False

        # fell out - default False
        # print("fellout mode: ", vehicles[vehIdx].mode, " tracking:", vehIdx)
        return False

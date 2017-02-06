#!/usr/bin/python
"""
roadGrid.py: version 0.1.0
A quick 2D Voxel implemenation.

History:
2017/01/29: coding style phase1:
    reformat to python-guide.org code style
    http://docs.python-guide.org/en/latest/writing/style/
    which uses PEP 8 as a base: http://pep8.org/.
2017/01/23: Initial version converted to a class
"""

import numpy as np
import re


# a class for helping us map the road surface
# using voxel for quick 3D reconstruction
class RoadGrid():

    # initialize
    def __init__(self, x0, y0, nlanes, mainLaneIdx):
        self.nlanes = nlanes
        self.mainIdx = mainLaneIdx
        self.maxkey = 55
        self.mapping = {}
        self.x0 = x0
        self.y0 = y0

        # list of objects and their boxes
        # trigged by setting found and occlusion checks
        self.object_list = []

        # list of vehicles and their boxes
        # trigged by setting insert and track checks
        self.vehicle_list = {}

    # we are enforcing some constraints into the road grid
    def map_boxes(self, laneIdx, boxes):
        numkeys = [int(key) for key in boxes.keys()]
        maxkey = max(numkeys)
        if self.maxkey < maxkey:
            self.maxkey = maxkey
        laneAlpha = chr(laneIdx + 65)
        for i in numkeys:
            box = '%s+%02d' % (laneAlpha, self.maxkey - i)
            self.mapping[box] = {
                'window': boxes['%d' % (i)],
                'found': False,
                'occluded': False,
                'tracked': False,
                'object': None,
                'vehicle': None}

    def getMapping(self):
        return self.mapping

    def setVehicle(self, boxlist, vehIdx):
        for box in boxlist:
            self.mapping[box]['vehicle'] = vehIdx
            if not self.mapping[box]['occluded']:
                vehStr = '%d' % (vehIdx)
                self.vehicle_list[vehStr] = box

    def setFound(self, box):
        self.mapping[box]['found'] = True
        # print("from setFound...")
        self.calculateVoxelOcclusionAndObjectSeparation(
            box, forceIntoOne=True)

    def setOccluded(self, box):
        if box in self.mapping:
            self.mapping[box]['occluded'] = True

    def getKey(self, lane, y):
        box = '%s+%02d' % (chr(lane + 65), y)
        return box

    def getBox(self, lane, y):
        box = '%s+%02d' % (chr(lane + 65), y)
        if box in self.mapping:
            return self.mapping[box]
        return None

    def getAllWindows(self):
        return [self.mapping[map]['window'] for map in self.mapping.keys()]

    def getBoxWindow(self, box):
        return self.mapping[box]['window']

    def getFoundWindows(self):
        return [self.mapping[map]['window'] for map in self.mapping.keys()
                if self.mapping[map]['found']]

    def getOccludedWindows(self):
        return [self.mapping[map]['window'] for map in self.mapping.keys()
                if self.mapping[map]['occluded']]

    def getFoundAndNotOccludedWindows(self):
        return [self.mapping[map]['window'] for map in self.mapping.keys()
                if self.mapping[map]['found'] and
                not self.mapping[map]['occluded']]

    def getFoundAndNotOccludedWindowsInObject(self, objIdx):
        return [self.mapping[map]['window']
                for map in self.object_list[objIdx]
                if self.mapping[map]['found'] and
                not self.mapping[map]['occluded']]

    def getFoundAndNotOccludedWindowsInVehicle(self, vehIdx):
        return [self.mapping[map]['window']
                for map in self.mapping.keys()
                if self.mapping[map]['found'] and
                not self.mapping[map]['occluded'] and
                self.mapping[map]['vehicle'] is not None and
                self.mapping[map]['vehicle'] == vehIdx]

    def getFoundAndNotOccludedBoxesInObject(self, objIdx):
        return [map for map in self.object_list[objIdx]
                if self.mapping[map]['found'] and
                not self.mapping[map]['occluded']]

    def getFoundAndNotOccludedBoxesInVehicle(self, vehIdx):
        return [map for map in self.mapping.keys()
                if self.mapping[map]['found'] and
                not self.mapping[map]['occluded'] and
                self.mapping[map]['vehicle'] is not None and
                self.mapping[map]['vehicle'] == vehIdx]

    def gridCoordinates(self, box):
        lane, y = box.split('+')
        return ord(lane)-65, int(y)

    def gridSize(self):
        return self.nlanes, self.maxkey

    def generatePolyRay(self, x0, y0, x1, y1):
        allY = np.array([y0, y1])
        allX = np.array([x0, x1])
        return np.poly1d(np.polyfit(allY, allX, 1))

    def getNumObjects(self):
        return len(self.object_list)

    def getObjects(self):
        return self.object_list

    def getObjectList(self, i):
        return self.object_list[i]

    def getObjectListWindows(self, i):
        return [self.mapping[map]['window'] for map in self.object_list[i]]

    # we will use constrain propagation to limit our search for vehicle testing
    # by using voxel occlusion testing to find occluded boxes in the grid
    def calculateVoxelOcclusionAndObjectSeparation(
            self, box, vehicle=None, forceIntoOne=False):
        if not self.mapping[box]['occluded']:
            # find the two rays from our camera that hits the edges
            # of our box and generate a set of ray polys.
            window = self.mapping[box]['window']
            x1, y1 = window[0][0], window[0][1]
            x2, y2 = window[1][0], window[1][1]
            polyRay1 = self.generatePolyRay(self.x0, self.y0, x1, y1)
            polyRay2 = self.generatePolyRay(self.x0, self.y0, x2, y2)

            newobject = True
            # until our rays hit something found before
            # then we are a new object
            # else we are part of a larger object.
            # (or it could be something that is too close
            #  to tell apart using this method)
            mapping = [n for n in self.mapping.keys()]
            mapping.sort()
            for map in mapping:
                window = self.mapping[map]['window']
                boxX1, boxX2 = window[0][0], window[1][0]
                boxMidY = (window[0][1] + window[1][1])/2
                rayX1 = polyRay1(np.array([boxMidY]))[0]
                rayX2 = polyRay2(np.array([boxMidY]))[0]
                # print("rayX1", rayX1, "rayX2", rayX2, boxMidY)
                # print("boxX1", boxX1, "boxX2", boxX2, boxMidY)
                # print("box", box, "map", map)

                # three choices for a box to be occluded by our
                # box: ray1 hits, ray2 hits, or the box is
                #      completely within the two rays
                if (((boxX1 <= rayX1 and boxX2 >= rayX1) or
                        (boxX1 <= rayX2 and boxX2 >= rayX2) or
                        (rayX1 < boxX1 and rayX1 < boxX2 and
                         rayX2 > boxX1 and rayX2 > boxX2)) and
                        (y1 > boxMidY)):
                    # print("Hit!")

                    # is our box is a vehicle...?
                    if vehicle is not None:
                        if self.mapping[map]['vehicle'] is None:
                            self.mapping[map]['vehicle'] = vehicle
                            self.mapping[map]['found'] = True
                            self.mapping[map]['occluded'] = True
                            # print("10. vehicle is none.!", box, map)

                        # we are the same!
                        elif self.mapping[map]['vehicle'] == vehicle:
                            # update the vehicle to be us.
                            vehStr = '%d' % (vehicle)
                            self.vehicle_list[vehStr] = box
                            # print("11. vehicle is same.!", box, map)

                        # the other box is a vehicle too, but not us!
                        # occlude it
                        else:
                            # print("1. this should not happen!")
                            self.mapping[map]['occluded'] = True

                    # stop! we found something already
                    # occluded - this box maybe be something
                    # larger, so adopt its object or vehicle
                    elif self.mapping[map]['occluded'] and forceIntoOne:
                        # the other voxel is a vehicle!
                        if self.mapping[map]['vehicle'] is not None:
                            # the vehicle is being tracked by
                            # vehicle tracker, don't try to move it!
                            # vehicle is not being tracked...
                            if self.mapping[map]['tracked']:
                                # print("2. tracked detected!", box, map)
                                # print("setting ourselves occluded")
                                vehIdx = self.mapping[map]['vehicle']
                                vehStr = '%d' % (vehIdx)
                                self.mapping[box]['occluded'] = True
                                self.mapping[box]['vehicle'] = \
                                    self.mapping[map]['vehicle']
                                self.vehicle_list[vehStr] = map
                                self.mapping[box]['vehicle'] = vehIdx
                                self.mapping[box]['found'] = True

                            else:
                                # print("2. new location detected!", box, map)
                                # need to inform the vehicle
                                # it has a new location!
                                vehIdx = self.mapping[map]['vehicle']
                                self.mapping[map]['occluded'] = True
                                vehStr = '%d' % (vehIdx)
                                self.vehicle_list[vehStr] = box
                                self.mapping[box]['vehicle'] = vehIdx
                                self.mapping[box]['found'] = True

                        elif self.mapping[box]['object'] is not None:
                            # if self.mapping[map]['object'] is None:
                            #     print("That's not suppose to happen!")
                            # else:
                            #     print("objectlist", self.object_list)
                            # print("3. new location detected!", box, map)
                            idx = self.mapping[map]['object']
                            if idx is not None:
                                self.mapping[box]['object'] = idx
                                # and add ourselves to their list
                                # print("idx=", idx)
                                self.object_list[idx].append(box)

                        # our objects do not match!
                        elif self.mapping[box]['object'] is None and \
                                self.mapping[map]['object'] is not None:
                            # print("4. new location detected!", box, map)
                            idx = self.mapping[map]['object']
                            self.mapping[box]['object'] = idx
                            # else:
                            #     print("newer list than ours!!!")
                            # otherwise we are the same object already
                            # - nothing to do.

                    # other box is not occluded
                    # elif not self.mapping[map]['occluded'] and \
                    #      not self.mapping[map]['tracked']:
                    elif not self.mapping[map]['occluded']:
                        # the other voxel is also a vehicle!
                        if self.mapping[map]['vehicle'] is not None:
                            # print("5. new location detected!", box, map)
                            # need to inform the vehicle
                            # it has a new location!
                            vehIdx = self.mapping[map]['vehicle']
                            self.mapping[map]['occluded'] = True
                            vehStr = '%d' % (vehIdx)
                            self.vehicle_list[vehStr] = box
                            self.mapping[box]['vehicle'] = vehIdx
                            self.mapping[box]['found'] = True

                        # but we don't belong in the same object
                        if self.mapping[box]['object'] is not None and \
                                self.mapping[map]['object'] is not None \
                                and self.mapping[box]['object'] != \
                                self.mapping[map]['object']:
                            # we seem to be occluding another object!
                            # just set their occluded flag
                            # print("6. new location detected!")
                            self.mapping[map]['occluded'] = True

                        # we thought we were our own list, we need
                        # the other object is not in a different
                        # object list and we don't either!
                        elif self.mapping[box]['object'] is None:
                            # we must be a new object then!
                            # create a new object list,
                            # and add this to our object.
                            idx = len(self.object_list)
                            self.mapping[box]['object'] = idx
                            self.mapping[map]['object'] = idx
                            self.object_list.append([box, map])
                            # and set the occlusion!
                            self.mapping[map]['occluded'] = True
                        else:
                            # are in our own list, just add this one to
                            # it and set to our object.
                            idx = self.mapping[box]['object']
                            self.object_list[idx].append(map)
                            self.mapping[map]['object'] = idx
                            # and set the occlusion!
                            self.mapping[map]['occluded'] = True

                    # the other box is occluded already, but we cannot
                    # force our objects into one
                    else:
                        # this item is already occluded.
                        # add ourselves to their list.
                        if self.mapping[map]['object'] is not None and \
                                self.mapping[box]['object'] is None:
                            # we may be something larger after all!
                            idx = self.mapping[map]['object']
                            self.mapping[box]['object'] = idx
                            # and add ourselves to their list
                            self.object_list[idx].append(box)
                            # and set the occlusion!
                            self.mapping[map]['occluded'] = True
        # for debugging objs
        # print("objs = ", len(self.object_list))

    def calculateObjectPosition(self, lane, ycoordinate):
        # first check to see if the coordinates falls into an existing box
        laneAscii = chr(lane + 65)
        boxpattern = re.compile('^%s[0123456789]+$' % (laneAscii))
        for box in self.mapping.keys():
            if boxpattern.match(box):
                window = self.mapping[box]['window']
                if (window[0][1] < ycoordinate and
                        ycoordinate < window[1][1]):
                    return int(box.replace(laneAscii, ''))

        # if not, return an estimate
        return int((1.0-ycoordinate/self.y0) * self.maxkey)-3

    def insertTrackedObject(self, lane, yidx, window, vehIdx, tracking=False):
        box = '%s+%02d' % (chr(lane + 65), yidx)
        # its not there - insert it
        if box not in self.mapping:
            self.mapping[box] = {
                'window': window,
                'found': True,
                'tracked': tracking,
                'occluded': False,
                'object': None,
                'vehicle': vehIdx}

        # its already there - replace it
        else:
            self.mapping[box]['window'] = window
            self.mapping[box]['found'] = True
            self.mapping[box]['tracked'] = tracking
            self.mapping[box]['vehicle'] = vehIdx

        vehStr = '%d' % (vehIdx)
        self.vehicle_list[vehStr] = box
        # print("from insertTrackedObject...")
        self.calculateVoxelOcclusionAndObjectSeparation(box, vehicle=vehIdx)
        return self.vehicle_list[vehStr]

    def isOccluded(self, box):
        if box in self.mapping:
            return self.mapping[box]['occluded']
        return False

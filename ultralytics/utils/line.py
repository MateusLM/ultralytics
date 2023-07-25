import string

import numpy as np
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString, GeometryCollection

from math_operations import degree_to_range, add_degree, calculate_direction_pt, midpoint, calculate_distance, \
    is_in_degree_range, average_dir

from global_variables import *


class BaseLine(object):
    _count = 0
    _letters = list(string.ascii_uppercase)

    @staticmethod
    def next_id():
        BaseLine._count += 1
        return BaseLine._count

    @staticmethod
    def remove_id():
        BaseLine._count -= 1
        return BaseLine._count

    @staticmethod
    def name(type):
        if type == "Entry":
            return str(BaseLine._letters[BaseLine._count - 1])
        else:
            return str(BaseLine._count)

    @staticmethod
    def reset():
        BaseLine._count = 0


class Line(BaseLine):
    def __init__(self, coordinates, r=90):
        self.r = r
        self.coords = LineString(coordinates)
        self.left_point, self.right_point = coordinates[0], coordinates[1]
        self.center_point = midpoint(self.left_point, self.right_point, is_round=False)
        self.line_id = self.next_id()
        self._dir = {
            "Entry": degree_to_range(add_degree(calculate_direction_pt(self.left_point, self.right_point), 90)),
            "Exit": degree_to_range(add_degree(calculate_direction_pt(self.right_point, self.left_point), 90))}
        self.dir_l = {"Entry": [],
                      "Exit": []}
        self.intersection_points = {"Entry": [], "Exit": []}
        self._avg_point = {
            "Entry": self.center_point,
            "Exit": self.center_point}
        self.stop = False


    def point_intersection(self, track):
        min_distance = float("inf")
        intersection = None

        #print(track)
        inter = track.intersection(self.coords)
        #print(inter)
        if isinstance(inter, Point):
            intersection = inter
        elif isinstance(inter, MultiPoint):
            for p in inter:
                distance = track.distance(p)
                if distance < min_distance:
                    min_distance = distance
                    intersection = p
        elif isinstance(inter, LineString):
            #print(track)
            closest_point = inter.interpolate(inter.project(Point(track.coords[-2])))
            #print(closest_point)
            intersection = closest_point
        elif isinstance(inter, MultiLineString):
            for ls in inter:
                closest_point = ls.interpolate(ls.project(Point(track.coords[-2])))
                distance = track.distance(closest_point)
                if distance < min_distance:
                    min_distance = distance
                    intersection = closest_point
        elif isinstance(inter, GeometryCollection):
            for geom in inter.geoms:
                if isinstance(geom, Point):
                    distance = track.distance(geom)
                    if distance < min_distance:
                        min_distance = distance
                        intersection = geom
                elif isinstance(geom, LineString):
                    closest_point = geom.interpolate(geom.project(Point(track.coords[-2])))
                    distance = track.distance(closest_point)
                    if distance < min_distance:
                        min_distance = distance
                        intersection = closest_point

        return intersection.x, intersection.y

    def valid_exit_point(self, track, exit_point_intersection):
        if track.reset:
            if not track.entry and track.last_exit[0].name("Exit") == self.name("Exit"):
                return False
        if not track.entry:
            return True
        else:

            entry_point_intersection_index = track.entry[1]
            entry_point_intersection = track.tlbr_to_bottom_center_point(track.tracking_tlbr[entry_point_intersection_index])
            # print("VALID EXIT POINT")
            # print(entry_point_intersection, exit_point_intersection, calculate_distance(entry_point_intersection, exit_point_intersection) )
            # print(track, "VALID EXIT", calculate_distance(entry_point_intersection, exit_point_intersection) < 20, calculate_distance(entry_point_intersection, exit_point_intersection))
            if calculate_distance(entry_point_intersection, exit_point_intersection) < MIN_DIST_ENTRY_EXIT:
                return False
        return True

    def valid_entry_point(self, track):
        if track.reset:
            #print("SUM DIST", sum([dist for _, dist, _, _ in track.info_tracking]))
            return sum([dist for _, dist, _,_ in track.info_tracking]) > MIN_DIST_ENTRY_EXIT
        return True


    def is_valid_direction(self, t, mode):
        # TODO: Do average range
        if is_in_degree_range(self.dir(mode), t.dir):
            return True
        else:
            return False

    def update_dir(self, track, mode):
        #print(self._dir)
        if not self.stop:
            if len(self.dir_l[mode]) < MAX_BUFFER_MOVS:
                self.dir_l[mode].append(track.dir)
                self._dir[mode] = average_dir(self.dir_l[mode], range=True, r=90, only_dir=True)
            else:
                self.stop = True
                #print("Enough of dirs")

    def dir(self, mode):
        return self._dir[mode]

    def name(self, type):
        if type == "Entry":
            return str(BaseLine._letters[self.line_id - 1])
        else:
            return str(self.line_id)

    def avg_point(self, mode):
        return self._avg_point[mode]



    def update_avg_point(self, cross_point, mode="Entry"):
        if not self.stop:
            if len(self.intersection_points[mode]) < MAX_BUFFER_MOVS:
                self.intersection_points[mode].append(cross_point)
                self._avg_point[mode] = np.average(self.intersection_points[mode], axis=0)
            else:
                self.stop = True
                #print("Enough of intersection_points")

    def update_position(self, coordinates):
        self.coords = LineString(coordinates)
        self.left_point, self.right_point = coordinates[0], coordinates[1]
        self.mid_point = midpoint(self.left_point, self.right_point, is_round=False)
        self._dir = {
            "Entry": degree_to_range(add_degree(calculate_direction_pt(self.left_point, self.right_point), 90)),
            "Exit": degree_to_range(add_degree(calculate_direction_pt(self.right_point, self.left_point), 90))}
        self.dir_l = {"Entry": [],
                      "Exit": []}
        self.intersection_points = {"Entry": [], "Exit": []}
        self._avg_point = {
            "Entry": self.mid_point,
            "Exit": self.mid_point}

    def __repr__(self):
        return 'L_{}{}:IN:{}ºOUT:{}º'.format(self.name("Entry"), self.name("Exit"), self._dir["Entry"],
                                             self._dir["Exit"])

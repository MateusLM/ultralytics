import string

import numpy as np
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString, GeometryCollection, Polygon

from global_variables import *
from math_operations import degree_to_range, add_degree, calculate_direction_pt, calculate_distance, \
    is_in_degree_range


class BaseRegion(object):
    _count = 0
    _letters = list(string.ascii_uppercase)

    @staticmethod
    def next_id():
        BaseRegion._count += 1
        return BaseRegion._count

    @staticmethod
    def remove_id():
        BaseRegion._count -= 1
        return BaseRegion._count

    @staticmethod
    def name(type):
        if type == "Entry":
            return str(BaseRegion._letters[BaseRegion._count - 1])
        else:
            return str(BaseRegion._count)

    @staticmethod
    def reset():
        BaseRegion._count = 0


class Region(BaseRegion):
    def __init__(self, coordinates, r=90):
        print("REACHED HERE!!")
        self.r = r
        print(coordinates)
        self.coordinates = coordinates
        self.tl, self.tr, self.br, self.bl = self.coordinates
        self.stop = False

        # self.left_point, self.right_point = coordinates[0], coordinates[1]
        # self.mid_point = midpoint(self.left_point, self.right_point, is_round=False)
        def get_width_height(tl, tr, br, bl):
            return max(abs(tl[0] - tr[0]), abs(tl[0] -bl[0])), max(abs(br[1] - tr[1]), abs(bl[1]-bl[1]))

        self.width, self.height = get_width_height(self.tl, self.tr, self.br, self.bl)
        print("WIDTH", self.width, "HEIGHT", self.height)
        self.polygon = Polygon([self.tl, self.tr, self.br, self.bl])


        self.exit_midpoint = (np.array(self.tl) + np.array(self.bl)) / 2
        self.entry_midpoint = (np.array(self.br) + np.array(self.tr)) / 2

        self.center_point = (np.array(self.exit_midpoint) + np.array(self.entry_midpoint)) / 2

        # Calculate the centers of the two regions
        self.exit_center = np.rint((np.array(self.exit_midpoint) + np.array(self.center_point)) / 2)
        self.entry_center = np.rint((np.array(self.entry_midpoint) + np.array(self.center_point)) / 2)

        print("LEFT", self.exit_midpoint, "right", self.entry_midpoint, "center_point", self.center_point, "exit_center", self.exit_center, "entry_center", self.entry_center)

        self.min_dist =min([calculate_distance(self.tl, self.tr), calculate_distance(self.tr, self.br), calculate_distance(self.br, self.bl), calculate_distance(self.bl, self.tl)])

        # print(self.mid_point_left, self.mid_point_right)
        self.line_id = self.next_id()
        self.dir_l = {"Entry": [],
                      "Exit": []}
        self.coords = LineString([self.tl, self.tr])
        self._dir = {
            "Entry": degree_to_range(add_degree(calculate_direction_pt(self.tl, self.tr), 90)),
            "Exit": degree_to_range(add_degree(calculate_direction_pt(self.tr, self.tl), 90))}

        self.max_to_stop = 50
        self.count_to_stop = 0
        self._avg_point = {
            "Entry": self.center_point,
            "Exit": self.center_point}
        self.intersection_points = {"Entry": [], "Exit": []}

    def valid_exit_point(self, track):

        if track.last_exit:
            if track.last_exit[0].name("Exit") == self.name("Exit"):
                return False
        if track.entry:
            if track.entry[0].name("Entry") == self.name("Entry"):
                #print("INVERSE???", track, track.dist, self.min_dist)
                if track.dist < self.min_dist:
                    return False

        return True

    def point_intersection(self, track):
        min_distance = float("inf")
        intersection = None

        # print(track)
        inter = track.intersection(self.polygon)

        if inter.is_empty:
            print("Does it intersects?", self.polygon.intersects(track), track.intersects(self.polygon))
            print("it should not be empty", inter)
            print("track", track, "polygon", self.polygon)
            exit(1)

        if isinstance(inter, Point):
            intersection = inter
        elif isinstance(inter, MultiPoint):
            for p in inter:
                distance = track.distance(p)
                if distance < min_distance:
                    min_distance = distance
                    intersection = p
        elif isinstance(inter, LineString):
            # print(track)
            closest_point = inter.interpolate(inter.project(Point(track.coords[-2])))
            # print(closest_point)
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

    def update_avg_point(self, cross_point, mode="Entry"):
        if not self.stop:
            if len(self.intersection_points[mode]) < MAX_BUFFER_MOVS:
                self.intersection_points[mode].append(cross_point)
                self._avg_point[mode] = np.average(self.intersection_points[mode], axis=0)
            else:
                self.stop = True
                #print("Enough of intersection_points")

    def update_min_dist(self, dist):
        if self.min_dist < dist:
            self.min_dist = dist
        self.count_to_stop += 1

    def is_update(self):
        if self.max_to_stop == self.count_to_stop:
            return False
        return True

    def valid_entry_point(self, track):
        if track.reset:
            # print("SUM DIST", sum([dist for _, dist, _, _ in track.info_tracking]))
            if track.last_exit[0].name("Entry") == self.name("Entry"):
                return False
        return True

    def is_valid_direction(self, t, mode):
        # TODO: Do average range
        if is_in_degree_range(self.dir(mode), t.dir):
            return True
        else:
            return False

    def dir(self, mode):
        return self._dir[mode]

    def name(self, type):
        if type == "Entry":
            return str(BaseRegion._letters[self.line_id - 1])
        else:
            return str(self.line_id)

    def avg_point(self, mode):
        return self._avg_point[mode]

    def __repr__(self):
        return 'R_{}{}:IN:{}ºOUT:{}º'.format(self.name("Entry"), self.name("Exit"), self._dir["Entry"],
                                             self._dir["Exit"])

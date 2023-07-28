import itertools
import math
from copy import deepcopy
from math import sin, cos, radians

import numpy as np
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import nearest_points

import ultralytics.utils.global_variables as GV

def tlwh_to_center_point(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret[:2]


def yolobbox2bbox(yolobbox, dw, dh):
    x, y, w, h = yolobbox
    x1 = int((x - w / 2) * dw)
    x2 = int((x + w / 2) * dw)
    y1 = int((y - h / 2) * dh)
    y2 = int((y + h / 2) * dh)

    if x1 < 0:
        x1 = 0
    if x2 > dw - 1:
        x2 = dw - 1
    if y1 < 0:
        y1 = 0
    if y2 > dh - 1:
        y2 = dh - 1

    return x1, y1, x2, y2


def dir_suddenly_changes(old_dir, new_dir):
    old_dir_reversed = add_degree(old_dir, 180)
    new_dir_range = degree_to_range(new_dir, r=115)
    return is_in_degree_range(new_dir_range, old_dir_reversed)


def nearest_track_to_point0(coords, tracks):
    p = Point(coords)

    dist = [LineString(t.show_tracking_line).distance(p) for t in tracks]
    return tracks[get_min_index(dist)]


def nearest_track_to_point(coords, tracks):
    print("calculating nearest track to point")
    p = Point(coords)

    closest_track = min(tracks, key=lambda t: LineString(t.show_tracking_line).distance(p))
    return closest_track


def tlbr_array_to_bottom_center_point_array(tlbr_array):
    tlwh_array = tlbr_array.copy()
    tlwh_array[:, 2:] -= tlwh_array[:, :2]
    tlwh_array[:, :2] += tlwh_array[:, 2:] / 2
    bottom_left_y_array = tlbr_array[:, 1] + tlwh_array[:, 3]
    return np.stack([tlwh_array[:, 0], bottom_left_y_array], axis=-1)


def nearest_track_to_point1(coords, tracks):
    p = Point(coords)
    w = pysal.lib.weights.DistanceBand.from_iterable([t.tlbr_to_bottom_center_point(p) for t in tracks])
    d, idx = w.query(p, k=1)
    closest_track = tracks[idx[0]]
    return closest_track


def nearest_vehicle_to_point(coords, vehicles):
    p = Point(coords)

    dist = [box(*v.bbox, ccw=True).distance(p) for v in vehicles]
    return vehicles[get_min_index(dist)]


def tracks_inside_rectangle(rectangle, tracks):
    init, end = rectangle
    p = Polygon([(init[0], init[1]), (end[0], init[1]), (end[0], end[1]), (init[0], end[1])])
    return [track for track in tracks if
            p.intersects(LineString(track.show_tracking_line))]


def vehicles_inside_rectangle(rectangle, vehicles):
    init, end = rectangle
    p = Polygon([(init[0], init[1]), (end[0], init[1]), (end[0], end[1]), (init[0], end[1])])
    return [v for v in vehicles if
            p.intersects(box(*v.bbox, ccw=True))]


def has_category_in_common(t, c):
    # print(t.category_counting, c.category_counting)
    for key in t.category_counting:
        if key in c.category_counting:
            # print("THERE ARE CATEGORIES IN COMMON")
            return True
    # print("THERE aren't ANY CATEGORY IN COMMON")
    return False


# TODO: fix !!!!
def calculate_num_column(movs, mov, categories, category, CLASSES, is_mov=False):
    print(movs, mov)
    section, movements = movs[0], movs[1]

    if is_mov:
        index = movements.index(mov) + len(section)
    else:
        index = section.index(mov)

    column = 3
    # print(mov, index)
    # print(movs, mov, categories, category)
    return column + index * len(categories) + categories.index(CLASSES[int(category)])


def unique_char(word):
    unique_characters = []

    for character in word:
        if character not in unique_characters:
            unique_characters.append(character)
    return unique_characters


def flatten(list_of_lists):
    new_list = []

    for sel in list_of_lists:
        if isinstance(sel, list):
            new_list.extend(sel)
        else:
            new_list.append(sel)
    return new_list


def find_x_y_min_max(x_min, x_max, y_min, y_max, point):
    if x_min > point[0]:
        x_min = point[0]
    if x_max < point[0]:
        x_max = point[0]
    if y_min > point[1]:
        y_min = point[1]
    if y_max < point[1]:
        y_max = point[1]
    return x_min, x_max, y_min, y_max


def calculate_distance(point1, point2):
    # return math.sqrt(((point1[0] - point2[0]) ** 2) + (((point1[1] - point2[1])) ** 2))  # *GV.ASPECT_RATIO
    return math.dist(point1, point2)


def midpoint(p1, p2, is_round=True):
    if is_round:
        m_point = int(round(((p1[0] + p2[0]) / 2), 0)), int(round((p1[1] + p2[1]) / 2, 0))
    else:
        m_point = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    return m_point


def new_point_f(begin_point, dist, dir_x, dir_y, slope_, r):
    return begin_point[0] + dir_x * (dist / r), begin_point[1] + dir_y * (dist * slope_ / r)


def calculate_velocity(distances, n_frames):
    return sum(distances) / n_frames


def find_change_direction_index(info_tracking, index=15):
    default = len(info_tracking)
    if -default <= index < default:
        pass

    else:
        index = default
    dir = info_tracking[-1][2]
    for i in range(-2, -index, -1):
        if dir_suddenly_changes(dir, info_tracking[i][2]) and info_tracking[i][
            1] > 2:  # when track is stopped he might change direction often but it does not matter
            return i
        if info_tracking[i][1] > 2:
            dir = info_tracking[i][2]
    return -1


def calculate_direction_pt(start_point, end_point, range=False, r=90):
    x1 = start_point[0]
    y1 = start_point[1]
    x2 = end_point[0]
    y2 = end_point[1]
    # change Y
    degrees = (math.degrees(math.atan2(-(y2 - y1), x2 - x1)) + 360) % 360
    if not range:
        return degrees

    range_ = degree_to_range(degrees, r=r)
    return range_


def is_point_left_of_line(point, line):
    # Calculate the signed area of the triangle formed by the point and the endpoints of the line
    area = 0.5 * ((line.coords[1][0] - line.coords[0][0]) * (point.y - line.coords[0][1])
                  - (point.x - line.coords[0][0]) * (line.coords[1][1] - line.coords[0][1]))

    if area > 0:
        # The point is to the left of the line
        return 1
    else:
        # The point is to the right of the line or on the line
        return 0


def get_rect_points(pt1, pt2, pt3):
    # calculate the slope of the line passing through pt1 and pt2
    angle = calculate_direction_pt(pt2, pt1)

    # print()

    if is_point_left_of_line(Point(pt3), LineString([pt1, pt2])):
        perp_slope = add_degree(angle, 90)
    else:
        perp_slope = add_degree(angle, 270)

    # print(calculate_direction_pt(pt3, pt1),angle, perp_slope)

    # calculate the maximum distance between the line and pt3
    dist = LineString([pt1, pt2]).distance(Point(pt3))

    # print(pt1, pt2, pt3, dist)

    # calculate the coordinates of the predicted point

    pt3 = predict_point_pos(pt2, dist, perp_slope, integer=True)
    pt4 = predict_point_pos(pt1, dist, perp_slope, integer=True)

    return [pt1, pt2, pt3, pt4]


def degree_to_range(degrees, r=90):
    if (degrees - r) < 0 or (degrees + r) > 360:
        range_ = [((degrees - r) % 360, 360), (0, (degrees + r) % 360)]
    else:
        range_ = [degrees - r, degrees + r]
    return range_


def add_degree(degrees, r):
    if (degrees + r) >= 360:
        return (degrees + r) % 360
    else:
        return degrees + r


def nearest_point(l, track, point=True):
    if point:
        n_p = list(nearest_points(l.coords, Point(track.tlbr_to_center_point(track.tracking_tlbr[-1])))[0].coords)[0]
    else:
        n_p = list(
            nearest_points(l.coords, LineString(list(map(track.tlbr_to_center_point, track.tracking_tlbr[:-2]))))[
                0].coords)[0]
    return n_p


def is_in_degree_range(line_range_degree, track_degree):
    # print(line_range_degree, track_degree)
    if isinstance(line_range_degree[0], tuple):
        lower, upper = line_range_degree[0], line_range_degree[1]
        if lower[0] <= track_degree <= lower[1] or upper[0] <= track_degree <= upper[1]:
            return True
    else:
        lower, upper = line_range_degree[0], line_range_degree[1]
        if lower <= track_degree <= upper:
            return True
    return False


def get_min_index(inputlist):
    # get the minimum value in the list
    min_value = min(inputlist)

    # return the index of minimum value
    min_index = inputlist.index(min_value)
    return min_index


def get_max_index(inputlist):
    # get the minimum value in the list
    max_value = max(inputlist)

    # return the index of minimum value
    max_index = inputlist.index(max_value)
    return max_index


def first_item_list(list_of_lists):
    return [item[0] for item in list_of_lists]


def associate_track(t_new, t_old, track_id=False):
    def associate_category(cls_name_new, cls_name_old):
        for cls_name in cls_name_old:
            if cls_name not in cls_name_new:
                cls_name_new[cls_name] = cls_name_old[cls_name]
            else:
                cls_name_new[cls_name] += cls_name_old[cls_name]
        return cls_name_new

    print("ASSOCIATION")
    if track_id:
        t_new.track_id = t_old.track_id
    t_new.is_activated = True
    t_new.tracking_tlbr = t_old.tracking_tlbr + t_new.tracking_tlbr
    t_new.info_tracking = t_old.info_tracking + t_new.info_tracking
    t_new.start_frame = t_old.start_frame
    t_new.category_counting = associate_category(t_new.category_counting, t_old.category_counting)
    if t_old.entry:
        t_new.entry = t_old.entry

    t_new.info_video["Start"] = deepcopy(t_old.info_video["Start"])
    t_new.info_video["End"] = deepcopy(t_old.info_video["End"])


def is_valid_distance(dist, track_dist, percentage=0.3):
    if dist * (1 - percentage) <= track_dist <= dist * (1 + percentage):
        return True
    return False


def closest_line_distance(track, lines, mode="Entry"):
    if mode == "Entry":
        closest_point = track.tlbr_to_bottom_center_point(track.tracking_tlbr[0])
        dir = add_degree(track.calculate_direction_track(track, length=GV.LENGTH_AVG_DIR, newer=False, raw=False), 180)
        index = 0
    else:
        closest_point = track.tlbr_to_bottom_center_point(track.tracking_tlbr[-1])
        dir = track.calculate_direction_track(track, length=GV.LENGTH_AVG_DIR, newer=True, raw=False)
        index = len(track.tracking_tlbr) - 1
    # Check if intersects

    dist_l = []
    min_distance = float("inf")
    closest_line = None
    intersects = False
    for l in lines:
        avg_point = l.avg_point(mode)
        dist = calculate_distance(closest_point, avg_point)
        # dist = Point(closest_point).distance(l.coords)
        predict_point = predict_point_pos(closest_point, dist, dir)
        # print(track,"DIR", dir, "Closest Point", closest_point, "pred_point", predict_point, "dist")
        predicted_polyline = LineString([closest_point, predict_point])
        if predicted_polyline.intersects(l.coords):
            # print("predicted point intersects with", l)
            intersects = True
            intersection = l.point_intersection(predicted_polyline)
            dist = Point(intersection).distance(l.coords)
            if dist < min_distance:
                min_distance = dist
                closest_line = l
        else:
            dist_l.append(predicted_polyline.distance(l.coords))

    closest_line = closest_line if intersects else lines[get_min_index(dist_l)]

    return [closest_line, index]


def closest_region_distance(track, lines, mode="Entry"):
    if mode == "Entry":
        closest_point = track.tlbr_to_bottom_center_point(track.tracking_tlbr[0])
        dir = add_degree(track.calculate_direction_track(track, length=GV.LENGTH_AVG_DIR, newer=False, raw=False), 180)
        index = 0
    else:
        closest_point = track.tlbr_to_bottom_center_point(track.tracking_tlbr[-1])
        dir = track.calculate_direction_track(track, length=GV.LENGTH_AVG_DIR, newer=True, raw=False)
        index = len(track.tracking_tlbr) - 1
    # Check if intersects

    dist_l = []
    min_distance = float("inf")
    closest_region = None
    intersects = False
    for l in lines:
        #avg_point = l.avg_point(mode)
        avg_point = l.avg_point(mode)
        dist = calculate_distance(closest_point, avg_point)
        #dist = Point(closest_point).distance(l.coords)
        predict_point = predict_point_pos(closest_point, dist, dir)

        print(track, "DIR", dir, "Closest Point", closest_point, "pred_point", predict_point, "dist", "line", l)
        predicted_polyline = LineString([closest_point, predict_point])
        if mode == "Entry":
            if l == track.exit[0] and predicted_polyline.intersects(track.exit[0].polygon):
                print(track, "First point of track already is inside of exit region")
                dist_l.append(float("inf"))
                continue
        else:
            if l == track.entry[0] and predicted_polyline.intersects(track.entry[0].polygon):
                print(track, "First point of track already is inside of exit region")
                dist_l.append(float("inf"))
                continue

        print("go here")

        if predicted_polyline.intersects(l.coords):
            print("predicted point intersects with", l)
            intersects = True
            intersection = l.point_intersection(predicted_polyline)
            dist = Point(intersection).distance(l.coords)
            if dist < min_distance:
                min_distance = dist
                closest_region = l
        else:
            dist_l.append(predicted_polyline.distance(l.coords))

    closest_region = closest_region if intersects else lines[get_min_index(dist_l)]

    return [closest_region, index]


def aux():
    pass
    """if l_intersects:

    dists = [Point(point_intersection).distance(Point(closest_point)) for _, _, point_intersection in l_intersects]
        info_intersect = l_intersects[get_min_index(dists)]
    line_o, _, _ = info_intersect
    if mode == "Entry":
        # track.tracking_tlbr.insert(0, track.center_point_to_tlbr(point_intersection,
        # track.area))
        return [line_o, 0]  # Line, cross_indexº
    else:
        # track.tracking_tlbr.append(
        # track.center_point_to_tlbr(point_intersection, track.area))
        # TODO: Might Have a problem if it intersects more than 1 exit line
        return [line_o, len(track.tracking_tlbr) - 1]

    # If not intersect check what is the closest Point to Predicted Point
    for info in aux:
        diff_predicted_dist.append(calculate_distance(info[0], info[1]))
    # print(track, entry_lines, diff_predicted_dist)
    line_o = lines[get_min_index(diff_predicted_dist)]
    line_string = line_o.coords
    p2 = list(nearest_points(line_string, Point(closest_point))[0].coords)[0]
    if mode == "Entry":
        # track.tracking_tlbr.insert(0, track.center_point_to_tlbr(p2, track.area))
        return [line_o, 0]  # Line,  entry_cross_index
    else:
        track.tracking_tlbr.append(track.center_point_to_tlbr(p2, track.area))
        return [line_o, len(track.tracking_tlbr) - 1]  # Line,  exit_cross_index"""


def get_last_tracked_frame(removed_track, active_stracks):
    tracked_tracks_last_frames = [t.start_frame for t in active_stracks if
                                  t.start_frame > removed_track.end_frame]
    last_tracked_frame = None
    if tracked_tracks_last_frames:
        # print(tracked_tracks_last_frames)
        last_tracked_frame = tracked_tracks_last_frames[get_min_index(tracked_tracks_last_frames)]

    return last_tracked_frame


def is_valid_direction(dir1, dir2, valid_diff=GV.DEGREE_RANGE_CANDIDATE):
    if calculate_diff_degree(dir1, dir2) > valid_diff:
        return False
    else:
        return True


def is_in_reasonable_distance(t, c, actual_frame=None):
    # If no actual frame is given, associate candidates
    if actual_frame is None:
        vel, actual_point, last_detected_point, dir = c.vel, t.tlbr_to_center_point(
            t.tracking_tlbr[0]), c.tlbr_to_center_point(c.tracking_tlbr[-1]), c.dir
        newest_frame, oldest_frame = t.start_frame, c.end_frame
    else:
        # If there's only 1 frame detection
        vel, actual_point, last_detected_point, dir = t.vel, c.center_point, t.tlbr_to_center_point(
            t.tracking_tlbr[-1]), t.info_tracking[-1][2]
        newest_frame, oldest_frame = actual_frame, t.info_tracking[-1][0]

    # Calculate the difference in direction
    last_dir, computed_dir = dir, calculate_direction_pt(last_detected_point, actual_point)
    diff_angle = calculate_diff_degree(dir, computed_dir)
    if diff_angle > 45:
        # If the difference in direction is too large, return False
        return False

    # Calculate predicted point
    n_lost_frames = (newest_frame - oldest_frame)
    predict_dist = n_lost_frames * vel
    predict_point = predict_point_pos(last_detected_point, predict_dist, dir)

    # Check if the predicted point is valid
    thresh = GV.THRESH_VALID_PREDICT_POINT
    return is_valid_predict_point(predict_point, actual_point, threshold=thresh)



def overlap_tlbr(tlbr1, tlbr2):
    rect1, rect2 = tlbr1, tlbr2
    p1 = Polygon([(rect1[0], rect1[1]), (rect1[2], rect1[1]), (rect1[2], rect1[3]), (rect1[0], rect1[3])])
    p2 = Polygon([(rect2[0], rect2[1]), (rect2[2], rect2[1]), (rect2[2], rect2[3]), (rect2[0], rect2[3])])

    print("DISTANCE", p1.distance(p2))

    return p1.distance(p2) < GV.THRESH_VALID_PREDICT_BOX


def is_valid_predicted_tlbr(predict_tlbr, tlbr):
    print("PRED", predict_tlbr, "actual", tlbr, overlap_tlbr(predict_tlbr, tlbr))
    if overlap_tlbr(predict_tlbr, tlbr):
        return True
    return False


def is_valid_predict_point(predicted_point, real_point, threshold=GV.THRESH_VALID_PREDICT_POINT):
    # print("DIST", abs(calculate_distance(real_point, predicted_point)))
    # print(abs(calculate_distance(real_point, predicted_point)))
    if abs(calculate_distance(real_point, predicted_point)) < threshold:
        return True
    return False


def closest_entry_degree_range(entry_lines, track_degrees):
    diff = [abs(e.degrees - track_degrees) for e in entry_lines]

    return entry_lines[get_min_index(diff)]


def calculate_diff_degree(degree1, degree2):
    return 180 - abs(abs(degree1 - degree2) - 180)


def mean_degrees(list_degrees):
    return (sum(list_degrees) % 360) / len(list_degrees)


def predict_point_pos(point, d, theta, integer=False):
    theta_rad = radians(theta)
    if integer:
        return int(point[0] + d * cos(theta_rad)), int(point[1] + d * -sin(theta_rad))
    return point[0] + d * cos(theta_rad), point[1] + d * -sin(theta_rad)


def find_good_dir_index(track, length, newer, raw):
    if raw:
        if newer:
            return -1
        else:
            return 0

    default = len(track.info_tracking)
    if newer == False:
        index = 0

        for i in range(default):
            if track.info_tracking[i][1] > 2:  # only matters dirs where a distance is relevant
                index = i
                break
        default = len(track.info_tracking)
        if -default <= length + index < default:
            pass
        else:
            if default - length > 0:
                index = default - length
            else:
                index = 0
    else:
        index = -1
        for i in range(-1, -default, -1):
            if track.info_tracking[i][1] > 2:
                index = i
                break
        default = len(track.info_tracking)
        if -default <= -length - index + 1 < default:
            pass
        else:
            if -default + length - 1 > 0:
                index = -1
            else:
                index = -default + length - 1

    return index


def get_movement(t, l, movements, associate_entry=True):
    if associate_entry:
        mov = [m for m in movements if l.name("Entry") + str(t.exit[0].name("Exit")) == m.name][0]
    else:
        mov = [m for m in movements if str(t.entry[0].name("Entry")) + str(l.name("Exit")) == m.name][0]

    return mov


def closest_candidate_distance(t, list_c):
    closest_point_t = t.tlbr_to_center_point(t.tracking_tlbr[0])
    diff = [calculate_distance(c.tlbr_to_center_point(c.tracking_tlbr[-1]), closest_point_t) for c in list_c]
    # print(list_c, diff)
    return list_c[get_min_index(diff)]


def range_to_degrees(range_degree):
    if isinstance(range_degree[0], tuple):

        lower, upper = range_degree[0], range_degree[1]
        degrees = ((lower[1] + lower[0] + upper[1] + upper[0]) / 2) % 360
    else:
        lower, upper = range_degree[0], range_degree[1]
        degrees = (upper + lower) / 2
    return degrees


def calculate_exit(entry_coord, offset=50):
    exit = list(LineString(entry_coord).parallel_offset(offset).coords)
    exit = list(tuple(map(int, tup)) for tup in exit)
    return exit


def sum_line_distances(t, entry_cross_index, exit_cross_index):
    dist = 0
    tracking_line = list(map(t.tlbr_to_center_point, t.tracking_tlbr[entry_cross_index:exit_cross_index + 1]))
    for i in range(0, len(tracking_line)):
        if i + 1 == len(tracking_line):
            break
        dist += calculate_distance(tracking_line[i], tracking_line[i + 1])
    return dist


def find_tracking_tlbr_index(t, point_intersection):
    tracking_line = np.asarray(list(map(t.tlbr_to_center_point, t.tracking_tlbr)))
    dist_2 = np.sum((tracking_line - point_intersection) ** 2, axis=1)
    return np.argmin(dist_2)


def is_new_track_associated(track, list_tracks):
    for t2 in list_tracks:
        if t2.info_video == track.info_video:
            return False

    return True


def farest_point(pt, pts):
    pt_far = pts[0]
    for p in pts:
        if calculate_distance(p, pt) > calculate_distance(pt_far, pt):
            pt_far = p
    return pt_far


def get_avg_point(list):
    return [sum(x) / len(x) for x in zip(*list)]


def average_dir(info_tracking, range=True, r=45, only_dir=False):
    x = y = 0
    if not only_dir:
        for frame, dist, dir, _ in info_tracking:
            dir = radians(dir)
            x += dist * cos(dir)
            y += dist * sin(dir)

    else:
        for dir in info_tracking:
            dir = radians(dir)
            x += cos(dir)
            y += sin(dir)

    # print("average_angle", math.degrees(math.atan2(y, x)) % 360)
    average_angle = math.degrees(math.atan2(y, x)) % 360  # math.atan2(y, x)
    # average_angle_degree = degree_to_range(average_angle, r=r)

    # new_dist_dir = [dir for dir in dirs if is_in_degree_range(average_angle_degree, dir[2])]
    # print(new_dist_dir)
    # x,y, _ = good_avg_dir(new_dist_dir, length)
    # average_angle = math.degrees(math.atan2(y, x)) % 360  # math.atan2(y, x)

    if not range:
        return average_angle

    range_ = degree_to_range(average_angle, r=r)
    # print(average_angle)

    return range_


def find_end_frame(tracks):
    return min([track.end_frame for track in tracks])


"""def overlap_period(t1, t2):
    # t2 must be older than t1
    if t2.start_frame >= t1.end_frame:
        return 1
    start1, end1 = t1.start_frame, t1.end_frame
    start2, end2 = t2.start_frame, t2.end_frame
    how much does the range (start1, end1) overlap with (start2, end2)
    return max(max((end2 - start1), 0) - max((end2 - end1), 0) - max((start2 - start1), 0), 0)"""


def overlap_time(new_track, old_track):
    if old_track.start_frame >= new_track.end_frame:
        return -1
    overlap = min(new_track.end_frame, old_track.end_frame) - max(new_track.start_frame, old_track.start_frame)
    return max(overlap, 0)


def dir_to_str(tupl, track=True):
    switcher = {
        (0, -1): 'N',
        (1, -1): 'NE',
        (1, 0): 'E',
        (1, 1): 'SE',
        (0, 1): 'S',
        (-1, 1): 'SW',
        (-1, 0): 'W',
        (-1, -1): 'NW',
    }
    str_dir = switcher.get(tupl, "Invalid Orientation")
    if track:
        if len(str_dir) == 1:
            var = [v for k, v in switcher.items() if str_dir in v]
        else:
            var = [v for k, v in switcher.items() if v in str_dir]
        return tuple(var)
    else:
        return str_dir


def get_indexes_smaller_bigger(list):
    return np.array(list).argsort().tolist()


def long_distance(pt1, pt2):
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    return abs(x2 - x1) + abs(y2 - y1)


def check_class(cls_name_counting):
    return int(max(cls_name_counting, key=cls_name_counting.get))


def lines_distance(lines):
    for pair in list(itertools.permutations(lines, 2)):
        pair[0].distances[pair[1].name] = long_distance(pair[0].mid_point, pair[1].mid_point)


def is_inside_ignored_regions(pt, ignored_regions):
    for box in ignored_regions:

        x1, y1, w, h = float(box.get("left")), float(box.get("top")), float(box.get("width")), float(box.get("height"))

        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        if point_is_inside_box(pt, intbox):
            return True
    return False


def point_is_inside_box(point, bounding_box):
    # TODO: check manually, creating objects makes the software slower
    p1 = Polygon(
        [(bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[1]), (bounding_box[2], bounding_box[3]),
         (bounding_box[0], bounding_box[3])])
    return p1.contains(Point(point))


def is_same_vehicle(t1, t2):
    if calculate_distance(t1.tlbr_to_center_point(t1.tlbr), t2.tlbr_to_center_point(t2.tlbr)) < 5:
        return True
    return False


def overlap(t1, t2, percentage=65):
    # TODO: calculate area manually, creating objects makes the software slower
    rect1, rect2 = t1.tlbr, t2.tlbr
    p1 = Polygon([(rect1[0], rect1[1]), (rect1[2], rect1[1]), (rect1[2], rect1[3]), (rect1[0], rect1[3])])
    p2 = Polygon([(rect2[0], rect2[1]), (rect2[2], rect2[1]), (rect2[2], rect2[3]), (rect2[0], rect2[3])])

    if percentage == 0:
        return p1.intersects(p2)

    overlap_percentage_p2 = (p1.intersection(p2).area / p2.area) * 100
    overlap_percentage_p1 = (p1.intersection(p2).area / p1.area) * 100

    if overlap_percentage_p2 > percentage or overlap_percentage_p1 > percentage:
        # print(t1,t2, overlap_percentage_p1, overlap_percentage_p2)
        return True
    return False


def calculate_arrows(region_entries):
    arrows = []

    for i, region_entry in enumerate(region_entries):
        mid_point_entry = midpoint(region_entry[0], region_entry[1])
        slope = region_entry[-1]
        r = math.sqrt(1 + slope ** 2)
        # How to know the direction?????
        distance = 70
        end_point = (int(mid_point_entry[0] + (distance / r)),
                     int(mid_point_entry[1] + (distance * slope / r)))

        arrows.append((mid_point_entry, end_point))
    return arrows


"""def calculate_directional_path(self):

    entries = self.lines_coordinates["Entries"]
    exits = self.lines_coordinates["Exits"]

    for i, entry in enumerate(entries):
        for j, exit in enumerate(exits):
            letter = letters[i]
            number = j + 1
            mov = letter + str(number)
            entry_mid_point = midpoint(entry[0], entry[1])
            exit_mid_point = midpoint(exit[0], exit[1])

            print(mov, entry_mid_point, exit_mid_point)

            slope_ = abs(slope(entry_mid_point, exit_mid_point))
            r = math.sqrt(1 + slope_ ** 2)
            dir_x, dir_y = calculate_direction_pt(entry_mid_point, exit_mid_point)
            dist = -200
            begin_point = new_point_f(entry_mid_point, dist, dir_x, dir_y, slope_, r)
            self.movs_directional_path[mov] = [round_point(begin_point)]
            self.movs_directional_slope[mov] = [dir_x, dir_y, slope_, r]
            dist = 5

            end_point = new_point_f(begin_point, dist, dir_x, dir_y, slope_, r)
            d = Point(end_point).distance(Point(exit_mid_point))
            while True:
                self.movs_directional_path[mov].append(round_point(end_point))
                begin_point = end_point
                end_point = new_point_f(begin_point, dist, dir_x, dir_y, slope_, r)
                new_d = Point(end_point).distance(Point(exit_mid_point))
                print(mov, slope_)
                if d < new_d:
                    break
                d = new_d

            self.movs_directional_path[mov].append(exit_mid_point)
            
    def is_too_close_f(center_points, point):
    # print(center_points)
    for id in center_points:
        dist = calculate_distance(center_points[id], point)
        # print(id, dist, point)
        if dist < 50 and dist != 0:
            # print("DEMASIADO PERTO")
            return id, True
    return -1, False


def slope(p1, p2):
    denominador = p2[0] - p1[0]
    if denominador == 0:
        denominador = 0.00001
    return (p2[1] - p1[1]) / denominador


def search_nearest_point(directional_path, point):
    dist = 1000000
    index = 0
    for i, directional_point in enumerate(directional_path):
        new_dist = Point(directional_point).distance(Point(point))
        if new_dist < dist:
            dist = new_dist
            index = i
    if index + 1 < len(directional_path):
        return directional_path[index + 1]
    else:
        return directional_path[index]


"""

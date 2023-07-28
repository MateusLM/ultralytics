import copy
import itertools
import math
import sys
from copy import deepcopy

import numpy as np
import torch
from shapely import LineString, Point

from .utils import matching
from ..utils.bad_place import BadPlace
from ..utils.global_variables import *
from ..utils.logger import Logger
from ..utils.math_operations import calculate_distance, calculate_velocity, calculate_direction_pt, \
    calculate_diff_degree, average_dir, check_class, is_in_reasonable_distance, closest_candidate_distance, \
    associate_track, get_avg_point, is_valid_direction, overlap_time, has_category_in_common, find_tracking_tlbr_index, \
    closest_region_distance, closest_line_distance, find_end_frame, get_last_tracked_frame, overlap
from .basetrack import BaseTrack, TrackState
from ultralytics.utils.draw_operations import get_color
from .utils.kalman_filter import KalmanFilterXYAH
from ..utils.movement import Movement


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, tlwh, score, category):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # added
        # Count number of times the object is classified as a certain class

        self.color_cluster = (int(0), int(0), int(255))  # red
        self.color = None

        self.count = False
        self.is_counted = False
        self.excel_row = None
        self.entry = None
        self.exit = None
        self.last_exit = None
        self.associated = False

        self.info_video = {"Start": None, "End": None}
        self.cluster = None
        self.category_counting = {}
        self.category = self.update_category(category, score)
        self.reset = False

        self.tracking_tlbr = [self.tlbr]
        self.info_tracking = []
        self.update_end_video = None
        self.gave_info_to_region = False
        self._dir, self._area, self._vel = None, None, None

    def reset_count(self,info_video, frame_id):
        #print("RESET", self)
        self.reset = True
        self.state = TrackState.Tracked
        self.count = False
        self.is_counted = False
        self.excel_row = None
        self.entry = None
        self.last_exit = deepcopy(self.exit)
        self.exit = None
        self.associated = False
        self.info_video = {"Start": info_video, "End": None}
        self.category_counting = {}

        self.color = get_color(abs(self.track_id))
        self.tracklet_len = 2

        self.tracking_tlbr = self.tracking_tlbr[-2:]
        self.info_tracking = self.info_tracking[-2:]
        self.start_frame = self.frame_id
        self.frame_id = frame_id
        self.update_end_video = None
        self.last_frame_id = frame_id
        self._dir, self._area, self._vel = None, None, None

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, info_video):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.last_frame_id = deepcopy(self.frame_id)
        self.frame_id = frame_id
        self.color = get_color(abs(self.track_id))
        self.start_frame = frame_id
        self.info_video["Start"] = info_video

    def re_activate(self, new_track, frame_id, info_video, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self._tlwh = new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.last_frame_id = deepcopy(self.frame_id)
        self.frame_id = frame_id

        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        # added
        self._dir, self._area, self._vel = None, None, None
        self.info = None
        # added
        self.category = self.update_category(new_track.category, self.score)
        self.update_end_video = info_video

        self.tracking_tlbr.append(self.tlbr)
        if len(self.info_tracking) < 20 or calculate_diff_degree(self.info_tracking[-1][2], self.dir) < 90:
            self.info_tracking.append(
                (
                    self.frame_id,
                    calculate_distance(self.center_point, self.tlbr_to_center_point(self.tracking_tlbr[-2])),
                    self.dir, STrack.tlbr_to_area(self.tlbr)))

    def update(self, new_track, frame_id, info_video):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.last_frame_id = deepcopy(self.frame_id)
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # added
        self.category = self.update_category(new_track.category, self.score)

        self._dir, self._area, self._vel = None, None, None

        self.update_end_video = info_video
        self.tracking_tlbr.append(self.tlbr)
        """print("BEFORE ADDDING DisTDIR", self.track_id,
              calculate_distance(self.center_point, self.tlbr_to_center_point(self.tracking_tlbr[-2])),
              self.center_point, "AREA", STrack.tlbr_to_area(self.tlbr), "AVG_AREA",
              self.area if self.info_tracking else None)"""

        test= True
        if len(self.info_tracking) < 20 or (calculate_diff_degree(self.info_tracking[-1][2], self.dir) < 90 and STrack.tlbr_to_area(self.tlbr) / self.area > 0.80 if test else True):

            self.info_tracking.append(
                (
                    self.frame_id,
                    calculate_distance(self.center_point, self.tlbr_to_center_point(self.tracking_tlbr[-2])),
                    self.dir, STrack.tlbr_to_area(self.tlbr)))
            #print("%AREA", STrack.tlbr_to_area(self.tlbr) / self.area)

        else:
            pass
            #print(self.track_id, "WEIRD DIR")

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # added
    @property
    # @jit(nopython=True)
    def center_point(self):
        """Convert current position to format `(center x, center y)"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret[:2]

    def bottom_center_point(self):
        ret = np.asarray(self.tlwh).copy()

        center_x = ret[0] + ret[2] / 2
        bottom_left_y = ret[1] + ret[3]

        # new array of shape (center x, top left y)
        ret = np.array([center_x, bottom_left_y])

        return ret[:2]

    @property
    # @jit(nopython=True)
    def dir(self, length=5, newer=True, range=False):
        """Get track orientation of last 5 detections`.
        """
        if not self._dir:
            self._dir = self.calculate_direction_track(self, length=length, range=range, newer=newer, raw=True)
        return self._dir

    @property
    # @jit(nopython=True)
    def vel(self, newer=True):
        """Get track velocity of last 5 detections"""
        # print(self, "CALCULATE VEL")
        if not self._vel:
            self._vel = self.avg_vel(self, newer=newer)
        return self._vel

    @property
    # @jit(nopython=True)
    def area(self, newer=True):
        if not self._area:
            self._area = self.avg_area(self, frame_id=self.frame_id, length=10, newer=newer)
        return self._area

    @property
    def dist(self):
        if self.entry:
            first_point = self.tlbr_to_bottom_center_point(self.tracking_tlbr[self.entry[1]])
        else:
            first_point = self.tlbr_to_bottom_center_point(self.tracking_tlbr[0])
        #print(self, "entry?", self.entry, "DIST", calculate_distance(first_point, self.tlbr_to_bottom_center_point(self.tracking_tlbr[-1])))
        return calculate_distance(first_point, self.tlbr_to_bottom_center_point(self.tracking_tlbr[-1]))

    @property
    def is_stopped(self):
        if len(self.info_tracking) > 2:
            # print(t, t.vel)
            if self.vel < 0.2:
                # print(t, "is stopped")
                return True
        return False

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_area(tlbr):
        box = tlbr
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_center_point(tlbr):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        # print(tlbr)
        tlwh = STrack.tlbr_to_tlwh(tlbr)
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret[:2]

    @staticmethod
    def tlbr_to_bottom_center_point(tlbr):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        # print(tlbr)
        tlwh = STrack.tlbr_to_tlwh(tlbr)
        ret = np.asarray(tlwh).copy()

        center_x = ret[0] + ret[2] / 2
        bottom_left_y = ret[1] + ret[3]

        # new array of shape (center x, top left y)
        ret = np.array([center_x, bottom_left_y])

        return ret[:2]

    @staticmethod
    # @jit(nopython=True)
    def center_point_to_tlbr(center_point, area):
        """Convert center point to box (tlbr) to format `(min x, min y, max x, max y)`"""
        # square
        width = height = math.sqrt(area)
        minx = center_point[0] - width / 2
        maxx = center_point[0] + width / 2
        miny = center_point[1] - height / 2
        maxy = center_point[1] + height / 2

        return [minx, miny, maxx, maxy]

    @staticmethod
    # @jit(nopython=True)
    def bottom_center_point_to_tlbr(center_point, area):
        width = height = math.sqrt(area)
        minx = center_point[0] - width / 2
        maxx = center_point[0] + width / 2
        maxy = center_point[1]
        miny = center_point[1] - height

        return [minx, miny, maxx, maxy]

    @staticmethod
    # @jit(nopython=True)
    def info_avg_vel(t, frame_id, length=3):
        """Get track velocity of last 5 detections`.
        """
        list_frame_ids = list(zip(*t.info_tracking))[0]
        i_p = list_frame_ids.index(frame_id)
        info_tracking = t.info_tracking[:i_p + 1]

        info_tracking = info_tracking[-length:]
        frame = info_tracking[0][0]
        n_frames = (frame_id - frame) + 1
        # print(frame_id, frame, n_frames)
        return calculate_velocity([dist for _, dist, _ in info_tracking], n_frames)

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_area(tlwh):
        box = STrack.tlwh_to_tlbr(tlwh)
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def avg_vel(t, length=5, newer=True):
        """Get track velocity of last 5 detections`.
        """
        if newer:
            info_tracking = t.info_tracking[-length:]
        else:
            info_tracking = t.info_tracking[:length]

        frame_old = info_tracking[0][0]
        frame_new = info_tracking[-1][0]

        n_frames = (frame_new - frame_old) + 1
        """print("CALCULATING AVG", "frame_lost", frame_old, "frame_id", frame_new, "n_frames", n_frames, "info_tracking",
              info_tracking)"""
        return calculate_velocity([dist for _, dist, _, _ in info_tracking], n_frames)

    @staticmethod
    def avg_area(t, frame_id, length=5, newer=True):
        """Get track velocity of last 5 detections`.
        """
        if newer:
            info_tracking = t.info_tracking[-length:]
        else:
            info_tracking = t.info_tracking[:length]

        frame = info_tracking[0][0]

        n_frames = (frame_id - frame)
        """print("CALCULATING AVG", "frame_lost", frame, "frame_id", frame_id, "n_frames", n_frames, "info_tracking",
              info_tracking)"""
        return calculate_velocity([area for _, _, _, area in info_tracking], n_frames)

    @staticmethod
    def calculate_direction_track(t, length=5, r=90, range=False, newer=True, raw=False):
        if newer:
            tracking_tlbr = t.tracking_tlbr[-length:]
        else:
            tracking_tlbr = t.tracking_tlbr[:length]
        pts = list(map(t.tlbr_to_center_point, tracking_tlbr))
        """if not raw:
            while calculate_distance(pts[0], pts[-1]) < MIN_DIRECTION_DIST and length <= len(t.tracking_tlbr):
                length += 20
                if newer:
                    tracking_tlbr = t.tracking_tlbr[-length:]
                else:
                    tracking_tlbr = t.tracking_tlbr[:length]
                pts = list(map(t.tlbr_to_center_point, tracking_tlbr))"""

        if raw:
            dir = calculate_direction_pt(pts[0], pts[-1], range=range, r=r)
        else:
            if newer:
                info_tracking = t.info_tracking[-length:]
            else:
                info_tracking = t.info_tracking[:length]

            mean_avg_dir = calculate_direction_pt(pts[0], pts[-1], range=False, r=r)
            meaningful_dirs = [dir for dir in info_tracking if
                               calculate_diff_degree(mean_avg_dir, dir[2]) <= DEGREE_RANGE_CANDIDATE]

            dir = average_dir(meaningful_dirs, range, r)

        return dir

    """def dir_suddenly_changes(self):
        if len(self.info_tracking) < 2:
            return False
        # print(self.track_id)
        return dir_suddenly_changes(self.info_tracking[-2][2], self.info_tracking[-1][2])"""

    def __repr__(self):
        return 'OT_{}_({}-{})_{}'.format(self.track_id, self.start_frame, self.end_frame, self.state)

    # added
    def update_category(self, category, score):
        if category not in self.category_counting:
            self.category_counting[category] = score
        else:
            self.category_counting[category] += score
        return check_class(self.category_counting)


class BYTETracker_Counter:
    def __init__(self, args, lines=None, ROI_coordinates=None, frame_id=1, frame_rate=30):
        # Tracking Variables
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = frame_id - 1
        self.args = args
        self.CLASSES = args.CLASSES
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # added
        self.marked_places = []  # type: list[BadPlace]
        self.lines = lines

        BaseTrack.reset()

        self.log = Logger(args.debug)
        self.categories = self.args.categories
        self.CLASSES = self.args.CLASSES
        self.is_preprocessing = True

        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        if self.lines:
            #print("INIT TRACKER")
            self.counter = {}
            self.movements = []  # type: list[Movement]
            self.ROI_coordinates = ROI_coordinates
            self.is_preprocessing = False

    def update(self, output_results, img_info, img_size, info_video, excel_row=None):
        self.excel_row = excel_row
        self.info_video = info_video

        self.frame_id += 1
        # added

        if output_results is None:
            output_results = torch.empty((0, 6))
        indexes = [np.where(output_results[:, -1].cpu().numpy() == self.CLASSES.index(cls)) for cls in
                   self.categories]
        indexes = [item for sublist in indexes for subsublist in sublist for item in subsublist]
        output_results = output_results[indexes]
        print(self.frame_id, output_results)

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        self.log.namestr(frame_id=self.frame_id)
        self.log.namestr(track_buffer=self.buffer_size, max_time_lost=self.max_time_lost)
        # added
        # only consider classes i want to show

        # print(len(output_results))
        """if self.args.use_ignoring_regions:
            indexes = [np.where(output_results[:, :4].cpu().numpy()) for bbox in
                       output_results[:, :4].cpu().numpy()]
            bboxes = output_results[:, :4]"""

        if output_results.shape[1] == 5:

            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            # added
            categories = output_results[:, -1]
        # added
        elif output_results.shape[1] == 6:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]  # x1y1x2y2
            # added
            categories = output_results[:, -1]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            # added
            categories = output_results[:, -1]

        # self.log.namestr(categorys=categorys, scores=scores)
        # print(bboxes)
        # print(yolov7)
        if not (self.args.yolov7 or self.args.yolor or self.args.yolov8):
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale
        # bboxes = self.update_ROI_coordinates(bboxes)
        # print(bboxes)
        # exit(1)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # added
        category = categories[remain_inds]
        category_second = categories[inds_second]

        self.log.namestr(scores_keep=scores_keep,
                         dets_second=dets_second,
                         scores_second=scores_second)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, category) for
                          (tlbr, s, category) in zip(dets, scores_keep, category)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        self.log.print("Step 1: Add newly detected tracklets to tracked_stracks")
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        self.log.namestr(unconfirmed=unconfirmed,
                         tracked_stracks=tracked_stracks)

        ''' Step 2: First association, with high score detection boxes'''
        self.log.print("Step 2: First association, with high score detection boxes (update and refind)")
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # self.log.print("strack_pool", strack_pool)
        # Predict the current location with KF
        # print(strack_pool)
        STrack.multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, self.info_video)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, self.info_video, new_id=False)
                refind_stracks.append(track)

        u_d = [detections[u].center_point for u in u_detection]
        self.log.namestr(refind_stracks=refind_stracks, activated_starcks=activated_starcks, u_detection=u_d)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        self.log.print("Step 3: Second association, with low score detection boxes (update, reactivate)")
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, category) for
                                 (tlbr, s, category) in
                                 zip(dets_second, scores_second, category_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_second = matching.iou_distance(r_tracked_stracks, detections_second)
        matches_second, u_track_second, u_detection_second = matching.linear_assignment(dists_second, thresh=0.5)
        # print(matches_second)
        for itracked, idet in matches_second:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            # self.log.namestr(track=track, det=det)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.info_video)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            # self.log.namestr(unconfirmed_track=track)
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        self.log.namestr(refind_stracks=refind_stracks, activated_starcks=activated_starcks, lost_stracks=lost_stracks)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # THEY DONT DEAL WITH UNCONFIRMED WITH LOW SCORE DETECTION BOXES
        self.log.print(
            "Deal with unconfirmed tracks, usually tracks with only one beginning frame (update of unconfirmed)")
        u_detections = [detections[i] for i in u_detection]
        # u_detections_cnt = [u.center_point for u in u_detections]
        # self.log.namestr(u_detections_cnt=u_detections_cnt)
        u_dists = matching.iou_distance(unconfirmed, u_detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(u_dists, thresh=0.9)
        for itracked, idet in matches:
            unconfirmed[itracked].update(u_detections[idet], self.frame_id, self.info_video)
            # print(unconfirmed[itracked], "VEM FAZER TRACK")
            # print("UNCONF",unconfirmed[itracked].center_point)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            # print("REMOV", track.center_point)
            track.mark_removed()
            removed_stracks.append(track)
            # print(track, "SO TEVE UMA DETECAO",)

        self.check_absent(lost_stracks)
        # self.check_occluded(lost_stracks)

        self.log.namestr(activated_starcks=activated_starcks, removed_stracks=removed_stracks,
                         lost_stracks=lost_stracks)

        # TODO: Before initializing a new track, check if this track was failed to be recognized by the tracker.
        all_tracked_tracks = self.joint_stracks(tracked_stracks, refind_stracks)
        all_entry_tracked_tracks = [track for track in all_tracked_tracks if track.entry]
        all_lost_stracks = self.joint_stracks(self.lost_stracks, lost_stracks)
        self.log.print(all_entry_tracked_tracks)
        lost_absent_stracks = [l for l in all_lost_stracks if
                               l.state == TrackState.Absent]
        """lost_occluded_stracks = [l for l in all_lost_stracks if
                                 l.state == TrackState.Occluded]"""
        # MAYBE ALSO DEAL WITH UNCONFIRMED FOR A SINGLE DETECTION ONLY, we doint want to miss information
        # one_det_tracks = [track for track in removed_stracks]
        two_det_tracks = [track for track in activated_starcks if len(track.tracking_tlbr) == 2]
        # print("one_det", one_det_tracks,"two_det", two_det_tracks)
        if self.lines:
            for two_det_track in two_det_tracks:
                bool, lost_absent_candidates = self.is_track_from_absent(two_det_track, lost_absent_stracks)
                if bool:
                    track = closest_candidate_distance(two_det_track, lost_absent_candidates)
                    self.log.print("REFOUND ABSENT", track.tlbr_to_center_point(track.tracking_tlbr[-1]),
                                   "with TRack", two_det_track)

                    associate_track(two_det_track, track, track_id=True)
                    self.lost_stracks = self.sub_stracks(self.lost_stracks, [track])

                    continue
        """ Step 5: Init new stracks"""
        self.log.print("Step 5: Init new stracks (activate)")
        for inew in u_detection:
            det = u_detections[inew]

            if det.score < self.det_thresh:
                # print("Score Not enough")
                continue
            """if self.args.use_ignoring_regions:
                if self.is_point_from_ignoring_region(det):
                    continue"""
            if self.lines:
                if self.is_point_fake(det):
                    continue
                """if not test:
                    bool, lost_candidates = self.is_point_from_absent(det, lost_absent_stracks)
                    if bool:
                        track = closest_candidate_distance(det, lost_candidates)
                        self.log.print("REFOUND ABSENT", det.center_point, "FOR POINT", track.center_point)

                        #print("Associate", track, track.track_id)
                        track.update(det, self.frame_id, self.info_video)
                        activated_starcks.append(track)
                        continue"""

                if self.is_point_overlapping(det, all_entry_tracked_tracks, percentage=50): #or self.is_point_overlapping(
                        #det,
                        #all_tracked_tracks,
                        #percentage=80):
                    continue

            det.activate(self.kalman_filter, self.frame_id, self.info_video)
            activated_starcks.append(det)

        self.log.namestr(activated_starcks=activated_starcks, refind_stracks=refind_stracks)
        """ Step 5: Update state"""
        self.log.print("Step 5: Update state")

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)

        # removes tracked tracks from lost tracks
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)

        # self.lost_stracks = self.sub_stracks(self.lost_stracks, remove_counted_stracks)

        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        self.check_places()
        # added
        """ Step 6: Count Cars """
        self.log.print("Step 6: Count Cars (only if not in pre_processing)")
        if self.args.draw_option == "regions":
            self.update_counter_region()
        else:
            self.update_counter_line()
        self.count_movement(self.lost_stracks)
        # self.lost_stracks = self.sub_stracks(self.lost_stracks, removed_counted_tracks)

        # self.check_absent(self.lost_stracks)
        # self.check_occluded(self.lost_stracks)

        self.removed_stracks.extend(self.add_lost_to_removed(removed_stracks))
        # removes removed_tracks from lost tracks
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks = self.remove_removed(self.removed_stracks)
        # print("self", self.tracked_stracks, self.lost_stracks, self.removed_stracks)

        self.log.namestr(frame_id=self.frame_id, tracked_stracks=self.tracked_stracks, lost_stracks=self.lost_stracks,
                         removed_stracks=self.removed_stracks, marked_places=self.marked_places)

        self.check_counted_tracked_stracks()
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # print(self.info_video)
        # print("REMOVED size", len(self.removed_stracks))

        return output_stracks

    def check_counted_tracked_stracks(self):
        if self.lines:
            reset_tracks = []
            count_tracked = [track for track in self.tracked_stracks if track.count]

            for track in count_tracked:
                if len(track.tracking_tlbr) > 3:
                    # print(track, calculate_direction_pt(track.tlbr_to_center_point(track.tracking_tlbr[-3]),track.tracking_tlbr[-2]),calculate_direction_pt(track.tlbr_to_center_point(track.tracking_tlbr[-2]),track.tracking_tlbr[-1]),calculate_diff_degree(calculate_direction_pt(track.tlbr_to_center_point(track.tracking_tlbr[-3]),track.tracking_tlbr[-2]),calculate_direction_pt(track.tlbr_to_center_point(track.tracking_tlbr[-2]),track.tracking_tlbr[-1])) )

                    before_last_dir = calculate_direction_pt(track.tlbr_to_center_point(track.tracking_tlbr[-3]),
                                                             track.tlbr_to_center_point(track.tracking_tlbr[-2]))
                    last_dir = calculate_direction_pt(track.tlbr_to_center_point(track.tracking_tlbr[-2]),
                                                      track.tlbr_to_center_point(track.tracking_tlbr[-1]))
                    if not is_valid_direction(before_last_dir, last_dir):
                        reset_tracks.append(track)
            self.count_movement(reset_tracks)
            #[track.reset_count() for track in reset_tracks]
            # self.tracked_stracks = self.sub_stracks(self.tracked_stracks, removed_counted_tracks)

    def init_counter(self):

        entry = [l for l in self.lines]
        exit_ = [l for l in self.lines]
        entry_exit = []
        print(entry, exit_)

        directional_mov_lines = [(tup[0], tup[1]) for tup in itertools.product(entry, exit_)]
        directional_mov_names = [tup[0].name("Entry") + str(tup[1].name("Exit")) for tup in
                                 itertools.product(entry, exit_)]

        for i in range(len(entry)):
            entry_exit.append(entry[i].name("Entry"))
            entry_exit.append(exit_[i].name("Exit"))
        for i in range(len(directional_mov_names)):
            self.movements.append(Movement(directional_mov_lines[i][0], directional_mov_lines[i][1]))
            entry_exit.append(directional_mov_names[i])

        for mov in entry_exit:
            self.counter[mov] = {}
            for class_ in self.categories:
                self.counter[mov][class_] = []

        self.counter_init = copy.deepcopy(self.counter)
        return self.counter

    def count_section(self, track, mode="Entry"):
        mov = None
        if mode == "Entry":
            self.log.print(track, "Count as Entry")
            # print(track, "Count as Entry")
            # self.log.print(track.entry, track.dir)
            line = track.entry[0]
            mov = line.name("Entry")
        elif mode == "Exit":
            self.log.print(track, "Count as Exit")
            # print(track, "Count as Exit")
            line_exit = track.exit[0]
            mov = line_exit.name("Exit")
        if mov is None:
            sys.exit("Error mov can't be None")
        category = check_class(track.category_counting)
        # print(self.CLASSES[category])
        if self.CLASSES[category] in self.counter[mov]:
            self.counter[mov][self.CLASSES[category]].append(track.track_id)
        else:
            sys.exit("Class must exist to be counted")


    def update_counter_region(self):
        def sort_key(line):
            if line.is_valid_direction(track, "Entry") and line.valid_entry_point(track):
                return 0  # sort valid entry lines first
            elif line.is_valid_direction(track, "Exit"):
                return 1  # then sort valid exit lines next
            else:
                return 2  # sort all other lines last


        if not self.is_preprocessing:
            for track in self.tracked_stracks:
                if len(track.tracking_tlbr) > 1:

                    tracking_line = list(map(track.tlbr_to_bottom_center_point, track.tracking_tlbr[-BUFFER_LINE:]))
                    # print(track, tracking_line)
                    polyline = LineString(tracking_line)
                    # self.log.print(obj_id)
                    # print(track, track.count)
                    if not track.count:  # and not is_stopped(track):

                        #tracking_line = list(map(track.tlbr_to_bottom_center_point, track.tracking_tlbr[-BUFFER_LINE:]))
                        # print(track, tracking_line)
                        #polyline = LineString(tracking_line)
                        #print(track.track_id, polyline)
                        crosses = []
                        for r in self.lines:
                            if not r.polygon.intersection(polyline).is_empty:
                                self.log.print(track, "INTERSECTS", r)
                                crosses.append((r, r.point_intersection(polyline)))
                        if crosses:

                            crosses = sorted(crosses, key=lambda c: sort_key(c[0]))
                            for region, point_intersection in crosses:
                                self.log.print(track, "INTERSECTS", track.dir, region)
                                has_appeared = False
                                if not track.entry:
                                    if region.is_valid_direction(track, "Entry") and region.valid_entry_point(track):
                                        self.log.print(track, "CROSSES ENTRY", region.name("Entry"))
                                        entry_cross_index = len(track.tracking_tlbr) - 1
                                        track.entry = [region, entry_cross_index, track.dir]
                                        region.update_avg_point(point_intersection, "Entry")
                                        has_appeared = True
                                if region.is_valid_direction(track, "Exit") and not has_appeared and region.valid_exit_point(track):
                                    self.log.print(track, "CROSSES EXIT", region.name("Exit"))

                                    # tlbr = STrack.bottom_center_point_to_tlbr(point_intersection, track.area)
                                    # track.tracking_tlbr.insert(-1, tlbr)
                                    exit_cross_index = len(track.tracking_tlbr) - 1
                                    track.exit = [region, exit_cross_index, track.dir]
                                    region.update_avg_point(point_intersection, "Entry")
                        else:
                            if track.entry and not track.gave_info_to_region:
                                region = track.entry[0]
                                #print(track, "giving?")
                                if region.is_update() and len(track.tracking_tlbr) >= 2:
                                    point_intersection = track.tlbr_to_bottom_center_point(
                                        track.tracking_tlbr[track.entry[1]])
                                    last_point = track.tlbr_to_bottom_center_point(track.tracking_tlbr[-2])
                                    if len(track.info_tracking) >= 2 and track.last_frame_id == self.frame_id - 1 and region.polygon.intersects(Point(last_point)):
                                        print("UPDATE REGION", track, region.min_dist, calculate_distance(point_intersection, last_point))
                                        region.update_min_dist(calculate_distance(point_intersection, last_point))
                                        track.gave_info_to_region = True




                        if track.exit and not track.entry:
                            valid_exit = self.associate_entry(track)
                            if not valid_exit:
                                self.log.print(track, "FALSE EXIT", track.exit[0])
                                track.exit = None

                        if track.entry and track.exit:
                            track.count = True

    def update_counter_line(self):
        if not self.is_preprocessing:
            for track in self.tracked_stracks:
                if len(track.tracking_tlbr) > 1:
                    # self.log.print(obj_id)
                    # TODO: why comparing with the whole polyline (only need two last points)
                    # TODO: why LineString(line_coordinate).distance(Point(polyline.coords[-1]))
                    # print(track, track.count)
                    if not track.count:  # and not is_stopped(track):

                        tracking_line = list(map(track.tlbr_to_bottom_center_point, track.tracking_tlbr[-BUFFER_LINE:]))
                        # print(track, tracking_line)
                        polyline = LineString(tracking_line)
                        #print(track.track_id, polyline)
                        crosses = []
                        for l in self.lines:
                            if polyline.intersects(l.coords):
                                self.log.print(track, "INTERSECTS", l)
                                crosses.append((l, l.point_intersection(polyline)))
                        if crosses:
                            line = crosses[0][0]
                            point_intersection = crosses[0][1]
                            self.log.print(track, "INTERSECTS", track.dir, line)
                            has_crossed = False

                            if not track.entry:
                                if line.is_valid_direction(track, "Entry") and line.valid_entry_point(track):
                                    self.log.print(track, "CROSSES ENTRY", line.name("Entry"))

                                    # tlbr = STrack.bottom_center_point_to_tlbr(point_intersection, track.area)
                                    # track.tracking_tlbr.insert(-1, tlbr)
                                    entry_cross_index = len(track.tracking_tlbr) - 1
                                    track.entry = [line, entry_cross_index, track.dir]

                                    line.update_avg_point(point_intersection, "Entry")
                                    # line.update_dir(track, "Entry")
                                    has_crossed = True
                            if line.is_valid_direction(track, "Exit") and line.valid_exit_point(track,
                                                                                                point_intersection) and not has_crossed:
                                self.log.print(track, "CROSSES EXIT", line.name("Exit"))

                                # tlbr = STrack.bottom_center_point_to_tlbr(point_intersection, track.area)
                                # track.tracking_tlbr.insert(-1, tlbr)
                                exit_cross_index = len(track.tracking_tlbr) - 1
                                track.exit = [line, exit_cross_index, track.dir]

                                line.update_avg_point(point_intersection, "Exit")
                                # line.update_dir(track, "Exit")
                        # print(track, track.entry, track.exit)
                        if track.exit and not track.entry:
                            valid_exit = self.associate_entry(track)
                            if not valid_exit:
                                self.log.print(track, "FALSE EXIT", track.exit[0])
                                track.exit = None

                        if track.entry and track.exit:
                            track.count = True

    def count_movement(self, counted_tracks=None, remove=False):
        removed_tracked_counting = []
        if not self.is_preprocessing:
            for track in counted_tracks:
                if track.count and not track.is_counted:
                    self.log.print(track, "Count entry and exit", "CATEGORY", track.category)
                    track.category = check_class(track.category_counting)
                    line_entry = track.entry[0]
                    line_exit = track.exit[0]
                    mov = line_entry.name("Entry") + line_exit.name("Exit")
                    track.excel_row = self.excel_row
                    track.is_counted = True

                    track.info_video["End"] = track.update_end_video

                    # print("UPDATE CATEGORY")
                    # print(track, mov, self.CLASSES[track.category])
                    self.count_section(track, mode="Entry")
                    self.count_section(track, mode="Exit")
                    print(track, "MOV", mov, "COUNTED", "associated:", track.associated, "Category:",
                          self.CLASSES[track.category])
                    # if self.CLASSES[track.category] in self.counter[mov]:
                    self.counter[mov][self.CLASSES[track.category]].append(track.track_id)

                    if self.args.excel:
                        bool_ = False
                        for m in self.movements:
                            if m.name == mov:
                                bool_ = True
                                m.update(deepcopy(track))
                                break
                        if not bool_:
                            print("ERROR, not mov associated")
                    if self.lines:
                        track.reset_count(self.info_video, self.frame_id)
                    if remove:
                        removed_tracked_counting.append(track)
        if remove:
            return removed_tracked_counting

    def associate_entry(self, t):

        self.log.print(t, "ASSOCIATION")

        old_size = len(t.tracking_tlbr)
        tracks = []
        tracks.extend(self.removed_stracks)
        tracks.extend(self.lost_stracks)
        tracks = [t_c for t_c in tracks if
                  not t_c.count and len(t_c.info_tracking) > 2]
        t.associated = True
        new_candidates = True
        self.log.print(
            "Step 1 : Check if there are lost tracks to append to the track for a better prediction of an entry")
        # print(tracks)

        while new_candidates:
            candidates = []
            remove_noise_stracks = []
            for c in tracks:

                dist_candidate_track = calculate_distance(c.tlbr_to_center_point(c.tracking_tlbr[-1]),
                                                          c.tlbr_to_center_point(c.tracking_tlbr[0]))
                # print(c, dist_candidate_track)
                if dist_candidate_track > MIN_DIST_CANDIDATE:
                    if not overlap_time(t, c) and is_in_reasonable_distance(t, c) and has_category_in_common(
                        t, c):
                        candidates.append(c)
                else:
                    remove_noise_stracks.append(c)

            tracks = self.sub_stracks(tracks, remove_noise_stracks)
            self.removed_stracks = self.sub_stracks(self.removed_stracks, remove_noise_stracks)

            # Choose best candidate
            self.log.print("CANDIDATES", candidates)
            if candidates:
                c_best = closest_candidate_distance(t, candidates)
                self.log.print(t, "THE MOST PROBABLE Lost Track is", c_best, "VEL", c_best.vel)

                associate_track(t, c_best)
                self.log.print("REMOVING", c_best)
                self.lost_stracks = self.sub_stracks(self.lost_stracks, [c_best])
                self.removed_stracks = self.sub_stracks(self.removed_stracks, [c_best])
                tracks = self.sub_stracks(tracks, [c_best])

            else:
                new_candidates = False

        self.log.print("Step 2 : After having the full track, associate the entry")
        # if len(t.tracking_tlbr) < MIN_TRACK_EXIT:
        # return False, False
        # t_end_dir = calculate_direction_track(t, length=5, newer=True)
        # t_end_dir = calculate_direction_track(t, length=5, newer=True)
        vehicle_dist = sum(dist for _, dist, _, _ in t.info_tracking)
        # vehicle_dist = calculate_distance(t.tlbr_to_center_point(t.tracking_tlbr[0]),
        # t.tlbr_to_center_point(t.tracking_tlbr[-1]))
        # print(vehicle_dist)
        if vehicle_dist < MIN_DIST_MOV:
            t.associated = False
            return False
        valid_entries = []

        self.log.print("OLD_Size", old_size, "NEWSIZE", len(t.tracking_tlbr))
        for l in self.lines:
            # print(polyline.intersects(l.coords))
            # TRACKS WERE ADDED

            if old_size != len(t.tracking_tlbr):
                tracking_line = list(
                    map(t.tlbr_to_bottom_center_point,
                        t.tracking_tlbr[:-1]))  # -1 : remove last tlbr to not intersect with exit line
                polyline = LineString(tracking_line)
                if polyline.intersects(l.coords):
                    point_intersection = l.point_intersection(polyline)
                    i = find_tracking_tlbr_index(t, point_intersection)

                    # tlbr = STrack.bottom_center_point_to_tlbr(point_intersection, t.area)
                    # t.tracking_tlbr.insert(i, tlbr)

                    t.entry = [l, i]

                    print("Association track with an entry that intersects", t.entry[0].name("Entry"))
                    return True
                else:
                    if l == t.exit[0]:
                        continue
            else:

                # NO TRACKS WERE ADDED
                self.log.print("Does not Intersects ", l.name("Entry"))
                self.log.print(l.name("Entry") + t.exit[0].name("Exit"))
                valid_entries.append(l)

        # check distance
        if self.args.draw_option == "regions":
            if valid_entries:
                t.entry = closest_region_distance(t, valid_entries, mode="Entry")
            else:
                t.entry = closest_region_distance(t, self.lines, mode="Entry")
        else:
            if valid_entries:
                t.entry = closest_line_distance(t, valid_entries, mode="Entry")
            else:
                t.entry = closest_line_distance(t, self.lines, mode="Entry")
        print("Association the track with the closest entry ", t.entry[0].name("Entry"))

        self.log.print(t, "THE MOST PROBABLE Entry is", t.entry)
        # print(t.dist_dir)

        """else:
            self.log.print(t, "THERE'S NO PROBABLE ENTRY!!!!!!!!!!")
            exit(1)"""
        return True

    def add_lost_to_removed(self, lost_stracks):
        # print(removed_stracks)
        # print(self.lost_stracks)
        # print("ADD TO REMOVED")
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                if track.state != TrackState.Absent_Removed:
                    if track.state == TrackState.Absent or track.state == TrackState.Occluded:
                        # print("MARK_UNKNOWN REMOVED")
                        track.mark_absent_removed()

                if track.state != TrackState.Removed and not track.state == TrackState.Absent_Removed:
                    track.mark_removed()
                    self.log.print(track, "LOST AND REMOVED")
                    # print("2")
                    # print("ADD here to REMOVED")
                lost_stracks.append(track)
        return lost_stracks

    def remove_removed(self, removed_stracks, max_time_keep=MAX_FRAMES_REMOVED):
        if max_time_keep == 0:
            print("LAST TIME COUNT IF THERE ARE REMOVED TRACKS WITH ENTRY")
        if self.lines and not self.is_preprocessing:
            removed_tracks_aux = []
            for track in removed_stracks:
                if track.state == TrackState.Absent_Removed or max_time_keep == 0:
                    possible_loosen_tracks = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
                    possible_loosen_tracks = self.joint_stracks(possible_loosen_tracks, [track])
                    end_frame = find_end_frame(possible_loosen_tracks)

                    print(track.track_id, "END_FRAME", end_frame)

                    # print("REMOVING?", self.frame_id, end_frame, self.max_time_lost)

                    if self.frame_id - end_frame > self.max_time_lost or max_time_keep == 0:
                        # TODO: Know if track is counted in the x minutes before
                        print(track, track.entry)
                        dist = calculate_distance(track.tlbr_to_center_point(track.tracking_tlbr[track.entry[1]]),
                                                  track.tlbr_to_center_point(track.tracking_tlbr[-1]))
                        if dist > MIN_DIST_MOV or max_time_keep == 0:
                            if self.args.draw_option == "regions":
                                track.exit = closest_region_distance(track, self.lines, mode="Exit")
                            else:
                                track.exit = closest_line_distance(track, self.lines, mode="Exit")

                            track.associated = True
                            track.count = True
                            print(track, "EXIT ASSOCIATED TO ABSENT REMOVED: ", track.associated,
                                  track.entry[0].name("Exit"))
                        else:
                            print(track, "Fake track")
                            removed_tracks_aux.append(track)

                active_tracks = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
                last_tracked_frame = get_last_tracked_frame(track, active_tracks)
                if last_tracked_frame is None:
                    last_tracked_frame = self.frame_id
                # print(last_tracked_frame, last_tracked_frame - track.end_frame, max_time_keep)
                if last_tracked_frame - track.end_frame > max_time_keep:
                    removed_tracks_aux.append(track)

            removed_tracks_counting = self.count_movement(removed_stracks, remove=True)
            # print("self", self.removed_stracks, "remove", removed_tracks_aux)
            removed_stracks = self.sub_stracks(removed_stracks, removed_tracks_aux)
            # print("self", self.removed_stracks)
            removed_stracks = self.sub_stracks(removed_stracks, removed_tracks_counting)
            # print(removed_tracks_aux,self.removed_stracks)
        return removed_stracks

    def check_absent(self, lost_stracks, empty_frame=False):
        if self.lines:
            lost_stracks = [track for track in lost_stracks if not track.count]
            for track in lost_stracks:
                # print(track)
                if empty_frame:
                    track.mark_lost()
                if track.entry and not track.exit:
                    if not track.state == TrackState.Absent:
                        track.mark_absent()

    def check_places(self):
        def remove_false_places():
            remove = []
            for p in self.marked_places:
                if self.frame_id - p.end_frame > 100 and p.length < 10 or self.frame_id - p.end_frame > 400 and p.length < 20:
                    remove.append(p)
            self.marked_places = self.sub_splaces(self.marked_places, remove)

        # self.log.print("CHEKCING PLACES")
        if self.lines:
            remove = []
            irrelevant_removed_stracks = [track for track in self.removed_stracks if
                                          track.state != TrackState.Absent_Removed]
            for t in irrelevant_removed_stracks:
                if len(t.tracking_tlbr) == 1 or calculate_distance(t.tlbr_to_center_point(t.tracking_tlbr[-1]),
                                                                   t.tlbr_to_center_point(
                                                                       t.tracking_tlbr[0])) < MAX_DIST_FALSE_PLACE:
                    t.mark_removed()  # want to remove
                    self.log.print(t)
                    is_old = False
                    for p in self.marked_places:
                        if calculate_distance(p.avg_point, t.center_point) < MAX_DIST_FALSE_PLACE:
                            self.log.print("OLD")
                            self.log.print(t.center_point)
                            is_old = True
                            p.update(t, self.frame_id, length=len(t.tracking_tlbr))

                    if not is_old:
                        self.log.print("NEW")
                        self.marked_places.append(BadPlace(t, self.frame_id, length=len(t.tracking_tlbr)))

                    remove.append(t)
            remove_t = []

            for t in [t for t in self.tracked_stracks if not t.entry]:
                if len(t.tracking_tlbr) > 3600:  # 3 min of being constantly identifying the same stopped vehicle
                    print(t, "TRacked for too long")
                    t.mark_removed()
                    self.log.print(t)
                    is_old = False
                    for p in self.marked_places:
                        if calculate_distance(p.avg_point, get_avg_point(
                                list(map(t.tlbr_to_center_point, t.tracking_tlbr)))) < MAX_DIST_FALSE_PLACE:
                            self.log.print("OLD")
                            self.log.print(t.center_point)
                            is_old = True
                            p.update(t, self.frame_id, length=len(t.tracking_tlbr))

                    if not is_old:
                        self.log.print("NEW")
                        self.marked_places.append(BadPlace(t, self.frame_id, length=len(t.tracking_tlbr)))

                    remove_t.append(t)
                    print(self.marked_places)

            self.removed_stracks = self.sub_stracks(self.removed_stracks, remove)
            self.tracked_stracks = self.sub_stracks(self.tracked_stracks, remove_t)
            remove_false_places()

    """"def clean_tracker(self):
        # self.log.print("CLEANING")
        removed_stracks = []
        x_min, y_min, x_max, y_max = self.ROI_coordinates
        rect = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        polygon_ROI_bounds = Polygon(rect)

        for track in self.tracked_stracks:

            if not polygon_ROI_bounds.contains(Point(track.center_point)):
                # self.log.print(track.track_id, track.center_point)
                # print("ADD here to REMOVED")
                removed_stracks.append(track)
        self.tracked_stracks = self.sub_stracks(self.tracked_stracks, removed_stracks)
        # print(self.removed_stracks)"""

    def is_point_fake(self, det):
        is_fake = []
        for p in self.marked_places:
            if calculate_distance(p.avg_point, det.center_point) < MAX_DIST_FALSE_PLACE:
                is_fake.append(p)
        if is_fake:
            self.log.print(det.center_point, "IS_FAKE")
            for p in is_fake:
                p.update(det, self.frame_id)
            return True
        return False

    def is_track_from_absent(self, two_det_track, lost_absent_stracks):
        lost_candidates = []
        self.log.print("Absent", lost_absent_stracks)

        for lost_t in lost_absent_stracks:
            if two_det_track.category != lost_t.category:
                continue
            self.log.print("TRYING TO FIND", lost_t, "for center point", two_det_track.center_point,
                           "N_FRAMES_LOST:", self.frame_id - lost_t.end_frame)

            if is_in_reasonable_distance(lost_t, two_det_track,
                                         actual_frame=self.frame_id, ) and self.frame_id != lost_t.end_frame:
                lost_candidates.append(lost_t)
        if lost_candidates:
            return True, lost_candidates
        else:
            return False, []

    def is_point_from_absent(self, det, lost_absent_stracks):
        lost_candidates = []
        self.log.print("Absent", lost_absent_stracks)
        for lost_t in lost_absent_stracks:
            self.log.print("TRYING TO FIND", lost_t, "is_count", lost_t.count, "for center point", det.center_point,
                           "N_FRAMES_LOST:", self.frame_id - lost_t.end_frame)

            if is_in_reasonable_distance(lost_t, det, actual_frame=self.frame_id):  # self.frame_id != lost_t.end_frame
                lost_candidates.append(lost_t)
        if lost_candidates:
            return True, lost_candidates
        else:
            return False, []

    def is_point_from_occluded(self, det, lost_occluded_stracks):
        lost_candidates = []
        self.log.print("OCCLUDED", lost_occluded_stracks)
        for occluded_t in lost_occluded_stracks:
            if is_in_reasonable_distance(occluded_t, det, actual_frame=self.frame_id):
                lost_candidates.append(occluded_t)
        if lost_candidates:
            return True, lost_candidates
        else:
            return False, []

    def is_point_overlapping(self, det, tracks, percentage=0):
        overlaps = []
        for t in tracks:
            bool_ = overlap(t, det, percentage=percentage)
            if bool_:
                self.log.print(det.center_point, "OVERLAPS", t)
            overlaps.append(bool_)
        # print(inside_l)
        if any(overlaps):
            self.log.print(det.center_point, "OVERLAPS")
            return True
        return False

    def is_point_from_ignoring_region(self, det):
        ignoring_regions = self.args.ignoring_regions[self.args.filename_id]
        return any(region.overlaps(det) for region in ignoring_regions)

    # added

    def reset_counter(self):
        self.counter = deepcopy(self.counter_init)

    def reset_movements(self):
        for mov in self.movements:
            mov.tracks = []

    def update_ROI_coordinates(self, bboxes):
        if self.lines:
            bboxes[:, 0] = bboxes[:, 0] + self.ROI_coordinates[0]
            bboxes[:, 1] = bboxes[:, 1] + self.ROI_coordinates[1]
            bboxes[:, 2] = bboxes[:, 2] + self.ROI_coordinates[0]
            bboxes[:, 3] = bboxes[:, 3] + self.ROI_coordinates[1]

        return bboxes

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IOU and fuses scores."""
        dists = matching.iou_distance(tracks, detections)
        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """Returns the predicted tracks using the YOLOv8 network."""
        STrack.multi_predict(tracks)

    def reset_id(self):
        """Resets the ID counter of STrack."""
        STrack.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combine two lists of stracks into a single one."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_splaces(tlista, tlistb):
        stracks = {}
        for t in tlista:
            stracks[t.place_id] = t
        for t in tlistb:
            tid = t.place_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
        """
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Remove duplicate stracks with non-maximum IOU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

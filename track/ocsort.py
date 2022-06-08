"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import cv2
import numpy as np
from .association import *
import time
import torch


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, cate, delta_t=3, orig=False, max_age=70, overlap_threshold=0.6, n_init=3):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
            0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.last_prediction = np.array([-1, -1, -1, -1])
        self.hits = 0
        self.n_init = n_init
        self.hit_streak = 0
        self.age = 0
        self.max_age = max_age
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.history_frame_num = []
        self.velocity = None
        self.delta_t = delta_t
        self.cate = cate

        self.overlap_threshold = overlap_threshold
        self.move_streak = 0
        self.static_box = None
        self.current_timestamp = 0
        self.first_timestamp = 0
        self.alarm = False
        self.state = TrackState.Tentative

    def update(self, bbox, timestamp, frame_num=-1):
        """
        Updates the state vector with observed bbox.
        """
        if frame_num > 0:
            self.history_frame_num.append(frame_num)

        # 框被遮挡时不应该更新时间
        self.current_timestamp = timestamp
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

                """
                    Determine if the target is stationary
                """
                # 运动状态估计
                self.state_estimation(bbox)
                if self.move_streak > 3:
                    self.first_timestamp = timestamp
                    self.static_box = bbox
            else:
                self.static_box = bbox
                self.first_timestamp = timestamp
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            if self.state == TrackState.Tentative and self.hits >= self.n_init:
                self.state = TrackState.Confirmed
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        self.last_prediction = self.history[-1][0]
        if np.any(np.isnan(self.history[-1])):
            return True
        else:
            return False

    def mark_missed(self):
        """
            Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age or \
                (self.time_since_update > int(self.max_age * 0.2) and self.move_streak > 3):
            self.state = TrackState.Deleted

        if self.is_deleted():
            self._save_record()

    def is_tentative(self):
        """
            Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """
            Returns True if this track is confirmed.
        """
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """
            Returns True if this track is dead and should be deleted.
        """
        return self.state == TrackState.Deleted

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def state_estimation(self, detection):
        iou = ima_batch(self.static_box[np.newaxis, :4], detection[np.newaxis, :4])[0][0]
        if iou < self.overlap_threshold:
            self.move_streak += 1
        else:
            self.move_streak = 0

    def _save_record(self, folder="data/delete"):
        with open(os.path.join(folder, str(self.id) + ".txt"), "w") as f:
            for frame_num in self.history_frame_num:
                f.write("{} ".format(frame_num))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {"iou": iou_batch,
              "giou": giou_batch,
              "ciou": ciou_batch,
              "diou": diou_batch,
              "ct_dist": ct_dist,
              "ima": ima_batch}


class OCSort(object):
    def __init__(self, areas, max_age=150, min_hits=3,
                 iou_threshold=0.7, delta_t=3,
                 inertia=0.2, alarm_threshold=120):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.delta_t = delta_t
        self.inertia = inertia
        self.alarm_threshold = alarm_threshold
        self.areas = areas
        KalmanBoxTracker.count = 1

    def update(self, detections, timestamp, frame_num=-1):
        detections = self.box_filter(detections)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.predict():  # 框预测为Nan，删除
                self.trackers.pop(i)

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.trackers[track_idx].update(detections[detection_idx][:5], timestamp, frame_num)

        for track_idx in unmatched_tracks:
            self.trackers[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self.trackers.append(
                KalmanBoxTracker(detections[detection_idx][:5], int(detections[detection_idx][5]), max_age=self.max_age))

        self.trackers = [t for t in self.trackers if not t.is_deleted()]

        outputs = []
        for track in self.trackers:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            if track.current_timestamp - track.first_timestamp > self.alarm_threshold and track.hits > self.alarm_threshold:
                track.alarm += 1

            outputs.append((track.last_prediction, track.last_observation[:4], track.id, track.alarm,
                            track.current_timestamp - track.first_timestamp))

        return outputs

    def _match(self, detections):
        # split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.trackers) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.trackers) if not t.is_confirmed()]

        # 先匹配静止的目标
        unmatched_detections = list(range(len(detections)))
        track_indices = [k for k in confirmed_tracks if self.trackers[k].move_streak == 0]
        matches_static, unmatched_tracks, unmatched_detections = associate(self.trackers,
                                                                           detections,
                                                                           track_indices,
                                                                           unmatched_detections,
                                                                           associate_func=iou_batch,
                                                                           iou_threshold=self.iou_threshold,
                                                                           vdc_weight=self.inertia,
                                                                           delta_t=self.delta_t)

        # 然后匹配运动的目标
        track_indices = unmatched_tracks + [k for k in confirmed_tracks if self.trackers[k].move_streak != 0]
        matches_move, unmatched_tracks, unmatched_detections = associate(self.trackers,
                                                                         detections,
                                                                         track_indices,
                                                                         unmatched_detections,
                                                                         associate_func=iou_batch,
                                                                         iou_threshold=self.iou_threshold,
                                                                         vdc_weight=self.inertia,
                                                                         delta_t=self.delta_t)

        # 再匹配未确定态的目标
        track_indices = unconfirmed_tracks + unmatched_tracks
        matches_unconfirmed, unmatched_tracks, unmatched_detections = associate(self.trackers,
                                                                                detections,
                                                                                track_indices,
                                                                                unmatched_detections,
                                                                                associate_func=iou_batch,
                                                                                iou_threshold=self.iou_threshold - 0.2,
                                                                                vdc_weight=self.inertia,
                                                                                delta_t=self.delta_t)
        # 剩下的匹配
        matches_rest, unmatched_tracks, unmatched_detections = associate_iou(self.trackers,
                                                                             detections,
                                                                             unmatched_tracks,
                                                                             unmatched_detections,
                                                                             associate_func=ima_batch,
                                                                             iou_threshold=self.iou_threshold - 0.3)

        matches = matches_static + matches_move + matches_unconfirmed + matches_rest
        return matches, unmatched_tracks, unmatched_detections

    def box_filter(self, detections):
        #  非机动车转换为人骑车
        detections[detections[:, 5] == 3, 5] = 1

        remind_classes = [1, 2, 3]
        class_threshold = {0: 0.5, 1: 0.5, 2: 0.6, 3: 0.5}

        # 判断目标框和区域的关系
        remind = [False] * detections.shape[0]
        for i in range(detections.shape[0]):
            if int(detections[i][5]) == 2:
                point = ((detections[i][0] + detections[i][2]) / 2, (detections[i][1] + detections[i][3]) / 2)  # 目标中心点
            else:
                point = ((detections[i][0] + detections[i][2]) / 2, detections[i][3])  # 底部中心点
            for counter in self.areas:
                distance = cv2.pointPolygonTest(np.array(counter), point, False)
                if distance != -1 and int(detections[i][5]) in remind_classes and \
                        detections[i][4] > class_threshold[int(detections[i][5])]:
                    remind[i] = True
                    break
        return detections[remind]

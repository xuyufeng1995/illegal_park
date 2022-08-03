import numpy as np
import torch
import cv2
import os
from loguru import logger
from trackers.deepsort_tracker import kalman_filter, linear_assignment, iou_matching
from .reid_model import Extractor
from .detection import Detection
from .track import Track


def save_image(detections, frame):
    image = frame.copy()
    for detection in detections:
        cv2.rectangle(image, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (0, 204, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite("data/test.jpg", image)


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class Tracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, areas=[], occlusion_threshold=30, fps=5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.areas = areas
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.areas = areas
        self.occlusion_threshold = occlusion_threshold
        self.fps = fps

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections, classes, timestamp, frame_num=-1):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        if frame_num == 77:
            print(frame_num)
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].history_frame_number.append(frame_num)
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            if self.region_judge(detections[detection_idx].to_xyah()):
                if self.tracks[track_idx].last_timestamp is None or self.tracks[track_idx].out_region > \
                        self.occlusion_threshold * self.fps:
                    self.tracks[track_idx].last_timestamp = timestamp
                self.tracks[track_idx].out_region = 0
            else:
                self.tracks[track_idx].out_region += 1
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], classes[detection_idx].item())
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        iou_track_candidates = unmatched_tracks_b + [k for k in unmatched_tracks_a if
                                                     1 < self.tracks[k].time_since_update <= 10]  # 2秒之外的只能外观匹配
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update > 10]  # 2秒之外的只能外观匹配

        matches_c, unmatched_tracks_c, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iom_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_c))
        matches = matches_a + matches_b + matches_c
        # 对于匹配上的目标需要判断他们之间的外观距离
        delete_matches = []
        for track_idx, detect_idx in matches:
            if self.tracks[track_idx].is_confirmed():
                features = np.array([detections[detect_idx].feature])
                targets = np.array([self.tracks[track_idx].track_id])
                cost_matrix = self.metric.distance(features, targets)
                if cost_matrix[0][0] > 0.2:
                    logger.info("(track_idx, detect_idx): ({},{}),  track id: {}, cost matrix {}", track_idx, detect_idx, self.tracks[track_idx].track_id, cost_matrix[0][0])
                    delete_matches.append((track_idx, detect_idx))

        for track_idx, detect_idx in delete_matches:
            matches.remove((track_idx, detect_idx))
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detect_idx)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def region_judge(self, detection):
        point = (detection[0], detection[1])  # 目标中心点
        for counter in self.areas:
            distance = cv2.pointPolygonTest(np.array(counter), point, False)
            if distance != -1:
                return True

        return False


class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class DeepSort(object):
    def __init__(self, max_dist=0.2, min_confidence=0.4, nms_max_overlap=1.0,
                 max_iou_distance=0.5, max_age=70, n_init=6, nn_budget=100, use_cuda=True,
                 areas=[], alarm_threshold=120, occlusion_threshold=30):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor('trackers/deepsort_tracker/ckpt.t7', use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.areas = areas
        self.alarm_threshold = alarm_threshold
        self.occlusion_threshold = occlusion_threshold

        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age,
                               n_init=n_init, occlusion_threshold=occlusion_threshold, areas=areas)
        self.alarmed_detections = []

    def update(self, detections, ori_img, timestamp, frame_num):
        output_results = detections[detections[:, 5] == 2]

        save_image(output_results, ori_img)

        self.height, self.width = ori_img.shape[:2]
        # post process detections
        confidences = output_results[:, 4]
        bbox_xyxy = output_results[:, :4]  # x1y1x2y2
        bbox_tlwh = self._xyxy_to_tlwh_array(bbox_xyxy)
        remain_inds = confidences > self.min_confidence
        bbox_tlwh = bbox_tlwh[remain_inds]
        confidences = confidences[remain_inds]

        # generate detections
        features = self._get_features(bbox_tlwh, ori_img)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]
        classes = output_results[:, 5]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, timestamp, frame_num)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1 or track.out_region > 0:
                continue
            pred_box = self._tlwh_to_xyxy_noclip(track.to_tlwh())
            det_box = self._tlwh_to_xyxy_noclip(track.det_box)
            track_id = track.track_id
            if timestamp - track.last_timestamp > self.alarm_threshold:
                track.alarm_num += 1
                if track.alarm_num == 1:
                    logger.info("now frame number: {}, track id: {}", frame_num, track_id)
                    if self.suppress_duplicate_alarms(track.to_tlwh(), timestamp, track.det_feature):
                        logger.info("track id {} has already alarm!!!!!!!!!!!!!!!!!!!!!!!!!!!", track_id)
                        track.alarm_num += 1   # 已经告警过
                    else:
                        self.alarmed_detections.append(Detection(track.to_tlwh(), timestamp, track.det_feature))
                        logger.info("alarmed detection length is {}", len(self.alarmed_detections))
                        if len(self.alarmed_detections) > 100:
                            self.alarmed_detections.pop(0)

            outputs.append([pred_box, track_id, track.alarm_num, timestamp - track.last_timestamp])

        regions = []
        for area in self.areas:
            regions.append({"region": area, "park": len(outputs)})

        return regions, outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh
    
    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy_noclip(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return [x1, y1, x2, y2]

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def suppress_duplicate_alarms(self, box, timestamp, feature):
        for detect in self.alarmed_detections:
            if timestamp != detect.confidence:
                iou_cost = iou_matching.iou(box, np.expand_dims(detect.tlwh, axis=0))[0]
                feat_cost = _nn_cosine_distance(np.expand_dims(feature, axis=0), np.expand_dims(detect.feature, axis=0))[0]
                logger.info("iou_cost: {}, feat_cost: {}", iou_cost, feat_cost)
                if iou_cost > 0.6 and feat_cost < 0.2:
                    logger.info("duplicate alarm################")
                    return True
        logger.info("______________________not duplicate alarm_________________")
        return False

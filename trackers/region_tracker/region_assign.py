import cv2
import math
import numpy as np
from loguru import logger


class Region(object):
    def __init__(self, area, fps=6, alarm_threshold=120, alarm_prop=0.9):
        self.area = area
        self.alarm_threshold = alarm_threshold
        self.fps = fps
        # 车辆、非机动车状态列表
        self.vehicle_state_list = list()
        self.motor_state_list = list()

        self.alarm_prop = alarm_prop
        self.empty_area = 0

    def update(self, vehicle, non_vehicle, amount, timestamp):
        if amount == 0:
            self.empty_area += 1
        else:
            self.empty_area = 0
        # 状态更新
        self.vehicle_state_list.append(timestamp if vehicle > 0 else -timestamp)
        self.motor_state_list.append(timestamp if non_vehicle > 0 else -timestamp)
        logger.info("vehicle state length: {}, non_vehicle state length: {}", len(self.vehicle_state_list), len(self.motor_state_list))

        # 状态修正
        self.revise_state()

        return self.judge(self.vehicle_state_list), self.judge(self.motor_state_list)

    def point_polygon(self, detection):
        point = ((detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2)  # 目标中心点
        distance = cv2.pointPolygonTest(np.array(self.area), point, False)
        if distance != -1:
            return True
        else:
            return False

    def judge(self, state_list, length=5):
        # 当前状态列表为空
        if sum(np.array(state_list) > 0) == 0 and len(state_list) < self.alarm_threshold:
            return False
        logger.info("threshold: {}, start: [{}], end: [{}]", math.fabs(state_list[-1]) - math.fabs(state_list[0]), state_list[0] > 0, state_list[-1] > 0)
        if math.fabs(state_list[-1]) - math.fabs(state_list[0]) > self.alarm_threshold and \
                sum(np.array(state_list) > 0) / len(state_list) > self.alarm_prop and \
                sum(np.array(state_list[:length]) > 0) >= length and \
                sum(np.array(state_list[-length:]) > 0) >= length:
            logger.info("time: {}, will alarm!!!!!!!!!!!!!!!", math.fabs(state_list[-1]) - math.fabs(state_list[0]))
            return True
        else:
            return False

    def revise_state(self):
        min_length = int(self.alarm_threshold * self.fps * (1 - self.alarm_prop + 0.05) + 1e-3)
        if self.empty_area >= self.fps * 5:  # 5秒内无任何目标
            min_length = min(min_length, self.fps * 5)

        # 机动车
        if len(self.vehicle_state_list) > min_length:
            if sum(np.array(self.vehicle_state_list[-min_length:]) > 0) == 0:
                self.vehicle_state_list = list()
            else:
                start, end = 0, len(self.vehicle_state_list) - 1
                if math.fabs(self.vehicle_state_list[end]) - math.fabs(self.vehicle_state_list[start]) > self.alarm_threshold:
                    while end > start and math.fabs(self.vehicle_state_list[end]) - math.fabs(self.vehicle_state_list[start + 1]) > self.alarm_threshold:
                        start += 1
                        logger.info("vehicle move one square，time: {}",
                                    math.fabs(self.vehicle_state_list[end]) - math.fabs(self.vehicle_state_list[start + 1]))
                for i in range(start):
                    logger.info("vehicle will pop {} position, length {}", i, len(self.vehicle_state_list))
                    self.vehicle_state_list.pop(0)

        # 非机动车
        if len(self.motor_state_list) > min_length:
            if sum(np.array(self.motor_state_list[-min_length:]) > 0) == 0:
                self.motor_state_list = list()
            else:
                start, end = 0, len(self.motor_state_list) - 1
                if math.fabs(self.motor_state_list[end]) - math.fabs(
                        self.motor_state_list[start]) > self.alarm_threshold:
                    while end > start and math.fabs(self.motor_state_list[end]) - math.fabs(
                            self.motor_state_list[start + 1]) > self.alarm_threshold:
                        start += 1
                        logger.info("motor move one square，time: {}", math.fabs(self.motor_state_list[end]) - math.fabs(self.motor_state_list[start + 1]))
                for i in range(start):
                    logger.info("motor will pop {} position, length {}", i, len(self.motor_state_list))
                    self.motor_state_list.pop(0)


class Assign(object):
    def __init__(self, areas, target_type=(1, 2), alarm_interval=30,
                 alarm_threshold=60, fps=5):
        # 1 非机动车和非机动车， 2 机动车
        self.target_type = target_type
        self.alarm_interval = alarm_interval
        self.alarm_threshold = alarm_threshold
        self.fps = fps
        self.last_alarm = 0
        self.regions = self.new_region(areas)

    def update(self, detections, timestamp, frame_num):
        logger.info("frame_num {}", frame_num)
        reset = 0
        inner_detections = list()
        outputs = []
        #  非机动车转换为人骑车
        if len(detections):
            detections[detections[:, 5] == 3, 5] = 1
        alarm = False
        for region in self.regions:
            result = dict()
            result["region"] = region.area
            vehicle, non_vehicle, inner_detections_amount = 0, 0, len(inner_detections)
            for det in detections:
                if region.point_polygon(det):
                    inner_detections.append(det)
                    if int(det[5]) == 1 and int(det[5]) in self.target_type:
                        non_vehicle += 1
                    elif int(det[5]) == 2 and int(det[5]) in self.target_type:
                        vehicle += 1
            vehicle_state, motor_state = region.update(vehicle,
                                                       non_vehicle,
                                                       len(inner_detections) - inner_detections_amount,
                                                       timestamp)
            result["vehicle"] = sum(np.array(region.vehicle_state_list) > 0)
            result["non_vehicle"] = sum(np.array(region.motor_state_list) > 0)
            result["alarm"] = False
            reset += result["vehicle"] + result["non_vehicle"]
            logger.info("vehicle positive sample {}， non_vechile positive sample {}", result["vehicle"], result["non_vehicle"])
            if (vehicle_state or motor_state) and timestamp - self.last_alarm > self.alarm_interval:
                alarm = True
                result["alarm"] = True
            outputs.append(result)

        if alarm:
            self.last_alarm = timestamp

        # 上次告警时间重置
        if reset == 0:
            logger.info("No target detected for a while, will reset!!!")
            self.last_alarm = 0

        return outputs, alarm, inner_detections

    def new_region(self, areas):
        regions = []
        for area in areas:
            regions.append(Region(area, alarm_threshold=self.alarm_threshold, fps=self.fps))
        return regions

import torch


class Region(object):
    count = 1

    def __init__(self):
        self.id = Region.count
        Region.count += 1
        self.hits = 1
        self.age = 1

    def update(self, box, timestamp, frame_num):
        pass

    def update_vehicle(self, box):
        pass

    def update_non_vehicle(self, box):
        pass


class Assign(object):
    def __init__(self, areas, target_type=(0, 1), alarm_interval=30, alarm_threshold=60, conf_threshold=0.5):
        # 0 非机动车， 1 机动车
        self.target_type = target_type
        self.alarm_interval = alarm_interval
        self.alarm_threshold = alarm_threshold
        self.conf_threshold = conf_threshold
        self.last_alarm = 0
        self.areas = areas

    def update(self, detections, timestamp, frame_num):
        for det in detections:
            if det[4] > self.conf_threshold and int(det[5]) != 0:
                if int(det[5]) == 2:  # vehicle
                    for region in self.areas:
                        region.update(detections, timestamp, frame_num)

import yaml
import os
import cv2
import time
import uuid
import shutil
from trackers.deepsort_tracker.deepsort import DeepSort
from trackers.region_tracker.region_assign import Assign
from detector.onnx_detect import YOLOV5_ONNX
from loguru import logger


def get_config(config_file, camera_id):
    config = {}
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    for device in cfg["deviceList"]:
        if device["deviceId"] == camera_id:
            config = device
            break

    return config


def park_process(param):
    # 解析参数
    image_queue = param["pictureQueue"]
    result_queue = param["resultQueue"]
    algo_cfg = get_config(param["configPath"], param["cameraId"])
    build_type = param["buildType"]

    model = YOLOV5_ONNX(onnx_path="detector/best_yolov7.onnx")
    non_vehicle_track = Assign(target_type=[1], areas=algo_cfg["areas"],
                               alarm_interval=algo_cfg['config']["alarm_interval"],
                               alarm_threshold=algo_cfg['config']["alarm_threshold"])
    vehicle_track = DeepSort(areas=algo_cfg["areas"],
                             alarm_threshold=algo_cfg['config']["alarm_threshold"],
                             occlusion_threshold=algo_cfg['config']["occlusion_threshold"])

    frame_num = 0
    regions = []
    consume_time = 0
    start_timestamp = 0
    while True:
        frame = image_queue.get()
        if frame is None:
            result_queue.put(None)
            break
        frame, timestamp = frame
        start_time = time.time()
        predict = model.infer(frame)[0]
        end_time = time.time()
        consume_time += end_time - start_time
        if frame_num % 5 == 0:
            logger.info("detect past 5 frame, model time: {} s, total time: {}s", consume_time, timestamp - start_timestamp)
            consume_time = 0
            start_timestamp = timestamp
        alarm_flag = False
        alarm_id = ""
        if algo_cfg["config"]["classes"] == "non_vehicle":
            regions, alarm_flag, inner_detections = non_vehicle_track.update(predict.numpy(), timestamp, frame_num)
            # 绘制目标框
            for bbox in inner_detections:
                if bbox[5] == 1 or bbox[5] == 3:
                    model.plot_one_box(list(map(int, bbox)),
                                       frame,
                                       color=(0, 255, 0),
                                       line_thickness=1)

        elif algo_cfg["config"]["classes"] == "vehicle":
            regions, tracker = vehicle_track.update(predict.numpy(), frame, timestamp, frame_num)
            for pred_box, track_id, alarm_count, stay_time in tracker:
                if alarm_count == 1:
                    if pred_box[2] - pred_box[0] > 80 or pred_box[3] - pred_box[1] > 80:
                        logger.info("An alarm occurs， track id {}", track_id)
                        alarm_flag = True
                        alarm_id += "   " + str(track_id)
                        model.plot_one_box(list(map(int, pred_box)),
                                           frame,
                                           label="{}__{}".format(int(track_id), int(stay_time)),
                                           color=(0, 0, 204),
                                           line_thickness=2)
                    else:
                        logger.info("The target box area is too small， track id: {}, width: {}, height: {}",
                                    str(track_id), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1])
                else:
                    if build_type == "debug":
                        model.plot_one_box(list(map(int, pred_box)),
                                           frame,
                                           label="{}__{}".format(int(track_id), int(stay_time)),
                                           color=(0, 153, 255),
                                           line_thickness=2)

        # 绘制区域框
        for region in regions:
            color = (0, 0, 0)
            number = 0
            if "non_vehicle" in region and region["non_vehicle"] != 0:
                if region["alarm"]:
                    color = (0, 0, 255)
                else:
                    color = (255, 128, 255)
                number = region["non_vehicle"]

            if "park" in region and region["park"] > 0:
                color = (0, 0, 255)
                number = ""

            model.plot_one_region(region["region"], frame, color=color, label=str(number))

        # 保存视频调试
        content = "frame: " + str(frame_num) + alarm_id
        cv2.putText(frame, content, (100, 100), 0, 2, [0, 0, 255], thickness=2,
                    lineType=cv2.LINE_AA)

        if build_type == "debug":
            cv2.imwrite("data/frame/{}.jpg".format(frame_num), frame)

        if alarm_flag:
            image_folder = "data/alarm/" + param["cameraId"]
            os.makedirs(image_folder, exist_ok=True)
            image_name = image_folder + "/{}.{}.jpg".format(time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp)),
                                                            round(timestamp * 1000) % 1000)
            cv2.imwrite(image_name, frame)
            result = dict()
            result["msg_uuid"] = str(uuid.uuid4())
            result["taskid"] = param["taskId"]
            result["cameraid"] = param["cameraId"]
            result["algo_type"] = param["algorithmId"]
            result["time"] = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))
            result["image_url"] = "http://{}:{}/".format(param["host_ip"], param["host_port"]) + image_name
            result_queue.put(result)
            logger.info(image_name)

        frame_num += 1

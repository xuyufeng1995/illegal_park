import os
import numpy as np
import torch
import time
from detect.detector import YOLOV5_ONNX
from track.region_assign import Assign
from track.ocsort import OCSort
import cv2
import random
import queue
import threading
import uuid
import shutil
import torchvision


def vas(rtsp, image_queue):
    cap = cv2.VideoCapture(rtsp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    print(size)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    count, read = 0, 5
    while cap.isOpened():
        count += 1
        cap.grab()
        if count % read == 0:
            success, frame = cap.retrieve()
            if success:
                image_queue.put((frame, time.time()))
            else:
                break
        # time.sleep(1 / fps)

    image_queue.put(None)


def event(regions, image_queue):
    class_name = ["person", "rider", "vehicle", "non-vehicle"]
    model = YOLOV5_ONNX(onnx_path="detect/best.onnx")
    track = OCSort(areas=regions["areas"], alarm_threshold=regions['config']["parking_time_shreshold"])

    cv2.namedWindow("abc", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("abc", 1920, 1080)
    # writer = cv2.VideoWriter("data/output.mp4", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 25, (1920, 1080))

    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(50000)]
    frame_num = 0
    while True:
        frame = image_queue.get()
        frame_num += 1
        if frame_num == 152:
            cv2.imwrite("test.jpg", frame[0])
        if frame is None:
            break
        frame, timestamp = frame
        predict = model.infer(frame)[0]
        if predict is not None:
            tracker = track.update(predict.numpy(), timestamp, frame_num)

            to_save = False
            for pred_box, det_box, track_id, alarm_count, stay_time in tracker:
                if alarm_count == 1:
                    to_save = True
                    color = (0, 0, 255)
                elif alarm_count == 0:
                    color = (0, 0, 0)
                else:
                    color = (0, 255, 0)

                model.plot_one_box(list(map(int, pred_box)),
                                   frame,
                                   frame_id="frame: " + str(frame_num),
                                   regions=regions["areas"],
                                   label="{}__{}".format(int(track_id), int(stay_time)),
                                   color=color,
                                   line_thickness=1)

                model.plot_one_box([int(det_box[0]), int(det_box[1]), int(det_box[2]), int(det_box[3])],
                                   frame,
                                   color=(0, 255, 255),
                                   line_thickness=1)

            if to_save:
                os.makedirs("data/alarm", exist_ok=True)
                image_name = "data/alarm/{}_{}.jpg".format(
                    time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp)),
                    uuid.uuid4())
                cv2.imwrite(image_name, frame)
                print(image_name)
        # writer.write(frame)
        cv2.imshow("abc", frame)
        cv2.waitKey(1)

    # writer.release()
    cv2.destroyAllWindows()


def run(rtsp, regions):
    data_queue = queue.Queue(maxsize=100)
    vas_thread = threading.Thread(target=vas, args=(rtsp, data_queue), daemon=True)
    vas_thread.start()
    event(regions, data_queue)

    vas_thread.join()


if __name__ == "__main__":
    # region = {'deviceId': '694f9224-307c-44fd-99dc-ea706438ef4a',
    #           'areas': [[[666, 613], [32, 1237], [5, 1416], [602, 1434], [1101, 744]],
    #                     [[1541, 208], [1810, 408], [2165, 394], [1661, 144]],
    #                     [[1397, 682], [1216, 1362], [2453, 1426], [2341, 680]]], 'lines': [],
    #           'config': {'fps': 5, 'parking_time_shreshold': 30, 'alarm_interval': 300, 'occlusion_time_shreshold': 60}}
    shutil.rmtree("data/alarm", ignore_errors=True)
    region = {'deviceId': '694f9224-307c-44fd-99dc-ea706438ef4a',
              'areas': [[[0, 0], [1920, 0], [1920, 1080], [0, 1080]]], 'lines': [],
              'config': {'fps': 5, 'parking_time_shreshold': 120, 'alarm_interval': 300, 'occlusion_time_shreshold': 60}}
    # run("data/val_video/步行街长郡中学门口（三期）_2022-04-01 07：00：53_2022-04-01 14：27：37_FE20AC87_3.mp4", region)
    run("/home/xuyufeng/Projectes/datasets/人非车/金色卡通幼儿园25栋东南2（三期）_1654476750_7FB3224E/金色卡通幼儿园25栋东南2（三期）_1C6887F8_1654476750_1.mp4", region)

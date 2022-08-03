import time
import cv2
import json
import queue
import threading
import requests
import ctypes
from loguru import logger
from park import park_process


def decode(rtsp, buffer_queue):
    cap = cv2.VideoCapture(rtsp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    buffer_queue.put(fps)
    size = (
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    logger.info("size: {}, fps: {}", size, fps)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            buffer_queue.put(frame)
        else:
            break
    buffer_queue.put(None)


def vas(buffer_queue, image_queue):
    fps = buffer_queue.get()
    if fps == 0:
        image_queue.put(None)
        return
    count, read = 0, int(fps / 5 + 1e-3)
    interval = int(fps / read + 1e-3)
    logger.info("take frame interval {}, sleep interval {}", read, 1/interval)
    while True:
        frame = buffer_queue.get()
        if frame is None:
            break

        if count % read == 0:
            image_queue.put((frame, time.time()))
            time.sleep(1 / interval)

        count += 1

    image_queue.put(None)


class Worker(threading.Thread):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.state = True

    def run(self):
        buffer_queue = queue.Queue(maxsize=10)
        data_queue = queue.Queue(maxsize=50)
        result_queue = queue.Queue(maxsize=50)
        self.param["pictureQueue"] = data_queue
        self.param["resultQueue"] = result_queue
        self.param["buildType"] = "release"

        decode_thread = threading.Thread(target=decode, args=(self.param["cameraAddress"], buffer_queue))
        vas_thread = threading.Thread(target=vas, args=(buffer_queue, data_queue))
        event_thread = threading.Thread(target=park_process, args=(self.param,))
        decode_thread.start()
        vas_thread.start()
        event_thread.start()

        url = self.param["alarm_url"]
        headers = {'content-type': "application/json"}
        while self.state:
            try:
                alarm_data = result_queue.get(timeout=1)
            except queue.Empty:
                continue

            if alarm_data is None:
                break

            logger.info(alarm_data)
            res = requests.post(url, data=json.dumps(alarm_data), headers=headers)
            logger.info(res.content)

        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(event_thread.ident),
                                                   ctypes.py_object(SystemExit))
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(vas_thread.ident),
                                                   ctypes.py_object(SystemExit))
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(decode_thread.ident),
                                                   ctypes.py_object(SystemExit))

        logger.warning("*******************************************")


if __name__ == "__main__":
    task_param = dict()
    task_param["cameraId"] = "9a8dc1ae-f1ef-46a9-8315-a08245a38f5f"
    task_param["cameraAddress"] = "/home/xuyufeng/Projectes/datasets/jicheweiting/永安小学门口3（三期）_2022-04-25_16：40：00_2022-04-25_16：55：00_85B06747_1.mp4"
    task_param["configPath"] = "data/algo_42.yaml"
    task_param["taskId"] = "ac0c4149-f42a-41fb-9530-f7cab3fbb74b"
    task_param["algorithmId"] = "42"
    task_param["alarm_url"] = "http://5.5.3.229:7777/alarm/commit"
    task_param["host_ip"] = "5.5.5.238"
    task_param["host_port"] = 9810
    task_thread = Worker(task_param)
    task_thread.start()
    task_thread.join()

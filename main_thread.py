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
    # rtsp = rtsp.replace("/home/videos/feijiweiting/", "/home/xuyufeng/Projectes/datasets/非机动车违规停放/")
    rtsp = rtsp.replace("/home/videos/weiguitingche/", "/home/xuyufeng/Projectes/datasets/非机动车违规停放/")
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

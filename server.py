import os
import time

from sanic import Sanic, response
import requests
from main_thread import Worker
import yaml
import ctypes
from loguru import logger


app = Sanic(__name__)


def parser_region(url, algo_id):
    yaml_path = "data/algo_{}.yaml".format(algo_id)
    f = open(yaml_path, "w")
    res = requests.get(url)
    if res.status_code == 200:
        content = res.content
        content = yaml.load(content, Loader=yaml.FullLoader)
    else:
        content = {}
    yaml.dump(content, f)
    f.close()
    return yaml_path


@app.route('/rest/behaviorAnalysis/addTask', methods=["POST"])
async def add_task(request):
    param = request.json[0]
    task_id = param["taskId"]
    cameras = param["cameras"]
    for camera in cameras:
        task_param = dict()
        task_param["cameraId"] = camera["id"]
        task_param["cameraAddress"] = camera["address"]
        task_param["configPath"] = parser_region(param["configPath"], param["algorithmId"])
        task_param["taskId"] = param["taskId"]
        task_param["algorithmId"] = param["algorithmId"]
        task_param["alarm_url"] = alarm_url
        task_param["host_ip"] = host_ip
        task_param["host_port"] = host_port
        task_thread = Worker(task_param)
        task_thread.start()
        task[(task_id, camera["id"])] = task_thread
    return response.json({"ret": 0, "desc": 'success'})


@app.route('/rest/behaviorAnalysis/TaskStatus', methods=["POST"])
async def task_status(request):
    param = request.json
    result = dict()
    result["taskId"] = param["taskId"]
    result["cameraList"] = list()
    for cam_id in param["cameraList"]:
        if (param["taskId"], cam_id) in task:
            if task[(param["taskId"], cam_id)].is_alive():
                status = {"id": cam_id, "status": 1}
            else:
                status = {"id": cam_id, "status": 2}
                task.pop((param["taskId"], cam_id))
            result["cameraList"].append(status)
    return response.json(result)


@app.route('/rest/behaviorAnalysis/delTask', methods=["POST"])
async def delete_task(request):
    param = request.json
    to_del = list()
    for cam_id in param[0]["cameras"]:
        if (param[0]["taskId"], cam_id) in task:
            task[(param[0]["taskId"], cam_id)].state = False
            time.sleep(1.5)
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(task[(param[0]["taskId"], cam_id)].ident), ctypes.py_object(SystemExit))
            to_del.append(cam_id)
    for cam_id in to_del:
        task.pop((param[0]["taskId"], cam_id))
    return response.json({"ret": 0, "desc": 'success'})


@app.route('/data/alarm/<cam_id>/<name>', methods=["GET"])
async def get_image(request, cam_id, name):
    image = os.path.join("data/alarm", cam_id, name)
    if os.path.exists(image):
        return await response.file(image)
    else:
        return response.json({})


if __name__ == '__main__':
    logger.add("data/logs/runtime.log", rotation="100 MB")
    task = {}
    alarm_url = "http://5.5.3.229:7777/alarm/commit"
    host_ip = "5.5.5.238"
    host_port = 9810
    app.run(host=host_ip, port=host_port)

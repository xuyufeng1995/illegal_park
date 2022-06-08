import cv2
import torch
import numpy as np
from detect.detector import YOLOV5_ONNX
import os
from pathlib import Path


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def only_detect(root):
    model = YOLOV5_ONNX(onnx_path="detect/best.onnx")
    for file in [_ for _ in os.listdir(root) if _.endswith("jpg")]:
        image_path = os.path.join(root, file)
        frame = cv2.imread(image_path)
        predict = model.infer(frame)[0].numpy()
        # model.draw(frame, predict, save_path="runs/test.jpg")
        with open(image_path.replace("jpg", "txt"), "w") as f:
            # 数据格式：类别id 中心点x坐标 中心点y坐标 w h（相对于图片宽高）
            predict[..., :4] = xyxy2xywh(predict[..., :4])
            for *xywh, conf, cls in predict:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (cls,
                                                      xywh[0] / frame.shape[1],
                                                      xywh[1] / frame.shape[0],
                                                      xywh[2] / frame.shape[1],
                                                      xywh[3] / frame.shape[0]))


def collect_sampler(video_folder, col_time="20220426", address="wlxysyey", only=None):
    model = YOLOV5_ONNX(onnx_path="detect/best.onnx")
    for file in os.listdir(video_folder):
        if isinstance(only, list) and file not in only:
            continue
        print(file)
        video_path = os.path.join(video_folder, file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        if fps > 25:
            fps = 25
        fps = int(fps)

        i = 0

        floder_name = Path(video_folder).name.split("_")[0]
        os.makedirs(os.path.join("results", floder_name), exist_ok=True)
        count = (len(os.listdir(os.path.join("results", floder_name))) + 1) // 2
        while cap.isOpened():
            cap.grab()
            i += 1
            if i > 3 * fps:
                i = 0
                success, image = cap.retrieve()
                if success:
                    batch_bbox_info = model.infer(image)
                    if batch_bbox_info[0] is None or len(batch_bbox_info[0]) == 0:
                        continue
                    save_path = os.path.join("results", floder_name, "%s_sw_sun_day_pub_%s_%07d.jpg" % (address, col_time, count))
                    cv2.imwrite(save_path, image)
                    bbox_info = batch_bbox_info[0]
                    with open(save_path.replace("jpg", "txt"), "w") as f:
                        # 数据格式：类别id 中心点x坐标 中心点y坐标 w h（相对于图片宽高）
                        bbox_info[..., :4] = xyxy2xywh(bbox_info[..., :4])
                        for *xywh, conf, cls in bbox_info.numpy():
                            f.write("%d %.6f %.6f %.6f %.6f\n" % (cls,
                                                                  xywh[0]/image.shape[1],
                                                                  xywh[1]/image.shape[0],
                                                                  xywh[2]/image.shape[1],
                                                                  xywh[3]/image.shape[0]))
                            # model.draw(image, bbox_info[0], save_path=save_path)
                    count += 1
                else:
                    break

        print("finish {}".format(file))


if __name__ == "__main__":
    only_detect(root="/home/xuyufeng/Projectes/pycharm/illegal_park/results/暮云幼儿园前（三期）")
    #  collect_sampler("/home/xuyufeng/Projectes/pycharm/illegal_park/data/val_video/", col_time="20220606", address="ctglyeymk", only=["biaozhu.mp4"])
    # collect_sampler("/home/xuyufeng/Projectes/datasets/人非车/暮云幼儿园前（三期）_1654476981_D8A6B2F8", col_time="20220606",
    #                address="myyeyq")

import os
import cv2
import random
import numpy as np


def read_data():
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(4)]
    root = "/home/xuyufeng/dataset/cstw/22/JPEGimages/"
    count = 0
    for name in [_ for _ in os.listdir(root) if _.endswith("jpg")]:
        file = os.path.join(root, name)
        image = cv2.imread(file)
        height, width, _ = image.shape
        label_file = "/home/xuyufeng/dataset/cstw/22/labels/" + name[:-3] + "txt"
        with open(label_file, "r") as f:
            label = f.read().strip().split("\n")
        for box_str in label:
            box = box_str.split(" ")
            cls = int(box[0])
            cx = float(box[1]) * width
            cy = float(box[2]) * height
            w = float(box[3]) * width
            h = float(box[4]) * height
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)

            cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], thickness=2, lineType=cv2.LINE_AA)

        cv2.imwrite("test/"+name, image)
        count += 1
        if count > 100:
            break


def show_box():
    img = cv2.imread("./test.jpg")
    rect1 = ((1325, 615), (1529, 840))
    rect2 = ((1187, 662), (1358, 894))
    cv2.rectangle(img, rect1[0], rect1[1], (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, rect2[0], rect2[1], (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite("black.jpg", img)


if __name__ == "__main__":
    show_box()

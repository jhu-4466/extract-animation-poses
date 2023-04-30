import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")

# 设置读取的文件夹路径和文件类型（图片或视频）
folder_path = 'F:/CodeProjects/transform_animation/3danimations/output/Magic Heal/blender/'
file_type = 'png'
file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_type)]

nums = 0
# 处理每一帧图像
for file_path in file_list:
    oriImg = cv2.imread(file_path)

    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)
    canvas = util.draw_handpose(canvas, all_hand_peaks)

    nums += 1
    cv2.imwrite(f'F:/CodeProjects/transform_animation/3danimations/output/Magic Heal/prompts/{nums}.png', canvas)

# 释放资源并关闭窗口
cv2.destroyAllWindows()


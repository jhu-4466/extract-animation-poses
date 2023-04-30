# -*- coding: utf-8 -*- #

# -----------------------------
# Topic: ai animation from 3d to 2d
# Author: motm14
# Created: 2023.04.29
# Description: by blender and openpose, transform 3d animation to 2d animation
# History:
#    <autohr>    <version>    <time>        <desc>
#    motm14         v0.1    2023/04/        basic build
# -----------------------------


import argparse

import bpy
import math

import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
import datetime

from openpose.src import model
from openpose.src import util
from openpose.src.body import Body


def render_animation(fbx_file, output, frames, camera_x, camera_z):
    # 导入fbx文件
    bpy.ops.import_scene.fbx(filepath=fbx_file)

    # 设置相机位置和角度
    bpy.context.scene.camera.location = (camera_x, 0, camera_z)
    bpy.context.scene.camera.rotation_euler = (math.radians(90), 0, math.radians(90))

    # 设置渲染输出路径和格式
    bpy.context.scene.render.filepath = output
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    # 设置帧范围
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames

    # 设置渲染引擎为Cycles
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # CYCLES

    # 设置分辨率
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    # 添加渲染队列
    for obj in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(obj)
    bpy.ops.transform.translate(value=(0, 0, 1), orient_type='GLOBAL')

    # 渲染
    bpy.ops.render.render(animation=True)


def extract_poses(png_folder, output_folder):
    # 读取显卡信息，设置读取的文件夹路径和文件类型（图片或视频）
    print(f"Torch device: {torch.cuda.get_device_name()}")
    
    file_list = [os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith('png')]

    # 处理每一帧图像
    body_estimation = Body('./openpose/model/body_pose_model.pth')
    for file_path in file_list:
        oriImg = cv2.imread(file_path)

        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_png = output_folder + file_path.split('/')[-1].split('.')[0] + '.png'
        print("Saved: '", output_png, "'")
        cv2.imwrite(output_png, canvas)

    # 释放资源并关闭窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--animation_3d", type=str,
                        required=True, help="3d animation file")
    parser.add_argument("-o", "--output_path", type=str,
                        required=True, help="output file abs path")
    parser.add_argument("-x", "--camera_x", type=float,
                        required=True, help="x index of camera")
    parser.add_argument("-z", "--camera_z", type=float,
                        required=True, help="z index of camera")
    parser.add_argument("-f", "--frames", type=int,
                        required=True, help="num of frames")
    
    args = parser.parse_args()
    
    animation_file = args.animation_3d
    output_filepath = args.output_path + animation_file.split('/')[-1].split('.')[0]
    blender_output = output_filepath + '/blender/'
    prompt_output = output_filepath + '/prompts/'
    
    # Blender 渲染
    render_animation(args.animation_3d, blender_output, args.frames, args.camera_x, args.camera_z)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': render animation successfullly')
    
    # Openpose 提取姿势
    extract_poses(blender_output, prompt_output)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': extract poses successfullly')
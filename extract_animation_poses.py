# -*- coding: utf-8 -*- #

# -----------------------------
# Topic: Intelligently transforms a animation from 3d to 2d
# Author: motm14
# Created: 2023.04.29
# Description: by blender and openpose, transforms 3d animation to 2d animation
# History:
#    <autohr>    <version>    <time>        <desc>
#    motm14         v0.1    2023/04/30      basic build
# -----------------------------


import bpy
import cv2
import numpy as np
import torch

import argparse
import math
import os
import shutil
import datetime

from pytorch_openpose.src import util
from pytorch_openpose.src.body import Body


def render_animation(fbx_file, output, camera_x, camera_z, frames=60):
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
    bpy.ops.render.render(animation=True, write_still=True)


def extract_poses(blender_folder, output_folder):
    # 读取显卡信息，设置读取的文件夹路径和文件类型（图片或视频）
    print(f"Torch device: {torch.cuda.get_device_name()}")
    
    file_list = [os.path.join(blender_folder, f) for f in os.listdir(blender_folder) if f.endswith('png')]

    # 处理每一帧图像
    body_estimation = Body('./pytorch_openpose/model/body_pose_model.pth')
    for file_path in file_list:
        oriImg = cv2.imread(file_path)
        img_height, img_width, _ = cv2.imread(file_list[0]).shape

        candidate, subset = body_estimation(oriImg)
        canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_png = output_folder + file_path.split('/')[-1].split('.')[0] + '.png'
        print("Saved: '", output_png, "'")
        cv2.imwrite(output_png, canvas)

    # 释放资源并关闭窗口
    cv2.destroyAllWindows()


def stitch_pngs(prompt_folder, output_folder):
    # 设置文件路径
    file_list = [os.path.join(prompt_folder, f) for f in os.listdir(prompt_folder) if f.endswith('png')]
    
    # 设置画布
    img_height, img_width, _ = cv2.imread(file_list[0]).shape
    canvas_width = img_width * len(file_list)
    canvas = np.zeros((img_height, canvas_width, 3), dtype=np.uint8)
    
    # 拼接
    for i, file_path in enumerate(file_list):
        img = cv2.imread(file_path)
        canvas[:, i*img_width:(i+1)*img_width, :] = img

    # 保存并释放资源
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(output_folder + '/' + output_folder.split('/')[-1] + '.png', canvas)
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
    output_path = args.output_path + animation_file.split('/')[-1].split('.')[0]
    blender_folder = output_path + '/blender/'
    prompt_folder = output_path + '/prompts/'
    stitch_folder = output_path
    
    # Blender 渲染
    render_animation(fbx_file=args.animation_3d, output=blender_folder, camera_x=args.camera_x, camera_z=args.camera_z, frames=args.frames)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': render animation successfullly')
    
    # Openpose 提取姿势
    extract_poses(blender_folder, prompt_folder)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': extract poses successfullly')
    
    # 拼接图片
    stitch_pngs(prompt_folder, stitch_folder)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': stitch pngs successfullly')
    
    # 清空渲染、提取姿势文件夹
    shutil.rmtree(blender_folder)
    shutil.rmtree(prompt_folder)

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

import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T



def render_animation(fbx_file, output, frames):
    # 导入fbx文件
    bpy.ops.import_scene.fbx(filepath=fbx_file)

    # 设置相机位置和角度
    bpy.context.scene.camera.location = (6, 0, 2)
    bpy.context.scene.camera.rotation_euler = (math.radians(90), 0, math.radians(90))

    # 设置渲染输出路径和格式
    bpy.context.scene.render.filepath = output
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    # 设置帧范围
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 2

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--animation_3d", type=str,
                        required=True, help="3d animation file")
    parser.add_argument("-o", "--output_path", type=str,
                        required=True, help="output file abs path")
    parser.add_argument("-f", "--frames", type=int,
                        required=True, help="num of frames")
    
    args = parser.parse_args()
    
    animation_file = args.animation_3d
    output_filepath = args.output_path + animation_file.split('/')[-1].split('.')[0]
    blender_output = output_filepath + '/blender/'
    prompt_output = output_filepath + '/prompt/'
    
    # Blender 渲染
    # render_animation(args.animation_3d, blender_output, args.frames)
    
    # Openpose 提取姿势
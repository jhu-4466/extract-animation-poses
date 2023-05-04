# 提取3d动画的姿势（Extract 3d animation poses）

## 资源（Resources）

1.  小红书博主“芒果和小猫（9632517736）”的《我写了一个脚本AI生成游戏角色动画》
2.  pytorch-openpose：<https://github.com/Hzzone/pytorch-openpose>
3.  openpose 使用教程： <https://blog.csdn.net/liaoqingjian/article/details/115366866>
4.  CUDA 安装教程： <https://blog.csdn.net/Jin1Yang/article/details/124754015>
5.  Mixamo 官网： https://www.mixamo.com/#/

## 描述（Describtions）

将一个3D的人物动画，通过blender、openpose进行“侧面”渲染、提取姿势等操作，完成3D与2D动画的转换（尚未完成2d动画的生成）。

## 使用步骤（Get Started & Run）

1. 安装环境（Install Requriements）

首先，需要更新GPU驱动以及安装CUDA、CUDNN，可以参考 <https://blog.csdn.net/Jin1Yang/article/details/124754015>

然后安装python依赖库（可以参考自己情况选择在虚拟环境（venv）、或容器（docker）下操作）
```
pip install -r requirements.txt
```
其中torch、torch-vision、torch-audio需要找到对应的版本，具体参考网页 <https://pytorch.org/get-started/locally/>，根据自身情况进行选择

2. 安装预训练模型（Download the Models）

参考 <https://github.com/Hzzone/pytorch-openpose> 的README.md中Download the Models部分。

3. 运行（Run）
```
python .\transform_animation.py -a 'fbx_file_path' -o 'output_path' -f int32 -x float -z float
```
参数解释：

    -a | --animation_3d： FBX文件，以.fbx结尾
    -o | --output_path: 输出文件夹
    -f | --frames: 动画帧数，默认60
    -x | --camera_x: blender中摄像头的x坐标
    -z | --camera_z: blender中摄像头的z坐标

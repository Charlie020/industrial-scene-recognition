# Industrial-Scene-Recognition
Industrial scene recognition / 工业场景识别



## 项目描述

本项目属于图像分类任务，需要搭建神经网络在数据集上进行训练，判断一张图片是否属于工业场景，为2分类。

项目中部分代码参考了李沐老师的代码：[动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)

默认情况下，第`X`次训练结果保存在`runs/expX`中。

项目中已搭建了 `Resnet18` 网络，并在数据集上完成了为期300个epoch的训练，结果保存在 `runs/ResNet18_300` 中，test acc为74.8%，FPS为283。



## 数据集说明

数据集小部分来源于百度收集，大部分筛自Places365数据集。对于从Places365数据集中筛取的图片，我们将 `auto_factory`, `assembly_line`, `chemistry_lab`, `construction_site`, `engine_room`, `excavation`, `hangar-indoor`, `hardware_store`, `industrial_area`,  `oilrig`, `physics_laboratory`, `repair_shop` 视为工业场景，其余所有类别均为非工业场景。

整个数据集拥有11862张图片，其中训练集8896张，测试集2966张，比例为3 : 1。



数据集下载：

百度网盘链接：https://pan.baidu.com/s/1SdGQnOVe-iK4FTeEWE1xtw

提取码：0621



其中：`factory` 中的文件源自百度收集；`places365_standard` 中的文件筛自Places365数据集，包含365个分类。
`train.txt`为训练集，`test.txt`为测试集，用“|”分割了文件路径和标签，0代表是工业场景，1代表不是工业场景。



## 项目结构

```
Industrial-Scene-Recognition
            │  train.py            # 训练
            │  detect.py           # 检测
            │
            ├─ Data
            │   │  test.txt
            │   │  train.txt
            │   │
            │   ├─ factory
            │   └─ places365_standard
            |
            ├─ runs
            │   └─ expX             # 第X次训练结果
            │        │  log.txt
            │        │  result.png
            │        │
            │        └─ weights
            │                best.pt
            │                last.pt
            │
            ├─ utils
            |   │  datasets.py      # 数据预处理
            |   │  model.py         # 模型
            |   └─ tool.py          # 项目中常用的函数
            |
            └─ ......
         
```


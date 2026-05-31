# 模型文件说明

本目录用于存放视觉模块需要的模型文件。当前项目仍以轻量模型和规则为主，适配 N150 级别工控机时要避免使用过大的实时模型。

## 当前模型结构

```text
models/
├── README.md
├── face_detection/
│   └── face_detection_yunet_2023mar.onnx
└── age_gender/
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    └── gender_net.caffemodel
```

## 人脸检测模型

默认优先使用 OpenCV YuNet：

```text
models/face_detection/face_detection_yunet_2023mar.onnx
```

如果 YuNet 不可用，会回退到 OpenCV Haar。Haar 可继续运行，但弱光、侧脸、遮挡和小人脸稳定性更差。

当前靠近检测也是先基于低分辨率人脸框面积占比实现：

```text
largestFaceRatio >= proximity_close_face_ratio
```

后续如果人脸在俯拍场景不稳定，可以替换为轻量人体检测模型。

## 年龄和性别模型

默认路径：

```text
models/age_gender/age_deploy.prototxt
models/age_gender/age_net.caffemodel
models/age_gender/gender_deploy.prototxt
models/age_gender/gender_net.caffemodel
```

如果模型不可用，年龄和性别返回 `unknown`。年龄和性别只建议作为弱参考，不建议作为核心推荐条件。

## N150 工控机上的模型建议

推荐：

```text
低分辨率输入
轻量人脸/人体检测
低频推理
多帧统计
OpenVINO / ONNX Runtime 优化
```

避免：

```text
1080p 30fps 全量推理
大 YOLO 模型高频运行
OpenPose 类重模型
多路摄像头同时识别
```

如果后续采购带 NPU 或独立 AI 加速的工控机，可以考虑替换为更稳定的人体检测和人体属性模型。

## 检查模型状态

启动服务后访问：

```text
http://127.0.0.1:7892/health
```

重点字段：

```text
modelReady
ageGenderReady
ageGenderMode
```

## 后续规划

- 增加轻量人体检测模型，用于替代当前人脸面积靠近判断。
- 增加 OpenVINO 推理路径。
- 记录每个模型的版本、来源、输入尺寸和输出含义。
- 建立现场数据回归测试集，避免模型替换后行为退化。

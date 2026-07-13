# 模型文件说明

本目录用于存放视觉模块需要的模型文件。当前项目仍以轻量模型和规则为主，适配 N150 级别工控机时要避免使用过大的实时模型。

## 当前模型结构

```text
models/
├── README.md
├── person_detection/
│   ├── README.md
│   └── person_yolov8n.onnx
├── face_detection/
│   ├── README.md
│   └── face_detection_yunet_2023mar.onnx
└── age_gender/
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    └── gender_net.caffemodel
```

生产 `.onnx`、`.caffemodel` 权重使用 Git LFS 跟踪，并由
`model-manifest.json` 固定角色、路径和 SHA-256。候选构建会拒绝未解析的
LFS pointer、缺失文件或 hash 不匹配；现场不得补充或替换模型。

## 人体检测模型

推荐放置轻量人体检测 ONNX：

```text
models/person_detection/person_yolov8n.onnx
```

默认按 COCO 类别解析，`person` 类别 ID 为 `0`。当前解析器兼容常见 YOLOv5/YOLOv8 ONNX 输出形状；如果后续换成其他模型，需要在 `vision/person_detector.py` 里适配输出格式。

多数现成 YOLOv8 ONNX 是固定 640x640 输入，当前默认配置也按 640x640。若自行导出 416x416 动态模型，可以同步调整 `person_detector_input_width` 和 `person_detector_input_height`。

人体检测就绪时，proximity 优先使用人体框面积占比：

```text
largestPersonRatio >= proximity_close_person_ratio
```

人体检测比人脸面积更适合侧脸、低头操作、轻微遮挡和俯拍场景。

## 人脸检测模型

默认优先使用 OpenCV YuNet：

```text
models/face_detection/face_detection_yunet_2023mar.onnx
```

如果 YuNet 不可用，会回退到 OpenCV Haar。Haar 可继续运行，但弱光、侧脸、遮挡和小人脸稳定性更差。

人脸检测作为靠近判断的补充，也用于年龄/性别和主脸裁剪。当前人脸靠近判断基于低分辨率人脸框面积占比：

```text
largestFaceRatio >= proximity_close_face_ratio
```

当人体检测模型缺失或未就绪时，会回退到 MediaPipe Pose 的姿态框辅助判断：

```text
bodyBoxRatio >= proximity_close_body_ratio
```

人体检测或人脸已经明确达到 close 时，会跳过姿态回退，避免低功耗工控机每次 proximity 都额外跑姿态推理。

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
轻量人体检测
轻量人脸检测作为补充
姿态辅助只做回退
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

- 基于现场数据选择更稳定的轻量人体检测模型和阈值。
- 增加 OpenVINO 推理路径。
- 记录每个模型的版本、来源、输入尺寸和输出含义。
- 建立现场数据回归测试集，避免模型替换后行为退化。

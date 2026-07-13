# 人脸检测模型目录

默认优先使用 OpenCV YuNet 人脸检测模型：

```text
face_detection_yunet_2023mar.onnx
```

对应默认配置：

```text
models/face_detection/face_detection_yunet_2023mar.onnx
```

该 ONNX 模型通过 Git LFS 进入不可变候选 bundle，售货机现场不得单独放置或替换。

Haar 仅是开发诊断回退。清单模型缺失或 hash 不匹配时候选安装验收必须报告 `modelReady=false`。

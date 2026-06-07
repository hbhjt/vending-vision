# 人脸检测模型目录

默认优先使用 OpenCV YuNet 人脸检测模型：

```text
face_detection_yunet_2023mar.onnx
```

对应默认配置：

```text
models/face_detection/face_detection_yunet_2023mar.onnx
```

该 ONNX 模型不建议上传到 GitHub。部署时请在售货机本机把模型文件放到本目录。

如果该模型不存在，服务会自动回退到 OpenCV Haar 人脸检测。Haar 可以保证服务继续运行，但弱光、侧脸、遮挡和小人脸场景的稳定性会下降。

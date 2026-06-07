# 人体检测模型目录

将轻量人体检测 ONNX 模型放在本目录，默认文件名：

```text
person_yolov8n.onnx
```

对应配置：

```json
{
  "person_detector_model": "models/person_detection/person_yolov8n.onnx",
  "person_detector_input_width": 640,
  "person_detector_input_height": 640,
  "person_detector_score_threshold": 0.35,
  "person_detector_nms_threshold": 0.45,
  "person_detector_person_class_id": 0
}
```

当前解析器按 COCO `person` 类别 ID `0` 读取常见 YOLOv5/YOLOv8 ONNX 输出。多数现成 YOLOv8 ONNX 是固定 640x640 输入，因此默认输入尺寸为 640。模型不存在或推理失败时，视觉服务会自动回退到人脸面积和 MediaPipe Pose 姿态框辅助，不会影响服务启动。

# 售货机 Win10 部署说明

本文档记录视觉模块在售货机 Windows 10 机器上的部署流程。目标机器直接安装 Python，不使用虚拟环境。

## 1. 环境要求

```text
OS: Windows 10
Python: 3.10.x
项目目录: D:\ai-cv\vending_vision
HTTP: http://127.0.0.1:7892
WebSocket: ws://127.0.0.1:7892/ws
```

确认 Python：

```bat
python --version
python -m pip --version
```

如果 `python` 不可用，检查 Python 是否加入了系统 PATH。

## 2. 安装依赖

```bat
cd /d D:\ai-cv\vending_vision
python -m pip install -r requirements.txt
```

如果 `mediapipe` 安装失败，优先确认 Python 是 3.10。当前依赖按 Python 3.10 验证。

## 3. 检查模型文件

部署前确认这些文件存在：

```text
models\person_detection\person_yolov8n.onnx
models\face_detection\face_detection_yunet_2023mar.onnx
models\age_gender\age_deploy.prototxt
models\age_gender\age_net.caffemodel
models\age_gender\gender_deploy.prototxt
models\age_gender\gender_net.caffemodel
```

年龄/性别模型缺失时服务仍可运行，但 `ageRange` 或 `gender` 可能返回 `unknown`。人体检测模型缺失时，会回退到人脸面积和姿态框辅助判断。

## 4. 配置摄像头

售货机摄像头为 `index=0` 时，修改 `config.json`：

```json
{
  "camera_index": 0,
  "camera_backend": "dshow",
  "camera_width": 1280,
  "camera_height": 720,
  "camera_fps": 30,
  "camera_fourcc": "MJPG",
  "camera_keep_open": true
}
```

如果不确定摄像头编号：

```bat
python test_camera.py --probe --max-index 8
```

拍照验证：

```bat
python test_camera.py --index 0 --backend dshow --width 1280 --height 720 --fps 30 --fourcc MJPG --output debug_outputs\camera_test.jpg
```

如果画面打不开或分辨率不对，再调整 `camera_backend`、`camera_width`、`camera_height`、`camera_fourcc`。

## 5. 启动服务

普通模式：

```bat
scripts\start_server.bat
```

该模式用于正式联调和上线，默认关闭过程追踪。

过程追踪模式：

```bat
scripts\start_trace_server.bat
```

该模式用于演示或排查问题，会保存中间图片到：

```text
debug_outputs\process_traces\
```

## 6. 健康检查

启动后访问：

```text
GET http://127.0.0.1:7892/health
GET http://127.0.0.1:7892/camera/status
GET http://127.0.0.1:7892/dashboard
```

重点字段：

```text
mockScenario: off
cameraReady: true
modelReady: true
```

摄像头异常时可手动重开：

```text
POST http://127.0.0.1:7892/camera/reopen
```

## 7. WebSocket 联调

原生层或后端连接：

```text
ws://127.0.0.1:7892/ws
```

通信流程：

```text
1. 客户端连接 /ws
2. 客户端发送 vision.hello
3. 服务返回 vision.ready
4. 客户端定期发送 vision.ping
5. 服务返回 vision.pong
6. 用户靠近后，服务主动推送 vision.profile_result
```

当前流程不需要客户端发送 `vision.start_profile` 或 `vision.cancel`。

## 8. 现场验证命令

```bat
python test_proximity.py --runs 20 --interval 0.5
python test_ws_client.py --wait-seconds 60
python test_integration.py
```

如果没有收到画像推送，改用过程追踪模式查看 `statusReason`：

```bat
scripts\start_trace_server.bat
python test_ws_client.py --wait-seconds 60
```

## 9. 上线建议

- 正式运行使用 `scripts\start_server.bat`。
- 上线时保持 `mock_scenario=off`、`process_trace_enabled=false`。
- 确保 `logs\`、`debug_outputs\` 可写。
- 需要开机自启时，可用 Windows 任务计划程序启动 `scripts\start_server.bat`。
- 长期运行时关注 `/camera/status` 中的 `stream.reconnectCount` 和 `stream.lastError`。

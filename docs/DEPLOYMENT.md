# 售货机 Windows 部署步骤

本文档描述当前双摄生产版本的部署和现场验证流程。默认项目目录：

```text
D:\ai-cv\vending_vision
```

## 1. 环境要求

```text
OS: Windows 10/11
Python: 3.10.x
HTTP: http://127.0.0.1:7892
WebSocket: ws://127.0.0.1:7892/ws
Dashboard: http://127.0.0.1:7892/dashboard
```

确认 Python：

```bat
python --version
python -m pip --version
```

## 2. 安装依赖

```bat
cd /d D:\ai-cv\vending_vision
python -m pip install -r requirements.txt
```

安装后确认 OpenCV 仍支持 Caffe 模型加载：

```bat
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2.dnn, 'readNetFromCaffe'))"
```

预期输出应为 OpenCV 4.x 且第二行为 `True`。如果显示 OpenCV 5.x 或第二行为 `False`，年龄/性别 Caffe 模型会降级为 `unknown`，需要重新安装：

```bat
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python -m pip install opencv-python==4.10.0.84
```

如果 `mediapipe` 安装失败，优先确认 Python 是否为 3.10。

## 3. 放置模型文件

部署机本地需要准备：

```text
models\person_detection\person_yolov8n.onnx
models\face_detection\face_detection_yunet_2023mar.onnx
models\age_gender\age_deploy.prototxt
models\age_gender\age_net.caffemodel
models\age_gender\gender_deploy.prototxt
models\age_gender\gender_net.caffemodel
```

年龄/性别模型缺失时服务仍可运行，但 `ageRange` 或 `gender` 可能返回 `unknown`。人体检测模型缺失时，会回退到人脸面积和姿态框辅助判断。

## 4. 配置双摄

先在 Windows 设备管理器、相机工具或现场调试工具中确认摄像头编号，然后修改 `config.json`：

```json
{
  "cameras": {
    "top": {
      "index": 0,
      "backend": "dshow",
      "width": 1280,
      "height": 720,
      "fps": 30,
      "fourcc": "MJPG",
      "role": "presence",
      "keep_open": true
    },
    "front": {
      "index": 1,
      "backend": "dshow",
      "width": 1280,
      "height": 720,
      "fps": 30,
      "fourcc": "MJPG",
      "role": "profile_tryon",
      "keep_open": true
    }
  }
}
```

启动服务后用接口确认：

```text
GET http://127.0.0.1:7892/camera/roles/status
GET http://127.0.0.1:7892/camera/top/status
GET http://127.0.0.1:7892/camera/front/status
```

如果编号调整后摄像头仍不可读，可调用：

```text
POST http://127.0.0.1:7892/camera/top/reopen
POST http://127.0.0.1:7892/camera/front/reopen
```

## 5. 启动服务

正式/联调模式：

```bat
scripts\start_server.bat
```

停止服务：

```bat
scripts\stop_server.bat
```

## 6. 健康检查

```text
GET http://127.0.0.1:7892/health
GET http://127.0.0.1:7892/version
GET http://127.0.0.1:7892/camera/roles/status
GET http://127.0.0.1:7892/camera/front/owner
GET http://127.0.0.1:7892/session/status
GET http://127.0.0.1:7892/dashboard
```

重点确认：

```text
mockScenario: off
cameraReady: true
modelReady: true
cameras.top.index: 顶部摄像头编号
cameras.front.index: 中部摄像头编号
front_owner.owner: idle
vision_session.state: empty 或当前状态
```

## 7. WebSocket 联调

客户端连接：

```text
ws://127.0.0.1:7892/ws
```

基础流程：

```text
1. 客户端连接 /ws
2. 客户端发送 vision.hello
3. 服务返回 vision.ready
4. 客户端定期发送 vision.ping
5. 服务返回 vision.pong
6. 顶部摄像头推送 presence_status
7. 单人且画面可用时推送 profile_result
8. 前端试衣页发送 vision.try_on.start
9. 服务返回 previewUrl
10. 前端退出试衣发送 vision.try_on.stop
```

## 8. 现场验证顺序

1. 确认顶部和中部摄像头在系统中的编号。
2. 修改 `config.json` 的 `top/front` index。
3. 启动 `scripts\start_server.bat`。
4. 打开 `/camera/roles/status`，确认双摄可读。
5. 打开 `/dashboard`，观察 presence、profile 和试衣模拟层。
6. 单人站到交互区，确认 `profile_result`。
7. 多人进入，确认只推 `presence_status`，且 `payload.occupancy.state=multiple`。
8. 进入试衣，确认 `vision.try_on.started.previewUrl` 可播放。
9. 试衣期间离开，确认收到独立的 `vision.person_departed`，退出试衣后等待下一次画像。

## 9. 上线建议

- 正式运行使用 `scripts\start_server.bat`。
- 保持 `mock_scenario=off`。
- `logs/` 是本机运行数据，不作为代码交付内容。
- 长期运行时关注 `/camera/roles/status` 的 `reconnectCount` 和 `lastError`。

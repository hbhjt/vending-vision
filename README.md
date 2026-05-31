# Vending Vision Module

智能售货机机器视觉模块。

当前版本采用 **视觉端主动推送** 模式：售货机应用连接 WebSocket 后只发送握手和心跳；视觉服务自行管理摄像头、检测是否有人、采集多帧、推理、统计结果，并在有可用画像时主动推送 `vision.profile_result`。

## 当前定位

视觉模块负责：

- 常驻管理工业摄像头。
- 自行检测是否有用户进入识别范围。
- 多帧采样、过滤低质量帧、聚合画像结果。
- 主动通过 WebSocket 推送画像。
- 提供 HTTP 调试接口、摄像头状态接口和采集脚本。

售货机应用负责：

- 连接 `ws://127.0.0.1:7892/ws`。
- 发送 `vision.hello` 和 `vision.ping`。
- 接收 `vision.profile_result` 后刷新推荐。
- 长时间未收到画像时继续展示默认商品。

## 快速启动

```bash
conda activate vending_vision
pip install -r requirements.txt
scripts\start_server.bat
```

服务地址：

```text
HTTP: http://127.0.0.1:7892
WebSocket: ws://127.0.0.1:7892/ws
```

## 摄像头配置

当前已按工业摄像头配置：

```json
{
  "camera_index": 1,
  "camera_backend": "dshow",
  "camera_width": 1280,
  "camera_height": 720,
  "camera_fps": 30,
  "camera_fourcc": "MJPG"
}
```

摄像头探测：

```bash
python test_camera.py --probe --max-index 8
```

拍样张：

```bash
python test_camera.py --index 1 --backend dshow --width 1280 --height 720 --fps 30 --fourcc MJPG --output debug_outputs/camera_test.jpg
```

## 主动推送配置

`config.json` 中的关键字段：

```json
{
  "profile_push_enabled": true,
  "profile_push_interval_ms": 1000,
  "profile_push_cooldown_ms": 8000,
  "profile_sample_count": 5,
  "profile_sample_interval_ms": 300,
  "profile_min_confidence": 0.45,
  "profile_min_valid_frames": 2,
  "profile_detection_width": 416,
  "profile_detection_height": 234
}
```

含义：

| 字段 | 说明 |
| --- | --- |
| `profile_push_enabled` | 是否启用主动推送 |
| `profile_push_interval_ms` | 未得到可用画像时，下一轮检测间隔 |
| `profile_push_cooldown_ms` | 成功推送后冷却时间，避免同一用户频繁刷新 |
| `profile_sample_count` | 每轮采集多少张候选帧 |
| `profile_sample_interval_ms` | 候选帧间隔 |
| `profile_min_confidence` | 单帧进入统计的最低置信度 |
| `profile_min_valid_frames` | 至少多少张有效帧才推送 |
| `profile_detection_width` | 推理前缩放宽度，降低工控机负载 |
| `profile_detection_height` | 推理前缩放高度，降低工控机负载 |

当前逻辑是：摄像头按 1280x720 采集，推理前缩放到 416x234。这样保留较清晰原始画面，同时控制 N150 工控机压力。

## 靠近检测配置

当前先用低分辨率人脸框面积判断用户是否靠近：

```json
{
  "proximity_enabled": true,
  "proximity_monitor_width": 416,
  "proximity_monitor_height": 234,
  "proximity_present_face_ratio": 0.003,
  "proximity_close_face_ratio": 0.015,
  "proximity_close_consecutive_frames": 2
}
```

含义：

| 字段 | 说明 |
| --- | --- |
| `proximity_present_face_ratio` | 最大人脸框面积占画面的比例，超过则认为有人 |
| `proximity_close_face_ratio` | 最大人脸框面积占画面的比例，超过则认为靠近 |
| `proximity_close_consecutive_frames` | 连续多少帧靠近后才启动完整画像 |

现场调试：

```bash
python test_proximity.py --runs 20 --interval 0.5
```

也可以启动服务后访问：

```text
GET /proximity/check
```

如果用户已经靠近但 `close=false`，降低 `proximity_close_face_ratio`；如果用户很远就触发，升高该值。

## 单用户主目标规则

当前业务默认售货机一次服务一个主用户。画面中如果出现多张脸，系统不会尝试识别所有人，而是选择一个主用户：

```text
1. 如果姿态模型检测到头部关键点，选择距离头部关键点最近的人脸。
2. 如果头部关键点不可用或距离过远，选择面积更大且更靠近画面中心的人脸。
3. 周报展示图中，primary 红框表示主用户，绿色框表示其他检测到的人脸。
```

相关配置：

```json
{
  "primary_face_max_head_distance_ratio": 0.18
}
```

如果主用户脸框经常和姿态不匹配，可以适当降低该值；如果经常匹配不到，可以适当提高该值。

## WebSocket 协议

详细协议见 [PROTOCOL.md](PROTOCOL.md)。

### 握手

客户端发送：

```json
{
  "protocol": "vem.vision.v1",
  "type": "vision.hello",
  "messageId": "hello-001",
  "timestamp": "2026-05-29T12:00:00.000Z",
  "payload": {
    "clientRole": "machine",
    "machineCode": "M001",
    "protocolVersion": 1,
    "capabilities": ["profile_push"]
  }
}
```

服务端返回：

```json
{
  "protocol": "vem.vision.v1",
  "type": "vision.ready",
  "messageId": "ready-001",
  "timestamp": "2026-05-29T12:00:00.100Z",
  "payload": {
    "serverName": "vem-vision-python",
    "serverVersion": "0.2.0",
    "cameraReady": true,
    "modelReady": true,
    "capabilities": ["profile_push"]
  }
}
```

### 主动画像推送

```json
{
  "protocol": "vem.vision.v1",
  "type": "vision.profile_result",
  "messageId": "result-vision-event-xxx",
  "timestamp": "2026-05-29T12:00:04.000Z",
  "payload": {
    "eventId": "vision-event-xxx",
    "detectedAt": "2026-05-29T12:00:03.900Z",
    "profile": {
      "personPresent": true,
      "heightCm": 172,
      "shoulderWidthCm": 43,
      "ageRange": "adult",
      "gender": "unknown",
      "bodyType": "regular",
      "upperColor": "dark",
      "confidence": 0.86
    },
    "quality": {
      "overall": "fair",
      "warnings": [],
      "sampleCount": 5,
      "validFrameCount": 3
    }
  }
}
```

新版协议不再使用 `vision.start_profile` 和 `vision.cancel`。

## HTTP 调试接口

```text
GET /health
GET /version
GET /camera/status
GET /camera/probe?max_index=8&backend=dshow
GET /proximity/check
POST /infer
GET /capture_infer
```

## 测试命令

WebSocket 主动推送测试：

```bash
python test_ws_client.py --wait-seconds 30
```

真实摄像头批量测试：

```bash
python test_real_camera_batch.py --runs 10 --interval 2 --save-frames
```

现场采集一个人的数据：

```bash
python collect_person_dataset.py --person-id p001 --height-cm 170 --age 24 --gender male --body-type medium --samples-per-group 10 --setup-height-cm 180 --setup-distance-cm 20 --setup-tilt-deg 45
```

生成甲方展示中间结果和周报：

```bash
python generate_weekly_report.py --input datasets/field_capture --limit 40
```

输出目录位于 `reports/weekly_xxx/`，包含人脸框图、姿态骨架图、人脸裁剪、识别结果 CSV 和 `WEEKLY_REPORT.md`。

## 当前限制

- 单目 RGB 摄像头在 1.8m 高、20cm 距离、45 度俯拍下，精确身高和性别年龄都不稳定。
- 当前“靠近检测”是通过多帧可用画像和置信度间接判断，后续可替换为更轻量的人体/人脸框面积模型。
- 年龄和性别只能作为弱参考，低光、侧脸、遮挡时建议返回 `unknown`。
- N150 工控机可支撑该流程，但不适合 30fps 全量深度推理。

## 后续计划

- 增加更轻量的人体靠近检测模型，只在靠近后启动完整画像推理。
- 将每个字段拆成独立置信度。
- 基于现场采集数据重新标定身高、体型和颜色规则。
- 增加 OpenVINO 推理路径，进一步适配 N150 工控机。

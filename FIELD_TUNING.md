# 现场调参速查

本文档用于售货机现场排查和快速调参。部署流程见 [DEPLOYMENT.md](DEPLOYMENT.md)，协议说明见 [PROTOCOL.md](PROTOCOL.md)。

## 先确认状态

```bat
scripts\start_server.bat
```

```text
http://127.0.0.1:7892/health
http://127.0.0.1:7892/camera/status
http://127.0.0.1:7892/proximity/check
http://127.0.0.1:7892/dashboard
```

测试命令：

```bat
python test_camera.py --probe --max-index 8
python test_proximity.py --runs 20 --interval 0.5
python test_ws_client.py --wait-seconds 60
```

需要看中间图时启动：

```bat
scripts\start_trace_server.bat
```

然后查看：

```text
debug_outputs\process_traces\
```

## 摄像头问题

| 现象 | 先看哪里 | 处理 |
| --- | --- | --- |
| `cameraReady=false` | `/camera/status` | 跑 `test_camera.py --probe`，确认 `config.json` 的 `camera_index`，售货机通常为 `0` |
| 画面黑 | 测试图片 | 检查摄像头占用、USB、曝光和驱动 |
| 分辨率不是预期 | `/camera/status` 的 actual | 调 `camera_width`、`camera_height`、`camera_fourcc` |
| 读帧偶发失败 | `stream.lastError` | 检查线材供电，必要时提高 `camera_read_retry_count` |
| 每次识别卡顿 | `mode` | 保持 `camera_keep_open=true` |

售货机常用摄像头配置：

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

## 检测不到人

| 现象 | 先看字段 | 调整 |
| --- | --- | --- |
| `/proximity/check` 的 `present=false` | `personReady`、`faceCount`、`bodyVisiblePointCount` | 检查模型文件、光线和画面位置 |
| 人在远处完全不识别 | `largestPersonRatio`、`largestFaceRatio` | 降低 `proximity_present_person_ratio` 或 `proximity_present_face_ratio` |
| 人体模型没工作 | `personReady=false` | 检查 `models/person_detection/person_yolov8n.onnx` |
| 人脸检测不到 | `faceCount=0` | 改善光线，让脸在画面中间 |

如果检测分辨率太低，可尝试：

```json
{
  "proximity_monitor_width": 640,
  "proximity_monitor_height": 360
}
```

CPU 压力高时退回：

```json
{
  "proximity_monitor_width": 512,
  "proximity_monitor_height": 288
}
```

## 有人但不触发靠近

trace 常见：

```text
statusReason=person_present_but_not_close
```

| 现象 | 先看字段 | 调整 |
| --- | --- | --- |
| 人体框明显但 `personCloseNow=false` | `largestPersonRatio` | 降低 `proximity_close_person_ratio` |
| 脸很近但 `faceCloseNow=false` | `largestFaceRatio` | 降低 `proximity_close_face_ratio` |
| 偶尔 close 但不连续 | `closeStreak` | 降低 `proximity_close_consecutive_frames` |
| 用户站位偏侧 | 人体框/人脸框图 | 优先调整摄像头角度 |

可尝试：

```json
{
  "proximity_close_person_ratio": 0.16,
  "proximity_close_face_ratio": 0.012,
  "proximity_close_consecutive_frames": 2
}
```

## 误触发太多

| 现象 | 调整 |
| --- | --- |
| 路过就触发 | 提高 `proximity_close_person_ratio` 或 `proximity_close_face_ratio` |
| 画面晃动导致触发 | 提高 `proximity_close_consecutive_frames` |
| 多人混人 | 提高 `profile_track_min_match_score`，降低 `profile_track_max_center_shift` |
| 同一用户重复推送过频 | 提高 `profile_push_cooldown_ms` |

稳定优先可尝试：

```json
{
  "proximity_close_person_ratio": 0.20,
  "proximity_close_face_ratio": 0.018,
  "proximity_close_consecutive_frames": 3,
  "profile_track_min_match_score": 0.55
}
```

## 推送慢或 CPU 高

| 现象 | 调整 |
| --- | --- |
| 人靠近后推送慢 | 降低 `profile_push_interval_ms` |
| CPU 高 | 降低 `proximity_monitor_width/height`，提高 `profile_push_interval_ms` |
| trace 很慢 | 正式模式关闭 `process_trace_enabled` |

速度优先可尝试：

```json
{
  "profile_push_interval_ms": 500,
  "proximity_monitor_width": 512,
  "proximity_monitor_height": 288
}
```

## 画像字段很多 unknown

| 字段 | 可能原因 | 处理 |
| --- | --- | --- |
| `heightCm=null` | 身体不完整 | 调整摄像头角度，增加 body buffer |
| `bodyType=unknown` | 肩/髋关键点不足 | 保证上半身入画 |
| `upperColor=unknown` | 上衣区域不清晰 | 改善光线，避免背景混入 |
| `ageRange=unknown` | 主脸不清晰 | 让用户靠近后脸部居中 |
| `gender=unknown` | 年龄性别模型缺失或脸图差 | 检查模型文件和 face crop |

身体字段优化可尝试：

```json
{
  "profile_body_buffer_max_frames": 12,
  "profile_body_buffer_ttl_ms": 5000
}
```

## 字段可信度说明

| 字段 | 稳定性 | 建议 |
| --- | --- | --- |
| `personPresent` | 较高 | 可作为是否有人的主判断 |
| `upperColor` | 中等 | 光线稳定时可用 |
| `bodyType` | 中等偏低 | 只做粗分类 |
| `heightCm` | 中等偏低 | 只做粗略分层 |
| `ageRange` | 低 | 弱参考 |
| `gender` | 低 | 弱参考 |

## 正式上线前

确认：

```json
{
  "mock_scenario": "off",
  "process_trace_enabled": false
}
```

并确认：

- `/health` 中 `cameraReady=true`。
- `/health` 中 `modelReady=true`。
- `/camera/status` 中 `stream.opened=true`。
- `test_ws_client.py --wait-seconds 60` 能收到 `vision.profile_result`，或在无人场景下不会误推送。

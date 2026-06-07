# 摄像头和画像标定步骤

本文档用于固定售货机摄像头后做一次现场标定。日常排查见 [FIELD_TUNING.md](FIELD_TUNING.md)。

## 标定目标

- 无人时不误推送。
- 用户靠近售货机时能及时推送。
- 远处路过不触发画像。
- 身体、颜色、人脸字段尽量减少 `unknown`。
- CPU 和磁盘占用可接受。

## 1. 固定现场条件

先固定这些条件，再开始调参：

```text
摄像头型号
摄像头 index
安装高度
俯仰角度
用户典型站位
现场光线
售货机屏幕反光情况
```

摄像头位置改变后，原来的阈值需要重新验证。

## 2. 确认摄像头

售货机摄像头通常配置为：

```json
{
  "camera_index": 0,
  "camera_backend": "dshow",
  "camera_width": 1280,
  "camera_height": 720,
  "camera_fps": 30,
  "camera_fourcc": "MJPG"
}
```

探测：

```bat
python test_camera.py --probe --max-index 8
```

拍照：

```bat
python test_camera.py --index 0 --backend dshow --width 1280 --height 720 --fps 30 --fourcc MJPG --output debug_outputs/camera_test.jpg
```

启动服务后查看：

```text
http://127.0.0.1:7892/camera/status
```

重点确认 `stream.opened=true`、实际分辨率符合预期。

## 3. 标定靠近阈值

先跑：

```bat
python test_proximity.py --runs 20 --interval 0.5
```

分别记录三种情况：

| 场景 | 期望 |
| --- | --- |
| 无人 | `present=false` |
| 远处路过 | `present` 可为 true，但 `close=false` |
| 正常靠近操作 | `closeNow=true` 或 `close=true` |

主要看这些字段：

```text
personReady
largestPersonRatio
personCloseNow
largestFaceRatio
faceCloseNow
closeStreak
close
```

调参原则：

- 靠近也不触发：降低 `proximity_close_person_ratio` 或 `proximity_close_face_ratio`。
- 远处也触发：提高 `proximity_close_person_ratio` 或 `proximity_close_face_ratio`。
- 偶发误触发：提高 `proximity_close_consecutive_frames`。
- 人体模型不可用：检查 `models/person_detection/person_yolov8n.onnx`。

## 4. 验证 WebSocket 推送

普通模式启动：

```bat
scripts\start_server.bat
```

测试：

```bat
python test_ws_client.py --wait-seconds 60
```

用户靠近后应收到 `vision.profile_result`。无人时长时间没有画像推送是正常情况。

如果有人靠近但没有推送，启动过程追踪：

```bat
scripts\start_trace_server.bat
python test_ws_client.py --wait-seconds 60
```

查看最新目录：

```text
debug_outputs/process_traces/
```

重点看 `manifest.json`：

| `statusReason` | 含义 |
| --- | --- |
| `person_present_but_not_close` | 有人但未达到靠近阈值 |
| `not_enough_valid_frames` | 采样结果有效帧不足 |
| `confidence_below_threshold` | 聚合结果置信度低 |

## 5. 标定画像字段

建议找 3 到 5 个已知身高的人，在相同站位下测试。

| 字段 | 调整方向 |
| --- | --- |
| `heightCm` 整体偏高/偏低 | 调 `height_offset` |
| 高矮变化趋势不对 | 调 `height_scale` |
| `bodyType` 经常偏 slim/strong | 调体型阈值 |
| `upperColor` 不准 | 改善光线和上半身入画 |
| `ageRange` / `gender` 不稳 | 作为弱参考，优先保证人脸清晰 |

注意：单目 RGB 摄像头无法保证精确身高，身高和体型建议作为粗略分层。

## 6. 保存最终配置

每次调整只改一组参数，并记录：

```text
修改参数
无人误触发次数
靠近成功推送次数
平均推送延迟
字段 unknown 情况
CPU 占用
是否开启 trace
```

正式上线前确认：

```json
{
  "mock_scenario": "off",
  "process_trace_enabled": false
}
```

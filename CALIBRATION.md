# 现场标定说明

本文档用于指导真实售货机场景下的摄像头安装、靠近检测阈值、多帧画像推送和结果标定。

当前现场条件：

```text
摄像头高度：约 1.8m
用户距离：约 20cm
摄像头俯拍角度：约 45度
摄像头：工业 USB 摄像头，已验证 1280x720 / 30fps / MJPG
```

这个视角和桌面正面测试差异很大。20cm 距离非常近，45度俯拍会让头肩被放大、腿部被压缩，手臂操作屏幕时也容易遮挡身体。因此不要用正面测试数据直接调售货机场景参数。

## 当前识别流程

正式流程为：

```text
摄像头常驻采集
↓
低分辨率靠近检测
↓
连续多帧判断用户靠近
↓
启动完整多帧画像采样
↓
过滤低质量/低置信帧
↓
聚合最可信结果
↓
主动推送 vision.profile_result
```

无人或未靠近时，视觉服务保持静默，不向后端推送 `no_person`。

## 摄像头配置

当前推荐配置：

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

检查摄像头：

```bash
python test_camera.py --probe --max-index 8
python test_camera.py --index 1 --backend dshow --width 1280 --height 720 --fps 30 --fourcc MJPG --output debug_outputs/camera_test.jpg
```

启动服务后检查：

```text
http://127.0.0.1:7892/camera/status
http://127.0.0.1:7892/health
```

## 靠近检测标定

当前靠近检测先使用低分辨率人脸框面积占比：

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

现场调试：

```bash
python test_proximity.py --runs 20 --interval 0.5
```

重点看输出字段：

```text
largestFaceRatio
present
closeNow
close
closeStreak
```

调参规则：

| 现象 | 调整 |
| --- | --- |
| 用户已经靠近但 `close=false` | 降低 `proximity_close_face_ratio` |
| 用户还很远就 `close=true` | 提高 `proximity_close_face_ratio` |
| 靠近状态忽真忽假 | 提高 `proximity_close_consecutive_frames` |
| 人脸检测不到 | 检查光照、角度、遮挡，或后续改成人体检测模型 |

建议今晚采集时记录不同站位下的 `largestFaceRatio`，后续用数据确定阈值，而不是凭感觉调。

## 多帧画像配置

当前画像推送配置：

```json
{
  "profile_sample_count": 5,
  "profile_sample_interval_ms": 300,
  "profile_min_confidence": 0.45,
  "profile_min_valid_frames": 2,
  "profile_detection_width": 416,
  "profile_detection_height": 234,
  "profile_push_cooldown_ms": 8000
}
```

含义：

- 每次靠近后采 5 张候选帧。
- 每张候选帧间隔 300ms。
- 单帧置信度低于 0.45 不进入统计。
- 至少 2 张有效帧才推送结果。
- 推理前缩放到 416x234，降低 N150 工控机负载。
- 推送后冷却 8 秒，避免同一个用户频繁刷新推荐。

## 单用户主目标标定

当前系统默认只识别一个主用户。多人同时出现在画面中时：

```text
优先选择与姿态头部关键点最近的人脸
匹配不到时，选择面积较大且靠近画面中心的人脸
```

展示材料中：

```text
红色 primary 框：系统选择的主用户
绿色 face 框：其他检测到的人脸
```

相关配置：

```json
{
  "primary_face_max_head_distance_ratio": 0.18
}
```

如果脸和姿态经常错配，应优先检查：

```text
画面里是否多人重叠
主用户是否站在固定识别区域
摄像头角度是否导致头部关键点偏移
primary_face_max_head_distance_ratio 是否过大
```

## 现场标定步骤

1. 固定摄像头高度、角度和位置。
2. 在地面或机器前贴站位标识。
3. 用 `test_camera.py` 拍样图，确认画面范围。
4. 用 `test_proximity.py` 采远/近两组靠近检测数据。
5. 调整 `proximity_close_face_ratio`。
6. 启动服务，运行 `test_ws_client.py --wait-seconds 30`。
7. 让用户靠近，确认是否收到 `vision.profile_result`。
8. 用 `collect_person_dataset.py` 采集多人、多姿态数据。
9. 用 `test_real_camera_batch.py --runs 10 --interval 2 --save-frames` 做稳定性测试。
10. 根据 CSV 和图片调整身高、体型、颜色和置信度规则。

## 身高和体型注意事项

在 1.8m 高、20cm 距离、45度俯拍条件下，不建议把 `heightCm` 当作精确身高。更现实的业务目标是：

```text
是否有人
是否靠近
大致身高区间
大致体型
上衣颜色
整体置信度
```

如果后续要稳定估计身高，建议融合距离传感器、深度摄像头或更严格的站位约束。

## 交付物

完成现场标定后保留：

```text
config.json
datasets/field_capture/
test_reports/
debug_outputs/ 中的典型样图
logs/vision.log
```

不同安装角度可能需要不同配置，不建议跨机器直接复用。

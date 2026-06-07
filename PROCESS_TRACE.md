# 实时过程追踪说明

过程追踪用于保存一次实时识别事件的完整中间结果。它是演示和排查模式，不是默认生产模式。

## 使用场景

建议在这些情况下开启：

- 给负责人或甲方展示识别过程。
- 排查为什么有人但没有推送。
- 查看哪几帧被采样、哪些帧被判定有效。
- 查看人体检测框、人脸框、姿态回退靠近判断、主脸选择、姿态骨架是否合理。

正式联调或上线时建议关闭，只保留标签推送。

## 启动方式

普通模式，不保存过程图：

```bat
scripts\start_server.bat
```

过程追踪模式，保存过程图：

```bat
scripts\start_trace_server.bat
```

也可以手动开启：

```bat
set VISION_PROCESS_TRACE_ENABLED=true
python -m uvicorn app:app --host 127.0.0.1 --port 7892
```

## 输出位置

默认输出到：

```text
debug_outputs/process_traces/
```

每一次事件一个目录：

```text
debug_outputs/process_traces/20260601_203000_vision-event-xxxx/
```

目录结构：

```text
00_proximity/
  proximity_raw.jpg
  proximity_face_boxes.jpg
  proximity_person_boxes.jpg
01_raw_samples/
  sample_01_raw.jpg
  sample_02_raw.jpg
02_inference_frames/
  sample_01_inference.jpg
  sample_02_inference.jpg
03_face_boxes/
  sample_01_faces.jpg
  sample_02_faces.jpg
04_face_crops/
  sample_01_primary_face.jpg
05_pose/
  sample_01_pose.jpg
  sample_02_pose.jpg
manifest.json
```

## 文件含义

| 路径 | 含义 |
| --- | --- |
| `00_proximity/proximity_raw.jpg` | 判断是否有人/是否靠近时的原始画面 |
| `00_proximity/proximity_face_boxes.jpg` | 靠近检测画面中的人脸框；人体辅助判断字段记录在 `manifest.json` |
| `00_proximity/proximity_person_boxes.jpg` | 轻量人体检测模型可用时保存的人体框图 |
| `01_raw_samples/` | 接近过程缓存帧和当前近距离帧的原始图 |
| `02_inference_frames/` | 缩放后输入识别流程的帧 |
| `03_face_boxes/` | 每帧人脸检测结果，红色 `primary` 是主用户 |
| `04_face_crops/` | 年龄/性别识别使用的主脸裁剪图 |
| `05_pose/` | MediaPipe 姿态骨架图 |
| `manifest.json` | 本次事件的结构化记录 |

## manifest 字段

`manifest.json` 里最常看的字段：

```json
{
  "eventId": "vision-event-xxx",
  "status": "pushed",
  "statusReason": null,
  "proximity": {
    "present": true,
    "close": true,
    "personReady": true,
    "personPresent": true,
    "facePresent": true,
    "bodyPresent": false,
    "largestPersonRatio": 0.21,
    "largestFaceRatio": 0.018,
    "bodyBoxRatio": 0.0,
    "method": "person_detector+face_area_ratio"
  },
  "samples": [],
  "payload": {}
}
```

`proximity` 字段用于判断为什么进入或没有进入主流程：

| 字段 | 说明 |
| --- | --- |
| `personReady` | 轻量人体检测模型是否已加载 |
| `personPresent` / `personCloseNow` | 人体检测框是否满足有人/靠近阈值 |
| `largestPersonRatio` | 最大人体框面积比例 |
| `facePresent` / `faceCloseNow` | 人脸面积是否满足有人/靠近阈值 |
| `bodyPresent` / `bodyCloseNow` | 姿态回退框是否满足有人/靠近阈值 |
| `bodySkipped` | 人体检测或人脸已明确靠近时会跳过姿态回退，降低性能开销 |
| `largestFaceRatio` | 最大人脸框面积比例 |
| `bodyVisiblePointCount` | 可见人体姿态点数量 |
| `bodyBoxRatio` | 人体姿态点外接框面积比例 |
| `closeNow` / `closeStreak` | 当前帧是否已到近距离阈值 / 连续满足靠近的帧数 |
| `closeTrigger` | 本次推送触发方式，常见为 `close_now` 或 `close_streak` |

状态说明：

| status | statusReason | 含义 |
| --- | --- | --- |
| `pushed` | `null` | 已成功推送 `vision.profile_result` |
| `not_pushed` | `person_present_but_not_close` | 检测到有人，但当前帧还没到近距离阈值 |
| `not_pushed` | `not_enough_valid_frames` | 已采样，但有效帧数量不足 |
| `not_pushed` | `confidence_below_threshold` | 聚合后置信度低于阈值 |

如果 `person_present_but_not_close` 很多，优先查看 `personCloseNow`、`faceCloseNow`、`bodyCloseNow` 和 `closeStreak`。如果 `personReady=false`，说明正在使用人脸和姿态回退；如果 `personReady=true` 但 `largestPersonRatio` 偏低，可以根据现场站位调低 `proximity_close_person_ratio`。

## 和 WebSocket 返回的关系

普通协议只返回画像标签，不传图片：

```json
{
  "type": "vision.profile_result",
  "payload": {
    "eventId": "vision-event-xxx",
    "profile": {},
    "quality": {}
  }
}
```

开启过程追踪后，推送结果里会额外带本地目录：

```json
{
  "quality": {
    "trace": {
      "eventDir": "debug_outputs\\process_traces\\20260601_203000_vision-event-xxx"
    }
  }
}
```

这个字段只用于本机调试和展示，不要求原生层/后端依赖它。

## 配置

```json
{
  "process_trace_enabled": false,
  "process_trace_output_dir": "debug_outputs/process_traces",
  "process_trace_max_events": 50
}
```

说明：

| 字段 | 说明 |
| --- | --- |
| `process_trace_enabled` | 是否保存实时过程追踪 |
| `process_trace_output_dir` | 追踪目录 |
| `process_trace_max_events` | 最多保留多少个事件目录，超过后自动清理旧目录 |

## 推荐使用方式

演示时：

1. 启动 `scripts\start_trace_server.bat`。
2. 运行 `python test_ws_client.py --wait-seconds 60`。
3. 用户站到摄像头前并靠近。
4. 成功推送后打开 `debug_outputs/process_traces/` 最新目录。
5. 展示 `proximity`、`raw_samples`、`face_boxes`、`face_crops`、`pose` 和 `manifest.json`。

联调时：

1. 启动 `scripts\start_server.bat`。
2. 后端只消费 `vision.profile_result`。
3. 不保存中间图片，降低磁盘占用和额外推理开销。

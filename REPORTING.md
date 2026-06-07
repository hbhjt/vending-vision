# 展示材料和周报说明

项目提供两类展示材料：

1. 实时过程追踪：跟随 WebSocket 实时流程产生，适合演示“用户靠近后系统如何处理”。
2. 离线周报：对一批已有图片做分析，生成 Markdown 周报和 CSV，适合每周汇报。

## 实时过程追踪

启动：

```bat
scripts\start_trace_server.bat
```

测试：

```bat
python test_ws_client.py --wait-seconds 60
```

输出：

```text
debug_outputs/process_traces/
```

每个事件包含：

- 摄像头发现人时的图片。
- 靠近检测人体框、人脸框，以及姿态回退判断字段。
- 多帧采样原图。
- 输入识别流程前的缩放图。
- 人脸框和主脸裁剪。
- 姿态骨架图。
- 每帧画像结果和最终 payload。

详细说明见 [PROCESS_TRACE.md](PROCESS_TRACE.md)。

## 离线周报

脚本：

```bat
python generate_weekly_report.py --input datasets/field_capture --limit 40
```

指定输出目录：

```bat
python generate_weekly_report.py --input datasets/field_capture --output reports/week_01 --limit 40
```

输出结构：

```text
reports/week_01/
  WEEKLY_REPORT.md
  weekly_results.csv
  manifest.json
  original/
  face_boxes/
  face_crops/
  pose/
```

| 文件/目录 | 用途 |
| --- | --- |
| `WEEKLY_REPORT.md` | 可直接给负责人看的周报草稿 |
| `weekly_results.csv` | 每张图片的结构化识别结果 |
| `manifest.json` | 本次生成任务摘要 |
| `original/` | 原始图片副本 |
| `face_boxes/` | 人脸框结果图 |
| `face_crops/` | 人脸裁剪图 |
| `pose/` | 姿态骨架图 |

## 展示建议

建议展示顺序：

1. 原始摄像头画面。
2. 检测到人脸/靠近的画面。
3. `proximity` 字段：展示人体检测、人脸和姿态回退分别是否触发。
4. 被选中的采样帧。
5. 主用户人脸框和裁剪图。
6. 姿态骨架图。
7. 单帧画像结果。
8. 多帧聚合后的最终画像。
9. 摄像头长期运行状态和当前限制。

需要谨慎说明：

- 年龄和性别在弱光、侧脸、遮挡条件下不稳定，目前应作为弱参考。
- 单目 RGB 摄像头无法保证精确身高，当前身高更适合做粗略分层。
- 现场 1.8m 高度、20cm 距离、45 度倾斜安装会改变画面比例，需要现场重新调参。
- 轻量人体检测可以缓解侧脸和轻微遮挡；如果模型缺失会自动回退到姿态辅助，但多人、严重遮挡、强反光仍需要现场验证。
- 长期运行时建议定期查看 `/camera/status` 的 `stream.reconnectCount` 和 `stream.lastError`，确认摄像头连接稳定。



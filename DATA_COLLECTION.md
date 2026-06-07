# 现场数据采集说明

该文档用于说明如何采集真实售货机场景数据。数据主要用于离线评估、调参和后续模型改进。

## 采集原则

采集环境应尽量复现最终部署条件：

```text
摄像头高度：约 1.8m
用户距离：约 20cm
摄像头倾斜：约 45 度
摄像头参数：1280x720 / 30fps / MJPG
运行方式：正式服务长期持有摄像头连接，采集脚本单独打开摄像头
```

如果摄像头位置、角度、距离或光线明显变化，应视为新的采集批次。

采集前建议先确认正式服务没有占用摄像头。如果 `scripts\start_server.bat` 正在运行，采集脚本可能打不开同一个工业摄像头。采集数据时可以先停止视觉服务；采集完成后再启动服务，并访问：

```text
GET http://127.0.0.1:7892/camera/status
```

确认 `mode=persistent`、`stream.opened=true`，再做 WebSocket 联调。

## 每个人采集哪些动作

建议每个人采集 3 到 5 组，每组 8 到 12 张。

推荐动作：

```text
front_still      正面自然站立
operate_screen   模拟操作屏幕
lean_forward     轻微前倾看屏幕
turn_left        身体轻微左转
turn_right       身体轻微右转
face_occluded    轻微遮挡/戴口罩，验证人体检测和姿态回退
```

时间有限时，至少采集：

```text
front_still
operate_screen
lean_forward
turn_left
```

## 标签怎么填

图片由脚本自动拍摄，标签由你在命令行里人工填写。

建议记录：

- 匿名编号，例如 `p001`
- 真实身高
- 年龄
- 性别
- 大致体型
- 备注，例如光线暗、戴帽子、戴口罩、穿深色衣服
- 是否能稳定看到人脸
- 是否能稳定看到上半身或肩部

不要记录真实姓名。

## 一键采集命令

每换一个人，运行一次命令：

```bat
python collect_person_dataset.py --person-id p001 --height-cm 170 --age 24 --gender male --body-type medium
```

完整现场命令：

```bat
python collect_person_dataset.py --person-id p001 --height-cm 170 --age 24 --gender male --body-type medium --samples-per-group 10 --interval 0.4 --setup-height-cm 180 --setup-distance-cm 20 --setup-tilt-deg 45
```

只采核心三组：

```bat
python collect_person_dataset.py --person-id p001 --height-cm 170 --age 24 --gender male --body-type medium --groups front_still,operate_screen,lean_forward --samples-per-group 10
```

脚本会在每组开始前暂停，按回车后倒计时并开始采集。

## 输出结构

数据保存到：

```text
datasets/field_capture/
```

每个人一次采集生成：

```text
datasets/field_capture/20260529_203000_p001/
  metadata.json
  manifest.csv
  images/
    front_still/
    operate_screen/
    lean_forward/
    turn_left/
    turn_right/
```

说明：

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 人员标签、摄像头配置、现场安装参数 |
| `manifest.csv` | 每张图片路径、分组、亮度、清晰度和标签 |
| `images/` | 实际采集图片 |

## 质量建议

1. 每个人采完后检查目录里是否有图片和 `manifest.csv`。
2. 光线变化明显时，在 `--note` 里记录。
3. 有帽子、口罩、反光、遮挡、深色衣服时也记录。
4. 尽量覆盖不同身高、体型、衣服颜色、年龄和性别。
5. 增加侧脸、轻微遮挡、低头操作屏幕等动作，用于验证轻量人体检测 proximity。
6. 摄像头位置改变后，重新开始新的采集批次。

## 最小采集量建议

如果时间有限，优先保证：

```text
至少 8 到 12 人
每人至少 3 组
每组至少 8 张
覆盖不同身高、体型、衣服颜色和性别
```

这批数据主要用于判断当前规则是否可用，以及是否需要替换为更适合现场的人体检测或人体属性模型。

## 和当前线上流程的关系

当前线上流程会先用低分辨率 proximity 判断，再进入多帧画像识别。采集数据时建议额外记录这些场景：

| 场景 | 目的 |
| --- | --- |
| 正脸靠近 | 验证人脸面积靠近阈值 |
| 侧脸靠近 | 验证轻量人体检测是否能补足人脸漏检 |
| 低头操作屏幕 | 验证主流程在真实操作动作下是否能得到有效画像 |
| 远处路过 | 验证不会误触发 `close=true` |
| 无人空场景 | 验证长期运行时不会持续误推送 |

这些样本后续可用于回放调参：重点比较 `personPresent`、`largestPersonRatio`、`facePresent`、`bodyPresent`、`closeStreak`、`bodyBoxRatio` 和最终 `profile.confidence`。

# 现场数据采集说明

本文档用于采集真实售货机场景数据。目标是获得和最终安装条件一致的图片与标签，用于离线评估、调参和后续模型改进。

## 采集原则

现场采集一定要尽量复现最终部署条件：

```text
摄像头高度：约 1.8m
用户距离：约 20cm
摄像头倾斜：约 45度
摄像头参数：1280x720 / 30fps / MJPG
```

摄像头位置、角度或站位一旦改变，就应该视为新的采集批次。

## 每个人采集内容

建议每个人采 5 组，每组 8 到 12 张：

```text
front_still      正面自然站立
operate_screen   模拟点击屏幕
lean_forward     轻微前倾看屏幕
turn_left        身体轻微左转
turn_right       身体轻微右转
```

如果时间有限，至少采：

```text
front_still
operate_screen
lean_forward
```

## 标签怎么填

照片由脚本自动拍摄，标签由你在命令里人工填写。

每个人建议记录：

```text
匿名编号，例如 p001
真实身高
年龄
性别
大致体型
备注，例如光线暗、戴帽子、穿深色衣服
```

不要填写真实姓名。

## 一键采集命令

每换一个人，运行一次命令：

```bash
python collect_person_dataset.py --person-id p001 --height-cm 170 --age 24 --gender male --body-type medium
```

完整现场命令：

```bash
python collect_person_dataset.py --person-id p001 --height-cm 170 --age 24 --gender male --body-type medium --samples-per-group 10 --interval 0.4 --setup-height-cm 180 --setup-distance-cm 20 --setup-tilt-deg 45
```

只采核心三组：

```bash
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
├── metadata.json
├── manifest.csv
└── images/
    ├── front_still/
    ├── operate_screen/
    ├── lean_forward/
    ├── turn_left/
    └── turn_right/
```

`metadata.json` 保存人员标签、摄像头配置和现场安装参数。

`manifest.csv` 保存每张图片路径、分组、亮度、清晰度和标签。

## 采集质量建议

1. 每个人采完后检查目录里是否有图片和 `manifest.csv`。
2. 现场光线变化明显时，在 `--note` 里记录。
3. 有帽子、口罩、反光、遮挡、深色衣服时也记录。
4. 尽量覆盖不同身高、体型、衣服颜色、年龄和性别。
5. 如果摄像头位置改变，重新开始新的采集批次。

## 推荐最小数据量

如果时间有限，优先保证：

```text
至少 8 到 12 人
每人至少 3 组
每组至少 8 张
覆盖不同身高、体型、衣服颜色和性别
```

这批数据会用于判断当前规则是否可用，以及是否需要换成更适合现场的人体检测或人体属性模型。

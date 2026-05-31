# 展示材料和周报生成说明

本项目提供 `generate_weekly_report.py`，用于把现场照片处理成可展示的中间结果，并生成一份周报 Markdown。

## 生成内容

输入一批图片后，会输出：

```text
reports/weekly_xxx/
├── WEEKLY_REPORT.md
├── weekly_results.csv
├── manifest.json
├── original/
├── face_boxes/
├── face_crops/
└── pose/
```

说明：

| 目录/文件 | 用途 |
| --- | --- |
| `original/` | 原始图片副本 |
| `face_boxes/` | 画出人脸框后的图片 |
| `face_crops/` | 裁剪出的人脸图片 |
| `pose/` | MediaPipe 姿态骨架图 |
| `weekly_results.csv` | 每张图片的识别结果和质量指标 |
| `WEEKLY_REPORT.md` | 可发给负责人/甲方的周报 |

在人脸框图片中：

```text
红色 primary：系统选择的主用户
绿色 face：其他检测到的人脸
```

当前项目默认单用户场景。多人入镜时，报告会展示多张脸，但最终画像只对应主用户。

## 使用命令

处理现场采集数据：

```bash
python generate_weekly_report.py --input datasets/field_capture --limit 40
```

指定输出目录：

```bash
python generate_weekly_report.py --input datasets/field_capture --output reports/week_01 --limit 40
```

处理单个目录：

```bash
python generate_weekly_report.py --input datasets/field_capture/20260529_203000_p001 --output reports/p001_demo
```

## 建议给甲方展示的内容

建议展示顺序：

1. 原始摄像头画面。
2. 人脸框检测结果。
3. 人脸裁剪结果。
4. 姿态骨架结果。
5. 单张图片画像输出。
6. 多张图片统计后的稳定性说明。
7. 当前限制和下周计划。

## 周报口径建议

建议强调：

- 当前已经完成工业摄像头接入。
- 已切换为视觉端主动推送协议。
- 已增加靠近检测、多帧采样和结果聚合。
- 当前可生成中间处理结果，便于验证算法过程。
- 45度俯拍和20cm近距离会影响身高、体型和性别年龄稳定性。

建议谨慎表达：

- 年龄和性别只作为弱参考。
- 精确身高不是当前阶段的强承诺。
- 后续会基于现场数据继续调靠近阈值和画像规则。

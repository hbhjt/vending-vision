# GitHub 独立仓库上传说明

本文档用于把视觉模块单独上传到个人 GitHub 仓库，方便负责人查看代码和运行方式。

## 建议上传

上传这些内容即可复现代码结构和运行流程：

```text
app.py
config.json
requirements.txt
vision/
dashboard/
scripts/
models/README.md
models/person_detection/README.md
models/face_detection/README.md
README.md
DEPLOYMENT.md
PROTOCOL.md
FIELD_TUNING.md
PROCESS_TRACE.md
CALIBRATION.md
DATA_COLLECTION.md
REPORTING.md
GITHUB_UPLOAD.md
test_camera.py
test_proximity.py
test_ws_client.py
test_integration.py
test_real_camera_batch.py
collect_person_dataset.py
generate_weekly_report.py
```

说明：

- `dashboard/` 需要上传，负责人可以看到本机展示面板。
- `scripts/start_server.bat` 是正式启动脚本。
- `scripts/start_trace_server.bat` 是排查和演示脚本。
- `config.json` 当前已经按售货机摄像头 `camera_index=0` 配置。

## 不建议上传

这些文件属于本地运行产物、现场数据或大模型权重，不适合传 GitHub：

```text
logs/
debug_outputs/
reports/
datasets/
test_reports/
demo_inputs/
output_*.jpg
output_*.png
__pycache__/
.pytest_cache/
.idea/
.vscode/
models/**/*.onnx
models/**/*.caffemodel
models/**/*.pb
models/**/*.tflite
models/**/*.pth
models/**/*.pt
```

原因：

- `logs/`、`debug_outputs/`、`test_reports/` 是运行或测试输出。
- `datasets/` 可能包含现场采集图片，不应公开上传。
- `reports/` 是汇报材料和生成图，可按需单独发送，不建议放代码仓库。
- `models/**/*.onnx`、`models/**/*.caffemodel` 通常较大，且可能涉及模型来源和授权。

## 模型文件如何处理

GitHub 仓库不上传模型权重，但部署机器需要本地放置：

```text
models/person_detection/person_yolov8n.onnx
models/face_detection/face_detection_yunet_2023mar.onnx
models/age_gender/age_deploy.prototxt
models/age_gender/age_net.caffemodel
models/age_gender/gender_deploy.prototxt
models/age_gender/gender_net.caffemodel
```

如果模型缺失：

- 年龄和性别会返回 `unknown`。
- 人体检测会回退到人脸面积和姿态框辅助判断。
- 服务仍可启动，但识别效果会下降。

## 售货机运行方式

目标机器为 Windows 10，直接安装 Python，不使用虚拟环境。

首次安装依赖：

```bat
cd /d D:\ai-cv\vending_vision
python -m pip install -r requirements.txt
```

普通模式启动：

```bat
scripts\start_server.bat
```

过程追踪模式启动：

```bat
scripts\start_trace_server.bat
```

启动后检查：

```text
http://127.0.0.1:7892/health
http://127.0.0.1:7892/camera/status
http://127.0.0.1:7892/dashboard
```

WebSocket 地址：

```text
ws://127.0.0.1:7892/ws
```

## 上传前检查

建议上传前执行：

```bat
git status --short
git check-ignore -v debug_outputs logs datasets reports test_reports demo_inputs
```

如果旧仓库里已经跟踪过测试报告或图片，仅修改 `.gitignore` 不会自动取消跟踪。新建独立仓库时，按本文件清单添加文件即可。

如果要在当前 Git 仓库取消跟踪历史产物，可执行：

```bat
git rm -r --cached test_reports demo_inputs reports datasets debug_outputs logs
```

这条命令只取消 Git 跟踪，不删除本地文件。执行前请确认这些目录不需要继续进入仓库。

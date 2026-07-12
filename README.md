# Vending Vision Module

售货机视觉服务，运行在售货机本机。当前项目已经收敛为“双摄 + WebSocket 主动推送 + 试衣 MJPEG 预览”的生产流程。

## 当前能力

- 顶部摄像头 `top`：常开，用于检测有人、无人、离开和多人状态。
- 中部摄像头 `front`：按需占用，用于人物画像采集和前端试衣预览。
- 视觉端通过 WebSocket 主动推送 `vision.presence_status`、`vision.profile_result`、`vision.person_departed`。
- 前端通过 `vision.try_on.start/stop` 申请和归还试衣预览流。
- 多人场景只推状态，不推人物画像字段。

## 快速启动

```bat
cd /d D:\ai-cv\vending_vision
python -m pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2.dnn, 'readNetFromCaffe'))"
scripts\start_server.bat
```

启动后访问：

```text
http://127.0.0.1:7892/health
http://127.0.0.1:7892/dashboard
http://127.0.0.1:7892/metrics
ws://127.0.0.1:7892/ws
```

停止服务：

```bat
scripts\stop_server.bat
```

## 摄像头配置

`config.json` 使用双摄配置：

```json
{
  "cameras": {
    "top": {
      "index": 0,
      "role": "presence",
      "keep_open": true
    },
    "front": {
      "index": 1,
      "role": "profile_tryon",
      "keep_open": true
    }
  }
}
```

现场部署时先确认 Windows 摄像头编号，再修改 `cameras.top.index` 和 `cameras.front.index`。修改后用下面接口确认两路摄像头状态：

```text
GET http://127.0.0.1:7892/camera/roles/status
```

## 目录结构

```text
app.py              FastAPI 服务入口
config.json         默认配置
requirements.txt    Python 依赖
vision/             摄像头、画像、协议、试衣和会话状态核心代码
dashboard/          本地调试 Dashboard
scripts/            Windows 启停脚本
docs/               架构、协议和部署文档
models/             本地模型文件和模型说明
logs/               本机运行日志
```

## 文档入口

- [架构说明](docs/ARCHITECTURE.md)
- [部署步骤](docs/DEPLOYMENT.md)
- [通信协议](docs/PROTOCOL.md)

## 主要接口

```text
GET  /health
GET  /version
GET  /camera/roles/status
GET  /camera/{role}/status
GET  /camera/{role}/snapshot.jpg
POST /camera/{role}/reopen
GET  /camera/front/owner
GET  /session/status
GET  /proximity/debug
GET  /metrics
GET  /try-on/{sessionId}.mjpeg
WS   /ws
```

## 测试与现场标定

运行单元测试：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
```

分析现场录制视频的 presence 阈值：

```powershell
python scripts\calibrate_presence.py camera_left_index0_20260707_203720.mp4 --frame-step 5
```

脚本会输出 `reports/presence_calibration.csv` 和 `reports/presence_calibration_summary.json`，用于观察 `largestPersonRatio`、`largestFaceRatio`、`bodyBoxRatio` 的分布。

使用根目录中的顶部/中部 MP4 一键生成同步数据集和稳定性报告：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_video_stability_test.ps1
```

脚本按接近生产轮询频率抽帧，自动打开 `reports/video_dataset/test01/stability_report.html`。输出同时包括：

- `top_occupancy_auto_labels.csv`：顶部占用状态和检测指标。
- `front_profile_auto_labels.csv`：正面画像字段和质量指标。
- `stability_timeline.csv`：双摄按时间戳对齐后的完整结果。
- `review_labels.csv`：可用 Excel 填写人工真值的复核表。
- `stability_summary.json`：适合程序读取的稳定性指标。

自动标签只能用于快速筛查。填写 `review_labels.csv` 后重新运行 `analyze_video_stability.py`，报告会额外显示基于人工标签的召回率。

## 现场摄像头方向

每路摄像头可在 `config.json` 的 `cameras` 中设置 `rotate`：

```json
{
  "front": {
    "rotate": 270
  }
}
```

`rotate` 表示服务读取后对画面做的校正角度，单位是顺时针度数。中部摄像头如果物理顺时针旋转 90 度安装，通常用 `270` 做逆时针校正。启动后打开：

```text
http://127.0.0.1:7892/camera/front/snapshot.jpg
```

确认人脸和身体在截图中是正的，再测年龄、性别和试衣。

顶部摄像头可以设置 ROI，只检测售货机前方交互区：

```json
{
  "top": {
    "roi": {
      "enabled": true,
      "x": 0.0,
      "y": 0.0,
      "width": 1.0,
      "height": 1.0
    }
  }
}
```

现场调 ROI 时打开：

```text
http://127.0.0.1:7892/proximity/debug
```

观察 `roi`、`largestPersonRatio`、`largestFaceRatio` 和 `bodyBoxRatio`，再调整 `proximity_*_ratio` 阈值。

## 上线注意

- 推荐 Python 3.10。
- 模型文件不提交到代码仓库时，需要在部署机器本地放入 `models/`。
- 正式运行保持 `mock_scenario=off`。
- 先完成双摄编号确认，再做顶部多人阈值和中部画像质量联调。
- 长期运行可查看 `/metrics` 和 `logs/vision.log`，日志默认按 5MB 滚动保留 5 份。

## 编码说明

项目源码、文档和 Dashboard 统一使用 UTF-8。Windows PowerShell 读取中文文件时建议显式指定编码：

```powershell
Get-Content -Raw -Encoding UTF8 README.md
Get-Content -Raw -Encoding UTF8 docs\DEPLOYMENT.md
```

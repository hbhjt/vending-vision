# Vending Vision Module

售货机视觉服务，运行在售货机本机。当前项目已经收敛为“双摄 + WebSocket 主动推送 + V2 Fast 生成结果”的生产流程。

## 当前能力

- 顶部摄像头 `top`：常开，用于检测有人、无人、离开和多人状态。
- 中部摄像头 `front`：按需占用，用于人物画像采集和 V2 Fast 试衣采集。
- 视觉端通过 WebSocket 主动推送 `vision.presence_status`、`vision.profile_result`、`vision.person_departed`。
- Machine uses the strict V2 attempt protocol for generated try-on.
- 多人场景只推状态，不推人物画像字段。

## 快速启动

```bat
cd /d D:\ai-cv\vending_vision
python -m pip download --only-binary=:all: --require-hashes -d wheelhouse -r requirements.txt
python -m pip install --no-index --find-links wheelhouse --require-hashes -r requirements.txt
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2.dnn, 'readNetFromCaffe'))"
scripts\start_server.bat
```

开发启动后访问：

```text
http://127.0.0.1:7892/health
http://127.0.0.1:7892/dashboard
http://127.0.0.1:7892/metrics
ws://127.0.0.1:7892/ws
```

`/dashboard` 和旧 snapshot 诊断默认关闭。仅供应方开发时可在启动前显式设置
`VISION_DEVELOPMENT_DASHBOARD=true`；带 `--config` 的 VEM 托管生产启动固定关闭。

VEM 托管运行不使用上述源码启动方式。候选发布物必须由受保护的 RC tag
构建为自包含 Windows bundle，VEM 以
`vending-vision.exe --config C:\ProgramData\VEM\vision\config\site.json`
启动。`--config` 会启用严格、失败即停的外部配置模式；示例和 schema 位于
`config/`。

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
      "role": "presence",
      "keep_open": true
    },
    "front": {
      "role": "profile_fast_try_on",
      "keep_open": true
    }
  }
}
```

现场部署不写入 Windows 摄像头编号。Vision 仅在同一 Windows DirectShow 枚举边界能
证明稳定身份与捕获源的关系时才公开可用候选；否则明确保持非就绪。操作员通过
plain loopback maintenance contract 预览、测试并确认 top/front 角色；Vision 保持
稳定身份的唯一所有者。具体合同见
[摄像头维护合同](docs/CAMERA-MAINTENANCE.md)。

```text
GET http://127.0.0.1:7892/camera/roles/status
GET http://127.0.0.1:7892/maintenance/cameras
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
GET  /maintenance/cameras
POST /maintenance/cameras/refresh
GET  /maintenance/cameras/{candidateId}/preview.jpg
POST /maintenance/cameras/{role}/test
POST /maintenance/cameras/{role}/confirm
GET  /proximity/debug
GET  /metrics
GET  /v2/try-on/results/{attemptId}?token={grant}
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

`rotate` 表示服务读取后对画面做的校正角度，单位是顺时针度数。中部摄像头如果物理顺时针旋转 90 度安装，通常用 `270` 做逆时针校正。通过 loopback v2 维护合同的 `/maintenance/cameras/{candidateId}/preview.jpg` 预览确认人脸和身体在画面中是正的，再测年龄、性别和 V2 Fast 试衣。生产环境不存在 `/camera/{role}/snapshot.jpg`。

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

- 开发、CI 和 Candidate 打包统一使用 `.python-version` 固定的 Python 3.11.9，且只消费同一份完整、传递闭包、SHA-256 锁定的 `requirements.txt`；先下载 wheelhouse，再以 `--no-index --require-hashes` 离线安装。
- Windows 正式枚举使用严格 pin 的 `cv2-enumerate-cameras` DirectShow moniker/index 边界；同一稳定 moniker 在 replug 后可解析为新 index。
- 生产模型由 `models/model-manifest.json` 声明并通过 Git LFS 进入候选 bundle；现场不得补模型。
- 正式运行保持 `mock_scenario=off`。
- 先完成双摄编号确认，再做顶部多人阈值和中部画像质量联调。
- 长期运行可查看 `/metrics` 和 `logs/vision.log`，日志默认按 5MB 滚动保留 5 份。

## 候选发布边界

- PR 和普通 `main` 只运行验证，不发布可部署 bundle；仓库不再维护另一套 Development 构建。
- `main` CI 每次构建同一 commit 的 Windows runtime、录播 fixture 与候选交付包，并上传 `vending-vision-main-<commit>` 和 `vending-vision-candidate-<commit>` 两个 Actions artifacts；`vending-vision-main-artifacts.json` 以 SHA-256 锁定 runtime 与 fixture。
- VEM 按 commit 下载成对 artifacts，校验候选 manifest 的 source commit、交付清单 SHA-256 与 fixture 摘要后原样使用；供应仓不安装、不批准、也不重打包候选。
- `scripts/verify_real_camera_capability.py` 用于现场真实双摄核心能力验收，强制 `mockScenario=off` 并验证 presence、单人可用画像和离开。
- `/dashboard` 与旧 `/camera/{role}/snapshot.jpg` 仅在供应方开发启动显式设置 `VISION_DEVELOPMENT_DASHBOARD=true` 时开放；托管生产模式固定关闭。

## 编码说明

项目源码、文档和 Dashboard 统一使用 UTF-8。Windows PowerShell 读取中文文件时建议显式指定编码：

```powershell
Get-Content -Raw -Encoding UTF8 README.md
Get-Content -Raw -Encoding UTF8 docs\DEPLOYMENT.md
```

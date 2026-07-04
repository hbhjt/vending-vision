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
POST /camera/{role}/reopen
GET  /camera/front/owner
GET  /session/status
GET  /try-on/{sessionId}.mjpeg
WS   /ws
```

## 上线注意

- 推荐 Python 3.10。
- 模型文件不提交到代码仓库时，需要在部署机器本地放入 `models/`。
- 正式运行保持 `mock_scenario=off`。
- 先完成双摄编号确认，再做顶部多人阈值和中部画像质量联调。

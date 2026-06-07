# Vending Vision Module

智能售货机视觉识别模块。服务在 Win10 售货机本机运行，长期读取摄像头画面；检测到用户靠近后，多帧采样并聚合画像标签，再通过 WebSocket 主动推送给原生层或后端。

## 当前流程

```text
启动视觉服务
-> 原生层/后端连接 ws://127.0.0.1:7892/ws
-> 发送 vision.hello，服务返回 vision.ready
-> 视觉端持续检测摄像头画面
-> 有人进入画面但未靠近时，缓存适合身体识别的帧
-> 用户靠近时，补采近距离帧并聚合画像
-> 推送 vision.profile_result
```

原生层/后端只需要连接 WebSocket、发送 `vision.hello` / `vision.ping`，然后等待视觉端主动推送。当前流程不需要客户端发送 `vision.start_profile`。

## 部署环境

推荐售货机环境：

```text
OS: Windows 10
Python: 3.10
Project: D:\ai-cv\vending_vision
HTTP: http://127.0.0.1:7892
WebSocket: ws://127.0.0.1:7892/ws
Dashboard: http://127.0.0.1:7892/dashboard
```

如果售货机直接安装 Python、没有虚拟环境，可以直接用系统 Python 安装依赖：

```bat
cd /d D:\ai-cv\vending_vision
python -m pip install -r requirements.txt
```

## 摄像头配置

摄像头编号写在 `config.json`。售货机本机摄像头为 `index=0` 时，配置为：

```json
{
  "camera_index": 0,
  "camera_backend": "dshow",
  "camera_width": 1280,
  "camera_height": 720,
  "camera_fps": 30,
  "camera_fourcc": "MJPG",
  "camera_keep_open": true
}
```

如果摄像头打不开，先探测编号：

```bat
python test_camera.py --probe --max-index 8
```

拍一张测试图：

```bat
python test_camera.py --index 0 --backend dshow --width 1280 --height 720 --fps 30 --fourcc MJPG --output debug_outputs/camera_test.jpg
```

## 启动

普通模式用于联调和上线：

```bat
scripts\start_server.bat
```

过程追踪模式用于演示和排查，会保存中间图片：

```bat
scripts\start_trace_server.bat
```

启动后检查：

```text
http://127.0.0.1:7892/health
http://127.0.0.1:7892/camera/status
http://127.0.0.1:7892/dashboard
```

## 常用测试

```bat
python test_ws_client.py --wait-seconds 60
python test_proximity.py --runs 20 --interval 0.5
python test_integration.py
```

如果没有推送，使用过程追踪模式，再查看：

```text
debug_outputs/process_traces/
```

常见原因：

| 原因 | 含义 | 处理 |
| --- | --- | --- |
| `person_present_but_not_close` | 有人但未达到靠近阈值 | 调低 close 阈值或调整摄像头角度 |
| `not_enough_valid_frames` | 有采样但有效帧不足 | 降低最低置信度或改善画面 |
| `confidence_below_threshold` | 聚合结果置信度低 | 改善光线、站位或调整画像参数 |

## 主要目录

```text
app.py                  FastAPI 服务入口
config.json             默认配置
vision/                 视觉识别核心代码
models/                 模型文件目录
dashboard/              本机画像展示面板
scripts/                启动和运维脚本
debug_outputs/          调试输出和过程追踪
reports/                汇报材料
datasets/               现场采集数据
test_reports/           测试输出
```

## 重要文档

- [DEPLOYMENT.md](DEPLOYMENT.md)：售货机 Win10 部署流程。
- [PROTOCOL.md](PROTOCOL.md)：WebSocket 通信协议。
- [FIELD_TUNING.md](FIELD_TUNING.md)：现场调参速查。
- [PROCESS_TRACE.md](PROCESS_TRACE.md)：过程追踪说明。
- [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)：单独上传 GitHub 仓库时的文件清单。
- [models/README.md](models/README.md)：模型文件说明。

## GitHub 上传说明

建议上传代码、配置、启动脚本、协议文档和模型目录说明；不要上传模型权重、日志、现场数据、测试输出和过程追踪图片。

模型文件需要在部署机器本地放置，GitHub 仓库只保留模型路径和说明文档。详细清单见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。

## 当前限制

- 年龄和性别受光线、角度、遮挡影响较大，建议作为弱参考字段。
- 单目 RGB 摄像头无法保证精确身高，身高更适合作为粗略分层。
- 多人、强反光、弱光、严重遮挡和极端俯拍仍需要现场调参。
- 正式上线建议关闭过程追踪，避免磁盘持续增长。

# Windows 候选发布物打包

Windows 候选版本使用 PyInstaller `onedir`。本地构建只属于开发产物；只有
受保护 RC tag 工作流可以正式发布候选版本。

```powershell
cd D:\ai-cv\vending_vision
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 -Wheelhouse .\wheelhouse
```

输出位于 `dist\vending-vision\vending-vision.exe`。必须分发完整
`dist\vending-vision` 目录，不能只复制 EXE。

每次构建后运行打包契约测试：

```powershell
.\.venv-packaging\Scripts\python.exe scripts\verify_packaged_exe.py
```

该测试检查 bundle 资源（包括 Windows DirectShow 枚举 adapter 与 Ed25519
验签运行时）、`/health`、`/version`、显式开发 Dashboard、metrics、严格
machine 角色 WebSocket 握手、ping/pong 和画像消息契约。真实摄像头、旋转、
试衣 MJPEG 以及长会话续租必须通过现场硬件验收。生产启动不开放 Dashboard
或旧的无认证 camera snapshot；冒烟测试仅通过显式 development flag 验证供应方
调试资源已经被正确打包。

VEM 托管选择使用外部现场配置启动：

```text
vending-vision.exe --no-browser --config C:\ProgramData\VEM\vision\config\site.json
```

打包约束：

- 依赖使用 Python 3.11.9 的单一完整传递 hash lock；构建只从已由 `pip download --require-hashes` 取得的 wheelhouse 以 `--no-index --require-hashes` 安装。Python 运行时和全部依赖进入 onedir bundle。
- `models/model-manifest.json` 中的 Git LFS 模型必须已解析且 hash 校验通过。
- 托管启动不读取相邻可编辑 `config.json`，`--config` 缺失或无效时失败关闭。
- `config.json`、`config/`、Dashboard 和模型作为供应方 bundle 资源进入产物；
  具体机器的现场配置保持在 bundle 外部。
- `VISION_WORKDIR` 只指定日志等可变运行数据目录，不改变发布物身份。

供应仓只发布原始 bundle 和供应方证据，不复制源码到 `C:\VEM\vision`、
不在现场安装 Python，也不注册 `VEM\StartVisionServer`。版本选择、安装、
健康验收和回滚均由 VEM factory/update 基础设施负责。

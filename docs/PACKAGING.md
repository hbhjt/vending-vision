# Windows 候选发布物打包

Windows 候选版本使用 PyInstaller `onedir`。本地构建只属于开发产物；每次
`main` CI 构建并上传同 commit 的候选交付包，供 VEM 验收使用。

每个 Windows CI（含 PR）都会构建、验证并检查以下交付布局；只有成功的
`main` Windows CI 会上传两个同提交的 Actions artifacts：

- `vending-vision-windows-x86_64.zip`：自包含 runtime；不含录播 fixture。
- `vending-vision-test-fixtures.zip`：`recorded-video` 的 top/front MP4 和 expected manifest。

两个 ZIP 都包含同一提交 SHA 的 `vision-artifact.json`，同伴
`vending-vision-main-artifacts.json` 列出两个 archive 的 SHA-256。消费方按该
commit 下载并原样安装或解压，不重新打包 Vision 内容。

AI 模型是独立的 `vending-vision-ai-models.zip`，绝不嵌入 runtime ZIP。
解压后目录必须只含 `ai-model-manifest.json` 和其精确 allowlist 文件；manifest
逐项绑定 `zhengchong/CatVTON` 的不可变 revision、相对路径、大小和 SHA-256。
部署先验证 pack，再以 `VEM_AI_MODEL_PACK` 指向该目录。启动仅校验 manifest、
代码和轻量 worker import，不加载完整模型或推理；顾客启动禁止下载。缺 pack
只使 AI readiness 为 false，Fast 与核心 Vision 仍可用。

官方 CatVTON worker 作为独立 `vending-vision-ai-worker.exe` onedir 与主
runtime 同 ZIP 交付，但不是服务、不是相机 owner、也不常驻。主 Vision 只按
attempt/probe 通过 supervisor 启动该 artifact-relative worker；source/dev 才使用
当前 Python 的 `vision.ai_attempt_worker` module。AI 依赖使用独立
`requirements-ai.txt` 的 exact direct versions，并由 `requirements-ai.lock.json`
描述 Windows x64 release wheelhouse；`scripts/verify_ai_wheelhouse.py` 在缺少
release-provided wheel manifest 时 fail closed，现场不得临时 pip install。不要把
torch/diffusers/SCHP 运行依赖或 4.5GB 权重折进核心 Vision hash lock，也不要把
权重打入任一 PyInstaller archive。正式 attempt child 使用 `VEM_AI_MODEL_PACK`
指向的 verified pack、vendored CatVTON source、`HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1` 和 CatVTON `local_files_only=True`；startup probe 只
import torch/torchvision/diffusers/accelerate/safetensors/PIL/numpy/cv2 与 vendored
CatVTON/SCHP，解析 3 个 JSON，不 `torch.load`、不 `from_pretrained`、不推理。
失败时只让 AI readiness/attempt failclosed。

CI 会验证 runtime ZIP 根目录包含 `vending-vision.exe` 和 manifest，fixture ZIP
包含 `recorded-video/top.mp4`、`front.mp4`、expected manifest 与
`vision-artifact.json`；
runtime ZIP 不得携带 fixture 路径或 MP4。

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

该测试检查 bundle 资源（包括 Windows DirectShow 枚举 adapter）、`/health`、`/version`、显式开发 Dashboard、metrics、严格
machine 角色 WebSocket 握手、ping/pong 和画像消息契约。真实摄像头、旋转、
V2 试衣尝试与结果读取必须通过现场硬件验收。生产启动不开放 Dashboard
或旧的开发 camera snapshot；冒烟测试仅通过显式 development flag 验证供应方
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

供应仓只发布原始 bundle 与 SHA-256 交付清单，不复制源码到 `C:\VEM\vision`、
不在现场安装 Python，也不注册 `VEM\StartVisionServer`。版本选择、安装、
健康验收和回滚均由 VEM factory/update 基础设施负责。

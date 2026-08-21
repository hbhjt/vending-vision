# Windows 部署与现场验收

## 职责边界

本仓库负责生成不可变的 Windows 视觉交付物，包括运行时、模型、依赖、
现场配置 schema、bundle 和同 commit 的 SHA-256 交付清单。本仓库不在售货机
上复制源码、联网下载依赖、注册 Windows 任务或重新打包
已经发布的 bundle。

VEM 负责按同一 commit 的交付清单校验 archive SHA-256 与字节数、解压到
`C:\VEM\vision\app`、
`C:\ProgramData\VEM\vision`、`VEM\StartVisionServer` 任务、安装验收和回滚。

## 同提交产物对发布

1. 将修改合并到 `main`。
2. `.github/workflows/ci.yml` 在测试、Windows 测试与区域证据合同通过后，
   构建 onedir EXE、执行机器协议打包冒烟测试，并上传同 commit 的
   `vending-vision-main-<commit>` Actions artifact。
3. `vending-vision-main-artifacts.json` 以 SHA-256 锁定 runtime 与录播
   fixture、同一 commit 与 archive 文件名。
4. VEM 按 commit 下载成对 artifacts，校验清单与摘要后原样安装；本仓库
   永远不生成 VEM approval，也不依赖任何签名 secret。

## VEM 托管现场配置

VEM 使用以下方式启动已选中的可执行文件：

```powershell
vending-vision.exe --no-browser `
  --config C:\ProgramData\VEM\vision\config\site.json
```

外部文件必须满足 `config/vending-vision-site-config-v1.schema.json`。
托管模式遇到文件缺失、JSON 损坏、未知字段、摄像头角色错误、非 loopback
host、非法端口或 mock 配置时会直接启动失败。环境变量和相邻 `config.json`
回退只保留给不带 `--config` 的供应方开发流程。

相机维护使用与运行时相同的 loopback v2 合同，不引入独立的 daemon token、
JWT、session、replay ledger 或 keyring。Vision 继续拥有并持久化稳定的
camera identity；VEM 只消费合同中的不透明 candidate ID 和 role readiness。

安装验收必须证明：

- delivery manifest 的同提交、archive SHA-256、字节数与内部 digest；
- `models/model-manifest.json` 中每个模型都存在且 hash 正确；
- 即使现场摄像头暂时离线，完整模型仍使 `modelReady=true`；
- HTTP `/health` 和严格机器端 `ws://127.0.0.1:7892/ws` 握手成功；
- 运行时 build identity 与交付 commit 一致。

摄像头不可用属于 degraded，不回滚其他部分有效的软件。模型、配置、进程、
HTTP 或 WebSocket 契约失败属于安装失败，必须回滚。

## 真实 VEM 现场验收

通过安装验收后，使用 Vision 的本机维护合同枚举候选、预览、测试并确认
top/front 角色。维护请求直接发送到 loopback 合同；稳定 DirectShow moniker 和当前 OpenCV index 从同一
`cv2-enumerate-cameras` 边界取得。重插后的 index 改变会被刷新解析到原绑定；
无法证明映射的异常 adapter 才产生 explicit ambiguous/non-ready。backend index
仅是 loopback v2 维护观察值，不能写入现场配置或 VEM：

```text
GET http://127.0.0.1:7892/maintenance/cameras
GET http://127.0.0.1:7892/maintenance/cameras/{candidateId}/preview.jpg
POST http://127.0.0.1:7892/maintenance/cameras/top/test
POST http://127.0.0.1:7892/maintenance/cameras/top/confirm
```

confirm 请求必须同时带 `testEvidenceId`、`operatorVisualConfirmation: true`
和 `expectedGeneration`；它们与同 role 的真实测试结果在一个原子确认中校验。
运行时保持打开的流会在预览/测试期间受保护地 handoff，维护完成后恢复。

操作员依次靠近、单人站入交互区域、进入试衣并离开，同时运行：

```powershell
python scripts\verify_real_camera_capability.py `
  --machine-code VEM-WIN10-REAL-01 `
  --observation-timeout 180 `
  --evidence C:\ProgramData\VEM\vision\evidence\capability-acceptance.json
```

该命令拒绝 mock 模式，且只有观察到以下事实才通过：

- 真实的有人 `vision.presence_status`；
- 单人且可用于推荐的 `vision.profile_result`；
- 去抖后的 `vision.person_departed` 边沿事件；
- strict V2 `vision.ready` and independent core capability events.

现场还需确认多人产生 `occupancy.state=multiple` 且不生成可用个体推荐、
摄像头拔插后恢复。这些真实硬件观察不能由
打包 mock 冒烟测试替代。

## 仅供供应方开发

```powershell
python -m pip download --only-binary=:all: --require-hashes -d wheelhouse -r requirements.txt
python -m pip install --no-index --find-links wheelhouse --require-hashes -r requirements.txt
scripts\start_server.bat
python -m pytest -q
```

开发启动可以使用相邻 `config.json` 和 mock 场景，但它们都不是正式安装方式，
也不能作为生产验收证据。

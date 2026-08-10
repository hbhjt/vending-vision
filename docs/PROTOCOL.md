# Vision 通信协议：V2 握手与就绪

协议版本：`vem.vision.v2`

默认 WebSocket 地址：`ws://127.0.0.1:7892/ws`

本文只描述 V2 的连接、合同身份和就绪边界。画像、presence 和 Fast 试衣尝试必须在 V2
就绪之后按生成的合同消费；本文不定义旧式会话流程。

## 唯一合同来源

`contracts/vem_vision_v2/manifest.json` 是运行时身份的唯一来源。客户端和服务端都
必须从其加载以下四项，不能在代码、脚本或部署描述中手写替代值：

- `protocol`
- `schemaVersion`
- `bundleVersion`
- `contractDigest`

manifest、JSON Schema、Python 生成模型及共享 fixtures 必须一起校验。manifest 缺失或
损坏时，服务端只可在仍有严格解析器时以受限的 `contract_bundle_unavailable` ready
报告；不能回退接受 V1 或宽松 hello。

## 通用信封

每帧都是严格 UTF-8 JSON 对象；未知信封字段或 payload 字段均无效：

```json
{
  "protocol": "vem.vision.v2",
  "type": "vision.hello",
  "messageId": "hello-001",
  "timestamp": "2026-08-10T12:00Z",
  "payload": {}
}
```

| 字段 | 规则 |
| --- | --- |
| `protocol` | 必须等于 manifest 的 `protocol` |
| `type` | 必须是合同已定义的 discriminator |
| `messageId` | 1–128 个 Unicode code point；它是传输追踪 ID，不等同业务事件 ID |
| `timestamp` | 严格 UTC ISO 格式：大写 `T`/`Z`，接受 `HH:MMZ`、带秒和带小数秒；拒绝空格、小写和 offset |
| `payload` | 必须匹配该 `type` 的严格对象模型 |

时间值还必须是实际日历时间。例如 `2026-02-30T00:00Z` 无效。合同采用 Unicode code
point 数而非 UTF-16 code unit 或 UTF-8 字节数计算 `messageId`、服务器元数据和
capability 的长度。

## 连接与握手

机器连接后首先发送严格的 `vision.hello`。hello 的合同身份字段必须与本地加载的
manifest 相同：

```json
{
  "protocol": "vem.vision.v2",
  "type": "vision.hello",
  "messageId": "hello-machine-001",
  "timestamp": "2026-08-10T12:00Z",
  "payload": {
    "clientRole": "machine",
    "machineCode": "VEM-WIN10-01",
    "schemaVersion": "<manifest schemaVersion>",
    "bundleVersion": "<manifest bundleVersion>",
    "contractDigest": "<manifest contractDigest>",
    "capabilities": ["profile_push", "presence_status"]
  }
}
```

服务端处于 `awaiting_ready` 时，只允许返回 `vision.ready` 或 `vision.error`。任何业务
事件、未知类型、额外字段或错误 phase 都是协议失败，客户端不得忽略后继续等待。
V1 hello、`protocolVersion`、`modelReady` 与所有 V1 成功条件均不受支持。

## `vision.ready`

成功解析 hello 后服务端返回一条严格的 ready：

```json
{
  "protocol": "vem.vision.v2",
  "type": "vision.ready",
  "messageId": "ready-001",
  "timestamp": "2026-08-10T12:00:00.125Z",
  "payload": {
    "serverName": "vem-vision-python",
    "serverVersion": "0.2.0",
    "schemaVersion": "<manifest schemaVersion>",
    "bundleVersion": "<manifest bundleVersion>",
    "contractDigest": "<manifest contractDigest>",
    "cameraReady": true,
    "fastReady": true,
    "visionBusinessReady": true,
    "capabilities": ["profile_push", "presence_status"],
    "diagnostics": []
  }
}
```

`serverName` 与 `schemaVersion` 最多 128 个 Unicode code point；`serverVersion`、
`bundleVersion` 和每项 capability 最多 64 个 Unicode code point。`contractDigest` 必须是
64 位小写十六进制摘要。机器必须逐项
比较 ready 的四个身份值；任何不匹配都是稳定的合同错误，不能被业务 readiness 覆盖。

若严格模型可用但 bundle 身份不可用，服务端可返回受限 ready：

```json
{
  "schemaVersion": "unavailable",
  "bundleVersion": "unavailable",
  "contractDigest": "0000000000000000000000000000000000000000000000000000000000000000",
  "fastReady": false,
  "visionBusinessReady": false,
  "diagnostics": ["contract_bundle_unavailable"]
}
```

这是可观测的降级状态，而不是合同匹配；客户端必须保留诊断并保持 readiness 为 false。
若严格解析器或 schema 也不可用，服务端必须稳定报错并关闭连接。

## `vision.error` 与就绪后的消息

`vision.error` 是 awaiting-ready 阶段唯一合法的失败帧，其 payload 使用合同定义的诊断
枚举和 retryable 标志。ready 建立后，客户端才可按合同处理 `vision.pong` 及业务事件；每个
业务事件的 `eventId` 是领域 ID，允许具有领域前缀（例如
`departure-event-<uuid>`），仍不得替代信封的 `messageId`。

运行时健康检查与 WebSocket 合同是不同边界；健康成功不能证明 V2 identity 已可用。

## Fast 尝试与结果

Machine 仅可在 ready 后发起一个 `vision.try_on.attempt.start`。Fast 尝试会依次发布
accepted 只是启动确认，不是生命周期 phase。客户试衣的公开 phase 是
`acquiring -> generating -> completed`，并以 `failed` 或 `canceled` 作为 terminal。采集阶段返回当前
active attempt 的 rotating-token loopback HTTP MJPEG preview（固定路径
`/v2/try-on/acquisition/preview.mjpeg?token=...`）、`streamType`、真实 occupancy、guidance
和 `manualCaptureAllowed`。Machine 发送 capture 或仅带 `user`/`route_leave` 的 cancel；
manual 只在 single/`hold_still` 时可用，只绕过 stability waiting，不能绕过 no-person、multiple-person
或 alignment 保护；`ready` 会立即自动采集，不公开 manual 动作。

Vision 直接拥有源帧，Machine 不从渲染后的 preview bytes 提取生成输入。capture 后 preview
关闭且 front-camera lease 在 generating 前释放；generating 只发布 `preparing`/`rendering`
等 coarse stage，不发布百分比。服务端 canceled reason 仅允许 `user`、`route_leave`、
`disconnect`、`departure`、`replaced`、`timeout`。一个 machine-wide 非 terminal attempt
以及 attempt identity fencing 保证 replacement、departure、disconnect、timeout 和 late
worker output 不会污染新交互。top-camera presence/departure/ambient 贯穿全流程，try-on
route 内 profile 不触发导航，也不存在 automaticFast/AI fallback。

## 验证与兼容性

- client/server 各自有 valid/invalid fixture corpus；Machine inbound 只接受 server schema，Vision
  inbound 只接受 client schema。
- generated manifest 的 canonical bytes、文件集合和 digest 必须通过本地 checker 验证。
- 当前兼容边界只接受 `vem.vision.v2`；V1 帧仅可作为拒绝性测试样本。
- 发送方或接收方不得因未知字段、未知 discriminator 或错误 phase 而静默降级。

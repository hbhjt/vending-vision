# Vision 通信协议：V2 握手与就绪

协议版本：`vem.vision.v2`

默认 WebSocket 地址：`ws://127.0.0.1:7892/ws`

本文只描述 V2 的连接、合同身份和就绪边界。画像、presence 等业务事件必须在 V2
就绪之后按生成的合同消费；本文不定义后续会话流程。

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
| `messageId` | 1–128 个 Unicode 字符；它是传输追踪 ID，不等同业务事件 ID |
| `timestamp` | 严格 UTC ISO 格式：大写 `T`/`Z`，接受 `HH:MMZ`、带秒和带小数秒；拒绝空格、小写和 offset |
| `payload` | 必须匹配该 `type` 的严格对象模型 |

时间值还必须是实际日历时间。例如 `2026-02-30T00:00Z` 无效。合同采用字符数而非
UTF-8 字节数计算 `messageId`、服务器元数据和 capability 的长度。

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

`serverName` 与 `schemaVersion` 最多 128 个字符；`serverVersion`、`bundleVersion` 和每项
capability 最多 64 个字符。`contractDigest` 必须是 64 位小写十六进制摘要。机器必须逐项
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

## 验证与兼容性

- 共享 valid/invalid fixtures 是 TS、Python 和 Rust 的共同证据。
- generated manifest 的 canonical bytes、文件集合和 digest 必须通过本地 checker 验证。
- 当前兼容边界只接受 `vem.vision.v2`；V1 帧仅可作为拒绝性测试样本。
- 发送方或接收方不得因未知字段、未知 discriminator 或错误 phase 而静默降级。
